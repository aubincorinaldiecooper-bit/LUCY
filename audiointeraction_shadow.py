"""Observational shadow client for AudioInteraction (https://github.com/xzf-thu/Audio-Interaction).

Purpose: measure whether AudioInteraction's turn-taking decisions (KEEP_SILENCE /
TEXT_BEGIN) are fast and useful enough to consider as a future turn-taking layer.
This module NEVER affects the live session:

- User audio frames are teed into a bounded queue; when the queue is full the
  oldest frames are dropped. The production STT path sees every frame unchanged.
- The sidecar connection runs as a background task with backoff reconnects.
  Slow, unavailable, or erroring sidecars only increment counters.
- Decisions are compared against LUCY's TurnPolicy at turn commit and logged.
  Nothing is injected into the LLM/TTS path.
- Privacy: raw audio is never logged. Sidecar text payloads are logged only when
  AUDIOINTERACTION_DEBUG_TEXT=true; otherwise only their length is logged.

Sidecar protocol (minimal, defensive): binary PCM frames are sent over a
WebSocket at AUDIOINTERACTION_ENDPOINT; the sidecar sends JSON text messages.
Decision values are normalized: silent/keep_silence/<no need to response> ->
KEEP_SILENCE, speak/text_begin/respond -> TEXT_BEGIN.
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

DECISION_KEEP_SILENCE = "KEEP_SILENCE"
DECISION_TEXT_BEGIN = "TEXT_BEGIN"

_KEEP_SILENCE_VALUES = {"keep_silence", "keep-silence", "silent", "silence", "<silent>", "no_response", "<no need to response>"}
_TEXT_BEGIN_VALUES = {"text_begin", "text-begin", "speak", "<speak>", "respond", "response_begin", "begin"}
_DECISION_KEYS = ("decision", "state", "action", "event", "type")
_TEXT_KEYS = ("text", "content", "transcript", "response")

MAX_QUEUE_FRAMES = 100
RECONNECT_BACKOFF_MAX_SECONDS = 5.0


def audiointeraction_mode() -> str:
    mode = os.getenv("AUDIOINTERACTION_MODE", "off").strip().lower()
    return mode if mode in {"off", "shadow"} else "off"


def audiointeraction_endpoint() -> str:
    return os.getenv("AUDIOINTERACTION_ENDPOINT", "").strip()


def audiointeraction_timeout_ms() -> int:
    try:
        return max(100, int(os.getenv("AUDIOINTERACTION_TIMEOUT_MS", "1000")))
    except Exception:
        return 1000


def audiointeraction_debug_text() -> bool:
    return os.getenv("AUDIOINTERACTION_DEBUG_TEXT", "false").strip().lower() in {"true", "1", "yes"}


@dataclass(slots=True)
class ShadowDecision:
    decision: str
    text: str
    received_at: float


def parse_decision_message(raw: Any) -> tuple[str, str, float | None] | None:
    """Parse a sidecar message into (decision, text, infer_ms). Returns None when unusable."""
    data: Any = raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:
            return None
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    decision_value = ""
    for key in _DECISION_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            decision_value = value.strip().lower()
            break
    if not decision_value:
        return None
    if decision_value in _KEEP_SILENCE_VALUES:
        decision = DECISION_KEEP_SILENCE
    elif decision_value in _TEXT_BEGIN_VALUES:
        decision = DECISION_TEXT_BEGIN
    else:
        return None
    text = ""
    for key in _TEXT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    infer_ms: float | None = None
    for key in ("infer_ms", "inference_ms", "latency_ms"):
        value = data.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            infer_ms = float(value)
            break
    return decision, text, infer_ms


def lucy_decision_commits(should_start_generation: bool) -> bool:
    return bool(should_start_generation)


def classify_disagreement(lucy_committed: bool, shadow_decision: str) -> str:
    """Returns 'agree' or a disagreement type for a comparison with a fresh shadow decision."""
    shadow_wants_speech = shadow_decision == DECISION_TEXT_BEGIN
    if lucy_committed and not shadow_wants_speech:
        return "lucy_committed_shadow_kept_silence"
    if not lucy_committed and shadow_wants_speech:
        return "lucy_held_shadow_would_speak"
    return "agree"


class AudioInteractionShadow:
    def __init__(
        self,
        endpoint: str,
        timeout_ms: int | None = None,
        debug_text: bool | None = None,
        ws_factory: Callable[..., Any] | None = None,
        max_queue_frames: int = MAX_QUEUE_FRAMES,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms if timeout_ms is not None else audiointeraction_timeout_ms()
        self.debug_text = debug_text if debug_text is not None else audiointeraction_debug_text()
        self._ws_factory = ws_factory or self._aiohttp_ws_factory
        self._frame_queue: deque[bytes] = deque(maxlen=max_queue_frames)
        self._frame_event = asyncio.Event()
        self._closed = False
        self._run_task: asyncio.Task | None = None
        self.latest_decision: ShadowDecision | None = None
        self._window_start_at = 0.0
        self._awaiting_late_decision: tuple[int, float] | None = None
        self.counters: dict[str, int] = {
            "frames_sent": 0,
            "frames_dropped": 0,
            "decisions_total": 0,
            "decisions_keep_silence": 0,
            "decisions_text_begin": 0,
            "parse_errors": 0,
            "connect_errors": 0,
            "send_errors": 0,
            "reconnects": 0,
            "comparisons": 0,
            "agreements": 0,
            "disagreements": 0,
            "disagreement_lucy_committed_shadow_kept_silence": 0,
            "disagreement_lucy_held_shadow_would_speak": 0,
            "no_decision_turns": 0,
            "late_decisions": 0,
        }
        self._decision_latency_total = 0.0
        self._decision_latency_max = 0.0
        self._late_latency_total = 0.0
        self._infer_ms_count = 0
        self._infer_ms_total = 0.0
        self._infer_ms_max = 0.0

    # ---------- audio fork (hot path: must never block or raise) ----------

    def feed_frame(self, frame: Any) -> None:
        if self._closed:
            return
        try:
            data = getattr(frame, "data", frame)
            payload = bytes(data) if not isinstance(data, bytes) else data
        except Exception:
            return
        if len(self._frame_queue) >= (self._frame_queue.maxlen or MAX_QUEUE_FRAMES):
            self.counters["frames_dropped"] += 1
        self._frame_queue.append(payload)
        self._frame_event.set()

    # ---------- sidecar connection ----------

    async def _aiohttp_ws_factory(self):
        import aiohttp

        timeout_seconds = self.timeout_ms / 1000
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=timeout_seconds)
        )
        try:
            ws = await session.ws_connect(self.endpoint, heartbeat=20)
        except Exception:
            await session.close()
            raise
        return session, ws

    def start(self) -> None:
        if self._run_task is None:
            self._run_task = asyncio.get_running_loop().create_task(self.run())

    async def run(self) -> None:
        backoff_seconds = 0.5
        first_attempt = True
        while not self._closed:
            session = None
            ws = None
            try:
                session, ws = await self._ws_factory()
                logger.info("audiointeraction_shadow_connected=true endpoint_present=true")
                backoff_seconds = 0.5
                if not first_attempt:
                    self.counters["reconnects"] += 1
                first_attempt = False
                await self._pump(ws)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.counters["connect_errors"] += 1
                logger.warning(
                    "audiointeraction_shadow_connection_error=true error_type=%s connect_errors=%s backoff_seconds=%.1f",
                    type(exc).__name__,
                    self.counters["connect_errors"],
                    backoff_seconds,
                )
            finally:
                for closable in (ws, session):
                    if closable is not None:
                        try:
                            await closable.close()
                        except Exception:
                            pass
            if self._closed:
                break
            try:
                await asyncio.sleep(backoff_seconds)
            except asyncio.CancelledError:
                break
            backoff_seconds = min(RECONNECT_BACKOFF_MAX_SECONDS, backoff_seconds * 2)

    async def _pump(self, ws: Any) -> None:
        sender = asyncio.create_task(self._send_loop(ws))
        try:
            async for message in ws:
                payload = getattr(message, "data", message)
                message_type = getattr(getattr(message, "type", None), "name", "")
                if message_type in {"CLOSED", "CLOSE", "ERROR"}:
                    break
                self.handle_sidecar_message(payload)
        finally:
            sender.cancel()
            try:
                await sender
            except (asyncio.CancelledError, Exception):
                pass

    async def _send_loop(self, ws: Any) -> None:
        while not self._closed:
            if not self._frame_queue:
                self._frame_event.clear()
                await self._frame_event.wait()
                continue
            payload = self._frame_queue.popleft()
            try:
                await asyncio.wait_for(ws.send_bytes(payload), timeout=self.timeout_ms / 1000)
                self.counters["frames_sent"] += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.counters["send_errors"] += 1
                raise

    # ---------- decisions ----------

    def handle_sidecar_message(self, raw: Any) -> None:
        parsed = parse_decision_message(raw)
        if parsed is None:
            self.counters["parse_errors"] += 1
            return
        decision, text, infer_ms = parsed
        if infer_ms is not None:
            self._infer_ms_count += 1
            self._infer_ms_total += infer_ms
            self._infer_ms_max = max(self._infer_ms_max, infer_ms)
        now = time.monotonic()
        self.latest_decision = ShadowDecision(decision=decision, text=text, received_at=now)
        self.counters["decisions_total"] += 1
        if decision == DECISION_KEEP_SILENCE:
            self.counters["decisions_keep_silence"] += 1
        else:
            self.counters["decisions_text_begin"] += 1
        if self._awaiting_late_decision is not None:
            late_turn_id, committed_at = self._awaiting_late_decision
            self._awaiting_late_decision = None
            lateness_seconds = now - committed_at
            self.counters["late_decisions"] += 1
            self._late_latency_total += lateness_seconds
            logger.warning(
                "audiointeraction_late_decision=true turn_id=%s shadow_decision=%s lateness_after_commit_seconds=%.3f",
                late_turn_id,
                decision,
                lateness_seconds,
            )
        text_field = text if self.debug_text else ""
        logger.info(
            "audiointeraction_decision=%s decision_text_length=%s sidecar_infer_ms=%s decision_text=%s",
            decision,
            len(text),
            "n/a" if infer_ms is None else f"{infer_ms:.1f}",
            text_field if text_field else "redacted",
        )

    def compare_at_turn_commit(self, turn_id: int, lucy_decision: str, lucy_should_start_generation: bool) -> dict[str, Any]:
        now = time.monotonic()
        self.counters["comparisons"] += 1
        lucy_committed = lucy_decision_commits(lucy_should_start_generation)
        decision = self.latest_decision
        fresh = decision is not None and decision.received_at > self._window_start_at
        if not fresh:
            self.counters["no_decision_turns"] += 1
            self._awaiting_late_decision = (turn_id, now)
            result = {
                "turn_id": turn_id,
                "lucy_decision": lucy_decision,
                "lucy_committed": lucy_committed,
                "shadow_decision": "none",
                "decision_before_commit": False,
                "agreement": "no_decision",
                "shadow_decision_age_seconds": None,
            }
        else:
            self._awaiting_late_decision = None
            age_seconds = now - decision.received_at
            self._decision_latency_total += age_seconds
            self._decision_latency_max = max(self._decision_latency_max, age_seconds)
            agreement = classify_disagreement(lucy_committed, decision.decision)
            if agreement == "agree":
                self.counters["agreements"] += 1
            else:
                self.counters["disagreements"] += 1
                self.counters[f"disagreement_{agreement}"] += 1
            result = {
                "turn_id": turn_id,
                "lucy_decision": lucy_decision,
                "lucy_committed": lucy_committed,
                "shadow_decision": decision.decision,
                "decision_before_commit": True,
                "agreement": agreement,
                "shadow_decision_age_seconds": age_seconds,
            }
        self._window_start_at = now
        logger.info(
            "audiointeraction_shadow_comparison turn_id=%s lucy_decision=%s lucy_committed=%s shadow_decision=%s decision_before_commit=%s shadow_decision_age_seconds=%s agreement=%s",
            result["turn_id"],
            result["lucy_decision"],
            result["lucy_committed"],
            result["shadow_decision"],
            result["decision_before_commit"],
            "n/a" if result["shadow_decision_age_seconds"] is None else f"{result['shadow_decision_age_seconds']:.3f}",
            result["agreement"],
        )
        return result

    # ---------- summary / lifecycle ----------

    def summary(self) -> dict[str, Any]:
        compared_with_decision = self.counters["comparisons"] - self.counters["no_decision_turns"]
        return {
            **self.counters,
            "decision_before_commit_count": compared_with_decision,
            "avg_decision_age_seconds": (self._decision_latency_total / compared_with_decision) if compared_with_decision else None,
            "max_decision_age_seconds": self._decision_latency_max if compared_with_decision else None,
            "avg_lateness_after_commit_seconds": (self._late_latency_total / self.counters["late_decisions"]) if self.counters["late_decisions"] else None,
            "avg_sidecar_infer_ms": (self._infer_ms_total / self._infer_ms_count) if self._infer_ms_count else None,
            "max_sidecar_infer_ms": self._infer_ms_max if self._infer_ms_count else None,
        }

    def log_summary(self) -> None:
        summary = self.summary()
        logger.info(
            "audiointeraction_shadow_summary comparisons=%s decisions_total=%s keep_silence=%s text_begin=%s "
            "decision_before_commit_count=%s no_decision_turns=%s late_decisions=%s agreements=%s disagreements=%s "
            "lucy_committed_shadow_kept_silence=%s lucy_held_shadow_would_speak=%s "
            "avg_decision_age_seconds=%s max_decision_age_seconds=%s avg_lateness_after_commit_seconds=%s "
            "avg_sidecar_infer_ms=%s max_sidecar_infer_ms=%s "
            "frames_sent=%s frames_dropped=%s connect_errors=%s send_errors=%s parse_errors=%s reconnects=%s",
            summary["comparisons"],
            summary["decisions_total"],
            summary["decisions_keep_silence"],
            summary["decisions_text_begin"],
            summary["decision_before_commit_count"],
            summary["no_decision_turns"],
            summary["late_decisions"],
            summary["agreements"],
            summary["disagreements"],
            summary["disagreement_lucy_committed_shadow_kept_silence"],
            summary["disagreement_lucy_held_shadow_would_speak"],
            "n/a" if summary["avg_decision_age_seconds"] is None else f"{summary['avg_decision_age_seconds']:.3f}",
            "n/a" if summary["max_decision_age_seconds"] is None else f"{summary['max_decision_age_seconds']:.3f}",
            "n/a" if summary["avg_lateness_after_commit_seconds"] is None else f"{summary['avg_lateness_after_commit_seconds']:.3f}",
            "n/a" if summary["avg_sidecar_infer_ms"] is None else f"{summary['avg_sidecar_infer_ms']:.1f}",
            "n/a" if summary["max_sidecar_infer_ms"] is None else f"{summary['max_sidecar_infer_ms']:.1f}",
            summary["frames_sent"],
            summary["frames_dropped"],
            summary["connect_errors"],
            summary["send_errors"],
            summary["parse_errors"],
            summary["reconnects"],
        )

    async def aclose(self) -> None:
        self._closed = True
        self._frame_event.set()
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
        self.log_summary()


def build_shadow_from_env(ws_factory: Callable[..., Any] | None = None) -> AudioInteractionShadow | None:
    mode = audiointeraction_mode()
    if mode != "shadow":
        return None
    endpoint = audiointeraction_endpoint()
    if not endpoint:
        logger.warning("audiointeraction_shadow_disabled=true reason=mode_shadow_but_endpoint_missing")
        return None
    return AudioInteractionShadow(endpoint=endpoint, ws_factory=ws_factory)

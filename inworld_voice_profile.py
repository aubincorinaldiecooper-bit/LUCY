"""Inworld voice-profile layer: weak emotional/vocal context, never user-facing.

Inworld's streaming STT (model ``inworld/inworld-stt-1``) returns a ``voiceProfile``
alongside each transcript — emotion / vocalStyle / accent / age / pitch, each an
array of ``{label, confidence}`` sorted by confidence — when
``inworldConfig.voiceProfileThreshold`` is set. We normalize that raw profile into
the weak-signal schema the planner consumes:

    {energy, tension, certainty, emotion_confidence, pitch, vocal_style, accent}

Hard rules baked in here:
  - raw emotion labels (sad/angry/…) are NEVER surfaced to the planner summary or
    the user — we only expose the derived dims + pitch/vocal_style/accent, so the
    model can't parrot "you sound sad",
  - it's a weak signal: low-confidence emotion collapses to neutral so it can't
    force expressive tags downstream.

Pure (config + message builders + parse + normalize) so it's unit testable without
a live WebSocket. Inworld voice profiling is English-only today; for other
languages this simply yields a neutral profile.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Inworld emotion labels -> derived dimensions. Raw labels stay internal.
_ENERGY = {
    "happy": "high", "angry": "high", "surprised": "high",
    "sad": "low", "calm": "low", "tender": "low",
    "frustrated": "medium", "fearful": "medium",
}
_TENSION = {
    "angry": "high", "fearful": "high", "frustrated": "high",
    "calm": "low", "tender": "low",
    "happy": "medium", "sad": "medium", "surprised": "medium",
}
_CERTAINTY = {
    "happy": "high", "calm": "high", "angry": "high",
    "fearful": "low", "surprised": "low", "sad": "low",
    "tender": "medium", "frustrated": "medium",
}
_NEUTRAL = {"energy": "medium", "tension": "medium", "certainty": "medium"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class InworldConfig:
    enabled: bool
    ws_url: str
    api_key: str
    model_id: str
    voice_profile_threshold: float
    sample_rate: int
    # Below this, the top emotion is treated as unreliable -> neutral profile.
    emotion_confidence_floor: float
    # Inworld's base64 clientId:clientSecret key is a Basic credential (matches the
    # working snippet); override to "Bearer" only if you use a bearer token.
    auth_scheme: str = "Basic"

    @classmethod
    def from_env(cls) -> "InworldConfig":
        return cls(
            enabled=_env_bool("INWORLD_ENABLED", False) and _env_bool("INWORLD_VOICE_PROFILE_ENABLED", False),
            ws_url=(os.getenv("INWORLD_STT_WS_URL")
                    or "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional").strip(),
            api_key=(os.getenv("INWORLD_API_KEY") or "").strip(),
            model_id=(os.getenv("INWORLD_MODEL_ID") or os.getenv("INWORLD_STT_MODEL_ID") or "inworld/inworld-stt-1").strip(),
            voice_profile_threshold=float(os.getenv("INWORLD_VOICE_PROFILE_THRESHOLD", "0.5") or "0.5"),
            sample_rate=int(os.getenv("INWORLD_STT_SAMPLE_RATE", "16000") or "16000"),
            emotion_confidence_floor=float(
                os.getenv("INWORLD_EMOTION_CONFIDENCE_FLOOR", "0.5") or "0.5"
            ),
            auth_scheme=(os.getenv("INWORLD_AUTH_SCHEME") or "Basic").strip() or "Basic",
        )

    def is_usable(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "inworld_disabled"
        if not self.api_key:
            return False, "inworld_api_key_missing"
        if not self.ws_url:
            return False, "inworld_ws_url_missing"
        return True, "ok"

    def authorization_header(self) -> str:
        """Authorization header value for the STT websocket.

        Inworld's base64 ``clientId:clientSecret`` key is a *Basic* credential (the
        default); set ``INWORLD_AUTH_SCHEME=Bearer`` only for a bearer token. The
        scheme was previously hardcoded to Bearer, which 401'd Basic keys (the
        common case) and silently dropped the analyzer into reconnect/fallback.
        """
        scheme = (self.auth_scheme or "Basic").strip() or "Basic"
        canonical = {"basic": "Basic", "bearer": "Bearer"}.get(scheme.lower(), scheme)
        return f"{canonical} {self.api_key}"


@dataclass
class NormalizedVoiceProfile:
    energy: str = "medium"
    tension: str = "medium"
    certainty: str = "medium"
    confidence: float = 0.0
    pitch: str = ""
    vocal_style: str = ""
    accent: str = ""

    @property
    def emotion_confidence(self) -> float:
        return self.confidence

    def to_dict(self) -> dict:
        return asdict(self)

    def planner_summary(self) -> str:
        """Short neutral string for the planner. No raw emotion label, ever."""
        parts = [
            f"energy {self.energy}",
            f"tension {self.tension}",
            f"certainty {self.certainty}",
        ]
        if self.pitch:
            parts.append(f"pitch {self.pitch}")
        if self.vocal_style:
            parts.append(f"vocal style {self.vocal_style}")
        return ", ".join(parts)


NEUTRAL_PROFILE = NormalizedVoiceProfile()


def build_config_message(config: InworldConfig, *, enable_language_detection: bool = False) -> str:
    """The transcribe_config WS frame that turns on voice profiling."""
    return json.dumps(
        {
            "transcribe_config": {
                "modelId": config.model_id,
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": config.sample_rate,
                "numberOfChannels": 1,
                "enableLanguageDetection": enable_language_detection,
                "inworldConfig": {"voiceProfileThreshold": config.voice_profile_threshold},
            }
        }
    )


def build_audio_chunk_message(pcm: bytes) -> str:
    return json.dumps({"audio_chunk": {"content": base64.b64encode(pcm).decode()}})


def _profile_node(message: dict) -> dict | None:
    """Find the voiceProfile object regardless of camel/snake nesting."""
    if not isinstance(message, dict):
        return None
    result = message.get("result", message)
    if not isinstance(result, dict):
        return None
    for key in ("voiceProfile", "voice_profile"):
        node = result.get(key)
        if isinstance(node, dict):
            return node
    return None


def _top_label(profile: dict, *keys: str) -> tuple[str, float]:
    """Top (label, confidence) for the first present key; arrays are conf-sorted."""
    for key in keys:
        arr = profile.get(key)
        if isinstance(arr, list) and arr:
            best = max(
                arr,
                key=lambda e: float(e.get("confidence", 0.0)) if isinstance(e, dict) else 0.0,
            )
            if isinstance(best, dict) and best.get("label"):
                return str(best["label"]).strip().lower(), float(best.get("confidence", 0.0))
    return "", 0.0


def normalize_voice_profile(
    profile: dict | None, *, emotion_confidence_floor: float = 0.5
) -> NormalizedVoiceProfile:
    """Map a raw Inworld voiceProfile into the weak-signal schema."""
    if not isinstance(profile, dict) or not profile:
        return NormalizedVoiceProfile()

    emotion, emo_conf = _top_label(profile, "emotion")
    pitch, _ = _top_label(profile, "pitch")
    vocal_style, _ = _top_label(profile, "vocalStyle", "vocal_style")
    accent, _ = _top_label(profile, "accent")

    # Weak signal: an unreliable emotion read collapses the derived dims to neutral
    # so it can never force an expressive tag.
    if emotion and emo_conf >= emotion_confidence_floor:
        energy = _ENERGY.get(emotion, "medium")
        tension = _TENSION.get(emotion, "medium")
        certainty = _CERTAINTY.get(emotion, "medium")
    else:
        energy, tension, certainty = _NEUTRAL["energy"], _NEUTRAL["tension"], _NEUTRAL["certainty"]

    # Vocal style nudges certainty (mumbling/whispering = less certain).
    if vocal_style in {"mumbling", "whispering"}:
        certainty = "low"
    elif vocal_style == "shouting":
        certainty = "high"

    return NormalizedVoiceProfile(
        energy=energy,
        tension=tension,
        certainty=certainty,
        confidence=round(emo_conf, 3),
        pitch=pitch,
        vocal_style=vocal_style,
        accent=accent,
    )


def normalize_from_message(
    message: dict, *, emotion_confidence_floor: float = 0.5
) -> NormalizedVoiceProfile:
    """Extract + normalize a voice profile from a raw STT response message."""
    return normalize_voice_profile(
        _profile_node(message), emotion_confidence_floor=emotion_confidence_floor
    )


MAX_QUEUE_FRAMES = 100
RECONNECT_BACKOFF_MAX_SECONDS = 5.0


class InworldVoiceProfileShadow:
    """Best-effort audio sidecar for Inworld voice profile context.

    The production STT/TTS path never waits on this class. Audio frames are copied
    into a bounded queue, sent to Inworld in a background websocket task, and the
    latest normalized weak context can be read at turn commit.
    """

    def __init__(self, config: InworldConfig, ws_factory: Callable[..., Any] | None = None, max_queue_frames: int = MAX_QUEUE_FRAMES) -> None:
        self.config = config
        self._ws_factory = ws_factory or self._aiohttp_ws_factory
        self._frame_queue: deque[bytes] = deque(maxlen=max_queue_frames)
        self._frame_event = asyncio.Event()
        self._closed = False
        self._run_task: asyncio.Task | None = None
        self.latest_profile: NormalizedVoiceProfile | None = None
        self.latest_received_at = 0.0
        self.last_skip_reason = "not_started"
        self.counters = {
            "frames_queued": 0,
            "frames_sent": 0,
            "frames_dropped": 0,
            "responses_received": 0,
            "connect_errors": 0,
            "send_errors": 0,
            "parse_errors": 0,
            "reconnects": 0,
        }
        self._started_at = 0.0
        self._last_audio_sent_at = 0.0

    def feed_frame(self, frame: Any) -> None:
        if self._closed:
            self.last_skip_reason = "closed"
            return
        try:
            data = getattr(frame, "data", frame)
            payload = bytes(data) if not isinstance(data, bytes) else data
        except Exception:
            self.last_skip_reason = "invalid_audio_frame"
            return
        if len(self._frame_queue) >= (self._frame_queue.maxlen or MAX_QUEUE_FRAMES):
            self.counters["frames_dropped"] += 1
        self._frame_queue.append(payload)
        self.counters["frames_queued"] += 1
        self._frame_event.set()

    async def _aiohttp_ws_factory(self):
        import aiohttp

        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=3.0))
        try:
            ws = await session.ws_connect(
                self.config.ws_url,
                headers={"Authorization": self.config.authorization_header()},
                heartbeat=20,
            )
        except Exception:
            await session.close()
            raise
        return session, ws

    def start(self) -> None:
        usable, reason = self.config.is_usable()
        logger.info(
            "inworld_voice_profile_enabled=%s inworld_enabled=%s voice_profile_enabled=%s usable=%s skip_reason=%s model_id=%s",
            self.config.enabled,
            _env_bool("INWORLD_ENABLED", False),
            _env_bool("INWORLD_VOICE_PROFILE_ENABLED", False),
            usable,
            "none" if usable else reason,
            self.config.model_id,
        )
        if not usable:
            self.last_skip_reason = reason
            return
        if self._run_task is None:
            self._started_at = time.monotonic()
            self._run_task = asyncio.get_running_loop().create_task(self.run())

    async def run(self) -> None:
        backoff_seconds = 0.5
        first_attempt = True
        while not self._closed:
            session = None
            ws = None
            try:
                session, ws = await self._ws_factory()
                await ws.send_str(build_config_message(self.config))
                self.last_skip_reason = "none"
                logger.info("inworld_voice_profile_connected=true model_id=%s", self.config.model_id)
                if not first_attempt:
                    self.counters["reconnects"] += 1
                first_attempt = False
                await self._pump(ws)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.counters["connect_errors"] += 1
                self.last_skip_reason = type(exc).__name__
                logger.warning("inworld_voice_profile_fallback=true skip_reason=%s connect_errors=%s", type(exc).__name__, self.counters["connect_errors"])
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
                self.handle_message(payload)
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
                await ws.send_str(build_audio_chunk_message(payload))
                self.counters["frames_sent"] += 1
                self._last_audio_sent_at = time.monotonic()
                logger.info("inworld_voice_profile_audio_sent=true bytes=%s frames_sent=%s", len(payload), self.counters["frames_sent"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.counters["send_errors"] += 1
                self.last_skip_reason = type(exc).__name__
                logger.warning("inworld_voice_profile_fallback=true skip_reason=%s send_errors=%s", type(exc).__name__, self.counters["send_errors"])
                raise

    def handle_message(self, raw: Any) -> None:
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            data = json.loads(raw) if isinstance(raw, str) else raw
            profile = normalize_from_message(data, emotion_confidence_floor=self.config.emotion_confidence_floor)
        except Exception as exc:
            self.counters["parse_errors"] += 1
            self.last_skip_reason = type(exc).__name__
            logger.warning("inworld_voice_profile_parse_error=true error_type=%s", type(exc).__name__)
            return
        self.latest_profile = profile
        self.latest_received_at = time.monotonic()
        self.counters["responses_received"] += 1
        added_latency = self.latest_received_at - self._last_audio_sent_at if self._last_audio_sent_at else None
        logger.info(
            "inworld_voice_profile_response_received=true normalized_context=%s confidence=%.3f added_latency_seconds=%s",
            json.dumps(profile.to_dict(), sort_keys=True),
            profile.confidence,
            "n/a" if added_latency is None else f"{added_latency:.3f}",
        )

    def context_for_turn(self, turn_started_at: float) -> tuple[NormalizedVoiceProfile | None, str, float | None]:
        if self.latest_profile is None:
            return None, self.last_skip_reason or "no_response", None
        latency = max(0.0, self.latest_received_at - turn_started_at) if turn_started_at else None
        return self.latest_profile, "none", latency

    async def aclose(self) -> None:
        self._closed = True
        self._frame_event.set()
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("inworld_voice_profile_summary frames_sent=%s frames_dropped=%s responses_received=%s connect_errors=%s send_errors=%s parse_errors=%s fallback_skip_reason=%s", self.counters["frames_sent"], self.counters["frames_dropped"], self.counters["responses_received"], self.counters["connect_errors"], self.counters["send_errors"], self.counters["parse_errors"], self.last_skip_reason)


def build_inworld_shadow_from_env(ws_factory: Callable[..., Any] | None = None) -> InworldVoiceProfileShadow | None:
    config = InworldConfig.from_env()
    usable, reason = config.is_usable()
    logger.info("inworld_voice_profile_config enabled=%s usable=%s skip_reason=%s model_id=%s", config.enabled, usable, "none" if usable else reason, config.model_id)
    if not usable:
        return None
    return InworldVoiceProfileShadow(config=config, ws_factory=ws_factory)


def emotion_analyzer_status() -> dict:
    """Get the current status of the emotion analyzer.
    
    Returns a dictionary with information about the Inworld voice profile
    analyzer's operational status, including whether it's enabled, connected,
    and recent performance metrics.
    """
    config = InworldConfig.from_env()
    usable, reason = config.is_usable()
    
    return {
        "enabled": config.enabled,
        "usable": usable,
        "skip_reason": reason if not usable else "none",
        "model_id": config.model_id,
        "voice_profile_threshold": config.voice_profile_threshold,
        "emotion_confidence_floor": config.emotion_confidence_floor,
    }

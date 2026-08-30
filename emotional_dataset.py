"""Durable dataset collection for the emotional analyzer (Postgres).

The goal: every voice session accumulates labeled data we can later use to
evaluate and improve the emotional analyzer. Two row kinds:

  voice_profile_events           every voiceProfile capture on the Realtime
                                 path — the raw label arrays (the analyzer's
                                 output), the normalized weak-signal dims,
                                 and the transcript on the carrying event.
  emotional_calibration_moments  ground truth — the open, conversational
                                 reflection Arche voiced and the user's
                                 free-form reply to it.

Raw emotion labels ARE stored here (that's the point of a training/eval
dataset), but they are server-side only: nothing in this module flows back to
the model or the user. Writes are best-effort and happen on a background task
through a bounded queue so the realtime audio path never blocks on Postgres.

The CalibrationTracker carries the calibration heuristics for the Inworld
Realtime path: at most one pending question at a
time, a minimum user-turn cadence, questions only when emotionally useful,
always phrased as gentle or-questions the user can correct — never "I
detected" / "you sound".
"""

from __future__ import annotations

import asyncio
import logging
import os

from env_utils import env_bool as _env_bool
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from memory_layer import EMOTIONAL_PATTERN_PREFIX

logger = logging.getLogger(__name__)

MAX_QUEUE_ROWS = 200
CALIBRATION_MIN_USER_TURNS_BETWEEN = 3

_PROFILE_EVENT_SQL = """
INSERT INTO voice_profile_events
    (session_id, source_event, transcript, raw_profile, normalized_profile, emotion_confidence)
VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s)
"""

_CALIBRATION_MOMENT_SQL = """
INSERT INTO emotional_calibration_moments
    (session_id, turn_id, transcript, arche_question, user_answer,
     user_confirmed_or_corrected, inferred_emotional_pattern, normalized_profile)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
"""


@dataclass
class EmotionalDatasetConfig:
    enabled: bool
    db_url: str

    @classmethod
    def from_env(cls) -> "EmotionalDatasetConfig":
        return cls(
            enabled=_env_bool("VOICE_PROFILE_DATASET_ENABLED", False),
            db_url=(os.getenv("DATABASE_URL") or "").strip(),
        )

    def is_usable(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "dataset_disabled"
        if not self.db_url:
            return False, "database_url_missing"
        return True, "ok"


class EmotionalDatasetWriter:
    """Bounded-queue background writer; enqueue methods never block or raise."""

    def __init__(
        self,
        config: EmotionalDatasetConfig,
        db_writer: Callable[[str, tuple], None] | None = None,
        max_queue_rows: int = MAX_QUEUE_ROWS,
    ) -> None:
        self.config = config
        self._db_writer = db_writer or self._psycopg_writer
        self._queue: deque[tuple[str, tuple]] = deque(maxlen=max_queue_rows)
        self._queue_event = asyncio.Event()
        self._closed = False
        self._flush_task: asyncio.Task | None = None
        self.counters = {
            "rows_queued": 0,
            "rows_written": 0,
            "rows_dropped": 0,
            "write_errors": 0,
        }

    def _psycopg_writer(self, sql: str, params: tuple) -> None:
        import psycopg

        with psycopg.connect(self.config.db_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

    def _enqueue(self, sql: str, params: tuple) -> None:
        if self._closed:
            return
        if len(self._queue) >= (self._queue.maxlen or MAX_QUEUE_ROWS):
            self.counters["rows_dropped"] += 1
        self._queue.append((sql, params))
        self.counters["rows_queued"] += 1
        self._queue_event.set()

    def record_profile_event(
        self,
        *,
        session_id: str,
        source_event: str,
        raw_profile: dict[str, Any],
        normalized_profile: dict[str, Any],
        transcript: str | None,
        emotion_confidence: float,
    ) -> None:
        import json

        self._enqueue(
            _PROFILE_EVENT_SQL,
            (
                session_id,
                source_event,
                transcript,
                json.dumps(raw_profile, ensure_ascii=False),
                json.dumps(normalized_profile, ensure_ascii=False),
                emotion_confidence,
            ),
        )

    def record_calibration_moment(self, moment: dict[str, Any]) -> None:
        import json

        self._enqueue(
            _CALIBRATION_MOMENT_SQL,
            (
                str(moment.get("session_id") or ""),
                str(moment.get("turn_id") or ""),
                moment.get("transcript"),
                moment.get("arche_question"),
                moment.get("user_answer"),
                bool(moment.get("user_confirmed_or_corrected")),
                moment.get("inferred_emotional_pattern"),
                json.dumps(moment.get("normalized_profile") or {}, ensure_ascii=False),
            ),
        )

    def start(self) -> None:
        if self._flush_task is None and not self._closed:
            self._flush_task = asyncio.get_running_loop().create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while not self._closed:
            if not self._queue:
                self._queue_event.clear()
                await self._queue_event.wait()
                continue
            await self._flush_pending()

    async def _flush_pending(self) -> None:
        while self._queue:
            sql, params = self._queue.popleft()
            try:
                await asyncio.to_thread(self._db_writer, sql, params)
                self.counters["rows_written"] += 1
            except Exception as exc:  # noqa: BLE001 - dataset writes are best-effort
                self.counters["write_errors"] += 1
                logger.warning(
                    "emotional_dataset_write_failed=true error_type=%s write_errors=%s",
                    type(exc).__name__,
                    self.counters["write_errors"],
                )

    async def aclose(self) -> None:
        self._closed = True
        self._queue_event.set()
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):
                pass
            self._flush_task = None
        # Final best-effort drain so a short session's rows aren't lost.
        await self._flush_pending()
        logger.info(
            "emotional_dataset_summary rows_queued=%s rows_written=%s rows_dropped=%s write_errors=%s",
            self.counters["rows_queued"],
            self.counters["rows_written"],
            self.counters["rows_dropped"],
            self.counters["write_errors"],
        )


def build_emotional_dataset_writer_from_env() -> EmotionalDatasetWriter | None:
    config = EmotionalDatasetConfig.from_env()
    usable, reason = config.is_usable()
    logger.info(
        "emotional_dataset_config enabled=%s usable=%s skip_reason=%s",
        config.enabled,
        usable,
        "none" if usable else reason,
    )
    if not usable:
        return None
    return EmotionalDatasetWriter(config)


# ---------------------------------------------------------------------------
# Calibration (ground truth) heuristics.
# ---------------------------------------------------------------------------

_EMOTIONALLY_RELEVANT_TOKENS = (
    "feel", "feeling", "felt", "stressed", "worry", "worried", "hard", "heavy",
    "mad", "angry", "upset", "confused", "pressure", "scared", "fear", "guilt",
    "disappointed", "frustrated", "anxious", "anxiety", "unclear",
)


def calibration_reflection_for(
    transcript: str, profile: Any, mood_summary: str | None = None
) -> tuple[str | None, str]:
    """Pick an open, conversational check-in direction for this turn, or None.

    Returns a *direction* for Arche to reflect in her own words — not a fixed
    either/or question. The point is a natural reflection that invites the user
    to say more, so their free-form reply (not an A/B pick) becomes the ground
    truth. When a rich mood read is available (mood_context), the reflection is
    grounded in it; otherwise it falls back to open, dial-based directions.

    ``profile`` is a NormalizedVoiceProfile (or None). Never claims anything was
    detected and never comments on how the user sounds. Cadence limits live in
    CalibrationTracker, not here.
    """
    if profile is None:
        return None, "no_voice_profile_context"
    words = transcript.split()
    if len(words) < 4:
        return None, "transcript_too_short"
    text = transcript.lower()
    emotionally_relevant = any(token in text for token in _EMOTIONALLY_RELEVANT_TOKENS)
    profile_signal = (
        profile.tension == "high"
        or profile.certainty == "low"
        or profile.energy in {"low", "high"}
    )
    if not (emotionally_relevant or profile_signal):
        return None, "not_emotionally_useful"
    # When a rich mood read is available, ground the open reflection in it —
    # bespoke per turn rather than a template.
    if mood_summary and mood_summary.strip():
        return (
            f"gently reflect this sense of where they're at — {mood_summary.strip()} — "
            "and invite them to say more in their own words",
            "mood_grounded",
        )
    # Fallback: open, dial-based reflection directions (no either/or, no yes/no).
    if profile.certainty == "low":
        return (
            "gently reflect that a lot seems to be swirling in this and invite them "
            "to stay with it a moment and say more",
            "low_certainty_or_ambiguous",
        )
    if "choice" in text or "decide" in text or "decision" in text:
        return (
            "openly reflect the weight around this choice and ask what the heaviest "
            "part of it is for them",
            "choice_pressure",
        )
    if "frustrat" in text or "mad" in text or "angry" in text:
        return (
            "openly reflect that something in this seems to be grating on them and "
            "invite what's underneath it",
            "frustration_ambiguous",
        )
    if "anx" in text or "worr" in text or profile.tension == "high":
        return (
            "gently reflect the tension in this and invite them to name where it's "
            "sitting for them",
            "high_tension_or_worry",
        )
    if profile.energy == "low":
        return (
            "openly reflect that this seems to sit heavy and invite whether it feels "
            "lighter or heavier to say out loud",
            "low_energy_reflection",
        )
    return None, "no_matching_calibration_prompt"


def format_confirmed_pattern_content(pattern: str, user_answer: str) -> str:
    """Build the durable-memory content string for a confirmed pattern.

    Always starts with EMOTIONAL_PATTERN_PREFIX (required for
    MemoryLayer.preload() -> partition_emotional_patterns() to file this under
    "what we've learned" rather than general memory) and, when present,
    includes the user's own words — that's the actual ground truth, not just
    our internal energy/tension/certainty labels.
    """
    content = f"{EMOTIONAL_PATTERN_PREFIX}{pattern}"
    user_answer = (user_answer or "").strip()
    if user_answer:
        content += f" — user said: {user_answer}"
    return content


def remember_confirmed_pattern(memory_layer: Any, moment: dict[str, Any]) -> None:
    """Best-effort durable write of a user-confirmed emotional pattern.

    No-ops for a missing memory layer, an unconfirmed/corrected-away moment, or
    an empty inferred pattern. Called by the Inworld Realtime bridge so a
    confirmed calibration moment informs future sessions via the same
    MemoryLayer.preload() path. Never raises — a memory write must not break
    the turn that triggered it.
    """
    if memory_layer is None or not moment.get("user_confirmed_or_corrected"):
        return
    pattern = str(moment.get("inferred_emotional_pattern") or "").strip()
    if not pattern or pattern == "none":
        return
    turn_id_raw = moment.get("turn_id")
    try:
        turn_id = int(turn_id_raw) if turn_id_raw not in (None, "") else None
    except (TypeError, ValueError):
        turn_id = None
    try:
        memory_layer.schedule_remember(
            role="emotional_calibration",
            content=format_confirmed_pattern_content(pattern, str(moment.get("user_answer") or "")),
            turn_id=turn_id,
        )
    except Exception as exc:  # noqa: BLE001 - memory write must never break the turn
        logger.warning(
            "emotional_calibration_pattern_remember_failed=true error_type=%s error=%s",
            type(exc).__name__, exc,
        )


@dataclass
class CalibrationTracker:
    """One pending open reflection at a time; the next user turn answers it.

    Pure state machine (no I/O) so it's unit-testable: feed it each completed
    user transcript, the freshest normalized profile, and (optionally) the
    current mood read; get back an optional completed ground-truth moment and
    an optional new open-reflection direction for Arche to voice. The user's
    free-form reply on the following turn is the ground-truth label.
    """

    session_id: str
    min_user_turns_between: int = CALIBRATION_MIN_USER_TURNS_BETWEEN
    pending: dict[str, Any] | None = None
    _user_turns_seen: int = 0
    _last_question_user_turn: int = field(default=-(10**9))

    def on_user_transcript(
        self, transcript: str, profile: Any, mood_summary: str | None = None
    ) -> tuple[dict[str, Any] | None, str | None]:
        self._user_turns_seen += 1

        completed: dict[str, Any] | None = None
        if self.pending is not None:
            completed = dict(self.pending)
            completed["user_answer"] = transcript
            completed["user_confirmed_or_corrected"] = bool(transcript.strip())
            self.pending = None

        reflection: str | None = None
        if self._user_turns_seen - self._last_question_user_turn >= self.min_user_turns_between:
            candidate, reason = calibration_reflection_for(transcript, profile, mood_summary)
            logger.info(
                "emotional_calibration_reflection_armed=%s reason=%s user_turn=%s mood_grounded=%s",
                bool(candidate),
                reason,
                self._user_turns_seen,
                bool(mood_summary and mood_summary.strip()),
            )
            if candidate:
                reflection = candidate
                self._last_question_user_turn = self._user_turns_seen
                # The analyzer's read at this moment — the dials, plus the mood
                # fusion when available (the richer inference the user's
                # free-form reply is the ground truth against).
                if profile is not None:
                    inferred = (
                        f"energy={profile.energy}; tension={profile.tension}; "
                        f"certainty={profile.certainty}"
                    )
                    if mood_summary and mood_summary.strip():
                        inferred += f"; mood: {mood_summary.strip()}"
                else:
                    inferred = "none"
                self.pending = {
                    "session_id": self.session_id,
                    "turn_id": str(self._user_turns_seen),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "transcript": transcript,
                    "normalized_profile": profile.to_dict() if profile is not None else {},
                    # Stores the reflection *direction* (what Arche was nudged to
                    # voice), not verbatim; the column name is kept for schema
                    # compatibility.
                    "arche_question": reflection,
                    "user_answer": "",
                    "inferred_emotional_pattern": inferred,
                    "user_confirmed_or_corrected": False,
                }
        return completed, reflection

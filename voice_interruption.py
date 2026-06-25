"""Pure logic for confirmed-interruption gating and assistant-speech tail-outcome
classification.

The live `interrupted=True` flag is too broad: noise, echo, breath, a brief blip,
or even Arche's own tail audio can flip it, and not every interruption is an
audible cutoff. This module separates *candidate* mic activity from a *confirmed*
barge-in, and classifies what actually happened to each assistant speech based on
playout timing + audio lifecycle — so observability and behavior stop assuming
every interruption cut the user off.

No agent/livekit imports, so it's fast and deterministic to unit test.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Single-word commands strong enough to confirm an interruption on their own.
STRONG_COMMANDS = {"stop", "wait", "pause", "hold", "no"}

# --- tail outcome labels ---
CLEAN_PLAYOUT = "clean_playout"
LIKELY_TAIL_CUT = "likely_tail_cut"  # interrupted before playout completed (audible)
INTERRUPTION_AFTER_PLAYOUT = "interruption_after_playout_complete"  # not a cutoff
GHOST_STALE_NO_AUDIO = "ghost_stale_no_audio"  # handle produced no Hume audio
STALE_CLEANUP_ONLY = "stale_cleanup_only"  # bookkeeping, not a real interruption

VALID_TURN_DETECTION_MODES = ("vad", "stt", "default")

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def resolve_turn_detection_mode(raw: str | None) -> tuple[str, bool]:
    """Map a raw LIVEKIT_TURN_DETECTION_MODE value to a resolved mode.

    Returns (resolved, recognized). Only 'vad' | 'stt' | 'default' are valid;
    anything else (notably 'audio', which is NOT a valid mode in this code path)
    is unrecognized and resolves to 'vad'.
    """
    norm = (raw or "").strip().lower()
    if norm in VALID_TURN_DETECTION_MODES:
        return norm, True
    return "vad", False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def meaningful_words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def has_strong_command(text: str) -> bool:
    return any(w in STRONG_COMMANDS for w in meaningful_words(text))


def is_echo(transcript: str, recent_assistant_texts, *, min_overlap: float = 0.6) -> bool:
    """True if the candidate transcript looks like an echo/duplicate of Arche's
    recent speech (high token overlap) rather than a genuine user utterance."""
    cand = set(meaningful_words(transcript))
    if not cand:
        return False
    for prior in recent_assistant_texts or []:
        prior_words = set(meaningful_words(prior))
        if not prior_words:
            continue
        overlap = len(cand & prior_words) / len(cand)
        if overlap >= min_overlap:
            return True
    return False


@dataclass
class InterruptionConfig:
    min_words: int = 2
    min_duration: float = 0.65
    resume_false_interruption: bool = True
    false_interruption_timeout: float = 1.0

    @classmethod
    def from_env(cls) -> "InterruptionConfig":
        return cls(
            min_words=_env_int("LIVEKIT_INTERRUPTION_MIN_WORDS", 2),
            min_duration=_env_float("LIVEKIT_INTERRUPTION_MIN_DURATION", 0.65),
            resume_false_interruption=_env_bool("LIVEKIT_RESUME_FALSE_INTERRUPTION", True),
            false_interruption_timeout=_env_float("LIVEKIT_FALSE_INTERRUPTION_TIMEOUT", 1.0),
        )


def classify_interruption_candidate(
    *,
    transcript: str,
    duration_s: float,
    recent_assistant_texts=None,
    config: InterruptionConfig,
) -> tuple[str, str]:
    """Decide whether mic/audio activity is a real interruption.

    Returns (decision, reason) where decision is one of:
      - "confirmed": genuine barge-in -> cancel assistant speech
      - "false":     not an interruption (echo, or no transcript past the timeout)
                      -> resume/continue assistant playout
      - "pending":   not enough evidence yet -> keep playing, keep listening

    Confirmed only if (2+ meaningful words OR a strong command) AND duration >=
    min_duration AND the transcript is not an echo of Arche's recent speech.
    """
    text = (transcript or "").strip()
    words = meaningful_words(text)

    if text and is_echo(text, recent_assistant_texts):
        return "false", "echo_of_assistant"

    long_enough = duration_s >= config.min_duration
    strong = has_strong_command(text)
    enough_words = len(words) >= config.min_words

    if long_enough and (strong or enough_words):
        return "confirmed", "strong_command" if strong else "enough_words"

    if not text:
        if duration_s >= config.false_interruption_timeout:
            return "false", "no_transcript_timeout"
        return "pending", "awaiting_transcript"

    if not long_enough:
        return "pending", "too_short"
    return "pending", "insufficient_words"


def classify_tail_outcome(
    *,
    generated_audio_duration_s: float | None,
    playout_started_at: float | None,
    playout_completed_at: float | None,
    interrupted_at: float | None,
    interrupted: bool,
    was_stale: bool = False,
    was_active: bool = True,
    hume_requests_during_speech: int = 0,
) -> str:
    """Classify what actually happened to an assistant speech, using playout
    timing + audio lifecycle as the source of truth (not the broad `interrupted`
    flag). See the *_OUTCOME constants for the labels."""
    # Stale handle cleanup is bookkeeping, never a user-facing interruption.
    if was_stale and not was_active:
        return STALE_CLEANUP_ONLY

    produced_audio = (
        (generated_audio_duration_s or 0) > 0
        and (hume_requests_during_speech or 0) > 0
        and playout_started_at is not None
    )
    if not produced_audio:
        # No audio reached the user -> ghost handle, not an audible cutoff.
        return GHOST_STALE_NO_AUDIO

    if not interrupted or interrupted_at is None:
        return CLEAN_PLAYOUT

    # Interrupted after the audio finished playing -> not an audible cutoff.
    if playout_completed_at is not None and interrupted_at >= playout_completed_at:
        return INTERRUPTION_AFTER_PLAYOUT

    # Interrupted while audio was still playing -> the user likely heard a cut tail.
    return LIKELY_TAIL_CUT


def is_audible_cutoff(outcome: str) -> bool:
    """Only a likely tail cut is a real user-facing audible cutoff."""
    return outcome == LIKELY_TAIL_CUT

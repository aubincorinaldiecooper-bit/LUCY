"""Explicit voice interaction state layer for the LUCY pipeline.

Borrowed from VITA-MLLM/LUCY (https://github.com/VITA-MLLM/LUCY):
- explicit conversation/control states instead of implicit flag soup
- full-duplex awareness: user speech during assistant output is its own state
  (USER_INTERRUPTING), not just an event log line
- separation of tool/action turns from normal companion conversation turns
- a visible state-transition trail so a bad turn can be reconstructed from logs

Deliberately NOT borrowed: the VITA model, its audio codecs, or any runtime
dependency. This layer is advisory and observational — it sits beside the
existing STT/VAD -> context layer -> TurnPolicy -> OpenRouter -> TTS pipeline,
mirrors what those components decide, and makes every major transition
loggable and testable. It never blocks or alters the live path; an unexpected
transition is logged and permitted rather than raised.

Log lines to search in Railway:
- interaction_state_transition       every state change with from/to/reason
- interaction_turn_kind              conversation | action | recovery | unclear_audio | filler
- assistant_speech_overlap_detected  assistant audio started while user active
- fallback_state_check               fallback allowed/suppressed with state context
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Interaction states
LISTENING = "LISTENING"
USER_SPEAKING = "USER_SPEAKING"
USER_TURN_CANDIDATE = "USER_TURN_CANDIDATE"
HOLDING_FRAGMENT = "HOLDING_FRAGMENT"
COMMITTED_TURN = "COMMITTED_TURN"
ASSISTANT_THINKING = "ASSISTANT_THINKING"
TOOL_CALL_PENDING = "TOOL_CALL_PENDING"
ASSISTANT_SPEAKING = "ASSISTANT_SPEAKING"
USER_INTERRUPTING = "USER_INTERRUPTING"
RECOVERY = "RECOVERY"

ALL_STATES = {
    LISTENING,
    USER_SPEAKING,
    USER_TURN_CANDIDATE,
    HOLDING_FRAGMENT,
    COMMITTED_TURN,
    ASSISTANT_THINKING,
    TOOL_CALL_PENDING,
    ASSISTANT_SPEAKING,
    USER_INTERRUPTING,
    RECOVERY,
}

# Turn kinds (VITA-style response-mode / tool-call separation)
TURN_KIND_CONVERSATION = "conversation"
TURN_KIND_ACTION = "action"
TURN_KIND_RECOVERY = "recovery"
TURN_KIND_UNCLEAR_AUDIO = "unclear_audio"
TURN_KIND_FILLER = "filler"

_ACTION_INTENTS = {
    "tool_request_search",
    "tool_request_email",
    "tool_request_document",
    "date_time_question",
    "timer_request",
    "reminder_request",
    "calculation_request",
    "counting_request",
    "voice_change_request",
    "language_request",
}

# Expected transitions. Anything outside this map is logged as unexpected but
# still applied: the live session must never be hostage to the state model.
EXPECTED_TRANSITIONS: dict[str, set[str]] = {
    LISTENING: {USER_SPEAKING, USER_TURN_CANDIDATE, ASSISTANT_THINKING, ASSISTANT_SPEAKING, COMMITTED_TURN, HOLDING_FRAGMENT, RECOVERY},
    USER_SPEAKING: {USER_TURN_CANDIDATE, LISTENING, USER_INTERRUPTING},
    USER_TURN_CANDIDATE: {COMMITTED_TURN, HOLDING_FRAGMENT, RECOVERY, LISTENING, USER_SPEAKING},
    HOLDING_FRAGMENT: {COMMITTED_TURN, USER_SPEAKING, USER_TURN_CANDIDATE, LISTENING, RECOVERY},
    COMMITTED_TURN: {ASSISTANT_THINKING, TOOL_CALL_PENDING, ASSISTANT_SPEAKING, USER_SPEAKING, USER_INTERRUPTING, LISTENING},
    ASSISTANT_THINKING: {ASSISTANT_SPEAKING, TOOL_CALL_PENDING, USER_INTERRUPTING, LISTENING, COMMITTED_TURN},
    TOOL_CALL_PENDING: {ASSISTANT_THINKING, ASSISTANT_SPEAKING, USER_INTERRUPTING, LISTENING},
    ASSISTANT_SPEAKING: {LISTENING, USER_INTERRUPTING, ASSISTANT_SPEAKING, ASSISTANT_THINKING},
    USER_INTERRUPTING: {USER_TURN_CANDIDATE, USER_SPEAKING, LISTENING, COMMITTED_TURN, HOLDING_FRAGMENT, RECOVERY},
    RECOVERY: {ASSISTANT_THINKING, ASSISTANT_SPEAKING, LISTENING, USER_SPEAKING},
}


@dataclass(slots=True)
class AudioEnvironmentDecision:
    noise_state: str  # clean | noisy | uncertain
    noise_confidence: float
    speech_stability: str  # stable | unstable | unknown
    transcript_stability: str  # stable | unstable | unknown
    false_speech_start_count_recent: int
    candidate_turn_count_recent: int
    short_noisy_fragment_detected: bool
    action_hint: str  # normal | hold | ask_repair | audio_status
    reason: str


def build_audio_environment_decision(
    *,
    false_speech_start_count_recent: int = 0,
    candidate_turn_count_recent: int = 0,
    short_noisy_fragment_detected: bool = False,
    unstable_partial_transcripts: bool = False,
    low_final_transcript_rate: bool = False,
    snr_db: float | None = None,
    is_audio_status_check: bool = False,
) -> AudioEnvironmentDecision:
    """Lightweight, deterministic audio/noise state from existing pipeline signals.

    Heuristic by design (no biometric/speaker work): repeated false speech starts,
    churn of turn candidates, short noisy fragments, and unstable/low transcripts
    indicate a noisy or unstable environment. An explicit SNR reading, when
    available, takes precedence.
    """
    instability = 0
    if false_speech_start_count_recent >= 3:
        instability += 1
    if candidate_turn_count_recent >= 3:
        instability += 1
    if short_noisy_fragment_detected:
        instability += 1
    if unstable_partial_transcripts:
        instability += 1
    if low_final_transcript_rate:
        instability += 1

    speech_stability = "unstable" if (false_speech_start_count_recent >= 3 or candidate_turn_count_recent >= 4) else "stable"
    transcript_stability = "unstable" if (unstable_partial_transcripts or low_final_transcript_rate) else "stable"

    if snr_db is not None:
        if snr_db >= 18.0:
            noise_state, noise_confidence = "clean", 0.85
        elif snr_db < 8.0:
            noise_state, noise_confidence = "noisy", 0.85
        else:
            noise_state, noise_confidence = "uncertain", 0.5
    elif instability >= 3:
        noise_state, noise_confidence = "noisy", min(0.5 + 0.1 * instability, 0.9)
    elif instability == 0:
        noise_state, noise_confidence = "clean", 0.7
    else:
        noise_state, noise_confidence = "uncertain", 0.5

    if is_audio_status_check:
        action_hint = "audio_status"
    elif noise_state == "noisy" and transcript_stability == "unstable":
        action_hint = "ask_repair"
    elif noise_state == "noisy":
        action_hint = "hold"
    else:
        action_hint = "normal"

    reason = (
        f"instability={instability} false_starts={false_speech_start_count_recent} "
        f"candidates={candidate_turn_count_recent} short_fragment={short_noisy_fragment_detected} "
        f"unstable_partials={unstable_partial_transcripts} low_finals={low_final_transcript_rate} "
        f"snr_db={'n/a' if snr_db is None else round(snr_db, 1)}"
    )
    return AudioEnvironmentDecision(
        noise_state=noise_state,
        noise_confidence=round(noise_confidence, 2),
        speech_stability=speech_stability,
        transcript_stability=transcript_stability,
        false_speech_start_count_recent=false_speech_start_count_recent,
        candidate_turn_count_recent=candidate_turn_count_recent,
        short_noisy_fragment_detected=short_noisy_fragment_detected,
        action_hint=action_hint,
        reason=reason,
    )


def classify_turn_kind(detected_intent: str | None, policy_classification: str | None, policy_decision: str | None) -> str:
    decision = (policy_decision or "").strip().upper()
    classification = (policy_classification or "").strip().upper()
    intent = (detected_intent or "").strip().lower()
    if decision == "IGNORE_LOW_INFORMATION_FILLER":
        return TURN_KIND_FILLER
    if decision == "RECOVER_FROM_SILENCE" or classification == "META_COMPLAINT":
        return TURN_KIND_RECOVERY
    if classification == "UNCLEAR_AUDIO":
        return TURN_KIND_UNCLEAR_AUDIO
    if intent in _ACTION_INTENTS:
        return TURN_KIND_ACTION
    return TURN_KIND_CONVERSATION


class InteractionStateMachine:
    """Mirrors pipeline decisions as explicit states. Logging-first, never raises."""

    def __init__(self) -> None:
        self.state = LISTENING
        self.previous_state = LISTENING
        self.state_entered_at = time.monotonic()
        self.turn_id = 0
        self.turn_kind = TURN_KIND_CONVERSATION
        self.detected_intent = "unknown"
        self.unexpected_transition_count = 0
        self.overlap_count = 0

    # ---------- core ----------

    def transition(self, to_state: str, reason: str) -> None:
        if to_state not in ALL_STATES:
            logger.warning("interaction_state_transition_invalid_target=true to=%s reason=%s", to_state, reason)
            return
        from_state = self.state
        if to_state == from_state:
            return
        expected = to_state in EXPECTED_TRANSITIONS.get(from_state, set())
        if not expected:
            self.unexpected_transition_count += 1
            logger.warning(
                "interaction_state_transition_unexpected=true from=%s to=%s reason=%s turn_id=%s",
                from_state,
                to_state,
                reason,
                self.turn_id,
            )
        now = time.monotonic()
        seconds_in_previous = now - self.state_entered_at
        self.previous_state = from_state
        self.state = to_state
        self.state_entered_at = now
        logger.info(
            "interaction_state_transition from=%s to=%s reason=%s turn_id=%s turn_kind=%s seconds_in_previous_state=%.3f",
            from_state,
            to_state,
            reason,
            self.turn_id,
            self.turn_kind,
            seconds_in_previous,
        )

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "previous_state": self.previous_state,
            "turn_id": self.turn_id,
            "turn_kind": self.turn_kind,
            "detected_intent": self.detected_intent,
            "seconds_in_state": time.monotonic() - self.state_entered_at,
        }

    # ---------- user speech (full-duplex awareness) ----------

    def on_user_speech_started(self) -> None:
        if self.state in {ASSISTANT_SPEAKING, ASSISTANT_THINKING, TOOL_CALL_PENDING}:
            self.transition(USER_INTERRUPTING, reason=f"user_started_speaking_during_{self.state.lower()}")
        else:
            self.transition(USER_SPEAKING, reason="user_started_speaking")

    def on_user_speech_stopped(self) -> None:
        if self.state in {USER_SPEAKING, USER_INTERRUPTING}:
            self.transition(USER_TURN_CANDIDATE, reason="user_stopped_speaking")

    # ---------- turn lifecycle ----------

    def begin_turn(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self.turn_kind = TURN_KIND_CONVERSATION
        self.detected_intent = "unknown"
        if self.state not in {USER_TURN_CANDIDATE, USER_SPEAKING, USER_INTERRUPTING}:
            self.transition(USER_TURN_CANDIDATE, reason="turn_committed_by_pipeline_without_observed_user_speech")

    def set_turn_kind(self, turn_kind: str, detected_intent: str | None = None) -> None:
        self.turn_kind = turn_kind
        self.detected_intent = (detected_intent or "unknown").strip() or "unknown"
        logger.info(
            "interaction_turn_kind turn_id=%s turn_kind=%s detected_intent=%s",
            self.turn_id,
            self.turn_kind,
            self.detected_intent,
        )

    def on_turn_policy(self, decision: str, classification: str, reason: str) -> None:
        decision = (decision or "").strip().upper()
        policy_reason = f"turn_policy_{decision.lower()}:{reason}"
        if decision == "HOLD_FOR_CONTINUATION":
            self.transition(HOLDING_FRAGMENT, reason=policy_reason)
        elif decision == "RECOVER_FROM_SILENCE":
            self.transition(RECOVERY, reason=policy_reason)
        elif decision == "IGNORE_LOW_INFORMATION_FILLER":
            self.transition(LISTENING, reason=policy_reason)
        else:
            # COMMIT_NOW, MERGE_WITH_HELD_FRAGMENT, FLUSH_HELD_AND_COMMIT_NEW
            self.transition(COMMITTED_TURN, reason=policy_reason)

    def on_hold_deadline_commit(self) -> None:
        self.transition(COMMITTED_TURN, reason="held_fragment_reply_deadline_expired")

    # ---------- assistant lifecycle ----------

    def on_llm_started(self) -> None:
        self.transition(ASSISTANT_THINKING, reason="llm_stream_starting")

    def on_tool_call_started(self, tool_name: str) -> None:
        self.transition(TOOL_CALL_PENDING, reason=f"tool_call_started:{tool_name}")

    def on_tool_call_finished(self, tool_name: str) -> None:
        if self.state == TOOL_CALL_PENDING:
            self.transition(ASSISTANT_THINKING, reason=f"tool_call_finished:{tool_name}")

    def on_assistant_speech_started(self, speech_id: str) -> bool:
        """Returns True when assistant audio is starting while the user is active (overlap)."""
        overlap = self.state in {USER_SPEAKING, USER_INTERRUPTING}
        if overlap:
            self.overlap_count += 1
            logger.warning(
                "assistant_speech_overlap_detected=true interaction_state=%s speech_id=%s turn_id=%s overlap_count=%s",
                self.state,
                speech_id,
                self.turn_id,
                self.overlap_count,
            )
        self.transition(ASSISTANT_SPEAKING, reason=f"assistant_speech_started:{speech_id}")
        return overlap

    def on_assistant_speech_finished(self, interrupted: bool) -> None:
        if self.state == ASSISTANT_SPEAKING:
            self.transition(LISTENING, reason="assistant_speech_finished_interrupted" if interrupted else "assistant_speech_finished")

    # ---------- fallback visibility ----------

    def on_fallback_decision(self, allowed: bool, reason: str, requires_repeat: bool) -> None:
        logger.warning(
            "fallback_state_check interaction_state=%s turn_id=%s turn_kind=%s fallback_allowed=%s fallback_reason=%s fallback_requires_user_repeat=%s",
            self.state,
            self.turn_id,
            self.turn_kind,
            allowed,
            reason,
            requires_repeat,
        )

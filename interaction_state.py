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

# How long after an observed user-speech event a pipeline turn commit is still
# attributable to that speech. Beyond this, a commit with the FSM already past
# the user-speech states is treated as a genuine no-observed-speech anomaly.
USER_SPEECH_OBSERVATION_WINDOW_SECONDS = 12.0

# Revalidation taxonomy for a tool/search result whose authority to speak was
# paused by user speech mid tool call. Only clearly-additive context lets the
# in-flight result keep its right to speak; everything else must re-run or defer
# so a stale result cannot regain conversational authority after the user moved on.
TOOL_RESULT_REVALIDATION_AUTHORITY: dict[str, bool] = {
    "additive_context": True,
    "additive": True,
    "continuation": True,
    "correction": False,
    "refinement": False,
    "cancellation": False,
    "pivot": False,
    "meta_complaint": False,
    "unrelated": False,
    "new_turn": False,
}

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
        # Ownership / lifecycle bookkeeping --------------------------------
        # Whether real user speech was observed for the turn currently being
        # committed. Consumed (reset) by begin_turn so each turn evaluates
        # fresh speech rather than inheriting a stale flag.
        self._user_speech_seen = False
        self._last_user_speech_at = 0.0
        # Speech objects that exist (LLM/TTS scheduled) but have not begun real
        # audio playout. ASSISTANT_SPEAKING is reserved for actual playout.
        self._pending_speech_ids: set[str] = set()
        # The speech id whose audio is actively playing out. Set when playout
        # begins and cleared the moment it finishes/cancels, so after an
        # interruption nothing still points at a dead speech.
        self.active_speech_id: str | None = None
        # True once the active assistant speech was interrupted by the user, so
        # the interruption is recorded immediately (not only when the handle
        # later resolves). Reset when a new speech starts.
        self.active_speech_interrupted = False
        # Tool/search result authority to speak. A result earns authority when
        # its tool call starts and loses it the moment the user speaks during
        # TOOL_CALL_PENDING, until the newer utterance is classified.
        self.tool_result_speak_authority = True
        self.tool_result_pending_revalidation = False
        self.tool_result_paused_reason = ""
        self.tool_result_relationship = ""
        self.tool_result_resume_decision = ""
        # Count of high-risk runtime gates this machine has blocked (observability).
        self.gate_blocked_count = 0

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
            "pending_speech_count": len(self._pending_speech_ids),
            "active_speech_id": self.active_speech_id,
            "active_speech_interrupted": self.active_speech_interrupted,
            "tool_result_speak_authority": self.tool_result_speak_authority,
            "tool_result_pending_revalidation": self.tool_result_pending_revalidation,
            "tool_result_resume_decision": self.tool_result_resume_decision,
            "gate_blocked_count": self.gate_blocked_count,
        }

    # ---------- user speech (full-duplex awareness) ----------

    def on_user_speech_started(self) -> None:
        # Record that real user speech was observed for the turn currently in
        # flight, so a later pipeline commit can be attributed correctly even if
        # the FSM state has already advanced past the user-speech states.
        self._user_speech_seen = True
        self._last_user_speech_at = time.monotonic()
        # User speech during a tool call must not silently cancel the in-flight
        # result, but it does pause the result's automatic right to speak until
        # the new utterance is classified (additive / correction / cancel / ...).
        if self.state == TOOL_CALL_PENDING:
            self._pause_tool_result_authority("user_spoke_during_tool_call")
        if self.state in {ASSISTANT_SPEAKING, ASSISTANT_THINKING, TOOL_CALL_PENDING}:
            # Mark the active assistant speech interrupted immediately, so the
            # interruption is known right away rather than only when the speech
            # handle later resolves.
            if self.active_speech_id is not None:
                self.active_speech_interrupted = True
                logger.info(
                    "active_speech_marked_interrupted=true speech_id=%s interaction_state=%s turn_id=%s",
                    self.active_speech_id,
                    self.state,
                    self.turn_id,
                )
            self.transition(USER_INTERRUPTING, reason=f"user_started_speaking_during_{self.state.lower()}")
        else:
            self.transition(USER_SPEAKING, reason="user_started_speaking")

    def on_user_speech_stopped(self) -> None:
        if self.state in {USER_SPEAKING, USER_INTERRUPTING}:
            self.transition(USER_TURN_CANDIDATE, reason="user_stopped_speaking")

    # ---------- turn lifecycle ----------

    def begin_turn(self, turn_id: int, *, user_speech_observed: bool | None = None) -> None:
        self.turn_id = turn_id
        self.turn_kind = TURN_KIND_CONVERSATION
        self.detected_intent = "unknown"
        # Consume the per-turn observed-speech flag up front so each turn is
        # evaluated against its own speech, never an inherited one.
        seen_flag = self._user_speech_seen
        last_seen_at = self._last_user_speech_at
        self._user_speech_seen = False
        if self.state in {USER_TURN_CANDIDATE, USER_SPEAKING, USER_INTERRUPTING}:
            # Normal path: the FSM already saw the user speak for this turn.
            return
        # The FSM state has already advanced (LISTENING / ASSISTANT_* / RECOVERY).
        # Decide whether this commit is an explained lag (we did observe speech,
        # the state simply moved on) or a genuine no-observed-speech anomaly.
        if user_speech_observed is None:
            observed = seen_flag and (
                last_seen_at > 0.0
                and (time.monotonic() - last_seen_at) <= USER_SPEECH_OBSERVATION_WINDOW_SECONDS
            )
        else:
            observed = bool(user_speech_observed)
        if observed:
            self.transition(
                USER_TURN_CANDIDATE,
                reason="turn_committed_after_state_advanced_post_user_speech",
            )
        else:
            self.transition(
                USER_TURN_CANDIDATE,
                reason="turn_committed_by_pipeline_without_observed_user_speech",
            )

    # ---------- tool/search result authority ----------

    def _pause_tool_result_authority(self, reason: str) -> None:
        self.tool_result_speak_authority = False
        self.tool_result_pending_revalidation = True
        self.tool_result_paused_reason = reason
        logger.info(
            "tool_result_authority_paused_pending_revalidation=true reason=%s turn_id=%s interaction_state=%s",
            reason,
            self.turn_id,
            self.state,
        )

    def revalidate_tool_result(self, classification: str | None) -> bool:
        """Classify a barge-in utterance to decide if a paused tool result may speak.

        Returns the resulting speak authority. Only clearly-additive context
        restores authority; everything else (correction, cancel, pivot, meta,
        unrelated) keeps the in-flight result from speaking without a re-run.
        """
        cls = (classification or "").strip().lower()
        if not self.tool_result_pending_revalidation:
            return self.tool_result_speak_authority
        regained = TOOL_RESULT_REVALIDATION_AUTHORITY.get(cls, False)
        self.tool_result_speak_authority = regained
        self.tool_result_pending_revalidation = False
        logger.info(
            "tool_result_revalidated=true classification=%s tool_result_speak_authority=%s turn_id=%s",
            cls or "unknown",
            regained,
            self.turn_id,
        )
        return regained

    def apply_tool_resume_decision(
        self,
        *,
        relationship: str,
        decision: str,
        resolution: str,
        additive_allowed: bool,
    ) -> bool:
        """Apply a composer resume decision to the in-flight tool result.

        Authority to speak the existing result is granted only when the decision
        is to compose it with the newer utterance. rerun/withhold/discard/defer/
        clarify all withhold the stale result's right to speak on its own.
        """
        self.tool_result_relationship = relationship
        self.tool_result_resume_decision = decision
        self.tool_result_speak_authority = bool(additive_allowed) and decision == "compose"
        self.tool_result_pending_revalidation = False
        logger.info(
            "tool_result_resume_decision=%s tool_revalidation_class=%s context_resolution=%s "
            "tool_result_composed_with_newer_user_utterance=%s tool_result_authority_restored=%s "
            "stale_tool_result_blocked=%s turn_id=%s",
            decision,
            relationship,
            resolution,
            decision == "compose",
            self.tool_result_speak_authority,
            not self.tool_result_speak_authority,
            self.turn_id,
        )
        return self.tool_result_speak_authority

    # ---------- runtime enforcement (act, not just observe) ----------

    def runtime_gate(self, action: str, allowed: bool, reason: str) -> bool:
        """Record and return a high-risk gate decision.

        The FSM observes and logs; the caller acts on the returned boolean. A
        blocked gate is always logged so a denied high-risk action is traceable.
        """
        if allowed:
            logger.info(
                "fsm_gate_blocked=false fsm_gate_action=%s fsm_gate_reason=%s interaction_state=%s turn_id=%s",
                action,
                reason,
                self.state,
                self.turn_id,
            )
        else:
            self.gate_blocked_count += 1
            logger.warning(
                "fsm_gate_blocked=true fsm_gate_action=%s fsm_gate_reason=%s interaction_state=%s turn_id=%s gate_blocked_count=%s",
                action,
                reason,
                self.state,
                self.turn_id,
                self.gate_blocked_count,
            )
        return allowed

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
        # A fresh tool call owns the right to speak its own result until/unless
        # the user barges in during it.
        self.tool_result_speak_authority = True
        self.tool_result_pending_revalidation = False
        self.tool_result_paused_reason = ""
        self.tool_result_relationship = ""
        self.tool_result_resume_decision = ""
        self.transition(TOOL_CALL_PENDING, reason=f"tool_call_started:{tool_name}")

    def on_tool_call_finished(self, tool_name: str) -> None:
        if self.state == TOOL_CALL_PENDING:
            self.transition(ASSISTANT_THINKING, reason=f"tool_call_finished:{tool_name}")

    def on_assistant_speech_created(self, speech_id: str) -> None:
        """A speech OBJECT was created (LLM/TTS scheduled) — not real audio yet.

        Deliberately does NOT transition to ASSISTANT_SPEAKING. The assistant is
        only SPEAKING once real audio playout begins (on_assistant_speech_started),
        so downstream logic never believes audio is playing on object creation.
        """
        self._pending_speech_ids.add(speech_id)
        logger.info(
            "assistant_speech_object_created=true speech_id=%s interaction_state=%s turn_id=%s pending_speech_count=%s",
            speech_id,
            self.state,
            self.turn_id,
            len(self._pending_speech_ids),
        )

    def on_assistant_speech_started(self, speech_id: str) -> bool:
        """Real audio playout is beginning.

        Returns True when assistant audio is starting while the user is active
        (overlap).
        """
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
        self._pending_speech_ids.discard(speech_id)
        self.active_speech_id = speech_id
        self.active_speech_interrupted = overlap
        self.transition(ASSISTANT_SPEAKING, reason=f"assistant_speech_started:{speech_id}")
        return overlap

    def on_assistant_speech_finished(self, interrupted: bool, speech_id: str | None = None) -> None:
        if speech_id is not None:
            self._pending_speech_ids.discard(speech_id)
        # Clear the active speech once it finishes/cancels so nothing keeps
        # pointing at a dead speech after an interruption. Only clear when it
        # matches (or no id was tracked) to avoid clobbering a newer speech.
        if speech_id is None or self.active_speech_id is None or self.active_speech_id == speech_id:
            self.active_speech_id = None
            self.active_speech_interrupted = False
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

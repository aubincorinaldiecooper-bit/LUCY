import unittest

import interaction_state
from interaction_state import (
    ASSISTANT_SPEAKING,
    ASSISTANT_THINKING,
    COMMITTED_TURN,
    HOLDING_FRAGMENT,
    LISTENING,
    RECOVERY,
    TOOL_CALL_PENDING,
    TURN_KIND_ACTION,
    TURN_KIND_CONVERSATION,
    TURN_KIND_FILLER,
    TURN_KIND_RECOVERY,
    TURN_KIND_UNCLEAR_AUDIO,
    USER_INTERRUPTING,
    USER_SPEAKING,
    USER_TURN_CANDIDATE,
    InteractionStateMachine,
    classify_turn_kind,
)


def committed_machine(turn_id: int = 1) -> InteractionStateMachine:
    sm = InteractionStateMachine()
    sm.on_user_speech_started()
    sm.on_user_speech_stopped()
    sm.begin_turn(turn_id)
    sm.on_turn_policy("COMMIT_NOW", "COMPLETE_THOUGHT", "complete_shape")
    return sm


class TurnKindClassificationTests(unittest.TestCase):
    def test_tool_and_action_intents_are_action_turns(self):
        for intent in (
            "tool_request_search",
            "tool_request_email",
            "tool_request_document",
            "date_time_question",
            "timer_request",
            "reminder_request",
            "calculation_request",
        ):
            self.assertEqual(classify_turn_kind(intent, "COMPLETE_THOUGHT", "COMMIT_NOW"), TURN_KIND_ACTION, intent)

    def test_normal_conversation_is_conversation_turn(self):
        self.assertEqual(classify_turn_kind("unknown", "EMOTIONAL_STATEMENT", "COMMIT_NOW"), TURN_KIND_CONVERSATION)
        self.assertEqual(classify_turn_kind(None, None, "COMMIT_NOW"), TURN_KIND_CONVERSATION)

    def test_meta_complaint_and_recovery_are_recovery_turns(self):
        self.assertEqual(classify_turn_kind("unknown", "META_COMPLAINT", "RECOVER_FROM_SILENCE"), TURN_KIND_RECOVERY)
        self.assertEqual(classify_turn_kind("unknown", "META_COMPLAINT", "COMMIT_NOW"), TURN_KIND_RECOVERY)

    def test_unclear_audio_turn_kind(self):
        self.assertEqual(classify_turn_kind("unknown", "UNCLEAR_AUDIO", "COMMIT_NOW"), TURN_KIND_UNCLEAR_AUDIO)

    def test_filler_turn_kind(self):
        self.assertEqual(classify_turn_kind("unknown", "LOW_INFORMATION", "IGNORE_LOW_INFORMATION_FILLER"), TURN_KIND_FILLER)

    def test_action_intent_during_recovery_stays_recovery(self):
        self.assertEqual(classify_turn_kind("tool_request_search", "META_COMPLAINT", "RECOVER_FROM_SILENCE"), TURN_KIND_RECOVERY)


class FragmentLifecycleTests(unittest.TestCase):
    def test_incomplete_fragment_holds(self):
        sm = InteractionStateMachine()
        sm.on_user_speech_started()
        self.assertEqual(sm.state, USER_SPEAKING)
        sm.on_user_speech_stopped()
        self.assertEqual(sm.state, USER_TURN_CANDIDATE)
        sm.begin_turn(1)
        sm.on_turn_policy("HOLD_FOR_CONTINUATION", "INCOMPLETE_THOUGHT", "structural_incomplete")
        self.assertEqual(sm.state, HOLDING_FRAGMENT)
        self.assertEqual(sm.unexpected_transition_count, 0)

    def test_held_fragment_merges_when_continuation_arrives(self):
        sm = InteractionStateMachine()
        sm.on_user_speech_started()
        sm.on_user_speech_stopped()
        sm.begin_turn(1)
        sm.on_turn_policy("HOLD_FOR_CONTINUATION", "INCOMPLETE_THOUGHT", "structural_incomplete")
        sm.on_user_speech_started()
        sm.on_user_speech_stopped()
        sm.begin_turn(2)
        sm.on_turn_policy("MERGE_WITH_HELD_FRAGMENT", "COMPLETE_THOUGHT", "related_continuation")
        self.assertEqual(sm.state, COMMITTED_TURN)
        self.assertEqual(sm.unexpected_transition_count, 0)

    def test_held_fragment_eventually_commits_on_deadline(self):
        sm = InteractionStateMachine()
        sm.on_user_speech_started()
        sm.on_user_speech_stopped()
        sm.begin_turn(1)
        sm.on_turn_policy("HOLD_FOR_CONTINUATION", "INCOMPLETE_THOUGHT", "structural_incomplete")
        self.assertEqual(sm.state, HOLDING_FRAGMENT)
        sm.on_hold_deadline_commit()
        self.assertEqual(sm.state, COMMITTED_TURN)

    def test_unrelated_new_turn_flushes_held_fragment(self):
        sm = InteractionStateMachine()
        sm.on_user_speech_started()
        sm.on_user_speech_stopped()
        sm.begin_turn(1)
        sm.on_turn_policy("HOLD_FOR_CONTINUATION", "INCOMPLETE_THOUGHT", "structural_incomplete")
        sm.on_user_speech_started()
        sm.on_user_speech_stopped()
        sm.begin_turn(2)
        sm.on_turn_policy("FLUSH_HELD_AND_COMMIT_NEW", "COMPLETE_THOUGHT", "unrelated_new_thought")
        self.assertEqual(sm.state, COMMITTED_TURN)


class InterruptionAndOverlapTests(unittest.TestCase):
    def test_user_speaking_during_assistant_speech_is_interrupting(self):
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_assistant_speech_started("speech_1")
        self.assertEqual(sm.state, ASSISTANT_SPEAKING)
        sm.on_user_speech_started()
        self.assertEqual(sm.state, USER_INTERRUPTING)

    def test_user_speaking_during_thinking_is_interrupting(self):
        sm = committed_machine()
        sm.on_llm_started()
        self.assertEqual(sm.state, ASSISTANT_THINKING)
        sm.on_user_speech_started()
        self.assertEqual(sm.state, USER_INTERRUPTING)

    def test_assistant_speech_during_user_activity_flags_overlap(self):
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_user_speech_started()
        with self.assertLogs("interaction_state", level="WARNING") as captured:
            overlap = sm.on_assistant_speech_started("speech_2")
        self.assertTrue(overlap)
        self.assertEqual(sm.overlap_count, 1)
        self.assertIn("assistant_speech_overlap_detected=true", "\n".join(captured.output))

    def test_assistant_speech_when_user_quiet_is_not_overlap(self):
        sm = committed_machine()
        sm.on_llm_started()
        overlap = sm.on_assistant_speech_started("speech_3")
        self.assertFalse(overlap)
        self.assertEqual(sm.overlap_count, 0)

    def test_assistant_finish_returns_to_listening(self):
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_assistant_speech_started("speech_4")
        sm.on_assistant_speech_finished(interrupted=False)
        self.assertEqual(sm.state, LISTENING)

    def test_interrupt_before_first_audio(self):
        # Speech object created (LLM/TTS scheduled) but real audio never started;
        # user barges in. The FSM must move to USER_INTERRUPTING and never have a
        # live active_speech_id.
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_assistant_speech_created("speech_pre")
        self.assertIsNone(sm.active_speech_id)  # no playout yet
        sm.on_user_speech_started()
        self.assertEqual(sm.state, USER_INTERRUPTING)
        sm.on_assistant_speech_finished(interrupted=True, speech_id="speech_pre")
        self.assertIsNone(sm.active_speech_id)
        self.assertNotIn("speech_pre", sm._pending_speech_ids)
        # User is still the active party — we did not snap back to LISTENING.
        self.assertEqual(sm.state, USER_INTERRUPTING)

    def test_interrupt_mid_playout_marks_and_clears_active_speech(self):
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_assistant_speech_started("speech_live")
        self.assertEqual(sm.active_speech_id, "speech_live")
        sm.on_user_speech_started()
        # Interruption is recorded immediately, before the handle resolves.
        self.assertEqual(sm.state, USER_INTERRUPTING)
        self.assertTrue(sm.active_speech_interrupted)
        sm.on_assistant_speech_finished(interrupted=True, speech_id="speech_live")
        self.assertIsNone(sm.active_speech_id)
        self.assertFalse(sm.active_speech_interrupted)

    def test_user_speaks_after_assistant_fully_finishes_is_clean(self):
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_assistant_speech_started("speech_done")
        sm.on_assistant_speech_finished(interrupted=False, speech_id="speech_done")
        self.assertEqual(sm.state, LISTENING)
        self.assertIsNone(sm.active_speech_id)
        # A new utterance after a clean finish is normal speech, NOT an interrupt.
        sm.on_user_speech_started()
        self.assertEqual(sm.state, USER_SPEAKING)
        self.assertFalse(sm.active_speech_interrupted)

    def test_next_user_turn_after_interruption_gets_fresh_turn_id(self):
        sm = committed_machine(turn_id=1)
        sm.on_llm_started()
        sm.on_assistant_speech_started("speech_live")
        sm.on_user_speech_started()  # interrupt
        sm.on_user_speech_stopped()
        sm.begin_turn(2)  # the next user utterance commits as a fresh turn
        self.assertEqual(sm.turn_id, 2)

    def test_newer_turn_while_assistant_thinking_supersedes_old_turn(self):
        sm = committed_machine(turn_id=1)
        sm.on_llm_started()
        sm.on_user_speech_started()
        self.assertEqual(sm.state, USER_INTERRUPTING)
        sm.on_user_speech_stopped()
        sm.begin_turn(2)
        sm.on_turn_policy("COMMIT_NOW", "COMPLETE_THOUGHT", "complete_shape")
        self.assertEqual(sm.state, COMMITTED_TURN)
        self.assertEqual(sm.turn_id, 2)


class ToolCallSeparationTests(unittest.TestCase):
    def test_tool_call_enters_and_exits_pending_state(self):
        sm = committed_machine()
        sm.set_turn_kind(TURN_KIND_ACTION, detected_intent="tool_request_search")
        sm.on_llm_started()
        sm.on_tool_call_started("internet_search")
        self.assertEqual(sm.state, TOOL_CALL_PENDING)
        sm.on_tool_call_finished("internet_search")
        self.assertEqual(sm.state, ASSISTANT_THINKING)
        self.assertEqual(sm.turn_kind, TURN_KIND_ACTION)

    def test_tool_finish_outside_pending_state_is_noop(self):
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_tool_call_finished("internet_search")
        self.assertEqual(sm.state, ASSISTANT_THINKING)


class AssistantSpeechLifecycleTests(unittest.TestCase):
    def test_speech_object_created_does_not_enter_speaking(self):
        sm = committed_machine()
        sm.on_llm_started()
        self.assertEqual(sm.state, ASSISTANT_THINKING)
        sm.on_assistant_speech_created("speech_x")
        # Object creation is not playout: must stay thinking, not SPEAKING.
        self.assertEqual(sm.state, ASSISTANT_THINKING)

    def test_real_playout_enters_speaking_after_object_created(self):
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_assistant_speech_created("speech_x")
        self.assertEqual(sm.state, ASSISTANT_THINKING)
        sm.on_assistant_speech_started("speech_x")
        self.assertEqual(sm.state, ASSISTANT_SPEAKING)

    def test_object_created_never_played_does_not_strand_in_speaking(self):
        # A suppressed/zero-audio speech object that never plays must not leave
        # the FSM believing the assistant is speaking.
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_assistant_speech_created("speech_x")
        sm.on_assistant_speech_finished(interrupted=True, speech_id="speech_x")
        self.assertNotEqual(sm.state, ASSISTANT_SPEAKING)


class BeginTurnAttributionTests(unittest.TestCase):
    def test_explicit_observed_signal_avoids_anomaly_reason(self):
        sm = InteractionStateMachine()
        # FSM never saw the user-speech event (state LISTENING), but the caller
        # passes a real observed signal -> explained, not the counted anomaly.
        with self.assertLogs(interaction_state.logger, level="INFO") as captured:
            sm.begin_turn(1, user_speech_observed=True)
        joined = "\n".join(captured.output)
        self.assertIn("turn_committed_after_state_advanced_post_user_speech", joined)
        self.assertNotIn("turn_committed_by_pipeline_without_observed_user_speech", joined)

    def test_explicit_no_speech_signal_emits_anomaly_reason(self):
        sm = InteractionStateMachine()
        with self.assertLogs(interaction_state.logger, level="INFO") as captured:
            sm.begin_turn(1, user_speech_observed=False)
        self.assertTrue(
            any("turn_committed_by_pipeline_without_observed_user_speech" in line for line in captured.output)
        )

    def test_observed_flag_does_not_leak_across_turns(self):
        sm = InteractionStateMachine()
        sm.on_user_speech_started()
        sm.on_user_speech_stopped()
        sm.begin_turn(1)  # consumes the observed flag
        # A later pipeline-only commit from LISTENING with no fresh speech is an
        # anomaly again, not silently explained by the prior turn's flag.
        sm.transition(LISTENING, reason="reset_for_test")
        with self.assertLogs(interaction_state.logger, level="INFO") as captured:
            sm.begin_turn(2)
        self.assertTrue(
            any("turn_committed_by_pipeline_without_observed_user_speech" in line for line in captured.output)
        )


class ToolResultAuthorityTests(unittest.TestCase):
    def _machine_in_tool_call(self):
        sm = committed_machine()
        sm.set_turn_kind(TURN_KIND_ACTION, detected_intent="tool_request_search")
        sm.on_llm_started()
        sm.on_tool_call_started("internet_search")
        return sm

    def test_tool_call_start_grants_authority(self):
        sm = self._machine_in_tool_call()
        self.assertEqual(sm.state, TOOL_CALL_PENDING)
        self.assertTrue(sm.tool_result_speak_authority)
        self.assertFalse(sm.tool_result_pending_revalidation)

    def test_user_speech_during_tool_call_pauses_authority(self):
        sm = self._machine_in_tool_call()
        with self.assertLogs(interaction_state.logger, level="INFO") as captured:
            sm.on_user_speech_started()
        self.assertEqual(sm.state, USER_INTERRUPTING)
        self.assertFalse(sm.tool_result_speak_authority)
        self.assertTrue(sm.tool_result_pending_revalidation)
        self.assertIn("tool_result_authority_paused_pending_revalidation=true", "\n".join(captured.output))

    def test_additive_context_restores_authority(self):
        sm = self._machine_in_tool_call()
        sm.on_user_speech_started()
        regained = sm.revalidate_tool_result("additive_context")
        self.assertTrue(regained)
        self.assertTrue(sm.tool_result_speak_authority)
        self.assertFalse(sm.tool_result_pending_revalidation)

    def test_correction_and_cancel_withhold_authority(self):
        for cls in ("correction", "cancellation", "pivot", "meta_complaint", "unrelated"):
            sm = self._machine_in_tool_call()
            sm.on_user_speech_started()
            regained = sm.revalidate_tool_result(cls)
            self.assertFalse(regained, cls)
            self.assertFalse(sm.tool_result_speak_authority, cls)

    def test_revalidate_without_pause_is_noop(self):
        sm = self._machine_in_tool_call()
        # No barge-in: authority intact, revalidate returns current authority.
        self.assertTrue(sm.revalidate_tool_result("unrelated"))
        self.assertTrue(sm.tool_result_speak_authority)

    def test_apply_compose_decision_grants_authority(self):
        sm = self._machine_in_tool_call()
        sm.on_user_speech_started()
        with self.assertLogs(interaction_state.logger, level="INFO") as captured:
            granted = sm.apply_tool_resume_decision(
                relationship="additive_context", decision="compose", resolution="resolved", additive_allowed=True
            )
        self.assertTrue(granted)
        self.assertTrue(sm.tool_result_speak_authority)
        self.assertEqual(sm.tool_result_resume_decision, "compose")
        joined = "\n".join(captured.output)
        self.assertIn("tool_result_resume_decision=compose", joined)
        self.assertIn("tool_result_composed_with_newer_user_utterance=True", joined)

    def test_apply_rerun_decision_withholds_authority(self):
        sm = self._machine_in_tool_call()
        sm.on_user_speech_started()
        granted = sm.apply_tool_resume_decision(
            relationship="major_correction", decision="rerun", resolution="resolved", additive_allowed=False
        )
        self.assertFalse(granted)
        self.assertFalse(sm.tool_result_speak_authority)
        self.assertEqual(sm.tool_result_resume_decision, "rerun")

    def test_apply_withhold_even_if_additive_flag_but_not_compose(self):
        sm = self._machine_in_tool_call()
        sm.on_user_speech_started()
        granted = sm.apply_tool_resume_decision(
            relationship="additive_context", decision="defer", resolution="unresolved", additive_allowed=True
        )
        self.assertFalse(granted)


class RuntimeGateTests(unittest.TestCase):
    def test_blocked_gate_logs_and_counts(self):
        sm = InteractionStateMachine()
        with self.assertLogs(interaction_state.logger, level="WARNING") as captured:
            allowed = sm.runtime_gate("turn_commit_owner", False, reason="turn_commit_no_owner")
        self.assertFalse(allowed)
        self.assertEqual(sm.gate_blocked_count, 1)
        joined = "\n".join(captured.output)
        self.assertIn("fsm_gate_blocked=true", joined)
        self.assertIn("fsm_gate_action=turn_commit_owner", joined)

    def test_allowed_gate_returns_true_without_counting(self):
        sm = InteractionStateMachine()
        with self.assertLogs(interaction_state.logger, level="INFO") as captured:
            allowed = sm.runtime_gate("turn_commit_owner", True, reason="valid_owner")
        self.assertTrue(allowed)
        self.assertEqual(sm.gate_blocked_count, 0)
        self.assertIn("fsm_gate_blocked=false", "\n".join(captured.output))


class FallbackVisibilityTests(unittest.TestCase):
    def test_suppressed_fallback_logged_with_state(self):
        sm = committed_machine()
        sm.on_llm_started()
        sm.on_user_speech_started()
        with self.assertLogs("interaction_state", level="WARNING") as captured:
            sm.on_fallback_decision(allowed=False, reason="user_speaking_or_newer_turn_pending", requires_repeat=False)
        joined = "\n".join(captured.output)
        self.assertIn("fallback_allowed=False", joined)
        self.assertIn("interaction_state=USER_INTERRUPTING", joined)

    def test_allowed_fallback_logged_with_repeat_flag(self):
        sm = committed_machine()
        sm.on_llm_started()
        with self.assertLogs("interaction_state", level="WARNING") as captured:
            sm.on_fallback_decision(allowed=True, reason="first_token_timeout", requires_repeat=False)
        joined = "\n".join(captured.output)
        self.assertIn("fallback_allowed=True", joined)
        self.assertIn("fallback_requires_user_repeat=False", joined)


class StateMachineSafetyTests(unittest.TestCase):
    def test_recovery_decision_enters_recovery_state(self):
        sm = InteractionStateMachine()
        sm.on_user_speech_started()
        sm.on_user_speech_stopped()
        sm.begin_turn(1)
        sm.on_turn_policy("RECOVER_FROM_SILENCE", "META_COMPLAINT", "meta_complaint")
        self.assertEqual(sm.state, RECOVERY)

    def test_filler_decision_returns_to_listening(self):
        sm = InteractionStateMachine()
        sm.on_user_speech_started()
        sm.on_user_speech_stopped()
        sm.begin_turn(1)
        sm.on_turn_policy("IGNORE_LOW_INFORMATION_FILLER", "LOW_INFORMATION", "filler")
        self.assertEqual(sm.state, LISTENING)

    def test_unexpected_transition_is_counted_but_applied(self):
        sm = InteractionStateMachine()
        with self.assertLogs("interaction_state", level="WARNING") as captured:
            sm.transition(USER_INTERRUPTING, reason="forced")
        self.assertEqual(sm.state, USER_INTERRUPTING)
        self.assertEqual(sm.unexpected_transition_count, 1)
        self.assertIn("interaction_state_transition_unexpected=true", "\n".join(captured.output))

    def test_invalid_target_state_is_rejected(self):
        sm = InteractionStateMachine()
        sm.transition("DANCING", reason="nope")
        self.assertEqual(sm.state, LISTENING)

    def test_transitions_are_logged_for_railway(self):
        sm = InteractionStateMachine()
        with self.assertLogs("interaction_state", level="INFO") as captured:
            sm.on_user_speech_started()
        joined = "\n".join(captured.output)
        self.assertIn("interaction_state_transition from=LISTENING to=USER_SPEAKING", joined)
        self.assertIn("reason=user_started_speaking", joined)

    def test_snapshot_fields(self):
        sm = committed_machine(turn_id=9)
        snapshot = sm.snapshot()
        self.assertEqual(snapshot["state"], COMMITTED_TURN)
        self.assertEqual(snapshot["turn_id"], 9)
        self.assertIn("seconds_in_state", snapshot)


if __name__ == "__main__":
    unittest.main()

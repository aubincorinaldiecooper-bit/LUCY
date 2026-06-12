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

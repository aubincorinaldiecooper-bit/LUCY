import unittest
from unittest.mock import patch

import agent
import transcript_context as tc
from transcript_context import ContextDecision, build_context_decision


def _decide(text, *, base_intent="unknown", classification="COMPLETE_THOUGHT", ambiguity=False, clarification=False, prior=None):
    return build_context_decision(
        text=text,
        base_intent=base_intent,
        classification=classification,
        ambiguity_detected=ambiguity,
        clarification_suggested=clarification,
        confidence=0.8,
        prior_decision=prior,
    )


class ContextDecisionIntentTests(unittest.TestCase):
    """The 8 contextual utterances from the spec must force recent visible context."""

    def assert_contextual(self, decision, expected_intent, expected_posture):
        self.assertEqual(decision.final_intent, expected_intent)
        self.assertEqual(decision.context_dependency, "high")
        self.assertTrue(decision.force_context_injection)
        self.assertEqual(decision.response_posture, expected_posture)

    def test_exactly_youre_getting_it(self):
        self.assert_contextual(_decide("Exactly. You’re getting it."), "user_evaluating_assistant", "contextual_acknowledgment")

    def test_correct_see(self):
        self.assert_contextual(_decide("Correct. See?"), "user_evaluating_assistant", "contextual_acknowledgment")

    def test_why_did_you_miss_it(self):
        self.assert_contextual(_decide("Why did you miss it?"), "meta_complaint", "missed_context_recovery")

    def test_do_you_know_why_carries_forward(self):
        prior = _decide("the thing we said", base_intent="reference_to_prior_context")
        decision = _decide("Do you know why?", prior=prior)
        self.assert_contextual(decision, "reference_to_prior_context", "contextual_acknowledgment")
        self.assertEqual(decision.decision_source, "carry_forward")

    def test_that_time_you_got_it(self):
        self.assert_contextual(_decide("That time you got it."), "user_evaluating_assistant", "contextual_acknowledgment")

    def test_no_the_last_thing(self):
        self.assert_contextual(_decide("No, the last thing."), "correction", "correction_received")

    def test_see_what_i_mean(self):
        self.assert_contextual(_decide("See what I mean?"), "user_evaluating_assistant", "contextual_acknowledgment")

    def test_thats_what_i_was_testing(self):
        self.assert_contextual(_decide("That’s what I was testing."), "user_evaluating_assistant", "contextual_acknowledgment")


class ContextDecisionRuleTests(unittest.TestCase):
    def test_complete_thought_does_not_override_context_dependency(self):
        # Turn-shape COMPLETE_THOUGHT must not downgrade a contextual turn.
        decision = _decide("Exactly, you got it.", classification="COMPLETE_THOUGHT")
        self.assertEqual(decision.context_dependency, "high")
        self.assertTrue(decision.force_context_injection)

    def test_ambiguous_clarification_is_not_generic_standalone(self):
        decision = _decide("that one", ambiguity=True, clarification=True)
        self.assertEqual(decision.response_posture, "unclear_reference_clarification")
        self.assertTrue(decision.force_context_injection)
        self.assertEqual(decision.context_dependency, "high")

    def test_plain_question_is_standalone(self):
        decision = _decide("What time is it?", base_intent="date_time_question")
        self.assertEqual(decision.final_intent, "unknown")
        self.assertEqual(decision.context_dependency, "none")
        self.assertFalse(decision.force_context_injection)
        self.assertEqual(decision.response_posture, "answer")

    def test_carry_forward_requires_prior_high_dependency(self):
        # Same vague utterance, no prior context -> stays standalone.
        decision = _decide("Do you know why?", prior=None)
        self.assertEqual(decision.context_dependency, "none")
        self.assertFalse(decision.force_context_injection)


class ConversationLedgerTests(unittest.TestCase):
    def setUp(self):
        agent._conversation_ledger.clear()

    def tearDown(self):
        agent._conversation_ledger.clear()

    def test_suppressed_and_empty_assistant_turns_excluded_from_canonical(self):
        agent._ledger_append("user", "remember the plan", visible=True, suppressed=False)
        agent._ledger_append("assistant", "", visible=False, suppressed=True)  # zero-audio/stale phantom
        agent._ledger_append("assistant", "Okay, noted.", visible=True, suppressed=False)
        recent = agent._ledger_recent_canonical(5)
        texts = [(e["role"], e["text"]) for e in recent]
        self.assertIn(("user", "remember the plan"), texts)
        self.assertIn(("assistant", "Okay, noted."), texts)
        self.assertNotIn(("assistant", ""), texts)
        self.assertEqual(agent._ledger_suppressed_count(), 1)
        self.assertEqual(agent._ledger_visible_canonical_count(), 2)

    def test_interrupted_assistant_turn_is_not_canonical(self):
        # Mirrors the conversation_item_added guard: interrupted/empty -> suppressed.
        agent._ledger_append("assistant", "half a sen", visible=False, suppressed=True)
        self.assertEqual(agent._ledger_recent_canonical(5), [])


class LedgerOutcomeDowngradeTests(unittest.TestCase):
    def setUp(self):
        agent._conversation_ledger.clear()

    def tearDown(self):
        agent._conversation_ledger.clear()

    def test_reason_zero_audio(self):
        self.assertEqual(
            agent._ledger_downgrade_reason_for_outcome(
                was_suppressed=False, interrupted="false", generated_bytes=0, playout_seconds=1.2
            ),
            "zero_audio",
        )

    def test_reason_suppressed(self):
        self.assertEqual(
            agent._ledger_downgrade_reason_for_outcome(
                was_suppressed=True, interrupted="false", generated_bytes=4096, playout_seconds=1.2
            ),
            "suppressed",
        )

    def test_reason_interrupted(self):
        self.assertEqual(
            agent._ledger_downgrade_reason_for_outcome(
                was_suppressed=False, interrupted="true", generated_bytes=4096, playout_seconds=1.2
            ),
            "interrupted",
        )

    def test_reason_no_playout(self):
        self.assertEqual(
            agent._ledger_downgrade_reason_for_outcome(
                was_suppressed=False, interrupted="false", generated_bytes=4096, playout_seconds=-1.0
            ),
            "no_playout",
        )

    def test_reason_audible_returns_none(self):
        self.assertIsNone(
            agent._ledger_downgrade_reason_for_outcome(
                was_suppressed=False, interrupted="false", generated_bytes=4096, playout_seconds=1.2
            )
        )

    def test_nonempty_text_zero_audio_is_downgraded_and_excluded(self):
        agent._ledger_append("user", "the plan", visible=True, suppressed=False, turn_id=7)
        agent._ledger_append("assistant", "Here is the plan.", visible=True, suppressed=False, turn_id=7, provisional=True)
        # canonical until reconciliation
        self.assertEqual(agent._ledger_visible_canonical_count(), 2)
        strategy = agent._ledger_downgrade_for_outcome(turn_id=7, speech_id="AS_x", reason="zero_audio")
        self.assertEqual(strategy, "turn_id_recent")
        recent = agent._ledger_recent_canonical(5)
        self.assertNotIn(("assistant", "Here is the plan."), [(e["role"], e["text"]) for e in recent])
        self.assertIn(("user", "the plan"), [(e["role"], e["text"]) for e in recent])

    def test_suppressed_outcome_is_downgraded(self):
        agent._ledger_append("assistant", "stale reply", visible=True, suppressed=False, turn_id=3, provisional=True)
        agent._ledger_downgrade_for_outcome(turn_id=3, speech_id=None, reason="suppressed")
        self.assertEqual(agent._ledger_recent_canonical(5), [])

    def test_successful_audible_remains_canonical(self):
        agent._ledger_append("assistant", "audible reply", visible=True, suppressed=False, turn_id=9, provisional=True)
        # No downgrade call (reason was None) -> stays canonical.
        recent = agent._ledger_recent_canonical(5)
        self.assertIn(("assistant", "audible reply"), [(e["role"], e["text"]) for e in recent])

    def test_speech_id_match_is_preferred(self):
        agent._ledger_append("assistant", "older same turn", visible=True, suppressed=False, turn_id=5, speech_id=None, provisional=True)
        agent._ledger_append("assistant", "target by speech", visible=True, suppressed=False, turn_id=5, speech_id="AS_1", provisional=True)
        strategy = agent._ledger_downgrade_for_outcome(turn_id=5, speech_id="AS_1", reason="zero_audio")
        self.assertEqual(strategy, "speech_id")
        texts = [(e["role"], e["text"]) for e in agent._ledger_recent_canonical(5)]
        self.assertIn(("assistant", "older same turn"), texts)
        self.assertNotIn(("assistant", "target by speech"), texts)

    def test_no_safe_match_logs_failure_and_does_not_corrupt_other_turns(self):
        agent._ledger_append("assistant", "turn 1 reply", visible=True, suppressed=False, turn_id=1, provisional=True)
        with self.assertLogs(agent.logger, level="WARNING") as captured:
            strategy = agent._ledger_downgrade_for_outcome(turn_id=0, speech_id=None, reason="zero_audio")
        self.assertEqual(strategy, "failed")
        self.assertTrue(any("ledger_downgrade_match_failed=true" in line for line in captured.output))
        # Unrelated turn is untouched.
        self.assertIn(("assistant", "turn 1 reply"), [(e["role"], e["text"]) for e in agent._ledger_recent_canonical(5)])


class ContextAwareFallbackTests(unittest.TestCase):
    def test_context_aware_fallback_for_contextual_timeout(self):
        with patch.object(agent, "CONTEXT_AWARE_FALLBACK_ENABLED", True):
            text = agent._context_aware_fallback_text("first_token_timeout", "high")
        self.assertEqual(text, agent.CONTEXT_AWARE_FALLBACK_TEXT)

    def test_generic_fallback_for_ordinary_timeout(self):
        with patch.object(agent, "CONTEXT_AWARE_FALLBACK_ENABLED", True):
            self.assertIsNone(agent._context_aware_fallback_text("first_token_timeout", "none"))
            self.assertIsNone(agent._context_aware_fallback_text("provider_error", "high"))

    def test_disabled_returns_none(self):
        with patch.object(agent, "CONTEXT_AWARE_FALLBACK_ENABLED", False):
            self.assertIsNone(agent._context_aware_fallback_text("first_token_timeout", "high"))


if __name__ == "__main__":
    unittest.main()

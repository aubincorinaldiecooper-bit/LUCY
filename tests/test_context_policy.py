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

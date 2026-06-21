import unittest

import agent


class MemoryRetrievalGateTests(unittest.TestCase):
    def test_recall_intent_allows_retrieval(self):
        allowed, reason = agent._memory_retrieval_policy_for_intent("memory_recall_request")
        self.assertTrue(allowed)
        self.assertEqual(reason, "memory_recall_intent")

    def test_non_recall_intents_skip_retrieval(self):
        for intent in (
            "unknown",
            "reference_to_prior_context",
            "tool_request_search",
            "calculation_request",
            None,
        ):
            allowed, reason = agent._memory_retrieval_policy_for_intent(intent)
            self.assertFalse(allowed, intent)
            self.assertEqual(reason, "no_recall_intent", intent)

    def test_recall_intent_is_blocked_from_web_search(self):
        # A "do you remember..." turn must not leak into the Exa search path.
        allowed, reason = agent._search_policy_for_intent("memory_recall_request", False)
        self.assertFalse(allowed)
        self.assertEqual(reason, "blocked_non_lookup_intent")

    def test_clear_search_intent_still_allowed(self):
        allowed, _ = agent._search_policy_for_intent("tool_request_search", False)
        self.assertTrue(allowed)


if __name__ == "__main__":
    unittest.main()

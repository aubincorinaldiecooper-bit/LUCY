import types
import unittest

import agent


def _msg(role: str, content: str):
    return types.SimpleNamespace(role=role, content=content)


class _FakeTurnCtx:
    def __init__(self):
        self.messages_added: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str):
        self.messages_added.append((role, content))


class SearchPolicyGateTests(unittest.TestCase):
    def setUp(self):
        self._orig = agent.SEARCH_REQUIRE_EXPLICIT_INTENT

    def tearDown(self):
        agent.SEARCH_REQUIRE_EXPLICIT_INTENT = self._orig

    def test_explicit_search_intent_always_allowed(self):
        agent.SEARCH_REQUIRE_EXPLICIT_INTENT = True
        allowed, reason = agent._search_policy_for_intent("tool_request_search", False)
        self.assertTrue(allowed)
        self.assertEqual(reason, "clear_search_intent")

    def test_explicit_search_intent_blocked_when_clarification_needed(self):
        allowed, reason = agent._search_policy_for_intent("tool_request_search", True)
        self.assertFalse(allowed)
        self.assertEqual(reason, "blocked_unclear_fragment")

    def test_unknown_intent_blocked_under_strict_gate(self):
        # The "I didn't ask you to look anything up" failure: a casual statement
        # classified as unknown must NOT auto-trigger search.
        agent.SEARCH_REQUIRE_EXPLICIT_INTENT = True
        allowed, reason = agent._search_policy_for_intent("unknown", False)
        self.assertFalse(allowed)
        self.assertEqual(reason, "blocked_no_explicit_search_intent")

    def test_unknown_intent_permissive_when_flag_disabled(self):
        # Legacy behavior remains reachable for anyone who wants it back.
        agent.SEARCH_REQUIRE_EXPLICIT_INTENT = False
        allowed, reason = agent._search_policy_for_intent("unknown", False)
        self.assertTrue(allowed)
        self.assertEqual(reason, "llm_tool_call")

    def test_conversational_intent_blocked_regardless_of_flag(self):
        for flag in (True, False):
            agent.SEARCH_REQUIRE_EXPLICIT_INTENT = flag
            allowed, reason = agent._search_policy_for_intent("memory_recall_request", False)
            self.assertFalse(allowed)
            self.assertEqual(reason, "blocked_non_lookup_intent")


class RecallAnchorTests(unittest.TestCase):
    def test_recent_user_messages_excludes_current_question_and_orders(self):
        ctx = [
            _msg("user", "I'm going live tomorrow"),
            _msg("assistant", "got it"),
            _msg("user", "do you remember my last question?"),
        ]
        recent = agent._recent_user_messages_from_chat_ctx(
            ctx, exclude_text="do you remember my last question?", limit=3
        )
        self.assertEqual(recent, ["I'm going live tomorrow"])

    def test_recent_user_messages_limit_and_newest_last(self):
        ctx = [
            _msg("user", "one"),
            _msg("user", "two"),
            _msg("user", "three"),
            _msg("user", "four"),
        ]
        recent = agent._recent_user_messages_from_chat_ctx(ctx, exclude_text="", limit=2)
        self.assertEqual(recent, ["three", "four"])

    def test_anchor_note_quotes_most_recent_message(self):
        ctx = _FakeTurnCtx()
        injected = agent._inject_recall_anchor_note(ctx, ["older thing", "my actual last question"])
        self.assertTrue(injected)
        self.assertEqual(len(ctx.messages_added), 1)
        role, content = ctx.messages_added[0]
        self.assertEqual(role, "developer")
        self.assertIn('"my actual last question"', content)
        self.assertIn("MOST RECENT", content)

    def test_anchor_note_noop_without_messages(self):
        ctx = _FakeTurnCtx()
        self.assertFalse(agent._inject_recall_anchor_note(ctx, []))
        self.assertEqual(ctx.messages_added, [])


if __name__ == "__main__":
    unittest.main()

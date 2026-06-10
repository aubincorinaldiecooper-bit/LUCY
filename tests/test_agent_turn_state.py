import unittest
from unittest.mock import patch

import agent


class AgentTurnStateTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            name: getattr(agent, name)
            for name in (
                "_current_turn_id",
                "_search_turn_id",
                "_search_tool_called",
                "_search_in_progress",
                "_search_specific_response_produced",
                "_last_search_tool_output",
                "_current_turn_search_allowed",
                "_current_turn_search_allowed_reason",
            )
        }

    def tearDown(self):
        for name, value in self.state.items():
            setattr(agent, name, value)

    def test_search_state_resets_on_each_turn(self):
        agent._current_turn_id = 1
        agent._mark_search_wait_started(turn_id=1)
        agent._mark_search_wait_completed(False, "result", turn_id=1)
        self.assertTrue(agent._search_tool_called)
        self.assertTrue(agent._search_specific_response_produced)

        agent._current_turn_id = 2
        agent._reset_search_state_for_turn(2)
        self.assertFalse(agent._search_tool_called)
        self.assertFalse(agent._search_in_progress)
        self.assertFalse(agent._search_specific_response_produced)
        self.assertEqual(agent._search_turn_id, 2)

    def test_stale_search_completion_is_ignored(self):
        agent._current_turn_id = 1
        agent._mark_search_wait_started(turn_id=1)
        agent._current_turn_id = 2
        agent._reset_search_state_for_turn(2)

        applied = agent._mark_search_wait_completed(False, "old result", turn_id=1)
        self.assertFalse(applied)
        self.assertEqual(agent._last_search_tool_output, "")
        self.assertFalse(agent._search_specific_response_produced)

    def test_previous_search_does_not_match_current_turn_for_fallback(self):
        agent._current_turn_id = 1
        agent._mark_search_wait_started(turn_id=1)
        agent._mark_search_wait_completed(False, "old result", turn_id=1)
        agent._current_turn_id = 2
        self.assertFalse(agent._search_turn_matches_current())
        self.assertFalse(agent._search_specific_response_for_current_turn())

    def test_unclear_fragment_does_not_allow_search(self):
        allowed, reason = agent._search_policy_for_intent("unclear_fragment", True)
        self.assertFalse(allowed)
        self.assertEqual(reason, "blocked_unclear_fragment")

    def test_unclear_search_intent_asks_clarification(self):
        allowed, reason = agent._search_policy_for_intent("tool_request_search", True)
        self.assertFalse(allowed)
        self.assertEqual(reason, "blocked_unclear_fragment")

    def test_clear_search_intent_can_call_exa(self):
        allowed, reason = agent._search_policy_for_intent("tool_request_search", False)
        self.assertTrue(allowed)
        self.assertEqual(reason, "clear_search_intent")

    def test_non_lookup_intents_block_search(self):
        for intent in ("numeric_fragment", "language_request", "counting_request", "calculation_request"):
            allowed, reason = agent._search_policy_for_intent(intent, False)
            self.assertFalse(allowed)
            self.assertEqual(reason, "blocked_non_lookup_intent")

    def test_preemptive_generation_defaults_disabled(self):
        self.assertFalse(agent.PREEMPTIVE_GENERATION_ENABLED)

    def test_cleanup_create_task_wrapper_receives_coroutine(self):
        scheduled = []

        class SpeechLike:
            def interrupt(self):
                return self

            def __await__(self):
                async def _done():
                    return None
                return _done().__await__()

        test_case = self

        class Loop:
            def create_task(self, value):
                scheduled.append(value)
                test_case.assertTrue(hasattr(value, "cr_await") or hasattr(value, "__await__"))
                test_case.assertNotIsInstance(value, SpeechLike)
                close = getattr(value, "close", None)
                if callable(close):
                    close()
                return value

        with patch("asyncio.get_running_loop", return_value=Loop()):
            ok, result = agent._test_invoke_cleanup_method(SpeechLike(), "interrupt", "speech_1", "unit_test")

        self.assertTrue(ok)
        self.assertEqual(result, "scheduled_awaitable")
        self.assertEqual(len(scheduled), 1)


class IncompleteFragmentHeuristicTests(unittest.TestCase):
    def test_trailing_comma_is_fragment(self):
        self.assertTrue(agent._looks_like_incomplete_fragment("Yeah. So,", "unknown", False))

    def test_trailing_conjunction_with_stt_period_is_fragment(self):
        self.assertTrue(agent._looks_like_incomplete_fragment("Now, why people are.", "unknown", False))

    def test_unclear_fragment_intent_is_fragment(self):
        self.assertTrue(agent._looks_like_incomplete_fragment("Yeah. So,", "unclear_fragment", True))

    def test_ellipsis_is_fragment(self):
        self.assertTrue(agent._looks_like_incomplete_fragment("I was thinking about...", "unknown", False))

    def test_empty_text_is_fragment(self):
        self.assertTrue(agent._looks_like_incomplete_fragment("", "unknown", False))

    def test_complete_question_is_not_fragment(self):
        self.assertFalse(agent._looks_like_incomplete_fragment("Isn't that why they just like to vent?", "unknown", False))

    def test_complete_sentence_is_not_fragment(self):
        self.assertFalse(agent._looks_like_incomplete_fragment("Sometimes people need a vent.", "unknown", False))


class ChatContextPruneTests(unittest.TestCase):
    def _build_ctx(self, non_system_count: int):
        from livekit.agents.llm import ChatContext

        ctx = ChatContext()
        ctx.add_message(role="system", content="sys")
        for i in range(non_system_count):
            ctx.add_message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        return ctx

    def test_chat_ctx_items_resolves_livekit_items_property(self):
        ctx = self._build_ctx(3)
        items = agent._chat_ctx_items(ctx)
        self.assertIsNotNone(items)
        self.assertEqual(len(items), 4)

    def test_prune_caps_non_system_messages_and_keeps_system(self):
        cap = agent.CONTEXT_WINDOW_TURNS * 2
        ctx = self._build_ctx(cap + 10)
        total, kept, dropped = agent._prune_turn_context_messages(ctx, turn_id=1)
        self.assertEqual(total, cap + 11)
        self.assertEqual(dropped, 10)
        self.assertEqual(len(ctx.items), kept)
        self.assertEqual(str(ctx.items[0].role), "system")
        non_system = [item for item in ctx.items if str(item.role) != "system"]
        self.assertEqual(len(non_system), cap)
        self.assertEqual(non_system[-1].text_content, f"m{cap + 9}")

    def test_prune_below_cap_is_noop(self):
        ctx = self._build_ctx(4)
        total, kept, dropped = agent._prune_turn_context_messages(ctx, turn_id=1)
        self.assertEqual((total, kept, dropped), (5, 5, 0))
        self.assertEqual(len(ctx.items), 5)


if __name__ == "__main__":
    unittest.main()

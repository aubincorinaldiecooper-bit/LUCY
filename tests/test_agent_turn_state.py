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


if __name__ == "__main__":
    unittest.main()

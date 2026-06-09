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
                "_latest_user_state_for_greeting",
                "_latest_user_speaking_at",
                "_latest_stt_partial_at",
                "_latest_stt_final_at",
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

    def test_trailing_comma_extends_endpointing(self):
        decision, reason, wait_ms = agent._endpointing_decision_for_transcript("I mean,", None)
        self.assertEqual(decision, "extend_wait")
        self.assertEqual(reason, "trailing_comma")
        self.assertGreaterEqual(wait_ms, 600)

    def test_natural_pause_fragments_extend_endpointing(self):
        for text in ("Yeah. So,", "Now,", "Because"):
            decision, reason, wait_ms = agent._endpointing_decision_for_transcript(text, None)
            self.assertEqual(decision, "extend_wait")
            self.assertGreater(wait_ms, 0)
            self.assertIn(reason, {"trailing_comma", "filler_phrase", "short_fragment"})

    def test_unclear_fragment_delays_commit(self):
        context = agent.detect_transcript_context("Sometimes")
        decision, reason, _ = agent._endpointing_decision_for_transcript("Sometimes", context)
        self.assertEqual(decision, "extend_wait")
        self.assertIn(reason, {"unclear_fragment", "short_fragment"})

    def test_direct_commands_commit_quickly(self):
        for text in ("What time is it?", "Stop.", "Count to ten.", "Search that.", "Yes.", "No.", "Okay."):
            decision, reason, wait_ms = agent._endpointing_decision_for_transcript(text, None)
            self.assertEqual(decision, "commit")
            self.assertEqual(reason, "none")
            self.assertEqual(wait_ms, 0)

    def test_generic_fallback_suppressed_when_user_speaking(self):
        agent._current_turn_id = 10
        agent._latest_user_state_for_greeting = "speaking"
        self.assertEqual(
            agent._generic_fallback_suppression_reason(10, 100.0),
            "user_speaking_or_newer_turn_pending",
        )

    def test_generic_fallback_suppressed_when_newer_turn_pending(self):
        agent._current_turn_id = 11
        agent._latest_user_state_for_greeting = "listening"
        self.assertEqual(
            agent._generic_fallback_suppression_reason(10, 100.0),
            "user_speaking_or_newer_turn_pending",
        )

    def test_generic_fallback_suppressed_when_partial_is_growing(self):
        agent._current_turn_id = 10
        agent._latest_user_state_for_greeting = "listening"
        agent._latest_stt_final_at = 101.0
        agent._latest_stt_partial_at = 102.0
        self.assertEqual(
            agent._generic_fallback_suppression_reason(10, 100.0),
            "user_speaking_or_newer_turn_pending",
        )

    def test_llm_stream_turn_id_values_can_remain_original(self):
        agent._current_turn_id = 47
        original_turn_id = agent._current_turn_id
        agent._current_turn_id = 48
        self.assertTrue(agent._is_stale_llm_turn(original_turn_id))
        self.assertEqual(original_turn_id, 47)

    def test_stale_llm_output_condition_detects_newer_turn(self):
        agent._current_turn_id = 48
        self.assertTrue(agent._is_stale_llm_turn(47))
        agent._current_turn_id = 47
        self.assertFalse(agent._is_stale_llm_turn(47))


class ContextPruningTests(unittest.TestCase):
    class Message:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

    class TurnCtx:
        def __init__(self, messages):
            self.messages = messages

    def test_pruning_drops_older_non_system_messages(self):
        messages = [
            self.Message("system", "system prompt"),
            self.Message("developer", "runtime note"),
        ]
        for index in range(10):
            messages.append(self.Message("user", f"user {index}"))
            messages.append(self.Message("assistant", f"assistant {index}"))
        ctx = self.TurnCtx(messages)

        with patch.object(agent, "CONTEXT_WINDOW_TURNS", 4):
            total, kept, dropped = agent._prune_turn_context_messages(ctx, turn_id=123)

        self.assertEqual(total, 22)
        self.assertEqual(kept, 10)
        self.assertEqual(dropped, 12)
        self.assertEqual([message.role for message in ctx.messages[:2]], ["system", "developer"])
        self.assertEqual([message.content for message in ctx.messages[2:]], [
            "user 6",
            "assistant 6",
            "user 7",
            "assistant 7",
            "user 8",
            "assistant 8",
            "user 9",
            "assistant 9",
        ])

    def test_system_prompt_is_retained_when_pruning(self):
        system = self.Message("system", "keep me")
        messages = [system]
        for index in range(6):
            messages.append(self.Message("user", f"user {index}"))
            messages.append(self.Message("assistant", f"assistant {index}"))
        ctx = self.TurnCtx(messages)

        with patch.object(agent, "CONTEXT_WINDOW_TURNS", 4):
            agent._prune_turn_context_messages(ctx, turn_id=124)

        self.assertIn(system, ctx.messages)
        self.assertEqual(ctx.messages[0], system)
        self.assertEqual(len([message for message in ctx.messages if message.role != "system"]), 8)

    def test_no_pruning_when_history_within_window(self):
        messages = [self.Message("system", "system")]
        for index in range(3):
            messages.append(self.Message("user", f"user {index}"))
            messages.append(self.Message("assistant", f"assistant {index}"))
        original_ids = [id(message) for message in messages]
        ctx = self.TurnCtx(messages)

        with patch.object(agent, "CONTEXT_WINDOW_TURNS", 4):
            total, kept, dropped = agent._prune_turn_context_messages(ctx, turn_id=125)

        self.assertEqual((total, kept, dropped), (7, 7, 0))
        self.assertEqual([id(message) for message in ctx.messages], original_ids)

    def test_context_window_zero_clamps_to_safe_minimum(self):
        with patch.dict("os.environ", {"CONTEXT_WINDOW_TURNS": "0"}):
            self.assertEqual(agent.env_int_clamped("CONTEXT_WINDOW_TURNS", 10, 4, 100), 4)


if __name__ == "__main__":
    unittest.main()

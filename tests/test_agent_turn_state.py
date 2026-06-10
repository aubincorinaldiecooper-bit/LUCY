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


class VoiceLifecycleObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.hume_keys = dict(agent._hume_recent_request_keys)
        self.hume_order = list(agent._hume_recent_request_order)

    def tearDown(self):
        agent._hume_recent_request_keys.clear()
        agent._hume_recent_request_keys.update(self.hume_keys)
        agent._hume_recent_request_order[:] = self.hume_order

    def test_deprecated_cleanup_warning_removed_from_source(self):
        with open(agent.__file__, "r", encoding="utf-8") as source_file:
            self.assertNotIn("Deprecated one-argument assistant speech cleanup call ignored", source_file.read())

    def test_stale_speech_ids_are_capped_and_pruned(self):
        stale_ids = {f"speech_{index}" for index in range(25)}
        stale_order = [f"speech_{index}" for index in range(25)]

        pruned = agent._cap_recent_ids(stale_ids, stale_order, max_size=20)

        self.assertEqual(pruned, 5)
        self.assertEqual(len(stale_ids), 20)
        self.assertNotIn("speech_0", stale_ids)
        self.assertIn("speech_24", stale_ids)
        self.assertEqual(len(stale_order), 20)

    def test_latency_audit_drops_impossible_or_inherited_values(self):
        audit = agent._build_voice_latency_audit(
            turn_id=9,
            speech_id="speech_9",
            user_speech_started_at=20.0,
            user_speech_stopped_at=19.0,
            final_stt_received_at=21.0,
            user_turn_committed_at=22.0,
            llm_request_started_at=0.0,
            llm_first_token_at=23.0,
            llm_completed_at=24.0,
            tts_request_started_at=18.0,
            tts_first_audio_at=25.0,
            tts_completed_at=26.0,
            assistant_playout_started_at=25.5,
            assistant_playout_completed_at=27.0,
        )

        self.assertIsNone(audit["user_speech_stopped_at"])
        self.assertIsNone(audit["llm_request_started_at"])
        self.assertIsNone(audit["tts_request_started_at"])
        self.assertIsNone(audit["user_stopped_to_first_audio"])

    def test_latency_audit_is_fresh_per_speech(self):
        first = agent._build_voice_latency_audit(
            turn_id=1,
            speech_id="speech_1",
            user_speech_started_at=1.0,
            user_speech_stopped_at=2.0,
            final_stt_received_at=3.0,
            user_turn_committed_at=4.0,
            llm_request_started_at=5.0,
            llm_first_token_at=6.0,
            llm_completed_at=7.0,
            tts_request_started_at=8.0,
            tts_first_audio_at=9.0,
            tts_completed_at=10.0,
            assistant_playout_started_at=9.5,
            assistant_playout_completed_at=11.0,
        )
        second = agent._build_voice_latency_audit(
            turn_id=2,
            speech_id="speech_2",
            user_speech_started_at=101.0,
            user_speech_stopped_at=102.0,
            final_stt_received_at=103.0,
            user_turn_committed_at=104.0,
            llm_request_started_at=None,
            llm_first_token_at=None,
            llm_completed_at=None,
            tts_request_started_at=None,
            tts_first_audio_at=None,
            tts_completed_at=None,
            assistant_playout_started_at=None,
            assistant_playout_completed_at=None,
        )

        self.assertEqual(first["speech_id"], "speech_1")
        self.assertEqual(second["speech_id"], "speech_2")
        self.assertIsNone(second["llm_request_started_at"])
        self.assertIsNone(second["tts_request_started_at"])

    def test_hume_duplicate_request_detection_for_same_speech_and_hash(self):
        first = agent._record_hume_request_metadata(
            path="default_agent_tts_node_fallback",
            speech_id="speech_1",
            normalized_text_hash="abc123",
            feeds_playout=True,
        )
        second = agent._record_hume_request_metadata(
            path="default_agent_tts_node_fallback",
            speech_id="speech_1",
            normalized_text_hash="abc123",
            feeds_playout=True,
        )

        self.assertFalse(first[1])
        self.assertTrue(second[1])
        self.assertEqual(first[0], second[0])

    def test_hume_request_dedupe_key_includes_speech_and_hash(self):
        key = agent._hume_request_dedupe_key("path", "speech_7", "hash_7")
        self.assertIn("speech_7", key)
        self.assertIn("hash_7", key)

    def test_hume_logs_include_latest_agent_state_field(self):
        with open(agent.__file__, "r", encoding="utf-8") as source_file:
            source = source_file.read()
        self.assertIn("latest_agent_state=%s", source)
        self.assertIn("global _latest_agent_state_for_hume", source)

    def test_cleanup_skips_current_turn_speech(self):
        self.assertEqual(
            agent._assistant_cleanup_action(
                cleanup_reason="before_new_assistant_speech",
                current_user_turn_id=12,
                speech_turn_id=12,
                latest_user_state="listening",
            ),
            "skip",
        )

    def test_cleanup_interrupts_stale_turn_speech(self):
        self.assertEqual(
            agent._assistant_cleanup_action(
                cleanup_reason="before_new_assistant_speech",
                current_user_turn_id=12,
                speech_turn_id=11,
                latest_user_state="listening",
            ),
            "interrupt",
        )

    def test_cleanup_interrupts_when_user_is_speaking(self):
        self.assertEqual(
            agent._assistant_cleanup_action(
                cleanup_reason="before_new_assistant_speech",
                current_user_turn_id=12,
                speech_turn_id=12,
                latest_user_state="speaking",
            ),
            "interrupt",
        )


class TurnPolicyTests(unittest.TestCase):
    def test_complete_emotional_statement_commits_immediately(self):
        result = agent._make_turn_policy_decision("I felt really hurt by that.")
        self.assertEqual(result.classification, "EMOTIONAL_STATEMENT")
        self.assertEqual(result.decision, "COMMIT_NOW")
        self.assertTrue(result.should_start_generation)

    def test_short_meaningful_statement_commits_immediately(self):
        result = agent._make_turn_policy_decision("What time is it?", agent.detect_transcript_context("What time is it?"))
        self.assertEqual(result.classification, "COMPLETE_THOUGHT")
        self.assertEqual(result.decision, "COMMIT_NOW")

    def test_structurally_incomplete_thought_is_held(self):
        result = agent._make_turn_policy_decision("I was thinking because")
        self.assertEqual(result.classification, "INCOMPLETE_THOUGHT")
        self.assertEqual(result.decision, "HOLD_FOR_CONTINUATION")
        self.assertFalse(result.should_start_generation)

    def test_held_meaningful_fragment_commits_after_reply_deadline_policy(self):
        result = agent._make_turn_policy_decision("I started to feel like")
        self.assertEqual(result.decision, "HOLD_FOR_CONTINUATION")
        self.assertEqual(agent.TURN_HOLD_FRAGMENT_REPLY_DEADLINE_SECONDS, agent.TURN_HOLD_FRAGMENT_REPLY_DEADLINE_SECONDS)

    def test_related_continuation_merges_with_held_fragment(self):
        result = agent._make_turn_policy_decision(
            "it was about my brother",
            held_text="I was thinking about my brother because",
            held_created_at=100.0,
            now=103.0,
        )
        self.assertEqual(result.decision, "MERGE_WITH_HELD_FRAGMENT")
        self.assertTrue(result.should_merge_held_fragment)

    def test_unrelated_continuation_does_not_merge(self):
        result = agent._make_turn_policy_decision(
            "what time is it?",
            agent.detect_transcript_context("what time is it?"),
            held_text="I was thinking about my brother because",
            held_created_at=100.0,
            now=103.0,
        )
        self.assertEqual(result.decision, "FLUSH_HELD_AND_COMMIT_NEW")
        self.assertFalse(result.should_merge_held_fragment)

    def test_meta_complaint_never_merges_into_held_fragment(self):
        result = agent._make_turn_policy_decision(
            "you did not answer me",
            held_text="I was talking about my brother because",
            held_created_at=100.0,
            now=102.0,
        )
        self.assertEqual(result.classification, "META_COMPLAINT")
        self.assertEqual(result.decision, "RECOVER_FROM_SILENCE")
        self.assertFalse(result.should_merge_held_fragment)

    def test_llm_timeout_with_good_transcript_does_not_ask_repeat(self):
        text = agent._fallback_text_for_reason("first_token_timeout", "EMOTIONAL_STATEMENT")
        self.assertNotIn("say that again", text.lower())
        self.assertFalse(agent._fallback_requires_user_repeat("first_token_timeout", "EMOTIONAL_STATEMENT"))

    def test_unclear_audio_fallback_asks_repeat(self):
        self.assertTrue(agent._fallback_requires_user_repeat("audio_unclear", "UNCLEAR_AUDIO"))
        text = agent._fallback_text_for_reason("audio_unclear", "UNCLEAR_AUDIO")
        self.assertIn("say", text.lower())

    def test_api_connection_error_before_first_token_retries_once(self):
        class APIConnectionError(Exception):
            pass

        allowed, reason = agent._should_retry_openrouter_connection_error(
            APIConnectionError("transport"),
            first_token_seen=False,
            chunk_count=0,
            text_length=0,
            llm_turn_id=agent._current_turn_id,
            tts_started_for_turn=False,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "eligible")

    def test_api_connection_error_after_partial_text_does_not_retry(self):
        class APIConnectionError(Exception):
            pass

        allowed, reason = agent._should_retry_openrouter_connection_error(
            APIConnectionError("transport"),
            first_token_seen=True,
            chunk_count=1,
            text_length=5,
            llm_turn_id=agent._current_turn_id,
            tts_started_for_turn=False,
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "first_token_seen")

    def test_single_turn_cleanup_invariant_current_turn_skip(self):
        self.assertEqual(
            agent._assistant_cleanup_action(
                cleanup_reason="legacy_cleanup_call",
                current_user_turn_id=5,
                speech_turn_id=5,
                latest_user_state="listening",
            ),
            "skip",
        )


class OnUserTurnCompletedRegressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = {
            name: getattr(agent, name)
            for name in (
                "_held_turn_fragment_text",
                "_held_turn_fragment_created_at",
                "_held_turn_fragment_classification",
                "_held_turn_fragment_incomplete",
                "_current_turn_policy_decision",
            )
        }

    def tearDown(self):
        for name, value in self.state.items():
            setattr(agent, name, value)

    class Message:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

    class TurnCtx:
        def __init__(self, messages):
            self.messages = messages

        def add_message(self, role: str, content: str):
            self.messages.append(OnUserTurnCompletedRegressionTests.Message(role, content))

    async def _run_turn(self, text: str):
        lucy = object.__new__(agent.LucyAgent)
        lucy.runtime_context = None
        turn_ctx = self.TurnCtx([self.Message("system", "prompt"), self.Message("user", text)])
        new_message = turn_ctx.messages[-1]
        async def fake_interpret(transcript, **kwargs):
            return agent.detect_transcript_context(transcript)
        with patch.object(agent, "interpret_transcript_context", side_effect=fake_interpret), \
             patch.object(agent, "_endpointing_decision_for_transcript", return_value=("commit", "none", 0)), \
             patch.object(agent, "_prune_turn_context_messages", wraps=agent._prune_turn_context_messages) as prune:
            await lucy.on_user_turn_completed(turn_ctx, new_message)
        return prune, turn_ctx

    async def test_on_user_turn_completed_commit_now_does_not_raise(self):
        prune, _ = await self._run_turn("I felt really hurt by that.")
        self.assertTrue(prune.called)
        self.assertEqual(agent._current_turn_policy_decision, "COMMIT_NOW")

    async def test_on_user_turn_completed_low_information_filler_does_not_raise(self):
        prune, _ = await self._run_turn("Yeah.")
        self.assertTrue(prune.called)
        self.assertEqual(agent._current_turn_policy_decision, "IGNORE_LOW_INFORMATION_FILLER")

    async def test_on_user_turn_completed_pruning_invoked_with_message_list(self):
        prune, turn_ctx = await self._run_turn("What time is it?")
        self.assertTrue(prune.called)
        self.assertIsInstance(turn_ctx.messages, list)

    async def test_on_user_turn_completed_no_held_fragment_commit_now_does_not_raise(self):
        agent._held_turn_fragment_text = ""
        agent._held_turn_fragment_created_at = 0.0
        prune, _ = await self._run_turn("I felt complete.")
        self.assertTrue(prune.called)
        self.assertEqual(agent._current_turn_policy_decision, "COMMIT_NOW")

    async def test_on_user_turn_completed_held_fragment_not_merged_does_not_raise(self):
        agent._held_turn_fragment_text = "I was talking about my brother because"
        agent._held_turn_fragment_created_at = 100.0
        agent._held_turn_fragment_classification = "INCOMPLETE_THOUGHT"
        agent._held_turn_fragment_incomplete = True
        with patch("agent.time.monotonic", return_value=103.0):
            prune, turn_ctx = await self._run_turn("What time is it?")
        self.assertTrue(prune.called)
        self.assertEqual(agent._current_turn_policy_decision, "FLUSH_HELD_AND_COMMIT_NEW")
        user_messages = [message for message in turn_ctx.messages if message.role == "user"]
        self.assertEqual(user_messages[-1].content, "What time is it?")


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

    def test_pruning_accepts_message_list_directly(self):
        messages = [self.Message("system", "system")]
        for index in range(6):
            messages.append(self.Message("user", f"user {index}"))
            messages.append(self.Message("assistant", f"assistant {index}"))

        with patch.object(agent, "CONTEXT_WINDOW_TURNS", 4):
            total, kept, dropped = agent._prune_turn_context_messages(messages, turn_id=126)

        self.assertEqual(total, 13)
        self.assertEqual(kept, 9)
        self.assertEqual(dropped, 4)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].content, "user 2")

    def test_context_window_zero_clamps_to_safe_minimum(self):
        with patch.dict("os.environ", {"CONTEXT_WINDOW_TURNS": "0"}):
            self.assertEqual(agent.env_int_clamped("CONTEXT_WINDOW_TURNS", 10, 4, 100), 4)


if __name__ == "__main__":
    unittest.main()

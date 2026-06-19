import unittest
from unittest.mock import patch

import agent
import interaction_state as ist
from interaction_state import (
    ASSISTANT_THINKING,
    LISTENING,
    USER_SPEAKING,
    InteractionStateMachine,
)


class SttCandidateBindingTests(unittest.TestCase):
    def setUp(self):
        agent._stt_candidates.clear()

    def tearDown(self):
        agent._stt_candidates.clear()

    def test_each_final_gets_immutable_id(self):
        a = agent._record_stt_final_candidate("hello there", user_state="speaking", partial_count=2)
        b = agent._record_stt_final_candidate("Why?", user_state="speaking", partial_count=1)
        self.assertNotEqual(a, b)
        self.assertEqual([c["candidate_id"] for c in agent._stt_candidates], [a, b])
        # metadata captured
        self.assertEqual(agent._stt_candidates[0]["text"], "hello there")
        self.assertEqual(agent._stt_candidates[1]["user_state"], "speaking")

    def test_commit_binds_latest_matching_candidate_no_drift(self):
        agent._record_stt_final_candidate("a longer earlier utterance", user_state="speaking", partial_count=4)
        latest = agent._record_stt_final_candidate("Why?", user_state="speaking", partial_count=1)
        cid, drift, _ = agent._bind_candidate_for_commit("Why?")
        self.assertEqual(cid, latest)
        self.assertFalse(drift)

    def test_committing_lagged_transcript_flags_drift(self):
        # Symptom: "Why?" final has arrived (newest), but the commit still carries
        # the previous transcript -> must be flagged as drift, bound to the old id.
        old = agent._record_stt_final_candidate("a longer earlier utterance", user_state="speaking", partial_count=4)
        agent._record_stt_final_candidate("Why?", user_state="speaking", partial_count=1)
        cid, drift, latest_hash = agent._bind_candidate_for_commit("a longer earlier utterance")
        self.assertEqual(cid, old)
        self.assertTrue(drift)
        self.assertEqual(latest_hash, agent._text_hash("Why?"))

    def test_no_candidates_binds_synthetic_id_without_drift(self):
        cid, drift, _ = agent._bind_candidate_for_commit("hello")
        self.assertFalse(drift)
        self.assertTrue(cid.startswith("commit-") or cid == "empty")


class PipelineWithoutObservedSpeechInvariantTests(unittest.TestCase):
    """`turn_committed_by_pipeline_without_observed_user_speech` must only occur
    when the pipeline commits a turn with no observed user speech."""

    def test_not_emitted_when_user_speech_was_observed(self):
        sm = InteractionStateMachine()
        sm.on_user_speech_started()  # LISTENING -> USER_SPEAKING
        self.assertEqual(sm.state, USER_SPEAKING)
        # begin_turn must NOT inject the synthetic pipeline transition when speech
        # was observed: state stays put, so no such transition/reason is emitted.
        sm.begin_turn(1)
        self.assertEqual(sm.state, USER_SPEAKING)
        self.assertEqual(sm.turn_id, 1)

    def test_emitted_only_when_no_speech_observed(self):
        sm = InteractionStateMachine()
        self.assertEqual(sm.state, LISTENING)  # no observed user speech
        with self.assertLogs(ist.logger, level="INFO") as captured:
            sm.begin_turn(1)
        self.assertTrue(
            any("turn_committed_by_pipeline_without_observed_user_speech" in line for line in captured.output),
            captured.output,
        )


class IncompleteFragmentContinuationTests(unittest.TestCase):
    def test_incomplete_fragment_holds(self):
        result = agent._make_turn_policy_decision("I was thinking because")
        self.assertEqual(result.decision, "HOLD_FOR_CONTINUATION")

    def test_continuation_merges_with_held_fragment(self):
        result = agent._make_turn_policy_decision(
            "because it really mattered to me",
            held_text="I was thinking",
            held_created_at=__import__("time").monotonic(),
        )
        self.assertIn(result.decision, {"MERGE_WITH_HELD_FRAGMENT", "FLUSH_HELD_AND_COMMIT_NEW", "COMMIT_NOW"})


class BargeInDuringThinkingTests(unittest.TestCase):
    def test_real_barge_in_during_thinking_suppresses_pending_reply(self):
        import time

        with patch.object(agent, "LLM_TO_TTS_HANDOFF_GUARD_ENABLED", True), patch.multiple(
            agent,
            _barge_in_during_thinking_turn_id=7,
            _barge_in_started_at=time.monotonic() - 1.0,
            _barge_in_confirmed_real=True,
        ):
            suppress, _ = agent._handoff_guard_should_suppress(7)
        self.assertTrue(suppress)


class SuppressedZeroAudioNotCanonicalTests(unittest.TestCase):
    def setUp(self):
        agent._conversation_ledger.clear()

    def tearDown(self):
        agent._conversation_ledger.clear()

    def test_zero_audio_assistant_reply_excluded_from_canonical(self):
        agent._ledger_append("user", "what's the plan", visible=True, suppressed=False, turn_id=11)
        agent._ledger_append("assistant", "Here's the plan.", visible=True, suppressed=False, turn_id=11, provisional=True)
        # Reconcile against a zero-audio outcome.
        reason = agent._ledger_downgrade_reason_for_outcome(
            was_suppressed=False, interrupted="false", generated_bytes=0, playout_seconds=1.0
        )
        self.assertEqual(reason, "zero_audio")
        agent._ledger_downgrade_for_outcome(turn_id=11, speech_id=None, reason=reason)
        texts = [(e["role"], e["text"]) for e in agent._ledger_recent_canonical(5)]
        self.assertNotIn(("assistant", "Here's the plan."), texts)
        self.assertIn(("user", "what's the plan"), texts)


class TranscriptDriftCategorizationTests(unittest.TestCase):
    def setUp(self):
        agent._stt_candidates.clear()

    def tearDown(self):
        agent._stt_candidates.clear()

    def test_no_candidates(self):
        self.assertEqual(agent._categorize_transcript_drift("anything"), "no_candidates")

    def test_equal_is_none(self):
        agent._record_stt_final_candidate("what time is it", user_state="speaking", partial_count=1)
        self.assertEqual(agent._categorize_transcript_drift("what time is it"), "none")

    def test_committed_superset_is_expected_merge(self):
        # Newest final is a fragment; committed already merged it into a longer turn.
        agent._record_stt_final_candidate("the weather", user_state="speaking", partial_count=1)
        self.assertEqual(
            agent._categorize_transcript_drift("the weather in paris tomorrow"),
            "expected_merge_or_superset",
        )

    def test_newest_superset_is_real_lag(self):
        # Newest final extends what committed: committed lags one segment behind.
        agent._record_stt_final_candidate("the weather in paris tomorrow", user_state="speaking", partial_count=3)
        self.assertEqual(agent._categorize_transcript_drift("the weather"), "real_lag")

    def test_disjoint_is_ambiguous(self):
        agent._record_stt_final_candidate("call my mother", user_state="speaking", partial_count=2)
        self.assertEqual(agent._categorize_transcript_drift("buy some milk"), "ambiguous_commit")


class ToolRevalidationClassTests(unittest.TestCase):
    def test_meta_complaint(self):
        self.assertEqual(agent._tool_revalidation_class("META_COMPLAINT", "unknown", "none"), "meta_complaint")

    def test_high_dependency_is_additive(self):
        self.assertEqual(agent._tool_revalidation_class("COMPLETE_THOUGHT", "unknown", "high"), "additive_context")

    def test_default_is_unrelated(self):
        self.assertEqual(agent._tool_revalidation_class("COMPLETE_THOUGHT", "tool_request_search", "none"), "unrelated")


class SearchResultAuthorityGateTests(unittest.TestCase):
    def setUp(self):
        self._sm = agent._interaction_state
        self._saved = (
            self._sm.tool_result_speak_authority,
            self._sm.tool_result_pending_revalidation,
            self._sm.tool_result_paused_reason,
        )

    def tearDown(self):
        (
            self._sm.tool_result_speak_authority,
            self._sm.tool_result_pending_revalidation,
            self._sm.tool_result_paused_reason,
        ) = self._saved

    def test_authority_intact_passes_output_through(self):
        self._sm.tool_result_speak_authority = True
        self._sm.tool_result_pending_revalidation = False
        self.assertEqual(agent._search_result_authority_gate("RESULT", 5), "RESULT")

    def test_revoked_authority_withholds_result(self):
        self._sm.tool_result_speak_authority = False
        self._sm.tool_result_pending_revalidation = False
        self._sm.tool_result_paused_reason = "user_spoke_during_tool_call"
        gated = agent._search_result_authority_gate("RESULT", 5)
        self.assertNotEqual(gated, "RESULT")
        self.assertIn("withheld", gated.lower())

    def test_pending_revalidation_allows_but_flags(self):
        self._sm.tool_result_speak_authority = False
        self._sm.tool_result_pending_revalidation = True
        with self.assertLogs(agent.logger, level="WARNING") as captured:
            gated = agent._search_result_authority_gate("RESULT", 5)
        self.assertEqual(gated, "RESULT")
        self.assertTrue(
            any("search_result_authority_pending_revalidation=true" in line for line in captured.output)
        )


if __name__ == "__main__":
    unittest.main()

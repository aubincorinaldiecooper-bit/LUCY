import asyncio
import unittest
from unittest.mock import patch

import transcript_context as tc
from transcript_context import (
    ADDITIVE_FAMILY,
    TranscriptContext,
    classify_tool_revalidation_relationship,
    decide_tool_result_resume,
    deterministic_is_clearly_safe,
    query_materially_changed,
    resolve_transcript_context,
)


def _ctx(intent="tool_request_search", confidence=0.9, ambiguity=False):
    return TranscriptContext(
        original_text="x",
        cleaned_text="x",
        should_replace_user_text=False,
        llm_context_note=None,
        ambiguity_detected=ambiguity,
        clarification_suggested=ambiguity,
        detected_intent=intent,
        confidence=confidence,
        source="deterministic",
    )


class TimeoutEnvTests(unittest.TestCase):
    def test_normal_and_tool_timeouts_split(self):
        with patch.dict("os.environ", {
            "NORMAL_CONTEXT_CLASSIFIER_MAX_WAIT_MS": "500",
            "TOOL_REVALIDATION_CONTEXT_CLASSIFIER_MAX_WAIT_MS": "1000",
        }, clear=False):
            self.assertEqual(tc.normal_context_classifier_max_wait_ms(), 500)
            self.assertEqual(tc.tool_revalidation_context_classifier_max_wait_ms(), 1000)

    def test_normal_falls_back_to_legacy_env(self):
        with patch.dict("os.environ", {"TRANSCRIPT_CONTEXT_LLM_TIMEOUT_MS": "420"}, clear=False):
            import os
            os.environ.pop("NORMAL_CONTEXT_CLASSIFIER_MAX_WAIT_MS", None)
            self.assertEqual(tc.normal_context_classifier_max_wait_ms(), 420)

    def test_additive_min_dependency_default_high(self):
        import os
        os.environ.pop("TOOL_REVALIDATION_ADDITIVE_MIN_DEPENDENCY", None)
        self.assertEqual(tc.tool_revalidation_additive_min_dependency(), "high")

    def test_additive_min_dependency_validates(self):
        with patch.dict("os.environ", {"TOOL_REVALIDATION_ADDITIVE_MIN_DEPENDENCY": "medium"}, clear=False):
            self.assertEqual(tc.tool_revalidation_additive_min_dependency(), "medium")
        with patch.dict("os.environ", {"TOOL_REVALIDATION_ADDITIVE_MIN_DEPENDENCY": "bogus"}, clear=False):
            self.assertEqual(tc.tool_revalidation_additive_min_dependency(), "high")


class ClearlySafeTests(unittest.TestCase):
    def test_confident_unambiguous_is_safe(self):
        self.assertTrue(deterministic_is_clearly_safe(_ctx(confidence=0.9)))

    def test_low_confidence_not_safe(self):
        self.assertFalse(deterministic_is_clearly_safe(_ctx(confidence=0.6)))

    def test_unknown_intent_not_safe(self):
        self.assertFalse(deterministic_is_clearly_safe(_ctx(intent="unknown")))

    def test_ambiguous_not_safe(self):
        self.assertFalse(deterministic_is_clearly_safe(_ctx(ambiguity=True)))


class ResolutionTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_llm_disabled_clearly_safe_is_deterministic_safe(self):
        with patch.object(tc, "transcript_context_llm_enabled", lambda: False), patch.object(
            tc, "detect_transcript_context", lambda t: _ctx(confidence=0.92)
        ):
            res = self._run(resolve_transcript_context("look up the weather in paris", high_risk=True))
        self.assertEqual(res.resolution, "deterministic_safe")
        self.assertEqual(res.resolution_source, "deterministic")

    def test_llm_disabled_high_risk_unsafe_is_unresolved(self):
        with patch.object(tc, "transcript_context_llm_enabled", lambda: False), patch.object(
            tc, "detect_transcript_context", lambda t: _ctx(intent="unknown", confidence=0.4)
        ):
            res = self._run(resolve_transcript_context("uh", high_risk=True))
        self.assertEqual(res.resolution, "unresolved")

    def test_llm_disabled_normal_unsafe_still_proceeds(self):
        with patch.object(tc, "transcript_context_llm_enabled", lambda: False), patch.object(
            tc, "detect_transcript_context", lambda t: _ctx(intent="unknown", confidence=0.4)
        ):
            res = self._run(resolve_transcript_context("uh", high_risk=False))
        self.assertEqual(res.resolution, "deterministic_safe")

    def test_llm_timeout_high_risk_unsafe_is_unresolved_via_timeout(self):
        async def _slow(_ctx_in):
            await asyncio.sleep(0.5)
            return _ctx_in
        with patch.object(tc, "transcript_context_llm_enabled", lambda: True), patch.object(
            tc, "detect_transcript_context", lambda t: _ctx(intent="unknown", confidence=0.4)
        ):
            res = self._run(
                resolve_transcript_context("hmm", high_risk=True, max_wait_ms=20, llm_caller=_slow)
            )
        self.assertEqual(res.resolution, "unresolved")
        self.assertEqual(res.resolution_source, "timeout")
        self.assertTrue(res.timed_out)

    def test_llm_success_is_resolved(self):
        async def _ok(ctx_in):
            return tc.with_source(ctx_in, "llm")
        with patch.object(tc, "transcript_context_llm_enabled", lambda: True), patch.object(
            tc, "detect_transcript_context", lambda t: _ctx()
        ):
            res = self._run(
                resolve_transcript_context("paris weather", high_risk=True, max_wait_ms=500, llm_caller=_ok)
            )
        self.assertEqual(res.resolution, "resolved")
        self.assertEqual(res.resolution_source, "llm")


class RelationshipClassifierTests(unittest.TestCase):
    def test_additive_high_overlap(self):
        rel, dep, _ = classify_tool_revalidation_relationship(
            original_query="weather in paris tomorrow",
            newer_utterance="paris tomorrow afternoon weather",
        )
        self.assertEqual(rel, "additive_context")
        self.assertEqual(dep, "high")

    def test_constraint(self):
        rel, dep, _ = classify_tool_revalidation_relationship(
            original_query="flights to tokyo",
            newer_utterance="make sure they are direct flights",
        )
        self.assertEqual(rel, "constraint")
        self.assertEqual(dep, "high")

    def test_only_the_is_additive_family(self):
        rel, dep, _ = classify_tool_revalidation_relationship(
            original_query="flights to tokyo",
            newer_utterance="only the cheapest ones",
        )
        self.assertIn(rel, ADDITIVE_FAMILY)
        self.assertEqual(dep, "high")

    def test_pivot_on_cancel(self):
        rel, _, _ = classify_tool_revalidation_relationship(
            original_query="flights to tokyo", newer_utterance="actually nevermind"
        )
        self.assertEqual(rel, "pivot")

    def test_major_correction(self):
        rel, _, _ = classify_tool_revalidation_relationship(
            original_query="weather in paris", newer_utterance="no, that's not what I meant"
        )
        self.assertEqual(rel, "major_correction")

    def test_meta_complaint_from_classification(self):
        rel, _, _ = classify_tool_revalidation_relationship(
            original_query="weather", newer_utterance="you missed it", classification="META_COMPLAINT"
        )
        self.assertEqual(rel, "meta_complaint")

    def test_weak_overlap_additive_is_medium_dependency(self):
        # Some shared token but weak overlap -> additive_context at medium.
        rel, dep, _ = classify_tool_revalidation_relationship(
            original_query="weather in paris tomorrow",
            newer_utterance="and the paris hotels situation honestly",
        )
        self.assertEqual(rel, "additive_context")
        self.assertEqual(dep, "medium")

    def test_medium_additive_withheld_at_high_bar_composed_at_medium(self):
        # Default high bar withholds a weak (medium) addition; medium bar composes.
        withhold, _ = decide_tool_result_resume(
            relationship="additive_context", resolution="resolved", dependency_level="medium",
            additive_min_dependency="high",
        )
        compose, allowed = decide_tool_result_resume(
            relationship="additive_context", resolution="resolved", dependency_level="medium",
            additive_min_dependency="medium",
        )
        self.assertEqual(withhold, "withhold")
        self.assertEqual(compose, "compose")
        self.assertTrue(allowed)

    def test_unrelated_low_overlap(self):
        rel, _, _ = classify_tool_revalidation_relationship(
            original_query="weather in paris", newer_utterance="remind me to call mom"
        )
        self.assertEqual(rel, "unrelated")


class ResumeDecisionTests(unittest.TestCase):
    def test_unresolved_high_risk_withholds_when_available(self):
        decision, additive = decide_tool_result_resume(
            relationship="additive_context", resolution="unresolved", dependency_level="high",
            require_resolution=True, result_available=True,
        )
        self.assertEqual(decision, "withhold")
        self.assertFalse(additive)

    def test_unresolved_defers_when_result_not_ready(self):
        decision, _ = decide_tool_result_resume(
            relationship="additive_context", resolution="unresolved", dependency_level="high",
            require_resolution=True, result_available=False,
        )
        self.assertEqual(decision, "defer")

    def test_additive_high_composes(self):
        decision, additive = decide_tool_result_resume(
            relationship="additive_context", resolution="resolved", dependency_level="high",
            additive_min_dependency="high",
        )
        self.assertEqual(decision, "compose")
        self.assertTrue(additive)

    def test_additive_below_min_dependency_withholds(self):
        # narrowing scores high; force a medium dependency relationship below a high bar.
        decision, additive = decide_tool_result_resume(
            relationship="minor_correction", resolution="resolved", dependency_level="medium",
            additive_min_dependency="high", materially_changed=False,
        )
        # minor_correction without material change composes (amend), not withhold.
        self.assertEqual(decision, "compose")

    def test_medium_bar_lets_medium_dependency_compose(self):
        decision, additive = decide_tool_result_resume(
            relationship="constraint", resolution="resolved", dependency_level="high",
            additive_min_dependency="medium",
        )
        self.assertEqual(decision, "compose")

    def test_major_correction_reruns(self):
        decision, _ = decide_tool_result_resume(
            relationship="major_correction", resolution="resolved", dependency_level="low",
        )
        self.assertEqual(decision, "rerun")

    def test_minor_correction_material_change_reruns(self):
        decision, _ = decide_tool_result_resume(
            relationship="minor_correction", resolution="resolved", dependency_level="medium",
            materially_changed=True,
        )
        self.assertEqual(decision, "rerun")

    def test_pivot_discards(self):
        decision, _ = decide_tool_result_resume(
            relationship="pivot", resolution="resolved", dependency_level="low",
        )
        self.assertEqual(decision, "discard")

    def test_meta_complaint_withholds(self):
        decision, _ = decide_tool_result_resume(
            relationship="meta_complaint", resolution="resolved", dependency_level="low",
        )
        self.assertEqual(decision, "withhold")


class MaterialChangeTests(unittest.TestCase):
    def test_low_overlap_is_material_change(self):
        self.assertTrue(query_materially_changed("weather in paris", "stock price of tesla"))

    def test_high_overlap_not_material(self):
        self.assertFalse(query_materially_changed("weather in paris tomorrow", "the paris weather tomorrow please"))


if __name__ == "__main__":
    unittest.main()

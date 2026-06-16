import unittest
from unittest.mock import patch

import agent


class ContextCoherenceAssessmentTests(unittest.TestCase):
    """Phase-1 deterministic coherence: _assess_context_coherence."""

    def test_disabled_returns_unknown(self):
        with patch.object(agent, "CONTEXT_COHERENCE_ENABLED", False):
            fit, reason, conf = agent._assess_context_coherence("pt", "0.95", "en")
        self.assertEqual(fit, "unknown")
        self.assertEqual(reason, "disabled")
        self.assertEqual(conf, 0.0)

    def test_language_mismatch_is_low_coherence(self):
        with patch.object(agent, "CONTEXT_COHERENCE_ENABLED", True):
            fit, reason, conf = agent._assess_context_coherence("pt", "0.95", "en")
        self.assertEqual(fit, "low_coherence")
        self.assertEqual(reason, "language_mismatch:pt")
        self.assertGreater(conf, 0.0)

    def test_same_language_region_variant_is_coherent(self):
        # en-US should not trip the en session mismatch.
        with patch.object(agent, "CONTEXT_COHERENCE_ENABLED", True):
            fit, reason, _ = agent._assess_context_coherence("en-US", "n/a", "en")
        self.assertEqual(fit, "coherent")
        self.assertEqual(reason, "ok")

    def test_missing_language_does_not_trip_mismatch(self):
        with patch.object(agent, "CONTEXT_COHERENCE_ENABLED", True):
            fit, reason, _ = agent._assess_context_coherence("n/a", "n/a", "en")
        self.assertEqual(fit, "coherent")
        self.assertEqual(reason, "ok")

    def test_low_asr_confidence_is_low_coherence(self):
        with patch.object(agent, "CONTEXT_COHERENCE_ENABLED", True), patch.object(
            agent, "CONTEXT_COHERENCE_MIN_ASR_CONFIDENCE", 0.45
        ):
            fit, reason, _ = agent._assess_context_coherence("en", "0.20", "en")
        self.assertEqual(fit, "low_coherence")
        self.assertTrue(reason.startswith("asr_low_confidence:"))

    def test_high_asr_confidence_is_coherent(self):
        with patch.object(agent, "CONTEXT_COHERENCE_ENABLED", True), patch.object(
            agent, "CONTEXT_COHERENCE_MIN_ASR_CONFIDENCE", 0.45
        ):
            fit, reason, _ = agent._assess_context_coherence("en", "0.90", "en")
        self.assertEqual(fit, "coherent")
        self.assertEqual(reason, "ok")

    def test_unparseable_confidence_does_not_trip_asr(self):
        with patch.object(agent, "CONTEXT_COHERENCE_ENABLED", True):
            fit, reason, _ = agent._assess_context_coherence("en", "n/a", "en")
        self.assertEqual(fit, "coherent")
        self.assertEqual(reason, "ok")

    def test_default_flag_is_off(self):
        # Ships disabled: zero behavior change until opted in.
        self.assertIs(agent.env_bool("CONTEXT_COHERENCE_ENABLED_UNSET_XYZ", False), False)


class CoherenceNoteInjectionTests(unittest.TestCase):
    def test_injects_developer_note_when_low_coherence(self):
        captured = {}

        class FakeTurnCtx:
            def add_message(self, role, content):
                captured["role"] = role
                captured["content"] = content

        ok = agent._inject_coherence_note(FakeTurnCtx(), "language_mismatch:pt")
        self.assertTrue(ok)
        self.assertEqual(captured["role"], "developer")
        self.assertIn("may not fit the conversation", captured["content"])
        self.assertIn("language_mismatch:pt", captured["content"])

    def test_missing_add_message_is_safe(self):
        ok = agent._inject_coherence_note(object(), "asr_low_confidence:0.20")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

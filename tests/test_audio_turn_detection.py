import unittest
from unittest.mock import patch

import agent
from interaction_state import build_audio_environment_decision


class AudioEnvironmentDecisionTests(unittest.TestCase):
    def test_clean_when_no_instability(self):
        d = build_audio_environment_decision()
        self.assertEqual(d.noise_state, "clean")
        self.assertEqual(d.action_hint, "normal")
        self.assertEqual(d.speech_stability, "stable")

    def test_noisy_when_multiple_instability_signals(self):
        d = build_audio_environment_decision(
            false_speech_start_count_recent=4,
            candidate_turn_count_recent=5,
            unstable_partial_transcripts=True,
        )
        self.assertEqual(d.noise_state, "noisy")
        self.assertEqual(d.speech_stability, "unstable")
        self.assertEqual(d.transcript_stability, "unstable")
        self.assertEqual(d.action_hint, "ask_repair")

    def test_noisy_stable_transcript_is_hold(self):
        d = build_audio_environment_decision(
            false_speech_start_count_recent=3,
            candidate_turn_count_recent=3,
            short_noisy_fragment_detected=True,
        )
        self.assertEqual(d.noise_state, "noisy")
        self.assertEqual(d.action_hint, "hold")

    def test_uncertain_when_mild(self):
        d = build_audio_environment_decision(short_noisy_fragment_detected=True)
        self.assertEqual(d.noise_state, "uncertain")

    def test_snr_override_clean(self):
        d = build_audio_environment_decision(false_speech_start_count_recent=9, snr_db=25.0)
        self.assertEqual(d.noise_state, "clean")

    def test_snr_override_noisy(self):
        d = build_audio_environment_decision(snr_db=4.0)
        self.assertEqual(d.noise_state, "noisy")

    def test_audio_status_check_sets_hint(self):
        d = build_audio_environment_decision(is_audio_status_check=True)
        self.assertEqual(d.action_hint, "audio_status")


class AudioStatusCheckTests(unittest.TestCase):
    def test_detects_audio_status_phrases(self):
        for phrase in [
            "Can you hear me?",
            "Can you hear me now?",
            "Do you hear me?",
            "Is it noisy?",
            "Can you tell if it’s noisy?",
            "Are you still there?",
        ]:
            self.assertTrue(agent._is_audio_status_check(phrase), phrase)

    def test_ignores_ordinary_questions(self):
        for phrase in ["What time is it?", "Tell me a joke.", "I was thinking about dinner."]:
            self.assertFalse(agent._is_audio_status_check(phrase), phrase)

    def test_response_text_matches_state(self):
        clean = build_audio_environment_decision(is_audio_status_check=True)
        self.assertIn("clearly", agent._audio_status_response_text(clean))
        noisy_stable = build_audio_environment_decision(
            is_audio_status_check=True, false_speech_start_count_recent=3, candidate_turn_count_recent=3, short_noisy_fragment_detected=True
        )
        self.assertIn("background noise", agent._audio_status_response_text(noisy_stable))
        noisy_unstable = build_audio_environment_decision(
            is_audio_status_check=True, false_speech_start_count_recent=4, candidate_turn_count_recent=5, unstable_partial_transcripts=True
        )
        self.assertIn("harder to catch", agent._audio_status_response_text(noisy_unstable))


class TurnDetectorBuilderTests(unittest.TestCase):
    def test_disabled_returns_none_with_reason(self):
        with patch.object(agent, "LIVEKIT_TURN_DETECTOR_ENABLED", False):
            det, info = agent.build_audio_turn_detector()
        self.assertIsNone(det)
        self.assertFalse(info["enabled"])
        self.assertEqual(info["error"], "disabled_by_config")

    def test_enabled_builds_detector_on_supported_sdk(self):
        # This environment has livekit-agents >= 1.6.1, so the detector is available.
        with patch.object(agent, "LIVEKIT_TURN_DETECTOR_ENABLED", True):
            det, info = agent.build_audio_turn_detector()
        if info["available"]:
            self.assertIsNotNone(det)
            self.assertEqual(info["class"], "TurnDetector")
            self.assertFalse(info["fallback_used"])
        else:  # pragma: no cover - only on older SDKs
            self.assertIsNone(det)
            self.assertTrue(info["fallback_used"])
            self.assertIn("inference.TurnDetector unavailable", info["error"])

    def test_unavailable_sdk_falls_back_cleanly(self):
        with patch.object(agent, "LIVEKIT_TURN_DETECTOR_ENABLED", True), patch.object(agent, "_lk_inference", None):
            det, info = agent.build_audio_turn_detector()
        self.assertIsNone(det)
        self.assertTrue(info["fallback_used"])
        self.assertFalse(info["available"])

    def test_invalid_threshold_is_ignored_not_fatal(self):
        with patch.object(agent, "LIVEKIT_TURN_DETECTOR_ENABLED", True), patch.object(
            agent, "LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD", "not-a-float"
        ):
            det, info = agent.build_audio_turn_detector()
        # Bad threshold must not crash the builder.
        self.assertEqual(info["error"] in ("none",) or info["available"] is False, True)


if __name__ == "__main__":
    unittest.main()

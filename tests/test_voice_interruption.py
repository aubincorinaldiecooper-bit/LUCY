import os
import unittest
from unittest import mock

import voice_interruption as vi


CFG = vi.InterruptionConfig()  # defaults: 2 words, 0.65s, resume, 1.0s timeout


class InterruptionGateTests(unittest.TestCase):
    def test_noise_no_transcript_does_not_interrupt(self):
        d, r = vi.classify_interruption_candidate(
            transcript="", duration_s=0.3, recent_assistant_texts=[], config=CFG)
        self.assertEqual(d, "pending")  # candidate only, keep playing
        self.assertEqual(r, "awaiting_transcript")

    def test_false_interruption_resumes_after_timeout(self):
        # mic active >1s but still no meaningful transcript -> false -> resume
        d, r = vi.classify_interruption_candidate(
            transcript="", duration_s=1.2, recent_assistant_texts=[], config=CFG)
        self.assertEqual(d, "false")
        self.assertEqual(r, "no_transcript_timeout")

    def test_strong_one_word_command_interrupts(self):
        for cmd in ("stop", "wait", "pause", "hold", "no"):
            d, r = vi.classify_interruption_candidate(
                transcript=cmd, duration_s=0.7, recent_assistant_texts=[], config=CFG)
            self.assertEqual((d, r), ("confirmed", "strong_command"), cmd)

    def test_two_plus_meaningful_words_interrupt(self):
        d, r = vi.classify_interruption_candidate(
            transcript="actually hold on a second", duration_s=0.8,
            recent_assistant_texts=[], config=CFG)
        self.assertEqual(d, "confirmed")

    def test_single_non_command_word_is_pending(self):
        d, r = vi.classify_interruption_candidate(
            transcript="okay", duration_s=0.8, recent_assistant_texts=[], config=CFG)
        self.assertEqual(d, "pending")
        self.assertEqual(r, "insufficient_words")

    def test_too_short_is_pending(self):
        d, r = vi.classify_interruption_candidate(
            transcript="hold on please", duration_s=0.3, recent_assistant_texts=[], config=CFG)
        self.assertEqual(d, "pending")
        self.assertEqual(r, "too_short")

    def test_echo_does_not_interrupt(self):
        assistant = ["so the first step is to write down what's weighing on you"]
        d, r = vi.classify_interruption_candidate(
            transcript="the first step is to write down what's weighing",
            duration_s=1.0, recent_assistant_texts=assistant, config=CFG)
        self.assertEqual(d, "false")
        self.assertEqual(r, "echo_of_assistant")


class TailOutcomeTests(unittest.TestCase):
    def test_clean_playout(self):
        out = vi.classify_tail_outcome(
            generated_audio_duration_s=3.0, playout_started_at=10.0,
            playout_completed_at=13.0, interrupted_at=None, interrupted=False,
            hume_requests_during_speech=1)
        self.assertEqual(out, vi.CLEAN_PLAYOUT)
        self.assertFalse(vi.is_audible_cutoff(out))

    def test_interruption_before_playout_complete_is_tail_cut(self):
        out = vi.classify_tail_outcome(
            generated_audio_duration_s=3.0, playout_started_at=10.0,
            playout_completed_at=None, interrupted_at=11.2, interrupted=True,
            hume_requests_during_speech=1)
        self.assertEqual(out, vi.LIKELY_TAIL_CUT)
        self.assertTrue(vi.is_audible_cutoff(out))

    def test_interruption_after_playout_complete_is_clean(self):
        out = vi.classify_tail_outcome(
            generated_audio_duration_s=3.0, playout_started_at=10.0,
            playout_completed_at=13.0, interrupted_at=13.4, interrupted=True,
            hume_requests_during_speech=1)
        self.assertEqual(out, vi.INTERRUPTION_AFTER_PLAYOUT)
        self.assertFalse(vi.is_audible_cutoff(out))

    def test_ghost_stale_handle_no_audio_is_not_cutoff(self):
        out = vi.classify_tail_outcome(
            generated_audio_duration_s=0.0, playout_started_at=None,
            playout_completed_at=None, interrupted_at=11.0, interrupted=True,
            hume_requests_during_speech=0)
        self.assertEqual(out, vi.GHOST_STALE_NO_AUDIO)
        self.assertFalse(vi.is_audible_cutoff(out))

    def test_stale_cleanup_only(self):
        out = vi.classify_tail_outcome(
            generated_audio_duration_s=0.0, playout_started_at=None,
            playout_completed_at=None, interrupted_at=None, interrupted=False,
            was_stale=True, was_active=False, hume_requests_during_speech=0)
        self.assertEqual(out, vi.STALE_CLEANUP_ONLY)
        self.assertFalse(vi.is_audible_cutoff(out))

    def test_audio_produced_when_hume_counter_is_zero(self):
        # Production reality: Hume request counter stays 0 without HTTP debug,
        # but generated audio duration proves audio was synthesized + played.
        # Must NOT be misclassified as a ghost handle.
        out = vi.classify_tail_outcome(
            generated_audio_duration_s=10.1, playout_started_at=100.0,
            playout_completed_at=113.7, interrupted_at=None, interrupted=False,
            hume_requests_during_speech=0)
        self.assertEqual(out, vi.CLEAN_PLAYOUT)
        self.assertFalse(vi.is_audible_cutoff(out))

    def test_interrupted_after_full_playout_with_zero_hume_counter(self):
        # Barge-in lands after the full audio already played (playout >> generated)
        # -> interruption after playout, not an audible tail cut.
        out = vi.classify_tail_outcome(
            generated_audio_duration_s=2.9, playout_started_at=100.0,
            playout_completed_at=104.8, interrupted_at=104.8, interrupted=True,
            hume_requests_during_speech=0)
        self.assertEqual(out, vi.INTERRUPTION_AFTER_PLAYOUT)
        self.assertFalse(vi.is_audible_cutoff(out))


class TailAudioSilenceTests(unittest.TestCase):
    def test_silent_tail_is_silence(self):
        # near-zero trailing amplitude -> clean decay, synthesis tail intact
        self.assertTrue(vi.tail_ends_in_silence(50))
        self.assertLess(vi.peak_dbfs(50), -40.0)

    def test_loud_tail_is_not_silence(self):
        # loud final sample -> TTS clipped mid-word
        self.assertFalse(vi.tail_ends_in_silence(20000))
        self.assertGreater(vi.peak_dbfs(20000), -40.0)

    def test_pure_silence_is_negative_infinity(self):
        self.assertEqual(vi.peak_dbfs(0), float("-inf"))
        self.assertTrue(vi.tail_ends_in_silence(0))

    def test_threshold_is_configurable(self):
        # 327 ~= -40 dBFS at int16 full scale; raising the bar flips the verdict
        self.assertTrue(vi.tail_ends_in_silence(300))
        self.assertFalse(vi.tail_ends_in_silence(300, silence_dbfs=-50.0))


class TurnDetectionResolveTests(unittest.TestCase):
    def test_audio_is_invalid_resolves_to_vad(self):
        resolved, ok = vi.resolve_turn_detection_mode("audio")
        self.assertEqual(resolved, "vad")
        self.assertFalse(ok)

    def test_default_is_valid(self):
        self.assertEqual(vi.resolve_turn_detection_mode("default"), ("default", True))

    def test_vad_and_stt_valid(self):
        self.assertEqual(vi.resolve_turn_detection_mode("VAD"), ("vad", True))
        self.assertEqual(vi.resolve_turn_detection_mode("stt"), ("stt", True))

    def test_missing_resolves_to_vad(self):
        self.assertEqual(vi.resolve_turn_detection_mode(None), ("vad", False))


class ConfigTests(unittest.TestCase):
    def test_defaults_match_target(self):
        c = vi.InterruptionConfig()
        self.assertEqual(c.min_words, 2)
        self.assertEqual(c.min_duration, 0.65)
        self.assertTrue(c.resume_false_interruption)
        self.assertEqual(c.false_interruption_timeout, 1.0)

    def test_from_env_overrides(self):
        env = {
            "LIVEKIT_INTERRUPTION_MIN_WORDS": "3",
            "LIVEKIT_INTERRUPTION_MIN_DURATION": "0.8",
            "LIVEKIT_RESUME_FALSE_INTERRUPTION": "false",
            "LIVEKIT_FALSE_INTERRUPTION_TIMEOUT": "1.5",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            c = vi.InterruptionConfig.from_env()
        self.assertEqual(c.min_words, 3)
        self.assertEqual(c.min_duration, 0.8)
        self.assertFalse(c.resume_false_interruption)
        self.assertEqual(c.false_interruption_timeout, 1.5)


if __name__ == "__main__":
    unittest.main()

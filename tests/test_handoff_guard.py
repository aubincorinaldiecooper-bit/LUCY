import time
import unittest
from unittest.mock import patch

import agent


class HandoffGuardSuppressionTests(unittest.TestCase):
    """Covers _handoff_guard_should_suppress: only a real barge-in during thinking
    (and only when the guard is enabled) suppresses the pending reply before TTS."""

    def _set(self, **kwargs):
        # Helper to patch the module-level barge-in state for one decision.
        defaults = {
            "LLM_TO_TTS_HANDOFF_GUARD_ENABLED": True,
            "HANDOFF_GUARD_MIN_SPEECH_MS": 350,
            "_barge_in_during_thinking_turn_id": 5,
            "_barge_in_started_at": time.monotonic(),
            "_barge_in_confirmed_real": False,
        }
        defaults.update(kwargs)
        return patch.multiple(agent, **defaults)

    def test_disabled_guard_never_suppresses(self):
        with self._set(LLM_TO_TTS_HANDOFF_GUARD_ENABLED=False, _barge_in_confirmed_real=True):
            suppress, reason = agent._handoff_guard_should_suppress(5)
        self.assertFalse(suppress)
        self.assertEqual(reason, "guard_disabled")

    def test_no_barge_in_does_not_suppress(self):
        with self._set(_barge_in_during_thinking_turn_id=0):
            suppress, reason = agent._handoff_guard_should_suppress(5)
        self.assertFalse(suppress)
        self.assertEqual(reason, "no_barge_in_for_turn")

    def test_barge_in_for_different_turn_does_not_suppress(self):
        with self._set(_barge_in_during_thinking_turn_id=4, _barge_in_confirmed_real=True):
            suppress, reason = agent._handoff_guard_should_suppress(5)
        self.assertFalse(suppress)
        self.assertEqual(reason, "no_barge_in_for_turn")

    def test_confirmed_real_barge_in_suppresses(self):
        with self._set(_barge_in_confirmed_real=True):
            suppress, reason = agent._handoff_guard_should_suppress(5)
        self.assertTrue(suppress)
        self.assertTrue(reason.startswith("real_barge_in_during_thinking"))

    def test_sustained_ongoing_barge_in_suppresses(self):
        # Still speaking, no stop yet, but elapsed already exceeds the threshold.
        started = time.monotonic() - 1.0  # 1000ms ago > 350ms
        with self._set(_barge_in_started_at=started, _barge_in_confirmed_real=False):
            suppress, reason = agent._handoff_guard_should_suppress(5)
        self.assertTrue(suppress)
        self.assertTrue(reason.startswith("real_barge_in_during_thinking"))

    def test_brief_blip_below_threshold_does_not_suppress(self):
        started = time.monotonic() - 0.05  # 50ms ago < 350ms
        with self._set(_barge_in_started_at=started, _barge_in_confirmed_real=False):
            suppress, reason = agent._handoff_guard_should_suppress(5)
        self.assertFalse(suppress)
        self.assertTrue(reason.startswith("barge_in_below_min_speech"))

    def test_default_flag_is_off(self):
        # Safety: the guard ships disabled so behavior is unchanged until opted in.
        self.assertIs(agent.env_bool("LLM_TO_TTS_HANDOFF_GUARD_ENABLED_UNSET_XYZ", False), False)


if __name__ == "__main__":
    unittest.main()

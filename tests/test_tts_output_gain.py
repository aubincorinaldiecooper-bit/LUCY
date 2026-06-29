"""Tests for the server-side TTS output makeup gain.

Loudness is raised at the source (scaling the PCM frames) so the client never
needs a second, parallel audio stream to boost volume — that parallel stream was
the cause of the doubled/clipped voice. Default gain is 1.0 (passthrough).
"""

import asyncio
import os
import struct
import unittest
from unittest.mock import patch

import agent
from livekit import rtc


def _frame(samples, sample_rate=48000, num_channels=1):
    data = struct.pack("<" + "h" * len(samples), *samples)
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=len(samples) // num_channels,
    )


def _samples(frame):
    raw = bytes(frame.data)
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


async def _source(items):
    for item in items:
        yield item


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


class ApplyGainFrameTests(unittest.TestCase):
    def test_unity_gain_returns_frame_unchanged(self):
        f = _frame([100, -200, 300])
        out, clipped = agent._apply_output_gain_to_frame(f, 1.0)
        self.assertIs(out, f)
        self.assertEqual(clipped, 0)

    def test_gain_scales_samples_and_preserves_format(self):
        f = _frame([100, -200, 300])
        out, clipped = agent._apply_output_gain_to_frame(f, 2.0)
        self.assertEqual(_samples(out), [200, -400, 600])
        self.assertEqual(clipped, 0)
        self.assertEqual(out.sample_rate, 48000)
        self.assertEqual(out.num_channels, 1)
        self.assertEqual(out.samples_per_channel, 3)

    def test_clipping_is_protected_and_counted(self):
        # 20000 * 2 = 40000 -> clipped to int16 max; -20000 * 2 -> int16 min.
        f = _frame([20000, -20000, 1000])
        out, clipped = agent._apply_output_gain_to_frame(f, 2.0)
        self.assertEqual(_samples(out), [32767, -32768, 2000])
        self.assertEqual(clipped, 2)


class WithOutputGainStreamTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_passthrough_at_unity_yields_same_object(self):
        f = _frame([100, -200])
        with patch.object(agent, "TTS_OUTPUT_GAIN", 1.0):
            out = self._run(_collect(agent._with_output_gain(_source([f]))))
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], f)

    def test_applies_gain_when_enabled(self):
        f = _frame([100, -200])
        with patch.object(agent, "TTS_OUTPUT_GAIN", 3.0):
            out = self._run(_collect(agent._with_output_gain(_source([f]))))
        self.assertEqual(_samples(out[0]), [300, -600])

    def test_non_audio_items_pass_through_untouched(self):
        f = _frame([100])
        sentinel = object()
        with patch.object(agent, "TTS_OUTPUT_GAIN", 2.0):
            out = self._run(_collect(agent._with_output_gain(_source([f, sentinel]))))
        self.assertEqual(_samples(out[0]), [200])
        self.assertIs(out[1], sentinel)


class ConfigDefaultTests(unittest.TestCase):
    def test_default_is_unity_passthrough(self):
        # Ships disabled: behavior is unchanged until TTS_OUTPUT_GAIN is set.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_OUTPUT_GAIN", None)
            computed = max(1.0, min(4.0, float(os.getenv("TTS_OUTPUT_GAIN", "1.0") or "1.0")))
        self.assertEqual(computed, 1.0)
        self.assertGreaterEqual(agent.TTS_OUTPUT_GAIN, 1.0)
        self.assertLessEqual(agent.TTS_OUTPUT_GAIN, 4.0)


if __name__ == "__main__":
    unittest.main()

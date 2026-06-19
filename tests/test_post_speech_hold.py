import unittest
from unittest.mock import patch

import agent
from livekit import rtc


def _frame(sample_rate=24000, num_channels=1, samples=480):
    return rtc.AudioFrame(
        data=b"\x11\x22" * (samples * num_channels),
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples,
    )


async def _source(frames):
    for frame in frames:
        yield frame


async def _drain(async_gen):
    out = []
    async for item in async_gen:
        out.append(item)
    return out


def _is_silent(frame: rtc.AudioFrame) -> bool:
    return set(bytes(frame.data)) <= {0}


class PostSpeechHoldTests(unittest.IsolatedAsyncioTestCase):
    """Regression coverage for NameError: name 'TTS_POST_SPEECH_HOLD_MS' is not defined."""

    async def test_no_crash_and_passthrough_when_env_absent(self):
        # Mirrors the production default (env var unset -> 0 -> disabled).
        self.assertEqual(agent.env_int_clamped("TTS_POST_SPEECH_HOLD_MS_UNSET_XYZ", 0, 0, 5000), 0)
        frames = [_frame(), _frame()]
        with patch.object(agent, "TTS_POST_SPEECH_HOLD_MS", 0):
            out = await _drain(agent._with_post_speech_hold(_source(frames)))
        self.assertEqual(out, frames)  # no crash, no silence appended

    async def test_appends_trailing_silence_when_configured(self):
        frames = [_frame(sample_rate=24000, num_channels=1, samples=480)]
        with patch.object(agent, "TTS_POST_SPEECH_HOLD_MS", 200):
            out = await _drain(agent._with_post_speech_hold(_source(frames)))

        self.assertGreater(len(out), len(frames))
        # Original frame passes through untouched, the appended tail is pure silence.
        self.assertIs(out[0], frames[0])
        appended = out[1:]
        self.assertTrue(appended and all(_is_silent(f) for f in appended))
        # ~200ms of 24kHz audio in ~20ms frames -> roughly 10 silent frames.
        total_samples = sum(f.samples_per_channel for f in appended)
        self.assertGreaterEqual(total_samples, int(24000 * 0.2) - 480)

    async def test_no_silence_when_source_yields_no_audio(self):
        with patch.object(agent, "TTS_POST_SPEECH_HOLD_MS", 200):
            out = await _drain(agent._with_post_speech_hold(_source([])))
        self.assertEqual(out, [])  # nothing to pad, no crash


if __name__ == "__main__":
    unittest.main()

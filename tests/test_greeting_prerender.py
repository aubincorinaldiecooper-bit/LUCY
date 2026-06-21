import io
import types
import unittest
import wave

import agent


def _fake_frame(data: bytes, sample_rate: int = 48000, num_channels: int = 1):
    return types.SimpleNamespace(data=data, sample_rate=sample_rate, num_channels=num_channels)


def _fake_event(frame):
    return types.SimpleNamespace(frame=frame)


class _FakeStream:
    """Async-iterable stand-in for hume.TTS.synthesize()'s ChunkedStream."""

    def __init__(self, events, *, raise_on_iter: bool = False):
        self._events = events
        self._raise_on_iter = raise_on_iter
        self.closed = False

    def __aiter__(self):
        async def _gen():
            if self._raise_on_iter:
                raise RuntimeError("synthesize boom")
            for event in self._events:
                yield event

        return _gen()

    async def aclose(self):
        self.closed = True


class _FakeTTS:
    def __init__(self, stream):
        self._stream = stream
        self.synthesize_calls = 0

    def synthesize(self, text):
        self.synthesize_calls += 1
        return self._stream


class GreetingPrerenderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_provider = agent.TTS_PROVIDER
        self._orig_build_tts = agent.build_tts
        agent.TTS_PROVIDER = "hume"
        agent._prerendered_greeting_wav = None
        agent._prerender_greeting_in_flight = False

    def tearDown(self):
        agent.TTS_PROVIDER = self._orig_provider
        agent.build_tts = self._orig_build_tts
        agent._prerendered_greeting_wav = None
        agent._prerender_greeting_in_flight = False

    async def test_prerender_populates_valid_wav_buffer(self):
        events = [_fake_event(_fake_frame(b"\x01\x02" * 480)) for _ in range(3)]
        stream = _FakeStream(events)
        agent.build_tts = lambda: _FakeTTS(stream)

        await agent._prerender_greeting()

        self.assertIsNotNone(agent._prerendered_greeting_wav)
        # The buffer must be a valid 16-bit WAV the cached-playback path accepts.
        agent._validate_cached_wav_audio(agent._prerendered_greeting_wav)
        with wave.open(io.BytesIO(agent._prerendered_greeting_wav), "rb") as wav:
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 48000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertGreater(wav.getnframes(), 0)
        self.assertTrue(stream.closed)
        self.assertFalse(agent._prerender_greeting_in_flight)

    async def test_prerender_is_noop_when_already_rendered(self):
        agent._prerendered_greeting_wav = b"already"
        called = {"build": False}

        def _build():
            called["build"] = True
            raise AssertionError("build_tts should not be called when already rendered")

        agent.build_tts = _build
        await agent._prerender_greeting()
        self.assertEqual(agent._prerendered_greeting_wav, b"already")
        self.assertFalse(called["build"])

    async def test_prerender_skips_when_provider_not_hume(self):
        agent.TTS_PROVIDER = "deepgram"
        agent.build_tts = lambda: self.fail("build_tts should not run for non-hume provider")
        await agent._prerender_greeting()
        self.assertIsNone(agent._prerendered_greeting_wav)

    async def test_prerender_failure_leaves_buffer_none_and_resets_flag(self):
        stream = _FakeStream([], raise_on_iter=True)
        agent.build_tts = lambda: _FakeTTS(stream)

        await agent._prerender_greeting()

        self.assertIsNone(agent._prerendered_greeting_wav)
        # Flag must reset so a later job can retry the render.
        self.assertFalse(agent._prerender_greeting_in_flight)
        self.assertTrue(stream.closed)

    async def test_prerender_no_frames_yields_no_buffer(self):
        stream = _FakeStream([])  # empty, no error
        agent.build_tts = lambda: _FakeTTS(stream)

        await agent._prerender_greeting()

        self.assertIsNone(agent._prerendered_greeting_wav)
        self.assertFalse(agent._prerender_greeting_in_flight)


if __name__ == "__main__":
    unittest.main()

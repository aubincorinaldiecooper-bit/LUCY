import io
import os
import unittest
import wave
from unittest.mock import patch

import hume_evi_bridge
import server


def _make_wav(pcm: bytes, sample_rate: int = 48000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class _FakeRoom:
    """Minimal stand-in; the real rtc.Room raises if touched before connect()."""


class _FakeJobContext:
    """Records connect() so tests can assert the room is connected first."""

    def __init__(self, calls, *, connect_error=None):
        self._calls = calls
        self._connect_error = connect_error
        self.room = _FakeRoom()

    async def connect(self):
        self._calls.append("connect")
        if self._connect_error is not None:
            raise self._connect_error


class HumeEVIBridgeConfigTests(unittest.TestCase):
    def test_voice_engine_defaults_current(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(hume_evi_bridge.voice_engine(), "current")

    def test_voice_engine_accepts_hume_evi(self):
        with patch.dict(os.environ, {"VOICE_ENGINE": "hume_evi"}):
            self.assertEqual(hume_evi_bridge.voice_engine(), "hume_evi")

    def test_missing_hume_evi_vars_fail_clearly(self):
        with patch.dict(os.environ, {"VOICE_ENGINE": "hume_evi"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HUME_API_KEY"):
                hume_evi_bridge.load_hume_evi_settings()

    def test_hume_evi_websocket_url_uses_server_side_api_key(self):
        with patch.dict(os.environ, {"HUME_API_KEY": "key", "HUME_SECRET_KEY": "secret", "HUME_EVI_CONFIG_ID": "cfg", "HUME_EVI_VERSION": "evi-3", "HUME_CLM_BEARER_TOKEN": "token"}, clear=True):
            settings = hume_evi_bridge.load_hume_evi_settings()
            self.assertIn("api_key=key", settings.websocket_url)
            self.assertIn("config_id=cfg", settings.websocket_url)
            self.assertIn("evi_version=evi-3", settings.websocket_url)


class VoiceEngineSelectorBootstrapTests(unittest.TestCase):
    """Regression coverage for the entrypoint NameError: voice_engine is not defined."""

    def test_agent_imports_voice_engine_selector(self):
        import agent

        self.assertTrue(callable(agent.voice_engine))
        self.assertIs(agent.voice_engine, hume_evi_bridge.voice_engine)

    def test_agent_imports_hume_evi_bridge_entrypoint(self):
        import agent

        self.assertTrue(callable(agent.run_hume_evi_bridge))
        self.assertIs(agent.run_hume_evi_bridge, hume_evi_bridge.run_hume_evi_bridge)

    def test_agent_selects_current_engine_by_default(self):
        import agent

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(agent.voice_engine(), "current")

    def test_agent_selects_hume_evi_through_config(self):
        import agent

        with patch.dict(os.environ, {"VOICE_ENGINE": "hume_evi"}):
            self.assertEqual(agent.voice_engine(), "hume_evi")

    def test_invalid_voice_engine_falls_back_to_current(self):
        import agent

        with patch.dict(os.environ, {"VOICE_ENGINE": "not-a-real-engine"}):
            self.assertEqual(agent.voice_engine(), "current")


class HumeEVIBootstrapOrderTests(unittest.IsolatedAsyncioTestCase):
    """Regression coverage for: cannot access local participant before connecting.

    The EVI bridge publishes a track / subscribes to audio, which requires a
    connected room. The cascaded pipeline connects via AgentSession.start(); the
    EVI path has no session, so the entrypoint helper must connect first.
    """

    async def test_room_connected_before_bridge_runs(self):
        import agent

        calls = []

        async def fake_bridge(room):
            calls.append("bridge")

        ctx = _FakeJobContext(calls)
        with patch.object(agent, "run_hume_evi_bridge", fake_bridge):
            await agent._run_hume_evi_voice_engine(ctx)

        self.assertEqual(calls, ["connect", "bridge"])

    async def test_bridge_not_run_when_connect_fails(self):
        import agent

        calls = []

        async def fake_bridge(room):
            calls.append("bridge")

        ctx = _FakeJobContext(calls, connect_error=Exception("connect boom"))
        with patch.object(agent, "run_hume_evi_bridge", fake_bridge):
            with self.assertRaisesRegex(Exception, "connect boom"):
                await agent._run_hume_evi_voice_engine(ctx)

        self.assertEqual(calls, ["connect"])

    async def test_bootstrap_failure_is_logged_clearly(self):
        import agent

        calls = []

        async def fake_bridge(room):
            calls.append("bridge")
            raise RuntimeError("HUME_API_KEY")

        ctx = _FakeJobContext(calls)
        with patch.object(agent, "run_hume_evi_bridge", fake_bridge):
            with self.assertLogs(agent.logger, level="ERROR") as captured:
                with self.assertRaises(RuntimeError):
                    await agent._run_hume_evi_voice_engine(ctx)

        self.assertTrue(
            any("voice_engine_bootstrap_failed=true" in line and "engine=hume_evi" in line for line in captured.output),
            captured.output,
        )


class HumeEVIAudioOutputFramingTests(unittest.TestCase):
    """Regression coverage for the assistant-audio clicking noise."""

    def test_parse_wav_strips_header_and_reads_format(self):
        pcm = bytes(range(0, 240)) * 8  # 1920 bytes of deterministic PCM
        wav = _make_wav(pcm, sample_rate=48000, channels=1)

        self.assertGreater(len(wav), len(pcm))  # header present
        decoded, sample_rate, channels = hume_evi_bridge._parse_wav(wav)
        self.assertEqual(decoded, pcm)  # header stripped, no leading garbage
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(channels, 1)

    def test_parse_wav_passes_through_raw_pcm(self):
        raw = b"\x01\x02\x03\x04" * 10
        decoded, sample_rate, channels = hume_evi_bridge._parse_wav(raw)
        self.assertEqual(decoded, raw)
        self.assertEqual(sample_rate, hume_evi_bridge.EVI_OUTPUT_SAMPLE_RATE)
        self.assertEqual(channels, hume_evi_bridge.EVI_CHANNELS)

    def test_take_full_frames_buffers_remainder_without_padding(self):
        frame_bytes = 1920
        buffer = bytearray()

        buffer.extend(b"\xaa" * 1000)
        self.assertEqual(hume_evi_bridge._take_full_frames(buffer, frame_bytes), [])
        self.assertEqual(len(buffer), 1000)  # remainder kept, not zero-padded

        buffer.extend(b"\xbb" * 1000)
        frames = hume_evi_bridge._take_full_frames(buffer, frame_bytes)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0]), frame_bytes)
        # The emitted frame is contiguous source bytes with no interior silence.
        self.assertEqual(frames[0], (b"\xaa" * 1000 + b"\xbb" * 920))
        self.assertEqual(len(buffer), 80)  # leftover carried to the next chunk

    def test_take_full_frames_emits_multiple_frames(self):
        frame_bytes = 1920
        buffer = bytearray(b"\x00" * (frame_bytes * 3 + 5))
        frames = hume_evi_bridge._take_full_frames(buffer, frame_bytes)
        self.assertEqual(len(frames), 3)
        self.assertEqual(len(buffer), 5)


class HumeCLMEndpointHelperTests(unittest.TestCase):
    def test_extract_hume_messages_discards_prosody_metadata(self):
        payload = {
            "messages": [
                {
                    "type": "user_message",
                    "message": {"role": "user", "content": "hello"},
                    "models": {"prosody": {"scores": {"Joy": 0.4}}},
                },
                {"role": "assistant", "content": "hi"},
            ]
        }
        self.assertEqual(
            server._extract_hume_clm_messages(payload),
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        )

    def test_openrouter_model_defaults_to_env_over_hume_hint(self):
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "openrouter/model", "HUME_CLM_HONOR_MODEL_HINT": "false"}):
            self.assertEqual(server._openrouter_model_for_hume({"model": "hume-hint"}), "openrouter/model")

    def test_openrouter_model_can_honor_explicit_hint_flag(self):
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "openrouter/model", "HUME_CLM_HONOR_MODEL_HINT": "true"}):
            self.assertEqual(server._openrouter_model_for_hume({"model": "hume-hint"}), "hume-hint")


if __name__ == "__main__":
    unittest.main()

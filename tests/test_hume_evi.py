import os
import unittest
from unittest.mock import patch

import hume_evi_bridge
import server


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

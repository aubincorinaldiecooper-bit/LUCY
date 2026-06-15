import os
import unittest
from unittest.mock import patch

import agent
import hume_evi_bridge
import server


class VoiceEngineWiringTests(unittest.TestCase):
    """Regression guard for the entrypoint NameError: agent.py must import the
    EVI selection helpers, not just call them."""

    def test_agent_imports_voice_engine_helpers(self):
        for name in ("voice_engine", "run_hume_evi_bridge", "load_hume_evi_settings"):
            self.assertTrue(hasattr(agent, name), f"agent.py is missing import: {name}")

    def test_agent_voice_engine_is_bridge_helper(self):
        self.assertIs(agent.voice_engine, hume_evi_bridge.voice_engine)
        self.assertIs(agent.run_hume_evi_bridge, hume_evi_bridge.run_hume_evi_bridge)

    def test_agent_voice_engine_defaults_current(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(agent.voice_engine(), "current")


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

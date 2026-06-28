import json
import os
import unittest
from unittest import mock

import inworld_voice_profile as ivp


def _profile(emotion=None, pitch=None, vocal_style=None, accent=None):
    p = {}
    if emotion:
        p["emotion"] = [{"label": e, "confidence": c} for e, c in emotion]
    if pitch:
        p["pitch"] = [{"label": l, "confidence": c} for l, c in pitch]
    if vocal_style:
        p["vocalStyle"] = [{"label": l, "confidence": c} for l, c in vocal_style]
    if accent:
        p["accent"] = [{"label": l, "confidence": c} for l, c in accent]
    return p


class NormalizeTests(unittest.TestCase):
    def test_empty_is_neutral(self):
        n = ivp.normalize_voice_profile(None)
        self.assertEqual((n.energy, n.tension, n.certainty), ("medium", "medium", "medium"))
        self.assertEqual(n.emotion_confidence, 0.0)
        self.assertIn("confidence", n.to_dict())
        self.assertNotIn("emotion_confidence", n.to_dict())

    def test_high_confidence_emotion_drives_dims(self):
        n = ivp.normalize_voice_profile(
            _profile(emotion=[("angry", 0.9), ("sad", 0.2)], pitch=[("high", 0.8)]),
        )
        self.assertEqual(n.energy, "high")
        self.assertEqual(n.tension, "high")
        self.assertEqual(n.emotion_confidence, 0.9)
        self.assertEqual(n.pitch, "high")

    def test_low_confidence_emotion_collapses_to_neutral(self):
        n = ivp.normalize_voice_profile(
            _profile(emotion=[("angry", 0.2)]), emotion_confidence_floor=0.5
        )
        self.assertEqual((n.energy, n.tension, n.certainty), ("medium", "medium", "medium"))
        # but the raw confidence is still reported
        self.assertEqual(n.emotion_confidence, 0.2)

    def test_top_label_is_highest_confidence(self):
        n = ivp.normalize_voice_profile(
            _profile(emotion=[("sad", 0.4), ("calm", 0.85)])
        )
        # calm wins -> low energy/tension
        self.assertEqual(n.energy, "low")
        self.assertEqual(n.tension, "low")

    def test_vocal_style_adjusts_certainty(self):
        whisper = ivp.normalize_voice_profile(
            _profile(emotion=[("calm", 0.9)], vocal_style=[("whispering", 0.9)])
        )
        self.assertEqual(whisper.certainty, "low")
        self.assertEqual(whisper.vocal_style, "whispering")

    def test_snake_case_and_nested_result(self):
        msg = {"result": {"voice_profile": _profile(emotion=[("happy", 0.9)])}}
        n = ivp.normalize_from_message(msg)
        self.assertEqual(n.energy, "high")

    def test_planner_summary_never_contains_raw_emotion(self):
        n = ivp.normalize_voice_profile(
            _profile(emotion=[("sad", 0.95)], pitch=[("low", 0.9)], vocal_style=[("crying", 0.8)])
        )
        summary = n.planner_summary()
        self.assertNotIn("sad", summary)
        self.assertIn("energy", summary)
        self.assertIn("pitch low", summary)


class MessageBuilderTests(unittest.TestCase):
    def test_config_message_enables_profiling(self):
        cfg = ivp.InworldConfig(
            enabled=True, ws_url="wss://x", api_key="k", model_id="inworld/inworld-stt-1",
            voice_profile_threshold=0.4, sample_rate=16000, emotion_confidence_floor=0.5,
        )
        msg = json.loads(ivp.build_config_message(cfg))
        tc = msg["transcribe_config"]
        self.assertEqual(tc["modelId"], "inworld/inworld-stt-1")
        self.assertEqual(tc["inworldConfig"]["voiceProfileThreshold"], 0.4)
        self.assertEqual(tc["sampleRateHertz"], 16000)

    def test_audio_chunk_is_base64(self):
        import base64
        msg = json.loads(ivp.build_audio_chunk_message(b"\x01\x02\x03"))
        self.assertEqual(base64.b64decode(msg["audio_chunk"]["content"]), b"\x01\x02\x03")


class ConfigTests(unittest.TestCase):
    def test_is_usable(self):
        base = dict(enabled=True, ws_url="wss://x", api_key="k",
                    model_id="m", voice_profile_threshold=0.5, sample_rate=16000,
                    emotion_confidence_floor=0.5)
        self.assertTrue(ivp.InworldConfig(**base).is_usable()[0])
        self.assertFalse(ivp.InworldConfig(**{**base, "enabled": False}).is_usable()[0])
        self.assertFalse(ivp.InworldConfig(**{**base, "api_key": ""}).is_usable()[0])

    def test_from_env(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "true", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc",
             "INWORLD_VOICE_PROFILE_THRESHOLD": "0.6", "INWORLD_MODEL_ID": "inworld/inworld-stt-1"},
            clear=False,
        ):
            c = ivp.InworldConfig.from_env()
        self.assertTrue(c.enabled)
        self.assertEqual(c.api_key, "abc")
        self.assertEqual(c.voice_profile_threshold, 0.6)
        self.assertEqual(c.model_id, "inworld/inworld-stt-1")
        self.assertTrue(c.ws_url.startswith("wss://api.inworld.ai"))

    def test_auth_scheme_defaults_to_basic(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INWORLD_AUTH_SCHEME", None)
            self.assertEqual(ivp.InworldConfig.from_env().auth_scheme, "Basic")
        with mock.patch.dict(os.environ, {"INWORLD_AUTH_SCHEME": "Bearer"}, clear=False):
            self.assertEqual(ivp.InworldConfig.from_env().auth_scheme, "Bearer")

    def test_model_id_alias(self):
        # Both INWORLD_MODEL_ID and the legacy INWORLD_STT_MODEL_ID are accepted.
        with mock.patch.dict(os.environ, {"INWORLD_STT_MODEL_ID": "legacy/model"}, clear=False):
            os.environ.pop("INWORLD_MODEL_ID", None)
            self.assertEqual(ivp.InworldConfig.from_env().model_id, "legacy/model")
        with mock.patch.dict(
            os.environ, {"INWORLD_MODEL_ID": "new/model", "INWORLD_STT_MODEL_ID": "legacy/model"},
            clear=False,
        ):
            self.assertEqual(ivp.InworldConfig.from_env().model_id, "new/model")


class ShadowTests(unittest.TestCase):
    def test_disabled_without_global_inworld_flag(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "false", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc"},
            clear=False,
        ):
            self.assertIsNone(ivp.build_inworld_shadow_from_env())

    def test_context_for_turn_returns_latest_profile_and_latency(self):
        cfg = ivp.InworldConfig(
            enabled=True, ws_url="wss://x", api_key="k", model_id="inworld/inworld-stt-1",
            voice_profile_threshold=0.4, sample_rate=16000, emotion_confidence_floor=0.5,
        )
        shadow = ivp.InworldVoiceProfileShadow(cfg)
        shadow.latest_profile = ivp.NormalizedVoiceProfile(energy="low", confidence=0.7)
        shadow.latest_received_at = 12.0
        profile, reason, latency = shadow.context_for_turn(10.0)
        self.assertEqual(profile.energy, "low")
        self.assertEqual(reason, "none")
        self.assertEqual(latency, 2.0)


class ShadowTests(unittest.TestCase):
    def test_disabled_without_global_inworld_flag(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "false", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc"},
            clear=False,
        ):
            self.assertIsNone(ivp.build_inworld_shadow_from_env())

    def test_context_for_turn_returns_latest_profile_and_latency(self):
        cfg = ivp.InworldConfig(
            enabled=True, ws_url="wss://x", api_key="k", model_id="inworld/inworld-stt-1",
            voice_profile_threshold=0.4, sample_rate=16000, emotion_confidence_floor=0.5,
        )
        shadow = ivp.InworldVoiceProfileShadow(cfg)
        shadow.latest_profile = ivp.NormalizedVoiceProfile(energy="low", confidence=0.7)
        shadow.latest_received_at = 12.0
        profile, reason, latency = shadow.context_for_turn(10.0)
        self.assertEqual(profile.energy, "low")
        self.assertEqual(reason, "none")
        self.assertEqual(latency, 2.0)


class ShadowTests(unittest.TestCase):
    def test_disabled_without_global_inworld_flag(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "false", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc"},
            clear=False,
        ):
            self.assertIsNone(ivp.build_inworld_shadow_from_env())

    def test_context_for_turn_returns_latest_profile_and_latency(self):
        cfg = ivp.InworldConfig(
            enabled=True, ws_url="wss://x", api_key="k", model_id="inworld/inworld-stt-1",
            voice_profile_threshold=0.4, sample_rate=16000, emotion_confidence_floor=0.5,
        )
        shadow = ivp.InworldVoiceProfileShadow(cfg)
        shadow.latest_profile = ivp.NormalizedVoiceProfile(energy="low", confidence=0.7)
        shadow.latest_received_at = 12.0
        profile, reason, latency = shadow.context_for_turn(10.0)
        self.assertEqual(profile.energy, "low")
        self.assertEqual(reason, "none")
        self.assertEqual(latency, 2.0)


class ShadowTests(unittest.TestCase):
    def test_disabled_without_global_inworld_flag(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "false", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc"},
            clear=False,
        ):
            self.assertIsNone(ivp.build_inworld_shadow_from_env())

    def test_context_for_turn_returns_latest_profile_and_latency(self):
        cfg = ivp.InworldConfig(
            enabled=True, ws_url="wss://x", api_key="k", model_id="inworld/inworld-stt-1",
            voice_profile_threshold=0.4, sample_rate=16000, emotion_confidence_floor=0.5,
        )
        shadow = ivp.InworldVoiceProfileShadow(cfg)
        shadow.latest_profile = ivp.NormalizedVoiceProfile(energy="low", confidence=0.7)
        shadow.latest_received_at = 12.0
        profile, reason, latency = shadow.context_for_turn(10.0)
        self.assertEqual(profile.energy, "low")
        self.assertEqual(reason, "none")
        self.assertEqual(latency, 2.0)


class ShadowTests(unittest.TestCase):
    def test_disabled_without_global_inworld_flag(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "false", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc"},
            clear=False,
        ):
            self.assertIsNone(ivp.build_inworld_shadow_from_env())

    def test_context_for_turn_returns_latest_profile_and_latency(self):
        cfg = ivp.InworldConfig(
            enabled=True, ws_url="wss://x", api_key="k", model_id="inworld/inworld-stt-1",
            voice_profile_threshold=0.4, sample_rate=16000, emotion_confidence_floor=0.5,
        )
        shadow = ivp.InworldVoiceProfileShadow(cfg)
        shadow.latest_profile = ivp.NormalizedVoiceProfile(energy="low", confidence=0.7)
        shadow.latest_received_at = 12.0
        profile, reason, latency = shadow.context_for_turn(10.0)
        self.assertEqual(profile.energy, "low")
        self.assertEqual(reason, "none")
        self.assertEqual(latency, 2.0)


class AuthHeaderTests(unittest.TestCase):
    """The STT websocket must authenticate with the configured scheme. Inworld
    keys are Basic credentials by default; the scheme was previously hardcoded to
    Bearer, which 401'd those keys and dropped the analyzer into fallback."""

    def _cfg(self, **over):
        base = dict(
            enabled=True, ws_url="wss://x", api_key="abc123", model_id="m",
            voice_profile_threshold=0.5, sample_rate=16000, emotion_confidence_floor=0.5,
        )
        base.update(over)
        return ivp.InworldConfig(**base)

    def test_defaults_to_basic(self):
        self.assertEqual(self._cfg().authorization_header(), "Basic abc123")

    def test_bearer_when_configured(self):
        self.assertEqual(self._cfg(auth_scheme="Bearer").authorization_header(), "Bearer abc123")

    def test_scheme_is_canonicalized_case_insensitively(self):
        self.assertEqual(self._cfg(auth_scheme="basic").authorization_header(), "Basic abc123")
        self.assertEqual(self._cfg(auth_scheme="bearer").authorization_header(), "Bearer abc123")

    def test_blank_scheme_falls_back_to_basic(self):
        self.assertEqual(self._cfg(auth_scheme="  ").authorization_header(), "Basic abc123")

    def test_unknown_scheme_passed_through(self):
        self.assertEqual(self._cfg(auth_scheme="Custom").authorization_header(), "Custom abc123")

    def test_aiohttp_factory_uses_configured_scheme_not_hardcoded_bearer(self):
        # Regression for the auth bug: the real _aiohttp_ws_factory must send the
        # configured scheme (Basic by default), not a hardcoded Bearer. Inject a
        # fake aiohttp module (the factory does `import aiohttp` internally) and
        # capture the Authorization header actually passed to ws_connect.
        import asyncio
        import sys
        import types

        captured = {}

        class FakeWS:
            async def close(self):
                pass

        class FakeClientSession:
            def __init__(self, *a, **k):
                pass

            async def ws_connect(self, url, headers=None, heartbeat=None):
                captured["headers"] = headers
                return FakeWS()

            async def close(self):
                pass

        fake_aiohttp = types.SimpleNamespace(
            ClientSession=FakeClientSession,
            ClientTimeout=lambda **k: None,
        )
        shadow = ivp.InworldVoiceProfileShadow(self._cfg(auth_scheme="Basic"))
        with mock.patch.dict(sys.modules, {"aiohttp": fake_aiohttp}):
            asyncio.run(shadow._aiohttp_ws_factory())
        self.assertEqual(captured["headers"]["Authorization"], "Basic abc123")


class EmotionAnalyzerStatusTests(unittest.TestCase):
    """emotion_analyzer_status() is the single source of truth for the startup log."""

    def test_active_when_fully_configured(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "true", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc"},
            clear=False,
        ):
            status = ivp.emotion_analyzer_status()
        self.assertTrue(status["active"])
        self.assertEqual(status["reason"], "active")
        self.assertTrue(status["api_key_present"])

    def test_disabled_when_flags_off(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "false", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc"},
            clear=False,
        ):
            status = ivp.emotion_analyzer_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["reason"], "inworld_disabled")

    def test_disabled_reason_is_missing_api_key(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "true", "INWORLD_VOICE_PROFILE_ENABLED": "true"},
            clear=False,
        ):
            os.environ.pop("INWORLD_API_KEY", None)
            status = ivp.emotion_analyzer_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["reason"], "inworld_api_key_missing")
        self.assertFalse(status["api_key_present"])


class ShadowTests(unittest.TestCase):
    def test_disabled_without_global_inworld_flag(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_ENABLED": "false", "INWORLD_VOICE_PROFILE_ENABLED": "true", "INWORLD_API_KEY": "abc"},
            clear=False,
        ):
            self.assertIsNone(ivp.build_inworld_shadow_from_env())

    def test_context_for_turn_returns_latest_profile_and_latency(self):
        cfg = ivp.InworldConfig(
            enabled=True, ws_url="wss://x", api_key="k", model_id="inworld/inworld-stt-1",
            voice_profile_threshold=0.4, sample_rate=16000, emotion_confidence_floor=0.5,
        )
        shadow = ivp.InworldVoiceProfileShadow(cfg)
        shadow.latest_profile = ivp.NormalizedVoiceProfile(energy="low", confidence=0.7)
        shadow.latest_received_at = 12.0
        profile, reason, latency = shadow.context_for_turn(10.0)
        self.assertEqual(profile.energy, "low")
        self.assertEqual(reason, "none")
        self.assertEqual(latency, 2.0)


if __name__ == "__main__":
    unittest.main()

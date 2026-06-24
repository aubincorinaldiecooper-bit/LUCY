import os
import unittest
from unittest import mock

import aiohttp
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, tts

import agent
import omnivoice_tts as ov


# --- fakes for the aiohttp session used by synthesize_via_sidecar ---
class _FakeResp:
    def __init__(self, status=200, data=b"", text_body=""):
        self.status = status
        self._data = data
        self._text = text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def read(self):
        return self._data

    async def text(self):
        return self._text


class _RaisingCtx:
    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, *, resp=None, raise_exc=None):
        self._resp = resp
        self._raise = raise_exc
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._raise is not None:
            return _RaisingCtx(self._raise)
        return self._resp


def _cfg(**over):
    base = dict(
        enabled=True,
        base_url="http://omnivoice.local",
        api_key="",
        device="cpu",
        model_path="",
        default_language="en",
        expressive_tags_enabled=True,
        timeout_seconds=5.0,
        sample_rate=24000,
        audio_format="wav",
        min_audio_bytes=2048,
    )
    base.update(over)
    return ov.OmniVoiceConfig(**base)


class OmniVoiceConfigTests(unittest.TestCase):
    def test_from_env_parses_fields(self):
        env = {
            "OMNIVOICE_ENABLED": "true",
            "OMNIVOICE_URL": "https://tts.example.com/",  # trailing slash trimmed
            "OMNIVOICE_DEVICE": "CUDA",
            "OMNIVOICE_DEFAULT_LANGUAGE": "fr",
            "OMNIVOICE_EXPRESSIVE_TAGS_ENABLED": "false",
            "OMNIVOICE_TIMEOUT_SECONDS": "7",
            "OMNIVOICE_AUDIO_FORMAT": "pcm_s16le",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            c = ov.OmniVoiceConfig.from_env()
        self.assertTrue(c.enabled)
        self.assertEqual(c.base_url, "https://tts.example.com")
        self.assertEqual(c.device, "cuda")
        self.assertEqual(c.default_language, "fr")
        self.assertFalse(c.expressive_tags_enabled)
        self.assertEqual(c.timeout_seconds, 7.0)
        self.assertEqual(c.audio_format, "pcm_s16le")

    def test_is_usable(self):
        self.assertEqual(_cfg(enabled=False).is_usable()[0], False)
        self.assertEqual(_cfg(enabled=True, base_url="").is_usable()[0], False)
        self.assertEqual(_cfg(enabled=True, base_url="http://x").is_usable()[0], True)

    def test_mime_for_format(self):
        self.assertEqual(ov.mime_for_format("pcm_s16le"), "audio/pcm")
        self.assertEqual(ov.mime_for_format("wav"), "audio/wav")
        self.assertEqual(ov.mime_for_format("anything_else"), "audio/wav")

    def test_build_payload_defaults_language_and_voice(self):
        p = ov.build_synthesis_payload("hi", config=_cfg(), voice=None, language=None)
        self.assertEqual(p["text"], "hi")
        self.assertIsNone(p["voice"])
        self.assertEqual(p["language"], "en")
        self.assertTrue(p["expressive_tags"])

    def test_validate_audio(self):
        with self.assertRaises(ov.OmniVoiceError):
            ov.validate_audio(b"", min_bytes=10)
        with self.assertRaises(ov.OmniVoiceError):
            ov.validate_audio(b"short", min_bytes=10)
        ov.validate_audio(b"x" * 20, min_bytes=10)  # no raise


class SynthesizeViaSidecarTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_returns_audio(self):
        audio = b"R" * 4096
        sess = _FakeSession(resp=_FakeResp(status=200, data=audio))
        out = await ov.synthesize_via_sidecar(
            sess, text="hello", config=_cfg(api_key="k"), voice="warm_1", language="en"
        )
        self.assertEqual(out, audio)
        # request shape
        self.assertEqual(sess.calls[0]["url"], "http://omnivoice.local/synthesize")
        self.assertEqual(sess.calls[0]["headers"]["Authorization"], "Bearer k")
        self.assertEqual(sess.calls[0]["json"]["voice"], "warm_1")

    async def test_non_200_raises_status_error(self):
        sess = _FakeSession(resp=_FakeResp(status=503, text_body="unavailable"))
        with self.assertRaises(APIStatusError):
            await ov.synthesize_via_sidecar(sess, text="x", config=_cfg(), voice=None, language=None)

    async def test_invalid_audio_raises_connection_error(self):
        sess = _FakeSession(resp=_FakeResp(status=200, data=b"tiny"))
        with self.assertRaises(APIConnectionError):
            await ov.synthesize_via_sidecar(
                sess, text="x", config=_cfg(min_audio_bytes=2048), voice=None, language=None
            )

    async def test_timeout_raises_timeout_error(self):
        sess = _FakeSession(raise_exc=TimeoutError())
        with self.assertRaises(APITimeoutError):
            await ov.synthesize_via_sidecar(sess, text="x", config=_cfg(), voice=None, language=None)

    async def test_connection_error_raises_connection_error(self):
        sess = _FakeSession(raise_exc=aiohttp.ClientConnectionError("boom"))
        with self.assertRaises(APIConnectionError):
            await ov.synthesize_via_sidecar(sess, text="x", config=_cfg(), voice=None, language=None)


class BuildTtsProviderSelectionTests(unittest.TestCase):
    def setUp(self):
        self._orig = (agent.TTS_PROVIDER, agent.TTS_FALLBACK_PROVIDER)

    def tearDown(self):
        agent.TTS_PROVIDER, agent.TTS_FALLBACK_PROVIDER = self._orig

    def test_non_omnivoice_returns_bare_primary(self):
        marker = object()
        agent.TTS_PROVIDER, agent.TTS_FALLBACK_PROVIDER = "hume", "hume"
        with mock.patch.object(agent, "_build_single_tts", return_value=marker) as bld:
            out = agent.build_tts()
        self.assertIs(out, marker)
        bld.assert_called_once_with("hume")  # no fallback build for non-omnivoice

    def test_omnivoice_with_fallback_wraps_in_fallback_adapter(self):
        agent.TTS_PROVIDER, agent.TTS_FALLBACK_PROVIDER = "omnivoice", "hume"
        primary = ov.OmniVoiceTTS(config=_cfg())
        fallback = ov.OmniVoiceTTS(config=_cfg())  # stand-in concrete TTS

        def _fake(provider):
            return primary if provider == "omnivoice" else fallback

        with mock.patch.object(agent, "_build_single_tts", side_effect=_fake):
            out = agent.build_tts()
        self.assertIsInstance(out, tts.FallbackAdapter)
        self.assertEqual(out._tts_instances, [primary, fallback])

    def test_omnivoice_fallback_none_returns_bare_primary(self):
        marker = object()
        agent.TTS_PROVIDER, agent.TTS_FALLBACK_PROVIDER = "omnivoice", "none"
        with mock.patch.object(agent, "_build_single_tts", return_value=marker) as bld:
            out = agent.build_tts()
        self.assertIs(out, marker)
        bld.assert_called_once_with("omnivoice")

    def test_primary_build_failure_degrades_to_fallback_only(self):
        fallback_marker = object()
        agent.TTS_PROVIDER, agent.TTS_FALLBACK_PROVIDER = "omnivoice", "hume"

        def _fake(provider):
            if provider == "omnivoice":
                raise RuntimeError("omnivoice_url_missing")
            return fallback_marker

        with mock.patch.object(agent, "_build_single_tts", side_effect=_fake):
            out = agent.build_tts()
        self.assertIs(out, fallback_marker)


if __name__ == "__main__":
    unittest.main()

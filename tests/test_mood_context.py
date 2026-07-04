"""mood_context.py: fuse transcript + vocal dials into a human-style mood read.

Pins the product rules:
  - off unless MOOD_CONTEXT_ENABLED=true (cost + privacy safe default),
  - the mood model is its own env var, independent of other model knobs,
  - describe_mood() never raises — bad response, error status, network
    exception, missing key, empty transcript all degrade to None,
  - "neutral" from the model is treated as "no read" (None), not injected,
  - the instructions note never lets Arche say "you sound..." back.
"""

import asyncio
import types
import unittest
from unittest import mock

import mood_context as mc


def _config(**overrides):
    kwargs = dict(
        enabled=True,
        model="google/gemini-2.5-flash-lite",
        api_key="or-key",
        min_interval_seconds=8.0,
        request_timeout_seconds=5.0,
        max_transcript_chars=600,
    )
    kwargs.update(overrides)
    return mc.MoodContextConfig(**kwargs)


class LoadConfigTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(mc.load_mood_context_config().enabled)

    def test_model_independent_of_other_model_env_vars(self):
        env = {
            "MOOD_CONTEXT_ENABLED": "true",
            "MOOD_MODEL": "anthropic/claude-3-haiku",
            "OPENROUTER_API_KEY": "k",
            "INWORLD_REALTIME_MODEL": "openai/gpt-4o",
            "VISION_MODEL": "google/gemini-2.5-flash-lite",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            cfg = mc.load_mood_context_config()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.model, "anthropic/claude-3-haiku")


class BuildMessageTests(unittest.TestCase):
    def test_includes_words_and_vocal_summary(self):
        msg = mc.build_mood_user_message("I'm fine, really", "energy low, tension high")
        self.assertIn("I'm fine, really", msg)
        self.assertIn("energy low, tension high", msg)

    def test_handles_missing_vocal_summary(self):
        msg = mc.build_mood_user_message("hey", None)
        self.assertIn("hey", msg)
        self.assertIn("none available", msg)


class _FakeResponse:
    def __init__(self, status, json_body=None, text_body=""):
        self.status = status
        self._json_body = json_body
        self._text_body = text_body

    async def json(self):
        return self._json_body

    async def text(self):
        return self._text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response=None, raise_on_post=None):
        self._response = response
        self._raise_on_post = raise_on_post
        self.last_request = None

    def post(self, url, headers=None, json=None):
        if self._raise_on_post is not None:
            raise self._raise_on_post
        self.last_request = {"url": url, "headers": headers, "json": json}
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_aiohttp(fake_session):
    fake_aiohttp = types.SimpleNamespace(
        ClientSession=lambda timeout=None: fake_session,
        ClientTimeout=lambda **k: None,
    )
    return mock.patch.object(mc, "aiohttp", fake_aiohttp)


def _reply(content):
    return _FakeResponse(200, json_body={"choices": [{"message": {"content": content}}]})


class DescribeMoodTests(unittest.TestCase):
    def test_disabled_returns_none_without_call(self):
        result = asyncio.run(mc.describe_mood("hi", "energy low", _config(enabled=False)))
        self.assertIsNone(result)

    def test_empty_transcript_returns_none(self):
        result = asyncio.run(mc.describe_mood("   ", "energy low", _config()))
        self.assertIsNone(result)

    def test_missing_api_key_returns_none(self):
        result = asyncio.run(mc.describe_mood("hi", "energy low", _config(api_key="")))
        self.assertIsNone(result)

    def test_successful_read_returned(self):
        session = _FakeSession(response=_reply("Says she's fine but sounds like she's holding back."))
        with _patch_aiohttp(session):
            result = asyncio.run(mc.describe_mood("I'm fine", "energy low, tension high", _config()))
        self.assertEqual(result, "Says she's fine but sounds like she's holding back.")
        # model + both signals reached the request
        self.assertEqual(session.last_request["json"]["model"], "google/gemini-2.5-flash-lite")
        user_msg = session.last_request["json"]["messages"][1]["content"]
        self.assertIn("I'm fine", user_msg)
        self.assertIn("energy low, tension high", user_msg)

    def test_neutral_reply_treated_as_no_read(self):
        for neutral in ("neutral", "Neutral.", "  NEUTRAL  "):
            session = _FakeSession(response=_reply(neutral))
            with _patch_aiohttp(session):
                result = asyncio.run(mc.describe_mood("hi", "energy medium", _config()))
            self.assertIsNone(result, neutral)

    def test_error_status_returns_none(self):
        session = _FakeSession(response=_FakeResponse(500, text_body="boom"))
        with _patch_aiohttp(session):
            result = asyncio.run(mc.describe_mood("hi", "energy low", _config()))
        self.assertIsNone(result)

    def test_malformed_body_returns_none(self):
        session = _FakeSession(response=_FakeResponse(200, json_body={"unexpected": 1}))
        with _patch_aiohttp(session):
            result = asyncio.run(mc.describe_mood("hi", "energy low", _config()))
        self.assertIsNone(result)

    def test_network_exception_returns_none_never_raises(self):
        session = _FakeSession(raise_on_post=ConnectionError("down"))
        with _patch_aiohttp(session):
            result = asyncio.run(mc.describe_mood("hi", "energy low", _config()))
        self.assertIsNone(result)

    def test_transcript_is_truncated_to_max(self):
        session = _FakeSession(response=_reply("A read."))
        long_transcript = "x" * 5000
        with _patch_aiohttp(session):
            asyncio.run(mc.describe_mood(long_transcript, None, _config(max_transcript_chars=100)))
        sent = str(session.last_request["json"]["messages"])
        # 100 x's present, but not the full 5000
        self.assertIn("x" * 100, sent)
        self.assertNotIn("x" * 200, sent)


class BuildNoteTests(unittest.TestCase):
    def test_note_carries_summary_and_forbids_you_sound(self):
        note = mc.build_mood_context_note("winding down, a little tired")
        self.assertIn("winding down, a little tired", note)
        self.assertIn("you sound", note.lower())  # it explicitly forbids the phrase
        self.assertIn("never", note.lower())


if __name__ == "__main__":
    unittest.main()

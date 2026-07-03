"""vision_context.py: env config loading, best-effort OpenRouter vision calls.

Pins the product rules:
  - disabled unless VISION_CONTEXT_ENABLED=true (cost-safe default),
  - the vision model is a standalone env var, independent of any other model
    knob in this codebase (agnostic/swappable for cost),
  - describe_frame() never raises — a bad response, error status, or network
    exception all degrade to None,
  - the instructions note never mentions a camera/image to the model output.
"""

import asyncio
import types
import unittest
from unittest import mock

import vision_context as vc


def _config(**overrides):
    kwargs = dict(
        enabled=True,
        model="openai/gpt-4o-mini",
        api_key="or-key",
        min_interval_seconds=12.0,
        frame_sample_interval_seconds=2.0,
        max_frame_dimension=512,
        request_timeout_seconds=6.0,
    )
    kwargs.update(overrides)
    return vc.VisionContextConfig(**kwargs)


class LoadConfigTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            cfg = vc.load_vision_context_config()
        self.assertFalse(cfg.enabled)

    def test_reads_model_independently_of_other_model_env_vars(self):
        env = {
            "VISION_CONTEXT_ENABLED": "true",
            "VISION_MODEL": "google/gemini-2.0-flash-001",
            "OPENROUTER_API_KEY": "k",
            # Other model knobs present in the codebase; must NOT leak in.
            "INWORLD_REALTIME_MODEL": "openai/gpt-4o",
            "OPENROUTER_MODEL": "anthropic/claude-3-haiku",
            "LLM_MODEL": "google/gemini-2.5-flash-lite",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            cfg = vc.load_vision_context_config()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.model, "google/gemini-2.0-flash-001")

    def test_defaults_when_enabled_without_overrides(self):
        env = {"VISION_CONTEXT_ENABLED": "true", "OPENROUTER_API_KEY": "k"}
        with mock.patch.dict("os.environ", env, clear=True):
            cfg = vc.load_vision_context_config()
        self.assertEqual(cfg.model, "openai/gpt-4o-mini")
        self.assertEqual(cfg.min_interval_seconds, 12.0)
        self.assertEqual(cfg.max_frame_dimension, 512)


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
    # vision_context imports aiohttp at module level (not inside the function),
    # so the fake must replace the name bound in that module's namespace —
    # patching sys.modules would be a no-op against the already-bound reference.
    fake_aiohttp = types.SimpleNamespace(
        ClientSession=lambda timeout=None: fake_session,
        ClientTimeout=lambda **k: None,
    )
    return mock.patch.object(vc, "aiohttp", fake_aiohttp)


class DescribeFrameTests(unittest.TestCase):
    def test_disabled_returns_none_without_any_call(self):
        cfg = _config(enabled=False)
        result = asyncio.run(vc.describe_frame("data:image/jpeg;base64,AAAA", cfg))
        self.assertIsNone(result)

    def test_missing_api_key_returns_none(self):
        cfg = _config(api_key="")
        result = asyncio.run(vc.describe_frame("data:image/jpeg;base64,AAAA", cfg))
        self.assertIsNone(result)

    def test_empty_image_returns_none(self):
        cfg = _config()
        result = asyncio.run(vc.describe_frame("", cfg))
        self.assertIsNone(result)

    def test_successful_response_returns_stripped_content(self):
        response = _FakeResponse(
            200, json_body={"choices": [{"message": {"content": "  A red circle on white.  "}}]}
        )
        session = _FakeSession(response=response)
        cfg = _config()
        with _patch_aiohttp(session):
            result = asyncio.run(vc.describe_frame("data:image/jpeg;base64,AAAA", cfg))
        self.assertEqual(result, "A red circle on white.")
        self.assertEqual(session.last_request["json"]["model"], "openai/gpt-4o-mini")
        content_parts = session.last_request["json"]["messages"][0]["content"]
        self.assertEqual(content_parts[1]["image_url"]["url"], "data:image/jpeg;base64,AAAA")

    def test_error_status_returns_none(self):
        response = _FakeResponse(500, text_body="upstream error")
        session = _FakeSession(response=response)
        cfg = _config()
        with _patch_aiohttp(session):
            result = asyncio.run(vc.describe_frame("data:image/jpeg;base64,AAAA", cfg))
        self.assertIsNone(result)

    def test_malformed_response_body_returns_none(self):
        response = _FakeResponse(200, json_body={"unexpected": "shape"})
        session = _FakeSession(response=response)
        cfg = _config()
        with _patch_aiohttp(session):
            result = asyncio.run(vc.describe_frame("data:image/jpeg;base64,AAAA", cfg))
        self.assertIsNone(result)

    def test_network_exception_returns_none_never_raises(self):
        session = _FakeSession(raise_on_post=ConnectionError("boom"))
        cfg = _config()
        with _patch_aiohttp(session):
            result = asyncio.run(vc.describe_frame("data:image/jpeg;base64,AAAA", cfg))
        self.assertIsNone(result)


class BuildVisionContextNoteTests(unittest.TestCase):
    def test_note_contains_summary_and_never_mention_guard(self):
        note = vc.build_vision_context_note("a red circle on a white background")
        self.assertIn("a red circle on a white background", note)
        self.assertIn("never mention", note.lower())
        self.assertIn("camera", note.lower())


if __name__ == "__main__":
    unittest.main()

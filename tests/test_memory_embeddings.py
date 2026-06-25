import os
import types
import unittest
from unittest import mock

import memory_embeddings as me


class _FakeEmbeddings:
    def __init__(self, vec, raise_exc=None):
        self._vec = vec
        self._raise = raise_exc
        self.calls = []

    async def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        if self._raise:
            raise self._raise
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=self._vec)])


class _FakeClient:
    def __init__(self, vec, raise_exc=None):
        self.embeddings = _FakeEmbeddings(vec, raise_exc)


class ToPgvectorLiteralTests(unittest.TestCase):
    def test_format(self):
        self.assertEqual(me.to_pgvector_literal([0.1, 0.2, -0.3]), "[0.1,0.2,-0.3]")

    def test_handles_ints_and_floats(self):
        self.assertEqual(me.to_pgvector_literal([1, 2, 3]), "[1,2,3]")


class FlagTests(unittest.TestCase):
    def test_semantic_enabled_toggle(self):
        with mock.patch.dict(os.environ, {"MEMORY_SEMANTIC_RETRIEVAL_ENABLED": "true"}, clear=False):
            self.assertTrue(me.semantic_retrieval_enabled())
        with mock.patch.dict(os.environ, {"MEMORY_SEMANTIC_RETRIEVAL_ENABLED": "false"}, clear=False):
            self.assertFalse(me.semantic_retrieval_enabled())

    def test_semantic_timeout_default_and_floor(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_SEMANTIC_TIMEOUT_MS", None)
            self.assertEqual(me.semantic_timeout_ms(), 800)
        with mock.patch.dict(os.environ, {"MEMORY_SEMANTIC_TIMEOUT_MS": "10"}, clear=False):
            self.assertEqual(me.semantic_timeout_ms(), 100)  # floored


class EmbedTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_vector_with_client(self):
        vec = [0.1, 0.2, 0.3]
        out = await me.embed_text("hello", client=_FakeClient(vec))
        self.assertEqual(out, vec)

    async def test_empty_text_returns_none(self):
        self.assertIsNone(await me.embed_text("   ", client=_FakeClient([1.0])))

    async def test_no_key_and_no_client_returns_none(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            self.assertIsNone(await me.embed_text("hello"))

    async def test_api_error_returns_none(self):
        out = await me.embed_text("hello", client=_FakeClient(None, raise_exc=RuntimeError("boom")))
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()

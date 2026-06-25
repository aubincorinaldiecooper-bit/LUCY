import os
import types
import unittest
from unittest import mock

import memory_embeddings as me


# --- OpenAI-style fake (resp.data[0].embedding) ---
class _FakeOpenAIEmbeddings:
    def __init__(self, vec, raise_exc=None):
        self._vec, self._raise, self.calls = vec, raise_exc, []

    async def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        if self._raise:
            raise self._raise
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=self._vec)])


class _FakeOpenAIClient:
    def __init__(self, vec, raise_exc=None):
        self.embeddings = _FakeOpenAIEmbeddings(vec, raise_exc)


# --- Cohere-style fake (resp.embeddings.float_[0]) ---
class _FakeCohereClient:
    def __init__(self, vec, raise_exc=None):
        self._vec, self._raise, self.calls = vec, raise_exc, []

    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise:
            raise self._raise
        return types.SimpleNamespace(embeddings=types.SimpleNamespace(float_=[self._vec]))


class HelperTests(unittest.TestCase):
    def test_to_pgvector_literal(self):
        self.assertEqual(me.to_pgvector_literal([0.1, 0.2, -0.3]), "[0.1,0.2,-0.3]")

    def test_provider_defaults_to_cohere(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_EMBEDDING_PROVIDER", None)
            self.assertEqual(me.embedding_provider(), "cohere")
            self.assertEqual(me.embedding_model(), "embed-v4.0")

    def test_provider_openai(self):
        with mock.patch.dict(os.environ, {"MEMORY_EMBEDDING_PROVIDER": "openai"}, clear=False):
            os.environ.pop("MEMORY_EMBEDDING_MODEL", None)
            self.assertEqual(me.embedding_provider(), "openai")
            self.assertEqual(me.embedding_model(), "text-embedding-3-small")

    def test_unknown_provider_falls_back_to_cohere(self):
        with mock.patch.dict(os.environ, {"MEMORY_EMBEDDING_PROVIDER": "bogus"}, clear=False):
            self.assertEqual(me.embedding_provider(), "cohere")


class EmbedOpenAITests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_vector(self):
        with mock.patch.dict(os.environ, {"MEMORY_EMBEDDING_PROVIDER": "openai"}, clear=False):
            out = await me.embed_text("hello", client=_FakeOpenAIClient([0.1, 0.2]))
        self.assertEqual(out, [0.1, 0.2])

    async def test_error_returns_none(self):
        with mock.patch.dict(os.environ, {"MEMORY_EMBEDDING_PROVIDER": "openai"}, clear=False):
            out = await me.embed_text("hi", client=_FakeOpenAIClient(None, raise_exc=RuntimeError("x")))
        self.assertIsNone(out)


class EmbedCohereTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_vector_and_sets_v4_params(self):
        client = _FakeCohereClient([0.3, 0.4])
        with mock.patch.dict(os.environ, {"MEMORY_EMBEDDING_PROVIDER": "cohere"}, clear=False):
            os.environ.pop("MEMORY_EMBEDDING_MODEL", None)
            out = await me.embed_text("hello", input_type="search_query", client=client)
        self.assertEqual(out, [0.3, 0.4])
        kw = client.calls[0]
        self.assertEqual(kw["input_type"], "search_query")
        self.assertEqual(kw["model"], "embed-v4.0")
        self.assertEqual(kw["output_dimension"], me.EMBED_DIM)  # v4 gets output_dimension

    async def test_v3_model_omits_output_dimension(self):
        client = _FakeCohereClient([0.5])
        env = {"MEMORY_EMBEDDING_PROVIDER": "cohere", "MEMORY_EMBEDDING_MODEL": "embed-multilingual-v3.0"}
        with mock.patch.dict(os.environ, env, clear=False):
            await me.embed_text("hi", client=client)
        self.assertNotIn("output_dimension", client.calls[0])

    async def test_error_returns_none(self):
        client = _FakeCohereClient(None, raise_exc=RuntimeError("boom"))
        with mock.patch.dict(os.environ, {"MEMORY_EMBEDDING_PROVIDER": "cohere"}, clear=False):
            self.assertIsNone(await me.embed_text("hi", client=client))


class EmbedCommonTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_text_returns_none(self):
        self.assertIsNone(await me.embed_text("   ", client=_FakeCohereClient([1.0])))

    async def test_no_key_no_client_returns_none(self):
        env = {"MEMORY_EMBEDDING_PROVIDER": "cohere", "COHERE_API_KEY": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertIsNone(await me.embed_text("hello"))


if __name__ == "__main__":
    unittest.main()

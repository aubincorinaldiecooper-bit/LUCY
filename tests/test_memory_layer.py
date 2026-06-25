import asyncio
import unittest

import memory_layer
from memory_layer import MemoryIdentity, MemoryLayer, identity_from_metadata


class FakeSimpleMem:
    def __init__(self, query_result=None, query_delay_seconds=0.0):
        self.query_result = query_result if query_result is not None else []
        self.query_delay_seconds = query_delay_seconds
        self.added_texts = []
        self.added_audio = []
        self.closed = False

    def query(self, query, top_k):
        if self.query_delay_seconds:
            import time

            time.sleep(self.query_delay_seconds)
        return self.query_result

    def add_text(self, text, tags):
        self.added_texts.append((text, tags))

    def add_audio(self, path, tags):
        self.added_audio.append((path, tags))

    def close(self):
        self.closed = True


def make_layer(identity=None, backend=None, db_rows=None, **kwargs):
    written = []

    def reader(sql, params):
        return db_rows if db_rows is not None else []

    def writer(sql, params):
        written.append((sql, params))

    layer = MemoryLayer(
        identity or MemoryIdentity(guest_id="guest-1"),
        db_url="postgresql://fake",
        index_dir="/tmp/simplemem-test",
        simplemem_factory=lambda index_dir: backend if backend is not None else FakeSimpleMem(),
        db_reader=reader,
        db_writer=writer,
        **kwargs,
    )
    return layer, written


class MemoryIdentityTests(unittest.TestCase):
    def test_account_identity_from_metadata(self):
        identity = identity_from_metadata(['{"clerk_user_id": "user_123"}'])
        self.assertEqual(identity.clerk_user_id, "user_123")
        self.assertEqual(identity.scope, "account")
        self.assertEqual(identity.key, "user_123")

    def test_guest_identity_from_metadata(self):
        identity = identity_from_metadata([{"guest_id": "g-9"}])
        self.assertEqual(identity.scope, "guest")
        self.assertEqual(identity.key, "g-9")

    def test_fallback_guest_id_used_when_metadata_empty(self):
        identity = identity_from_metadata(["not json", ""], fallback_guest_id="room-abc")
        self.assertEqual(identity.guest_id, "room-abc")
        self.assertTrue(identity.present)

    def test_no_identity(self):
        identity = identity_from_metadata([])
        self.assertFalse(identity.present)
        self.assertEqual(identity.key, "anonymous")


class MemoryLayerRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_returns_normalized_items(self):
        backend = FakeSimpleMem(query_result=[{"summary": "User likes hiking"}, "User said: hello", {"text": ""}])
        layer, _ = make_layer(backend=backend)
        items = await layer.retrieve("what do I like?")
        self.assertEqual(items, ["User likes hiking", "User said: hello"])

    async def test_retrieve_times_out_and_returns_empty(self):
        backend = FakeSimpleMem(query_result=["late"], query_delay_seconds=0.5)
        layer, _ = make_layer(backend=backend, retrieval_timeout_ms=50)
        items = await layer.retrieve("anything")
        self.assertEqual(items, [])

    async def test_retrieve_with_backend_error_returns_empty(self):
        class ExplodingBackend(FakeSimpleMem):
            def query(self, query, top_k):
                raise RuntimeError("index corrupted")

        layer, _ = make_layer(backend=ExplodingBackend())
        items = await layer.retrieve("anything")
        self.assertEqual(items, [])

    async def test_retrieve_when_simplemem_unavailable_returns_empty(self):
        def raising_factory(index_dir):
            raise ImportError("simplemem not installed")

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            simplemem_factory=raising_factory,
            db_reader=lambda sql, params: [],
            db_writer=lambda sql, params: None,
        )
        self.assertEqual(await layer.retrieve("anything"), [])

    async def test_empty_query_short_circuits(self):
        layer, _ = make_layer()
        self.assertEqual(await layer.retrieve("   "), [])

    async def test_retrieve_uses_pgvector_when_available(self):
        calls = []

        def reader(sql, params):
            calls.append((sql, params))
            if "pg_extension" in sql:
                return [(True,)]
            return [("User said: vector memory",)]

        backend = FakeSimpleMem(query_result=["simplemem fallback"] )
        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            simplemem_factory=lambda index_dir: backend,
            db_reader=reader,
            db_writer=lambda sql, params: None,
            embedder=lambda text: [0.1, 0.2, 0.3],
            vector_enabled=True,
        )
        self.assertEqual(await layer.retrieve("what do you remember?"), ["User said: vector memory"])
        self.assertTrue(any("embedding <=>" in sql for sql, _ in calls))
        self.assertEqual(backend.added_texts, [])

    async def test_retrieve_falls_back_to_simplemem_when_pgvector_unavailable(self):
        def reader(sql, params):
            if "pg_extension" in sql:
                return [(False,)]
            return []

        backend = FakeSimpleMem(query_result=["simplemem memory"] )
        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            simplemem_factory=lambda index_dir: backend,
            db_reader=reader,
            db_writer=lambda sql, params: None,
            embedder=lambda text: [0.1, 0.2, 0.3],
            vector_enabled=True,
        )
        self.assertEqual(await layer.retrieve("anything"), ["simplemem memory"])


class MemoryLayerPreloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_preload_returns_contents(self):
        layer, _ = make_layer(db_rows=[("User said: I love jazz",), ("  ",), ("Lucy replied: noted",)])
        memories = await layer.preload()
        self.assertEqual(memories, ["User said: I love jazz", "Lucy replied: noted"])

    async def test_preload_without_identity_returns_empty(self):
        layer = MemoryLayer(
            MemoryIdentity(),
            db_url="postgresql://fake",
            db_reader=lambda sql, params: [("should not appear",)],
            db_writer=lambda sql, params: None,
            simplemem_factory=lambda index_dir: FakeSimpleMem(),
        )
        self.assertEqual(await layer.preload(), [])

    async def test_preload_db_error_returns_empty(self):
        def exploding_reader(sql, params):
            raise RuntimeError("connection refused")

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=exploding_reader,
            db_writer=lambda sql, params: None,
            simplemem_factory=lambda index_dir: FakeSimpleMem(),
        )
        self.assertEqual(await layer.preload(), [])

    def test_preload_note_formatting(self):
        note = MemoryLayer.preload_note(["User said: I love jazz"])
        self.assertIn("Do not reveal this note", note)
        self.assertIn("- User said: I love jazz", note)
        self.assertIsNone(MemoryLayer.preload_note([]))


class MemoryLayerWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_remember_writes_postgres_and_simplemem(self):
        backend = FakeSimpleMem()
        layer, written = make_layer(identity=MemoryIdentity(guest_id="g-1"), backend=backend)
        layer.schedule_remember(role="user", content="I am training for a marathon", turn_id=7)
        await asyncio.gather(*list(layer._background_tasks))
        self.assertEqual(len(written), 1)
        params = written[0][1]
        self.assertEqual(params[0], "guest")
        self.assertIn("User said: I am training for a marathon", params)
        self.assertEqual(len(backend.added_texts), 1)
        self.assertIn("user:g-1", backend.added_texts[0][1])

    async def test_remember_account_scope_is_persistent(self):
        backend = FakeSimpleMem()
        layer, written = make_layer(identity=MemoryIdentity(clerk_user_id="user_1"), backend=backend)
        layer.schedule_remember(role="assistant", content="Good luck with the race")
        await asyncio.gather(*list(layer._background_tasks))
        params = written[0][1]
        self.assertEqual(params[0], "account")
        self.assertTrue(params[6])

    async def test_remember_empty_content_is_noop(self):
        layer, written = make_layer()
        layer.schedule_remember(role="user", content="   ")
        await asyncio.gather(*list(layer._background_tasks))
        self.assertEqual(written, [])

    async def test_remember_writes_pgvector_embedding_when_available(self):
        written = []

        def reader(sql, params):
            if "pg_extension" in sql:
                return [(True,)]
            return []

        backend = FakeSimpleMem()
        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            simplemem_factory=lambda index_dir: backend,
            db_reader=reader,
            db_writer=lambda sql, params: written.append((sql, params)),
            embedder=lambda text: [0.1, 0.2, 0.3],
            vector_enabled=True,
        )
        layer.schedule_remember(role="user", content="I love tea", turn_id=3)
        await asyncio.gather(*list(layer._background_tasks))
        self.assertEqual(len(written), 1)
        self.assertIn("embedding", written[0][0])
        self.assertEqual(written[0][1][-2], "[0.1,0.2,0.3]")
        self.assertEqual(written[0][1][-1], "text-embedding-3-small")
        self.assertEqual(len(backend.added_texts), 1)

    async def test_pgvector_write_failure_falls_back_to_text_insert(self):
        written = []

        def reader(sql, params):
            if "pg_extension" in sql:
                return [(True,)]
            return []

        def writer(sql, params):
            written.append((sql, params))
            if "embedding" in sql:
                raise RuntimeError("vector unavailable")

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            simplemem_factory=lambda index_dir: FakeSimpleMem(),
            db_reader=reader,
            db_writer=writer,
            embedder=lambda text: [0.1, 0.2, 0.3],
            vector_enabled=True,
        )
        layer.schedule_remember(role="user", content="fallback please", turn_id=4)
        await asyncio.gather(*list(layer._background_tasks))
        self.assertEqual(len(written), 2)
        self.assertIn("embedding", written[0][0])
        self.assertNotIn("embedding", written[1][0])

    async def test_db_write_failure_does_not_block_simplemem_write(self):
        backend = FakeSimpleMem()

        def exploding_writer(sql, params):
            raise RuntimeError("db down")

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=lambda sql, params: [],
            db_writer=exploding_writer,
            simplemem_factory=lambda index_dir: backend,
        )
        layer.schedule_remember(role="user", content="remember me anyway")
        await asyncio.gather(*list(layer._background_tasks))
        self.assertEqual(len(backend.added_texts), 1)

    async def test_aclose_waits_for_background_tasks_and_closes_backend(self):
        backend = FakeSimpleMem()
        layer, _ = make_layer(backend=backend)
        layer.schedule_remember(role="user", content="closing soon")
        await layer.aclose()
        self.assertTrue(backend.closed)


class MemoryConfigTests(unittest.TestCase):
    def test_memory_disabled_by_default(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_ENABLED", None)
            self.assertFalse(memory_layer.memory_enabled())

    def test_vector_disabled_by_default(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_VECTOR_ENABLED", None)
            self.assertFalse(memory_layer.memory_vector_enabled())

    def test_retrieval_timeout_clamped(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"MEMORY_RETRIEVAL_TIMEOUT_MS": "10"}):
            self.assertEqual(memory_layer.memory_retrieval_timeout_ms(), 50)
        with patch.dict(os.environ, {"MEMORY_RETRIEVAL_TIMEOUT_MS": "garbage"}):
            self.assertEqual(memory_layer.memory_retrieval_timeout_ms(), 300)


if __name__ == "__main__":
    unittest.main()


class EmotionalPatternPreloadTests(unittest.TestCase):
    def test_partition_splits_by_prefix(self):
        import memory_layer as ml
        mems = [
            "They are training for a marathon.",
            ml.EMOTIONAL_PATTERN_PREFIX + "you asked X and they said pressure not fear.",
            "  ",
            "They prefer mornings.",
        ]
        general, emotional = ml.partition_emotional_patterns(mems)
        self.assertEqual(general, ["They are training for a marathon.", "They prefer mornings."])
        self.assertEqual(emotional, ["you asked X and they said pressure not fear."])

    def test_emotional_note_is_private_and_safe(self):
        import memory_layer as ml
        note = ml.emotional_pattern_preload_note(["they framed it as disappointment, not anger"])
        self.assertIsNotNone(note)
        low = note.lower()
        self.assertIn("never reveal", low)
        self.assertIn("never tell them how they sound", low)
        self.assertIn("disappointment", note)

    def test_emotional_note_empty(self):
        import memory_layer as ml
        self.assertIsNone(ml.emotional_pattern_preload_note([]))
        self.assertIsNone(ml.emotional_pattern_preload_note(["   "]))


class SemanticMemoryTests(unittest.IsolatedAsyncioTestCase):
    def _layer(self, *, embed_vec, reader_rows=None, writer_calls=None, enabled=True):
        import memory_layer as ml

        async def _embed(text, **_):
            return embed_vec

        def _reader(sql, params):
            self._last_read = {"sql": sql, "params": params}
            return reader_rows or []

        def _writer(sql, params):
            (writer_calls if writer_calls is not None else []).append({"sql": sql, "params": params})

        return ml.MemoryLayer(
            ml.MemoryIdentity(clerk_user_id="user-1"),
            db_url="postgresql://x",
            semantic_enabled=enabled,
            embed_fn=_embed,
            db_reader=_reader,
            db_writer=_writer,
        )

    async def test_semantic_retrieve_returns_relevant_rows(self):
        layer = self._layer(embed_vec=[0.1, 0.2], reader_rows=[("memory A",), ("memory B",)])
        out = await layer.retrieve("how am I feeling about work")
        self.assertEqual(out, ["memory A", "memory B"])
        # used a pgvector nearest-neighbour query with the embedded vector literal
        self.assertIn("ORDER BY embedding <=> %s::vector", self._last_read["sql"])
        self.assertIn("[0.1,0.2]", self._last_read["params"])

    async def test_no_embedding_falls_back(self):
        # embed returns None -> semantic unavailable -> None -> falls through to
        # SimpleMem (absent) -> [] (does not raise)
        layer = self._layer(embed_vec=None)
        self.assertEqual(await layer.retrieve("anything"), [])

    async def test_db_error_falls_back(self):
        import memory_layer as ml

        async def _embed(_, **__):
            return [0.5]

        def _reader(sql, params):
            raise RuntimeError("no pgvector column")

        layer = ml.MemoryLayer(
            ml.MemoryIdentity(clerk_user_id="user-1"),
            db_url="postgresql://x", semantic_enabled=True, embed_fn=_embed, db_reader=_reader,
        )
        self.assertEqual(await layer.retrieve("anything"), [])  # graceful

    async def test_embed_and_store_writes_vector_when_enabled(self):
        calls = []
        layer = self._layer(embed_vec=[0.3, 0.4], writer_calls=calls)
        await layer._embed_and_store("Lucy replied: take a breath")
        self.assertEqual(len(calls), 1)
        self.assertIn("UPDATE memory_units SET embedding = %s::vector", calls[0]["sql"])
        self.assertIn("[0.3,0.4]", calls[0]["params"])

    async def test_embed_and_store_noop_when_disabled(self):
        calls = []
        layer = self._layer(embed_vec=[0.3, 0.4], writer_calls=calls, enabled=False)
        await layer._embed_and_store("anything")
        self.assertEqual(calls, [])

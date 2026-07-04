import asyncio
import unittest

import memory_layer
from memory_layer import MemoryIdentity, MemoryLayer, identity_from_metadata


def make_layer(identity=None, db_rows=None, **kwargs):
    """Build a MemoryLayer with injected Postgres reader/writer (no real DB)."""
    written = []

    def reader(sql, params):
        return db_rows if db_rows is not None else []

    def writer(sql, params):
        written.append((sql, params))

    layer = MemoryLayer(
        identity or MemoryIdentity(guest_id="guest-1"),
        db_url="postgresql://fake",
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


class MemoryLayerRetrieveTests(unittest.IsolatedAsyncioTestCase):
    """retrieve() is pgvector-only: returns recalled rows when a pgvector-enabled
    Postgres is present, else [] (never blocks the turn)."""

    def _pgvector_reader(self, content_rows):
        def reader(sql, params):
            if "pg_extension" in sql:
                return [(True,)]
            return content_rows
        return reader

    async def test_retrieve_returns_pgvector_rows(self):
        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=self._pgvector_reader([("User said: I like hiking",), ("  ",)]),
            db_writer=lambda sql, params: None,
            embedder=lambda text: [0.1, 0.2, 0.3],
            vector_enabled=True,
        )
        self.assertEqual(await layer.retrieve("what do I like?"), ["User said: I like hiking"])

    async def test_retrieve_times_out_and_returns_empty(self):
        def slow_embed(text):
            import time
            time.sleep(0.5)
            return [0.1, 0.2, 0.3]

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=self._pgvector_reader([("late",)]),
            db_writer=lambda sql, params: None,
            embedder=slow_embed,
            vector_enabled=True,
            semantic_timeout_ms_override=50,
        )
        self.assertEqual(await layer.retrieve("anything"), [])

    async def test_retrieve_with_reader_error_returns_empty(self):
        def exploding_reader(sql, params):
            raise RuntimeError("db down")

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=exploding_reader,
            db_writer=lambda sql, params: None,
            embedder=lambda text: [0.1, 0.2, 0.3],
            vector_enabled=True,
        )
        self.assertEqual(await layer.retrieve("anything"), [])

    async def test_retrieve_returns_empty_when_pgvector_unavailable(self):
        def reader(sql, params):
            if "pg_extension" in sql:
                return [(False,)]
            return [("should not be used",)]

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=reader,
            db_writer=lambda sql, params: None,
            embedder=lambda text: [0.1, 0.2, 0.3],
            vector_enabled=True,
        )
        self.assertEqual(await layer.retrieve("anything"), [])

    async def test_retrieve_returns_empty_when_semantic_disabled(self):
        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=lambda sql, params: [("x",)],
            db_writer=lambda sql, params: None,
        )
        self.assertFalse(layer._semantic_enabled)
        self.assertEqual(await layer.retrieve("anything"), [])

    async def test_empty_query_short_circuits(self):
        layer, _ = make_layer()
        self.assertEqual(await layer.retrieve("   "), [])

    async def test_constructor_accepts_semantic_alias_args(self):
        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=lambda sql, params: [(False,)] if "pg_extension" in sql else [],
            db_writer=lambda sql, params: None,
            semantic_enabled=True,
            embed_fn=lambda text: [0.1, 0.2],
            semantic_timeout_ms_override=75,
        )
        self.assertTrue(layer._semantic_enabled)
        self.assertEqual(layer._semantic_timeout_ms, 75)
        self.assertEqual(layer._embedder("x"), [0.1, 0.2])

    async def test_broken_semantic_config_degrades_without_crashing(self):
        original = memory_layer.semantic_retrieval_enabled
        try:
            memory_layer.semantic_retrieval_enabled = lambda: (_ for _ in ()).throw(RuntimeError("bad config"))
            layer = MemoryLayer(
                MemoryIdentity(guest_id="g"),
                db_url="postgresql://fake",
                db_reader=lambda sql, params: [],
                db_writer=lambda sql, params: None,
            )
        finally:
            memory_layer.semantic_retrieval_enabled = original
        self.assertFalse(layer._semantic_enabled)
        self.assertEqual(await layer.retrieve("anything"), [])

    async def test_retrieve_uses_pgvector_embedding_query(self):
        calls = []

        def reader(sql, params):
            calls.append((sql, params))
            if "pg_extension" in sql:
                return [(True,)]
            return [("User said: vector memory",)]

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=reader,
            db_writer=lambda sql, params: None,
            embedder=lambda text: [0.1, 0.2, 0.3],
            vector_enabled=True,
        )
        self.assertEqual(await layer.retrieve("what do you remember?"), ["User said: vector memory"])
        self.assertTrue(any("embedding <=>" in sql for sql, _ in calls))


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
        )
        self.assertEqual(await layer.preload(), [])

    def test_preload_note_formatting(self):
        note = MemoryLayer.preload_note(["User said: I love jazz"])
        self.assertIn("Do not reveal this note", note)
        self.assertIn("- User said: I love jazz", note)
        self.assertIsNone(MemoryLayer.preload_note([]))


class MemoryLayerWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_remember_writes_postgres(self):
        layer, written = make_layer(identity=MemoryIdentity(guest_id="g-1"))
        layer.schedule_remember(role="user", content="I am training for a marathon", turn_id=7)
        await asyncio.gather(*list(layer._background_tasks))
        self.assertEqual(len(written), 1)
        params = written[0][1]
        self.assertEqual(params[0], "guest")
        self.assertIn("User said: I am training for a marathon", params)

    async def test_remember_account_scope_is_persistent(self):
        layer, written = make_layer(identity=MemoryIdentity(clerk_user_id="user_1"))
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

    async def test_remember_calls_embed_and_store_without_unhandled_exception(self):
        layer, _ = make_layer()
        calls = []

        async def fake_embed_and_store(compact, role, turn_id, modality, media_url):
            calls.append((compact, role, turn_id, modality, media_url))

        layer._embed_and_store = fake_embed_and_store
        await layer._remember("user", "hello", 8, "text", None)
        self.assertEqual(len(calls), 1)
        self.assertIn("User said: hello", calls[0][0])

    async def test_remember_preserves_emotional_pattern_prefix_unwrapped(self):
        """A confirmed-pattern note must round-trip through preload() with
        EMOTIONAL_PATTERN_PREFIX still at the very start (needed for
        partition_emotional_patterns to file it as a learned pattern, not
        general memory) — so it must NOT get the usual 'User said:'/'Lucy
        replied:' wrapping ordinary content gets.
        """
        layer, written = make_layer()
        content = memory_layer.EMOTIONAL_PATTERN_PREFIX + "energy=low; tension=high"
        layer.schedule_remember(role="emotional_calibration", content=content, turn_id=3)
        await asyncio.gather(*list(layer._background_tasks))
        self.assertEqual(len(written), 1)
        params = written[0][1]
        self.assertIn(content, params)
        for p in params:
            if isinstance(p, str):
                self.assertNotIn("Lucy replied:", p)
                self.assertNotIn("User said:", p)

    async def test_remember_still_wraps_ordinary_content(self):
        layer, written = make_layer()
        layer.schedule_remember(role="assistant", content="Good luck!", turn_id=1)
        await asyncio.gather(*list(layer._background_tasks))
        params = written[0][1]
        self.assertIn("Lucy replied: Good luck!", params)

    async def test_remember_observation_role_uses_noticed_phrasing(self):
        """Vision-context scene descriptions are Arche's own noticing, not a
        user or assistant line — the preload note should read naturally."""
        layer, written = make_layer()
        layer.schedule_remember(role="observation", content="a red mug on a desk", turn_id=2)
        await asyncio.gather(*list(layer._background_tasks))
        params = written[0][1]
        self.assertIn("Arche noticed: a red mug on a desk", params)
        for p in params:
            if isinstance(p, str):
                self.assertNotIn("Lucy replied:", p)
                self.assertNotIn("User said:", p)

    async def test_remember_writes_pgvector_embedding_when_available(self):
        written = []

        def reader(sql, params):
            if "pg_extension" in sql:
                return [(True,)]
            return []

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
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

    async def test_db_write_failure_is_swallowed(self):
        def exploding_writer(sql, params):
            raise RuntimeError("db down")

        layer = MemoryLayer(
            MemoryIdentity(guest_id="g"),
            db_url="postgresql://fake",
            db_reader=lambda sql, params: [],
            db_writer=exploding_writer,
        )
        layer.schedule_remember(role="user", content="remember me anyway")
        # A failing write must never surface as an unhandled task exception.
        await asyncio.gather(*list(layer._background_tasks), return_exceptions=True)

    async def test_aclose_waits_for_background_tasks(self):
        layer, _ = make_layer()
        layer.schedule_remember(role="user", content="closing soon")
        await layer.aclose()  # must drain the pending write without raising


class MemoryConfigTests(unittest.TestCase):
    def test_memory_disabled_by_default(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_ENABLED", None)
            self.assertFalse(memory_layer.memory_enabled())

    def test_semantic_alias_uses_vector_flag(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"MEMORY_VECTOR_ENABLED": "true"}):
            self.assertTrue(memory_layer.semantic_retrieval_enabled())

    def test_vector_disabled_by_default(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_VECTOR_ENABLED", None)
            self.assertFalse(memory_layer.memory_vector_enabled())


if __name__ == "__main__":
    unittest.main()

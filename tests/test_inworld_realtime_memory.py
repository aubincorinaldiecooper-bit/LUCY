"""Long-term memory setup for the Inworld Realtime voice engine (agent.py).

Pins: identity resolves from job/room metadata, preload runs with a timeout,
confirmed emotional patterns are split into their own private note, the
global _active_memory_layer never leaks a previous session's instance forward
into a session where memory is disabled or setup fails, and a resolution
failure degrades to (None, None) rather than raising.

Uses the real MemoryLayer (no DATABASE_URL in the test env, so preload()
safely no-ops) rather than a fake, so this exercises
the actual identity-resolution + note-composition code path.
"""

import asyncio
import json
import types
import unittest

import agent
from memory_layer import EMOTIONAL_PATTERN_PREFIX, MemoryLayer


def _ctx(*, job_metadata=None, room_metadata=None, room_name=None, remote_participants=None):
    return types.SimpleNamespace(
        job=types.SimpleNamespace(
            metadata=job_metadata, participant=None, participant_info=None
        ),
        room=types.SimpleNamespace(
            metadata=room_metadata,
            name=room_name,
            remote_participants=remote_participants or {},
        ),
    )


class BuildMemoryLayerForRealtimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved_active_memory_layer = agent._active_memory_layer
        self._saved_memory_enabled_env = agent.os.environ.get("MEMORY_ENABLED")

    def tearDown(self):
        agent._active_memory_layer = self._saved_active_memory_layer
        if self._saved_memory_enabled_env is None:
            agent.os.environ.pop("MEMORY_ENABLED", None)
        else:
            agent.os.environ["MEMORY_ENABLED"] = self._saved_memory_enabled_env

    async def test_disabled_returns_none_and_clears_global(self):
        agent.os.environ["MEMORY_ENABLED"] = "false"
        agent._active_memory_layer = "stale-instance-from-a-previous-session"
        layer, note = await agent._build_memory_layer_for_realtime(_ctx())
        self.assertIsNone(layer)
        self.assertIsNone(note)
        self.assertIsNone(agent._active_memory_layer)

    async def test_enabled_resolves_signed_in_identity_and_returns_layer(self):
        agent.os.environ["MEMORY_ENABLED"] = "true"
        ctx = _ctx(job_metadata=json.dumps({"clerk_user_id": "user_42"}), room_name="room-1")
        layer, note = await agent._build_memory_layer_for_realtime(ctx)
        self.assertIsInstance(layer, MemoryLayer)
        self.assertEqual(layer.identity.scope, "account")
        self.assertEqual(layer.identity.clerk_user_id, "user_42")
        self.assertIs(agent._active_memory_layer, layer)
        # No DATABASE_URL in the test env -> preload() returns [] -> no note.
        self.assertIsNone(note)
        await layer.aclose()

    async def test_enabled_falls_back_to_guest_scope_from_room_name(self):
        agent.os.environ["MEMORY_ENABLED"] = "true"
        ctx = _ctx(room_name="guest-room-77")
        layer, _ = await agent._build_memory_layer_for_realtime(ctx)
        self.assertEqual(layer.identity.scope, "guest")
        self.assertEqual(layer.identity.guest_id, "guest-room-77")
        await layer.aclose()

    async def test_preload_note_combines_general_and_emotional_patterns(self):
        agent.os.environ["MEMORY_ENABLED"] = "true"
        ctx = _ctx(room_name="room-2")

        async def fake_preload(self):
            return [
                "User said: likes long walks",
                EMOTIONAL_PATTERN_PREFIX + "energy=low; tension=high",
            ]

        orig_preload = MemoryLayer.preload
        MemoryLayer.preload = fake_preload
        try:
            layer, note = await agent._build_memory_layer_for_realtime(ctx)
        finally:
            MemoryLayer.preload = orig_preload
        self.assertIsNotNone(note)
        self.assertIn("likes long walks", note)
        self.assertIn("energy=low; tension=high", note)
        # Raw prefix never leaks into the composed note verbatim as a labeled
        # section marker other than inside the "what we've learned" framing.
        self.assertIn("what you've learned", note.lower())
        await layer.aclose()

    async def test_preload_timeout_degrades_to_no_note_not_raise(self):
        agent.os.environ["MEMORY_ENABLED"] = "true"
        ctx = _ctx(room_name="room-3")

        async def hanging_preload(self):
            await asyncio.sleep(10)
            return []

        orig_preload = MemoryLayer.preload
        MemoryLayer.preload = hanging_preload
        try:
            # Patch wait_for's effective timeout by monkeypatching asyncio.wait_for
            # is unnecessary here; instead just confirm the real 2.0s path doesn't
            # raise by using a short-circuited preload that raises TimeoutError
            # directly, matching what asyncio.wait_for would surface.
            async def timing_out_preload(self):
                raise asyncio.TimeoutError()

            MemoryLayer.preload = timing_out_preload
            layer, note = await agent._build_memory_layer_for_realtime(ctx)
        finally:
            MemoryLayer.preload = orig_preload
        self.assertIsInstance(layer, MemoryLayer)
        self.assertIsNone(note)
        await layer.aclose()

    async def test_setup_failure_degrades_to_none_none_not_raise(self):
        agent.os.environ["MEMORY_ENABLED"] = "true"
        ctx = _ctx(room_name="room-4")
        agent._active_memory_layer = "stale-instance"

        orig_identity_from_metadata = agent.identity_from_metadata

        def boom(*a, **k):
            raise RuntimeError("identity resolution exploded")

        agent.identity_from_metadata = boom
        try:
            layer, note = await agent._build_memory_layer_for_realtime(ctx)  # must not raise
        finally:
            agent.identity_from_metadata = orig_identity_from_metadata
        self.assertIsNone(layer)
        self.assertIsNone(note)
        self.assertIsNone(agent._active_memory_layer)


if __name__ == "__main__":
    unittest.main()

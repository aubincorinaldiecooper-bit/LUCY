import types
import unittest

import agent


class _FakeHandle:
    async def wait_for_playout(self):
        return None


class _FakeSession:
    def __init__(self):
        self.said: list[tuple[str, bool]] = []

    async def say(self, text, allow_interruptions=True, **_):
        self.said.append((text, allow_interruptions))
        return _FakeHandle()


class _FakeCtx:
    def __init__(self):
        self.deleted = 0
        self.room = types.SimpleNamespace(name="lucy-test")

    async def delete_room(self):
        self.deleted += 1


class SessionTimeLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig = {
            k: getattr(agent, k)
            for k in (
                "SESSION_TIME_LIMIT_ENABLED",
                "SESSION_MAX_DURATION_SECONDS",
                "SESSION_ENDING_WARNING_SECONDS",
                "SESSION_ENDING_WARNING_TEXT",
                "SESSION_ENDING_GOODBYE_TEXT",
            )
        }
        agent.SESSION_TIME_LIMIT_ENABLED = True
        agent.SESSION_MAX_DURATION_SECONDS = 0.2
        agent.SESSION_ENDING_WARNING_SECONDS = 0.1
        agent.SESSION_ENDING_WARNING_TEXT = "thirty seconds left"
        agent.SESSION_ENDING_GOODBYE_TEXT = "that's our time"

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(agent, k, v)

    async def test_warns_then_says_goodbye_then_terminates(self):
        session = _FakeSession()
        ctx = _FakeCtx()
        await agent._run_session_time_limit(session, ctx)

        # Warning first (interruptible), goodbye second (not interruptible).
        self.assertEqual([t for t, _ in session.said], ["thirty seconds left", "that's our time"])
        self.assertTrue(session.said[0][1])  # warning allows interruptions
        self.assertFalse(session.said[1][1])  # goodbye does not
        self.assertEqual(ctx.deleted, 1)

    async def test_disabled_does_nothing(self):
        agent.SESSION_TIME_LIMIT_ENABLED = False
        session = _FakeSession()
        ctx = _FakeCtx()
        await agent._run_session_time_limit(session, ctx)
        self.assertEqual(session.said, [])
        self.assertEqual(ctx.deleted, 0)

    async def test_zero_duration_does_nothing(self):
        agent.SESSION_MAX_DURATION_SECONDS = 0
        session = _FakeSession()
        ctx = _FakeCtx()
        await agent._run_session_time_limit(session, ctx)
        self.assertEqual(session.said, [])
        self.assertEqual(ctx.deleted, 0)

    async def test_no_warning_text_still_terminates(self):
        agent.SESSION_ENDING_WARNING_TEXT = ""
        agent.SESSION_ENDING_GOODBYE_TEXT = ""
        session = _FakeSession()
        ctx = _FakeCtx()
        await agent._run_session_time_limit(session, ctx)
        self.assertEqual(session.said, [])
        self.assertEqual(ctx.deleted, 1)

    async def test_terminate_falls_back_to_room_disconnect(self):
        session = _FakeSession()
        disconnects = {"n": 0}

        async def _disconnect():
            disconnects["n"] += 1

        ctx = types.SimpleNamespace(
            room=types.SimpleNamespace(name="lucy-test", disconnect=_disconnect)
        )
        # No delete_room / api on ctx -> falls back to room.disconnect().
        strategy = await agent._terminate_room(ctx)
        self.assertEqual(strategy, "room_disconnect")
        self.assertEqual(disconnects["n"], 1)


if __name__ == "__main__":
    unittest.main()

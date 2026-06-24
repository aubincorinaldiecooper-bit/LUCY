import json
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


class _FakeLocalParticipant:
    def __init__(self):
        self.published: list[dict] = []

    async def publish_data(self, payload, reliable=True, topic=None):
        self.published.append(
            {"payload": json.loads(payload.decode("utf-8")), "reliable": reliable, "topic": topic}
        )


class _FakeCtx:
    def __init__(self, with_local_participant=True):
        self.deleted = 0
        local = _FakeLocalParticipant() if with_local_participant else None
        self.local_participant = local
        self.room = types.SimpleNamespace(name="lucy-test", local_participant=local)

    async def delete_room(self):
        self.deleted += 1


class SessionTimeLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig = {
            k: getattr(agent, k)
            for k in (
                "SESSION_TIME_LIMIT_ENABLED",
                "SESSION_MAX_DURATION_SECONDS",
                "SESSION_ENDING_NOTICE_SECONDS",
                "SESSION_ENDING_GOODBYE_TEXT",
                "SESSION_ENDING_NOTICE_TOPIC",
            )
        }
        agent.SESSION_TIME_LIMIT_ENABLED = True
        agent.SESSION_MAX_DURATION_SECONDS = 0.2
        agent.SESSION_ENDING_NOTICE_SECONDS = 0.1
        agent.SESSION_ENDING_GOODBYE_TEXT = "that's our time"
        agent.SESSION_ENDING_NOTICE_TOPIC = "session"

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(agent, k, v)

    async def test_publishes_notice_then_goodbye_then_terminates(self):
        session = _FakeSession()
        ctx = _FakeCtx()
        await agent._run_session_time_limit(session, ctx)

        # Ending notice went to the client first.
        self.assertEqual(len(ctx.local_participant.published), 1)
        notice = ctx.local_participant.published[0]
        self.assertEqual(notice["payload"]["type"], "session_ending")
        self.assertTrue(notice["payload"]["seconds_remaining"] >= 0)
        self.assertEqual(notice["topic"], "session")
        self.assertTrue(notice["reliable"])
        # Goodbye spoken (not interruptible) and room ended.
        self.assertEqual([t for t, _ in session.said], ["that's our time"])
        self.assertFalse(session.said[0][1])
        self.assertEqual(ctx.deleted, 1)

    async def test_no_goodbye_text_still_notifies_and_terminates(self):
        agent.SESSION_ENDING_GOODBYE_TEXT = ""
        session = _FakeSession()
        ctx = _FakeCtx()
        await agent._run_session_time_limit(session, ctx)
        self.assertEqual(len(ctx.local_participant.published), 1)
        self.assertEqual(session.said, [])
        self.assertEqual(ctx.deleted, 1)

    async def test_missing_local_participant_still_terminates(self):
        session = _FakeSession()
        ctx = _FakeCtx(with_local_participant=False)
        # No participant to publish to -> notice skipped, but the hard end still runs.
        await agent._run_session_time_limit(session, ctx)
        self.assertEqual(ctx.deleted, 1)

    async def test_disabled_does_nothing(self):
        agent.SESSION_TIME_LIMIT_ENABLED = False
        session = _FakeSession()
        ctx = _FakeCtx()
        await agent._run_session_time_limit(session, ctx)
        self.assertEqual(session.said, [])
        self.assertEqual(ctx.local_participant.published, [])
        self.assertEqual(ctx.deleted, 0)

    async def test_zero_duration_does_nothing(self):
        agent.SESSION_MAX_DURATION_SECONDS = 0
        session = _FakeSession()
        ctx = _FakeCtx()
        await agent._run_session_time_limit(session, ctx)
        self.assertEqual(session.said, [])
        self.assertEqual(ctx.local_participant.published, [])
        self.assertEqual(ctx.deleted, 0)

    async def test_terminate_falls_back_to_room_disconnect(self):
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

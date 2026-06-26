"""Tests for handoff-guard suppression recovery.

When the handoff guard drops a thinking-phase reply on a barge-in, the barge-in is
*expected* to commit a replacement turn. When it does not, the conversation would
dead-end in silence (observed in production: a 75-char reply dropped, then ~20s of
dead air until the user said "Hello?"). These tests cover the recovery watchdog
that re-speaks the dropped reply only when no replacement turn took the floor.
"""

import asyncio
import types
import unittest
from unittest.mock import patch

import agent


class FakeSession:
    """Minimal AgentSession stand-in recording say() calls."""

    def __init__(self):
        self.calls = []

    async def say(self, text, *, allow_interruptions=None, add_to_chat_ctx=None):
        self.calls.append(
            {
                "text": text,
                "allow_interruptions": allow_interruptions,
                "add_to_chat_ctx": add_to_chat_ctx,
            }
        )
        return object()


class ShouldRecoverDecisionTests(unittest.TestCase):
    """Pure decision: re-speak only when the drop would otherwise leave dead air."""

    def test_no_text_never_recovers(self):
        ok, reason = agent._should_recover_suppressed_reply(
            armed_turn_id=8, current_turn_id=8, interaction_state="LISTENING", has_text=False
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "no_text")

    def test_new_turn_supersedes(self):
        # A replacement turn committed (current advanced past the armed turn): the
        # new turn owns the reply now, so re-speaking would double up.
        ok, reason = agent._should_recover_suppressed_reply(
            armed_turn_id=8, current_turn_id=9, interaction_state="LISTENING", has_text=True
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "superseded_by_new_turn")

    def test_floor_busy_states_block_recovery(self):
        for state in (
            "USER_SPEAKING",
            "USER_INTERRUPTING",
            "COMMITTED_TURN",
            "ASSISTANT_THINKING",
            "ASSISTANT_SPEAKING",
            "HOLDING_FRAGMENT",
        ):
            with self.subTest(state=state):
                ok, reason = agent._should_recover_suppressed_reply(
                    armed_turn_id=8, current_turn_id=8, interaction_state=state, has_text=True
                )
                self.assertFalse(ok)
                self.assertEqual(reason, f"floor_busy:{state.lower()}")

    def test_open_floor_states_recover(self):
        # LISTENING and a stale USER_TURN_CANDIDATE (the exact dead-air case where a
        # barge-in stopped but never committed a turn) both reopen the floor.
        for state in ("LISTENING", "USER_TURN_CANDIDATE", "RECOVERY", ""):
            with self.subTest(state=state):
                ok, reason = agent._should_recover_suppressed_reply(
                    armed_turn_id=8, current_turn_id=8, interaction_state=state, has_text=True
                )
                self.assertTrue(ok)
                self.assertTrue(reason.startswith("open_floor:"))

    def test_case_insensitive_state(self):
        ok, _ = agent._should_recover_suppressed_reply(
            armed_turn_id=8, current_turn_id=8, interaction_state="user_speaking", has_text=True
        )
        self.assertFalse(ok)


class RealLogScenarioTests(unittest.TestCase):
    """Reproduces the two suppressions from the production log and asserts the
    recovery decision matches what should have happened."""

    def test_turn6_user_kept_talking_does_not_recover(self):
        # Turn 6: reply "You want to go with me." suppressed, but the user kept
        # speaking (USER_SPEAKING at the grace check) and turn 7 took over. The
        # recovery must stay silent so it never talks over the real next turn.
        ok, reason = agent._should_recover_suppressed_reply(
            armed_turn_id=6, current_turn_id=6, interaction_state="USER_SPEAKING", has_text=True
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "floor_busy:user_speaking")

    def test_turn8_dead_air_recovers(self):
        # Turn 8: reply "It feels like a lot to even get to the point of noticing
        # all these changes." suppressed; the barge-in stopped (USER_TURN_CANDIDATE)
        # and never committed a turn -> ~20s of dead air. Recovery must fire.
        ok, reason = agent._should_recover_suppressed_reply(
            armed_turn_id=8,
            current_turn_id=8,
            interaction_state="USER_TURN_CANDIDATE",
            has_text=True,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "open_floor:user_turn_candidate")


class ArmAndClearTests(unittest.TestCase):
    def tearDown(self):
        agent._clear_suppressed_reply_recovery("test_teardown")
        agent._suppressed_reply_pending = False
        agent._suppressed_reply_text = ""
        agent._suppressed_reply_turn_id = 0

    def test_disabled_is_noop(self):
        with patch.object(agent, "HANDOFF_GUARD_RECOVERY_ENABLED", False):
            agent._arm_suppressed_reply_recovery(8, "some reply")
        self.assertFalse(agent._suppressed_reply_pending)
        self.assertIsNone(agent._suppressed_reply_recovery_task)

    def test_empty_text_is_noop(self):
        with patch.object(agent, "HANDOFF_GUARD_RECOVERY_ENABLED", True):
            agent._arm_suppressed_reply_recovery(8, "   ")
        self.assertFalse(agent._suppressed_reply_pending)

    def test_arm_without_running_loop_sets_state_no_task(self):
        # tts_node suppression may be exercised synchronously in tests; arming must
        # never raise just because there is no running event loop.
        with patch.object(agent, "HANDOFF_GUARD_RECOVERY_ENABLED", True):
            agent._arm_suppressed_reply_recovery(8, "It feels like a lot.")
        self.assertTrue(agent._suppressed_reply_pending)
        self.assertEqual(agent._suppressed_reply_turn_id, 8)
        self.assertEqual(agent._suppressed_reply_text, "It feels like a lot.")
        self.assertIsNone(agent._suppressed_reply_recovery_task)

    def test_clear_resets_pending(self):
        agent._suppressed_reply_pending = True
        agent._suppressed_reply_turn_id = 8
        agent._clear_suppressed_reply_recovery("new_turn_committed")
        self.assertFalse(agent._suppressed_reply_pending)


class WatchdogAsyncTests(unittest.TestCase):
    """End-to-end: arm inside a running loop, let the grace elapse, assert say()."""

    def _run(self, coro):
        return asyncio.run(coro)

    def tearDown(self):
        agent._clear_suppressed_reply_recovery("test_teardown")
        agent._suppressed_reply_pending = False

    def test_dead_air_triggers_respeak_without_duplicating_ctx(self):
        session = FakeSession()
        text = "It feels like a lot to even get to the point of noticing all these changes."

        async def scenario():
            with patch.multiple(
                agent,
                HANDOFF_GUARD_RECOVERY_ENABLED=True,
                HANDOFF_GUARD_RECOVERY_GRACE_MS=10,
                _active_agent_session=session,
                _current_turn_id=8,
                _interaction_state=types.SimpleNamespace(state="USER_TURN_CANDIDATE"),
                # A stale barge-in latch must be cleared by the recovery so the
                # re-spoken reply is not itself suppressed in tts_node.
                _barge_in_during_thinking_turn_id=8,
                _barge_in_started_at=1.0,
                _barge_in_confirmed_real=True,
            ):
                agent._arm_suppressed_reply_recovery(8, text)
                task = agent._suppressed_reply_recovery_task
                self.assertIsNotNone(task)
                await asyncio.wait_for(task, timeout=1.0)
                # Re-spoke the exact dropped reply, interruptible, WITHOUT re-adding
                # it to chat context (the suppressed item is already there).
                self.assertEqual(len(session.calls), 1)
                self.assertEqual(session.calls[0]["text"], text)
                self.assertTrue(session.calls[0]["allow_interruptions"])
                self.assertFalse(session.calls[0]["add_to_chat_ctx"])
                # Stale barge-in latch cleared.
                self.assertEqual(agent._barge_in_during_thinking_turn_id, 0)
                self.assertFalse(agent._suppressed_reply_pending)

        self._run(scenario())

    def test_new_turn_supersedes_no_respeak(self):
        session = FakeSession()

        async def scenario():
            with patch.multiple(
                agent,
                HANDOFF_GUARD_RECOVERY_ENABLED=True,
                HANDOFF_GUARD_RECOVERY_GRACE_MS=10,
                _active_agent_session=session,
                _current_turn_id=9,  # a replacement turn already committed
                _interaction_state=types.SimpleNamespace(state="LISTENING"),
            ):
                agent._arm_suppressed_reply_recovery(8, "stale reply")
                await asyncio.wait_for(agent._suppressed_reply_recovery_task, timeout=1.0)
                self.assertEqual(session.calls, [])

        self._run(scenario())

    def test_floor_busy_no_respeak(self):
        session = FakeSession()

        async def scenario():
            with patch.multiple(
                agent,
                HANDOFF_GUARD_RECOVERY_ENABLED=True,
                HANDOFF_GUARD_RECOVERY_GRACE_MS=10,
                _active_agent_session=session,
                _current_turn_id=8,
                _interaction_state=types.SimpleNamespace(state="USER_SPEAKING"),
            ):
                agent._arm_suppressed_reply_recovery(8, "stale reply")
                await asyncio.wait_for(agent._suppressed_reply_recovery_task, timeout=1.0)
                self.assertEqual(session.calls, [])

        self._run(scenario())

    def test_no_session_does_not_crash(self):
        async def scenario():
            with patch.multiple(
                agent,
                HANDOFF_GUARD_RECOVERY_ENABLED=True,
                HANDOFF_GUARD_RECOVERY_GRACE_MS=10,
                _active_agent_session=None,
                _current_turn_id=8,
                _interaction_state=types.SimpleNamespace(state="USER_TURN_CANDIDATE"),
            ):
                agent._arm_suppressed_reply_recovery(8, "stale reply")
                await asyncio.wait_for(agent._suppressed_reply_recovery_task, timeout=1.0)
                self.assertFalse(agent._suppressed_reply_pending)

        self._run(scenario())

    def test_clear_cancels_pending_watchdog(self):
        session = FakeSession()

        async def scenario():
            with patch.multiple(
                agent,
                HANDOFF_GUARD_RECOVERY_ENABLED=True,
                HANDOFF_GUARD_RECOVERY_GRACE_MS=10000,  # long grace; we cancel first
                _active_agent_session=session,
                _current_turn_id=8,
                _interaction_state=types.SimpleNamespace(state="USER_TURN_CANDIDATE"),
            ):
                agent._arm_suppressed_reply_recovery(8, "stale reply")
                task = agent._suppressed_reply_recovery_task
                agent._clear_suppressed_reply_recovery("new_turn_committed")
                self.assertFalse(agent._suppressed_reply_pending)
                # Give the cancelled task a chance to settle, then confirm no say().
                await asyncio.sleep(0.02)
                self.assertTrue(task.cancelled() or task.done())
                self.assertEqual(session.calls, [])

        self._run(scenario())


if __name__ == "__main__":
    unittest.main()

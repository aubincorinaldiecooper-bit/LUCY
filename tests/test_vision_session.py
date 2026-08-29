"""vision_session.py: bounded ingestion, lifecycle, telemetry, degradation.

Pins the rules that make live vision safe to ship next to the voice path:
  - vision NEVER raises into the caller; every failure degrades to voice-only,
  - the frame queue is bounded and drops the OLDEST frame, so latency stays
    bounded during a Modal cold start instead of climbing forever,
  - ingestion is rate-limited to VISION_TARGET_FPS,
  - the session is bound to the Lucy session id and cleans up completely on
    camera-stop, close, and timeout — no socket, task, frame or GPU session
    outlives the conversation,
  - a pending Gateway is a first-class state that never opens a socket,
  - cold-start latency is actually measured.

The provider/network boundary is injected via connection_factory, so nothing
here needs a GPU, Modal, or a socket.
"""

import asyncio
import unittest
from unittest import mock

import minicpmo_provider as mp
import vision_session as vsn
from minicpmo_provider import VisionProviderState

WORKER_URL = "https://aubincorinaldiecooper--minicpmo45-realtime-minicpmo-worker.modal.run"
GATEWAY_URL = "wss://minicpmo-gateway.example.com/v1/realtime?mode=video"
FRAME = "data:image/jpeg;base64,AAAA"


def _config(**overrides):
    kwargs = dict(
        enabled=True,
        worker_url=WORKER_URL,
        realtime_url=GATEWAY_URL,
        gateway_token="",
        connect_timeout_seconds=120.0,
        session_timeout_seconds=900.0,
        health_timeout_seconds=10.0,
        target_fps=2.0,
        max_queue_size=4,
        max_connect_attempts=2,
        max_total_connects=6,
        reconnect_backoff_seconds=0.0,
        min_state_update_interval_seconds=0.0,
    )
    kwargs.update(overrides)
    return mp.MiniCPMOConfig(**kwargs)


class FakeConnection:
    """Stands in for MiniCPMORealtimeConnection at the socket boundary."""

    def __init__(self, config, session_id, *, messages=None, fail_with=None,
                 send_error=None, hold_open=True):
        self.config = config
        self.session_id = session_id
        self._messages = list(messages or [])
        self._fail_with = fail_with
        self._send_error = send_error
        self._hold_open = hold_open
        self.sent = []
        self.closed = False
        self.connect_calls = 0

    async def connect(self):
        self.connect_calls += 1
        if self._fail_with is not None:
            raise self._fail_with

    async def send_frame(self, image):
        if self._send_error is not None:
            raise self._send_error
        self.sent.append(image)

    async def receive(self):
        for message in self._messages:
            yield message
        if self._hold_open:
            # Emulate a live socket that stays open after its messages.
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def _factory(**kwargs):
    made = []

    def build(config, session_id):
        conn = FakeConnection(config, session_id, **kwargs)
        made.append(conn)
        return conn

    build.made = made
    return build


async def _settle(times=8):
    """Let spawned tasks run without wall-clock sleeps."""
    for _ in range(times):
        await asyncio.sleep(0)


# --- bounded queue ------------------------------------------------------------


class BoundedFrameQueueTests(unittest.TestCase):
    def test_capacity_is_enforced(self):
        q = vsn.BoundedFrameQueue(4)
        for i in range(10):
            q.put(f"f{i}")
        self.assertEqual(len(q), 4)
        self.assertEqual(q.offered, 10)
        self.assertEqual(q.dropped, 6)

    def test_oldest_frames_are_dropped_not_newest(self):
        # For live understanding the newest frame is the valuable one.
        q = vsn.BoundedFrameQueue(3)
        for i in range(6):
            q.put(f"f{i}")

        async def drain():
            return [await q.get() for _ in range(3)]

        self.assertEqual(asyncio.run(drain()), ["f3", "f4", "f5"])

    def test_put_reports_whether_it_displaced(self):
        q = vsn.BoundedFrameQueue(2)
        self.assertTrue(q.put("a"))
        self.assertTrue(q.put("b"))
        self.assertFalse(q.put("c"))  # displaced "a"

    def test_get_awaits_until_a_frame_arrives(self):
        q = vsn.BoundedFrameQueue(2)

        async def scenario():
            task = asyncio.create_task(q.get())
            await _settle()
            self.assertFalse(task.done())
            q.put("late")
            return await asyncio.wait_for(task, timeout=1)

        self.assertEqual(asyncio.run(scenario()), "late")

    def test_zero_capacity_is_coerced_to_one(self):
        # A 0 cap would silently discard everything and look like a dead provider.
        q = vsn.BoundedFrameQueue(0)
        self.assertEqual(q.max_size, 1)
        q.put("a")
        self.assertEqual(len(q), 1)

    def test_clear_drops_everything_pending(self):
        q = vsn.BoundedFrameQueue(4)
        q.put("a")
        q.put("b")
        q.clear()
        self.assertEqual(len(q), 0)

    def test_stats_expose_capacity_and_counters(self):
        q = vsn.BoundedFrameQueue(2)
        q.put("a")
        self.assertEqual(q.stats()["capacity"], 2)
        self.assertEqual(q.stats()["pending"], 1)


# --- telemetry ----------------------------------------------------------------


class TelemetryTests(unittest.TestCase):
    def test_marks_and_derived_latencies(self):
        t = vsn.VisionLatencyTelemetry()
        t.mark("session_requested", now=0.0)
        t.mark("modal_request_started", now=0.5)
        t.mark("modal_connected", now=8.5)
        t.mark("model_ready", now=8.5)
        t.mark("first_frame_sent", now=9.0)
        t.mark("first_visual_event", now=10.0)
        t.cold = True
        self.assertEqual(t.modal_connect_ms, 8000.0)
        self.assertEqual(t.model_ready_ms, 8000.0)
        self.assertEqual(t.first_visual_context_ms, 10000.0)
        self.assertEqual(t.classify(), "cold")

    def test_marks_are_first_write_wins(self):
        t = vsn.VisionLatencyTelemetry()
        t.mark("first_frame_sent", now=1.0)
        t.mark("first_frame_sent", now=99.0)
        self.assertEqual(t.first_frame_sent_at, 1.0)

    def test_missing_marks_yield_none_not_zero(self):
        t = vsn.VisionLatencyTelemetry()
        self.assertIsNone(t.modal_connect_ms)
        self.assertIsNone(t.first_visual_context_ms)
        self.assertEqual(t.classify(), "unknown")

    def test_log_fields_are_key_value_and_carry_no_media(self):
        t = vsn.VisionLatencyTelemetry()
        t.mark("session_requested", now=0.0)
        fields = t.log_fields()
        self.assertIn("first_visual_context_ms=", fields)
        self.assertIn("start_class=", fields)
        self.assertNotIn("data:image", fields)


# --- lifecycle ----------------------------------------------------------------


class VisionSessionLifecycleTests(unittest.TestCase):
    def test_disabled_provider_never_starts(self):
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(enabled=False))
        self.assertFalse(session.start())
        self.assertIs(session.state, VisionProviderState.DISABLED)
        self.assertEqual(session.reason, "minicpmo_disabled")
        self.assertFalse(session.active)

    def test_build_vision_session_returns_none_when_disabled(self):
        self.assertIsNone(vsn.build_vision_session("lucy-1", config=_config(enabled=False)))
        self.assertIsNotNone(vsn.build_vision_session("lucy-1", config=_config()))

    def test_session_is_bound_to_the_lucy_session_id(self):
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-abc123", config=_config(), connection_factory=factory
        )

        async def scenario():
            session.start()
            await _settle()
            await session.aclose()

        asyncio.run(scenario())
        self.assertEqual(session.session_id, "lucy-abc123")
        # The Gateway gets a per-session stream id derived from (and traceable
        # to) the Lucy session id — see stream_id in MiniCPMOVisionSession.
        self.assertTrue(factory.made[0].session_id.startswith("lucy-abc123-"))
        self.assertEqual(session.stream_id, factory.made[0].session_id)

    def test_pending_gateway_is_unavailable_and_opens_no_socket(self):
        # The expected state today. Must cost nothing and wake no GPU.
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(realtime_url=""), connection_factory=factory
        )
        self.assertFalse(session.start())
        self.assertIs(session.state, VisionProviderState.UNAVAILABLE)
        self.assertEqual(session.reason, "realtime_gateway_not_configured")
        self.assertEqual(factory.made, [], "pending Gateway must not construct a connection")

    def test_worker_url_as_realtime_url_is_unavailable_not_attempted(self):
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-1",
            config=_config(realtime_url=WORKER_URL.replace("https://", "wss://")),
            connection_factory=factory,
        )
        self.assertFalse(session.start())
        self.assertEqual(session.reason, "realtime_url_is_worker_url")
        self.assertEqual(factory.made, [])

    def test_successful_connect_reaches_ready(self):
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=factory
        )

        async def scenario():
            self.assertTrue(session.start())
            await _settle()
            self.assertIs(session.state, VisionProviderState.READY)
            self.assertTrue(session.active)
            await session.aclose()

        asyncio.run(scenario())

    def test_start_is_idempotent(self):
        factory = _factory()
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            session.start()
            await _settle()
            await session.aclose()

        asyncio.run(scenario())
        self.assertEqual(len(factory.made), 1)


class FailureDegradationTests(unittest.TestCase):
    """Vision failure must always degrade to voice-only, never raise."""

    def test_connect_timeout_degrades_to_unavailable(self):
        factory = _factory(fail_with=mp.MiniCPMOConnectionError("connect_timeout", "cold"))
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(max_connect_attempts=1), connection_factory=factory
        )

        async def scenario():
            session.start()
            await _settle(20)
            await session.aclose()

        asyncio.run(scenario())
        self.assertIs(session.state, VisionProviderState.DISABLED)  # after aclose
        self.assertTrue(factory.made[0].closed)

    def test_connect_timeout_reports_unavailable_before_teardown(self):
        factory = _factory(fail_with=mp.MiniCPMOConnectionError("connect_timeout", "cold"))
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(max_connect_attempts=1), connection_factory=factory
        )

        async def scenario():
            session.start()
            await _settle(20)
            return session.state, session.reason

        state, reason = asyncio.run(scenario())
        self.assertIs(state, VisionProviderState.UNAVAILABLE)
        self.assertEqual(reason, "connect_timeout")

    def test_provider_unavailable_degrades(self):
        factory = _factory(fail_with=mp.MiniCPMOConnectionError("connect_failed", "refused"))
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(max_connect_attempts=1), connection_factory=factory
        )

        async def scenario():
            session.start()
            await _settle(20)
            return session.state, session.reason

        state, reason = asyncio.run(scenario())
        self.assertIs(state, VisionProviderState.DEGRADED)
        self.assertEqual(reason, "connect_failed")

    def test_retries_are_conservative_and_bounded(self):
        # Every attempt can wake an L40S; an outage must not become a GPU-spinning
        # retry storm.
        factory = _factory(fail_with=mp.MiniCPMOConnectionError("connect_failed", "boom"))
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(max_connect_attempts=2), connection_factory=factory
        )

        async def scenario():
            session.start()
            await _settle(40)
            await session.aclose()

        asyncio.run(scenario())
        self.assertEqual(len(factory.made), 2, "must stop at max_connect_attempts")

    def test_gateway_disconnect_mid_session_degrades(self):
        factory = _factory(messages=['{"scene_summary": "A desk."}'], hold_open=False)
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle(20)
            await session.aclose()

        asyncio.run(scenario())
        # The socket ending is not an exception anywhere; the session simply winds down.
        self.assertTrue(factory.made[0].closed)

    def test_send_failure_degrades_without_raising(self):
        factory = _factory(send_error=ConnectionResetError("gateway gone"))
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle()
            session.offer_frame(FRAME)
            await _settle(20)
            return session.state, session.reason

        state, reason = asyncio.run(scenario())
        self.assertIs(state, VisionProviderState.DEGRADED)
        self.assertEqual(reason, "send_failed")

    def test_a_consumer_callback_fault_does_not_break_vision(self):
        async def bad_callback(_state):
            raise RuntimeError("consumer exploded")

        factory = _factory(messages=['{"scene_summary": "A red book."}'])
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=factory,
            on_visual_update=bad_callback,
        )

        async def scenario():
            session.start()
            await _settle(20)
            state = session.state
            await session.aclose()
            return state

        # Degraded-or-ready is fine; raising is not.
        self.assertIsNotNone(asyncio.run(scenario()))

    def test_offer_frame_never_raises_and_is_false_when_inactive(self):
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(enabled=False))
        session.start()
        self.assertFalse(session.offer_frame(FRAME))
        self.assertFalse(session.offer_frame(None))
        self.assertFalse(session.offer_frame("not-a-data-url"))


class FrameIngestionTests(unittest.TestCase):
    def test_ingestion_is_rate_limited_to_target_fps(self):
        clock = {"t": 0.0}
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(target_fps=2.0),  # one frame per 0.5s
            connection_factory=factory, clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle()
            accepted = 0
            # 10 offers within 0.4s of each other: only ~1 per 0.5s may pass.
            for i in range(10):
                clock["t"] = 1.0 + i * 0.1
                accepted += bool(session.offer_frame(FRAME))
            await session.aclose()
            return accepted

        accepted = asyncio.run(scenario())
        self.assertEqual(accepted, 2, "0.9s of offers at 2 FPS should accept 2 frames")

    def test_backpressure_drops_stale_frames(self):
        clock = {"t": 0.0}
        # A connection that never drains: emulates the model being slower than
        # the camera, which is exactly the Modal cold-start case.
        factory = _factory(hold_open=True)
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(target_fps=100.0, max_queue_size=4),
            connection_factory=factory, clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle()
            for i in range(30):
                clock["t"] = 10.0 + i  # far apart, so the FPS gate never blocks
                session.offer_frame(f"data:image/jpeg;base64,frame{i}")
            stats = session.queue.stats()
            await session.aclose()
            return stats

        stats = asyncio.run(scenario())
        self.assertLessEqual(stats["pending"], 4, "queue must stay bounded")
        self.assertGreater(stats["dropped"], 0, "stale frames must be dropped")
        self.assertEqual(stats["offered"], 30)

    def test_frames_reach_the_provider(self):
        clock = {"t": 0.0}
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=factory,
            clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle()
            clock["t"] = 10.0
            session.offer_frame(FRAME)
            await _settle(20)
            sent = list(factory.made[0].sent)
            await session.aclose()
            return sent

        self.assertEqual(asyncio.run(scenario()), [FRAME])


class VisualStateFlowTests(unittest.TestCase):
    def test_notable_change_invokes_the_consumer_once(self):
        seen = []

        async def on_update(state):
            seen.append(state)

        factory = _factory(messages=[
            '{"scene_summary": "A person at a desk.", "objects": ["desk"]}',
            '{"scene_summary": "A person at a desk.", "objects": ["desk"]}',  # static
            '{"scene_summary": "User picked up a red book.", "objects": ["desk", "red book"],'
            ' "actions": ["picked up"]}',
        ])
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=factory, on_visual_update=on_update
        )

        async def scenario():
            session.start()
            await _settle(30)
            live_line = session.context_line()
            await session.aclose()
            return live_line, session.context_line()

        live_line, closed_line = asyncio.run(scenario())
        # First observation + the real change; the static repeat must not publish.
        self.assertEqual(len(seen), 2, [s.scene_summary for s in seen])
        self.assertIn("red book", seen[-1].objects)
        self.assertIn("red book", live_line)
        self.assertIsNone(closed_line, "context line must be cleared after close")

    def test_malformed_provider_messages_are_survived(self):
        factory = _factory(messages=[
            '{"type": "pong"}', "not json at all {", b"\xff\xfe", "",
            '{"scene_summary": "A lamp."}',
        ])
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle(30)
            state, line = session.latest_state, session.context_line()
            await session.aclose()
            return state, line

        state, line = asyncio.run(scenario())
        self.assertIsNotNone(state, "a good message after garbage must still land")
        self.assertIn("lamp", line)

    def test_first_visual_event_latency_is_recorded(self):
        clock = {"t": 0.0}
        factory = _factory(messages=['{"scene_summary": "A lamp."}'])
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=factory,
            clock=lambda: clock["t"],
        )

        async def scenario():
            clock["t"] = 0.0
            session.start()          # marks session_requested
            clock["t"] = 3.0         # connect takes 3s
            await _settle(30)
            t = session.telemetry
            await session.aclose()
            return t

        t = asyncio.run(scenario())
        self.assertIsNotNone(t.session_requested_at)
        self.assertIsNotNone(t.modal_connected_at)
        self.assertIsNotNone(t.first_visual_event_at)
        self.assertIsNotNone(t.first_visual_context_ms)

    def test_cold_start_is_classified_from_connect_duration(self):
        clock = {"t": 0.0}

        class SlowConnection(FakeConnection):
            async def connect(self):
                # Emulate Modal waking an L40S from scale-to-zero.
                clock["t"] += 45.0
                await super().connect()

        def factory(config, session_id):
            return SlowConnection(config, session_id, messages=[])

        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=factory,
            clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle(20)
            cls = session.telemetry.classify()
            connect_ms = session.telemetry.modal_connect_ms
            await session.aclose()
            return cls, connect_ms

        cls, connect_ms = asyncio.run(scenario())
        self.assertEqual(cls, "cold")
        self.assertEqual(connect_ms, 45000.0)

    def test_warm_start_is_classified(self):
        clock = {"t": 0.0}

        class FastConnection(FakeConnection):
            async def connect(self):
                clock["t"] += 0.2
                await super().connect()

        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(),
            connection_factory=lambda c, s: FastConnection(c, s, messages=[]),
            clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle(20)
            cls = session.telemetry.classify()
            await session.aclose()
            return cls

        self.assertEqual(asyncio.run(scenario()), "warm")


class CleanupTests(unittest.TestCase):
    def test_camera_stop_releases_everything(self):
        clock = {"t": 0.0}
        factory = _factory(hold_open=True)
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(target_fps=100.0), connection_factory=factory,
            clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle()
            for i in range(6):
                clock["t"] = 10.0 + i
                session.offer_frame(f"data:image/jpeg;base64,f{i}")
            await session.stop(reason="camera_stopped")
            return session

        s = asyncio.run(scenario())
        self.assertTrue(factory.made[0].closed, "Gateway socket must be closed")
        self.assertEqual(len(s.queue), 0, "queued frames must be dropped")
        self.assertIsNone(s.latest_state, "visual state must be cleared")
        self.assertIsNone(s.context_line())
        self.assertIs(s.state, VisionProviderState.DISABLED)
        self.assertEqual(s.reason, "camera_stopped")
        self.assertFalse(s.active)

    def test_stop_is_idempotent(self):
        factory = _factory()
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle()
            await session.stop()
            await session.stop()
            await session.aclose()

        asyncio.run(scenario())  # must not raise

    def test_stop_before_start_is_safe(self):
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config())
        asyncio.run(session.aclose())

    def test_no_tasks_survive_close(self):
        factory = _factory(hold_open=True)
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle()
            await session.aclose()
            # Nothing this session spawned may still be pending.
            return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        self.assertEqual(asyncio.run(scenario()), [])

    def test_frames_are_rejected_after_terminal_close(self):
        # aclose() ends the conversation: nothing may resurrect vision after it.
        factory = _factory()
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle()
            await session.aclose()
            return session.offer_frame(FRAME), session.start(), session.wants_frames

        offered, restarted, wants = asyncio.run(scenario())
        self.assertFalse(offered)
        self.assertFalse(restarted)
        self.assertFalse(wants)

    def test_camera_off_then_on_restores_live_vision(self):
        # The bug this guards: stop() used to leave the session permanently
        # spent, so a user toggling their camera got the OpenRouter path back
        # but MiniCPM-o stayed silently dark for the rest of the conversation.
        clock = {"t": 0.0}
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=factory,
            clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle()
            await session.stop(reason="camera_stopped")
            # Camera comes back on: a frame alone must revive the session.
            clock["t"] = 100.0
            accepted = session.offer_frame(FRAME)
            await _settle(20)
            state = session.state
            await session.aclose()
            return accepted, state

        accepted, state = asyncio.run(scenario())
        self.assertTrue(accepted, "a frame after camera-off must be accepted")
        self.assertIs(state, VisionProviderState.READY, "vision must reconnect")
        self.assertEqual(len(factory.made), 2, "a fresh connection is opened")
        self.assertTrue(factory.made[0].closed, "the first connection is released")

    def test_stop_resets_telemetry_for_the_next_camera_session(self):
        factory = _factory()
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle()
            await session.stop()
            return session.telemetry

        t = asyncio.run(scenario())
        self.assertIsNone(t.modal_connected_at,
                          "stale timings must not describe the next session")


class StatusReportingTests(unittest.TestCase):
    def test_status_reports_state_gateway_frames_and_telemetry(self):
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(realtime_url=""))
        session.start()
        status = session.status().as_dict()
        self.assertEqual(status["state"], "unavailable")
        self.assertEqual(status["gateway"], "pending_deployment")
        self.assertEqual(status["reason"], "realtime_gateway_not_configured")
        self.assertEqual(status["frames"]["capacity"], 4)
        self.assertIn("first_visual_context_ms", status["telemetry"])

    def test_status_carries_no_media_or_infrastructure_urls(self):
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config())
        blob = repr(session.status().as_dict())
        self.assertNotIn("data:image", blob)
        self.assertNotIn("modal.run", blob)


if __name__ == "__main__":
    unittest.main()


class HardeningTests(unittest.TestCase):
    """Regression guards for defects found in adversarial review."""

    def test_cancelled_stop_still_releases_the_socket_and_allows_later_teardown(self):
        # The worst bug class here: a cancelled stop() used to latch _stopping
        # True forever, so every later stop()/aclose() silently no-opped and the
        # Gateway socket plus its Modal GPU session leaked for good.
        factory = _factory(hold_open=True)
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle()
            task = asyncio.create_task(session.stop())
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await _settle(10)
            # Teardown must not be latched shut.
            await session.aclose()
            await _settle(10)
            return factory.made[0].closed, session._stopping

        closed, stopping = asyncio.run(scenario())
        self.assertTrue(closed, "the Gateway socket must still be released")
        self.assertFalse(stopping, "_stopping must never stay latched")

    def test_connect_budget_bounds_gpu_wakes_across_camera_toggles(self):
        # Re-arm must not reset the budget, or toggling the camera would wake an
        # L40S an unbounded number of times.
        clock = {"t": 0.0}
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(max_total_connects=3),
            connection_factory=factory, clock=lambda: clock["t"],
        )

        async def scenario():
            for i in range(12):
                clock["t"] = 100.0 * (i + 1)
                session.offer_frame(FRAME)
                await _settle(6)
                await session.stop(reason="camera_off")
            state, reason = session.state, session.reason
            await session.aclose()
            return state, reason

        asyncio.run(scenario())
        self.assertEqual(len(factory.made), 3, "budget must cap total connects")

    def test_exhausted_budget_stops_wanting_frames(self):
        factory = _factory(fail_with=mp.MiniCPMOConnectionError("connect_failed", "x"))
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(max_total_connects=2, max_connect_attempts=2),
            connection_factory=factory,
        )

        async def scenario():
            session.start()
            await _settle(30)
            return session.wants_frames

        self.assertFalse(asyncio.run(scenario()))

    def test_degrading_clears_the_stale_visual_note(self):
        # A frozen last-known scene would shadow the OpenRouter fallback for the
        # rest of the conversation, because that fallback is only reached when
        # the live context line is empty.
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config())
        session.rolling.ingest({"scene_summary": "A red book."}, now=1.0)
        self.assertIsNotNone(session.context_line())
        session._set_state(VisionProviderState.DEGRADED, "gateway_disconnected")
        self.assertIsNone(session.context_line(),
                          "a degraded provider must not keep describing the room")

    def test_clean_gateway_close_degrades_and_cancels_the_sender(self):
        # asyncio.wait returns on FIRST_COMPLETED; the send loop would otherwise
        # stay parked on queue.get() forever with state still READY.
        factory = _factory(messages=['{"scene_summary": "A desk."}'], hold_open=False)
        session = vsn.MiniCPMOVisionSession("lucy-1", config=_config(),
                                            connection_factory=factory)

        async def scenario():
            session.start()
            await _settle(30)
            state, reason = session.state, session.reason
            stray = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            await session.aclose()
            return state, reason, len(stray)

        state, reason, stray = asyncio.run(scenario())
        self.assertIs(state, VisionProviderState.DEGRADED)
        self.assertEqual(reason, "gateway_closed")
        self.assertEqual(stray, 0, "no loop may outlive _run()")

    def test_oversized_frames_are_refused(self):
        # Without a per-frame ceiling an API client could pin tens of megabytes.
        clock = {"t": 0.0}
        factory = _factory()
        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=factory,
            clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle()
            clock["t"] = 50.0
            huge = "data:image/jpeg;base64," + "A" * (mp.MAX_FRAME_CHARS + 1)
            accepted = session.offer_frame(huge)
            pending = len(session.queue)
            await session.aclose()
            return accepted, pending

        accepted, pending = asyncio.run(scenario())
        self.assertFalse(accepted)
        self.assertEqual(pending, 0)

    def test_cleared_frames_are_counted_as_dropped(self):
        q = vsn.BoundedFrameQueue(4)
        for i in range(3):
            q.put(f"f{i}")
        q.clear()
        stats = q.stats()
        self.assertEqual(stats["dropped"], 3)
        self.assertEqual(
            stats["offered"], stats["delivered"] + stats["dropped"] + stats["pending"],
            "frame accounting must balance",
        )

    def test_cold_warm_is_judged_on_the_successful_attempt_not_the_retries(self):
        clock = {"t": 0.0}
        calls = {"n": 0}

        class FlakyThenFast:
            def __init__(self, cfg, sid):
                self.closed = False
                self.session_id = sid
            async def connect(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    clock["t"] += 30.0          # slow failure
                    raise mp.MiniCPMOConnectionError("connect_failed", "boom")
                clock["t"] += 0.2               # the real container is warm
            async def send_frame(self, i): pass
            async def receive(self):
                await asyncio.Event().wait()
                yield None
            async def aclose(self): self.closed = True

        session = vsn.MiniCPMOVisionSession(
            "lucy-1", config=_config(), connection_factory=FlakyThenFast,
            clock=lambda: clock["t"],
        )

        async def scenario():
            session.start()
            await _settle(30)
            t = session.telemetry
            await session.aclose()
            return t.classify(), t.last_attempt_connect_ms, t.modal_connect_ms

        cls, last_ms, total_ms = asyncio.run(scenario())
        self.assertEqual(cls, "warm", "a retry must not make a warm container look cold")
        self.assertEqual(last_ms, 200.0)
        self.assertGreater(total_ms, last_ms, "total wait still spans the failed attempt")

    def test_gateway_identity_is_unique_per_session_even_on_a_shared_lucy_id(self):
        # Lucy's session id can be a process-wide env var or a millisecond
        # timestamp, so two conversations can share one. The Gateway is shared
        # infrastructure keyed on what we send it, so it must get a unique id
        # or two users' video could be conflated.
        factory_a, factory_b = _factory(), _factory()
        a = vsn.MiniCPMOVisionSession("lucy-same-id", config=_config(),
                                      connection_factory=factory_a)
        b = vsn.MiniCPMOVisionSession("lucy-same-id", config=_config(),
                                      connection_factory=factory_b)

        async def scenario():
            a.start(); b.start()
            await _settle(10)
            ids = (factory_a.made[0].session_id, factory_b.made[0].session_id)
            await a.aclose(); await b.aclose()
            return ids

        id_a, id_b = asyncio.run(scenario())
        self.assertNotEqual(id_a, id_b, "Gateway identities must not collide")
        # Lucy-side association is still the real session id, for logs/status.
        self.assertEqual(a.session_id, "lucy-same-id")
        self.assertEqual(b.session_id, "lucy-same-id")
        self.assertTrue(id_a.startswith("lucy-same-id-"))

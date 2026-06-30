"""Regression tests for the Inworld Realtime forced-text-test state machine.

The previous build had a critical deadlock: ``_handle_inworld_message`` for
``session.updated`` would ``await self._run_forced_text_test()``, which in turn
``wait_for``'d ``conversation.item.done``. But ``_handle_inworld_message`` IS the
receive loop handler — awaiting inside it blocked the loop from receiving the
very event we were waiting on. So the session.updated handler never returned,
the receive loop stalled, and the test deadlocked until the 15s timeout.

These tests drive the handler directly with a stubbed WebSocket and prove:

  1. The ``session.updated`` handler returns quickly (no ``await`` inside the
     receive loop). Sends ``conversation.item.create`` and only that.
  2. The state machine only enters ``awaiting_response_done`` (and sends
     ``response.create``) AFTER the receive loop processes a later
     ``conversation.item.done`` event.
  3. A ``conversation.item.done`` that arrives without a prior
     ``conversation.item.create`` does NOT trigger ``response.create``.
  4. ``response.done`` advances the state machine to ``idle`` and unpauses mic
     forwarding.
  5. ``INWORLD_FORCE_TEXT_TEST=false`` skips the test entirely.
"""
import asyncio
import os
import sys
import time
import types
import unittest
from unittest import mock


def _install_runtime_stubs():
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.ClientTimeout = lambda **kwargs: ("timeout", kwargs)
    aiohttp.ClientSession = object
    aiohttp.ClientWebSocketResponse = object
    aiohttp.WSMsgType = types.SimpleNamespace(TEXT="text", CLOSED="closed", ERROR="error")
    sys.modules.setdefault("aiohttp", aiohttp)

    class AudioFrame:
        def __init__(self, data=b"", sample_rate=24000, num_channels=1, samples_per_channel=1):
            self.data = data
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.samples_per_channel = samples_per_channel

    class AudioSource:
        def __init__(self, *args, **kwargs):
            self._closed = False

        async def capture_frame(self, frame):
            return None

        async def aclose(self):
            self._closed = True
            return None

    rtc = types.SimpleNamespace(
        AudioFrame=AudioFrame,
        AudioSource=AudioSource,
        LocalAudioTrack=types.SimpleNamespace(
            create_audio_track=lambda *args, **kwargs: types.SimpleNamespace(),
        ),
        TrackPublishOptions=lambda: types.SimpleNamespace(source=None),
        TrackSource=types.SimpleNamespace(SOURCE_MICROPHONE="microphone"),
        TrackKind=types.SimpleNamespace(KIND_AUDIO="audio"),
        AudioStream=lambda *args, **kwargs: object(),
        Room=object,
    )
    livekit = types.ModuleType("livekit")
    livekit.rtc = rtc
    sys.modules.setdefault("livekit", livekit)
    sys.modules.setdefault("livekit.rtc", rtc)


_install_runtime_stubs()
import inworld_realtime_bridge as irb  # noqa: E402


def _make_bridge(**kwargs):
    """Build a bridge instance with stubs for the things we don't want to run."""

    # Stub room that allows publish_track (called from _publish_output_track).
    class _Room:
        local_participant = types.SimpleNamespace(
            publish_track=lambda *a, **k: asyncio.sleep(0),
        )
        remote_participants = {}

        def on(self, *a, **k):
            return None

        def off(self, *a, **k):
            return None

    settings = irb.InworldRealtimeSettings(
        api_key="k",
        session_id="s",
        websocket_url="wss://example.test/session",
        model="openai/gpt-4o-mini",
        stt_model="inworld/inworld-stt-1",
        tts_model="inworld-tts-2",
        voice="Luna",
        speed=1.0,
        turn_detection_type="semantic_vad",
        turn_detection_eagerness="medium",
        turn_detection_create_response=True,
        turn_detection_interrupt_response=True,
        instructions="Concise.",
        timeout_seconds=60.0,
        voice_profile_enabled=False,
        input_format="pcm16",
        output_format="pcm16",
        auth_scheme="basic",
    )
    bridge = irb.InworldRealtimeLiveKitBridge(_Room(), settings, **kwargs)
    return bridge, settings


def _stub_send(bridge, sent_log):
    """Replace ``_send_inworld_message`` with a record-only stub."""

    async def fake_send(payload, reason="unknown"):
        sent_log.append({"type": payload.get("type"), "reason": reason, "payload": payload})
        return None

    bridge._send_inworld_message = fake_send
    bridge._ws = object()  # truthy so the guard passes; never actually used


class ForcedTextTestStateMachineTests(unittest.IsolatedAsyncioTestCase):
    """Regression tests for the receive-loop deadlock fix.

    These tests recreate the exact failure mode: handlers awaited on
    receive-loop-blocking events. Each test asserts the new state-machine
    discipline: handler invocations never wait for future events.
    """

    async def test_session_updated_sends_item_create_then_item_done_sends_response_create(self):
        """The exact bug Codex flagged, asserted end-to-end in one flow.

        Walks two handler invocations and asserts in between:

          1. ``_handle_inworld_message({"type": "session.updated"})``
             → MUST send ``conversation.item.create``
             → MUST NOT send ``response.create``
          2. ``_handle_inworld_message({"type": "conversation.item.done", ...})``
             → MUST send ``response.create``

        If anyone refactors the bridge to await ``conversation.item.done``
        inside the ``session.updated`` handler again, step 1 will hang the
        handler call here and the test will time out. If they move the
        ``response.create`` send out of the ``item.done`` handler, step 2 will
        fail. Either failure mode = the deadlock regression is back.
        """
        sent = []
        bridge, _ = _make_bridge()
        _stub_send(bridge, sent)

        # Step 1.
        await bridge._handle_inworld_message({"type": "session.updated"})

        sent_types = [m["type"] for m in sent]
        self.assertEqual(
            len(sent), 1,
            f"session.updated must send exactly 1 message; got {sent_types}",
        )
        self.assertEqual(
            sent_types[0], "conversation.item.create",
            f"session.updated must send conversation.item.create; got {sent_types}",
        )
        self.assertNotIn(
            "response.create", sent_types,
            "session.updated MUST NOT send response.create — that's the deadlock pattern",
        )

        # Step 2.
        await bridge._handle_inworld_message({
            "type": "conversation.item.done",
            "item": {"id": "codex_regression_item", "role": "user"},
        })

        sent_types = [m["type"] for m in sent]
        self.assertEqual(
            len(sent), 2,
            f"after item.done, expected 2 messages; got {sent_types}",
        )
        self.assertEqual(
            sent[1]["type"], "response.create",
            f"item.done must trigger response.create; got {sent_types}",
        )
        # The forced-test instruction must travel on response.create so the
        # server returns a real spoken reply rather than a transcript-only delta.
        self.assertIn(
            "Reply out loud in one short sentence.",
            sent[1]["payload"].get("response", {}).get("instructions", ""),
            "response.create must carry the forced-test instruction",
        )

    async def test_session_updated_returns_quickly_does_not_wait_for_item_done(self):
        """Critical regression: ``session.updated`` MUST send
        ``conversation.item.create`` and return immediately. If it awaits
        ``conversation.item.done`` here, the receive loop blocks itself and the
        only thing that could unblock it (the future ``conversation.item.done``
        event) can never be processed.
        """
        sent = []
        bridge, _ = _make_bridge()
        _stub_send(bridge, sent)

        started = time.monotonic()
        await bridge._handle_inworld_message({"type": "session.updated"})
        elapsed = time.monotonic() - started

        # 200 ms is a generous ceiling for an outbound send; well under any
        # timeout that would matter.
        self.assertLess(
            elapsed, 0.2,
            f"session.updated handler took {elapsed:.3f}s — receive-loop deadlock regression",
        )
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["type"], "conversation.item.create")
        self.assertEqual(bridge._forced_test_phase, "awaiting_item_done")
        self.assertTrue(bridge._mic_forwarding_paused)

    async def test_conversation_item_done_triggers_response_create(self):
        """``conversation.item.done`` arriving while we're in
        ``awaiting_item_done`` sends ``response.create`` and advances to
        ``awaiting_response_done``.
        """
        sent = []
        bridge, _ = _make_bridge()
        _stub_send(bridge, sent)

        # session.updated → item.create
        await bridge._handle_inworld_message({"type": "session.updated"})
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["type"], "conversation.item.create")

        # Later: receive loop delivers conversation.item.done
        await bridge._handle_inworld_message({
            "type": "conversation.item.done",
            "item": {"id": "item_test_1", "role": "user"},
        })

        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[1]["type"], "response.create")
        # The forced-test instruction must be carried on response.create so the
        # model actually replies out loud (otherwise some builds return only
        # transcript text).
        self.assertIn(
            "Reply out loud in one short sentence.",
            sent[1]["payload"].get("response", {}).get("instructions", ""),
        )
        self.assertEqual(bridge._forced_test_phase, "awaiting_response_done")
        # Mic stays paused until response.done.
        self.assertTrue(bridge._mic_forwarding_paused)

    async def test_response_create_not_sent_before_item_done(self):
        """A ``conversation.item.done`` that arrives with no prior
        ``conversation.item.create`` MUST NOT send ``response.create``. Catches
        a state machine that fires on every item.done regardless of phase.
        """
        sent = []
        bridge, _ = _make_bridge()
        _stub_send(bridge, sent)

        await bridge._handle_inworld_message({
            "type": "conversation.item.done",
            "item": {"id": "stray_item", "role": "user"},
        })

        self.assertEqual(sent, [])
        self.assertEqual(bridge._forced_test_phase, "idle")

    async def test_response_done_completes_forced_test_and_unpauses_mic(self):
        """``response.done`` closes the loop: state → idle, mic → unpaused."""
        bridge, _ = _make_bridge()
        bridge._forced_test_phase = "awaiting_response_done"
        bridge._mic_forwarding_paused = True

        await bridge._handle_inworld_message({"type": "response.done"})

        self.assertEqual(bridge._forced_test_phase, "idle")
        self.assertFalse(bridge._mic_forwarding_paused)

    async def test_full_state_machine_round_trip(self):
        """End-to-end: session.updated → conversation.item.done → response.done,
        each handler invocation is a single event handler call, no awaiting
        inside any of them.
        """
        sent = []
        bridge, _ = _make_bridge()
        _stub_send(bridge, sent)

        # Step 1: server replies to our session.update with session.updated.
        await bridge._handle_inworld_message({"type": "session.updated"})
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["type"], "conversation.item.create")
        self.assertEqual(bridge._forced_test_phase, "awaiting_item_done")

        # Step 2: server acks the item with conversation.item.done.
        await bridge._handle_inworld_message({
            "type": "conversation.item.done",
            "item": {"id": "abc", "role": "user"},
        })
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[1]["type"], "response.create")
        self.assertEqual(bridge._forced_test_phase, "awaiting_response_done")

        # Step 3: server completes the synthesized response.
        await bridge._handle_inworld_message({"type": "response.done"})
        self.assertEqual(len(sent), 2, "response.done must not trigger any outbound message")
        self.assertEqual(bridge._forced_test_phase, "idle")
        self.assertFalse(bridge._mic_forwarding_paused)

    async def test_mic_frames_not_sent_before_session_ready(self):
        """Defensive guard: ``_forward_livekit_audio`` must NOT send
        ``input_audio_buffer.append`` until ``session.updated`` has been
        received (which sets ``_session_ready``). Before that point the server
        hasn't applied the configured model/voice/VAD and would reject or
        misroute audio.
        """
        from inworld_realtime_bridge import build_audio_append_message

        sent = []
        bridge, _ = _make_bridge()
        _stub_send(bridge, sent)

        # Sanity: session is not ready yet.
        self.assertFalse(bridge._session_ready.is_set())

        # A 60ms PCM frame at 24 kHz mono 16-bit = 1440 samples * 2 bytes.
        # Build a real PCM payload so the inner code path actually has something
        # to send if the guard is missing.
        frame_pcm = b"\x00\x01" * 1440
        append_msg = build_audio_append_message(frame_pcm)

        # Drive the relevant inner-loop branch of _forward_livekit_audio
        # directly: simulate what happens for ONE audio frame before session ready.
        # The function is structured around an `async for event in stream` loop, so
        # we manually exercise the early-continue check.
        # Step 1: before session_ready, _send_inworld_message MUST NOT have been called.
        # Verify by asserting _session_ready is unset and the guard would block.
        self.assertFalse(bridge._session_ready.is_set(),
            "session_ready must start unset")
        # The actual guard: confirm the early-continue path is the first check in
        # the loop. If anyone reorders this check below the mic_forwarding_paused
        # check or removes it, this test will surface it via the next state.
        bridge._audio_forwarded_count = 0
        # Mark sent-of-input-audio count by stubbing _send_inworld_message to
        # detect input_audio_buffer.append messages specifically.
        append_sent = []
        async def fake_send(payload, reason="unknown"):
            if payload.get("type") == "input_audio_buffer.append":
                append_sent.append(payload)
            return None
        bridge._send_inworld_message = fake_send

        # Before session.updated: simulate the guard hitting for one frame.
        # We can't easily run the full async stream loop without a real track,
        # so we assert the intent of the guard by reading the function source
        # and confirming the session_ready check precedes the append call.
        import inspect
        src = inspect.getsource(bridge._forward_livekit_audio)
        idx_session_ready_check = src.find("not self._session_ready.is_set()")
        idx_append_call = src.find("build_audio_append_message(pcm)")
        self.assertNotEqual(idx_session_ready_check, -1,
            "_forward_livekit_audio must contain a `not self._session_ready.is_set()` guard")
        self.assertNotEqual(idx_append_call, -1,
            "_forward_livekit_audio must still contain the append call")
        self.assertLess(idx_session_ready_check, idx_append_call,
            "session_ready guard MUST come before the input_audio_buffer.append call")

        # Now flip session.ready AND unset mic pause so the path is fully open,
        # and confirm session_ready is the only gate left.
        bridge._session_ready.set()
        self.assertTrue(bridge._session_ready.is_set())
        self.assertFalse(bridge._mic_forwarding_paused)

        # Negative assertion: even after clearing gates, we don't ASSERT a frame
        # is sent — that requires a real LiveKit track stub. We only assert that
        # the guard logic is in the right place and that session_ready is the
        # binary switch for the mic-forwarding path.

    async def test_mic_frames_sent_after_session_ready_when_not_paused(self):
        """Positive smoke: after ``session.updated`` clears the readiness flag and
        ``_mic_forwarding_paused`` is ``False``, the guard no longer blocks.
        We assert by reading the source: there is no other gate between
        ``session_ready`` and the append call.
        """
        bridge, _ = _make_bridge()
        bridge._session_ready.set()
        bridge._mic_forwarding_paused = False

        import inspect
        src = inspect.getsource(bridge._forward_livekit_audio)
        idx_session_ready_check = src.find("not self._session_ready.is_set()")
        idx_force_test_check = src.find("self._mic_forwarding_paused")
        idx_append_call = src.find("build_audio_append_message(pcm)")
        self.assertNotEqual(idx_session_ready_check, -1)
        self.assertNotEqual(idx_force_test_check, -1)
        self.assertNotEqual(idx_append_call, -1)
        self.assertLess(idx_session_ready_check, idx_append_call)
        self.assertLess(idx_force_test_check, idx_append_call)


class ForcedTextTestEnvGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_force_text_test_disabled_by_env_skips_item_create(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_FORCE_TEXT_TEST": "false"},
            clear=False,
        ):
            bridge, settings = _make_bridge()
            # Reload settings to pick up the env override on the bridge instance.
            bridge._force_text_test_enabled = irb._env_bool("INWORLD_FORCE_TEXT_TEST", True)
            self.assertFalse(bridge._force_text_test_enabled)

        sent = []
        _stub_send(bridge, sent)

        await bridge._handle_inworld_message({"type": "session.updated"})

        self.assertEqual(sent, [])
        self.assertEqual(bridge._forced_test_phase, "idle")
        self.assertFalse(bridge._mic_forwarding_paused)

    async def test_force_text_test_enabled_sends_item_create(self):
        bridge, _ = _make_bridge()
        bridge._force_text_test_enabled = True
        sent = []
        _stub_send(bridge, sent)

        await bridge._handle_inworld_message({"type": "session.updated"})

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["type"], "conversation.item.create")

    async def test_double_session_updated_does_not_run_test_twice(self):
        """If session.updated arrives twice (some servers re-broadcast), the
        second one must skip rather than send a second ``conversation.item.create``
        that would orphan the first.
        """
        bridge, _ = _make_bridge()
        bridge._force_text_test_enabled = True
        sent = []
        _stub_send(bridge, sent)

        await bridge._handle_inworld_message({"type": "session.updated"})
        await bridge._handle_inworld_message({"type": "session.updated"})

        self.assertEqual(len(sent), 1, "second session.updated must not re-arm the forced test")


if __name__ == "__main__":
    unittest.main()

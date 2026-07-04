"""Public Arche API: auth gating, transport behavior, event relay scrubbing.

Pins the product rules:
  - everything refuses until ARCHE_API_ENABLED=true AND keys are configured,
  - key comparison accepts any configured key, rejects everything else,
  - the WS transport emits OpenAI-shaped audio deltas and relays events,
  - voiceProfile nodes are scrubbed from every relayed event (raw emotion
    labels never leave the server),
  - the session core relays whitelisted events through the transport and
    keeps unlisted ones server-side,
  - user_ref sanitization only admits safe stable ids.
"""

import asyncio
import base64
import types
import unittest
from unittest import mock

import arche_api
import inworld_realtime_bridge as irb


class ApiAuthTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(arche_api.api_enabled())

    def test_key_validation_requires_configured_keys(self):
        with mock.patch.dict("os.environ", {"ARCHE_API_KEYS": ""}, clear=True):
            self.assertFalse(arche_api.api_key_valid("anything"))

    def test_key_validation_accepts_any_configured_key(self):
        env = {"ARCHE_API_KEYS": "key-one, key-two"}
        with mock.patch.dict("os.environ", env, clear=True):
            self.assertTrue(arche_api.api_key_valid("key-one"))
            self.assertTrue(arche_api.api_key_valid("key-two"))
            self.assertFalse(arche_api.api_key_valid("key-three"))
            self.assertFalse(arche_api.api_key_valid(""))
            self.assertFalse(arche_api.api_key_valid(None))

    def test_key_from_bearer_header_preferred_over_query(self):
        headers = {"authorization": "Bearer from-header"}
        query = {"api_key": "from-query"}
        self.assertEqual(arche_api.api_key_from_scope(headers, query), "from-header")

    def test_key_from_query_when_no_header(self):
        self.assertEqual(arche_api.api_key_from_scope({}, {"api_key": "q"}), "q")

    def test_user_ref_sanitization(self):
        self.assertEqual(arche_api.sanitize_user_ref("user_42.a:b-c"), "user_42.a:b-c")
        self.assertIsNone(arche_api.sanitize_user_ref(""))
        self.assertIsNone(arche_api.sanitize_user_ref(None))
        self.assertIsNone(arche_api.sanitize_user_ref("has spaces"))
        self.assertIsNone(arche_api.sanitize_user_ref("x" * 200))
        self.assertIsNone(arche_api.sanitize_user_ref("semi;colon"))


class _FakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []
        self.closed_with: int | None = None
        self.accepted = False

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code: int = 1000):
        self.closed_with = code

    async def accept(self):
        self.accepted = True


class WebSocketApiTransportTests(unittest.TestCase):
    def test_write_output_pcm_emits_openai_shaped_delta(self):
        ws = _FakeWebSocket()
        transport = arche_api.WebSocketApiTransport(ws)

        async def run():
            return await transport.write_output_pcm(b"\x01\x02\x03\x04")

        units = asyncio.run(run())
        self.assertEqual(units, 1)
        self.assertEqual(len(ws.sent), 1)
        event = ws.sent[0]
        self.assertEqual(event["type"], "response.output_audio.delta")
        self.assertEqual(base64.b64decode(event["delta"]), b"\x01\x02\x03\x04")

    def test_emit_event_forwards_payload(self):
        ws = _FakeWebSocket()
        transport = arche_api.WebSocketApiTransport(ws)
        asyncio.run(transport.emit_event({"type": "session.updated"}))
        self.assertEqual(ws.sent, [{"type": "session.updated"}])

    def test_closed_transport_swallows_writes(self):
        ws = _FakeWebSocket()
        transport = arche_api.WebSocketApiTransport(ws)

        async def run():
            await transport.aclose()
            wrote = await transport.write_output_pcm(b"\x01\x02")
            await transport.emit_event({"type": "x"})
            return wrote

        self.assertEqual(asyncio.run(run()), 0)
        self.assertEqual(ws.sent, [])


class ScrubEventTests(unittest.TestCase):
    def test_voice_profile_removed_at_any_depth(self):
        payload = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hello",
            "voiceProfile": {"emotion": [{"label": "sad", "confidence": 0.9}]},
            "item": {
                "nested": [{"voice_profile": {"emotion": []}, "keep": 1}],
            },
        }
        scrubbed = irb.scrub_event_for_client(payload)
        self.assertEqual(scrubbed["transcript"], "hello")
        self.assertNotIn("voiceProfile", scrubbed)
        self.assertNotIn("voice_profile", scrubbed["item"]["nested"][0])
        self.assertEqual(scrubbed["item"]["nested"][0]["keep"], 1)
        self.assertNotIn("sad", str(scrubbed))

    def test_original_payload_not_mutated(self):
        payload = {"voiceProfile": {"emotion": []}, "keep": True}
        irb.scrub_event_for_client(payload)
        self.assertIn("voiceProfile", payload)


def _settings(**overrides):
    kwargs = dict(
        api_key="k",
        session_id="s",
        websocket_url="wss://example.test/session",
        model="openai/gpt-4o-mini",
        stt_model="assemblyai/u3-rt-pro",
        tts_model="inworld-tts-2",
        voice="Luna",
        speed=1.0,
        turn_detection_type="semantic_vad",
        turn_detection_eagerness="medium",
        turn_detection_create_response=True,
        turn_detection_interrupt_response=True,
        instructions="Be concise.",
        timeout_seconds=60.0,
        voice_profile_enabled=True,
        input_format="pcm16",
        output_format="pcm16",
        auth_scheme="basic",
        tts_delivery_mode="CREATIVE",
        tts_segmenter_strategy="full_turn",
        tts_steering_handling="emit_once",
        emotion_confidence_floor=0.5,
    )
    kwargs.update(overrides)
    return irb.InworldRealtimeSettings(**kwargs)


class _RecordingTransport(irb.ArcheTransport):
    def __init__(self):
        self.events: list[dict] = []
        self.pcm_writes: list[bytes] = []

    def bind(self, session):
        self.session = session

    async def start(self):
        return

    async def write_output_pcm(self, pcm: bytes) -> int:
        self.pcm_writes.append(pcm)
        return 1

    async def emit_event(self, payload):
        self.events.append(payload)

    async def aclose(self):
        return


class SessionEventRelayTests(unittest.TestCase):
    """The core relays whitelisted, scrubbed events through any transport."""

    def _run(self, *events):
        async def scenario():
            transport = _RecordingTransport()
            session = irb.ArcheRealtimeSession(_settings(), transport)
            session._session_ready.set()
            for event in events:
                await session._handle_inworld_message(event)
            return transport, session

        return asyncio.run(scenario())

    def test_transcription_event_relayed_and_scrubbed(self):
        transport, _ = self._run(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "hi there",
                "voiceProfile": {"emotion": [{"label": "happy", "confidence": 0.9}]},
            }
        )
        relayed = [e for e in transport.events if e["type"].endswith("transcription.completed")]
        self.assertEqual(len(relayed), 1)
        self.assertEqual(relayed[0]["transcript"], "hi there")
        self.assertNotIn("voiceProfile", relayed[0])

    def test_unlisted_event_not_relayed(self):
        transport, _ = self._run({"type": "conversation.item.added"})
        self.assertEqual(transport.events, [])

    def test_audio_delta_goes_to_pcm_path_not_event_relay(self):
        pcm = bytes(1920)
        transport, session = self._run(
            {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()}
        )
        self.assertEqual(transport.events, [])
        self.assertEqual(transport.pcm_writes, [pcm])
        self.assertEqual(session._audio_frames_written, 1)

    def test_input_pcm_dropped_until_session_ready(self):
        async def scenario():
            transport = _RecordingTransport()
            session = irb.ArcheRealtimeSession(_settings(), transport)

            class _Ws:
                def __init__(self):
                    self.sent = []

                async def send_json(self, payload):
                    self.sent.append(payload)

            ws = _Ws()
            session._ws = ws
            await session.handle_input_pcm(b"\x01\x02")  # not ready -> dropped
            session._session_ready.set()
            await session.handle_input_pcm(b"\x03\x04")  # forwarded
            return session, ws

        session, ws = asyncio.run(scenario())
        self.assertEqual(session._dropped_before_session_ready, 1)
        self.assertEqual(session._mic_frames_forwarded, 1)
        self.assertEqual(len(ws.sent), 1)
        self.assertEqual(ws.sent[0]["type"], "input_audio_buffer.append")

    def test_set_video_frame_rejects_non_image_payload(self):
        async def scenario():
            transport = _RecordingTransport()
            session = irb.ArcheRealtimeSession(_settings(), transport)
            session.set_video_frame("data:image/jpeg;base64,AAAA")
            accepted = session._latest_video_frame_data_url
            session.set_video_frame("javascript:alert(1)")
            after_bad = session._latest_video_frame_data_url
            return accepted, after_bad

        accepted, after_bad = asyncio.run(scenario())
        self.assertEqual(accepted, "data:image/jpeg;base64,AAAA")
        self.assertEqual(after_bad, "data:image/jpeg;base64,AAAA")


class HandshakeRejectionTests(unittest.TestCase):
    def test_disabled_api_closes_before_accept(self):
        ws = _FakeWebSocket()
        with mock.patch.dict("os.environ", {}, clear=True):
            asyncio.run(arche_api.handle_realtime_api_websocket(ws))
        self.assertFalse(ws.accepted)
        self.assertEqual(ws.closed_with, 4404)

    def test_bad_key_closes_before_accept(self):
        ws = _FakeWebSocket()
        ws.headers = {"authorization": "Bearer wrong"}
        ws.query_params = {}
        env = {"ARCHE_API_ENABLED": "true", "ARCHE_API_KEYS": "right"}
        with mock.patch.dict("os.environ", env, clear=True):
            asyncio.run(arche_api.handle_realtime_api_websocket(ws))
        self.assertFalse(ws.accepted)
        self.assertEqual(ws.closed_with, 4401)


class VisionMemoryTests(unittest.TestCase):
    """Seen scenes persist to memory when VISION_MEMORY_ENABLED, deduped."""

    def _vision_config(self, remember_enabled=True):
        from vision_context import VisionContextConfig

        return VisionContextConfig(
            enabled=True,
            model="m",
            api_key="k",
            min_interval_seconds=12.0,
            frame_sample_interval_seconds=2.0,
            max_frame_dimension=512,
            request_timeout_seconds=6.0,
            remember_enabled=remember_enabled,
        )

    def _run_update(self, *, remember_enabled, summaries):
        remembered = []

        class _FakeMemory:
            def schedule_remember(self, role, content, turn_id=None, **kwargs):
                remembered.append((role, content))

        async def scenario():
            transport = _RecordingTransport()
            session = irb.ArcheRealtimeSession(_settings(), transport)
            session._session_ready.set()
            session._vision_config = self._vision_config(remember_enabled=remember_enabled)
            session._memory_layer = _FakeMemory()
            session._latest_video_frame_data_url = "data:image/jpeg;base64,AAAA"
            summary_iter = iter(summaries)

            async def fake_describe(frame, config):
                return next(summary_iter)

            with mock.patch.object(irb, "describe_frame", fake_describe):
                for _ in summaries:
                    await session._run_vision_context_update()
            return session

        asyncio.run(scenario())
        return remembered

    def test_summary_remembered_once_with_observation_role(self):
        remembered = self._run_update(
            remember_enabled=True, summaries=["a red mug", "a red mug"]
        )
        self.assertEqual(remembered, [("observation", "a red mug")])

    def test_changed_summary_remembered_again(self):
        remembered = self._run_update(
            remember_enabled=True, summaries=["a red mug", "a stack of books"]
        )
        self.assertEqual(
            remembered,
            [("observation", "a red mug"), ("observation", "a stack of books")],
        )

    def test_disabled_flag_never_remembers(self):
        remembered = self._run_update(remember_enabled=False, summaries=["a red mug"])
        self.assertEqual(remembered, [])


if __name__ == "__main__":
    unittest.main()

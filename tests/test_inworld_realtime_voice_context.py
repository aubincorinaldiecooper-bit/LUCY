"""Emotional context wiring on the Inworld Realtime bridge.

Pins the product rules:
  - voiceProfile metadata is found wherever it sits inside a Realtime event
    (top level, nested item/transcription, camelCase or snake_case),
  - the latest normalized profile is stored on the bridge,
  - the session receives a partial instructions-only session.update carrying
    profile.planner_summary() plus a never-mention guard,
  - raw emotion labels NEVER appear in anything sent to Inworld,
  - unchanged summaries and rapid-fire changes don't spam session.update.

Bridge construction happens inside a running event loop (rtc.AudioSource
requires one), so each test drives an async scenario via asyncio.run().
"""

import asyncio
import time
import types
import unittest
from unittest import mock

import inworld_realtime_bridge as irb
from inworld_voice_profile import NormalizedVoiceProfile
from vision_context import VisionContextConfig


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


class _FakeWs:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


def _make_bridge():
    room = types.SimpleNamespace(
        local_participant=types.SimpleNamespace(publish_track=lambda *a, **k: asyncio.sleep(0)),
        remote_participants={},
        on=lambda *a, **k: None,
        off=lambda *a, **k: None,
    )
    bridge = irb.InworldRealtimeLiveKitBridge(room, _settings())
    bridge._ws = _FakeWs()
    return bridge


def _run_scenario(*events, session_ready=False, between=None):
    """Build a bridge in a running loop, feed it events, return it."""

    async def scenario():
        bridge = _make_bridge()
        if session_ready:
            bridge._session_ready.set()
        for i, event in enumerate(events):
            if between is not None and i > 0:
                between(bridge)
            await bridge._handle_inworld_message(event)
        return bridge

    return asyncio.run(scenario())


# A realtime transcription event carrying a high-confidence "happy" profile.
def _transcription_event_with_profile(confidence=0.9):
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item-1",
        "transcript": "hello there",
        "voiceProfile": {
            "emotion": [{"label": "happy", "confidence": confidence}],
            "pitch": [{"label": "high", "confidence": 0.8}],
            "vocalStyle": [{"label": "normal", "confidence": 0.7}],
        },
    }


def _calm_event():
    event = _transcription_event_with_profile()
    event["voiceProfile"]["emotion"] = [{"label": "calm", "confidence": 0.9}]
    return event


class FindVoiceProfileContainerTests(unittest.TestCase):
    def test_top_level_camel_case(self):
        event = _transcription_event_with_profile()
        self.assertIs(irb.find_voice_profile_container(event), event)

    def test_nested_under_item_snake_case(self):
        inner = {"voice_profile": {"emotion": [{"label": "sad", "confidence": 0.8}]}}
        event = {"type": "response.done", "response": {"output": [{"content": [inner]}]}}
        self.assertIs(irb.find_voice_profile_container(event), inner)

    def test_absent_returns_none(self):
        self.assertIsNone(irb.find_voice_profile_container({"type": "response.created"}))

    def test_non_dict_profile_value_ignored(self):
        self.assertIsNone(irb.find_voice_profile_container({"voiceProfile": "not-a-dict"}))

    def test_depth_limit(self):
        event = {"type": "x"}
        node = event
        for _ in range(10):
            node["nested"] = {}
            node = node["nested"]
        node["voiceProfile"] = {"emotion": []}
        self.assertIsNone(irb.find_voice_profile_container(event))


class CaptureAndStoreTests(unittest.TestCase):
    def test_profile_stored_on_bridge(self):
        bridge = _run_scenario(_transcription_event_with_profile())
        profile = bridge.latest_voice_profile
        self.assertIsInstance(profile, NormalizedVoiceProfile)
        # happy @0.9 maps to high energy / medium tension / high certainty.
        self.assertEqual(profile.energy, "high")
        self.assertEqual(profile.certainty, "high")
        self.assertEqual(profile.pitch, "high")
        self.assertGreater(bridge.latest_voice_profile_at, 0.0)

    def test_low_confidence_collapses_to_neutral(self):
        bridge = _run_scenario(_transcription_event_with_profile(confidence=0.2))
        profile = bridge.latest_voice_profile
        self.assertEqual(
            (profile.energy, profile.tension, profile.certainty),
            ("medium", "medium", "medium"),
        )

    def test_event_without_profile_does_not_overwrite(self):
        seen = []
        bridge = _run_scenario(
            _transcription_event_with_profile(),
            {"type": "response.created"},
            between=lambda b: seen.append(b.latest_voice_profile),
        )
        self.assertIs(bridge.latest_voice_profile, seen[0])


class VoiceContextSessionUpdateTests(unittest.TestCase):
    def test_no_send_before_session_ready(self):
        bridge = _run_scenario(_transcription_event_with_profile(), session_ready=False)
        self.assertEqual(bridge._ws.sent, [])

    def test_partial_instructions_update_sent_when_ready(self):
        bridge = _run_scenario(_transcription_event_with_profile(), session_ready=True)
        self.assertEqual(len(bridge._ws.sent), 1)
        update = bridge._ws.sent[0]
        self.assertEqual(update["type"], "session.update")
        session = update["session"]
        # Partial update: instructions only (plus the required type field) —
        # audio/model config must not be re-asserted mid-conversation.
        self.assertEqual(sorted(session.keys()), ["instructions", "type"])
        self.assertIn("Be concise.", session["instructions"])
        self.assertIn(
            bridge.latest_voice_profile.planner_summary(), session["instructions"]
        )
        self.assertIn("never mention", session["instructions"].lower())

    def test_raw_emotion_label_never_sent(self):
        bridge = _run_scenario(_transcription_event_with_profile(), session_ready=True)
        sent_text = str(bridge._ws.sent).lower()
        for label in ("happy", "sad", "angry", "fearful", "frustrated", "surprised", "tender"):
            self.assertNotIn(label, sent_text)

    def test_unchanged_summary_not_resent(self):
        bridge = _run_scenario(
            _transcription_event_with_profile(),
            _transcription_event_with_profile(),
            session_ready=True,
        )
        self.assertEqual(len(bridge._ws.sent), 1)

    def test_changed_summary_throttled_within_min_interval(self):
        # A different profile right away: change is real but inside the window.
        bridge = _run_scenario(
            _transcription_event_with_profile(),
            _calm_event(),
            session_ready=True,
        )
        self.assertEqual(len(bridge._ws.sent), 1)

    def test_changed_summary_sent_after_min_interval(self):
        def age_last_send(bridge):
            bridge._last_voice_context_sent_at = (
                time.monotonic() - irb.VOICE_CONTEXT_MIN_INTERVAL_SECONDS - 1
            )

        bridge = _run_scenario(
            _transcription_event_with_profile(),
            _calm_event(),
            session_ready=True,
            between=age_last_send,
        )
        self.assertEqual(len(bridge._ws.sent), 2)
        self.assertIn("energy low", bridge._ws.sent[1]["session"]["instructions"])

    def test_session_updated_flushes_early_capture(self):
        bridge = _run_scenario(
            _transcription_event_with_profile(),
            {"type": "session.updated"},
            session_ready=False,
        )
        self.assertEqual(len(bridge._ws.sent), 1)
        self.assertEqual(bridge._ws.sent[0]["type"], "session.update")


class BuilderTests(unittest.TestCase):
    def test_note_contains_summary_and_guard(self):
        note = irb.build_voice_context_note("energy high, tension medium, certainty high")
        self.assertIn("energy high", note)
        self.assertIn("never mention", note.lower())

    def test_session_update_builder_keeps_base_instructions(self):
        update = irb.build_voice_context_session_update(_settings(), "energy low, tension low, certainty medium")
        self.assertEqual(update["session"]["type"], "realtime")
        self.assertTrue(update["session"]["instructions"].startswith("Be concise."))
        self.assertIn("energy low", update["session"]["instructions"])

    def test_image_conversation_item_carries_image_and_text_parts(self):
        item = irb.build_image_conversation_item_create(
            "data:image/png;base64,AAAA", "What do you see?"
        )
        self.assertEqual(item["type"], "conversation.item.create")
        self.assertEqual(item["item"]["type"], "message")
        self.assertEqual(item["item"]["role"], "user")
        content = item["item"]["content"]
        self.assertEqual(
            content[0], {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        )
        self.assertEqual(content[1], {"type": "input_text", "text": "What do you see?"})

    def test_handshake_failure_reason_maps_billing_and_auth_statuses(self):
        self.assertEqual(irb.inworld_handshake_failure_reason(402), "payment_required")
        self.assertEqual(irb.inworld_handshake_failure_reason(401), "unauthorized")
        self.assertEqual(irb.inworld_handshake_failure_reason(403), "forbidden")
        self.assertEqual(irb.inworld_handshake_failure_reason(500), "handshake_rejected")

    def test_session_update_builder_includes_vision_note(self):
        update = irb.build_instructions_session_update(
            _settings(), vision_summary="a desk with a laptop"
        )
        instructions = update["session"]["instructions"]
        self.assertTrue(instructions.startswith("Be concise."))
        self.assertIn("a desk with a laptop", instructions)
        self.assertIn("never mention", instructions.lower())

    def test_session_update_builder_composes_vision_with_voice_context(self):
        update = irb.build_instructions_session_update(
            _settings(),
            voice_context_summary="energy low, tension low, certainty medium",
            vision_summary="a desk with a laptop",
        )
        instructions = update["session"]["instructions"]
        self.assertIn("energy low", instructions)
        self.assertIn("a desk with a laptop", instructions)

    def test_session_update_builder_includes_mood_note(self):
        update = irb.build_instructions_session_update(
            _settings(), mood_summary="downplaying it, sounds worn out"
        )
        instructions = update["session"]["instructions"]
        self.assertTrue(instructions.startswith("Be concise."))
        self.assertIn("downplaying it, sounds worn out", instructions)
        self.assertIn("never", instructions.lower())


def _vision_config(**overrides):
    kwargs = dict(
        enabled=True,
        model="openai/gpt-4o-mini",
        api_key="or-key",
        min_interval_seconds=12.0,
        frame_sample_interval_seconds=2.0,
        max_frame_dimension=512,
        request_timeout_seconds=6.0,
    )
    kwargs.update(overrides)
    return VisionContextConfig(**kwargs)


_SPEECH_STARTED = {"type": "input_audio_buffer.speech_started"}


class VisionContextBridgeTests(unittest.TestCase):
    """speech_started → background describe → instructions update, all gated."""

    def _drain_tasks(self, bridge):
        async def drain():
            pending = [t for t in bridge._tasks if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return drain()

    def _scenario(self, *, config, frame="data:image/jpeg;base64,AAAA", describe=None, events=(_SPEECH_STARTED,), between=None):
        async def run():
            bridge = _make_bridge()
            bridge._session_ready.set()
            bridge._vision_config = config
            bridge._latest_video_frame_data_url = frame
            fake_describe = describe or mock.AsyncMock(return_value="a red mug on a desk")
            with mock.patch.object(irb, "describe_frame", fake_describe):
                for i, event in enumerate(events):
                    if between is not None and i > 0:
                        between(bridge)
                    await bridge._handle_inworld_message(event)
                    # Let the background describe task finish before the next
                    # event — in production consecutive speech_started events
                    # are min_interval_seconds apart, far beyond the call
                    # timeout, so the previous call is never still in flight.
                    await self._drain_tasks(bridge)
            return bridge, fake_describe

        return asyncio.run(run())

    def test_disabled_config_never_calls_describe(self):
        bridge, fake_describe = self._scenario(config=_vision_config(enabled=False))
        fake_describe.assert_not_awaited()
        self.assertEqual(bridge._ws.sent, [])

    def test_no_held_frame_never_calls_describe(self):
        bridge, fake_describe = self._scenario(config=_vision_config(), frame=None)
        fake_describe.assert_not_awaited()
        self.assertEqual(bridge._ws.sent, [])

    def test_speech_started_sends_vision_instructions_update(self):
        bridge, fake_describe = self._scenario(config=_vision_config())
        fake_describe.assert_awaited_once()
        self.assertEqual(bridge.latest_vision_summary, "a red mug on a desk")
        self.assertEqual(len(bridge._ws.sent), 1)
        update = bridge._ws.sent[0]
        self.assertEqual(update["type"], "session.update")
        self.assertIn("a red mug on a desk", update["session"]["instructions"])

    def test_failed_describe_sends_nothing(self):
        bridge, _ = self._scenario(
            config=_vision_config(), describe=mock.AsyncMock(return_value=None)
        )
        self.assertIsNone(bridge.latest_vision_summary)
        self.assertEqual(bridge._ws.sent, [])

    def test_second_speech_started_within_interval_is_throttled(self):
        bridge, fake_describe = self._scenario(
            config=_vision_config(), events=(_SPEECH_STARTED, _SPEECH_STARTED)
        )
        fake_describe.assert_awaited_once()
        self.assertEqual(len(bridge._ws.sent), 1)

    def test_second_speech_started_after_interval_sends_again(self):
        def age_last_send(bridge):
            bridge._last_vision_context_sent_at = (
                time.monotonic() - bridge._vision_config.min_interval_seconds - 1
            )

        summaries = iter(["a red mug on a desk", "a person holding a book"])

        async def next_summary(*args, **kwargs):
            return next(summaries)

        bridge, _ = self._scenario(
            config=_vision_config(),
            describe=mock.AsyncMock(side_effect=next_summary),
            events=(_SPEECH_STARTED, _SPEECH_STARTED),
            between=age_last_send,
        )
        self.assertEqual(len(bridge._ws.sent), 2)
        self.assertIn("a person holding a book", bridge._ws.sent[1]["session"]["instructions"])

    def test_video_sampler_not_started_when_disabled(self):
        async def run():
            bridge = _make_bridge()
            # Default env: disabled. A video track must be ignored entirely.
            bridge._vision_config = _vision_config(enabled=False)
            track = types.SimpleNamespace(kind=irb.rtc.TrackKind.KIND_VIDEO)
            bridge.transport._maybe_start_video_sampler(track)
            return bridge

        bridge = asyncio.run(run())
        self.assertFalse(bridge.transport._video_sampler_started)
        self.assertEqual(len(bridge.transport._tasks), 0)

    def test_audio_track_never_starts_video_sampler(self):
        async def run():
            bridge = _make_bridge()
            bridge._vision_config = _vision_config()
            track = types.SimpleNamespace(kind=irb.rtc.TrackKind.KIND_AUDIO)
            bridge.transport._maybe_start_video_sampler(track)
            return bridge

        bridge = asyncio.run(run())
        self.assertFalse(bridge.transport._video_sampler_started)


class _FakeVideoFrame:
    """Stands in for rtc.VideoFrame: convert() returns RGBA self-description."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def convert(self, buffer_type):
        return types.SimpleNamespace(
            width=self.width,
            height=self.height,
            data=bytes(self.width * self.height * 4),  # opaque-black RGBA
        )


class EncodeVideoFrameTests(unittest.TestCase):
    def test_encodes_to_jpeg_data_url(self):
        encoded = irb._encode_video_frame_jpeg(_FakeVideoFrame(64, 48), max_dimension=512)
        self.assertIsNotNone(encoded)
        self.assertTrue(encoded.startswith("data:image/jpeg;base64,"))

    def test_downscales_to_max_dimension(self):
        import base64 as b64
        import io as io_

        from PIL import Image

        encoded = irb._encode_video_frame_jpeg(_FakeVideoFrame(1280, 720), max_dimension=512)
        raw = b64.b64decode(encoded.split(",", 1)[1])
        image = Image.open(io_.BytesIO(raw))
        self.assertEqual(max(image.width, image.height), 512)

    def test_bad_frame_returns_none_never_raises(self):
        class Broken:
            def convert(self, buffer_type):
                raise RuntimeError("no buffer")

        self.assertIsNone(irb._encode_video_frame_jpeg(Broken(), max_dimension=512))


def _mood_config(**overrides):
    from mood_context import MoodContextConfig

    kwargs = dict(
        enabled=True,
        model="m",
        api_key="k",
        min_interval_seconds=8.0,
        request_timeout_seconds=5.0,
        max_transcript_chars=600,
    )
    kwargs.update(overrides)
    return MoodContextConfig(**kwargs)


def _user_transcript_event(text="I'm okay I guess"):
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": text,
    }


class MoodContextBridgeTests(unittest.TestCase):
    """completed user transcript → background mood fusion → instructions update."""

    def _drain_tasks(self, bridge):
        async def drain():
            pending = [t for t in bridge._tasks if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return drain()

    def _scenario(self, *, config, describe=None, events=None, between=None):
        events = events or (_user_transcript_event(),)

        async def run():
            bridge = _make_bridge()
            bridge._session_ready.set()
            bridge._mood_config = config
            fake_describe = describe or mock.AsyncMock(
                return_value="says she's okay but sounds like she's holding back"
            )
            with mock.patch.object(irb, "describe_mood", fake_describe):
                for i, event in enumerate(events):
                    if between is not None and i > 0:
                        between(bridge)
                    await bridge._handle_inworld_message(event)
                    await self._drain_tasks(bridge)
            return bridge, fake_describe

        return asyncio.run(run())

    def test_disabled_config_never_calls_describe(self):
        bridge, fake_describe = self._scenario(config=_mood_config(enabled=False))
        fake_describe.assert_not_awaited()
        self.assertEqual(bridge._ws.sent, [])

    def test_completed_transcript_sends_mood_instructions_update(self):
        bridge, fake_describe = self._scenario(config=_mood_config())
        fake_describe.assert_awaited_once()
        self.assertEqual(
            bridge.latest_mood_summary, "says she's okay but sounds like she's holding back"
        )
        self.assertEqual(len(bridge._ws.sent), 1)
        update = bridge._ws.sent[0]
        self.assertEqual(update["type"], "session.update")
        self.assertIn("holding back", update["session"]["instructions"])

    def test_none_read_sends_nothing(self):
        bridge, _ = self._scenario(
            config=_mood_config(), describe=mock.AsyncMock(return_value=None)
        )
        self.assertIsNone(bridge.latest_mood_summary)
        self.assertEqual(bridge._ws.sent, [])

    def test_second_transcript_within_interval_is_throttled(self):
        bridge, fake_describe = self._scenario(
            config=_mood_config(),
            events=(_user_transcript_event("a"), _user_transcript_event("b")),
        )
        fake_describe.assert_awaited_once()
        self.assertEqual(len(bridge._ws.sent), 1)

    def test_second_transcript_after_interval_sends_again(self):
        def age_last_send(bridge):
            bridge._last_mood_context_sent_at = (
                time.monotonic() - bridge._mood_config.min_interval_seconds - 1
            )

        reads = iter(["first read", "second read"])

        async def next_read(*args, **kwargs):
            return next(reads)

        bridge, _ = self._scenario(
            config=_mood_config(),
            describe=mock.AsyncMock(side_effect=next_read),
            events=(_user_transcript_event("a"), _user_transcript_event("b")),
            between=age_last_send,
        )
        self.assertEqual(len(bridge._ws.sent), 2)
        self.assertIn("second read", bridge._ws.sent[1]["session"]["instructions"])

    def test_vocal_dials_paired_with_words_when_fresh(self):
        captured = {}

        async def capture(transcript, vocal_summary, config):
            captured["transcript"] = transcript
            captured["vocal_summary"] = vocal_summary
            return None

        async def run():
            bridge = _make_bridge()
            bridge._session_ready.set()
            bridge._mood_config = _mood_config()
            bridge.latest_voice_profile = NormalizedVoiceProfile(energy="low", tension="high")
            bridge.latest_voice_profile_at = time.monotonic()  # fresh
            with mock.patch.object(irb, "describe_mood", capture):
                await bridge._handle_inworld_message(_user_transcript_event("I'm fine"))
                await self._drain_tasks(bridge)

        asyncio.run(run())
        self.assertEqual(captured["transcript"], "I'm fine")
        self.assertIn("energy low", captured["vocal_summary"])
        self.assertIn("tension high", captured["vocal_summary"])


if __name__ == "__main__":
    unittest.main()

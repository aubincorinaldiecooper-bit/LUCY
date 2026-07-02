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

import inworld_realtime_bridge as irb
from inworld_voice_profile import NormalizedVoiceProfile


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


if __name__ == "__main__":
    unittest.main()

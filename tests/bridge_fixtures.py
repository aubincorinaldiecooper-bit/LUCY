"""Shared fixtures for the Inworld Realtime bridge test files.

The InworldRealtimeSettings factory below was copied nearly verbatim in three
files (test_inworld_realtime_voice_context, test_emotional_dataset,
test_arche_api) — 27 keyword defaults each, meaning any new required settings
field forced three synchronized edits. Same bare-name import mechanism as
vision_fixtures.py; not named test_*.py, so pytest never collects it.
"""

import inworld_realtime_bridge as irb


def make_realtime_settings(**overrides):
    """An InworldRealtimeSettings with every field defaulted for tests."""
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


class FakeWs:
    """Records session.update / conversation payloads the bridge sends."""

    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)

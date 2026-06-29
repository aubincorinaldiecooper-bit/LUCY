import base64
import os
import sys
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

    rtc = types.SimpleNamespace(
        AudioFrame=AudioFrame,
        AudioSource=lambda *args, **kwargs: types.SimpleNamespace(aclose=lambda: None),
        LocalAudioTrack=types.SimpleNamespace(create_audio_track=lambda *args, **kwargs: object()),
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
import inworld_realtime_bridge as irb


class InworldRealtimeConfigTests(unittest.TestCase):
    def test_settings_use_basic_auth_and_realtime_protocol(self):
        with mock.patch.dict(
            os.environ,
            {
                "INWORLD_API_KEY": "base64-api-key",
                "INWORLD_REALTIME_SESSION_ID": "session-123",
                "INWORLD_REALTIME_WS_URL": "wss://example.test/realtime/session",
            },
            clear=False,
        ):
            settings = irb.load_inworld_realtime_settings(instructions="Be concise.")

        self.assertEqual(settings.auth_headers["Authorization"], "Basic base64-api-key")
        self.assertEqual(
            settings.connection_url,
            "wss://example.test/realtime/session?key=session-123&protocol=realtime",
        )
        self.assertEqual(settings.instructions, "Be concise.")

    def test_session_update_replaces_stt_tts_and_vad_with_inworld(self):
        settings = irb.InworldRealtimeSettings(
            api_key="k",
            session_id="s",
            websocket_url="wss://example.test/session",
            model="openai/gpt-4o-mini",
            stt_model="inworld/inworld-stt-1",
            tts_model="inworld-tts-2",
            voice="Dennis",
            speed=1.0,
            turn_detection_type="semantic_vad",
            turn_detection_eagerness="medium",
            instructions="Talk naturally.",
            timeout_seconds=60.0,
            voice_profile_enabled=True,
            input_format="pcm16",
            output_format="pcm16",
        )

        update = irb.build_session_update(settings)
        session = update["session"]

        self.assertEqual(update["type"], "session.update")
        self.assertEqual(session["audio"]["input"]["transcription"]["model"], "inworld/inworld-stt-1")
        self.assertEqual(session["audio"]["input"]["turn_detection"]["type"], "semantic_vad")
        self.assertTrue(session["audio"]["input"]["turn_detection"]["create_response"])
        self.assertTrue(session["audio"]["input"]["turn_detection"]["interrupt_response"])
        self.assertEqual(session["audio"]["output"]["model"], "inworld-tts-2")
        self.assertEqual(session["audio"]["output"]["voice"], "Dennis")
        self.assertTrue(session["providerData"]["stt"]["voice_profile"])

    def test_audio_append_message_is_base64_realtime_event(self):
        msg = irb.build_audio_append_message(b"\x01\x02")
        self.assertEqual(msg["type"], "input_audio_buffer.append")
        self.assertEqual(base64.b64decode(msg["audio"]), b"\x01\x02")


if __name__ == "__main__":
    unittest.main()

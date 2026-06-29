"""Smoke + structural tests for ``inworld_realtime_bridge``.


These tests don't connect to Inworld; they pin down the shape of the WebSocket

messages we send so a future refactor can't accidentally regress to the broken

``content_type: input_text`` shape that triggered server-side parse errors.

"""

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

import inworld_realtime_bridge as irb  # noqa: E402



class InworldRealtimeSettingsTests(unittest.TestCase):

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


    def test_settings_use_bearer_auth_when_scheme_is_bearer(self):

        with mock.patch.dict(

            os.environ,

            {

                "INWORLD_API_KEY": "jwt-token",

                "INWORLD_REALTIME_SESSION_ID": "session-123",

                "INWORLD_AUTH_SCHEME": "bearer",

            },

            clear=False,

        ):

            settings = irb.load_inworld_realtime_settings()


        self.assertEqual(settings.auth_headers["Authorization"], "Bearer jwt-token")

        self.assertEqual(settings.auth_scheme, "bearer")


    def test_default_voice_is_luna(self):

        with mock.patch.dict(

            os.environ,

            {"INWORLD_API_KEY": "key", "INWORLD_REALTIME_SESSION_ID": "s"},

            clear=False,

        ):

            settings = irb.load_inworld_realtime_settings()

        self.assertEqual(settings.voice, "Luna")


    def test_voice_profile_disabled_by_default(self):

        with mock.patch.dict(

            os.environ,

            {"INWORLD_API_KEY": "key", "INWORLD_REALTIME_SESSION_ID": "s"},

            clear=False,

        ):

            settings = irb.load_inworld_realtime_settings()

        self.assertFalse(settings.voice_profile_enabled)



class InworldRealtimeMessageShapeTests(unittest.TestCase):

    """Pin down the WebSocket message shapes so a future refactor can't break

    the OpenAI-ReaAPI-compatible contract Inworld expects."""


    def test_session_update_replaces_stt_tts_and_vad_with_inworld(self):

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

            instructions="Talk naturally.",

            timeout_seconds=60.0,

            voice_profile_enabled=False,

            input_format="pcm16",

            output_format="pcm16",

            auth_scheme="basic",

        )


        update = irb.build_session_update(settings)

        session = update["session"]


        self.assertEqual(update["type"], "session.update")

        self.assertEqual(

            session["audio"]["input"]["transcription"]["model"],

            "inworld/inworld-stt-1",

        )

        self.assertEqual(session["audio"]["input"]["turn_detection"]["type"], "semantic_vad")

        self.assertTrue(session["audio"]["input"]["turn_detection"]["create_response"])

        self.assertTrue(session["audio"]["input"]["turn_detection"]["interrupt_response"])

        self.assertEqual(session["audio"]["output"]["model"], "inworld-tts-2")

        self.assertEqual(session["audio"]["output"]["voice"], "Luna")

        self.assertFalse(session["providerData"]["stt"]["voice_profile"])


    def test_session_update_input_output_format_pcm16(self):

        with mock.patch.dict(

            os.environ,

            {"INWORLD_API_KEY": "k", "INWORLD_REALTIME_SESSION_ID": "s"},

            clear=False,

        ):

            settings = irb.load_inworld_realtime_settings()

        update = irb.build_session_update(settings)

        session = update["session"]

        self.assertEqual(session["audio"]["input"]["format"]["type"], "pcm16")

        self.assertEqual(session["audio"]["output"]["format"]["type"], "pcm16")


    def test_audio_append_message_is_base64_realtime_event(self):

        msg = irb.build_audio_append_message(b"\x01\x02")

        self.assertEqual(msg["type"], "input_audio_buffer.append")

        self.assertEqual(base64.b64decode(msg["audio"]), b"\x01\x02")


    def test_conversation_item_create_message(self):

        """The shape MUST match OpenAI Realtime / ReaAPI: ``item.content`` is an

        array of content parts with ``type: input_text``."""

        msg = irb.build_conversation_item_create("Say hello.")

        self.assertEqual(msg["type"], "conversation.item.create")

        self.assertEqual(msg["item"]["type"], "message")

        self.assertEqual(msg["item"]["role"], "user")

        # array of content parts, not a single dict

        self.assertIsInstance(msg["item"]["content"], list)

        self.assertEqual(len(msg["item"]["content"]), 1)

        part = msg["item"]["content"][0]

        self.assertEqual(part["type"], "input_text")

        self.assertNotIn("content_type", part)  # the bug this fixes

        self.assertEqual(part["text"], "Say hello.")


    def test_response_create_message(self):

        msg = irb.build_response_create()

        self.assertEqual(msg["type"], "response.create")

        self.assertIn("audio", msg["response"]["output_modalities"])

        self.assertIn("text", msg["response"]["output_modalities"])



class InworldAudioExtractionTests(unittest.TestCase):

    def setUp(self):

        # 400 bytes of PCM, well above the 80-byte real-audio threshold.

        self.pcm = b"\x00\x01" * 200

        self.b64 = base64.b64encode(self.pcm).decode()


    def test_top_level_delta(self):

        self.assertEqual(irb._event_audio_bytes({"delta": self.b64}), self.pcm)


    def test_top_level_audio(self):

        self.assertEqual(irb._event_audio_bytes({"audio": self.b64}), self.pcm)


    def test_nested_content(self):

        self.assertEqual(

            irb._event_audio_bytes({"content": [{"delta": self.b64}]}),

            self.pcm,

        )


    def test_nested_item_content(self):

        self.assertEqual(

            irb._event_audio_bytes({"item": {"content": [{"audio": self.b64}]}}),

            self.pcm,

        )



if __name__ == "__main__":

    unittest.main()


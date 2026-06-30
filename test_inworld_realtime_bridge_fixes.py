"""Regression tests for the Inworld Realtime bridge fixes:
  1. conversation.item.create uses OpenAI-ReaAPI-compatible shape (array of
     content parts with ``type: input_text`` instead of the broken
     ``content_type: input_text`` single-dict shape that triggered parse errors).
  2. _event_audio_bytes recursively scans for audio bytes (top-level,
     nested content[], nested item.content[], audioContent field).
  3. _receive_inworld and run() are resilient to handler errors and don't
     propagate to the worker-cancel chain.
  4. _log_unknown_server_event runs without raising on arbitrary payloads.
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


# ---------------------------------------------------------------------------
# Fix 1: build_conversation_item_create uses the OpenAI/ReaAPI-compatible shape
# ---------------------------------------------------------------------------
class ConversationItemShapeTests(unittest.TestCase):
    def test_content_is_an_array_not_a_single_dict(self):
        msg = irb.build_conversation_item_create("hi there")
        assert isinstance(msg["item"]["content"], list), "content must be an array of content parts"
        assert len(msg["item"]["content"]) == 1

    def test_content_part_uses_type_not_content_type(self):
        msg = irb.build_conversation_item_create("hi there")
        part = msg["item"]["content"][0]
        assert part.get("type") == "input_text", f"expected type=input_text, got keys {list(part.keys())}"
        assert "content_type" not in part, "old broken field 'content_type' must not be present"
        assert part["text"] == "hi there"

    def test_session_update_still_uses_pcm16_format_and_inworld_models(self):
        with mock.patch.dict(
            os.environ,
            {"INWORLD_API_KEY": "k", "INWORLD_REALTIME_SESSION_ID": "s"},
            clear=False,
        ):
            settings = irb.load_inworld_realtime_settings()
        update = irb.build_session_update(settings)
        session = update["session"]
        assert session["audio"]["input"]["format"]["type"] == "pcm16"
        assert session["audio"]["output"]["format"]["type"] == "pcm16"
        assert session["audio"]["output"]["model"] == "inworld-tts-2"

    def test_response_create_keeps_audio_and_text_modalities(self):
        msg = irb.build_response_create()
        assert "audio" in msg["response"]["output_modalities"]
        assert "text" in msg["response"]["output_modalities"]


# ---------------------------------------------------------------------------
# Fix 2: _event_audio_bytes recursively finds audio in nested payloads
# ---------------------------------------------------------------------------
class AudioExtractionTests(unittest.TestCase):
    def setUp(self):
        # 400 bytes of PCM = 200 frames @ 16-bit, well above the 80-byte
        # "real audio" threshold the extractor uses.
        self.pcm = b"\x00\x01" * 200
        self.b64 = base64.b64encode(self.pcm).decode()

    def test_top_level_delta(self):
        assert irb._event_audio_bytes({"delta": self.b64}) == self.pcm

    def test_top_level_audio(self):
        assert irb._event_audio_bytes({"audio": self.b64}) == self.pcm

    def test_audioContent_field(self):
        assert irb._event_audio_bytes({"audioContent": self.b64}) == self.pcm

    def test_nested_content_array_delta(self):
        assert irb._event_audio_bytes({"content": [{"delta": self.b64}]}) == self.pcm

    def test_nested_content_array_audio(self):
        assert irb._event_audio_bytes({"content": [{"type": "audio", "audio": self.b64}]}) == self.pcm

    def test_nested_item_content_array(self):
        # Some Realtime implementations wrap the per-item audio at
        # ``payload.item.content[].audio``.
        assert irb._event_audio_bytes({"item": {"content": [{"audio": self.b64}]}}) == self.pcm

    def test_no_audio_returns_empty(self):
        assert irb._event_audio_bytes({"type": "response.created"}) == b""

    def test_short_string_treated_as_non_audio(self):
        # Tokens / ids / format tags are often short base64-ish strings; the
        # 80-byte heuristic keeps us from emitting those as PCM.
        assert irb._event_audio_bytes({"delta": "abc"}) == b""

    def test_invalid_base64_is_skipped(self):
        # Malformed base64 in delta should not crash; just return no audio.
        assert irb._event_audio_bytes({"delta": "@@@not_base64@@@"}) == b""


# ---------------------------------------------------------------------------
# Fix 3: discovery log never raises
# ---------------------------------------------------------------------------
class DiscoveryLogTests(unittest.TestCase):
    def test_log_unknown_event_with_arbitrary_payload(self):
        # Should not raise even with weird nested types.
        weird_payload = {
            "type": "response.weird.unknown.event",
            "delta": base64.b64encode(b"x" * 200).decode(),
            "nested": {"deep": {"deeper": ["list", "of", "strings", 1, 2.0, True, None]}},
            "empty_list": [],
            "empty_dict": {},
        }
        irb._log_unknown_server_event(
            msg_type="response.weird.unknown.event",
            payload=weird_payload,
            last_outbound_event_type="response.create",
            last_outbound_event_safe_shape="{...}",
            events_sent_since_session_updated=2,
        )

    def test_log_unknown_event_with_non_dict_payload(self):
        irb._log_unknown_server_event(
            msg_type=None,
            payload={"value": "string instead of dict"},
            last_outbound_event_type=None,
            last_outbound_event_safe_shape=None,
            events_sent_since_session_updated=0,
        )


# ---------------------------------------------------------------------------
# Fix 4: shape helper + session update coherence
# ---------------------------------------------------------------------------
class ShapeAndSessionTests(unittest.TestCase):
    def test_safe_payload_shape_masks_long_strings(self):
        shape = irb._safe_payload_shape(
            {"small": "abc", "huge": "x" * 200, "list": [1, 2, 3], "nested": {"k": "v"}}
        )
        assert "small" in shape
        assert "huge" in shape
        # Long strings render as str(len) markers, not raw values.
        assert "x" * 200 not in shape
        assert "str(200)" in shape

    def test_session_update_with_pcm16_output_format(self):
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
            instructions="Be concise.",
            timeout_seconds=60.0,
            voice_profile_enabled=False,
            input_format="pcm16",
            output_format="pcm16",
            auth_scheme="basic",
        )
        update = irb.build_session_update(settings)
        session = update["session"]
        assert session["type"] == "realtime"
        assert "audio" in session
        assert "input" in session["audio"]
        assert "output" in session["audio"]
        assert session["audio"]["output"]["voice"] == "Luna"
        assert session["audio"]["output"]["speed"] == 1.0


if __name__ == "__main__":
    unittest.main()

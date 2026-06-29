"""LiveKit ⇄ Inworld Realtime speech-to-speech bridge.

This path is selected with VOICE_ENGINE=inworld_realtime and keeps LiveKit as
Arche's browser media transport while Inworld owns STT, TTS, and semantic VAD / turn
detection. The existing cascaded LiveKit AgentSession remains available by leaving
VOICE_ENGINE unset or set to current.

Architecture:
  LiveKit mic/audio input
    -> livekit_to_inworld_audio_loop
    -> Inworld WebSocket
  Inworld WebSocket
    -> inworld_to_livekit_receive_loop
    -> decode output audio
    -> write PCM frames to LiveKit AudioSource
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp
from livekit import rtc

logger = logging.getLogger(__name__)

INWORLD_INPUT_SAMPLE_RATE = 24000
INWORLD_OUTPUT_SAMPLE_RATE = 24000
INWORLD_CHANNELS = 1
INWORLD_FRAME_MS = 60


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def inworld_realtime_debug() -> bool:
    return _env_bool("INWORLD_REALTIME_DEBUG", False)


def inworld_realtime_log_audio() -> bool:
    return _env_bool("INWORLD_REALTIME_LOG_AUDIO", False)


def inworld_realtime_session_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.getenv("INWORLD_REALTIME_SESSION_TIMEOUT_SECONDS", "1800")))
    except Exception:
        return 1800.0


@dataclass(frozen=True)
class InworldRealtimeSettings:
    api_key: str
    session_id: str
    websocket_url: str
    model: str
    stt_model: str
    tts_model: str
    voice: str
    speed: float
    turn_detection_type: str
    turn_detection_eagerness: str
    turn_detection_create_response: bool
    turn_detection_interrupt_response: bool
    instructions: str
    timeout_seconds: float
    voice_profile_enabled: bool
    input_format: str
    output_format: str
    auth_scheme: str

    @property
    def connection_url(self) -> str:
        query = {"key": self.session_id, "protocol": "realtime"}
        separator = "&" if "?" in self.websocket_url else "?"
        return f"{self.websocket_url}{separator}{urlencode(query)}"

    @property
    def auth_headers(self) -> dict[str, str]:
        # Inworld server-side WebSocket auth uses the Portal API key directly as
        # an already-base64-encoded Basic credential.
        if self.auth_scheme.lower() == "bearer":
            return {"Authorization": f"Bearer {self.api_key}"}
        else:
            return {"Authorization": f"Basic {self.api_key}"}


def load_inworld_realtime_settings(*, instructions: str | None = None) -> InworldRealtimeSettings:
    api_key = (os.getenv("INWORLD_API_KEY") or "").strip()
    session_id = (os.getenv("INWORLD_REALTIME_SESSION_ID") or os.getenv("LIVEKIT_ROOM_NAME") or f"lucy-{int(time.time() * 1000)}").strip()
    if not api_key:
        raise RuntimeError("VOICE_ENGINE=inworld_realtime requires INWORLD_API_KEY")
    if not session_id:
        raise RuntimeError("VOICE_ENGINE=inworld_realtime requires INWORLD_REALTIME_SESSION_ID or a room-derived fallback")

    auth_scheme = (os.getenv("INWORLD_AUTH_SCHEME") or "basic").strip().lower()
    logger.info(
        "inworld_auth_config raw_INWORLD_AUTH_SCHEME=%s inworld_auth_mode=%s",
        auth_scheme,
        "bearer_jwt" if auth_scheme == "bearer" else "basic_base64_api_key",
    )

    return InworldRealtimeSettings(
        api_key=api_key,
        session_id=session_id,
        websocket_url=(os.getenv("INWORLD_REALTIME_WS_URL") or "wss://api.inworld.ai/api/v1/realtime/session").strip(),
        model=(os.getenv("INWORLD_REALTIME_MODEL") or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip(),
        stt_model=(os.getenv("INWORLD_MODEL_ID") or os.getenv("INWORLD_STT_MODEL_ID") or "inworld/inworld-stt-1").strip(),
        tts_model=(os.getenv("INWORLD_TTS_MODEL") or "inworld-tts-2").strip(),
        voice=(os.getenv("INWORLD_TTS_VOICE") or "Luna").strip(),
        speed=float(os.getenv("INWORLD_TTS_SPEED", "1.0") or "1.0"),
        turn_detection_type=(os.getenv("INWORLD_TURN_DETECTION_TYPE") or "semantic_vad").strip(),
        turn_detection_eagerness=(os.getenv("INWORLD_TURN_DETECTION_EAGERNESS") or "medium").strip(),
        turn_detection_create_response=_env_bool("INWORLD_TURN_DETECTION_CREATE_RESPONSE", True),
        turn_detection_interrupt_response=_env_bool("INWORLD_TURN_DETECTION_INTERRUPT_RESPONSE", True),
        instructions=(instructions or os.getenv("INWORLD_REALTIME_INSTRUCTIONS") or "You are a concise, warm voice assistant.").strip(),
        timeout_seconds=inworld_realtime_session_timeout_seconds(),
        voice_profile_enabled=_env_bool("INWORLD_VOICE_PROFILE_ENABLED", False),
        input_format=(os.getenv("INWORLD_REALTIME_INPUT_FORMAT") or "pcm16").strip(),
        output_format=(os.getenv("INWORLD_REALTIME_OUTPUT_FORMAT") or "pcm16").strip(),
        auth_scheme=auth_scheme,
    )


def build_session_update(settings: InworldRealtimeSettings) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": settings.model,
            "instructions": settings.instructions,
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "format": {"type": settings.input_format, "sample_rate": INWORLD_INPUT_SAMPLE_RATE},
                    "transcription": {"model": settings.stt_model},
                    "turn_detection": {
                        "type": settings.turn_detection_type,
                        "eagerness": settings.turn_detection_eagerness,
                        "create_response": settings.turn_detection_create_response,
                        "interrupt_response": settings.turn_detection_interrupt_response,
                    },
                },
                "output": {
                    "format": {"type": settings.output_format, "sample_rate": INWORLD_OUTPUT_SAMPLE_RATE},
                    "model": settings.tts_model,
                    "voice": settings.voice,
                    "speed": settings.speed,
                },
            },
            "providerData": {
                "stt": {"voice_profile": settings.voice_profile_enabled},
            },
        },
    }


def build_conversation_item_create(text: str) -> dict[str, Any]:
    """Build a forced text test message."""
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": {
                "content_type": "input_text",
                "text": text,
            },
        },
    }


def build_response_create() -> dict[str, Any]:
    """Build a response.create message."""
    return {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio", "text"],
        },
    }


def build_audio_append_message(pcm: bytes) -> dict[str, str]:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}


def _frame_bytes(frame: rtc.AudioFrame) -> bytes:
    data = getattr(frame, "data", b"")
    try:
        return bytes(data)
    except Exception:
        return b""


def _event_audio_bytes(payload: dict[str, Any]) -> bytes:
    data = payload.get("delta") or payload.get("audio") or payload.get("data")
    if not isinstance(data, str) or not data:
        return b""
    try:
        return base64.b64decode(data)
    except Exception:
        return b""


def _iter_pcm_frames(pcm: bytes, *, sample_rate: int = INWORLD_OUTPUT_SAMPLE_RATE, channels: int = INWORLD_CHANNELS):
    bytes_per_sample = 2
    samples_per_channel = max(1, int(sample_rate * INWORLD_FRAME_MS / 1000))
    frame_size = samples_per_channel * channels * bytes_per_sample
    for offset in range(0, len(pcm), frame_size):
        chunk = pcm[offset : offset + frame_size]
        if len(chunk) < frame_size:
            chunk = chunk + (b"\x00" * (frame_size - len(chunk)))
        yield rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=samples_per_channel,
        )


class InworldRealtimeLiveKitBridge:
    def __init__(self, room: rtc.Room, settings: InworldRealtimeSettings) -> None:
        self.room = room
        self.settings = settings
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = asyncio.Event()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._output_source = rtc.AudioSource(INWORLD_OUTPUT_SAMPLE_RATE, INWORLD_CHANNELS)
        self._published = False
        self._session_ready = asyncio.Event()
        self._audio_forwarded_count = 0
        self._last_outbound_event_type: str | None = None
        self._last_inbound_event_time = time.monotonic()

    async def run(self) -> None:
        started_at = time.monotonic()
        logger.info(
            "inworld_realtime_bridge_started=true voice_engine_selected=inworld_realtime stt_model=%s tts_model=%s tts_voice=%s turn_detection=%s voice_profile_enabled=%s",
            self.settings.stt_model,
            self.settings.tts_model,
            self.settings.voice,
            self.settings.turn_detection_type,
            self.settings.voice_profile_enabled,
        )
        await self._publish_output_track()
        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(self.settings.connection_url, headers=self.settings.auth_headers, heartbeat=20) as ws:
                    self._ws = ws
                    logger.info("inworld_realtime_connected=true session_id_present=%s", bool(self.settings.session_id))

                    # Start receive loop immediately after socket open
                    receiver = asyncio.create_task(self._receive_inworld(ws))
                    self._tasks.add(receiver)
                    logger.info("inworld_receive_task_created=true")

                    # Subscribe to existing tracks and future tracks
                    self._subscribe_existing_audio_tracks()
                    self.room.on("track_subscribed", self._on_track_subscribed)

                    try:
                        await asyncio.wait_for(self._closed.wait(), timeout=self.settings.timeout_seconds)
                    except asyncio.TimeoutError:
                        logger.info(
                            "inworld_realtime_bridge_closed=true close_reason=session_timeout timeout_seconds=%s audio_forwarded_count=%s last_outbound_event_type=%s receive_task_done=%s websocket_closed=%s",
                            self.settings.timeout_seconds,
                            self._audio_forwarded_count,
                            self._last_outbound_event_type or "none",
                            receiver.done(),
                            ws.closed,
                        )
                    finally:
                        self.room.off("track_subscribed", self._on_track_subscribed)
                        await ws.close()
                        await self.aclose()
        except Exception as exc:
            logger.error("inworld_realtime_bridge_error=true error_type=%s error=%s", type(exc).__name__, exc)
            await self.aclose()
            raise
        finally:
            logger.info("inworld_realtime_bridge_closed=true duration_seconds=%.3f", time.monotonic() - started_at)

    async def aclose(self) -> None:
        self._closed.set()
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        try:
            await self._output_source.aclose()
        except Exception:
            pass

    async def _publish_output_track(self) -> None:
        if self._published:
            return
        track = rtc.LocalAudioTrack.create_audio_track("arche-inworld-realtime", self._output_source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.room.local_participant.publish_track(track, options)
        self._published = True
        logger.info("inworld_realtime_audio_published_to_livekit=true track_name=arche-inworld-realtime")

    def _subscribe_existing_audio_tracks(self) -> None:
        for participant in self.room.remote_participants.values():
            for publication in getattr(participant, "track_publications", {}).values():
                track = getattr(publication, "track", None)
                if track is not None:
                    self._maybe_start_audio_forwarder(track)

    def _on_track_subscribed(self, track, publication=None, participant=None) -> None:
        self._maybe_start_audio_forwarder(track)

    def _maybe_start_audio_forwarder(self, track: Any) -> None:
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        task = asyncio.create_task(self._forward_livekit_audio(track))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _forward_livekit_audio(self, track: Any) -> None:
        ws = self._ws
        if ws is None:
            return
        stream = rtc.AudioStream(track, sample_rate=INWORLD_INPUT_SAMPLE_RATE, num_channels=INWORLD_CHANNELS, frame_size_ms=INWORLD_FRAME_MS)
        try:
            async for event in stream:
                frame = getattr(event, "frame", None)
                pcm = _frame_bytes(frame)
                if not pcm:
                    continue
                await ws.send_json(build_audio_append_message(pcm))
                self._audio_forwarded_count += 1
                self._last_outbound_event_type = "input_audio_buffer.append"
                logger.info("inworld_realtime_audio_input_forwarded=true bytes=%s", len(pcm) if inworld_realtime_log_audio() else "redacted")
        finally:
            await stream.aclose()

    async def _receive_inworld(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        logger.info("inworld_receive_loop_started=true")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except Exception as exc:
                        logger.error(
                            "inworld_raw_message_parse_error=true error=%s preview=%s",
                            exc,
                            msg.data[:100] if len(msg.data) > 100 else msg.data,
                        )
                        continue

                    # Log raw message receipt
                    logger.info("inworld_raw_message_received=true bytes=%s", len(msg.data))

                    await self._handle_inworld_message(payload)
                    self._last_inbound_event_time = time.monotonic()
                elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                    logger.info("inworld_websocket_closed=true msg_type=%s", msg.type)
                    break
        except asyncio.CancelledError:
            logger.info("inworld_receive_task_cancelled=true")
            raise
        except Exception as exc:
            logger.error("inworld_receive_task_exception=true error_type=%s error=%s", type(exc).__name__, exc)
            raise
        finally:
            self._closed.set()

    async def _handle_inworld_message(self, payload: dict[str, Any]) -> None:
        msg_type = str(payload.get("type") or "")

        # Log server event type
        logger.info("inworld_server_event_received type=%s", msg_type)

        if msg_type == "session.created":
            logger.info("inworld_session_created=true")
            # Send session.update immediately after session.created
            await self._send_session_update()

        elif msg_type == "session.updated":
            logger.info("inworld_session_updated=true")
            self._session_ready.set()
            # Run forced text test to prove output path works
            await self._run_forced_text_test()

        elif msg_type == "response.output_audio.delta":
            pcm = _event_audio_bytes(payload)
            if pcm:
                logger.info("inworld_audio_delta_received=true encoded_bytes=%s", len(payload.get("delta", "")))
                logger.info("inworld_audio_decoded=true pcm_bytes=%s", len(pcm))

                frame_count = 0
                for frame in _iter_pcm_frames(pcm):
                    await self._output_source.capture_frame(frame)
                    frame_count += 1

                logger.info(
                    "inworld_audio_written_to_livekit=true frames=%s samples_per_frame=%s sample_rate=%s channels=%s",
                    frame_count,
                    INWORLD_OUTPUT_SAMPLE_RATE * INWORLD_FRAME_MS // 1000,
                    INWORLD_OUTPUT_SAMPLE_RATE,
                    INWORLD_CHANNELS,
                )

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(payload.get("transcript") or "")
            logger.info("inworld_transcription_completed=true transcript_length=%s", len(transcript))

        elif msg_type == "conversation.item.input_audio_transcription.delta":
            delta = str(payload.get("delta") or "")
            logger.info("inworld_transcription_delta=true delta_length=%s", len(delta))

        elif msg_type in {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.turn_suggestion"}:
            logger.info("inworld_vad_event=true event_type=%s", msg_type)

        elif msg_type == "response.created":
            logger.info("inworld_response_created=true")

        elif msg_type == "response.output_item.added":
            logger.info("inworld_response_output_item_added=true")

        elif msg_type == "response.output_text.delta":
            delta = str(payload.get("delta") or "")
            logger.info("inworld_response_output_text_delta=true delta_length=%s", len(delta))

        elif msg_type == "response.output_audio.done":
            logger.info("inworld_response_output_audio_done=true")

        elif msg_type == "response.done":
            logger.info("inworld_response_done=true")

        elif msg_type == "error":
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            logger.error("inworld_server_error=true code=%s message=%s", error.get("code"), error.get("message"))

        else:
            logger.info("inworld_server_event_unhandled type=%s", msg_type)

    async def _send_session_update(self) -> None:
        """Send session.update after session.created."""
        if self._ws is None:
            return

        update = build_session_update(self.settings)
        await self._ws.send_json(update)
        self._last_outbound_event_type = "session.update"

        logger.info(
            "inworld_session_update_sent=true inworld_session_model=%s inworld_stt_model=%s inworld_tts_model=%s inworld_tts_voice=%s inworld_turn_detection_type=%s inworld_turn_detection_create_response=%s inworld_turn_detection_interrupt_response=%s effective_voice_profile_enabled=%s",
            self.settings.model,
            self.settings.stt_model,
            self.settings.tts_model,
            self.settings.voice,
            self.settings.turn_detection_type,
            self.settings.turn_detection_create_response,
            self.settings.turn_detection_interrupt_response,
            self.settings.voice_profile_enabled,
        )

    async def _run_forced_text_test(self) -> None:
        """Send a forced text test to prove the output path works."""
        if self._ws is None:
            return

        # Send conversation.item.create with test text
        item_create = build_conversation_item_create("Say hello in one short sentence.")
        await self._ws.send_json(item_create)
        self._last_outbound_event_type = "conversation.item.create"
        logger.info("inworld_force_text_test_sent=true")

        # Send response.create
        response_create = build_response_create()
        await self._ws.send_json(response_create)
        self._last_outbound_event_type = "response.create"
        logger.info("inworld_response_create_sent=true")


async def run_inworld_realtime_bridge(room: rtc.Room, *, instructions: str | None = None) -> None:
    settings = load_inworld_realtime_settings(instructions=instructions)
    bridge = InworldRealtimeLiveKitBridge(room, settings)
    await bridge.run()


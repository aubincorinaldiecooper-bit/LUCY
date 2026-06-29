"""LiveKit ⇄ Inworld Realtime speech-to-speech bridge.

This path is selected with ``VOICE_ENGINE=inworld_realtime`` and keeps LiveKit as
Arche's browser media transport while Inworld owns STT, TTS, and semantic VAD / turn
detection. The existing cascaded LiveKit AgentSession remains available by leaving
VOICE_ENGINE unset or set to ``current``.
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
    instructions: str
    timeout_seconds: float
    voice_profile_enabled: bool
    input_format: str
    output_format: str

    @property
    def connection_url(self) -> str:
        query = {"key": self.session_id, "protocol": "realtime"}
        separator = "&" if "?" in self.websocket_url else "?"
        return f"{self.websocket_url}{separator}{urlencode(query)}"

    @property
    def auth_headers(self) -> dict[str, str]:
        # Inworld server-side WebSocket auth uses the Portal API key directly as
        # an already-base64-encoded Basic credential.
        return {"Authorization": f"Basic {self.api_key}"}


def load_inworld_realtime_settings(*, instructions: str | None = None) -> InworldRealtimeSettings:
    api_key = (os.getenv("INWORLD_API_KEY") or "").strip()
    session_id = (os.getenv("INWORLD_REALTIME_SESSION_ID") or os.getenv("LIVEKIT_ROOM_NAME") or f"lucy-{int(time.time() * 1000)}").strip()
    if not api_key:
        raise RuntimeError("VOICE_ENGINE=inworld_realtime requires INWORLD_API_KEY")
    if not session_id:
        raise RuntimeError("VOICE_ENGINE=inworld_realtime requires INWORLD_REALTIME_SESSION_ID or a room-derived fallback")
    return InworldRealtimeSettings(
        api_key=api_key,
        session_id=session_id,
        websocket_url=(os.getenv("INWORLD_REALTIME_WS_URL") or "wss://api.inworld.ai/api/v1/realtime/session").strip(),
        model=(os.getenv("INWORLD_REALTIME_MODEL") or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip(),
        stt_model=(os.getenv("INWORLD_MODEL_ID") or os.getenv("INWORLD_STT_MODEL_ID") or "inworld/inworld-stt-1").strip(),
        tts_model=(os.getenv("INWORLD_TTS_MODEL") or "inworld-tts-2").strip(),
        voice=(os.getenv("INWORLD_TTS_VOICE") or "Dennis").strip(),
        speed=float(os.getenv("INWORLD_TTS_SPEED", "1.0") or "1.0"),
        turn_detection_type=(os.getenv("INWORLD_TURN_DETECTION_TYPE") or "semantic_vad").strip(),
        turn_detection_eagerness=(os.getenv("INWORLD_TURN_DETECTION_EAGERNESS") or "medium").strip(),
        instructions=(instructions or os.getenv("INWORLD_REALTIME_INSTRUCTIONS") or "You are a concise, warm voice assistant.").strip(),
        timeout_seconds=inworld_realtime_session_timeout_seconds(),
        voice_profile_enabled=_env_bool("INWORLD_VOICE_PROFILE_ENABLED", True),
        input_format=(os.getenv("INWORLD_REALTIME_INPUT_FORMAT") or "pcm16").strip(),
        output_format=(os.getenv("INWORLD_REALTIME_OUTPUT_FORMAT") or "pcm16").strip(),
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
                        "create_response": True,
                        "interrupt_response": True,
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

    async def run(self) -> None:
        started_at = time.monotonic()
        logger.info(
            "inworld_realtime_bridge_started=true voice_engine_selected=inworld_realtime stt_model=%s tts_model=%s turn_detection=%s voice_profile_enabled=%s",
            self.settings.stt_model,
            self.settings.tts_model,
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
                    await ws.send_json(build_session_update(self.settings))
                    self._subscribe_existing_audio_tracks()
                    self.room.on("track_subscribed", self._on_track_subscribed)
                    receiver = asyncio.create_task(self._receive_inworld(ws))
                    self._tasks.add(receiver)
                    try:
                        await asyncio.wait_for(self._closed.wait(), timeout=self.settings.timeout_seconds)
                    except asyncio.TimeoutError:
                        logger.info("inworld_realtime_bridge_closed=true close_reason=session_timeout timeout_seconds=%s", self.settings.timeout_seconds)
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
                logger.info("inworld_realtime_audio_input_forwarded=true bytes=%s", len(pcm) if inworld_realtime_log_audio() else "redacted")
        finally:
            await stream.aclose()

    async def _receive_inworld(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except Exception:
                    continue
                await self._handle_inworld_message(payload)
            elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                break
        self._closed.set()

    async def _handle_inworld_message(self, payload: dict[str, Any]) -> None:
        msg_type = str(payload.get("type") or "")
        if msg_type == "response.output_audio.delta":
            pcm = _event_audio_bytes(payload)
            for frame in _iter_pcm_frames(pcm):
                await self._output_source.capture_frame(frame)
            logger.info("inworld_realtime_audio_delta_published=true audio_output_bytes=%s", len(pcm) if inworld_realtime_log_audio() else "redacted")
        elif msg_type == "conversation.item.input_audio_transcription.completed":
            logger.info("inworld_realtime_transcript_completed=true transcript_length=%s", len(str(payload.get("transcript") or "")))
        elif msg_type in {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.turn_suggestion", "input_audio_buffer.turn_suggestion_revoked"}:
            logger.info("inworld_realtime_vad_event=true event_type=%s", msg_type)
        elif msg_type == "response.done":
            logger.info("inworld_realtime_response_done=true")
        elif msg_type == "response.output_audio.done":
            logger.info("inworld_realtime_audio_done=true")
        elif msg_type == "error":
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            logger.error("inworld_realtime_error=true code=%s message=%s", error.get("code"), error.get("message"))
        elif inworld_realtime_debug():
            logger.info("inworld_realtime_event_received=true event_type=%s", msg_type or "unknown")


async def run_inworld_realtime_bridge(room: rtc.Room, *, instructions: str | None = None) -> None:
    settings = load_inworld_realtime_settings(instructions=instructions)
    bridge = InworldRealtimeLiveKitBridge(room, settings)
    await bridge.run()

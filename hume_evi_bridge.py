"""LiveKit ⇄ Hume EVI bridge for the optional Hume voice engine.

The bridge keeps LiveKit as the browser-facing room/media layer while Hume EVI
owns speech timing, turn-taking, interruption, and assistant audio. It is only
used when VOICE_ENGINE=hume_evi; the existing cascaded pipeline remains the
rollback/default path.
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

EVI_INPUT_SAMPLE_RATE = 48000
EVI_OUTPUT_SAMPLE_RATE = 48000
EVI_CHANNELS = 1
EVI_FRAME_MS = 20


def voice_engine() -> str:
    value = (os.getenv("VOICE_ENGINE") or "current").strip().lower()
    return value if value in {"current", "hume_evi"} else "current"


def evi_bridge_debug() -> bool:
    return (os.getenv("HUME_EVI_BRIDGE_DEBUG") or "false").strip().lower() in {"1", "true", "yes", "on"}


def evi_log_audio() -> bool:
    return (os.getenv("HUME_EVI_BRIDGE_LOG_AUDIO") or "false").strip().lower() in {"1", "true", "yes", "on"}


def evi_session_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.getenv("HUME_EVI_SESSION_TIMEOUT_SECONDS", "1800")))
    except Exception:
        return 1800.0


@dataclass(frozen=True)
class HumeEVISettings:
    api_key: str
    config_id: str
    version: str
    timeout_seconds: float

    @property
    def websocket_url(self) -> str:
        base_url = os.getenv("HUME_EVI_WEBSOCKET_URL", "wss://api.hume.ai/v0/evi/chat").strip()
        query = {
            "api_key": self.api_key,
            "config_id": self.config_id,
            "evi_version": self.version,
        }
        separator = "&" if "?" in base_url else "?"
        return f"{base_url}{separator}{urlencode(query)}"


def load_hume_evi_settings() -> HumeEVISettings:
    required = ("HUME_API_KEY", "HUME_SECRET_KEY", "HUME_EVI_CONFIG_ID", "HUME_CLM_BEARER_TOKEN")
    missing = [name for name in required if not (os.getenv(name) or "").strip()]
    if missing:
        raise RuntimeError(
            "VOICE_ENGINE=hume_evi requires "
            + ", ".join(missing)
            + "; set HUME_API_KEY, HUME_SECRET_KEY, HUME_EVI_CONFIG_ID, HUME_EVI_VERSION, and HUME_CLM_BEARER_TOKEN."
        )
    return HumeEVISettings(
        api_key=(os.getenv("HUME_API_KEY") or "").strip(),
        config_id=(os.getenv("HUME_EVI_CONFIG_ID") or "").strip(),
        version=(os.getenv("HUME_EVI_VERSION") or "evi-3").strip() or "evi-3",
        timeout_seconds=evi_session_timeout_seconds(),
    )


def _frame_bytes(frame: rtc.AudioFrame) -> bytes:
    data = getattr(frame, "data", b"")
    try:
        return bytes(data)
    except Exception:
        return b""


def _audio_output_bytes(message: dict[str, Any]) -> bytes:
    data = message.get("data") or message.get("audio") or message.get("chunk")
    if not isinstance(data, str) or not data:
        return b""
    try:
        return base64.b64decode(data)
    except Exception:
        return b""


def _parse_wav(data: bytes) -> tuple[bytes, int, int]:
    """Return (pcm_bytes, sample_rate, channels) from an EVI audio_output payload.

    Hume EVI sends each audio_output chunk as a base64-encoded WAV file, so the
    RIFF/``fmt ``/``data`` header must be stripped before the bytes are treated as
    PCM; feeding the header into the audio stream is heard as a click on every
    chunk boundary. Payloads without a RIFF header are treated as raw PCM at the
    EVI defaults so a header-once/raw-after stream still plays.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data, EVI_OUTPUT_SAMPLE_RATE, EVI_CHANNELS
    sample_rate = EVI_OUTPUT_SAMPLE_RATE
    channels = EVI_CHANNELS
    pcm = b""
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = data[pos + 8 : pos + 8 + chunk_size]
        if chunk_id == b"fmt " and len(body) >= 16:
            channels = int.from_bytes(body[2:4], "little") or EVI_CHANNELS
            sample_rate = int.from_bytes(body[4:8], "little") or EVI_OUTPUT_SAMPLE_RATE
        elif chunk_id == b"data":
            pcm = body
        pos += 8 + chunk_size + (chunk_size & 1)  # chunk bodies are word-aligned
    return pcm, sample_rate, channels


def _take_full_frames(buffer: bytearray, frame_bytes: int) -> list[bytes]:
    """Drain whole 20ms frames from ``buffer``, leaving any remainder in place.

    Buffering the remainder across audio_output messages (instead of zero-padding
    each message's tail) keeps assistant audio continuous; interior padding
    inserts silence gaps that are heard as clicking between chunks.
    """
    frames: list[bytes] = []
    while len(buffer) >= frame_bytes:
        frames.append(bytes(buffer[:frame_bytes]))
        del buffer[:frame_bytes]
    return frames


class HumeEVILiveKitBridge:
    def __init__(self, room: rtc.Room, settings: HumeEVISettings) -> None:
        self.room = room
        self.settings = settings
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = asyncio.Event()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._output_source = rtc.AudioSource(EVI_OUTPUT_SAMPLE_RATE, EVI_CHANNELS)
        self._published = False
        self._output_buffer = bytearray()
        self._output_frame_bytes = int(EVI_OUTPUT_SAMPLE_RATE * EVI_FRAME_MS / 1000) * EVI_CHANNELS * 2
        self._output_samples_per_channel = int(EVI_OUTPUT_SAMPLE_RATE * EVI_FRAME_MS / 1000)
        self._pending_user_final_at: float | None = None
        self._format_warning_logged = False

    async def run(self) -> None:
        started_at = time.monotonic()
        logger.info("evi_bridge_started=true voice_engine_selected=hume_evi evi_version=%s", self.settings.version)
        await self._publish_output_track()
        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(self.settings.websocket_url, heartbeat=20) as ws:
                    self._ws = ws
                    logger.info("evi_connected=true evi_config_id_present=%s evi_version=%s", bool(self.settings.config_id), self.settings.version)
                    await self._send_session_settings(ws)
                    self._subscribe_existing_audio_tracks()
                    self.room.on("track_subscribed", self._on_track_subscribed)
                    receiver = asyncio.create_task(self._receive_evi(ws))
                    self._tasks.add(receiver)
                    try:
                        await asyncio.wait_for(self._closed.wait(), timeout=self.settings.timeout_seconds)
                    except asyncio.TimeoutError:
                        logger.info("evi_bridge_closed=true close_reason=session_timeout timeout_seconds=%s", self.settings.timeout_seconds)
                    finally:
                        self.room.off("track_subscribed", self._on_track_subscribed)
                        await ws.close()
                        await self.aclose()
        except Exception as exc:
            logger.error("evi_bridge_error=true error_type=%s error=%s", type(exc).__name__, exc)
            await self.aclose()
            raise
        finally:
            logger.info("evi_bridge_closed=true duration_seconds=%.3f", time.monotonic() - started_at)

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
        track = rtc.LocalAudioTrack.create_audio_track("arche-evi", self._output_source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.room.local_participant.publish_track(track, options)
        self._published = True
        logger.info("evi_audio_published_to_livekit=true track_name=arche-evi")

    async def _send_session_settings(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        token = (os.getenv("HUME_CLM_BEARER_TOKEN") or "").strip()
        if token:
            await ws.send_json({"type": "session_settings", "language_model_api_key": token})
        await ws.send_json(
            {
                "type": "session_settings",
                "audio": {"encoding": "linear16", "sample_rate": EVI_INPUT_SAMPLE_RATE, "channels": EVI_CHANNELS},
            }
        )

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
        stream = rtc.AudioStream(track, sample_rate=EVI_INPUT_SAMPLE_RATE, num_channels=EVI_CHANNELS, frame_size_ms=EVI_FRAME_MS)
        try:
            async for event in stream:
                frame = getattr(event, "frame", None)
                pcm = _frame_bytes(frame)
                if not pcm:
                    continue
                await ws.send_json({"type": "audio_input", "data": base64.b64encode(pcm).decode("ascii")})
                if evi_log_audio():
                    logger.info("evi_audio_input_forwarded=true bytes=%s", len(pcm))
        finally:
            await stream.aclose()

    async def _receive_evi(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except Exception:
                    continue
                await self._handle_evi_message(payload)
            elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                break
        self._closed.set()

    async def _enqueue_output_pcm(self, pcm: bytes) -> None:
        if not pcm:
            return
        self._output_buffer.extend(pcm)
        for chunk in _take_full_frames(self._output_buffer, self._output_frame_bytes):
            await self._output_source.capture_frame(
                rtc.AudioFrame(
                    data=chunk,
                    sample_rate=EVI_OUTPUT_SAMPLE_RATE,
                    num_channels=EVI_CHANNELS,
                    samples_per_channel=self._output_samples_per_channel,
                )
            )

    async def _flush_output(self) -> None:
        """Emit any buffered tail at end-of-utterance, zero-padding only the final frame."""
        while self._output_buffer:
            chunk = bytes(self._output_buffer[: self._output_frame_bytes])
            del self._output_buffer[: self._output_frame_bytes]
            if len(chunk) < self._output_frame_bytes:
                chunk = chunk + (b"\x00" * (self._output_frame_bytes - len(chunk)))
            await self._output_source.capture_frame(
                rtc.AudioFrame(
                    data=chunk,
                    sample_rate=EVI_OUTPUT_SAMPLE_RATE,
                    num_channels=EVI_CHANNELS,
                    samples_per_channel=self._output_samples_per_channel,
                )
            )

    async def _handle_evi_message(self, payload: dict[str, Any]) -> None:
        msg_type = str(payload.get("type") or "")
        if msg_type == "audio_output":
            pcm, sample_rate, channels = _parse_wav(_audio_output_bytes(payload))
            if (sample_rate != EVI_OUTPUT_SAMPLE_RATE or channels != EVI_CHANNELS) and not self._format_warning_logged:
                self._format_warning_logged = True
                logger.warning(
                    "evi_audio_output_format_unexpected=true sample_rate=%s channels=%s expected_sample_rate=%s expected_channels=%s",
                    sample_rate,
                    channels,
                    EVI_OUTPUT_SAMPLE_RATE,
                    EVI_CHANNELS,
                )
            if self._pending_user_final_at is not None:
                logger.info(
                    "evi_turn_latency_user_final_to_first_audio_ms=%.1f",
                    (time.monotonic() - self._pending_user_final_at) * 1000.0,
                )
                self._pending_user_final_at = None
            logger.info("evi_audio_output_received=true")
            await self._enqueue_output_pcm(pcm)
            logger.info("evi_audio_published_to_livekit=true audio_output_bytes=%s", len(pcm) if evi_log_audio() else "redacted")
        elif msg_type == "user_message":
            interim = bool(payload.get("interim"))
            if not interim:
                self._pending_user_final_at = time.monotonic()
            logger.info("evi_user_message_received=true interim=%s", interim)
        elif msg_type == "assistant_message":
            logger.info("evi_assistant_message_received=true")
        elif msg_type == "assistant_end":
            await self._flush_output()
            logger.info("evi_assistant_end_received=true")
        elif msg_type in {"user_interruption", "assistant_interrupted", "interruption"}:
            self._output_buffer.clear()
            self._pending_user_final_at = None
            logger.info("evi_interruption_detected=true event_type=%s", msg_type)
        elif msg_type == "error":
            logger.error("evi_bridge_error=true error_type=hume_error code=%s reason=%s", payload.get("code"), payload.get("message"))
        elif evi_bridge_debug():
            logger.info("evi_bridge_event_received=true event_type=%s", msg_type or "unknown")


async def run_hume_evi_bridge(room: rtc.Room) -> None:
    settings = load_hume_evi_settings()
    bridge = HumeEVILiveKitBridge(room, settings)
    await bridge.run()

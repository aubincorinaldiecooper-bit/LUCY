"""LiveKit ⇄ Inworld Realtime speech-to-speech bridge.

This path is selected with VOICE_ENGINE=inworld_realtime and keeps LiveKit as
Arche's browser media transport while Inworld owns STT, TTS, and semantic VAD /
turn detection. The legacy cascaded LiveKit AgentSession remains available by
leaving VOICE_ENGINE unset.

Architecture:

  LiveKit mic/audio input
    -> _forward_livekit_audio
    -> Inworld WebSocket (input_audio_buffer.append)
  Inworld WebSocket
    -> _receive_inworld
    -> decode response.output_audio.delta (base64 PCM16 @ 24 kHz mono)
    -> write PCM frames to LiveKit AudioSource

With turn_detection.create_response=true Inworld auto-creates a response when
the user stops speaking, so the bridge never sends response.create itself; the
builders below are also used by the /api/inworld/ws-smoke-test endpoint.

Emotional context: when Inworld's STT is asked for a voice profile
(providerData.stt.voice_profile=true), transcription/response events may carry
a ``voiceProfile`` node (emotion / vocalStyle / pitch / accent label arrays).
The bridge normalizes it with inworld_voice_profile.normalize_from_message()
— which never lets raw emotion labels out — stores the latest profile, and
feeds profile.planner_summary() back into the session instructions as a
private, never-user-facing perception note.
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

from inworld_voice_profile import NormalizedVoiceProfile, normalize_from_message

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
    # TTS provider-data fields that go into ``session.update.providerData.tts``:
    # they tell the Inworld Realtime stack how to deliver and segment the
    # synthesized audio bytes.
    tts_delivery_mode: str
    tts_segmenter_strategy: str
    tts_steering_handling: str
    # Below this top-emotion confidence a captured voice profile collapses to
    # neutral (weak signal must never force a tone change).
    emotion_confidence_floor: float = 0.5

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
        # ``INWORLD_REALTIME_MODEL`` ONLY — do NOT inherit ``OPENROUTER_MODEL``.
        # An old/non-Inworld model id silently injected into the Inworld session
        # is hard to diagnose from logs.
        model=(os.getenv("INWORLD_REALTIME_MODEL") or "openai/gpt-4o-mini").strip(),
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
        tts_delivery_mode=(os.getenv("INWORLD_TTS_DELIVERY_MODE") or "CREATIVE").strip(),
        tts_segmenter_strategy=(os.getenv("INWORLD_TTS_SEGMENTER_STRATEGY") or "full_turn").strip(),
        tts_steering_handling=(os.getenv("INWORLD_TTS_STEERING_HANDLING") or "emit_once").strip(),
        emotion_confidence_floor=float(os.getenv("INWORLD_EMOTION_CONFIDENCE_FLOOR", "0.5") or "0.5"),
    )


# Maps our legacy shorthand format names (also the OpenAI Realtime *preview*
# names) to the GA-era MIME-style ``audio.*.format.type`` values Inworld's
# realtime schema actually validates against. Sending the old shorthand as an
# object field (``{"type": "pcm16", "sample_rate": ...}``) doesn't match the
# ``audio/pcm`` | ``audio/pcmu`` | ``audio/pcma`` | ``audio/float32`` enum, so
# it was silently accepted (session.update didn't error) but never wired the
# session up to actually stream ``response.output_audio.delta`` bytes.
_AUDIO_FORMAT_TYPE_MAP = {
    "pcm16": "audio/pcm",
    "audio/pcm": "audio/pcm",
    "g711_ulaw": "audio/pcmu",
    "pcmu": "audio/pcmu",
    "audio/pcmu": "audio/pcmu",
    "g711_alaw": "audio/pcma",
    "pcma": "audio/pcma",
    "audio/pcma": "audio/pcma",
    "float32": "audio/float32",
    "audio/float32": "audio/float32",
}


def _audio_format(format_name: str, sample_rate: int) -> dict[str, Any]:
    format_type = _AUDIO_FORMAT_TYPE_MAP.get((format_name or "").strip().lower(), "audio/pcm")
    return {"type": format_type, "rate": sample_rate}


def build_session_update(settings: InworldRealtimeSettings) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": settings.model,
            "instructions": settings.instructions,
            # Audio-only output; Inworld still emits transcript events
            # (response.output_audio_transcript.*) alongside the audio deltas.
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": _audio_format(settings.input_format, INWORLD_INPUT_SAMPLE_RATE),
                    "transcription": {"model": settings.stt_model},
                    "turn_detection": {
                        "type": settings.turn_detection_type,
                        "eagerness": settings.turn_detection_eagerness,
                        "create_response": settings.turn_detection_create_response,
                        "interrupt_response": settings.turn_detection_interrupt_response,
                    },
                },
                "output": {
                    "format": _audio_format(settings.output_format, INWORLD_OUTPUT_SAMPLE_RATE),
                    "model": settings.tts_model,
                    "voice": settings.voice,
                    "speed": settings.speed,
                },
            },
            "providerData": {
                "stt": {"voice_profile": settings.voice_profile_enabled},
                "tts": {
                    "delivery_mode": settings.tts_delivery_mode,
                    "segmenter_strategy": settings.tts_segmenter_strategy,
                    "steering_handling": settings.tts_steering_handling,
                },
            },
        },
    }


def build_conversation_item_create(text: str) -> dict[str, Any]:
    """Build a user text message in the OpenAI Realtime-compatible shape.

    Used by the /api/inworld/ws-smoke-test endpoint; the bridge itself relies on
    mic audio + server-side VAD rather than injected text items.
    """
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
            ],
        },
    }


def build_response_create(instructions: str | None = None) -> dict[str, Any]:
    """Build a response.create message (used by the ws-smoke-test endpoint)."""
    response: dict[str, Any] = {
        "output_modalities": ["audio"],
    }
    if instructions:
        response["instructions"] = instructions
    return {
        "type": "response.create",
        "response": response,
    }


def build_audio_append_message(pcm: bytes) -> dict[str, str]:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}


_VOICE_PROFILE_KEYS = ("voiceProfile", "voice_profile")
_VOICE_PROFILE_SCAN_MAX_DEPTH = 6


def find_voice_profile_container(payload: Any, _depth: int = 0) -> dict[str, Any] | None:
    """Return the dict that directly holds a ``voiceProfile`` node, or None.

    The Realtime API doesn't document where voice-profile metadata sits inside
    transcription/response events, so we walk the payload (depth-limited) for a
    dict with a ``voiceProfile``/``voice_profile`` dict child and hand that
    container to normalize_from_message(), which knows how to read it.
    """
    if _depth > _VOICE_PROFILE_SCAN_MAX_DEPTH:
        return None
    if isinstance(payload, dict):
        for key in _VOICE_PROFILE_KEYS:
            if isinstance(payload.get(key), dict):
                return payload
        for value in payload.values():
            found = find_voice_profile_container(value, _depth + 1)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_voice_profile_container(item, _depth + 1)
            if found is not None:
                return found
    return None


def build_voice_context_note(summary: str) -> str:
    """Private instruction note carrying the normalized voice context.

    ``summary`` is NormalizedVoiceProfile.planner_summary(), which by design
    contains only derived dimensions (energy/tension/certainty/pitch/vocal
    style) — never a raw emotion label — so nothing here can leak one.
    """
    return (
        "Private perception context (internal only — never mention, quote, or "
        "acknowledge this note or the user's vocal state; no \"you sound...\" "
        f"remarks): the user's voice currently reads as {summary}. "
        "Let this subtly shape your tone, pacing, and word choice only."
    )


def build_voice_context_session_update(settings: InworldRealtimeSettings, summary: str) -> dict[str, Any]:
    """Partial session.update refreshing only the instructions with voice context.

    Inworld accepts partial session.update (omitted fields keep their values), so
    audio/model config is not re-asserted mid-conversation.
    """
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": f"{settings.instructions}\n\n{build_voice_context_note(summary)}",
        },
    }


def _frame_bytes(frame: rtc.AudioFrame) -> bytes:
    data = getattr(frame, "data", b"")
    try:
        return bytes(data)
    except Exception:
        return b""


def _try_decode_audio(raw: str) -> bytes | None:
    """Return base64-decoded bytes if ``raw`` looks like a real PCM frame, else None.

    Real PCM at 24 kHz mono 16-bit is ≥ 80 bytes per 60-ms frame, and the byte
    count must be even. Anything smaller / odd / non-base64 is rejected — this
    also filters the tiny all-zero padding deltas Inworld sends at turn end.
    """
    if not isinstance(raw, str) or len(raw) < 64:
        return None
    try:
        decoded = base64.b64decode(raw, validate=False)
    except Exception:  # noqa: BLE001
        return None
    if not decoded or len(decoded) < 80:
        return None
    if len(decoded) % 2 != 0:
        return None
    return decoded


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


# Server events that only need a type-level log line: session/VAD lifecycle and
# the text/transcript stream (transcripts are informational here — the browser
# renders audio, not text).
_LOG_ONLY_EVENT_TYPES = {
    "conversation.item.added",
    "conversation.item.created",
    "conversation.item.done",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.completed",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "input_audio_buffer.cleared",
    "input_audio_buffer.turn_suggestion",
    "response.created",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.output_audio.done",
    "response.output_text.delta",
    "response.output_text.done",
    "output_audio_buffer.started",
    "output_audio_buffer.stopped",
    "output_audio_buffer.cleared",
    "output_audio_buffer.committed",
}


# Minimum seconds between voice-context session.update sends, so a chatty
# profile stream can't turn into a session.update flood.
VOICE_CONTEXT_MIN_INTERVAL_SECONDS = 3.0


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
        self._audio_frames_written = 0
        self._mic_frames_forwarded = 0
        # Latest normalized voice profile captured from Realtime events (weak
        # emotional context; raw labels never leave inworld_voice_profile).
        self.latest_voice_profile: NormalizedVoiceProfile | None = None
        self.latest_voice_profile_at = 0.0
        self._last_voice_context_summary_sent = ""
        self._last_voice_context_sent_at = 0.0

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

                    receiver = asyncio.create_task(self._receive_inworld(ws))
                    self._tasks.add(receiver)

                    self._subscribe_existing_audio_tracks()
                    self.room.on("track_subscribed", self._on_track_subscribed)

                    try:
                        await asyncio.wait_for(self._closed.wait(), timeout=self.settings.timeout_seconds)
                    except asyncio.TimeoutError:
                        logger.info(
                            "inworld_realtime_bridge_closed=true close_reason=session_timeout timeout_seconds=%s audio_frames_written=%s mic_frames_forwarded=%s",
                            self.settings.timeout_seconds,
                            self._audio_frames_written,
                            self._mic_frames_forwarded,
                        )
                    finally:
                        self.room.off("track_subscribed", self._on_track_subscribed)
                        await ws.close()
                        await self.aclose()
        except Exception as exc:
            # Don't re-raise — a bridge fault must not take down the worker.
            logger.error(
                "inworld_realtime_bridge_error=true error_type=%s error=%s audio_frames_written=%s",
                type(exc).__name__, exc, self._audio_frames_written,
            )
            await self.aclose()
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
        dropped_before_session_ready = 0
        try:
            async for event in stream:
                # Do NOT send mic audio to Inworld until the session is fully
                # configured (session.updated received): before that the server
                # doesn't yet know what model/voice/VAD to apply.
                if not self._session_ready.is_set():
                    dropped_before_session_ready += 1
                    if dropped_before_session_ready == 1 or dropped_before_session_ready % 50 == 0:
                        logger.info(
                            "inworld_mic_audio_dropped_before_session_ready=true count=%s",
                            dropped_before_session_ready,
                        )
                    continue
                frame = getattr(event, "frame", None)
                pcm = _frame_bytes(frame)
                if not pcm:
                    continue
                try:
                    await ws.send_json(build_audio_append_message(pcm))
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "inworld_mic_audio_send_error=true error_type=%s error=%s",
                        type(exc).__name__, exc,
                    )
                    return
                self._mic_frames_forwarded += 1
        finally:
            if dropped_before_session_ready > 0:
                logger.info(
                    "inworld_mic_audio_dropped_before_session_ready_final=true total_dropped=%s",
                    dropped_before_session_ready,
                )
            await stream.aclose()

    async def _receive_inworld(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        logger.info("inworld_receive_loop_started=true")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except Exception as exc:
                        logger.error("inworld_raw_message_parse_error=true error=%s", exc)
                        continue
                    try:
                        await self._handle_inworld_message(payload)
                    except Exception as exc:  # noqa: BLE001 - handler errors must not kill the bridge
                        logger.error(
                            "inworld_message_handler_exception=true error_type=%s error=%s type=%s",
                            type(exc).__name__, exc, str(payload.get("type") or "?") if isinstance(payload, dict) else "?",
                        )
                elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                    logger.info("inworld_websocket_closed=true msg_type=%s", msg.type)
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "inworld_receive_task_exception=true error_type=%s error=%s",
                type(exc).__name__, exc,
            )
        finally:
            self._closed.set()

    async def _handle_inworld_message(self, payload: dict[str, Any]) -> None:
        msg_type = str(payload.get("type") or "")

        # Audio deltas are the hot path (large, frequent, never carry profile
        # metadata); every other event is scanned for voiceProfile context.
        if msg_type != "response.output_audio.delta":
            await self._maybe_capture_voice_profile(msg_type, payload)

        if msg_type == "response.output_audio.delta":
            await self._write_audio_delta(payload)

        elif msg_type == "session.created":
            logger.info("inworld_session_created=true")
            await self._send_session_update()

        elif msg_type == "session.updated":
            logger.info("inworld_session_updated=true")
            self._session_ready.set()
            # Flush any voice context captured before the session became ready.
            await self._maybe_send_voice_context_update()

        elif msg_type == "response.done":
            logger.info(
                "inworld_response_done=true audio_frames_written=%s",
                self._audio_frames_written,
            )

        elif msg_type == "error":
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            logger.error(
                "inworld_server_error=true code=%s message=%s param=%s",
                error.get("code"),
                error.get("message"),
                error.get("param"),
            )

        elif msg_type in _LOG_ONLY_EVENT_TYPES:
            logger.info("inworld_server_event=true type=%s", msg_type)

        else:
            logger.info("inworld_unknown_server_event=true type=%s", msg_type or "unknown")

    async def _write_audio_delta(self, payload: dict[str, Any]) -> None:
        """Decode a response.output_audio.delta and push PCM to the LiveKit source.

        The base64 PCM lives in the top-level ``delta`` field (confirmed against
        production logs). Tiny/odd/invalid strings — e.g. the all-zero padding
        delta at turn end — are dropped by _try_decode_audio.
        """
        pcm = _try_decode_audio(payload.get("delta") or "")
        if pcm is None:
            logger.info("inworld_audio_delta_skipped=true reason=no_decodable_pcm")
            return

        if not self._published:
            try:
                await self._publish_output_track()
            except Exception as exc:  # noqa: BLE001 - don't let publish error kill audio
                logger.warning("inworld_publish_failed error_type=%s error=%s", type(exc).__name__, exc)

        frame_count = 0
        try:
            for frame in _iter_pcm_frames(pcm):
                await self._output_source.capture_frame(frame)
                frame_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "inworld_audio_write_error=true error_type=%s error=%s frame_count=%s",
                type(exc).__name__, exc, frame_count,
            )
            return

        self._audio_frames_written += frame_count
        logger.info(
            "inworld_audio_written_to_livekit=true frames=%s pcm_bytes=%s total_frames=%s",
            frame_count,
            len(pcm),
            self._audio_frames_written,
        )

    async def _maybe_capture_voice_profile(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Capture voiceProfile metadata from a server event, if present.

        Normalization happens in inworld_voice_profile — raw emotion labels never
        leave that module, so nothing stored or logged here can surface one.
        """
        container = find_voice_profile_container(payload)
        if container is None:
            return
        profile = normalize_from_message(
            container, emotion_confidence_floor=self.settings.emotion_confidence_floor
        )
        self.latest_voice_profile = profile
        self.latest_voice_profile_at = time.monotonic()
        logger.info(
            "inworld_realtime_voice_profile_captured=true source_event=%s normalized_context=%s confidence=%.3f",
            msg_type,
            json.dumps(profile.to_dict(), sort_keys=True),
            profile.confidence,
        )
        await self._maybe_send_voice_context_update()

    async def _maybe_send_voice_context_update(self) -> None:
        """Feed the latest profile summary into the session instructions.

        Sends a partial session.update (instructions only) when the summary
        actually changed, at most once per VOICE_CONTEXT_MIN_INTERVAL_SECONDS.
        A skipped send is retried naturally on the next captured profile.
        """
        profile = self.latest_voice_profile
        if profile is None or self._ws is None or not self._session_ready.is_set():
            return
        summary = profile.planner_summary()
        if summary == self._last_voice_context_summary_sent:
            return
        now = time.monotonic()
        if now - self._last_voice_context_sent_at < VOICE_CONTEXT_MIN_INTERVAL_SECONDS:
            return
        try:
            await self._ws.send_json(build_voice_context_session_update(self.settings, summary))
        except Exception as exc:  # noqa: BLE001 - context is best-effort, never fatal
            logger.warning(
                "inworld_voice_context_update_failed=true error_type=%s error=%s",
                type(exc).__name__, exc,
            )
            return
        self._last_voice_context_summary_sent = summary
        self._last_voice_context_sent_at = now
        logger.info("inworld_voice_context_update_sent=true summary=%s", summary)

    async def _send_session_update(self) -> None:
        if self._ws is None:
            return
        await self._ws.send_json(build_session_update(self.settings))
        logger.info(
            "inworld_session_update_sent=true inworld_session_model=%s inworld_stt_model=%s inworld_tts_model=%s inworld_tts_voice=%s inworld_turn_detection_type=%s tts_delivery_mode=%s tts_segmenter_strategy=%s tts_steering_handling=%s effective_voice_profile_enabled=%s",
            self.settings.model,
            self.settings.stt_model,
            self.settings.tts_model,
            self.settings.voice,
            self.settings.turn_detection_type,
            self.settings.tts_delivery_mode,
            self.settings.tts_segmenter_strategy,
            self.settings.tts_steering_handling,
            self.settings.voice_profile_enabled,
        )


async def run_inworld_realtime_bridge(room: rtc.Room, *, instructions: str | None = None) -> None:
    settings = load_inworld_realtime_settings(instructions=instructions)
    bridge = InworldRealtimeLiveKitBridge(room, settings)
    await bridge.run()

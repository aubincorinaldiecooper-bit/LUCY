"""Arche realtime session core + LiveKit ⇄ Inworld speech-to-speech bridge.

This path is selected with VOICE_ENGINE=inworld_realtime and keeps LiveKit as
Arche's browser media transport while Inworld owns STT, TTS, and semantic VAD /
turn detection. The legacy cascaded LiveKit AgentSession remains available by
leaving VOICE_ENGINE unset.

Architecture (OpenAI-style: one engine, pluggable transports):

  ArcheTransport (seam)
    - LiveKitTransport: browser product — mic/camera tracks in, published
      audio track out (this module)
    - WebSocketApiTransport: public API — OpenAI Realtime-shaped JSON events
      over a WebSocket (arche_api.py)
  ArcheRealtimeSession (core, transport-agnostic)
    transport -> handle_input_pcm -> Inworld WS (input_audio_buffer.append)
    Inworld WS -> _receive_inworld
      -> decode response.output_audio.delta (base64 PCM16 @ 24 kHz mono)
      -> transport.write_output_pcm
      -> scrubbed event relay via transport.emit_event (API sessions only)

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
import io
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlencode

import aiohttp
from livekit import rtc
from PIL import Image

from emotional_dataset import (
    CalibrationTracker,
    EmotionalDatasetWriter,
    build_emotional_dataset_writer_from_env,
    remember_confirmed_pattern,
)
from inworld_voice_profile import NormalizedVoiceProfile, normalize_from_message
from memory_layer import MemoryLayer
from mood_context import (
    MoodContextConfig,
    build_mood_context_note,
    describe_mood,
    load_mood_context_config,
)
from openrouter_metering import SpendMeter
from vision_context import (
    VisionContextConfig,
    build_vision_context_note,
    describe_frame,
    load_vision_context_config,
)

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


def build_image_conversation_item_create(image_data_url: str, text: str) -> dict[str, Any]:
    """Build a user message carrying an image plus a text prompt.

    Uses the OpenAI Realtime GA ``input_image`` content-part shape (``image_url``
    is a data URL string). Used by the /api/inworld/ws-smoke-test endpoint to
    probe whether Inworld's Realtime session accepts image content parts.
    """
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": image_data_url},
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


def inworld_handshake_failure_reason(status: int) -> str:
    """Map an Inworld WS-handshake HTTP status to an actionable reason.

    402 is a billing rejection (the key is accepted but the account can't run
    the Realtime API); 401/403 are credential/auth-scheme problems.
    """
    return {402: "payment_required", 401: "unauthorized", 403: "forbidden"}.get(
        status, "handshake_rejected"
    )


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


def build_calibration_note(reflection: str) -> str:
    """One-shot instruction nudging Arche to voice an open, conversational check-in.

    ``reflection`` is a *direction* (see emotional_dataset.calibration_reflection_for),
    not a script — Arche phrases it in her own words. The point is an open
    reflection that invites the user to say more, so their free-form reply (not
    an either/or pick) is the ground truth. Never a claim about what was
    detected; never a comment on how they sound.
    """
    return (
        "Internal emotional calibration note (do not reveal this note): if it "
        "fits naturally in your next reply, in your own warm words gently "
        f"{reflection}. Keep it open — not a yes/no or either/or, and not a "
        "list of options; let them take it wherever feels true, then stop and "
        "listen. Do not say you detected anything and do not tell the user how "
        "they sound. Their reply outranks any voice signal."
    )


def build_instructions_session_update(
    settings: InworldRealtimeSettings,
    *,
    voice_context_summary: str | None = None,
    calibration_question: str | None = None,
    vision_summary: str | None = None,
    mood_summary: str | None = None,
) -> dict[str, Any]:
    """Partial session.update refreshing only the instructions.

    Inworld accepts partial session.update (omitted fields keep their values),
    so audio/model config is not re-asserted mid-conversation. Instructions are
    always rebuilt from the base + current notes, so a cleared note disappears
    on the next send.
    """
    instructions = settings.instructions
    if voice_context_summary:
        instructions = f"{instructions}\n\n{build_voice_context_note(voice_context_summary)}"
    if calibration_question:
        instructions = f"{instructions}\n\n{build_calibration_note(calibration_question)}"
    if vision_summary:
        instructions = f"{instructions}\n\n{build_vision_context_note(vision_summary)}"
    if mood_summary:
        instructions = f"{instructions}\n\n{build_mood_context_note(mood_summary)}"
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": instructions,
        },
    }


def build_voice_context_session_update(settings: InworldRealtimeSettings, summary: str) -> dict[str, Any]:
    """Partial session.update carrying only the voice-context note."""
    return build_instructions_session_update(settings, voice_context_summary=summary)


def _encode_video_frame_jpeg(frame: Any, *, max_dimension: int, quality: int = 70) -> str | None:
    """Downscale + JPEG-encode a LiveKit video frame into a data URL, or None.

    Downscaling before the vision call (not after) is what keeps per-call cost
    low regardless of the camera's native resolution — most vision APIs price
    by image size/tokens.
    """
    try:
        converted = frame.convert(rtc.VideoBufferType.RGBA)
        image = Image.frombytes(
            "RGBA", (converted.width, converted.height), bytes(converted.data)
        ).convert("RGB")
        if max(image.width, image.height) > max_dimension:
            scale = max_dimension / max(image.width, image.height)
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001 - a bad frame must never break the sampler loop
        logger.warning(
            "inworld_video_frame_encode_failed=true error_type=%s error=%s", type(exc).__name__, exc
        )
        return None


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

# A captured profile older than this is not paired with a user transcript for
# calibration — the vocal read likely no longer describes the current turn.
PROFILE_FRESHNESS_SECONDS = 30.0


# Server events relayed to API clients (arche_api.py). LiveKit sessions ignore
# these (media-plane only). Everything NOT listed stays server-side, and
# voiceProfile nodes are scrubbed even from listed events — raw emotion labels
# never leave the server.
_CLIENT_RELAY_EVENT_TYPES = {
    "session.created",
    "session.updated",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "conversation.item.input_audio_transcription.completed",
    "response.created",
    "response.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.output_audio.done",
    "output_audio_buffer.started",
    "output_audio_buffer.stopped",
    "output_audio_buffer.cleared",
    "error",
}

_VOICE_PROFILE_SCRUB_KEYS = {"voiceProfile", "voice_profile"}


def scrub_event_for_client(payload: Any) -> Any:
    """Deep-copy an event with every voiceProfile node removed.

    The privacy rule for raw emotion labels (they never leave the server; see
    inworld_voice_profile) must hold for API clients exactly as it does for the
    model and the end user.
    """
    if isinstance(payload, dict):
        return {
            key: scrub_event_for_client(value)
            for key, value in payload.items()
            if key not in _VOICE_PROFILE_SCRUB_KEYS
        }
    if isinstance(payload, list):
        return [scrub_event_for_client(item) for item in payload]
    return payload


class ArcheTransport:
    """Media/event seam between the Arche session core and the outside world.

    The core (ArcheRealtimeSession) is transport-agnostic: it speaks Inworld on
    one side and this interface on the other. LiveKitTransport carries the
    browser product; arche_api.WebSocketApiTransport carries the public API.
    Adapters push caller audio into session.handle_input_pcm() / camera frames
    into session.set_video_frame(), and receive Arche's audio + a scrubbed
    event stream back through this interface.
    """

    def bind(self, session: "ArcheRealtimeSession") -> None:
        raise NotImplementedError

    async def start(self) -> None:
        """Begin delivering input media; called once before the Inworld connect."""
        raise NotImplementedError

    async def write_output_pcm(self, pcm: bytes) -> int:
        """Deliver Arche's speech (PCM16 @ 24 kHz mono); returns units written."""
        raise NotImplementedError

    async def emit_event(self, payload: dict[str, Any]) -> None:
        """Deliver a scrubbed, whitelisted server event to the client."""
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


class LiveKitTransport(ArcheTransport):
    """LiveKit room adapter: mic/camera tracks in, published audio track out."""

    def __init__(self, room: rtc.Room) -> None:
        self.room = room
        self._session: "ArcheRealtimeSession | None" = None
        self._output_source = rtc.AudioSource(INWORLD_OUTPUT_SAMPLE_RATE, INWORLD_CHANNELS)
        self._published = False
        self._video_sampler_started = False
        self._listening = False
        self._tasks: set[asyncio.Task[Any]] = set()

    def bind(self, session: "ArcheRealtimeSession") -> None:
        self._session = session

    async def start(self) -> None:
        await self._publish_output_track()
        self._subscribe_existing_tracks()
        self.room.on("track_subscribed", self._on_track_subscribed)
        self._listening = True

    async def _publish_output_track(self) -> None:
        if self._published:
            return
        track = rtc.LocalAudioTrack.create_audio_track("arche-inworld-realtime", self._output_source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.room.local_participant.publish_track(track, options)
        self._published = True
        logger.info("inworld_realtime_audio_published_to_livekit=true track_name=arche-inworld-realtime")

    def _subscribe_existing_tracks(self) -> None:
        for participant in self.room.remote_participants.values():
            for publication in getattr(participant, "track_publications", {}).values():
                track = getattr(publication, "track", None)
                if track is not None:
                    self._maybe_start_audio_forwarder(track)
                    self._maybe_start_video_sampler(track)

    def _on_track_subscribed(self, track, publication=None, participant=None) -> None:
        self._maybe_start_audio_forwarder(track)
        self._maybe_start_video_sampler(track)

    def _maybe_start_audio_forwarder(self, track: Any) -> None:
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        task = asyncio.create_task(self._forward_livekit_audio(track))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _maybe_start_video_sampler(self, track: Any) -> None:
        # Gated on the feature flag, not just track kind: with vision context
        # disabled (the default) we never touch a published camera track at
        # all, so there is zero CPU/behavior change unless deliberately
        # enabled.
        session = self._session
        if session is None or not session.vision_enabled:
            return
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_VIDEO:
            return
        if self._video_sampler_started:
            return
        self._video_sampler_started = True
        task = asyncio.create_task(self._sample_video_track(track))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _sample_video_track(self, track: Any) -> None:
        """Hold the most recent camera frame as a downscaled JPEG data URL.

        Sampled at a fixed cadence (frame_sample_interval_seconds), not every
        incoming frame — encoding every frame would burn CPU for no benefit,
        since a frame is used at most once per min_interval_seconds anyway.
        """
        session = self._session
        if session is None:
            return
        stream = rtc.VideoStream(track)
        last_sampled_at = 0.0
        try:
            async for event in stream:
                now = time.monotonic()
                if now - last_sampled_at < session.vision_config.frame_sample_interval_seconds:
                    continue
                frame = getattr(event, "frame", None)
                if frame is None:
                    continue
                encoded = _encode_video_frame_jpeg(
                    frame, max_dimension=session.vision_config.max_frame_dimension
                )
                if encoded is not None:
                    session.set_video_frame(encoded)
                    last_sampled_at = now
        finally:
            await stream.aclose()

    async def _forward_livekit_audio(self, track: Any) -> None:
        session = self._session
        if session is None:
            return
        stream = rtc.AudioStream(track, sample_rate=INWORLD_INPUT_SAMPLE_RATE, num_channels=INWORLD_CHANNELS, frame_size_ms=INWORLD_FRAME_MS)
        try:
            async for event in stream:
                frame = getattr(event, "frame", None)
                pcm = _frame_bytes(frame)
                if not pcm:
                    continue
                await session.handle_input_pcm(pcm)
        finally:
            await stream.aclose()

    async def write_output_pcm(self, pcm: bytes) -> int:
        if not self._published:
            try:
                await self._publish_output_track()
            except Exception as exc:  # noqa: BLE001 - don't let publish error kill audio
                logger.warning("inworld_publish_failed error_type=%s error=%s", type(exc).__name__, exc)
        frame_count = 0
        for frame in _iter_pcm_frames(pcm):
            await self._output_source.capture_frame(frame)
            frame_count += 1
        return frame_count

    async def emit_event(self, payload: dict[str, Any]) -> None:
        # Media-plane transport: the browser gets audio via the published
        # track; events stay server-side.
        return

    async def aclose(self) -> None:
        if self._listening:
            try:
                self.room.off("track_subscribed", self._on_track_subscribed)
            except Exception:  # noqa: BLE001
                pass
            self._listening = False
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        try:
            await self._output_source.aclose()
        except Exception:  # noqa: BLE001
            pass


class ArcheRealtimeSession:
    """Transport-agnostic Arche session core.

    Owns the Inworld Realtime connection plus all product logic (memory,
    emotional context, vision, calibration, instructions). Caller media
    arrives via handle_input_pcm()/set_video_frame(); Arche's speech and a
    scrubbed event stream leave via the bound ArcheTransport.
    """

    def __init__(
        self,
        settings: InworldRealtimeSettings,
        transport: ArcheTransport,
        *,
        dataset_writer: EmotionalDatasetWriter | None = None,
        calibration_enabled: bool = False,
        memory_task: "asyncio.Task[tuple[MemoryLayer | None, str | None]] | None" = None,
        vision_config: VisionContextConfig | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        transport.bind(self)
        # Vision gating is transport-specific: the product camera path stays
        # behind VISION_CONTEXT_ENABLED (ambient capture needs a deliberate
        # flag), while API sessions inject a forced-enabled config — a client
        # explicitly sending input_image IS the opt-in (see arche_api).
        self._vision_config_override = vision_config
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = asyncio.Event()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session_ready = asyncio.Event()
        self._audio_frames_written = 0
        self._mic_frames_forwarded = 0
        self._dropped_before_session_ready = 0
        # Latest normalized voice profile captured from Realtime events (weak
        # emotional context; raw labels never leave inworld_voice_profile).
        self.latest_voice_profile: NormalizedVoiceProfile | None = None
        self.latest_voice_profile_at = 0.0
        self._last_voice_context_sent_at = 0.0
        # Full instructions text of the last session.update we sent, for dedupe.
        self._last_instructions_sent = ""
        # Dataset collection (best-effort, server-side only) + ground-truth
        # calibration. Both default off; see emotional_dataset.py.
        self._dataset_writer = dataset_writer
        self._calibration = (
            CalibrationTracker(session_id=settings.session_id) if calibration_enabled else None
        )
        self._active_calibration_question: str | None = None
        # Long-term memory resolves in the background (identity lookup + up to
        # a 2s Postgres preload) concurrently with WebSocket connect + LiveKit
        # audio-track subscription below, so a fast talker's opening words
        # are never dropped waiting on the database. _ensure_memory_resolved()
        # awaits memory_task exactly once, lazily, the first time memory is
        # actually needed (building the first session.update). A simple
        # per-session counter stands in for the legacy pipeline's turn id.
        self._memory_task = memory_task
        self._memory_layer: MemoryLayer | None = None
        self._memory_preload_note: str | None = None
        self._memory_resolved = asyncio.Event()
        self._turn_counter = 0
        # Vision context: off unless VISION_CONTEXT_ENABLED=true, in which case
        # a camera track (if the user publishes one) is sampled at a fixed
        # cadence and described via vision_context.describe_frame(). See
        # vision_context.py for the model-agnostic OpenRouter call.
        self._vision_config: VisionContextConfig = (
            self._vision_config_override
            if self._vision_config_override is not None
            else load_vision_context_config()
        )
        self.latest_vision_summary: str | None = None
        self._last_vision_context_sent_at = 0.0
        self._latest_video_frame_data_url: str | None = None
        self._vision_call_in_flight = False
        self._last_vision_summary_remembered: str | None = None
        # Mood context: off unless MOOD_CONTEXT_ENABLED=true. On each completed
        # user transcript it fuses the words + the current vocal dials into one
        # human-style read (mood_context.describe_mood) and folds it into the
        # session instructions. Never a conversation item; rate-limited.
        self._mood_config: MoodContextConfig = load_mood_context_config()
        self.latest_mood_summary: str | None = None
        self._last_mood_context_sent_at = 0.0
        self._mood_call_in_flight = False
        # OpenRouter spend metering: one persisted row per vision/mood call,
        # attributed to the user (billing key) and session. Off unless
        # OPENROUTER_METERING_ENABLED=true (and DATABASE_URL set); inert
        # otherwise. user_ref is set lazily once memory identity resolves.
        self._spend_meter = SpendMeter(session_id=settings.session_id)

    @property
    def vision_enabled(self) -> bool:
        return self._vision_config.enabled

    @property
    def vision_config(self) -> VisionContextConfig:
        return self._vision_config

    async def handle_input_pcm(self, pcm: bytes) -> None:
        """Feed one chunk of caller audio (PCM16 @ 24 kHz mono) toward Inworld.

        Called by the transport for every input chunk. Audio is dropped until
        the Inworld session is fully configured (session.updated received):
        before that the server doesn't yet know what model/voice/VAD to apply.
        """
        if not pcm:
            return
        ws = self._ws
        if ws is None or not self._session_ready.is_set():
            self._dropped_before_session_ready += 1
            if self._dropped_before_session_ready == 1 or self._dropped_before_session_ready % 50 == 0:
                logger.info(
                    "inworld_mic_audio_dropped_before_session_ready=true count=%s",
                    self._dropped_before_session_ready,
                )
            return
        try:
            await ws.send_json(build_audio_append_message(pcm))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "inworld_mic_audio_send_error=true error_type=%s error=%s",
                type(exc).__name__, exc,
            )
            return
        self._mic_frames_forwarded += 1

    def set_video_frame(self, data_url: str) -> None:
        """Hold the most recent camera frame (as an image data URL) for vision.

        Called by the LiveKit video sampler and by API clients' input_image
        events. Only ever read at speech-start (rate-limited), so setting it
        frequently costs nothing.
        """
        if isinstance(data_url, str) and data_url.startswith("data:image/"):
            self._latest_video_frame_data_url = data_url

    async def _ensure_memory_resolved(self) -> None:
        """Await the background memory-resolution task exactly once.

        No-ops instantly once resolved (or if no task was given). Errors are
        already handled inside _build_memory_layer_for_realtime (it never
        raises, only degrades to (None, None)); this is a last-resort net for
        cancellation or an unexpected failure so memory never crashes a turn.
        """
        if self._memory_resolved.is_set():
            return
        if self._memory_task is not None:
            try:
                self._memory_layer, self._memory_preload_note = await self._memory_task
            except Exception as exc:  # noqa: BLE001 - memory resolution must never break the bridge
                logger.warning(
                    "inworld_memory_task_failed=true error_type=%s error=%s",
                    type(exc).__name__, exc,
                )
                self._memory_layer, self._memory_preload_note = None, None
        # Attribute metered spend to the resolved billing key (durable per-user
        # identity), if any. Guest/anonymous sessions stay user_ref=None and are
        # metered by session_id alone. Read defensively: reading an identity
        # must never crash memory resolution / the turn.
        identity = getattr(self._memory_layer, "identity", None)
        if identity is not None and getattr(identity, "present", False):
            self._spend_meter.user_ref = identity.key
        self._memory_resolved.set()

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
        await self.transport.start()
        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(self.settings.connection_url, headers=self.settings.auth_headers, heartbeat=20) as ws:
                    self._ws = ws
                    logger.info("inworld_realtime_connected=true session_id_present=%s", bool(self.settings.session_id))

                    receiver = asyncio.create_task(self._receive_inworld(ws))
                    self._tasks.add(receiver)

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
                        await ws.close()
                        await self.aclose()
        except aiohttp.WSServerHandshakeError as exc:
            # Inworld rejected the WebSocket handshake before any audio. The HTTP
            # status is the actionable signal — surface it as its own greppable
            # flag so an account/billing problem isn't buried in a generic error.
            logger.error(
                "inworld_realtime_handshake_failed=true inworld_http_status=%s reason=%s "
                "audio_frames_written=%s (402=check Inworld account billing; "
                "401/403=check INWORLD_API_KEY/INWORLD_AUTH_SCHEME)",
                exc.status, inworld_handshake_failure_reason(exc.status), self._audio_frames_written,
            )
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
        if self._memory_task is not None:
            if self._memory_task.done():
                await self._ensure_memory_resolved()
            else:
                # Session ended before memory ever resolved (very short call);
                # nothing was constructed, so just cancel cleanly.
                self._memory_task.cancel()
                try:
                    await self._memory_task
                except (asyncio.CancelledError, Exception):
                    pass
        if self._memory_layer is not None:
            try:
                await self._memory_layer.aclose()
            except Exception:
                pass
        try:
            await self._spend_meter.aclose()
        except Exception:
            pass
        try:
            await self.transport.aclose()
        except Exception:
            pass

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

        # API clients get a scrubbed subset of the event stream (transcripts,
        # lifecycle, VAD). No-op for LiveKit sessions. Audio reaches clients
        # via transport.write_output_pcm, not here.
        if msg_type in _CLIENT_RELAY_EVENT_TYPES:
            try:
                await self.transport.emit_event(scrub_event_for_client(payload))
            except Exception as exc:  # noqa: BLE001 - a client hiccup must not break the session
                logger.warning(
                    "client_event_relay_failed=true type=%s error_type=%s error=%s",
                    msg_type, type(exc).__name__, exc,
                )

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

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(payload.get("transcript") or "")
            logger.info(
                "inworld_user_transcription_completed=true transcript_length=%s",
                len(transcript),
            )
            if transcript:
                self._turn_counter += 1
                if self._memory_layer is not None:
                    self._memory_layer.schedule_remember(
                        role="user", content=transcript, turn_id=self._turn_counter
                    )
                await self._on_user_transcript(transcript)
                # Fuse these words with the current vocal dials into a human
                # mood read (background, rate-limited). No-op unless enabled.
                await self._maybe_send_mood_context_update(transcript)

        elif msg_type == "response.output_audio_transcript.done":
            transcript = str(payload.get("transcript") or "")
            logger.info(
                "inworld_assistant_transcript_done=true transcript_length=%s",
                len(transcript),
            )
            if transcript and self._memory_layer is not None:
                self._memory_layer.schedule_remember(
                    role="assistant", content=transcript, turn_id=self._turn_counter
                )

        elif msg_type == "input_audio_buffer.speech_started":
            logger.info("inworld_server_event=true type=%s", msg_type)
            # Refresh vision context right as the user starts talking, so
            # Arche's next reply is grounded in a fresh-ish frame rather than
            # a stale one from earlier in the call. No-ops entirely unless
            # VISION_CONTEXT_ENABLED=true and a camera frame has been sampled.
            await self._maybe_send_vision_context_update()

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
        """Decode a response.output_audio.delta and hand PCM to the transport.

        The base64 PCM lives in the top-level ``delta`` field (confirmed against
        production logs). Tiny/odd/invalid strings — e.g. the all-zero padding
        delta at turn end — are dropped by _try_decode_audio.
        """
        pcm = _try_decode_audio(payload.get("delta") or "")
        if pcm is None:
            logger.info("inworld_audio_delta_skipped=true reason=no_decodable_pcm")
            return

        try:
            frame_count = await self.transport.write_output_pcm(pcm)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "inworld_audio_write_error=true error_type=%s error=%s",
                type(exc).__name__, exc,
            )
            return

        self._audio_frames_written += frame_count
        # Log key predates the transport seam (dashboards grep for it); for API
        # sessions "livekit" here just means "the transport".
        logger.info(
            "inworld_audio_written_to_livekit=true frames=%s pcm_bytes=%s total_frames=%s",
            frame_count,
            len(pcm),
            self._audio_frames_written,
        )

    async def _maybe_capture_voice_profile(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Capture voiceProfile metadata from a server event, if present.

        Normalization happens in inworld_voice_profile — raw emotion labels never
        leave that module toward the model or user. The raw label arrays ARE
        persisted server-side (dataset writer) so sessions build a labeled
        dataset for improving the analyzer.
        """
        container = find_voice_profile_container(payload)
        if container is None:
            return
        raw_node: dict[str, Any] = {}
        for key in _VOICE_PROFILE_KEYS:
            node = container.get(key)
            if isinstance(node, dict):
                raw_node = node
                break
        profile = normalize_from_message(
            container, emotion_confidence_floor=self.settings.emotion_confidence_floor
        )
        self.latest_voice_profile = profile
        self.latest_voice_profile_at = time.monotonic()
        logger.info(
            "inworld_realtime_voice_profile_captured=true source_event=%s normalized_context=%s confidence=%.3f dataset_write=%s",
            msg_type,
            json.dumps(profile.to_dict(), sort_keys=True),
            profile.confidence,
            self._dataset_writer is not None,
        )
        if self._dataset_writer is not None:
            transcript = payload.get("transcript")
            self._dataset_writer.record_profile_event(
                session_id=self.settings.session_id,
                source_event=msg_type,
                raw_profile=raw_node,
                normalized_profile=profile.to_dict(),
                transcript=transcript if isinstance(transcript, str) and transcript else None,
                emotion_confidence=profile.confidence,
            )
        await self._maybe_send_voice_context_update()

    async def _on_user_transcript(self, transcript: str) -> None:
        """Advance calibration with a completed user utterance.

        Completes any pending open reflection (the user's free-form reply is
        the ground truth, persisted via the dataset writer) and may arm a new
        one, injected into the session instructions as a one-shot nudge. When a
        recent mood read is available it grounds the reflection.
        """
        if self._calibration is None:
            return
        profile = None
        if (
            self.latest_voice_profile is not None
            and time.monotonic() - self.latest_voice_profile_at <= PROFILE_FRESHNESS_SECONDS
        ):
            profile = self.latest_voice_profile
        completed, question = self._calibration.on_user_transcript(
            transcript, profile, mood_summary=self.latest_mood_summary
        )
        changed = False
        if completed is not None:
            if self._dataset_writer is not None:
                self._dataset_writer.record_calibration_moment(completed)
            remember_confirmed_pattern(self._memory_layer, completed)
            logger.info(
                "emotional_calibration_moment_stored=true session_id=%s user_answer_present=%s dataset_write=%s memory_write=%s",
                completed.get("session_id"),
                bool(str(completed.get("user_answer") or "").strip()),
                self._dataset_writer is not None,
                self._memory_layer is not None and bool(completed.get("user_confirmed_or_corrected")),
            )
            if self._active_calibration_question is not None:
                self._active_calibration_question = None
                changed = True
        if question:
            self._active_calibration_question = question
            changed = True
        if changed:
            # Calibration arms/clears are rare one-shots (cadence-limited), so
            # they bypass the voice-context throttle.
            await self._send_instructions_update(reason="calibration_change")

    def _compose_instructions_update(self) -> dict[str, Any]:
        summary = (
            self.latest_voice_profile.planner_summary()
            if self.latest_voice_profile is not None
            else None
        )
        return build_instructions_session_update(
            self.settings,
            voice_context_summary=summary,
            calibration_question=self._active_calibration_question,
            vision_summary=self.latest_vision_summary,
            mood_summary=self.latest_mood_summary,
        )

    async def _send_instructions_update(self, reason: str) -> bool:
        """Send the composed partial instructions update if it changed."""
        if self._ws is None or not self._session_ready.is_set():
            return False
        update = self._compose_instructions_update()
        instructions = update["session"]["instructions"]
        if instructions == self._last_instructions_sent:
            return False
        try:
            await self._ws.send_json(update)
        except Exception as exc:  # noqa: BLE001 - context is best-effort, never fatal
            logger.warning(
                "inworld_instructions_update_failed=true reason=%s error_type=%s error=%s",
                reason, type(exc).__name__, exc,
            )
            return False
        self._last_instructions_sent = instructions
        logger.info(
            "inworld_instructions_update_sent=true reason=%s calibration_question_active=%s",
            reason,
            self._active_calibration_question is not None,
        )
        return True

    async def _maybe_send_voice_context_update(self) -> None:
        """Feed the latest profile summary into the session instructions.

        Sends when the composed instructions actually changed, at most once per
        VOICE_CONTEXT_MIN_INTERVAL_SECONDS. A skipped send is retried naturally
        on the next captured profile.
        """
        if self.latest_voice_profile is None:
            return
        now = time.monotonic()
        if now - self._last_voice_context_sent_at < VOICE_CONTEXT_MIN_INTERVAL_SECONDS:
            return
        if await self._send_instructions_update(reason="voice_context"):
            self._last_voice_context_sent_at = now
            logger.info(
                "inworld_voice_context_update_sent=true summary=%s",
                self.latest_voice_profile.planner_summary(),
            )

    async def _maybe_send_vision_context_update(self) -> None:
        """Kick a background vision-description call, rate-limited.

        Fire-and-forget: the OpenRouter round trip (up to
        VISION_CONTEXT_TIMEOUT_SECONDS) must never block the receive loop or
        delay the user's turn. The result folds into instructions via the same
        session.update path as voice context — never a new conversation item —
        so a slow or failed call can only skip an update, never interrupt or
        duplicate a turn.
        """
        if not self._vision_config.enabled or self._vision_call_in_flight:
            return
        if self._latest_video_frame_data_url is None:
            return
        now = time.monotonic()
        if now - self._last_vision_context_sent_at < self._vision_config.min_interval_seconds:
            return
        self._last_vision_context_sent_at = now
        self._vision_call_in_flight = True
        task = asyncio.create_task(self._run_vision_context_update())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_vision_context_update(self) -> None:
        try:
            frame = self._latest_video_frame_data_url
            if frame is None:
                return
            summary = await describe_frame(
                frame,
                self._vision_config,
                on_usage=lambda data: self._spend_meter.record_from_response(
                    "vision", self._vision_config.model, data
                ),
            )
            if not summary:
                return
            self.latest_vision_summary = summary
            # Optionally persist what was seen (VISION_MEMORY_ENABLED) so Arche
            # can recall objects across sessions. Deduped on the exact summary:
            # a static scene produces one memory, not one per speech turn.
            if (
                self._vision_config.remember_enabled
                and self._memory_layer is not None
                and summary != self._last_vision_summary_remembered
            ):
                self._memory_layer.schedule_remember(
                    role="observation", content=summary, turn_id=self._turn_counter
                )
                self._last_vision_summary_remembered = summary
            await self._send_instructions_update(reason="vision_context")
            logger.info("inworld_vision_context_update_sent=true summary_length=%s", len(summary))
        finally:
            self._vision_call_in_flight = False

    async def _maybe_send_mood_context_update(self, transcript: str) -> None:
        """Kick a background mood-fusion call for a completed user turn.

        Fire-and-forget and rate-limited: the OpenRouter round trip must never
        block the receive loop. The result folds into instructions via the same
        session.update path as voice/vision context — never a conversation item.
        No-ops entirely unless MOOD_CONTEXT_ENABLED=true.
        """
        transcript = (transcript or "").strip()
        if not self._mood_config.enabled or self._mood_call_in_flight or not transcript:
            return
        now = time.monotonic()
        if now - self._last_mood_context_sent_at < self._mood_config.min_interval_seconds:
            return
        self._last_mood_context_sent_at = now
        self._mood_call_in_flight = True
        # Pair the words with the freshest vocal read, if it's recent enough to
        # still describe this turn (same freshness bound calibration uses).
        vocal_summary = None
        if (
            self.latest_voice_profile is not None
            and time.monotonic() - self.latest_voice_profile_at <= PROFILE_FRESHNESS_SECONDS
        ):
            vocal_summary = self.latest_voice_profile.planner_summary()
        task = asyncio.create_task(self._run_mood_context_update(transcript, vocal_summary))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_mood_context_update(self, transcript: str, vocal_summary: str | None) -> None:
        try:
            summary = await describe_mood(
                transcript,
                vocal_summary,
                self._mood_config,
                on_usage=lambda data: self._spend_meter.record_from_response(
                    "mood", self._mood_config.model, data
                ),
            )
            if not summary:
                return
            self.latest_mood_summary = summary
            await self._send_instructions_update(reason="mood_context")
            logger.info("inworld_mood_context_update_sent=true summary_length=%s", len(summary))
        finally:
            self._mood_call_in_flight = False

    async def _send_session_update(self) -> None:
        if self._ws is None:
            return
        # Resolve memory here (not earlier): this is the first point the
        # preload note is actually needed, and by now WebSocket connect +
        # LiveKit audio-track subscription are already underway/complete, so
        # waiting on Postgres here no longer risks missing the user's mic.
        await self._ensure_memory_resolved()
        if self._memory_preload_note:
            self.settings = replace(
                self.settings,
                instructions=f"{self.settings.instructions}\n\n{self._memory_preload_note}",
            )
        await self._ws.send_json(build_session_update(self.settings))
        logger.info(
            "inworld_session_update_sent=true inworld_session_model=%s inworld_stt_model=%s inworld_tts_model=%s inworld_tts_voice=%s inworld_turn_detection_type=%s tts_delivery_mode=%s tts_segmenter_strategy=%s tts_steering_handling=%s effective_voice_profile_enabled=%s memory_preload_note_present=%s",
            self.settings.model,
            self.settings.stt_model,
            self.settings.tts_model,
            self.settings.voice,
            self.settings.turn_detection_type,
            self.settings.tts_delivery_mode,
            self.settings.tts_segmenter_strategy,
            self.settings.tts_steering_handling,
            self.settings.voice_profile_enabled,
            bool(self._memory_preload_note),
        )


class InworldRealtimeLiveKitBridge(ArcheRealtimeSession):
    """Back-compat constructor: LiveKit room in, same session core.

    Kept so agent.py and the test suite construct the LiveKit-transported
    session exactly as before the transport seam existed.
    """

    def __init__(
        self,
        room: rtc.Room,
        settings: InworldRealtimeSettings,
        *,
        dataset_writer: EmotionalDatasetWriter | None = None,
        calibration_enabled: bool = False,
        memory_task: "asyncio.Task[tuple[MemoryLayer | None, str | None]] | None" = None,
    ) -> None:
        super().__init__(
            settings,
            LiveKitTransport(room),
            dataset_writer=dataset_writer,
            calibration_enabled=calibration_enabled,
            memory_task=memory_task,
        )
        self.room = room


async def run_inworld_realtime_bridge(
    room: rtc.Room,
    *,
    instructions: str | None = None,
    memory_task: "asyncio.Task[tuple[MemoryLayer | None, str | None]] | None" = None,
) -> None:
    settings = load_inworld_realtime_settings(instructions=instructions)
    dataset_writer = build_emotional_dataset_writer_from_env()
    calibration_enabled = _env_bool("EMOTIONAL_CALIBRATION_REALTIME_ENABLED", False)
    logger.info(
        "emotional_dataset_wiring dataset_writer_active=%s calibration_enabled=%s memory_resolution_pending=%s",
        dataset_writer is not None,
        calibration_enabled,
        memory_task is not None,
    )
    bridge = InworldRealtimeLiveKitBridge(
        room,
        settings,
        dataset_writer=dataset_writer,
        calibration_enabled=calibration_enabled,
        memory_task=memory_task,
    )
    if dataset_writer is not None:
        dataset_writer.start()
    try:
        await bridge.run()
    finally:
        if dataset_writer is not None:
            await dataset_writer.aclose()
        # bridge.aclose() (called from bridge.run()'s own finally/except paths)
        # already resolves-or-cancels memory_task and closes the memory layer,
        # so there is nothing left to do with it here.

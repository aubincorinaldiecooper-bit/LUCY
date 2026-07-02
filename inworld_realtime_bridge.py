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

    # TTS provider-data fields that go into ``session.update.providerData.tts``.

    # These tell the Inworld Realtime stack HOW to deliver (and segment) the

    # synthesized audio bytes; in our smoke-test session they were missing,

    # and Inworld responded with text-only events despite

    # ``output_modalities=["audio"]``. Confirmed-runtime defaults match the

    # values that have been observed to make ``response.output_audio.delta``

    # actually carry PCM on a Luna / inworld-tts-2 configuration.

    tts_delivery_mode: str

    tts_segmenter_strategy: str

    tts_steering_handling: str


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

        # ``INWORLD_REALTIME_MODEL`` ONLY — do NOT inherit ``OPENROUTER_MODEL``.

        # An old/non-Inworld model id silently injected into the Inworld session

        # is hard to diagnose from logs. Default is a known realtime-capable OpenAI

        # model id; override via the env var if Inworld ships another default.

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

        # TTS provider-data (see InworldRealtimeSettings docstring). Defaults

        # match the values confirmed-runtime for an inworld-tts-2 + Luna smoke

        # test session where ``response.output_audio.delta`` is expected to

        # carry PCM bytes.

        tts_delivery_mode=(os.getenv("INWORLD_TTS_DELIVERY_MODE") or "CREATIVE").strip(),

        tts_segmenter_strategy=(os.getenv("INWORLD_TTS_SEGMENTER_STRATEGY") or "full_turn").strip(),

        tts_steering_handling=(os.getenv("INWORLD_TTS_STEERING_HANDLING") or "emit_once").strip(),

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

            "output_modalities": ["audio"],  # AUDIO-ONLY EXPERIMENT: forcing audio only to verify Inworld returns PCM bytes

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

    """Build a forced text test message in OpenAI Realtime / Inworld-compatible shape.


    The previous version used ``content_type: input_text`` as a single dict — the

    Inworld server rejected that as malformed JSON. The standard shape (used by

    OpenAI's Realtime API which Inworld mirrors) is an array of content parts with

    each part keyed by ``type`` (= ``input_text`` for user text).

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

    """Build a response.create message.


    Per spec ``response.create`` accepts a ``response`` object that may include

    ``output_modalities`` (mirror of the session-level field), ``voice``,

    ``instructions``, ``tools``, etc. We declare both audio and text so the

    server is allowed to stream either (text keeps transcripts useful for

    logging; audio drives the LiveKit track).


    ``instructions`` is included when supplied to bias the model toward an

    actual spoken reply (rather than a transcript-only response).

    """

    response: dict[str, Any] = {

        # AUDIO-ONLY EXPERIMENT: forcing audio only to verify Inworld returns PCM

        # bytes rather than a transcript-only response. Revert to

        # ``["audio", "text"]`` once the audio path is proven.

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



def _frame_bytes(frame: rtc.AudioFrame) -> bytes:

    data = getattr(frame, "data", b"")

    try:

        return bytes(data)

    except Exception:

        return b""



def _event_audio_bytes(payload: dict[str, Any], *, aggressive: bool = False) -> bytes:

    """Recursively pull audio bytes out of an Inworld server event payload.


    Bytes-only convenience wrapper around ``_event_audio_candidate`` — discards

    the source-field path. Internal callers that want path logging use the

    candidate function directly. ``aggressive=True`` enables the Phase 2 fallback

    (try-every-string-value) intended only for smoke-test mode.

    """

    pcm, _ = _event_audio_candidate(payload, aggressive=aggressive)

    return pcm



def _event_audio_candidate(

    payload: dict[str, Any] | Any,

    *,

    aggressive: bool = False,

) -> tuple[bytes, str | None]:

    """Walk plausible audio-carrying paths; return the first valid base64 PCM and its path.


    The ReaAPI reference page does not name the exact event that carries the

    assistant's audio bytes for ``inworld-tts-2`` in the main response path —

    production logs show ``response.output_audio_transcript.delta`` and

    ``response.output_audio.done`` arrive, but the audio byte delta often shows

    up under less obvious field names. Two-phase strategy:


      Phase 1 (always). Try audio-named keys (``audio*``) and ``delta``/``data`` —

      the documented Realtime-compatible paths.

      Phase 2 (only with ``aggressive=True``). FALL BACK to *every other

      string value* in the payload. If Inworld names the field ``voice_bytes``,

      ``audio_blob``, or anything we haven't predicted, we'll still find it.

      Phase 2 is gated because a long non-audio string (transcript, error

      message) COULD in theory base64-decode to ≥ 80 bytes and satisfy the

      ``_try_decode_audio`` floor — the audio would sound like static, not

      speech. Smoke tests want it on; production wants it off.


    ``source_field_path`` is included in the ``inworld_audio_candidate_found``

    log so we can iterate from logs to figure out which path actually carries

    audio in *this* Inworld build. False-positive risk is bounded by

    ``_try_decode_audio`` (≥ 80 bytes decoded + even byte length).

    """

    audio_keys = ("audio", "audioContent", "audio_content", "audio_data")


    def _scan(obj: Any, path: str, known_already_tried: tuple[str, ...]) -> tuple[bytes, str | None]:

        if isinstance(obj, dict):

            # Phase 1: documented audio/delta/data names.

            for key in known_already_tried:

                raw = obj.get(key)

                if isinstance(raw, str) and raw:

                    decoded = _try_decode_audio(raw)

                    if decoded is not None:

                        return decoded, f"{path}.{key}"

            # Phase 2 (only with aggressive=True): every OTHER string value.

            if aggressive:

                for k, v in obj.items():

                    if k in known_already_tried:

                        continue

                    if isinstance(v, str) and v:

                        decoded = _try_decode_audio(v)

                        if decoded is not None:

                            return decoded, f"{path}.{k}"

            # Phase 3: recurse into nested objects (skip strings — already tried).

            for k, v in obj.items():

                if isinstance(v, (dict, list)):

                    found, found_path = _scan(v, f"{path}.{k}", known_already_tried)

                    if found:

                        return found, found_path

        elif isinstance(obj, list):

            for i, item in enumerate(obj):

                found, found_path = _scan(item, f"{path}[{i}]", known_already_tried)

                if found:

                    return found, found_path

        return b"", None


    known_already_tried = audio_keys + ("delta", "data")

    pcm, source = _scan(payload if isinstance(payload, dict) else {}, "payload", known_already_tried)

    if pcm:

        logger.info(

            "inworld_audio_candidate_found=true aggressive=%s source_field_path=%s decoded_bytes=%s",

            aggressive, source or "<unknown>", len(pcm),

        )

    return pcm, source



def _event_audio_payload_summary(payload: dict[str, Any] | Any) -> list[tuple[str, int]]:

    """Diagnostic: return the top-5 longest string values in the payload with their field paths.


    Used as a fallback diagnostic when ``_event_audio_candidate`` returns no

    bytes. Lets us answer "is there even a base64-shaped string in here, and if

    so where?" — which is the question we need to ask when ``inworld_audio_

    written_to_livekit`` never fires. Length threshold is 100 chars (≈75 decoded

    bytes) so very short transcripts/transcript-id-like strings don't pollute.

    """

    candidates: list[tuple[str, int]] = []


    def _walk(obj: Any, path: str) -> None:

        if isinstance(obj, dict):

            for k, v in obj.items():

                if isinstance(v, str) and len(v) >= 100:

                    candidates.append((f"{path}.{k}", len(v)))

                elif isinstance(v, (dict, list)):

                    _walk(v, f"{path}.{k}")

        elif isinstance(obj, list):

            for i, x in enumerate(obj):

                _walk(x, f"{path}[{i}]")


    _walk(payload if isinstance(payload, dict) else {}, "payload")

    candidates.sort(key=lambda x: -x[1])

    return candidates[:5]



def _try_decode_audio(raw: str) -> bytes | None:

    """Return base64-decoded bytes if ``raw`` looks like a real PCM frame, else None.


    Real PCM at 24 kHz mono 16-bit is ≥ 80 bytes per 60-ms frame, and the byte

    count must be even. Anything smaller / odd / non-base64 is rejected. The

    80-byte floor is intentionally low so we don't miss small deltas; the

    `_event_audio_payload_summary` log lets us eyeball whether we're being too

    lax if anything suspicious shows up downstream.

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



def _safe_payload_shape(payload: dict[str, Any]) -> str:

    """Return a safe representation of payload structure without secrets or audio data."""

    def shape_value(v: Any) -> str:

        if isinstance(v, dict):

            keys = sorted(v.keys())

            return "{" + ",".join(keys) + "}"

        elif isinstance(v, list):

            return "[...]"

        elif isinstance(v, str):

            if len(v) > 100:

                return f"str({len(v)})"

            return f'"{v}"'

        elif isinstance(v, (int, float, bool)):

            return str(type(v).__name__)

        else:

            return type(v).__name__


    try:

        keys = sorted(payload.keys())

        items = []

        for k in keys:

            v = payload[k]

            items.append(f"{k}:{shape_value(v)}")

        return "{" + ",".join(items) + "}"

    except Exception:

        return "<?>"



def _log_unknown_server_event(

    *,

    msg_type: str,

    payload: dict[str, Any],

    last_outbound_event_type: str | None,

    last_outbound_event_safe_shape: str | None,

    events_sent_since_session_updated: int,

) -> None:

    """Emit a structured ``inworld_unknown_server_event`` log so we can iterate

    on which fields carry audio/transcripts for *this* Inworld build.


    The ReaAPI reference page doesn't list every event Inworld emits; without

    this signal we can't tell from logs whether a given shape was *parsed and

    ignored* vs *never received at all*. Always log keys + nested shapes;

    never log audio bytes.

    """

    def shape_obj(obj: Any, depth: int = 0) -> str:

        if depth > 4:

            return "..."

        if isinstance(obj, dict):

            return "{" + ",".join(f"{k}:{shape_obj(obj[k], depth+1)}" for k in sorted(obj.keys())) + "}"

        if isinstance(obj, list):

            inner = shape_obj(obj[0], depth+1) if obj else "[]"

            return f"[{len(obj)}]{inner}"

        if isinstance(obj, str):

            return f'str({len(obj)})' if len(obj) > 60 else f'"{obj}"'

        return type(obj).__name__


    try:

        logger.info(

            "inworld_unknown_server_event type=%s safe_payload=%s last_outbound=%s last_outbound_shape=%s events_sent=%s",

            msg_type or "unknown",

            shape_obj(payload),

            last_outbound_event_type or "none",

            last_outbound_event_safe_shape or "none",

            events_sent_since_session_updated,

        )

    except Exception:  # noqa: BLE001 - logging must never break the bridge

        logger.info("inworld_unknown_server_event type=%s shape=<unrenderable>", msg_type or "unknown")



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

        self._last_outbound_event_safe_shape: str | None = None

        self._events_sent_since_session_updated = 0

        self._last_inbound_event_time = time.monotonic()

        # Per-item_id Events any future code path may use for waiting on a

        # particular ``conversation.item.done``. The forced text test uses a

        # different mechanism (a state-machine of handler transitions) so that no

        # handler ever awaits inside the receive loop.

        self._pending_item_done_events: dict[str, asyncio.Event] = {}

        self._next_item_done_event: asyncio.Event | None = None

        self._last_response_done_at: float | None = None

        # Forced-text-test state machine. ``idle`` is the steady state. After

        # ``session.updated`` we transition to ``awaiting_item_done`` (we send

        # ``conversation.item.create`` and return immediately). When the receive

        # loop later sees ``conversation.item.done`` for that item, we transition

        # to ``awaiting_response_done`` (we send ``response.create``). When the

        # receive loop sees ``response.done``, we transition back to ``idle``.

        # CRITICAL: none of this awaits anything inside ``_handle_inworld_message``.

        # The receive loop MUST stay free to receive the very events the state

        # machine depends on.

        self._forced_test_phase: str = "idle"  # idle | awaiting_item_done | awaiting_response_done

        self._mic_forwarding_paused: bool = False

        # ``INWORLD_FORCE_TEXT_TEST`` gates the forced text test. Default is

        # ``True`` for now because we still need to prove the audio output path

        # end-to-end; once proven, set to ``False`` so each session doesn't pay

        # the cost of an extra round-trip at startup.

        self._force_text_test_enabled: bool = _env_bool("INWORLD_FORCE_TEXT_TEST", True)

        # ``INWORLD_FORCE_TEXT_TEST_ONLY`` is the smoke-test mode — after the

        # forced text test completes (response.done), close the bridge cleanly.

        # Lets us assert the full pipeline ends-to-end in a controlled window

        # without leaving a long-lived session that could mask regressions.

        self._force_text_test_only: bool = _env_bool("INWORLD_FORCE_TEXT_TEST_ONLY", False)

        # ``INWORLD_AGGRESSIVE_AUDIO_PROBE`` enables the Phase 2 fallback that

        # tries base64-decode on EVERY string value in the payload (catches audio

        # bytes living under non-standard field names). Useful for the smoke test

        # — too aggressive for production (a long transcript could in theory

        # base64-decode to ≥80 bytes and be treated as PCM). Defaults to

        # auto-enabled when ``INWORLD_FORCE_TEXT_TEST_ONLY`` is set (smoke test

        # mode) and disabled otherwise. Operator can override explicitly:

        #   INWORLD_AGGRESSIVE_AUDIO_PROBE=true  forces on in any environment

        #   INWORLD_AGGRESSIVE_AUDIO_PROBE=false forces off (narrow Phase 1 only)

        if "INWORLD_AGGRESSIVE_AUDIO_PROBE" in os.environ:

            self._aggressive_audio_probe: bool = _env_bool("INWORLD_AGGRESSIVE_AUDIO_PROBE", False)

        else:

            self._aggressive_audio_probe = self._force_text_test_only


    async def run(self) -> None:

        started_at = time.monotonic()

        logger.info(

            "inworld_realtime_bridge_started=true voice_engine_selected=inworld_realtime stt_model=%s tts_model=%s tts_voice=%s turn_detection=%s voice_profile_enabled=%s force_text_test_enabled=%s force_text_test_only=%s aggressive_audio_probe=%s",

            self.settings.stt_model,

            self.settings.tts_model,

            self.settings.voice,

            self.settings.turn_detection_type,

            self.settings.voice_profile_enabled,

            self._force_text_test_enabled,

            self._force_text_test_only,

            self._aggressive_audio_probe,

        )

        if self._aggressive_audio_probe and not self._force_text_test_only:

            logger.warning(

                "inworld_aggressive_audio_probe_enabled=true "

                "reason=INWORLD_AGGRESSIVE_AUDIO_PROBE_set_explicitly_outside_smoke_test "

                "action_required=set_INWORLD_AGGRESSIVE_AUDIO_PROBE=false_after_audio_path_is_proven "

                "Phase2_fallback=try_every_string_value_in_payload"

            )

        if self._force_text_test_enabled:

            logger.warning(

                "inworld_force_text_test_enabled_default_rollout=true "

                "reason=INWORLD_FORCE_TEXT_TEST_default_true_temporarily "

                "action_required=set_INWORLD_FORCE_TEXT_TEST=false_after_audio_path_is_proven "

                "set_INWORLD_FORCE_TEXT_TEST_ONLY=true_to_run_smoke_test_then_close"

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

            # Don't re-raise — the worker cancellation chain was the previous symptom.

            logger.error(

                "inworld_realtime_bridge_error=true error_type=%s error=%s last_outbound=%s audio_forwarded_count=%s",

                type(exc).__name__, exc,

                self._last_outbound_event_type or "none",

                self._audio_forwarded_count,

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

        dropped_during_force_test = 0

        dropped_before_session_ready = 0

        try:

            async for event in stream:

                # Defensive: do NOT send mic audio to Inworld until the session

                # is fully configured (``session.updated`` has fired and turned

                # ``_session_ready`` on). Before that point the WebSocket is

                # open but the server doesn't yet know what model/voice/VAD to

                # apply; sending ``input_audio_buffer.append`` too early can be

                # rejected or, worse, silently buffered and played under the

                # wrong config. Drop and log instead.

                if not self._session_ready.is_set():

                    dropped_before_session_ready += 1

                    if dropped_before_session_ready == 1 or dropped_before_session_ready % 50 == 0:

                        # Sample at frame #1 and every 50th so we don't spam logs.

                        logger.info(

                            "inworld_mic_audio_dropped_before_session_ready=true count=%s",

                            dropped_before_session_ready,

                        )

                    continue

                # During the forced text test we deliberately stop streaming the

                # mic to Inworld so the test itself is the only conversation in

                # flight — no echo, no double-trigger, no race with the real turn.

                if self._mic_forwarding_paused:

                    dropped_during_force_test += 1

                    if dropped_during_force_test == 1 or dropped_during_force_test % 50 == 0:

                        # Sample at frame #1 and every 50th so we don't spam logs.

                        logger.info(

                            "inworld_mic_audio_dropped_during_force_test=true count=%s phase=%s",

                            dropped_during_force_test, self._forced_test_phase,

                        )

                    continue

                frame = getattr(event, "frame", None)

                pcm = _frame_bytes(frame)

                if not pcm:

                    continue

                await self._send_inworld_message(build_audio_append_message(pcm), reason="audio_frame_from_livekit")

                self._audio_forwarded_count += 1

        finally:

            if dropped_before_session_ready > 0:

                logger.info(

                    "inworld_mic_audio_dropped_before_session_ready_final=true total_dropped=%s",

                    dropped_before_session_ready,

                )

            if dropped_during_force_test > 0:

                logger.info(

                    "inworld_mic_audio_dropped_during_force_test_final=true total_dropped=%s",

                    dropped_during_force_test,

                )

            await stream.aclose()


    async def _receive_inworld(self, ws: aiohttp.ClientWebSocketResponse) -> None:

        logger.info("inworld_receive_loop_started=true")

        try:

            async for msg in ws:

                if msg.type == aiohttp.WSMsgType.TEXT:

                    raw = msg.data

                    try:

                        payload = json.loads(raw)

                    except Exception as exc:

                        logger.error(

                            "inworld_raw_message_parse_error=true error=%s preview=%s",

                            exc,

                            raw[:120] if isinstance(raw, (str, bytes)) else str(raw)[:120],

                        )

                        continue


                    try:

                        await self._handle_inworld_message(payload)

                        self._last_inbound_event_time = time.monotonic()

                    except Exception as exc:  # noqa: BLE001 - never let handler errors kill the bridge

                        logger.error(

                            "inworld_message_handler_exception=true error_type=%s error=%s type=%s",

                            type(exc).__name__, exc, str(payload.get("type") or "?") if isinstance(payload, dict) else "?",

                        )

                elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:

                    logger.info("inworld_websocket_closed=true msg_type=%s", msg.type)

                    break

        except asyncio.CancelledError:

            logger.info("inworld_receive_task_cancelled=true")

            raise

        except Exception as exc:

            # Log loudly but DO NOT re-raise — the worker should never die from a

            # single-streaming-side fault. ``run()`` is informed via ``_closed``.

            logger.error(

                "inworld_receive_task_exception=true error_type=%s error=%s last_outbound=%s",

                type(exc).__name__, exc, self._last_outbound_event_type or "none",

            )

        finally:

            self._closed.set()


    async def _send_inworld_message(self, payload: dict[str, Any], reason: str = "unknown") -> None:

        """Send a message to Inworld with comprehensive logging."""

        if self._ws is None:

            return


        event_type = str(payload.get("type") or "unknown")

        top_level_keys = sorted(payload.keys())

        safe_shape = _safe_payload_shape(payload)

        has_audio = "audio" in payload and isinstance(payload.get("audio"), str)

        audio_len = len(payload.get("audio", "")) if has_audio else 0


        logger.info(

            "inworld_outbound_message event_type=%s reason=%s top_level_keys=%s safe_shape=%s has_audio=%s audio_base64_len=%s",

            event_type,

            reason,

            ",".join(top_level_keys),

            safe_shape,

            has_audio,

            audio_len if has_audio else "n/a",

        )


        try:

            await self._ws.send_json(payload)

            self._last_outbound_event_type = event_type

            self._last_outbound_event_safe_shape = safe_shape

            self._events_sent_since_session_updated += 1

            logger.info("inworld_message_sent=true event_type=%s", event_type)

        except Exception as exc:

            logger.error(

                "inworld_message_send_error=true event_type=%s error_type=%s error=%s safe_shape=%s",

                event_type,

                type(exc).__name__,

                exc,

                safe_shape,

            )

            raise


    async def _handle_inworld_message(self, payload: dict[str, Any]) -> None:

        msg_type = str(payload.get("type") or "")


        # Log server event type

        logger.info("inworld_server_event_received type=%s", msg_type)


        if msg_type == "session.created":

            logger.info("inworld_session_created=true")

            # Use the helper so we get the detailed ``inworld_session_update_sent=true``

            # log on every reply (model, voice, vad, voice profile). Sends via

            # ``_send_inworld_message`` which also logs outbound diagnostics.

            await self._send_session_update()


        elif msg_type == "session.updated":

            logger.info("inworld_session_updated=true")

            self._session_ready.set()

            self._events_sent_since_session_updated = 0

            # Fire-and-forget. ``_start_forced_text_test_if_enabled`` only sends

            # ``conversation.item.create`` and arms the state machine; it MUST

            # return quickly so the receive loop can keep processing future

            # events (including the very ``conversation.item.done`` we'll need).

            await self._start_forced_text_test_if_enabled()


        elif msg_type == "conversation.item.created":

            # Mostly informational — we wait for ``.done`` before triggering a response.

            item = payload.get("item") if isinstance(payload.get("item"), dict) else {}

            item_id = str(item.get("id") or payload.get("item_id") or "")

            logger.info("inworld_conversation_item_created=true item_id=%s role=%s", item_id, item.get("role", "?"))


        elif msg_type == "conversation.item.done":

            item = payload.get("item") if isinstance(payload.get("item"), dict) else {}

            item_id = str(item.get("id") or payload.get("item_id") or "")

            role = str(item.get("role") or "?")

            logger.info(

                "inworld_conversation_item_done=true item_id=%s role=%s safe_shape=%s",

                item_id, role, _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

            )

            if item_id and item_id in self._pending_item_done_events:

                self._pending_item_done_events[item_id].set()

                # Don't hold the registry forever; drop entries the caller never consumed.

                self._pending_item_done_events.pop(item_id, None)

            # One-shot broadcast for any awaiters.

            if self._next_item_done_event is not None:

                self._next_item_done_event.set()


            # Forced text test state machine: when we're waiting for ``item.done``

            # to fire ``response.create``, do it HERE — NOT from a top-level awaiter

            # inside the receive handler. The receive loop must stay free to

            # process the very events the state machine waits on.

            if self._forced_test_phase == "awaiting_item_done":

                try:

                    response_create = build_response_create(

                        instructions="Reply out loud in one short sentence.",

                    )

                    await self._send_inworld_message(

                        response_create, reason="forced_text_test_after_item_done"

                    )

                    self._forced_test_phase = "awaiting_response_done"

                    logger.info(

                        "inworld_response_create_sent=true forced_test_phase=awaiting_response_done"

                    )

                except Exception as exc:  # noqa: BLE001

                    logger.error(

                        "inworld_response_create_error=true error_type=%s error=%s",

                        type(exc).__name__, exc,

                    )

                    self._forced_test_phase = "idle"

                    self._mic_forwarding_paused = False


        elif msg_type == "conversation.item.input_audio_transcription.delta":

            delta = str(payload.get("delta") or "")

            logger.info("inworld_transcription_delta=true delta_length=%s", len(delta))


        elif msg_type == "conversation.item.input_audio_transcription.completed":

            transcript = str(payload.get("transcript") or "")

            logger.info("inworld_transcription_completed=true transcript_length=%s", len(transcript))


        elif msg_type in {

            "input_audio_buffer.speech_started",

            "input_audio_buffer.speech_stopped",

            "input_audio_buffer.committed",

            "input_audio_buffer.cleared",

            "input_audio_buffer.turn_suggestion",

        }:

            logger.info("inworld_vad_event=true event_type=%s", msg_type)


        elif msg_type == "response.created":

            logger.info("inworld_response_created=true")


        elif msg_type == "response.output_item.added":

            # Treat as audio-probable for widest diagnostic net — a future

            # Inworld build could place audio bytes in the item-level envelope.

            summary = _event_audio_payload_summary(payload)

            logger.info(

                "inworld_response_output_item_added=true safe_shape=%s long_string_candidates=%s",

                _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

                summary,

            )


        elif msg_type in {

            "response.output_audio.delta",

            "response.audio.delta",

            "response.content_part.added",

            "response.content_part.done",

            "response.audio",

            "response.output_audio_buffer",

            "response.output_audio.done",

            "response.audio.done",

        }:

            # One handler for any event that might carry the audio bytes. The

            # docs don't name a single event for ``inworld-tts-2`` audio, and

            # some pipelines stash the full audio in the snapshot rather than

            # the delta. Always log the top-N longest strings (``inworld_audio_

            # probe_summary``) so we can see exactly what's in the payload even

            # when our extractor returns zero bytes.

            summary = _event_audio_payload_summary(payload)

            logger.info(

                "inworld_audio_probe_summary=true event=%s long_string_candidates=%s",

                msg_type, summary,

            )

            await self._write_inworld_audio_to_livekit(payload, msg_type)



        elif msg_type == "response.output_audio_transcript.delta":

            # Transcript events are unlikely to carry audio bytes themselves,

            # but logging the top-N long strings confirms that (and would

            # surprise us if Inworld ever put PCM in here).

            summary = _event_audio_payload_summary(payload)

            logger.info(

                "inworld_response_audio_transcript_delta=true safe_shape=%s long_string_candidates=%s",

                _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

                summary,

            )


        elif msg_type == "response.output_audio_transcript.done":

            summary = _event_audio_payload_summary(payload)

            logger.info(

                "inworld_response_audio_transcript_done=true safe_shape=%s long_string_candidates=%s",

                _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

                summary,

            )


        elif msg_type == "response.output_text.delta":

            logger.info(

                "inworld_response_output_text_delta=true safe_shape=%s",

                _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

            )


        elif msg_type == "response.output_text.done":

            logger.info(

                "inworld_response_output_text_done=true safe_shape=%s",

                _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

            )


        elif msg_type == "response.output_item.done":

            # Treat as audio-probable for widest diagnostic net.

            summary = _event_audio_payload_summary(payload)

            logger.info(

                "inworld_response_output_item_done=true safe_shape=%s long_string_candidates=%s",

                _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

                summary,

            )


        elif msg_type in {

            "output_audio_buffer.started",

            "output_audio_buffer.stopped",

            "output_audio_buffer.cleared",

            "output_audio_buffer.committed",

        }:

            logger.info(

                "inworld_output_audio_buffer=true event=%s safe_shape=%s",

                msg_type, _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

            )


        elif msg_type in {

            "response.backchannel.audio.delta",

            "response.backchannel.audio.done",

            "response.backchannel.delta",

        }:

            # Backchannel audio / acknowledgments — log shape only; we don't try

            # to play these to the user. If they ever show up carrying real

            # audio bytes, the shape log makes that diagnosable.

            logger.info(

                "inworld_backchannel_audio=true event=%s safe_shape=%s",

                msg_type, _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

            )


        elif msg_type == "response.done":

            self._last_response_done_at = time.monotonic()

            logger.info(

                "inworld_response_done=true audio_forwarded_count=%s elapsed_seconds=%.3f safe_shape=%s forced_test_phase_before=%s",

                self._audio_forwarded_count,

                time.monotonic() - (self._last_inbound_event_time or time.monotonic()),

                _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

                self._forced_test_phase,

            )

            # ``response.done`` carries the FULL response snapshot. Some Inworld

            # pipelines bake the audio bytes into this snapshot (rather than

            # streaming them in delta events). Probe it like any audio event.

            summary = _event_audio_payload_summary(payload)

            logger.info(

                "inworld_audio_probe_summary=true event=response.done long_string_candidates=%s",

                summary,

            )

            await self._write_inworld_audio_to_livekit(payload, msg_type)

            # Forced text test state machine: complete and unpause mic.

            if self._forced_test_phase == "awaiting_response_done":

                self._forced_test_phase = "idle"

                self._mic_forwarding_paused = False

                logger.info("inworld_force_text_test_completed=true mic_forwarding_resumed=true")

                # Smoke-test mode: close cleanly after the forced test so we can

                # assert the pipeline ends-to-end without leaving a session open.

                # ``run()`` is awaiting ``self._closed`` so flipping it here wakes

                # the run loop and lets it fall into the finally block.

                if self._force_text_test_only:

                    logger.info(

                        "inworld_realtime_bridge_closing_after_forced_test=true "

                        "close_reason=forced_text_test_completed "

                        "voice_engine_entrypoint_completed=true"

                    )

                    self._closed.set()

            # Clearing residual pending ``conversation.item.done`` waiters so a stale

            # response doesn't deadlock a future test run.

            for ev in self._pending_item_done_events.values():

                ev.set()

            self._pending_item_done_events.clear()


        elif msg_type == "error":

            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}

            logger.error(

                "inworld_server_error=true code=%s message=%s param=%s last_outbound_event_type=%s last_outbound_event_safe_shape=%s events_sent_since_session_updated=%s",

                error.get("code"),

                error.get("message"),

                error.get("param"),

                self._last_outbound_event_type or "none",

                self._last_outbound_event_safe_shape or "none",

                self._events_sent_since_session_updated,

            )


        else:

            # Discovery: every unhandled event still gets logged with a structured

            # shape so we can iterate on which fields actually carry audio.

            _log_unknown_server_event(

                msg_type=msg_type,

                payload=payload if isinstance(payload, dict) else {"value": payload},

                last_outbound_event_type=self._last_outbound_event_type,

                last_outbound_event_safe_shape=self._last_outbound_event_safe_shape,

                events_sent_since_session_updated=self._events_sent_since_session_updated,

            )


    async def _write_inworld_audio_to_livekit(self, payload: dict[str, Any], source_event: str) -> None:

        """Decode PCM out of any payload and push it to the LiveKit output source.


        ``_aggressive_audio_probe`` controls whether the Phase 2 "try every

        string value" fallback is enabled (smoke test mode) or skipped

        (production, narrow Phase 1 only).

        """

        pcm = _event_audio_bytes(payload, aggressive=self._aggressive_audio_probe)

        if not pcm:

            logger.info(

                "inworld_audio_no_bytes_decoded=true event=%s safe_shape=%s",

                source_event,

                _safe_payload_shape(payload) if isinstance(payload, dict) else "<!>",

            )

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

                "inworld_audio_write_error=true event=%s error_type=%s error=%s frame_count=%s",

                source_event, type(exc).__name__, exc, frame_count,

            )

            return


        self._audio_forwarded_count += frame_count

        logger.info(

            "inworld_audio_written_to_livekit=true event=%s frames=%s pcm_bytes=%s samples_per_frame=%s sample_rate=%s channels=%s audio_forwarded_total_frames=%s",

            source_event,

            frame_count,

            len(pcm),

            INWORLD_OUTPUT_SAMPLE_RATE * INWORLD_FRAME_MS // 1000,

            INWORLD_OUTPUT_SAMPLE_RATE,

            INWORLD_CHANNELS,

            self._audio_forwarded_count,

        )


    async def _send_session_update(self) -> None:

        """Send session.update after session.created."""

        if self._ws is None:

            return


        update = build_session_update(self.settings)

        await self._send_inworld_message(update, reason="session_created")


        # TTS provider-data confirmation — emitted so we can prove from logs

        # that the new ``providerData.tts`` block was included in the

        # session.update we just sent. Logs only the Inworld-controlled keys

        # we pass in (delivery_mode / segmenter_strategy / steering_handling)

        # and their values — no secrets, no audio bytes.

        logger.info(

            "inworld_tts_provider_data_enabled=true delivery_mode=%s segmenter_strategy=%s steering_handling=%s",

            self.settings.tts_delivery_mode,

            self.settings.tts_segmenter_strategy,

            self.settings.tts_steering_handling,

        )


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


    async def _start_forced_text_test_if_enabled(self) -> None:

        """Send ``conversation.item.create`` and arm the state machine. Returns fast.


        The previous build of this method awaited ``conversation.item.done`` inside

        ``_handle_inworld_message`` — but ``_handle_inworld_message`` is the receive

        loop handler, so awaiting inside it blocked the loop from receiving the very

        event we were waiting on. Fix: only send the outbound request here; the

        ``conversation.item.done`` and ``response.done`` handlers (separate handler

        invocations, each on a single event) advance the state machine and send the

        remaining outbound messages.

        """

        if not self._force_text_test_enabled:

            logger.info(

                "inworld_force_text_test_skipped=true reason=INWORLD_FORCE_TEXT_TEST_disabled"

            )

            return


        if self._ws is None:

            logger.info("inworld_force_text_test_skipped=true reason=ws_none")

            return


        if self._forced_test_phase != "idle":

            logger.info(

                "inworld_force_text_test_skipped=true reason=already_running phase=%s",

                self._forced_test_phase,

            )

            return


        try:

            item_create = build_conversation_item_create("Say hello in one short sentence.")

            await self._send_inworld_message(item_create, reason="forced_text_test")

            self._forced_test_phase = "awaiting_item_done"

            self._mic_forwarding_paused = True

            logger.info(

                "inworld_force_text_test_sent=true phase=awaiting_item_done mic_forwarding_paused=true"

            )

        except Exception as exc:  # noqa: BLE001

            logger.error(

                "inworld_force_text_test_item_create_error=true error_type=%s error=%s",

                type(exc).__name__, exc,

            )

            self._forced_test_phase = "idle"

            self._mic_forwarding_paused = False



async def run_inworld_realtime_bridge(room: rtc.Room, *, instructions: str | None = None) -> None:

    settings = load_inworld_realtime_settings(instructions=instructions)

    bridge = InworldRealtimeLiveKitBridge(room, settings)

    await bridge.run()



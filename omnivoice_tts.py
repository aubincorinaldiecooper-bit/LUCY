"""OmniVoice TTS provider (HTTP sidecar).

OmniVoice (https://github.com/k2-fsa/OmniVoice) is a heavy local inference model,
so rather than loading it inside the realtime LiveKit worker we talk to it as a
**sidecar HTTP service**: a separate (GPU) host runs OmniVoice and exposes a small
synth endpoint, and the worker calls it over HTTP. This keeps the worker light,
lets TTS scale/upgrade independently, and makes failover trivial — if the sidecar
errors, times out, or returns invalid audio, LiveKit's ``tts.FallbackAdapter``
moves on to Hume without crashing the session.

Sidecar HTTP contract (the service must implement this):

    POST {OMNIVOICE_URL}/synthesize
      headers: Authorization: Bearer {OMNIVOICE_API_KEY}   (only if a key is set)
      json body:
        {
          "text": "<text to speak>",
          "voice": "<preset id>" | null,
          "language": "en",
          "sample_rate": 24000,
          "audio_format": "wav" | "pcm_s16le",
          "expressive_tags": true,
          "device": "cuda" | "cpu",        # informational; sidecar owns the model
          "model_path": "<OMNIVOICE_MODEL_PATH>"  # informational
        }
      200 response: the synthesized audio bytes (WAV container, or raw little-endian
        s16 PCM mono at sample_rate when audio_format=pcm_s16le). May be chunked.
      non-200: treated as a provider failure -> fall back to Hume.

This module keeps the pure, testable pieces (config parsing, request payload,
audio validation, mime mapping) separate from the thin LiveKit glue
(``OmniVoiceTTS`` / ``_OmniVoiceChunkedStream``) so the contract can be unit-tested
without a live sidecar or GPU.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

logger = logging.getLogger("agent.omnivoice")

DEFAULT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1
# Below this many bytes a 200 response is treated as "invalid audio" (essentially
# empty) and we fall back rather than emit a click of silence. ~2KB ≈ 40ms of
# s16 PCM @24k, well under any real utterance.
DEFAULT_MIN_AUDIO_BYTES = 2048


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class OmniVoiceError(Exception):
    """Raised for an invalid/empty synthesis result from the sidecar."""


@dataclass
class OmniVoiceConfig:
    enabled: bool
    base_url: str
    api_key: str
    device: str
    model_path: str
    default_language: str
    expressive_tags_enabled: bool
    timeout_seconds: float
    sample_rate: int
    audio_format: str  # "wav" | "pcm_s16le"
    min_audio_bytes: int

    @classmethod
    def from_env(cls) -> "OmniVoiceConfig":
        return cls(
            enabled=_env_bool("OMNIVOICE_ENABLED", False),
            base_url=(os.getenv("OMNIVOICE_URL") or "").strip().rstrip("/"),
            api_key=(os.getenv("OMNIVOICE_API_KEY") or "").strip(),
            device=(os.getenv("OMNIVOICE_DEVICE") or "cuda").strip().lower(),
            model_path=(os.getenv("OMNIVOICE_MODEL_PATH") or "").strip(),
            default_language=(os.getenv("OMNIVOICE_DEFAULT_LANGUAGE") or "en").strip(),
            expressive_tags_enabled=_env_bool("OMNIVOICE_EXPRESSIVE_TAGS_ENABLED", True),
            timeout_seconds=float(os.getenv("OMNIVOICE_TIMEOUT_SECONDS", "10") or "10"),
            sample_rate=int(os.getenv("OMNIVOICE_SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE)) or DEFAULT_SAMPLE_RATE),
            audio_format=(os.getenv("OMNIVOICE_AUDIO_FORMAT") or "wav").strip().lower(),
            min_audio_bytes=int(
                os.getenv("OMNIVOICE_MIN_AUDIO_BYTES", str(DEFAULT_MIN_AUDIO_BYTES))
                or DEFAULT_MIN_AUDIO_BYTES
            ),
        )

    def is_usable(self) -> tuple[bool, str]:
        """Whether the sidecar can be called. Returns (ok, reason_if_not)."""
        if not self.enabled:
            return False, "omnivoice_disabled"
        if not self.base_url:
            return False, "omnivoice_url_missing"
        return True, "ok"


def mime_for_format(audio_format: str) -> str:
    """Map a configured audio format to the mime type the AudioEmitter expects."""
    fmt = (audio_format or "").strip().lower()
    if fmt in {"pcm", "pcm_s16le", "raw"}:
        return "audio/pcm"
    if fmt in {"wav", "wave"}:
        return "audio/wav"
    if fmt == "mp3":
        return "audio/mpeg"
    # Default to a self-describing container so the emitter can parse it.
    return "audio/wav"


def build_synthesis_payload(
    text: str,
    *,
    config: OmniVoiceConfig,
    voice: str | None,
    language: str | None,
) -> dict:
    """Build the JSON body for a sidecar /synthesize call (pure / testable)."""
    return {
        "text": text,
        "voice": voice or None,
        "language": (language or config.default_language or "en"),
        "sample_rate": config.sample_rate,
        "audio_format": config.audio_format,
        "expressive_tags": config.expressive_tags_enabled,
        "device": config.device,
        "model_path": config.model_path or None,
    }


def validate_audio(data: bytes | None, *, min_bytes: int) -> None:
    """Raise OmniVoiceError if the sidecar's audio is empty/too short to be real."""
    n = len(data) if data else 0
    if n < max(1, min_bytes):
        raise OmniVoiceError(f"omnivoice returned invalid audio: bytes={n} min={min_bytes}")


async def synthesize_via_sidecar(
    session: aiohttp.ClientSession,
    *,
    text: str,
    config: OmniVoiceConfig,
    voice: str | None,
    language: str | None,
) -> bytes:
    """POST to the sidecar, validate, and return the audio bytes.

    On any failure raises a LiveKit API error (APITimeoutError / APIStatusError /
    APIConnectionError) so a wrapping tts.FallbackAdapter fails over to Hume.
    Kept independent of ChunkedStream so the failure/fallback behaviour is unit
    testable with a mocked session.
    """
    payload = build_synthesis_payload(text, config=config, voice=voice, language=language)
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    url = f"{config.base_url}/synthesize"
    try:
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise APIStatusError(
                    f"omnivoice sidecar status {resp.status}: {body}",
                    status_code=resp.status,
                    request_id=None,
                    body=body,
                )
            audio = await resp.read()
    except (TimeoutError, aiohttp.ServerTimeoutError) as e:
        raise APITimeoutError() from e
    except (APIStatusError, APITimeoutError, APIConnectionError):
        raise
    except aiohttp.ClientError as e:
        raise APIConnectionError(f"omnivoice sidecar connection error: {e}") from e

    try:
        validate_audio(audio, min_bytes=config.min_audio_bytes)
    except OmniVoiceError as e:
        # Empty/garbage audio is a soft failure -> fall back to Hume.
        raise APIConnectionError(str(e)) from e
    return audio


class OmniVoiceTTS(tts.TTS):
    """LiveKit TTS adapter that synthesizes via the OmniVoice sidecar.

    Capabilities advertise non-streaming synthesis (we buffer a full utterance and
    validate it before emitting, so we never push half a clip and then fall back).
    ``voice`` / ``language`` are set per session and can be swapped at runtime via
    ``update_options`` (used by the voice-pool / language-switching work).
    """

    def __init__(
        self,
        *,
        config: OmniVoiceConfig,
        voice: str | None = None,
        language: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=config.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._config = config
        self._voice = voice
        self._language = language or config.default_language
        self._http_session = http_session
        self._owns_session = http_session is None

    @property
    def provider(self) -> str:
        return "omnivoice"

    @property
    def model(self) -> str:
        return os.path.basename(self._config.model_path) or "omnivoice"

    def update_options(self, *, voice: str | None = None, language: str | None = None) -> None:
        if voice is not None:
            self._voice = voice
        if language is not None:
            self._language = language

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
            self._owns_session = True
        return self._http_session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "tts.ChunkedStream":
        return _OmniVoiceChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._owns_session and self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()


class _OmniVoiceChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: OmniVoiceTTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: OmniVoiceTTS = tts

    async def _run(self, output_emitter: "tts.AudioEmitter") -> None:
        cfg = self._tts._config
        audio = await synthesize_via_sidecar(
            self._tts._ensure_session(),
            text=self.input_text,
            config=cfg,
            voice=self._tts._voice,
            language=self._tts._language,
        )
        output_emitter.initialize(
            request_id=shortuuid("omnivoice_"),
            sample_rate=cfg.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=mime_for_format(cfg.audio_format),
        )
        output_emitter.push(audio)
        output_emitter.flush()


def find_omnivoice_tts(tts_obj: object) -> "OmniVoiceTTS | None":
    """Return the OmniVoiceTTS inside a session TTS, whether bare or wrapped in a
    FallbackAdapter, so callers can update its voice/language at runtime."""
    if isinstance(tts_obj, OmniVoiceTTS):
        return tts_obj
    children = getattr(tts_obj, "_tts_instances", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            if isinstance(child, OmniVoiceTTS):
                return child
    return None

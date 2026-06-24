"""Inworld voice-profile layer: weak emotional/vocal context, never user-facing.

Inworld's streaming STT (model ``inworld/inworld-stt-1``) returns a ``voiceProfile``
alongside each transcript — emotion / vocalStyle / accent / age / pitch, each an
array of ``{label, confidence}`` sorted by confidence — when
``inworldConfig.voiceProfileThreshold`` is set. We normalize that raw profile into
the weak-signal schema the planner consumes:

    {energy, tension, certainty, emotion_confidence, pitch, vocal_style, accent}

Hard rules baked in here:
  - raw emotion labels (sad/angry/…) are NEVER surfaced to the planner summary or
    the user — we only expose the derived dims + pitch/vocal_style/accent, so the
    model can't parrot "you sound sad",
  - it's a weak signal: low-confidence emotion collapses to neutral so it can't
    force expressive tags downstream.

Pure (config + message builders + parse + normalize) so it's unit testable without
a live WebSocket. Inworld voice profiling is English-only today; for other
languages this simply yields a neutral profile.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass

# Inworld emotion labels -> derived dimensions. Raw labels stay internal.
_ENERGY = {
    "happy": "high", "angry": "high", "surprised": "high",
    "sad": "low", "calm": "low", "tender": "low",
    "frustrated": "medium", "fearful": "medium",
}
_TENSION = {
    "angry": "high", "fearful": "high", "frustrated": "high",
    "calm": "low", "tender": "low",
    "happy": "medium", "sad": "medium", "surprised": "medium",
}
_CERTAINTY = {
    "happy": "high", "calm": "high", "angry": "high",
    "fearful": "low", "surprised": "low", "sad": "low",
    "tender": "medium", "frustrated": "medium",
}
_NEUTRAL = {"energy": "medium", "tension": "medium", "certainty": "medium"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class InworldConfig:
    enabled: bool
    ws_url: str
    api_key: str
    model_id: str
    voice_profile_threshold: float
    sample_rate: int
    # Below this, the top emotion is treated as unreliable -> neutral profile.
    emotion_confidence_floor: float

    @classmethod
    def from_env(cls) -> "InworldConfig":
        return cls(
            enabled=_env_bool("INWORLD_VOICE_PROFILE_ENABLED", False),
            ws_url=(os.getenv("INWORLD_STT_WS_URL")
                    or "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional").strip(),
            api_key=(os.getenv("INWORLD_API_KEY") or "").strip(),
            model_id=(os.getenv("INWORLD_STT_MODEL_ID") or "inworld/inworld-stt-1").strip(),
            voice_profile_threshold=float(os.getenv("INWORLD_VOICE_PROFILE_THRESHOLD", "0.5") or "0.5"),
            sample_rate=int(os.getenv("INWORLD_STT_SAMPLE_RATE", "16000") or "16000"),
            emotion_confidence_floor=float(
                os.getenv("INWORLD_EMOTION_CONFIDENCE_FLOOR", "0.5") or "0.5"
            ),
        )

    def is_usable(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "inworld_disabled"
        if not self.api_key:
            return False, "inworld_api_key_missing"
        if not self.ws_url:
            return False, "inworld_ws_url_missing"
        return True, "ok"


@dataclass
class NormalizedVoiceProfile:
    energy: str = "medium"
    tension: str = "medium"
    certainty: str = "medium"
    emotion_confidence: float = 0.0
    pitch: str = ""
    vocal_style: str = ""
    accent: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def planner_summary(self) -> str:
        """Short neutral string for the planner. No raw emotion label, ever."""
        parts = [
            f"energy {self.energy}",
            f"tension {self.tension}",
            f"certainty {self.certainty}",
        ]
        if self.pitch:
            parts.append(f"pitch {self.pitch}")
        if self.vocal_style:
            parts.append(f"vocal style {self.vocal_style}")
        return ", ".join(parts)


NEUTRAL_PROFILE = NormalizedVoiceProfile()


def build_config_message(config: InworldConfig, *, enable_language_detection: bool = False) -> str:
    """The transcribe_config WS frame that turns on voice profiling."""
    return json.dumps(
        {
            "transcribe_config": {
                "modelId": config.model_id,
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": config.sample_rate,
                "numberOfChannels": 1,
                "enableLanguageDetection": enable_language_detection,
                "inworldConfig": {"voiceProfileThreshold": config.voice_profile_threshold},
            }
        }
    )


def build_audio_chunk_message(pcm: bytes) -> str:
    return json.dumps({"audio_chunk": {"content": base64.b64encode(pcm).decode()}})


def _profile_node(message: dict) -> dict | None:
    """Find the voiceProfile object regardless of camel/snake nesting."""
    if not isinstance(message, dict):
        return None
    result = message.get("result", message)
    if not isinstance(result, dict):
        return None
    for key in ("voiceProfile", "voice_profile"):
        node = result.get(key)
        if isinstance(node, dict):
            return node
    return None


def _top_label(profile: dict, *keys: str) -> tuple[str, float]:
    """Top (label, confidence) for the first present key; arrays are conf-sorted."""
    for key in keys:
        arr = profile.get(key)
        if isinstance(arr, list) and arr:
            best = max(
                arr,
                key=lambda e: float(e.get("confidence", 0.0)) if isinstance(e, dict) else 0.0,
            )
            if isinstance(best, dict) and best.get("label"):
                return str(best["label"]).strip().lower(), float(best.get("confidence", 0.0))
    return "", 0.0


def normalize_voice_profile(
    profile: dict | None, *, emotion_confidence_floor: float = 0.5
) -> NormalizedVoiceProfile:
    """Map a raw Inworld voiceProfile into the weak-signal schema."""
    if not isinstance(profile, dict) or not profile:
        return NormalizedVoiceProfile()

    emotion, emo_conf = _top_label(profile, "emotion")
    pitch, _ = _top_label(profile, "pitch")
    vocal_style, _ = _top_label(profile, "vocalStyle", "vocal_style")
    accent, _ = _top_label(profile, "accent")

    # Weak signal: an unreliable emotion read collapses the derived dims to neutral
    # so it can never force an expressive tag.
    if emotion and emo_conf >= emotion_confidence_floor:
        energy = _ENERGY.get(emotion, "medium")
        tension = _TENSION.get(emotion, "medium")
        certainty = _CERTAINTY.get(emotion, "medium")
    else:
        energy, tension, certainty = _NEUTRAL["energy"], _NEUTRAL["tension"], _NEUTRAL["certainty"]

    # Vocal style nudges certainty (mumbling/whispering = less certain).
    if vocal_style in {"mumbling", "whispering"}:
        certainty = "low"
    elif vocal_style == "shouting":
        certainty = "high"

    return NormalizedVoiceProfile(
        energy=energy,
        tension=tension,
        certainty=certainty,
        emotion_confidence=round(emo_conf, 3),
        pitch=pitch,
        vocal_style=vocal_style,
        accent=accent,
    )


def normalize_from_message(
    message: dict, *, emotion_confidence_floor: float = 0.5
) -> NormalizedVoiceProfile:
    """Extract + normalize a voice profile from a raw STT response message."""
    return normalize_voice_profile(
        _profile_node(message), emotion_confidence_floor=emotion_confidence_floor
    )

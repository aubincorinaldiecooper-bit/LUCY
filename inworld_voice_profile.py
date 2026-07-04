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

import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

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


@dataclass
class NormalizedVoiceProfile:
    energy: str = "medium"
    tension: str = "medium"
    certainty: str = "medium"
    confidence: float = 0.0
    pitch: str = ""
    vocal_style: str = ""
    accent: str = ""

    @property
    def emotion_confidence(self) -> float:
        return self.confidence

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
        confidence=round(emo_conf, 3),
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

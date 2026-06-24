"""OmniVoice rotating voice-preset pool.

We never synthesize a brand-new voice from scratch per session. Instead a curated
pool of presets (warm/young/natural/conversational voices) is selected from once
per session and kept stable for that whole session. Rotation across sessions can
be random, weighted (by preset weight), or round_robin.

The pool is a JSON file at OMNIVOICE_VOICE_POOL_PATH, either a bare list or
``{"presets": [...]}``; each preset is ``{"id", "name"?, "weight"?, "language"?,
"tags"?}``. Only ``id`` is required (it's what we hand OmniVoice as the voice).

Pure selection logic (load + pick) is kept separate from the per-session/per-worker
state (VoicePoolSelector) so it can be unit tested deterministically with a seeded
RNG and an explicit round-robin index.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field

logger = logging.getLogger("agent.omnivoice")

VALID_ROTATIONS = ("random", "weighted", "round_robin")


@dataclass(frozen=True)
class VoicePreset:
    id: str
    name: str = ""
    weight: float = 1.0
    language: str | None = None
    tags: tuple[str, ...] = ()


@dataclass
class VoicePoolConfig:
    enabled: bool
    path: str
    rotation: str

    @classmethod
    def from_env(cls) -> "VoicePoolConfig":
        rotation = (os.getenv("OMNIVOICE_VOICE_ROTATION") or "random").strip().lower()
        if rotation not in VALID_ROTATIONS:
            rotation = "random"
        enabled_raw = (os.getenv("OMNIVOICE_VOICE_POOL_ENABLED") or "").strip().lower()
        return cls(
            enabled=enabled_raw in {"1", "true", "yes", "on"},
            path=(os.getenv("OMNIVOICE_VOICE_POOL_PATH") or "").strip(),
            rotation=rotation,
        )


class VoicePoolError(Exception):
    """Raised when the configured pool can't be loaded into usable presets."""


def _coerce_preset(raw: dict) -> VoicePreset:
    pid = str(raw.get("id") or "").strip()
    if not pid:
        raise VoicePoolError("voice preset missing required 'id'")
    tags = raw.get("tags") or []
    if not isinstance(tags, (list, tuple)):
        tags = []
    try:
        weight = float(raw.get("weight", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    return VoicePreset(
        id=pid,
        name=str(raw.get("name") or ""),
        weight=weight,
        language=(str(raw["language"]).strip() if raw.get("language") else None),
        tags=tuple(str(t) for t in tags),
    )


def load_voice_pool(path: str) -> list[VoicePreset]:
    """Load and validate presets from a JSON file. Raises VoicePoolError."""
    if not path:
        raise VoicePoolError("voice pool path is empty")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise VoicePoolError(f"voice pool file not found: {path}") from e
    except (OSError, json.JSONDecodeError) as e:
        raise VoicePoolError(f"voice pool file unreadable: {e}") from e

    raw_list = data.get("presets") if isinstance(data, dict) else data
    if not isinstance(raw_list, list) or not raw_list:
        raise VoicePoolError("voice pool has no presets")
    presets = [_coerce_preset(r) for r in raw_list if isinstance(r, dict)]
    if not presets:
        raise VoicePoolError("voice pool has no usable presets")
    return presets


def select_preset(
    presets: list[VoicePreset],
    *,
    rotation: str,
    rng: random.Random,
    rr_index: int = 0,
) -> VoicePreset:
    """Pick one preset (pure; round_robin is deterministic given rr_index)."""
    if not presets:
        raise VoicePoolError("cannot select from an empty pool")
    if rotation == "round_robin":
        return presets[rr_index % len(presets)]
    if rotation == "weighted":
        weights = [max(0.0, p.weight) for p in presets]
        if sum(weights) <= 0:
            return rng.choice(presets)
        return rng.choices(presets, weights=weights, k=1)[0]
    return rng.choice(presets)


@dataclass
class VoicePoolSelector:
    """Loads the pool once and selects one stable preset per session call.

    Holds the round-robin cursor so rotation advances across sessions within a
    worker process. A failed/empty/disabled pool simply yields ``None`` (the
    caller keeps OmniVoice's default voice) rather than raising into the session.
    """

    config: VoicePoolConfig
    rng: random.Random = field(default_factory=random.Random)
    _presets: list[VoicePreset] = field(default_factory=list)
    _rr_index: int = 0
    load_error: str | None = None

    def __post_init__(self) -> None:
        if self.config.enabled and self.config.path:
            try:
                self._presets = load_voice_pool(self.config.path)
            except VoicePoolError as e:
                self.load_error = str(e)
                logger.warning(
                    "omnivoice_voice_pool_load_failed=true rotation=%s error=%s",
                    self.config.rotation,
                    e,
                )

    def available(self) -> bool:
        return bool(self._presets)

    def select(self) -> VoicePreset | None:
        if not self._presets:
            return None
        preset = select_preset(
            self._presets,
            rotation=self.config.rotation,
            rng=self.rng,
            rr_index=self._rr_index,
        )
        self._rr_index += 1
        return preset


_GLOBAL_SELECTOR: VoicePoolSelector | None = None


def get_session_selector() -> VoicePoolSelector:
    """Process-wide selector so round_robin advances across sessions."""
    global _GLOBAL_SELECTOR
    if _GLOBAL_SELECTOR is None:
        _GLOBAL_SELECTOR = VoicePoolSelector(VoicePoolConfig.from_env())
    return _GLOBAL_SELECTOR

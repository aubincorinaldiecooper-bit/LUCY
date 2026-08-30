"""Shared environment-variable parsing. Leaf module: imports nothing in-repo.

One canonical boolean parser for the modules that accept the documented
true/false spellings plus the common 1/yes/on variants. Deliberately NOT
adopted by the loaders that accept only the literal "true"
(vision_context, mood_context, arche_api, memory_layer's own flags):
widening their accepted spellings would flip flags for operators using
undocumented values, which is a product decision, not a cleanup.
"""

from __future__ import annotations

import os


def env_bool(name: str, default: bool = False) -> bool:
    """True for "1"/"true"/"yes"/"on" (any case); unset or blank -> default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

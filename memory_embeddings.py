"""Text embeddings for semantic memory recall (OpenAI -> pgvector).

An "embedding" turns text into a list of numbers that captures its meaning;
texts with similar meaning get similar number-lists. We embed each stored memory
and, at recall time, embed the user's current message and ask Postgres (pgvector)
for the memories whose vectors are closest — i.e. the most *relevant* memories,
not just the most recent.

This module is the only place that calls the embedding API, kept tiny and
fault-tolerant: any failure (no key, network, quota) returns None so the memory
layer silently falls back to recency recall — it never breaks a turn.

Requires OPENAI_API_KEY. Default model text-embedding-3-small (1536 dims), which
must match the vector(1536) column in migration 005.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("agent.memory")

EMBED_MODEL = (os.getenv("MEMORY_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
EMBED_DIM = int(os.getenv("MEMORY_EMBEDDING_DIM", "1536") or "1536")


def semantic_retrieval_enabled() -> bool:
    """Whether pgvector semantic recall is turned on (default off)."""
    return (os.getenv("MEMORY_SEMANTIC_RETRIEVAL_ENABLED", "false").strip().lower()
            in {"1", "true", "yes", "on"})


def semantic_timeout_ms() -> int:
    """Total budget for embed + vector query during a turn (default 800ms)."""
    try:
        return max(100, int(os.getenv("MEMORY_SEMANTIC_TIMEOUT_MS", "800")))
    except (TypeError, ValueError):
        return 800


def to_pgvector_literal(vector: list[float]) -> str:
    """Format a vector as a pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


async def embed_text(text: str, *, client: Any | None = None) -> list[float] | None:
    """Return the embedding for `text`, or None on any failure (graceful)."""
    text = (text or "").strip()
    if not text:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if client is None and not api_key:
        logger.warning("memory_embedding_skipped=true reason=missing_OPENAI_API_KEY")
        return None
    try:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)
        resp = await client.embeddings.create(model=EMBED_MODEL, input=text)
        vector = list(resp.data[0].embedding)
        return vector or None
    except Exception as exc:  # noqa: BLE001 - never raise into the turn pipeline
        logger.warning(
            "memory_embedding_failed=true error_type=%s model=%s", type(exc).__name__, EMBED_MODEL
        )
        return None

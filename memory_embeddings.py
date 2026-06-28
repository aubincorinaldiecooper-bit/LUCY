"""Text embeddings for semantic memory recall (pluggable provider -> pgvector).

An "embedding" turns text into a list of numbers that captures its meaning; texts
with similar meaning get similar number-lists. We embed each stored memory and, at
recall time, embed the user's current message and ask Postgres (pgvector) for the
memories whose vectors are closest — i.e. the most *relevant* memories, not just
the most recent.

Provider is pluggable via MEMORY_EMBEDDING_PROVIDER (default "cohere"):
  - cohere  -> embed-v4.0 (strong multilingual; output_dimension set to match the
               vector(1536) column). Needs COHERE_API_KEY + the `cohere` package.
  - openai  -> text-embedding-3-small (1536). Needs OPENAI_API_KEY.
You only need the key for the provider you actually use.

Asymmetric retrieval: queries are embedded with input_type="search_query" and
stored memories with "search_document" (Cohere/Voyage use this; OpenAI ignores it).

Fault-tolerant by design: any failure (missing key, missing SDK, network, quota,
dimension mismatch) returns None so the memory layer silently falls back to
recency recall — it never breaks a turn. The embedding dimension MUST match the
vector(N) column in migration 005 (1536); a different model dimension needs a new
migration.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("agent.memory")

EMBED_DIM = int(os.getenv("MEMORY_EMBEDDING_DIM", "1536") or "1536")

_DEFAULT_MODELS = {
    "cohere": "embed-v4.0",
    "openai": "text-embedding-3-small",
}


def embedding_provider() -> str:
    provider = (os.getenv("MEMORY_EMBEDDING_PROVIDER") or "cohere").strip().lower()
    return provider if provider in _DEFAULT_MODELS else "cohere"


def embedding_model() -> str:
    explicit = (os.getenv("MEMORY_EMBEDDING_MODEL") or "").strip()
    return explicit or _DEFAULT_MODELS[embedding_provider()]


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


async def _embed_openai(text: str, client: Any | None) -> list[float] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if client is None and not api_key:
        logger.warning("memory_embedding_skipped=true provider=openai reason=missing_OPENAI_API_KEY")
        return None
    if client is None:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
    resp = await client.embeddings.create(model=embedding_model(), input=text)
    return list(resp.data[0].embedding) or None


async def _embed_cohere(text: str, input_type: str, client: Any | None) -> list[float] | None:
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if client is None and not api_key:
        logger.warning("memory_embedding_skipped=true provider=cohere reason=missing_COHERE_API_KEY")
        return None
    if client is None:
        import cohere

        client = cohere.AsyncClientV2(api_key=api_key)
    model = embedding_model()
    kwargs: dict[str, Any] = {
        "texts": [text],
        "model": model,
        "input_type": input_type,
        "embedding_types": ["float"],
    }
    # output_dimension is an embed-v4 feature; v3 has a fixed 1024 dim.
    if "v4" in model:
        kwargs["output_dimension"] = EMBED_DIM
    resp = await client.embed(**kwargs)
    # SDK shape varies: embeddings.float_ (v2), embeddings.float, or a bare list.
    emb = getattr(resp, "embeddings", resp)
    vecs = getattr(emb, "float_", None) or getattr(emb, "float", None) or emb
    return list(vecs[0]) or None


async def embed_text(
    text: str, *, input_type: str = "search_query", client: Any | None = None
) -> list[float] | None:
    """Return the embedding for `text` via the configured provider, or None on any
    failure (graceful). `input_type` is 'search_query' for retrieval or
    'search_document' for stored memories (used by Cohere; ignored by OpenAI)."""
    text = (text or "").strip()
    if not text:
        return None
    provider = embedding_provider()
    try:
        if provider == "openai":
            return await _embed_openai(text, client)
        return await _embed_cohere(text, input_type, client)
    except ImportError:
        logger.warning(
            "memory_embedding_skipped=true provider=%s reason=sdk_not_installed "
            "install_hint=add_%s_to_dependencies",
            provider, provider,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - never raise into the turn pipeline
        logger.warning(
            "memory_embedding_failed=true provider=%s error_type=%s model=%s",
            provider, type(exc).__name__, embedding_model(),
        )
        return None

"""Long-term memory layer: Postgres durable store with optional pgvector recall.

Design (intent: voice-first, latency-protected):
- Every remembered item is written to the ``memory_units`` schema in Postgres
  (guest/account scoping, TTLs, soft deletes). Postgres is the source of truth.
- Session-start preload is recency-based (most recent memories first). When
  MEMORY_VECTOR_ENABLED=true and a pgvector-enabled Postgres is present,
  ``retrieve(query)`` does semantic (embedding) recall instead.
- Retrieval never blocks the turn pipeline beyond its timeout; on timeout or
  error it returns no memories and the turn proceeds without them.
- Writes are fire-and-forget background tasks; failures are logged, never raised.

The whole layer is disabled unless MEMORY_ENABLED=true.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

# NOTE: memory_embeddings (pluggable Cohere/OpenAI embeddings, commit bb011c5)
# is NOT wired in here anymore — a later rewrite of this module reverted to the
# local OpenAI-only embed_text below, and the leftover import of embed_text /
# semantic_retrieval_enabled / semantic_timeout_ms was fully shadowed by the
# local definitions (so MEMORY_SEMANTIC_RETRIEVAL_ENABLED and the Cohere
# provider currently have no effect on this layer; MEMORY_VECTOR_ENABLED is the
# live knob). Removing the dead import preserves runtime behavior exactly;
# re-integrating memory_embeddings is a separate, deliberate fix.

logger = logging.getLogger(__name__)

GUEST_MEMORY_TTL_HOURS = 24
DEFAULT_MEMORY_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MEMORY_EMBEDDING_DIMENSIONS = 1536


class _SemanticUnavailable(Exception):
    """Internal: semantic recall can't run (no embedding) -> fall back to recency."""


def memory_enabled() -> bool:
    return os.getenv("MEMORY_ENABLED", "false").strip().lower() in {"true", "1", "yes"}


def memory_retrieval_timeout_ms() -> int:
    try:
        return max(50, int(os.getenv("MEMORY_RETRIEVAL_TIMEOUT_MS", "300")))
    except Exception:
        return 300


def memory_vector_enabled() -> bool:
    return os.getenv("MEMORY_VECTOR_ENABLED", "false").strip().lower() in {"true", "1", "yes"}


def semantic_retrieval_enabled() -> bool:
    return memory_vector_enabled()


def semantic_timeout_ms() -> int:
    return memory_retrieval_timeout_ms()


def memory_embedding_model() -> str:
    return (os.getenv("MEMORY_EMBEDDING_MODEL") or DEFAULT_MEMORY_EMBEDDING_MODEL).strip() or DEFAULT_MEMORY_EMBEDDING_MODEL


def memory_embedding_dimensions() -> int:
    try:
        return max(1, int(os.getenv("MEMORY_EMBEDDING_DIMENSIONS", str(DEFAULT_MEMORY_EMBEDDING_DIMENSIONS))))
    except Exception:
        return DEFAULT_MEMORY_EMBEDDING_DIMENSIONS


def memory_preload_limit() -> int:
    try:
        return max(1, int(os.getenv("MEMORY_PRELOAD_LIMIT", "10")))
    except Exception:
        return 10


@dataclass(slots=True)
class MemoryIdentity:
    clerk_user_id: str | None = None
    guest_id: str | None = None

    @property
    def scope(self) -> str:
        return "account" if self.clerk_user_id else "guest"

    @property
    def key(self) -> str:
        return self.clerk_user_id or self.guest_id or "anonymous"

    @property
    def present(self) -> bool:
        return bool(self.clerk_user_id or self.guest_id)


def identity_from_metadata(metadata_values: list[Any], fallback_guest_id: str | None = None) -> MemoryIdentity:
    clerk_user_id: str | None = None
    guest_id: str | None = None
    for value in metadata_values:
        data: dict[str, Any] = {}
        if isinstance(value, dict):
            data = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                data = parsed if isinstance(parsed, dict) else {}
            except Exception:
                data = {}
        clerk_user_id = clerk_user_id or _clean_id(data.get("clerk_user_id") or data.get("user_id"))
        guest_id = guest_id or _clean_id(data.get("guest_id"))
    if not clerk_user_id and not guest_id:
        guest_id = _clean_id(fallback_guest_id)
    return MemoryIdentity(clerk_user_id=clerk_user_id, guest_id=guest_id)


def _clean_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def embed_text(text: str) -> list[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for semantic memory embeddings")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=memory_embedding_model(),
        input=text,
        dimensions=memory_embedding_dimensions(),
    )
    return [float(v) for v in response.data[0].embedding]


class MemoryLayer:
    """Per-session memory facade. All public methods are safe to call when degraded."""

    def __init__(
        self,
        identity: MemoryIdentity,
        session_id: str | None = None,
        companion_id: str | None = None,
        db_url: str | None = None,
        retrieval_timeout_ms: int | None = None,
        preload_limit: int | None = None,
        db_reader: Callable[[str, tuple], list[tuple]] | None = None,
        db_writer: Callable[[str, tuple], None] | None = None,
        embedder: Callable[[str], list[float]] | None = None,
        vector_enabled: bool | None = None,
        semantic_enabled: bool | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        semantic_timeout_ms_override: int | None = None,
    ) -> None:
        self.identity = identity
        self.session_id = session_id
        self.companion_id = companion_id
        self._db_url = db_url if db_url is not None else os.getenv("DATABASE_URL")
        self._retrieval_timeout_ms = retrieval_timeout_ms or memory_retrieval_timeout_ms()
        self._preload_limit = preload_limit or memory_preload_limit()
        self._db_reader = db_reader or self._psycopg_reader
        self._db_writer = db_writer or self._psycopg_writer
        try:
            if semantic_enabled is None:
                if vector_enabled is None:
                    semantic_on = semantic_retrieval_enabled()
                else:
                    semantic_on = bool(vector_enabled)
            else:
                semantic_on = bool(semantic_enabled)
            self._semantic_enabled = semantic_on
            self._vector_enabled = semantic_on
            self._embed_fn = embed_fn or embedder or embed_text
            self._embedder = self._embed_fn
            self._semantic_timeout_ms = semantic_timeout_ms_override or semantic_timeout_ms()
            self._embedding_model = memory_embedding_model()
            self._embedding_dimensions = memory_embedding_dimensions()
        except Exception as exc:
            logger.warning(
                "memory_semantic_config_invalid=true degraded_to_non_semantic=true error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            self._semantic_enabled = False
            self._vector_enabled = False
            self._embed_fn = embed_fn or embedder or embed_text
            self._embedder = self._embed_fn
            self._semantic_timeout_ms = self._retrieval_timeout_ms
            self._embedding_model = DEFAULT_MEMORY_EMBEDDING_MODEL
            self._embedding_dimensions = DEFAULT_MEMORY_EMBEDDING_DIMENSIONS
        self._vector_status = "not_checked" if self._vector_enabled else "disabled"
        self._background_tasks: set[asyncio.Task] = set()

    # ---------- backends ----------

    def _psycopg_reader(self, sql: str, params: tuple) -> list[tuple]:
        import psycopg

        with psycopg.connect(self._db_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def _psycopg_writer(self, sql: str, params: tuple) -> None:
        import psycopg

        with psycopg.connect(self._db_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()


    @staticmethod
    def _vector_literal(values: list[float]) -> str:
        return "[" + ",".join(f"{float(v):.8g}" for v in values) + "]"

    def _pgvector_available(self) -> bool:
        if not self._vector_enabled or not self._db_url:
            return False
        if self._vector_status == "ready":
            return True
        if self._vector_status in {"unavailable", "error"}:
            return False
        try:
            rows = self._db_reader(
                """
                SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')
                   AND EXISTS (
                     SELECT 1 FROM information_schema.columns
                     WHERE table_name = 'memory_units' AND column_name = 'embedding'
                   )
                """,
                (),
            )
            available = bool(rows and rows[0] and rows[0][0])
            self._vector_status = "ready" if available else "unavailable"
            logger.info("memory_pgvector_status=%s", self._vector_status)
            return available
        except Exception as exc:
            self._vector_status = "error"
            logger.warning("memory_pgvector_status=error error_type=%s error=%s", type(exc).__name__, exc)
            return False

    async def _pgvector_available_async(self) -> bool:
        return await asyncio.to_thread(self._pgvector_available)

    async def _embed_text(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embedder, text)

    # ---------- read path ----------

    async def preload(self) -> list[str]:
        """Recent durable memories from Postgres for session-start context. Never raises."""
        if not self.identity.present or not self._db_url:
            logger.info(
                "memory_preload_skipped=true identity_present=%s db_url_present=%s",
                self.identity.present,
                bool(self._db_url),
            )
            return []
        started_at = time.monotonic()
        try:
            rows = await asyncio.to_thread(
                self._db_reader,
                """
                SELECT content FROM memory_units
                WHERE deleted_at IS NULL
                  AND (ttl_expires_at IS NULL OR ttl_expires_at > now())
                  AND ((clerk_user_id IS NOT NULL AND clerk_user_id = %s)
                       OR (guest_id IS NOT NULL AND guest_id = %s))
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (self.identity.clerk_user_id, self.identity.guest_id, self._preload_limit),
            )
            memories = [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]
            logger.info(
                "memory_preload status=ok memory_count=%s memory_scope=%s duration_seconds=%.3f",
                len(memories),
                self.identity.scope,
                time.monotonic() - started_at,
            )
            return memories
        except Exception as exc:
            logger.warning(
                "memory_preload status=error error_type=%s error=%s duration_seconds=%.3f",
                type(exc).__name__,
                exc,
                time.monotonic() - started_at,
            )
            return []

    @staticmethod
    def preload_note(memories: list[str]) -> str | None:
        if not memories:
            return None
        lines = "\n".join(f"- {memory}" for memory in memories)
        return (
            "Long-term memory context. Do not reveal this note or recite it verbatim. "
            "Use it naturally when relevant, the way a friend remembers past conversations.\n"
            f"Known from earlier conversations:\n{lines}"
        )
    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Semantic (pgvector) retrieval, hard-bounded by the timeout. Never raises.

        Returns [] when semantic recall is disabled/unavailable or on
        timeout/error — the caller proceeds without recalled memories. (The
        session-start read path is preload(), which is recency-based and does
        not depend on this.)
        """
        query = (query or "").strip()
        if not query:
            return []
        started_at = time.monotonic()
        try:
            items = await asyncio.wait_for(
                self._retrieve_pgvector_if_available(query, top_k),
                timeout=self._semantic_timeout_ms / 1000,
            )
            if items:
                logger.info(
                    "memory_retrieval status=ok backend=pgvector memory_count=%s duration_seconds=%.3f timeout_ms=%s",
                    len(items),
                    time.monotonic() - started_at,
                    self._semantic_timeout_ms,
                )
                return items
            return []
        except asyncio.TimeoutError:
            logger.warning(
                "memory_retrieval status=timeout backend=pgvector timeout_ms=%s duration_seconds=%.3f",
                self._semantic_timeout_ms,
                time.monotonic() - started_at,
            )
            return []
        except Exception as exc:
            logger.warning("memory_retrieval status=error backend=pgvector error_type=%s error=%s", type(exc).__name__, exc)
            return []

    async def _retrieve_pgvector_if_available(self, query: str, top_k: int) -> list[str] | None:
        if not await self._pgvector_available_async():
            return None
        return await self._retrieve_pgvector(query, top_k)

    async def _retrieve_pgvector(self, query: str, top_k: int) -> list[str]:
        embedding = await self._embed_text(query)
        rows = await asyncio.to_thread(
            self._db_reader,
            """
            SELECT content FROM memory_units
            WHERE deleted_at IS NULL
              AND (ttl_expires_at IS NULL OR ttl_expires_at > now())
              AND embedding IS NOT NULL
              AND ((clerk_user_id IS NOT NULL AND clerk_user_id = %s)
                   OR (guest_id IS NOT NULL AND guest_id = %s))
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.identity.clerk_user_id, self.identity.guest_id, self._vector_literal(embedding), top_k),
        )
        return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]

    # ---------- write path ----------

    def schedule_remember(self, role: str, content: str, turn_id: int | None = None, modality: str = "text", media_url: str | None = None) -> None:
        """Fire-and-forget write to Postgres (durable row + optional pgvector embedding). Safe to call from event handlers."""
        content = (content or "").strip()
        if not content or not self.identity.present:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("memory_remember_skipped=true reason=no_running_event_loop")
            return
        task = loop.create_task(self._remember(role=role, content=content, turn_id=turn_id, modality=modality, media_url=media_url))
        self._background_tasks.add(task)

        def _on_done(t: "asyncio.Task") -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                # Last line of defense: retrieve+log so a background memory write can
                # never surface as an unhandled task exception or touch the voice loop.
                logger.warning(
                    "memory_background_task_failed=true role=%s turn_id=%s modality=%s fallback=recency error_type=%s error=%s",
                    role, turn_id, modality, type(exc).__name__, exc,
                )

        task.add_done_callback(_on_done)

    async def _remember(self, role: str, content: str, turn_id: int | None, modality: str, media_url: str | None) -> None:
        # A structured note (currently only confirmed emotional-calibration
        # patterns) must round-trip through preload() -> partition_emotional_patterns()
        # with EMOTIONAL_PATTERN_PREFIX still at the very start, so it is stored
        # verbatim rather than wrapped in "User said/Lucy replied".
        if content.startswith(EMOTIONAL_PATTERN_PREFIX):
            compact = content
        elif role == "observation":
            # Vision-context scene descriptions (see inworld_realtime_bridge):
            # phrased as Arche's own noticing so the preload note reads naturally.
            compact = f"Arche noticed: {content}"
        else:
            compact = f"{'User said' if role == 'user' else 'Lucy replied'}: {content}"
        try:
            await self._embed_and_store(compact, role, turn_id, modality, media_url)
        except Exception as exc:
            logger.warning(
                "memory_write status=error target=postgres role=%s turn_id=%s modality=%s fallback=recency error_type=%s error=%s",
                role, turn_id, modality, type(exc).__name__, exc,
            )

    async def _embed_and_store(self, compact: str, role: str, turn_id: int | None, modality: str, media_url: str | None) -> None:
        """Store durable memory with optional embedding, degrading to text-only writes. Never raises for semantic failures."""
        try:
            await self._write_postgres_memory(compact, role, turn_id, modality, media_url)
        except Exception:
            raise

    async def _write_postgres_memory(self, compact: str, role: str, turn_id: int | None, modality: str, media_url: str | None) -> None:
        metadata = json.dumps({"role": role, "turn_id": turn_id})
        base_params = (
            self.identity.scope,
            self.identity.clerk_user_id,
            self.identity.guest_id,
            self.companion_id,
            compact,
            _content_hash(compact),
            self.identity.scope == "account",
            self.identity.scope,
            GUEST_MEMORY_TTL_HOURS,
            modality,
            media_url,
            metadata,
        )
        if self._pgvector_available():
            try:
                embedding = await self._embed_text(compact)
                await asyncio.to_thread(
                    self._db_writer,
                    """
                    INSERT INTO memory_units
                      (memory_scope, clerk_user_id, guest_id, companion_id, content, content_hash,
                       is_persistent, ttl_expires_at, modality, media_url, metadata,
                       embedding, embedding_model, embedding_created_at)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s,
                       CASE WHEN %s = 'guest' THEN now() + make_interval(hours => %s) ELSE NULL END,
                       %s, %s, %s::jsonb, %s::vector, %s, now())
                    """,
                    (*base_params, self._vector_literal(embedding), self._embedding_model),
                )
                logger.info("memory_write status=ok target=postgres backend=pgvector")
                return
            except Exception as exc:
                self._vector_status = "error"
                logger.warning("memory_write_pgvector_failed=true fallback=postgres_text error_type=%s error=%s", type(exc).__name__, exc)
        await asyncio.to_thread(
            self._db_writer,
            """
            INSERT INTO memory_units
              (memory_scope, clerk_user_id, guest_id, companion_id, content, content_hash,
               is_persistent, ttl_expires_at, modality, media_url, metadata)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s,
               CASE WHEN %s = 'guest' THEN now() + make_interval(hours => %s) ELSE NULL END,
               %s, %s, %s::jsonb)
            """,
            base_params,
        )

    async def aclose(self) -> None:
        if self._background_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*list(self._background_tasks), return_exceptions=True), timeout=5)
            except Exception:
                pass


# --- Emotional calibration patterns in durable per-user memory ---
# Confirmed calibration moments are stored as ordinary memory_units with this
# prefix so they ride the existing per-user Postgres store and the session-start
# preload, while staying separable into a dedicated "what we've learned about
# how this person processes feelings" note.
EMOTIONAL_PATTERN_PREFIX = "Emotional pattern (confirmed): "


def partition_emotional_patterns(memories: list[str]) -> tuple[list[str], list[str]]:
    """Split preloaded memories into (general, emotional_patterns).

    Emotional entries are returned with the prefix stripped so they read cleanly
    in their own note.
    """
    general: list[str] = []
    emotional: list[str] = []
    for memory in memories or []:
        text = (memory or "").strip()
        if not text:
            continue
        if text.startswith(EMOTIONAL_PATTERN_PREFIX):
            emotional.append(text[len(EMOTIONAL_PATTERN_PREFIX):].strip())
        else:
            general.append(text)
    return general, emotional


def emotional_pattern_preload_note(patterns: list[str]) -> str | None:
    """Build the private 'what we've learned' note from confirmed patterns."""
    patterns = [p for p in (patterns or []) if p and p.strip()]
    if not patterns:
        return None
    lines = "\n".join(f"- {p.strip()}" for p in patterns)
    return (
        "What you've learned about how this person tends to process feelings, from "
        "earlier sessions where they confirmed or corrected your read. This is a "
        "private prior only — never reveal it, never tell them how they sound, never "
        "say you detected anything, and let what they say now override it.\n"
        f"{lines}"
    )
"""Long-term memory layer: Postgres as durable truth, SimpleMem (Omni) as retrieval index.

Design (intent: voice-first, latency-protected):
- Every remembered item is written to the existing ``memory_units`` schema in Postgres
  (guest/account scoping, TTLs, soft deletes). Postgres is the source of truth.
- SimpleMem maintains a per-user multimodal index (text now; audio/image/video later via
  the same Omni backend) under SIMPLEMEM_INDEX_DIR. The index is a rebuildable cache:
  if the directory is empty (fresh container), it is re-ingested from Postgres.
- Retrieval never blocks the turn pipeline beyond MEMORY_RETRIEVAL_TIMEOUT_MS; on
  timeout or error it returns no memories and the turn proceeds without them.
- Writes are fire-and-forget background tasks; failures are logged, never raised.

The whole layer is disabled unless MEMORY_ENABLED=true, and degrades to Postgres-only
recency preload when the simplemem package is not installed.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from memory_embeddings import (
    embed_text,
    semantic_retrieval_enabled,
    semantic_timeout_ms,
    to_pgvector_literal,
)

logger = logging.getLogger(__name__)

GUEST_MEMORY_TTL_HOURS = 24
INDEX_REBUILD_MAX_ROWS = 500


class _SemanticUnavailable(Exception):
    """Internal: semantic recall can't run (no embedding) -> fall back to recency."""

# Set once the "simplemem not installed" warning has been logged, so it isn't
# repeated for every per-session MemoryLayer instance.
_SIMPLEMEM_UNAVAILABLE_LOGGED = False


def memory_enabled() -> bool:
    return os.getenv("MEMORY_ENABLED", "false").strip().lower() in {"true", "1", "yes"}


def memory_retrieval_timeout_ms() -> int:
    try:
        return max(50, int(os.getenv("MEMORY_RETRIEVAL_TIMEOUT_MS", "300")))
    except Exception:
        return 300


def memory_preload_limit() -> int:
    try:
        return max(1, int(os.getenv("MEMORY_PRELOAD_LIMIT", "10")))
    except Exception:
        return 10


def simplemem_index_dir() -> str:
    return os.getenv("SIMPLEMEM_INDEX_DIR", "/data/simplemem")


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


def _normalize_retrieved_items(raw: Any) -> list[str]:
    items: list[str] = []
    if raw is None:
        return items
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, dict):
        raw = raw.get("results") or raw.get("items") or raw.get("memories") or []
    if not isinstance(raw, (list, tuple)):
        return items
    for entry in raw:
        if isinstance(entry, str):
            text = entry
        elif isinstance(entry, dict):
            text = entry.get("summary") or entry.get("content") or entry.get("text") or ""
        else:
            text = str(getattr(entry, "summary", "") or getattr(entry, "content", "") or "")
        text = str(text).strip()
        if text:
            items.append(text)
    return items


class MemoryLayer:
    """Per-session memory facade. All public methods are safe to call when degraded."""

    def __init__(
        self,
        identity: MemoryIdentity,
        session_id: str | None = None,
        companion_id: str | None = None,
        db_url: str | None = None,
        index_dir: str | None = None,
        retrieval_timeout_ms: int | None = None,
        preload_limit: int | None = None,
        simplemem_factory: Callable[[str], Any] | None = None,
        db_reader: Callable[[str, tuple], list[tuple]] | None = None,
        db_writer: Callable[[str, tuple], None] | None = None,
        semantic_enabled: bool | None = None,
        embed_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.identity = identity
        self.session_id = session_id
        self.companion_id = companion_id
        self._db_url = db_url if db_url is not None else os.getenv("DATABASE_URL")
        self._index_dir = index_dir or simplemem_index_dir()
        self._retrieval_timeout_ms = retrieval_timeout_ms or memory_retrieval_timeout_ms()
        self._preload_limit = preload_limit or memory_preload_limit()
        self._simplemem_factory = simplemem_factory or _default_simplemem_factory
        self._db_reader = db_reader or self._psycopg_reader
        self._db_writer = db_writer or self._psycopg_writer
        self._simplemem: Any = None
        self._simplemem_status = "not_initialized"
        self._background_tasks: set[asyncio.Task] = set()
        # pgvector semantic recall: embed memories on write and find the most
        # relevant (not just most recent) on retrieve. Off unless enabled + a key.
        self._semantic_enabled = (
            semantic_retrieval_enabled() if semantic_enabled is None else semantic_enabled
        )
        self._embed_fn = embed_fn or embed_text
        self._semantic_timeout_ms = semantic_timeout_ms()

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

    def _get_simplemem(self) -> Any:
        if self._simplemem is not None or self._simplemem_status in {"unavailable", "error"}:
            return self._simplemem
        try:
            user_dir = os.path.join(self._index_dir, self.identity.key)
            self._simplemem = self._simplemem_factory(user_dir)
            self._simplemem_status = "ready"
            logger.info("memory_simplemem_initialized=true index_dir=%s memory_scope=%s", user_dir, self.identity.scope)
        except ImportError:
            self._simplemem_status = "unavailable"
            # Log once per process: a fresh MemoryLayer is created per session, so
            # without this the same "not installed" warning repeats every session.
            global _SIMPLEMEM_UNAVAILABLE_LOGGED
            if not _SIMPLEMEM_UNAVAILABLE_LOGGED:
                _SIMPLEMEM_UNAVAILABLE_LOGGED = True
                logger.warning(
                    "memory_simplemem_unavailable=true reason=package_not_installed "
                    "install_hint=pip_install_simplemem note=logged_once_per_process "
                    "(retrieval falls back to Postgres recency preload)"
                )
        except Exception as exc:
            self._simplemem_status = "error"
            logger.warning("memory_simplemem_init_failed=true error_type=%s error=%s", type(exc).__name__, exc)
        return self._simplemem

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
        """Most-relevant memories for `query`. Prefers pgvector semantic recall when
        enabled, else falls back to the SimpleMem index. Hard-bounded; never raises."""
        query = (query or "").strip()
        if not query:
            return []
        if self._semantic_enabled and self._db_url and self.identity.present:
            items = await self._semantic_retrieve(query, top_k)
            if items is not None:  # None = semantic unavailable -> fall through
                return items
        backend = self._get_simplemem()
        if backend is None:
            return []
        started_at = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(backend.query, query, top_k),
                timeout=self._retrieval_timeout_ms / 1000,
            )
            items = _normalize_retrieved_items(raw)
            logger.info(
                "memory_retrieval status=ok memory_count=%s duration_seconds=%.3f timeout_ms=%s",
                len(items),
                time.monotonic() - started_at,
                self._retrieval_timeout_ms,
            )
            return items
        except asyncio.TimeoutError:
            logger.warning(
                "memory_retrieval status=timeout timeout_ms=%s duration_seconds=%.3f",
                self._retrieval_timeout_ms,
                time.monotonic() - started_at,
            )
            return []
        except Exception as exc:
            logger.warning(
                "memory_retrieval status=error error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return []

    async def _semantic_retrieve(self, query: str, top_k: int) -> list[str] | None:
        """pgvector nearest-neighbour recall. Returns items, [] for no matches, or
        None when semantic recall is unavailable (caller falls back). The whole
        embed+query is bounded by the semantic timeout so a turn never stalls."""
        started_at = time.monotonic()

        async def _run() -> list[str]:
            vector = await self._embed_fn(query)
            if not vector:
                raise _SemanticUnavailable("no_embedding")
            rows = await asyncio.to_thread(
                self._db_reader,
                """
                SELECT content FROM memory_units
                WHERE deleted_at IS NULL
                  AND embedding IS NOT NULL
                  AND (ttl_expires_at IS NULL OR ttl_expires_at > now())
                  AND ((clerk_user_id IS NOT NULL AND clerk_user_id = %s)
                       OR (guest_id IS NOT NULL AND guest_id = %s))
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (self.identity.clerk_user_id, self.identity.guest_id, to_pgvector_literal(vector), top_k),
            )
            return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]

        try:
            items = await asyncio.wait_for(_run(), timeout=self._semantic_timeout_ms / 1000)
        except _SemanticUnavailable:
            return None
        except asyncio.TimeoutError:
            logger.warning(
                "memory_retrieval status=timeout backend=pgvector timeout_ms=%s duration_seconds=%.3f",
                self._semantic_timeout_ms,
                time.monotonic() - started_at,
            )
            return None
        except Exception as exc:
            # e.g. embedding column / pgvector not present -> fall back, don't break.
            logger.warning(
                "memory_retrieval status=error backend=pgvector error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return None
        logger.info(
            "memory_retrieval status=ok backend=pgvector memory_count=%s duration_seconds=%.3f",
            len(items),
            time.monotonic() - started_at,
        )
        return items

    # ---------- write path ----------

    def schedule_remember(self, role: str, content: str, turn_id: int | None = None, modality: str = "text", media_url: str | None = None) -> None:
        """Fire-and-forget write to Postgres + SimpleMem. Safe to call from event handlers."""
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
        task.add_done_callback(self._background_tasks.discard)

    async def _remember(self, role: str, content: str, turn_id: int | None, modality: str, media_url: str | None) -> None:
        compact = f"{'User said' if role == 'user' else 'Lucy replied'}: {content}"
        if self._db_url:
            try:
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
                    (
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
                        json.dumps({"role": role, "turn_id": turn_id}),
                    ),
                )
            except Exception as exc:
                logger.warning("memory_write status=error target=postgres error_type=%s error=%s", type(exc).__name__, exc)
            await self._embed_and_store(compact)
        backend = self._get_simplemem()
        if backend is not None:
            try:
                tags = [f"user:{self.identity.key}", f"role:{role}"]
                if modality == "audio" and media_url:
                    await asyncio.to_thread(backend.add_audio, media_url, tags)
                else:
                    await asyncio.to_thread(backend.add_text, compact, tags)
            except Exception as exc:
                logger.warning("memory_write status=error target=simplemem error_type=%s error=%s", type(exc).__name__, exc)

    async def _embed_and_store(self, compact: str) -> None:
        """Best-effort: embed a just-written memory and store its vector by
        content_hash. A separate UPDATE so the base insert is never affected; any
        failure (semantic off, no key, no pgvector column) is swallowed."""
        if not self._semantic_enabled or not self._db_url:
            return
        try:
            vector = await self._embed_fn(compact)
            if not vector:
                return
            await asyncio.to_thread(
                self._db_writer,
                """
                UPDATE memory_units SET embedding = %s::vector
                WHERE content_hash = %s AND embedding IS NULL
                  AND ((clerk_user_id IS NOT NULL AND clerk_user_id = %s)
                       OR (guest_id IS NOT NULL AND guest_id = %s))
                """,
                (to_pgvector_literal(vector), _content_hash(compact),
                 self.identity.clerk_user_id, self.identity.guest_id),
            )
        except Exception as exc:
            logger.warning(
                "memory_write status=error target=embedding error_type=%s error=%s",
                type(exc).__name__, exc,
            )

    async def rebuild_index_if_empty(self) -> None:
        """Re-ingest recent Postgres memories when the container has a fresh/empty index."""
        backend = self._get_simplemem()
        if backend is None or not self._db_url or not self.identity.present:
            return
        user_dir = os.path.join(self._index_dir, self.identity.key)
        try:
            already_populated = any(os.scandir(user_dir)) if os.path.isdir(user_dir) else False
        except Exception:
            already_populated = False
        if already_populated:
            return
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
                (self.identity.clerk_user_id, self.identity.guest_id, INDEX_REBUILD_MAX_ROWS),
            )
            for row in rows:
                content = str(row[0]).strip()
                if content:
                    await asyncio.to_thread(backend.add_text, content, [f"user:{self.identity.key}", "source:rebuild"])
            logger.info("memory_index_rebuilt=true row_count=%s", len(rows))
        except Exception as exc:
            logger.warning("memory_index_rebuild status=error error_type=%s error=%s", type(exc).__name__, exc)

    async def aclose(self) -> None:
        if self._background_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*list(self._background_tasks), return_exceptions=True), timeout=5)
            except Exception:
                pass
        backend = self._simplemem
        if backend is not None:
            for method_name in ("finalize", "close"):
                method = getattr(backend, method_name, None)
                if callable(method):
                    try:
                        await asyncio.to_thread(method)
                    except Exception as exc:
                        logger.warning("memory_close status=error method=%s error_type=%s error=%s", method_name, type(exc).__name__, exc)


def _default_simplemem_factory(index_dir: str) -> Any:
    os.makedirs(index_dir, exist_ok=True)
    from simplemem import SimpleMem

    # Storage-path kwarg naming differs across simplemem releases; fall back gracefully.
    for kwargs in ({"data_dir": index_dir}, {"db_path": index_dir}, {}):
        try:
            return SimpleMem(**kwargs)
        except TypeError:
            continue
    return SimpleMem()


# --- Emotional calibration patterns in durable per-user memory ---
# Confirmed calibration moments are stored as ordinary memory_units with this
# prefix so they ride the existing per-user Postgres + SimpleMem store and the
# session-start preload, while staying separable into a dedicated "what we've
# learned about how this person processes feelings" note.
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

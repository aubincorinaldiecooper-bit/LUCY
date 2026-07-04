"""OpenRouter spend metering: attribute per-call token + cost usage to a user.

Every OpenRouter call the product makes (vision_context.describe_frame and
mood_context.describe_mood) can hand its raw response here; the meter pulls the
``usage`` block (token counts, and the real credit cost when the request set
``usage.include=true``) and persists one row per call attributed to a
``user_ref`` (the durable billing key) and ``session_id``. An operator can then
aggregate spend per user via ``fetch_spend_summary`` and forward it to paid
accounts as a meter.

Design mirrors memory_layer:
- Off by default (``OPENROUTER_METERING_ENABLED`` unset/false) AND a no-op
  without ``DATABASE_URL`` — nothing is written unless deliberately enabled.
- ``record_from_response`` is fire-and-forget: it schedules a background
  Postgres write and returns immediately. It never raises and never blocks the
  voice turn — a metering failure can only drop a usage row, never affect the
  call it measured.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


def metering_enabled() -> bool:
    return (os.getenv("OPENROUTER_METERING_ENABLED") or "false").strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }


def openrouter_usage_body() -> dict:
    """Request-body fragment that makes OpenRouter return real cost accounting.

    Merged into the chat-completions body by the callers so the response
    ``usage`` object carries a ``cost`` field (credits == USD) alongside the
    token counts. Cheap (no extra request, negligible tokens); safe to send
    even when metering is disabled — the response is simply ignored.
    """
    return {"usage": {"include": True}}


def parse_openrouter_usage(data: Any) -> dict | None:
    """Extract token counts + cost from an OpenRouter chat-completions response.

    Returns ``{prompt_tokens, completion_tokens, total_tokens, cost_usd}`` or
    ``None`` when the response carries no ``usage`` block. ``cost_usd`` is 0.0
    unless the request asked for cost accounting (openrouter_usage_body()).
    """
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return {
        "prompt_tokens": _int(usage.get("prompt_tokens")),
        "completion_tokens": _int(usage.get("completion_tokens")),
        "total_tokens": _int(usage.get("total_tokens")),
        # OpenRouter reports the charged amount as ``cost`` in credits (== USD).
        "cost_usd": _float(usage.get("cost")),
    }


@dataclass(frozen=True)
class UsageEvent:
    feature: str  # "vision" | "mood"
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    user_ref: str | None
    session_id: str | None


_INSERT_SQL = """
INSERT INTO openrouter_usage_events
  (user_ref, session_id, feature, model,
   prompt_tokens, completion_tokens, total_tokens, cost_usd)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


class SpendMeter:
    """Per-session OpenRouter spend meter. Fire-and-forget, never raises.

    ``user_ref`` is set lazily: the session core resolves memory identity in the
    background, so the billing key isn't known at construction. Rows written
    before it resolves carry ``user_ref=None`` (still attributable by
    ``session_id``); once resolved they carry the durable per-user key.
    """

    def __init__(
        self,
        *,
        session_id: str | None,
        user_ref: str | None = None,
        db_url: str | None = None,
        enabled: bool | None = None,
        db_writer: Callable[[str, tuple], None] | None = None,
    ) -> None:
        self.session_id = session_id
        self.user_ref = user_ref
        self._db_url = db_url if db_url is not None else os.getenv("DATABASE_URL")
        self._enabled = metering_enabled() if enabled is None else bool(enabled)
        self._db_writer = db_writer or self._psycopg_writer
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        # A meter with nowhere to write is inert: don't parse/allocate per call.
        return bool(self._enabled and self._db_url)

    def _psycopg_writer(self, sql: str, params: tuple) -> None:
        import psycopg

        with psycopg.connect(self._db_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

    def record_from_response(self, feature: str, model: str, data: Any) -> None:
        """Meter one OpenRouter call from its raw response dict. Never raises."""
        if not self.enabled:
            return
        try:
            usage = parse_openrouter_usage(data)
        except Exception:  # noqa: BLE001 - metering must never break the caller
            usage = None
        if not usage:
            return
        event = UsageEvent(
            feature=feature,
            model=model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cost_usd=usage["cost_usd"],
            user_ref=self.user_ref,
            session_id=self.session_id,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (called from sync context/tests without a loop):
            # write inline rather than silently dropping the row — but still
            # never raise (metering must not surface into the caller).
            try:
                self._write(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "openrouter_spend_write_failed=true feature=%s model=%s error_type=%s error=%s",
                    event.feature,
                    event.model,
                    type(exc).__name__,
                    exc,
                )
            return
        task = loop.create_task(self._persist(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _persist(self, event: UsageEvent) -> None:
        try:
            await asyncio.to_thread(self._write, event)
        except Exception as exc:  # noqa: BLE001 - background write, never surface
            logger.warning(
                "openrouter_spend_write_failed=true feature=%s model=%s error_type=%s error=%s",
                event.feature,
                event.model,
                type(exc).__name__,
                exc,
            )

    def _write(self, event: UsageEvent) -> None:
        self._db_writer(
            _INSERT_SQL,
            (
                event.user_ref,
                event.session_id,
                event.feature,
                event.model,
                event.prompt_tokens,
                event.completion_tokens,
                event.total_tokens,
                event.cost_usd,
            ),
        )

    async def aclose(self) -> None:
        """Drain in-flight background writes so a closing session isn't dropped."""
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
            self._background_tasks.clear()


def _psycopg_read(db_url: str, sql: str, params: tuple) -> list[tuple]:
    import psycopg

    with psycopg.connect(db_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def fetch_spend_summary(
    *,
    user_ref: str | None = None,
    session_id: str | None = None,
    since: str | None = None,
    db_url: str | None = None,
    db_reader: Callable[[str, tuple], list[tuple]] | None = None,
) -> dict:
    """Aggregate OpenRouter spend for billing/forwarding as a meter.

    Filter by ``user_ref`` and/or ``session_id`` (both optional; omit for a
    global total) and optionally ``since`` (ISO timestamp). Returns totals plus
    per-feature and per-model breakdowns. Read-only; safe to call from an
    operator/admin context.
    """
    db_url = db_url if db_url is not None else os.getenv("DATABASE_URL")
    reader = db_reader or (lambda sql, params: _psycopg_read(db_url, sql, params))

    where: list[str] = []
    params: list[Any] = []
    if user_ref is not None:
        where.append("user_ref = %s")
        params.append(user_ref)
    if session_id is not None:
        where.append("session_id = %s")
        params.append(session_id)
    if since is not None:
        where.append("created_at >= %s")
        params.append(since)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    rows = reader(
        f"""
        SELECT feature, model,
               COUNT(*) AS calls,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COALESCE(SUM(total_tokens), 0) AS total_tokens,
               COALESCE(SUM(cost_usd), 0) AS cost_usd
        FROM openrouter_usage_events{clause}
        GROUP BY feature, model
        """,
        tuple(params),
    )

    summary = {
        "user_ref": user_ref,
        "session_id": session_id,
        "since": since,
        "total_calls": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "by_feature": {},
        "by_model": {},
    }
    for feature, model, calls, prompt_tokens, completion_tokens, total_tokens, cost_usd in rows:
        calls = int(calls or 0)
        total_tokens = int(total_tokens or 0)
        cost_usd = float(cost_usd or 0.0)
        summary["total_calls"] += calls
        summary["total_tokens"] += total_tokens
        summary["total_cost_usd"] += cost_usd
        for bucket_key, bucket_name in (("by_feature", feature), ("by_model", model)):
            bucket = summary[bucket_key].setdefault(
                bucket_name or "unknown",
                {"calls": 0, "total_tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += calls
            bucket["total_tokens"] += total_tokens
            bucket["cost_usd"] += cost_usd
    # Round money to cents-of-a-cent so float dust doesn't leak into a bill.
    summary["total_cost_usd"] = round(summary["total_cost_usd"], 6)
    for bucket_key in ("by_feature", "by_model"):
        for bucket in summary[bucket_key].values():
            bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    return summary

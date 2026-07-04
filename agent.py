from __future__ import annotations

import os
import asyncio
import logging
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from inworld_realtime_bridge import run_inworld_realtime_bridge
from memory_layer import (
    EMOTIONAL_PATTERN_PREFIX,  # noqa: F401 - re-exported for calibration tests + callers
    MemoryLayer,
    emotional_pattern_preload_note,
    identity_from_metadata,
    memory_enabled,
    partition_emotional_patterns,
)

load_dotenv()
logger = logging.getLogger(__name__)


from system_prompt import DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPT  # noqa: E402,F401


_active_memory_layer: MemoryLayer | None = None


def _deployment_git_commit_sha() -> str:
    for env_name in (
        "RAILWAY_GIT_COMMIT_SHA",
        "RAILWAY_GIT_COMMIT",
        "GIT_COMMIT_SHA",
        "GIT_SHA",
        "SOURCE_COMMIT",
        "VERCEL_GIT_COMMIT_SHA",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")

RUN_DB_MIGRATIONS_ON_STARTUP = env_bool("RUN_DB_MIGRATIONS_ON_STARTUP", False)


def _run_db_migrations_on_startup() -> None:
    logger.info("database_migrations_startup_enabled=%s", RUN_DB_MIGRATIONS_ON_STARTUP)
    if not RUN_DB_MIGRATIONS_ON_STARTUP:
        logger.info("database_migration_status=skipped")
        return

    if not os.getenv("DATABASE_URL"):
        logger.error("database_migration_status=failed reason=missing_DATABASE_URL")
        raise RuntimeError("DATABASE_URL is required when RUN_DB_MIGRATIONS_ON_STARTUP=true")

    script_path = os.path.join(os.path.dirname(__file__), "scripts", "apply_migrations.py")
    if not os.path.exists(script_path):
        logger.error("database_migration_status=failed reason=migration_runner_missing")
        raise RuntimeError("Migration runner not found at scripts/apply_migrations.py")

    logger.info("database_migration_status=running")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (result.stdout or "").splitlines():
        logger.info("migration_runner: %s", line)
    for line in (result.stderr or "").splitlines():
        logger.error("migration_runner: %s", line)

    if result.returncode != 0:
        logger.error("database_migration_status=failed return_code=%s", result.returncode)
        raise RuntimeError("Database migration failed during startup")

    logger.info("database_migration_status=success")


def _realtime_metadata_candidates(ctx: JobContext) -> list[Any]:
    """Gather job/room/participant metadata for identity resolution.

    Deliberately does NOT reuse ``_metadata_candidates_from_context`` /
    ``_metadata_debug_entries_from_context``: both route every lookup through
    ``_safe_attr``, which is a logging helper that ``str()``-converts
    whatever it fetches. Chaining it (``_safe_attr(_safe_attr(ctx, "job"),
    "metadata")``) calls ``getattr`` on the *stringified* job object, which
    always fails and silently falls back to the default — so those two
    helpers never actually return real metadata. Uses plain ``getattr`` here
    so job-level metadata (how the signed-in user id actually arrives, per
    SESSION_IDENTITY_SHARED_SECRET) is genuinely read.
    """
    candidates: list[Any] = []
    job = getattr(ctx, "job", None)
    room = getattr(ctx, "room", None)
    for obj in (job, room):
        metadata = getattr(obj, "metadata", None)
        if metadata:
            candidates.append(metadata)
    for attr_name in ("participant", "participant_info"):
        participant = getattr(job, attr_name, None)
        metadata = getattr(participant, "metadata", None)
        if metadata:
            candidates.append(metadata)
    remote_participants = getattr(room, "remote_participants", None)
    if isinstance(remote_participants, dict):
        for participant in remote_participants.values():
            metadata = getattr(participant, "metadata", None)
            if metadata:
                candidates.append(metadata)
    return candidates


def _room_name_from_context(ctx: JobContext) -> str | None:
    room = getattr(ctx, "room", None)
    name = getattr(room, "name", None)
    return name if isinstance(name, str) and name.strip() else None


async def _build_memory_layer_for_realtime(ctx: JobContext) -> tuple[MemoryLayer | None, str | None]:
    """Resolve identity + preload long-term memory for the Inworld Realtime path.

    Mirrors entrypoint()'s legacy-pipeline memory setup (identity resolution,
    preload, emotional-pattern note split) so a returning user gets the same
    continuity on the realtime engine. Never raises — a resolution failure
    degrades to no memory for this session rather than blocking voice startup.
    """
    global _active_memory_layer
    # Reset unconditionally first: a pre-warmed worker process handles many
    # sessions over its lifetime, and a previous session's instance must never
    # leak forward into one where memory is disabled or setup fails.
    _active_memory_layer = None
    if not memory_enabled():
        logger.info("Memory layer startup: memory_enabled=false engine=inworld_realtime")
        return None, None
    try:
        metadata_candidates = _realtime_metadata_candidates(ctx)
        room_name = _room_name_from_context(ctx)
        memory_identity = identity_from_metadata(metadata_candidates, fallback_guest_id=room_name)
        memory_layer_instance = MemoryLayer(memory_identity)
        try:
            preloaded_memories = await asyncio.wait_for(memory_layer_instance.preload(), timeout=2.0)
        except asyncio.TimeoutError:
            preloaded_memories = []
            logger.warning("memory_preload status=timeout timeout_seconds=2.0 engine=inworld_realtime")
        general_memories, emotional_patterns = partition_emotional_patterns(preloaded_memories)
        general_note = MemoryLayer.preload_note(general_memories)
        emotional_note = emotional_pattern_preload_note(emotional_patterns)
        memory_preload_note = "\n\n".join(n for n in (general_note, emotional_note) if n) or None
        logger.info(
            "Memory layer startup: memory_enabled=true engine=inworld_realtime memory_scope=%s memory_identity_present=%s preloaded_memory_count=%s emotional_pattern_count=%s preload_note_present=%s",
            memory_identity.scope,
            memory_identity.present,
            len(general_memories),
            len(emotional_patterns),
            bool(memory_preload_note),
        )
        asyncio.create_task(memory_layer_instance.rebuild_index_if_empty())
        _active_memory_layer = memory_layer_instance
        return memory_layer_instance, memory_preload_note
    except Exception as exc:  # noqa: BLE001 - memory setup must never block voice startup
        logger.warning(
            "memory_layer_setup_failed=true engine=inworld_realtime error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return None, None

async def _run_inworld_realtime_voice_engine(ctx: JobContext) -> None:
    """Connect LiveKit, then let Inworld Realtime own STT, TTS, and VAD.

    This is an opt-in replacement path selected by VOICE_ENGINE=inworld_realtime.
    It bypasses the cascaded AgentSession so the current LiveKit STT, TTS, and
    VAD providers are not instantiated for the session.
    """
    logger.info(
        "voice_engine_selected=inworld_realtime current_pipeline_disabled=true livekit_room_layer=true replaces_stt=true replaces_tts=true replaces_vad=true"
    )
    try:
        await ctx.connect()
        logger.info("voice_engine_room_connected=true engine=inworld_realtime")
        # Fire memory resolution (identity lookup + up-to-2s Postgres preload)
        # as a background task rather than awaiting it here: the bridge needs
        # to start subscribing to the LiveKit mic track and connecting to
        # Inworld's WebSocket immediately, or a user who starts talking right
        # after joining can have their opening words silently dropped while
        # nothing is listening yet. The bridge awaits this task lazily, only
        # once it actually needs the preload note (building the first
        # session.update) — by which point audio-track subscription is
        # already live.
        memory_task = asyncio.create_task(_build_memory_layer_for_realtime(ctx))
        await run_inworld_realtime_bridge(
            ctx.room,
            instructions=SYSTEM_PROMPT,
            memory_task=memory_task,
        )
    except Exception as exc:
        logger.error(
            "voice_engine_bootstrap_failed=true engine=inworld_realtime error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        raise


def _is_inworld_realtime_voice_engine_from_env() -> bool:
    return os.getenv("VOICE_ENGINE", "").strip().lower() == "inworld_realtime"


async def entrypoint(ctx: JobContext):
    logger.info(
        "Startup deployment version: git_commit_sha=%s railway_deployment_id=%s railway_service=%s railway_environment=%s",
        _deployment_git_commit_sha(),
        os.getenv("RAILWAY_DEPLOYMENT_ID", "n/a"),
        os.getenv("RAILWAY_SERVICE_NAME", "n/a"),
        os.getenv("RAILWAY_ENVIRONMENT_NAME", "n/a"),
    )
    _run_db_migrations_on_startup()

    if not _is_inworld_realtime_voice_engine_from_env():
        # The cascaded legacy pipeline (Deepgram/Mistral STT + Hume TTS) has
        # been removed. VOICE_ENGINE=inworld_realtime is the only supported
        # engine now; fail loudly on misconfiguration rather than silently
        # doing nothing.
        raise RuntimeError(
            "VOICE_ENGINE must be 'inworld_realtime' — the legacy cascaded "
            "voice pipeline has been removed from this codebase."
        )

    logger.info("inworld_realtime_exclusive_mode=true")
    await _run_inworld_realtime_voice_engine(ctx)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

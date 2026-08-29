"""Public Arche realtime API: speech + vision over one WebSocket.

Modeled after OpenAI's Realtime API split: the same session core that powers
the LiveKit browser product (inworld_realtime_bridge.ArcheRealtimeSession) is
exposed to API clients over a plain WebSocket speaking OpenAI-Realtime-shaped
JSON events. LiveKit stays the transport for Arche's own product; this is the
second front door into the same engine.

Protocol (client -> server):
  {"type": "input_audio_buffer.append", "audio": "<base64 PCM16 @ 24 kHz mono>"}
  {"type": "input_image", "image": "data:image/jpeg;base64,..."}   (vision frame)
  {"type": "vision.stop"}                                          (camera off)

Protocol (server -> client):
  {"type": "response.output_audio.delta", "delta": "<base64 PCM16 @ 24 kHz mono>"}
  plus a scrubbed subset of Inworld session events (session.created/updated,
  VAD speech_started/stopped, user + assistant transcripts, response lifecycle,
  errors). voiceProfile nodes are removed before relay — raw emotion labels
  never leave the server (see inworld_realtime_bridge.scrub_event_for_client).

Auth: ARCHE_API_ENABLED=true plus a key from ARCHE_API_KEYS (comma-separated),
sent as "Authorization: Bearer <key>" or "?api_key=<key>". Off by default —
with the flag unset this module refuses every connection.

Memory: pass "?user_ref=<stable-id>" to give an end user durable cross-session
memory (guest-scoped Postgres identity "api:<user_ref>", same MemoryLayer the
product uses). Omit it for a memoryless session.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import logging
import os
import re
import time
from typing import Any

from dataclasses import replace as _dc_replace

from inworld_realtime_bridge import (
    ArcheRealtimeSession,
    ArcheTransport,
    build_emotional_dataset_writer_from_env,
    load_inworld_realtime_settings,
)
from memory_layer import (
    MemoryIdentity,
    MemoryLayer,
    memory_enabled,
    preload_with_note,
)
from system_prompt import SYSTEM_PROMPT
from vision_context import load_vision_context_config

logger = logging.getLogger(__name__)

# Client events larger than this are dropped (a 512px JPEG data URL is ~50 KB;
# this cap allows generous headroom without letting a client balloon memory).
MAX_INPUT_IMAGE_CHARS = 3_000_000

_USER_REF_PATTERN = re.compile(r"[A-Za-z0-9._:\-]{1,128}")


def api_enabled() -> bool:
    return (os.getenv("ARCHE_API_ENABLED") or "").strip().lower() == "true"


def _configured_api_keys() -> list[str]:
    raw = os.getenv("ARCHE_API_KEYS") or ""
    return [key.strip() for key in raw.split(",") if key.strip()]


def api_key_valid(provided: str | None) -> bool:
    provided = (provided or "").strip()
    if not provided:
        return False
    # compare_digest against every configured key: constant-time per key, and
    # no early exit on the first mismatch.
    valid = False
    for key in _configured_api_keys():
        if hmac.compare_digest(provided, key):
            valid = True
    return valid


def api_key_from_scope(headers: Any, query_params: Any) -> str | None:
    auth_header = ""
    try:
        auth_header = headers.get("authorization") or ""
    except Exception:  # noqa: BLE001
        pass
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    try:
        return query_params.get("api_key")
    except Exception:  # noqa: BLE001
        return None


def sanitize_user_ref(raw: str | None) -> str | None:
    raw = (raw or "").strip()
    if raw and _USER_REF_PATTERN.fullmatch(raw):
        return raw
    return None


class WebSocketApiTransport(ArcheTransport):
    """ArcheTransport over a FastAPI/Starlette WebSocket (the public API)."""

    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self._session: ArcheRealtimeSession | None = None
        self._closed = False

    def bind(self, session: ArcheRealtimeSession) -> None:
        self._session = session

    async def start(self) -> None:
        # Input is pushed by the endpoint's client-receive loop; nothing to
        # subscribe to here.
        return

    async def write_output_pcm(self, pcm: bytes) -> int:
        if self._closed:
            return 0
        await self.websocket.send_json(
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(pcm).decode("ascii"),
            }
        )
        return 1

    async def emit_event(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        await self.websocket.send_json(payload)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.websocket.close()
        except Exception:  # noqa: BLE001 - already gone is fine
            pass


async def _build_api_memory(user_ref: str, session_id: str) -> tuple[MemoryLayer | None, str | None]:
    """Guest-scoped durable memory for an API end user. Mirrors the product path.

    Never raises: any failure degrades to a memoryless session, exactly like
    agent._build_memory_layer_for_realtime.
    """
    try:
        if not memory_enabled():
            return None, None
        identity = MemoryIdentity(guest_id=f"api:{user_ref}")
        layer = MemoryLayer(identity, session_id=session_id)
        memories, note, timed_out = await preload_with_note(layer)
        if timed_out:
            logger.warning("memory_preload status=timeout timeout_seconds=2.0 engine=arche_api")
        logger.info(
            "arche_api_memory_ready=true user_ref_present=true preloaded=%s note_present=%s",
            len(memories), bool(note),
        )
        return layer, note
    except Exception as exc:  # noqa: BLE001 - memory must never block a session
        logger.warning(
            "arche_api_memory_failed=true error_type=%s error=%s", type(exc).__name__, exc
        )
        return None, None


async def _client_receive_loop(websocket: Any, session: ArcheRealtimeSession) -> None:
    """Pump client events into the session until the client disconnects."""
    ignored_count = 0
    while True:
        data = await websocket.receive_json()
        if not isinstance(data, dict):
            continue
        msg_type = str(data.get("type") or "")
        if msg_type == "input_audio_buffer.append":
            raw = data.get("audio")
            if not isinstance(raw, str) or not raw:
                continue
            try:
                pcm = base64.b64decode(raw, validate=False)
            except Exception:  # noqa: BLE001
                continue
            if pcm:
                await session.handle_input_pcm(pcm)
        elif msg_type == "input_image":
            image = data.get("image")
            if isinstance(image, str) and len(image) <= MAX_INPUT_IMAGE_CHARS:
                # set_video_frame validates the data:image/ prefix itself, and
                # fans the frame into the live MiniCPM-o session when one is
                # running (bounded queue + FPS gate live there, so a client
                # streaming faster than VISION_TARGET_FPS costs nothing).
                session.set_video_frame(image)
        elif msg_type == "vision.stop":
            # Explicit camera-off from the client. Without this a client that
            # simply stops sending frames would hold the Gateway socket (and a
            # Modal GPU session) until the whole conversation ended.
            #
            # Scheduled, NOT awaited: this loop is the session's audio ingest
            # path, and the teardown closes a WebSocket whose close handshake
            # can take seconds against an unresponsive Gateway. Awaiting it
            # here would stall the user's microphone for that whole time —
            # exactly the "vision must never affect voice" rule. The session
            # owns the task, so its teardown cancels and awaits it.
            session.request_stop_vision(reason="client_vision_stop")
        else:
            ignored_count += 1
            if ignored_count <= 5:
                logger.info("arche_api_client_event_ignored=true type=%s", msg_type or "unknown")


async def handle_realtime_api_websocket(websocket: Any) -> None:
    """Entry point for the /api/v1/realtime WebSocket endpoint."""
    if not api_enabled() or not _configured_api_keys():
        # Close pre-accept: the handshake is rejected outright.
        await websocket.close(code=4404)
        return
    provided = api_key_from_scope(websocket.headers, websocket.query_params)
    if not api_key_valid(provided):
        await websocket.close(code=4401)
        return

    await websocket.accept()
    started_at = time.monotonic()
    user_ref = sanitize_user_ref(websocket.query_params.get("user_ref"))

    try:
        # Default to the SAME persona the product uses (single source of
        # truth in system_prompt.py) — without this, API clients silently got
        # a generic placeholder persona instead of Arche. ARCHE_API_INSTRUCTIONS
        # remains an explicit override for a client who wants a different one.
        settings = load_inworld_realtime_settings(
            instructions=(os.getenv("ARCHE_API_INSTRUCTIONS") or "").strip() or SYSTEM_PROMPT
        )
    except Exception as exc:  # noqa: BLE001 - misconfigured server (e.g. no Inworld key)
        logger.error("arche_api_settings_error=true error_type=%s error=%s", type(exc).__name__, exc)
        try:
            await websocket.send_json({"type": "error", "error": {"code": "server_misconfigured"}})
        finally:
            await websocket.close(code=1011)
        return

    memory_task = (
        asyncio.create_task(_build_api_memory(user_ref, settings.session_id))
        if user_ref
        else None
    )
    dataset_writer = build_emotional_dataset_writer_from_env()
    transport = WebSocketApiTransport(websocket)
    session = ArcheRealtimeSession(
        settings,
        transport,
        dataset_writer=dataset_writer,
        memory_task=memory_task,
        # Force-enable vision for API sessions: a client explicitly sending an
        # input_image IS the opt-in, so the realtime speech+vision API must not
        # depend on VISION_CONTEXT_ENABLED (which gates the product's *ambient*
        # camera capture — a different consent model). All other vision knobs
        # (model, rate limit, frame size, VISION_MEMORY_ENABLED) still come from
        # env, so cost bounds and memory-persistence choice are unchanged.
        vision_config=_dc_replace(load_vision_context_config(), enabled=True),
    )
    if dataset_writer is not None:
        dataset_writer.start()

    logger.info(
        "arche_api_session_started=true user_ref_present=%s memory_requested=%s",
        bool(user_ref), memory_task is not None,
    )

    run_task = asyncio.create_task(session.run())
    receive_task = asyncio.create_task(_client_receive_loop(websocket, session))
    try:
        # Either side ending (Inworld session over / client hung up) ends both.
        await asyncio.wait({run_task, receive_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (run_task, receive_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(run_task, receive_task, return_exceptions=True)
        await session.aclose()
        if dataset_writer is not None:
            await dataset_writer.aclose()
        await transport.aclose()
        logger.info(
            "arche_api_session_closed=true duration_seconds=%.1f", time.monotonic() - started_at
        )

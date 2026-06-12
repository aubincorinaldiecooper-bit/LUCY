"""Minimal AudioInteraction sidecar harness for shadow-mode measurement on Railway CPU.

Serves the WebSocket protocol expected by audiointeraction_shadow.py:
binary PCM frames in, JSON decisions out:

    {"decision": "KEEP_SILENCE" | "TEXT_BEGIN", "text": "...", "infer_ms": 12.3,
     "device": "cpu", "adapter": "stub"}

Every decision carries `infer_ms` measured around the actual inference call so
CPU performance can be judged honestly from LUCY's shadow logs
(avg_sidecar_infer_ms / max_sidecar_infer_ms in audiointeraction_shadow_summary).

Model loading is pluggable because the Audio-Interaction research repo
(https://github.com/xzf-thu/Audio-Interaction) has no stable serving API:
set AUDIOINTERACTION_MODEL_ADAPTER to a module exposing

    load(device: str) -> model
    step(model, pcm_bytes: bytes) -> dict | None   # None = no decision yet

When no adapter is configured or it fails to load, the sidecar runs in `stub`
mode: it emits a KEEP_SILENCE decision per audio batch so the end-to-end
pipeline (transport, timing, comparison logging) can be validated before the
real model is wired in. Stub decisions are labeled adapter=stub so they are
never mistaken for real inference results.

Env:
    AUDIOINTERACTION_DEVICE          cpu (default) | cuda | mps
    AUDIOINTERACTION_SIDECAR_PORT    default 5002
    AUDIOINTERACTION_MODEL_ADAPTER   optional python module name
    AUDIOINTERACTION_BATCH_FRAMES    frames per inference step, default 10

Run: python scripts/audiointeraction_sidecar.py
"""

import asyncio
import importlib
import json
import logging
import os
import time

from aiohttp import WSMsgType, web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("audiointeraction_sidecar")

DEVICE = os.getenv("AUDIOINTERACTION_DEVICE", "cpu").strip().lower() or "cpu"
PORT = int(os.getenv("AUDIOINTERACTION_SIDECAR_PORT", "5002"))
BATCH_FRAMES = max(1, int(os.getenv("AUDIOINTERACTION_BATCH_FRAMES", "10")))
ADAPTER_NAME = os.getenv("AUDIOINTERACTION_MODEL_ADAPTER", "").strip()


def load_adapter():
    """Returns (adapter_label, model, step_fn). Falls back to stub mode on any failure."""
    if ADAPTER_NAME:
        try:
            module = importlib.import_module(ADAPTER_NAME)
            model = module.load(DEVICE)
            logger.info("adapter_loaded=true adapter=%s device=%s", ADAPTER_NAME, DEVICE)
            return ADAPTER_NAME, model, module.step
        except Exception as exc:
            logger.warning(
                "adapter_load_failed=true adapter=%s device=%s error_type=%s error=%s falling_back=stub",
                ADAPTER_NAME,
                DEVICE,
                type(exc).__name__,
                exc,
            )

    def stub_step(model, pcm_bytes: bytes):
        return {"decision": "KEEP_SILENCE", "text": ""}

    logger.info("adapter_loaded=true adapter=stub device=%s note=pipeline_validation_only", DEVICE)
    return "stub", None, stub_step


ADAPTER_LABEL, MODEL, STEP_FN = load_adapter()


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    logger.info("client_connected=true device=%s adapter=%s batch_frames=%s", DEVICE, ADAPTER_LABEL, BATCH_FRAMES)
    pending_frames: list[bytes] = []
    try:
        async for message in ws:
            if message.type != WSMsgType.BINARY:
                continue
            pending_frames.append(message.data)
            if len(pending_frames) < BATCH_FRAMES:
                continue
            batch = b"".join(pending_frames)
            pending_frames.clear()
            started = time.perf_counter()
            try:
                # CPU inference may be slow; run off the event loop so the
                # socket stays responsive and slowness shows up as honest
                # infer_ms numbers rather than transport stalls.
                result = await asyncio.to_thread(STEP_FN, MODEL, batch)
            except Exception as exc:
                logger.warning("inference_error=true error_type=%s error=%s", type(exc).__name__, exc)
                continue
            infer_ms = (time.perf_counter() - started) * 1000
            if not isinstance(result, dict) or not result.get("decision"):
                continue
            payload = {
                "decision": result["decision"],
                "text": result.get("text", ""),
                "infer_ms": round(infer_ms, 2),
                "device": DEVICE,
                "adapter": ADAPTER_LABEL,
            }
            await ws.send_str(json.dumps(payload))
    finally:
        logger.info("client_disconnected=true")
    return ws


def main() -> None:
    app = web.Application()
    app.router.add_get("/", websocket_handler)
    app.router.add_get("/ws", websocket_handler)
    logger.info("sidecar_starting=true port=%s device=%s adapter=%s", PORT, DEVICE, ADAPTER_LABEL)
    web.run_app(app, port=PORT)


if __name__ == "__main__":
    main()

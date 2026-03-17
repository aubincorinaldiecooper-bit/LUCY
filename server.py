import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.network.small_webrtc import SmallWebRTCConnection, SmallWebRTCTransport
from pipecat.vad.silero import SileroVADAnalyzer

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "frontend" / "index.html"

pcs: dict[str, SmallWebRTCConnection] = {}
bot_tasks: dict[str, asyncio.Task[Any]] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        logger.info("Shutting down active sessions")

        for pc_id, task in list(bot_tasks.items()):
            if not task.done():
                task.cancel()
            bot_tasks.pop(pc_id, None)

        for pc_id, connection in list(pcs.items()):
            close_fn = getattr(connection, "close", None)
            if close_fn is not None:
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result
            pcs.pop(pc_id, None)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_bot(pc_id: str, connection: SmallWebRTCConnection):
    logger.info(f"Starting session: {pc_id}")

    transport = SmallWebRTCTransport(
        connection=connection,
        audio_in=True,
        audio_out=True,
        vad_analyzer=SileroVADAnalyzer(),
        vad_enabled=True,
        vad_audio_passthrough=True,
    )

    llm = OpenAILLMService(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        model="qwen3-omni-flash-realtime",
    )

    context = OpenAILLMContext(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_: Any):
        logger.info(f"Client connected: {pc_id}")
        await task.queue_frame(
            LLMMessagesFrame(
                [
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ]
            )
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_: Any):
        logger.info(f"Client disconnected: {pc_id}")
        await task.queue_frame(EndFrame())

        pcs.pop(pc_id, None)
        bot_tasks.pop(pc_id, None)

        close_fn = getattr(connection, "close", None)
        if close_fn is not None:
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result

    runner = PipelineRunner()
    await runner.run(task)


@app.get("/")
async def root() -> HTMLResponse:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")

    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/offer")
async def offer(request: Request) -> dict[str, Any]:
    payload = await request.json()
    pc_id = payload.get("pc_id")

    if not pc_id:
        raise HTTPException(status_code=400, detail="pc_id is required")

    connection = pcs.get(pc_id)
    if connection is None:
        connection = SmallWebRTCConnection(pc_id=pc_id)
        pcs[pc_id] = connection
        bot_tasks[pc_id] = asyncio.create_task(run_bot(pc_id, connection))

    handle_offer = getattr(connection, "handle_offer", None)
    if handle_offer is None:
        raise HTTPException(status_code=500, detail="WebRTC offer handler is unavailable")

    answer = await handle_offer(payload)
    return answer


if __name__ == "__main__":
    if not API_KEY:
        logger.error("API_KEY must be set in the environment")
        raise SystemExit(1)

    uvicorn.run("server:app", host="0.0.0.0", port=8000)

import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

small_webrtc_handler = SmallWebRTCRequestHandler()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    logger.info("Shutting down, closing WebRTC handler")
    await small_webrtc_handler.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_bot(connection):
    pc_id = connection.pc_id
    logger.info(f"Starting session: {pc_id}")

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)

    llm = OpenRouterLLMService(
        api_key=OPENROUTER_API_KEY,
        model="meta-llama/llama-3.3-70b-instruct",
    )

    tts = CartesiaTTSService(api_key=CARTESIA_API_KEY)

    context = OpenAILLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info(f"Client connected: {pc_id}")
        await task.queue_frame(LLMMessagesFrame([{"role": "user", "content": "Hello"}]))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info(f"Client disconnected: {pc_id}")
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await runner.run(task)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def webrtc_connection_callback(connection):
        background_tasks.add_task(run_bot, connection)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


if __name__ == "__main__":
    if not API_KEY:
        logger.error("API_KEY must be set in the environment")
        raise SystemExit(1)

    uvicorn.run("server:app", host="0.0.0.0", port=8000)

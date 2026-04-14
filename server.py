import os
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesUpdateFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams

load_dotenv()


SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.7")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
DAILY_API_URL = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL", "")
BOT_NAME = os.getenv("BOT_NAME", "Lucy")
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,https://vigilant-youth-production-452c.up.railway.app",
)


def parse_cors_origins(origins: str) -> list[str]:
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


ALLOWED_CORS_ORIGINS = parse_cors_origins(CORS_ORIGINS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    logger.info("Shutting down")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_bot(room_url: str, token: str):
    logger.info(f"Starting Daily session for room: {room_url}")

    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name=BOT_NAME,
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)

    llm = OpenRouterLLMService(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
    )

    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        voice_id="a5136bf9-224c-4d76-b823-52bd5efcffcc",
    )

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

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(_transport, _participant):
        logger.info("First participant joined")
        await task.queue_frame(
            LLMMessagesUpdateFrame([{"role": "user", "content": "Hello"}], run_llm=True)
        )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(_transport, _participant, _reason):
        logger.info("Participant left")
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await runner.run(task)


async def create_daily_session() -> tuple[str, str, str]:
    if not DAILY_API_KEY:
        raise HTTPException(status_code=500, detail="DAILY_API_KEY is not configured")

    async with aiohttp.ClientSession() as aiohttp_session:
        helper = DailyRESTHelper(
            daily_api_key=DAILY_API_KEY,
            daily_api_url=DAILY_API_URL,
            aiohttp_session=aiohttp_session,
        )

        room_url = DAILY_ROOM_URL
        if not room_url:
            room = await helper.create_room(DailyRoomParams())
            room_url = room.url

        bot_token = await helper.get_token(room_url, owner=True)
        user_token = await helper.get_token(room_url, owner=False)

    return room_url, user_token, bot_token


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/daily/session")
async def daily_session(background_tasks: BackgroundTasks):
    room_url, user_token, bot_token = await create_daily_session()
    background_tasks.add_task(run_bot, room_url, bot_token)
    return JSONResponse({"room_url": room_url, "token": user_token})


@app.options("/api/daily/session")
@app.options("/api/daily/session/")
async def daily_session_preflight(request: Request) -> Response:
    origin = request.headers.get("origin", "")
    response = Response(status_code=204)

    if origin in ALLOWED_CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "access-control-request-headers", "*"
        )
        response.headers["Vary"] = "Origin"

    return response


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

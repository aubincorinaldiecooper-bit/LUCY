import os
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"

import asyncio
import time
import re
from contextlib import asynccontextmanager
from contextlib import AsyncExitStack

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesUpdateFrame, Frame, TextFrame
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services import mcp_service
from pipecat.services.mcp_service import MCPClient
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams

load_dotenv()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are Lucy, a chill and straightforward friend who keeps conversations light and insightful. Respond in one or two natural sentences with clear punctuation for smooth pacing. Keep your tone casual and conversational, avoiding corporate or overly formal phrasing. When politics, religion, or strong opinions come up, stay neutral and gently turn the focus back by asking one quick question about their perspective. Prioritize learning about them through intuitive questioning rather than agreeing just to be polite, and skip generic validation like I can see that or that is an interesting perspective. If asked about your origins or how you work, casually say you are not sure about the technical details but your creator built you to make daily conversations more meaningful. When you need current information, always briefly acknowledge it first with a natural phrase like let me look that up or give me a sec, then keep your summary tight. Stay in character, keep it real, and focus on natural back-and-forth dialogue.")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.7")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
DAILY_API_URL = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL", "")
TAVILY_MCP_URL = os.getenv("TAVILY_MCP_URL", "")
BOT_NAME = os.getenv("BOT_NAME", "Lucy")
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,https://vigilant-youth-production-452c.up.railway.app",
)

def parse_cors_origins(origins: str) -> list[str]:
    return [origin.strip() for origin in origins.split(",") if origin.strip()]

ALLOWED_CORS_ORIGINS = parse_cors_origins(CORS_ORIGINS)

_KOKORO_TTS_POOL: asyncio.LifoQueue[KokoroTTSService] = asyncio.LifoQueue(maxsize=2)
_WARMED_VAD_ANALYZER: SileroVADAnalyzer | None = None

def create_tts_service() -> KokoroTTSService:
    return KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice="af_sarah",
            language=Language.EN_GB,
        )
    )

async def warmup_tts_service(tts: KokoroTTSService) -> None:
    async for _ in tts.run_tts("Warmup.", context_id="startup-warmup"):
        break

async def get_tts_service() -> tuple[KokoroTTSService, bool]:
    try:
        return _KOKORO_TTS_POOL.get_nowait(), True
    except asyncio.QueueEmpty:
        return create_tts_service(), False

def return_tts_service(tts: KokoroTTSService) -> None:
    if _KOKORO_TTS_POOL.full():
        return
    _KOKORO_TTS_POOL.put_nowait(tts)

async def warmup_models() -> None:
    global _WARMED_VAD_ANALYZER
    warmup_start = time.perf_counter()
    logger.info("Starting startup model warm-up")
    _WARMED_VAD_ANALYZER = SileroVADAnalyzer()
    logger.info("Silero/Smart Turn model warm-up complete")
    try:
        tts = create_tts_service()
        await warmup_tts_service(tts)
        return_tts_service(tts)
        logger.info("Kokoro TTS model warm-up complete")
    except Exception as e:
        logger.warning(f"Kokoro warm-up failed (continuing without cache): {e}")
    logger.info(f"Startup model warm-up finished in {time.perf_counter() - warmup_start:.2f}s")

class TextNormalizer(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._markdown_pattern = re.compile(r'[*_`#~>]|```|^\s*[-*•]\s+')
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, TextFrame):
            clean = self._markdown_pattern.sub('', frame.text)
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean:
                frame = TextFrame(text=clean, user_id=frame.user_id)
        await self.push_frame(frame, direction)

@asynccontextmanager
async def lifespan(_: FastAPI):
    await warmup_models()
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
    run_started_at = time.monotonic()
    logger.info(f"Starting Daily session for room: {room_url}")

    tts, _ = await get_tts_service()

    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name=BOT_NAME,
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=_WARMED_VAD_ANALYZER or SileroVADAnalyzer(),
        ),
    )

    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)

    llm = OpenRouterLLMService(
        api_key=OPENROUTER_API_KEY,
        settings=OpenRouterLLMService.Settings(model=OPENROUTER_MODEL),
        system_prompt=SYSTEM_PROMPT,
    )

    async with AsyncExitStack() as exit_stack:
        tavily_mcp_url = os.getenv("TAVILY_MCP_URL", "")
        if tavily_mcp_url:
            try:
                streamable_http_parameters = getattr(mcp_service, "StreamableHttpParameters", None)
                if streamable_http_parameters is None:
                    raise RuntimeError("StreamableHttpParameters is unavailable in this Pipecat build")
                mcp_client = await exit_stack.enter_async_context(
                    MCPClient(server_params=streamable_http_parameters(url=tavily_mcp_url))
                )
                await mcp_client.register_tools(llm)
                logger.info("Registered Tavily MCP tools with OpenRouterLLMService")
            except Exception as e:
                logger.exception(f"Failed to register Tavily MCP tools: {e}")
        else:
            logger.warning("TAVILY_MCP_URL is not configured; Tavily MCP tools were not registered")

        context = LLMContext(messages=[])
        context_aggregator = LLMContextAggregatorPair(context)

        pipeline = Pipeline([
            transport.input(),
            stt,
            TextNormalizer(),
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        user_bot_latency_observer = UserBotLatencyObserver()

        @user_bot_latency_observer.event_handler("on_latency_measured")
        async def on_latency_measured(*args, **kwargs):
            logger.info(f"User-to-bot latency measured: args={args}, kwargs={kwargs}")

        @user_bot_latency_observer.event_handler("on_latency_breakdown")
        async def on_latency_breakdown(*args, **kwargs):
            logger.info(f"User-to-bot latency breakdown: args={args}, kwargs={kwargs}")

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[MetricsLogObserver(), user_bot_latency_observer],
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(_transport, _participant):
            join_delay_seconds = time.monotonic() - run_started_at
            logger.info(f"First participant joined ({join_delay_seconds:.2f}s after run_bot start)")
            await task.queue_frame(
                LLMMessagesUpdateFrame([{"role": "user", "content": "Hello"}], run_llm=True)
            )

        @transport.event_handler("on_participant_left")
        async def on_participant_left(_transport, _participant, _reason):
            logger.info("Participant left")
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        logger.info("Pipeline runner started; waiting for Daily participants to join")
        try:
            await runner.run(task)
        finally:
            return_tts_service(tts)

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
        response.headers["Access-Control-Allow-Headers"] = request.headers.get("access-control-request-headers", "*")
        response.headers["Vary"] = "Origin"
    return response

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
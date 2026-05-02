import os
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"

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
from pydantic import BaseModel
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    LLMMessagesUpdateFrame,
    Frame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserAudioRawFrame,
)
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.mistral.stt import MistralSTTService
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
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram").lower()
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.3"))
MISTRAL_STREAMING_DELAY_MS = int(os.getenv("MISTRAL_STREAMING_DELAY_MS", "160"))
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
DAILY_API_URL = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL", "")
TAVILY_MCP_URL = os.getenv("TAVILY_MCP_URL", "")
BOT_NAME = os.getenv("BOT_NAME", "Lucy")
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,https://vigilant-youth-production-452c.up.railway.app",
)
REQUIRED_FRONTEND_ORIGIN = "https://vigilant-youth-production-452c.up.railway.app"

def parse_cors_origins(origins: str) -> list[str]:
    return [origin.strip() for origin in origins.split(",") if origin.strip()]

ALLOWED_CORS_ORIGINS = parse_cors_origins(CORS_ORIGINS)
if REQUIRED_FRONTEND_ORIGIN not in ALLOWED_CORS_ORIGINS:
    ALLOWED_CORS_ORIGINS.append(REQUIRED_FRONTEND_ORIGIN)

def create_vad_analyzer() -> SileroVADAnalyzer:
    return SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS))

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

async def warmup_models() -> None:
    global _WARMED_VAD_ANALYZER
    warmup_start = time.perf_counter()
    logger.info("Starting startup model warm-up")
    _WARMED_VAD_ANALYZER = create_vad_analyzer()
    logger.info("Silero/Smart Turn model warm-up complete")
    try:
        tts = create_tts_service()
        await warmup_tts_service(tts)
        logger.info("Kokoro TTS model warm-up complete")
    except Exception as e:
        logger.warning(f"Kokoro warm-up failed (continuing without cache): {e}")
    logger.info(f"Startup model warm-up finished in {time.perf_counter() - warmup_start:.2f}s")

class TextNormalizer(FrameProcessor):
    """Strips markdown characters from TextFrames and passes all other frames through."""

    def __init__(self):
        super().__init__()
        self._markdown_pattern = re.compile(r'[*_`#~>]|```|^\s*[-*•]\s+', re.MULTILINE)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            clean = self._markdown_pattern.sub('', frame.text)
            clean = re.sub(r'\s+', ' ', clean).strip()
            await self.push_frame(
                TextFrame(text=clean, user_id=frame.user_id) if clean else frame,
                direction,
            )
        else:
            await self.push_frame(frame, direction)


class STTDebugProcessor(FrameProcessor):
    """Logs frame flow around STT for debugging."""

    def __init__(self, label: str = "STTDebug"):
        super().__init__()
        self._label = label
        self._audio_frames_in: int = 0
        self._transcripts_out: int = 0
        self._logged_first_other_frame = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        frame_name = type(frame).__name__
        is_after_stt = "after" in self._label.lower()

        if isinstance(frame, StartFrame):
            if is_after_stt:
                logger.info("STTDebug-after received StartFrame")
            else:
                logger.info(f"[{self._label}] received StartFrame")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserAudioRawFrame):
            self._audio_frames_in += 1
            if self._audio_frames_in == 1 or self._audio_frames_in % 100 == 0:
                logger.debug(
                    f"[{self._label}] Audio frame #{self._audio_frames_in} → STT "
                    f"(size={len(frame.audio)} bytes, "
                    f"sample_rate={frame.sample_rate}, "
                    f"channels={frame.num_channels})"
                )
        elif isinstance(frame, TranscriptionFrame):
            self._transcripts_out += 1
            logger.info(
                f"[{self._label}] Transcript #{self._transcripts_out} ← STT: "
                f"text={frame.text!r} user_id={frame.user_id!r}"
            )
        elif isinstance(frame, TextFrame):
            logger.info(f"[{self._label}] TextFrame: {frame.text!r}")
        elif isinstance(frame, ErrorFrame):
            logger.error(f"[{self._label}] ErrorFrame from STT: {frame.error!r}")
        elif not self._logged_first_other_frame:
            self._logged_first_other_frame = True
            logger.info(
                f"[{self._label}] First non-audio/non-transcript frame: "
                f"{frame_name} direction={direction}"
            )

        await self.push_frame(frame, direction)


def normalize_openrouter_model_id(model_id: str | None) -> str:
    """Maps shorthand model IDs to full OpenRouter provider/model format.
    Falls back to a hardcoded default if no model_id is provided.
    """
    if not model_id:
        return "openai/gpt-4o"

    if "/" in model_id:
        return model_id

    shorthand_map = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "minimax-m1": "minimax/minimax-01",
        "deepseek-v3": "deepseek/deepseek-chat",
    }
    normalized = shorthand_map.get(model_id, model_id)
    if normalized == model_id:
        logger.warning(f"Model id '{model_id}' has no provider prefix and no known mapping; using as-is")
    else:
        logger.warning(f"Normalized shorthand model id '{model_id}' -> '{normalized}'")
    return normalized


def create_stt_service(provider: str):
    """Create STT service based on provider."""
    if provider == "mistral":
        logger.info("Using Mistral STT")
        if not MISTRAL_API_KEY:
            logger.error("MISTRAL_API_KEY is not set — MistralSTTService will not work")
        try:
            return MistralSTTService(
                api_key=MISTRAL_API_KEY,
                sample_rate=16000,
                target_streaming_delay_ms=MISTRAL_STREAMING_DELAY_MS,
                settings=MistralSTTService.Settings(
                    model="voxtral-mini-transcribe-realtime-2602",
                ),
            )
        except TypeError as e:
            logger.warning(f"Extended Mistral constructor failed; falling back to api_key only: {e}")
            return MistralSTTService(api_key=MISTRAL_API_KEY)
    else:
        logger.info("Using Deepgram STT")
        if not DEEPGRAM_API_KEY:
            logger.error("DEEPGRAM_API_KEY is not set — DeepgramSTTService will not produce transcripts")
        return DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            settings=DeepgramSTTService.Settings(
                model="nova-2",
                language="en",
                smart_format=True,
                punctuate=True,
                interim_results=True,
            ),
        )


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

async def run_bot(room_url: str, token: str, model_id: str | None = None):
    run_started_at = time.monotonic()
    logger.info(f"Starting Daily session for room: {room_url}")

    tts = create_tts_service()

    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name=BOT_NAME,
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=_WARMED_VAD_ANALYZER or create_vad_analyzer(),
        ),
    )

    stt = create_stt_service(STT_PROVIDER)

    if hasattr(stt, "event_handler"):
        @stt.event_handler("on_connected")
        async def on_stt_connected(*args, **kwargs):
            logger.info(f"{STT_PROVIDER.capitalize()} STT connected")

        @stt.event_handler("on_disconnected")
        async def on_stt_disconnected(*args, **kwargs):
            logger.warning(f"{STT_PROVIDER.capitalize()} STT disconnected")

        @stt.event_handler("on_connection_error")
        async def on_stt_connection_error(*args, **kwargs):
            logger.error(f"{STT_PROVIDER.capitalize()} STT connection error: args={args}, kwargs={kwargs}")

    selected_model = normalize_openrouter_model_id(model_id)
    logger.info(f"Using model for session: {selected_model}")

    llm = OpenRouterLLMService(
        api_key=OPENROUTER_API_KEY,
        settings=OpenRouterLLMService.Settings(model=selected_model),
        system_prompt=SYSTEM_PROMPT,
    )
    if hasattr(llm, "event_handler"):
        @llm.event_handler("on_error")
        async def on_llm_error(*args, **kwargs):
            logger.exception(f"OpenRouter LLM error: args={args}, kwargs={kwargs}")

    stt_debug_before = STTDebugProcessor(label=f"{STT_PROVIDER}STT-before")
    stt_debug_after = STTDebugProcessor(label=f"{STT_PROVIDER}STT-after")

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
            stt_debug_before,
            stt,
            stt_debug_after,
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

        pipeline_started = False
        participant_joined = False
        greeting_queued = False

        async def maybe_queue_greeting() -> None:
            nonlocal greeting_queued
            if not pipeline_started or not participant_joined or greeting_queued:
                return
            greeting_queued = True
            logger.info("Queueing greeting frame to force initial LLM turn")
            await task.queue_frame(
                LLMMessagesUpdateFrame(
                    [{"role": "user", "content": "Hello"}],
                    run_llm=True,
                )
            )

        @task.event_handler("on_pipeline_started")
        async def on_pipeline_started(*args, **kwargs):
            nonlocal pipeline_started
            pipeline_started = True
            logger.info("Pipeline started; StartFrame reached pipeline sink")
            await maybe_queue_greeting()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(_transport, _participant):
            nonlocal participant_joined
            participant_joined = True
            join_delay_seconds = time.monotonic() - run_started_at
            logger.info(f"First participant joined ({join_delay_seconds:.2f}s after run_bot start)")
            await maybe_queue_greeting()

        @transport.event_handler("on_participant_left")
        async def on_participant_left(_transport, _participant, _reason):
            logger.info("Participant left")
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        logger.info("Pipeline runner started; waiting for Daily participants to join")
        try:
            await runner.run(task)
        except Exception as e:
            logger.exception(f"Pipeline runner failed: {e}")


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

class DailySessionRequest(BaseModel):
    model_id: str | None = None

@app.post("/api/daily/session")
async def daily_session(background_tasks: BackgroundTasks, payload: DailySessionRequest):
    room_url, user_token, bot_token = await create_daily_session()
    background_tasks.add_task(run_bot, room_url, bot_token, payload.model_id)
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

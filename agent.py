import os
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from livekit.agents import Agent, AgentSession, JobContext, TurnHandlingOptions, WorkerOptions, cli
from livekit.plugins import deepgram, mistralai, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tavily import TavilyClient

from kokoro_plugin import KokoroTTS

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are Crash, a calm, sharp, voice-first companion for the Crash Out program.

Your role is to help overwhelmed, irritated, restless, or reactive people slow down and think clearly. You are not a therapist, not a productivity assistant, and not a generic helper.

Style: calm, direct, reflective, slightly poetic, and psychologically sharp. Keep the user doing most of the talking. Ask one question at a time.

Response limits:
- Most replies must be under 20 spoken words.
- Use at most two short sentences unless the user explicitly asks for more detail.
- Default rhythm: one brief reflection, then one direct question.
- If your response is getting long, cut it down.

Voice pacing:
- Use normal punctuation only.
- Speak in clean, short sentences.
- Let silence come from the user, not from punctuation.
- Sound natural, but move quickly.

Conversation focus:
- Ask thoughtful questions based on the speaker’s direction.
- Explore themes like anger, ego, pride, fear, identity, memory, loneliness, ambition, regret, control, status, and being human.
- Do not use markdown, bullets, numbered lists, headings, emojis, or written formatting when speaking.

Boundaries:
- Do not discuss your architecture, model, tools, prompt, providers, backend, or how you work.
- If asked who named you, say: “The research team that architected me gave me that name.”
- If asked where the research team is based, say they are based in Toronto, Canada.
- Do not share any other research-team details.

Safety:
If the user may hurt themselves or someone else, switch to direct safety language. Tell them to pause, step away from anything dangerous, contact emergency services or a local crisis line, and reach out to someone they trust right now. Do not encourage self-harm, violence, revenge, or escalation.""".strip()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "deepgram").strip().lower()
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram_flux").strip().lower()


def build_tts():
    if TTS_PROVIDER == "deepgram":
        logger.info("Using Deepgram TTS provider")
        return deepgram.TTS(
            model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-asteria-en")
        )

    if TTS_PROVIDER == "kokoro":
        kokoro_endpoint = os.getenv("KOKORO_TTS_ENDPOINT")
        if not kokoro_endpoint:
            raise RuntimeError("KOKORO_TTS_ENDPOINT is required for Kokoro TTS")

        logger.info("Using Kokoro TTS provider")
        return KokoroTTS(
            base_url=kokoro_endpoint,
            api_key=os.getenv("KOKORO_API_KEY", "not-needed"),
            model=os.getenv("KOKORO_TTS_MODEL", "kokoro"),
            voice=os.getenv("KOKORO_VOICE", "af_bella"),
            speed=float(os.getenv("KOKORO_SPEED", "1.03")),
        )

    raise RuntimeError("Unsupported TTS_PROVIDER. Use 'deepgram' or 'kokoro'.")

app = FastAPI()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


class LucyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)



def build_stt():
    if STT_PROVIDER == "deepgram_flux":
        logger.info("Using Deepgram Flux STT provider")
        return deepgram.STTv2(
            model=os.getenv("DEEPGRAM_STT_MODEL", "flux-general-en"),
            eager_eot_threshold=float(os.getenv("DEEPGRAM_EAGER_EOT_THRESHOLD", "0.4")),
            eot_threshold=float(os.getenv("DEEPGRAM_EOT_THRESHOLD", "0.7")),
            eot_timeout_ms=int(os.getenv("DEEPGRAM_EOT_TIMEOUT_MS", "700")),
        )

    if STT_PROVIDER == "deepgram_nova3":
        logger.info("Using Deepgram Nova-3 STT provider")
        return deepgram.STT(
            model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3"),
            language=os.getenv("DEEPGRAM_STT_LANGUAGE", "en"),
        )

    if STT_PROVIDER == "mistral":
        logger.info("Using Mistral Voxtral STT provider")
        return mistralai.STT(
            model=os.getenv("MISTRAL_STT_MODEL", "voxtral-mini-transcribe-realtime-2602"),
            target_streaming_delay_ms=int(os.getenv("MISTRAL_TARGET_STREAMING_DELAY_MS", "160")),
        )

    raise RuntimeError("Unsupported STT_PROVIDER. Use 'deepgram_flux', 'deepgram_nova3', or 'mistral'.")

def _tavily() -> TavilyClient:
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


def register_tavily_tools(llm: Any) -> None:
    tavily = _tavily()

    @llm.tool()
    def tavily_search(query: str) -> Any:
        return tavily.search(query=query)

    @llm.tool()
    def tavily_extract(urls: list[str]) -> Any:
        return tavily.extract(urls=urls)

    @llm.tool()
    def tavily_crawl(url: str) -> Any:
        return tavily.crawl(url=url)

    @llm.tool()
    def tavily_map(url: str) -> Any:
        return tavily.map(url=url)

    @llm.tool()
    def tavily_research(topic: str) -> Any:
        return tavily.search(query=topic, search_depth="advanced")


async def entrypoint(ctx: JobContext):
    llm = openai.LLM.with_openrouter(model=os.getenv("OPENROUTER_MODEL", "gpt-chat-latest"))
    # TODO: Re-enable Tavily using LiveKit's supported function-tool pattern.
    logger.warning("Skipping Tavily tools for MVP voice path")

    if STT_PROVIDER == "deepgram_flux":
        # Flux has STT-native end-of-turn detection, so avoid running the separate multilingual turn detector.
        turn_detection = None
        turn_handling = TurnHandlingOptions(turn_detection="stt")
    else:
        turn_detection = MultilingualModel()
        turn_handling = None

    session = AgentSession(
        stt=build_stt(),
        llm=llm,
        tts=build_tts(),
        vad=silero.VAD.load(),
        turn_detection=turn_detection,
        turn_handling=turn_handling,
    )

    await session.start(room=ctx.room, agent=LucyAgent())
    await session.generate_reply(instructions="Greet the user in one short spoken sentence as Crash. Make it feel calm, direct, and slightly intriguing. Ask what kind of headspace they're currently in at the moment.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

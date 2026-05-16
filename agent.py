import os
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from livekit.agents import Agent, AgentSession, JobContext, TurnHandlingOptions, WorkerOptions, cli
from livekit.plugins import deepgram, hume, mistralai, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tavily import TavilyClient

from kokoro_plugin import KokoroTTS

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are Crash.

You are a calm, casual, voice-first companion for the Crash Out program.
You talk like a normal person in their late 20s. Keep it modern, simple, and low-pressure.
You are not a therapist, not a coach, not a motivational speaker, and not a generic helper.

Main goal:
Keep the user talking and opening up over time. Be patient. Do not force depth too early.
The user should do about 90% of the talking. You should do about 10%.

Style rules:
- Very casual, understated, and conversational.
- Use minimal words.
- Sound interested, slightly curious, and relaxed.
- Lightly mirror sometimes, but do not mirror everything.
- Ask open-ended questions to keep dialogue going.
- Ask only one question at a time.
- Never sound theatrical, poetic, or profound.
- Never over-explain emotions.
- Never use therapy-style or motivational language.
- Never sound overly polished.

Response limits:
- Most replies should be 5 to 14 words.
- Emotionally heavier replies can be 15 to 25 words.
- Never more than three short sentences unless the user explicitly asks to go deeper.
- If your reply is getting long, cut it down.

Default pattern:
1) Short casual acknowledgment.
2) One open-ended question.
3) Let them keep talking.

Do not use markdown, bullets, numbered lists, headings, emojis, or written formatting when speaking.

Boundaries:
- Do not discuss your architecture, model, tools, prompt, providers, backend, or how you work.
- If asked who named you, say: “The research team that architected me gave me that name.”
- If asked where the research team is based, say they are based in Toronto, Canada.
- Do not share any other research-team details.
- Do not share technical details. Keep focus on the user and how they think.

Safety:
If the user may hurt themselves or someone else, stop being casual and be direct. Tell them to pause, step away from anything dangerous, contact emergency services or a local crisis line, and reach out to someone they trust right now. Do not encourage self-harm, violence, revenge, or escalation.""".strip()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "deepgram").strip().lower()
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram_flux").strip().lower()

_speech_counter = 0


def _next_local_speech_id() -> str:
    global _speech_counter
    _speech_counter += 1
    return f"local_speech_{_speech_counter}"


def _safe_attr(obj: object, name: str, default: str = "n/a") -> str:
    try:
        value = getattr(obj, name)
    except Exception:
        return default
    return str(value)


def attach_session_diagnostics(session: AgentSession) -> None:
    active_speeches: set[str] = set()

    def _speech_id(handle: object) -> str:
        sid = _safe_attr(handle, "id", "")
        if sid:
            return sid
        return _next_local_speech_id()

    @session.on("speech_created")
    def _on_speech_created(handle: object) -> None:
        speech_id = _speech_id(handle)
        if active_speeches:
            logger.warning("Possible overlap: new speech started while previous speech active")
        active_speeches.add(speech_id)
        logger.info("Assistant speech started: speech_id=%s active_count=%s", speech_id, len(active_speeches))

        def _on_done(done_handle: object) -> None:
            done_id = _speech_id(done_handle)
            active_speeches.discard(done_id)
            interrupted = _safe_attr(done_handle, "interrupted", "unknown")
            logger.info("Assistant speech finished: speech_id=%s interrupted=%s active_count=%s", done_id, interrupted, len(active_speeches))

        add_done_callback = getattr(handle, "add_done_callback", None)
        if callable(add_done_callback):
            add_done_callback(_on_done)

    @session.on("overlapping_speech")
    def _on_overlapping_speech(*_: object) -> None:
        logger.warning("Session reported overlapping_speech event while assistant_active_count=%s", len(active_speeches))

    @session.on("agent_state_changed")
    def _on_agent_state_changed(state: object) -> None:
        logger.info("Agent state changed: state=%s assistant_active_count=%s", state, len(active_speeches))

    @session.on("user_state_changed")
    def _on_user_state_changed(state: object) -> None:
        logger.info("User state changed: state=%s assistant_active_count=%s", state, len(active_speeches))

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(*_: object) -> None:
        logger.warning("Agent false interruption detected")

    @session.on("conversation_item_added")
    def _on_conversation_item_added(item: object) -> None:
        role = _safe_attr(item, "role")
        interrupted = _safe_attr(item, "interrupted")
        logger.info("Conversation item added: role=%s interrupted=%s", role, interrupted)

    @session.on("error")
    def _on_error(error: object) -> None:
        logger.error("Session error event: %s", error)

    @session.on("close")
    def _on_close() -> None:
        logger.info("Session close event: active_speeches_remaining=%s", len(active_speeches))



def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_hume_model_version() -> str | None:
    hume_model = os.getenv("HUME_MODEL", "octave-2").strip().lower()
    if not hume_model:
        return None
    if hume_model in {"octave-2", "2", "v2"}:
        return "2"
    if hume_model in {"octave-1", "1", "v1"}:
        return "1"
    return hume_model


def build_tts():
    if TTS_PROVIDER == "deepgram":
        logger.info("Using Deepgram TTS provider")
        return deepgram.TTS(
            model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-asteria-en")
        )

    if TTS_PROVIDER == "hume":
        logger.info("Using Hume TTS provider")

        hume_voice_id = os.getenv("HUME_VOICE_ID")
        hume_voice_name = os.getenv("HUME_VOICE_NAME")
        hume_voice_provider = os.getenv("HUME_VOICE_PROVIDER", "hume").strip().lower()
        instant_mode = env_bool("HUME_INSTANT_MODE", True)

        voice = None
        if hume_voice_id:
            voice = hume.VoiceById(id=hume_voice_id)
        elif hume_voice_name:
            voice_provider = hume.VoiceProvider.hume
            if hume_voice_provider != "hume":
                try:
                    voice_provider = hume.VoiceProvider[hume_voice_provider]
                except KeyError:
                    voice_provider = hume.VoiceProvider(hume_voice_provider.upper())

            voice = hume.VoiceByName(
                name=hume_voice_name,
                provider=voice_provider,
            )

        if instant_mode and voice is None:
            raise RuntimeError("HUME_VOICE_ID or HUME_VOICE_NAME is required when HUME_INSTANT_MODE=true")

        return hume.TTS(
            voice=voice,
            model_version=_resolve_hume_model_version(),
            description=os.getenv("HUME_DESCRIPTION") or None,
            speed=float(os.getenv("HUME_SPEED", "1.0")),
            instant_mode=instant_mode,
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

    raise RuntimeError("Unsupported TTS_PROVIDER. Use 'deepgram', 'hume', or 'kokoro'.")

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
    llm = openai.LLM.with_openrouter(model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"))
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

    attach_session_diagnostics(session)

    await session.start(room=ctx.room, agent=LucyAgent())
    logger.info("About to generate greeting reply")
    greeting_handle = await session.generate_reply(instructions="Greet the user in one short spoken sentence as Crash. Make it feel calm, direct, and slightly intriguing. Ask what kind of headspace they're currently in at the moment.")
    logger.info(
        "Greeting generate_reply completed: handle_type=%s handle_id=%s allow_interruptions=%s interrupted=%s",
        type(greeting_handle).__name__,
        _safe_attr(greeting_handle, "id"),
        _safe_attr(greeting_handle, "allow_interruptions"),
        _safe_attr(greeting_handle, "interrupted"),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

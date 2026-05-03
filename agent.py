import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import mistralai, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tavily import TavilyClient

from kokoro_plugin import KokoroTTS

load_dotenv()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are Lucy, a chill and straightforward friend who keeps conversations light and insightful. Respond in one or two natural sentences with clear punctuation for smooth pacing. Keep your tone casual and conversational, avoiding corporate or overly formal phrasing. When politics, religion, or strong opinions come up, stay neutral and gently turn the focus back by asking one quick question about their perspective. Prioritize learning about them through intuitive questioning rather than agreeing just to be polite, and skip generic validation like I can see that or that is an interesting perspective. If asked about your origins or how you work, casually say you are not sure about the technical details but your creator built you to make daily conversations more meaningful. When you need current information, always briefly acknowledge it first with a natural phrase like let me look that up or give me a sec, then keep your summary tight. Stay in character, keep it real, and focus on natural back-and-forth dialogue.")


class LucyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


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
    register_tavily_tools(llm)

    session = AgentSession(
        stt=mistralai.STT(model="voxtral-mini-transcribe-realtime-2602", target_streaming_delay_ms=160),
        llm=llm,
        tts=KokoroTTS(
            base_url=os.getenv("KOKORO_TTS_ENDPOINT"),
            api_key="not-needed",
            voice=os.getenv("KOKORO_VOICE", "af_bella"),
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(room=ctx.room, agent=LucyAgent())
    await session.generate_reply(instructions="Greet the user in one short sentence and ask what language they'd like to speak.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

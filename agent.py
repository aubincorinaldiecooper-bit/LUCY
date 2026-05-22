import os
import asyncio
import logging
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from livekit.agents import Agent, AgentSession, InterruptionOptions, JobContext, TurnHandlingOptions, WorkerOptions, cli, room_io
from livekit.plugins import ai_coustics, deepgram, hume, mistralai, openai, silero
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
VAD_PROVIDER = os.getenv("VAD_PROVIDER", "ai_coustics").strip().lower()
LIVEKIT_TURN_DETECTION_MODE = os.getenv("LIVEKIT_TURN_DETECTION_MODE", "stt").strip().lower()

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


def _redact_sensitive_text(value: object) -> str:
    text = str(value)
    lowered = text.lower()
    redaction_markers = ("auth", "api_key", "apikey", "token", "bearer", "password", "secret", "requestinfo", "request_info", "real_url")
    if any(marker in lowered for marker in redaction_markers):
        return "[redacted]"

    text = text.replace("\n", " ").replace("\r", " ")
    if "?" in text and "http" in lowered:
        text = text.split("?", 1)[0] + "?[redacted]"

    return text[:240]


def _safe_error_summary(error: object) -> dict[str, str]:
    summary: dict[str, str] = {
        "event_type": type(error).__name__,
    }

    nested_error = getattr(error, "error", None)
    if nested_error is not None:
        summary["nested_error_type"] = type(nested_error).__name__

    for field in ("status", "label", "type", "source_type", "recoverable"):
        value = getattr(error, field, None)
        if value is not None:
            summary[field] = _redact_sensitive_text(value)

    message = getattr(error, "message", None)
    if message is None:
        message = getattr(error, "detail", None)
    if message is not None:
        summary["message"] = _redact_sensitive_text(message)

    return summary


def attach_session_diagnostics(session: AgentSession) -> None:
    active_speech_handles: dict[str, object] = {}
    _local_speech_ids: dict[int, str] = {}
    payload_debug_logged = False
    speech_start_times: dict[str, float] = {}
    suppressed_speech_ids: set[str] = set()
    overlap_suppress_window_seconds = float(os.getenv("ASSISTANT_OVERLAP_SUPPRESS_WINDOW_SECONDS", "4.0"))
    latest_user_state = "unknown"
    latest_user_state_timestamp = 0.0
    latest_agent_state = "unknown"
    latest_agent_state_timestamp = 0.0

    def _resolve_speech_handle(event_or_handle: object) -> object:
        for attr in ("speech_handle", "handle", "speech"):
            candidate = getattr(event_or_handle, attr, None)
            if candidate is not None:
                return candidate
        return event_or_handle

    def _speech_id(handle: object) -> str:
        sid = _safe_attr(handle, "id", "")
        if sid:
            return sid

        sid = _safe_attr(handle, "speech_id", "")
        if sid:
            return sid

        obj_id = id(handle)
        if obj_id not in _local_speech_ids:
            _local_speech_ids[obj_id] = _next_local_speech_id()
        return _local_speech_ids[obj_id]

    def _extract_agent_new_state(state_event: object) -> str:
        new_state = getattr(state_event, "new_state", None)
        if new_state is not None:
            return str(new_state).strip().lower()

        current_state = getattr(state_event, "state", None)
        if current_state is not None:
            return str(current_state).strip().lower()

        state_text = str(state_event)
        lowered = state_text.lower()
        if "new_state='listening'" in lowered or 'new_state="listening"' in lowered:
            return "listening"

        return lowered.strip()

    def _extract_user_new_state(state_event: object) -> str:
        new_state = getattr(state_event, "new_state", None)
        if new_state is not None:
            return str(new_state).strip().lower()

        current_state = getattr(state_event, "state", None)
        if current_state is not None:
            return str(current_state).strip().lower()

        state_text = str(state_event)
        lowered = state_text.lower()
        if "new_state='speaking'" in lowered or 'new_state="speaking"' in lowered:
            return "speaking"
        if "new_state='listening'" in lowered or 'new_state="listening"' in lowered:
            return "listening"

        return lowered.strip()

    def _clear_active_handles(reason: str) -> None:
        cleared_count = len(active_speech_handles)
        if cleared_count or speech_start_times or suppressed_speech_ids:
            logger.warning(
                "Clearing stale active speech handles: reason=%s cleared_count=%s suppressed_count=%s",
                reason,
                cleared_count,
                len(suppressed_speech_ids),
            )
        active_speech_handles.clear()
        speech_start_times.clear()
        suppressed_speech_ids.clear()


    @session.on("speech_created")
    def _on_speech_created(event_or_handle: object) -> None:
        nonlocal payload_debug_logged

        if not payload_debug_logged:
            payload_debug_logged = True
            attrs = ("id", "speech_id", "handle", "speech", "speech_handle", "interrupted", "add_done_callback", "interrupt", "wait_for_playout", "cancel", "stop", "close")
            attr_presence = {name: hasattr(event_or_handle, name) for name in attrs}
            logger.info("speech_created payload debug: type=%s attrs=%s suppress_window_seconds=%s", type(event_or_handle).__name__, attr_presence, overlap_suppress_window_seconds)

        resolved_handle = _resolve_speech_handle(event_or_handle)
        speech_id = _speech_id(resolved_handle)
        now = time.monotonic()
        speech_start_times[speech_id] = now

        suppressed = False
        suppression_attempted = False
        suppression_result = "not_needed"
        if active_speech_handles:
            active_ids_before_new = list(active_speech_handles.keys())
            current_speech = getattr(session, "current_speech", None)
            current_speech_id = _speech_id(current_speech) if current_speech is not None else "none"
            current_speech_type = type(current_speech).__name__ if current_speech is not None else "none"
            now_for_diag = time.monotonic()
            user_state_age = (now_for_diag - latest_user_state_timestamp) if latest_user_state_timestamp else -1.0
            agent_state_age = (now_for_diag - latest_agent_state_timestamp) if latest_agent_state_timestamp else -1.0
            latest_user_state_normalized = str(latest_user_state).strip().lower()
            latest_agent_state_normalized = str(latest_agent_state).strip().lower()
            logger.info(
                "Assistant overlap diagnostic: new_speech_id=%s active_speech_ids_before_new=%s session_current_speech_id=%s session_current_speech_type=%s latest_user_state=%s latest_agent_state=%s seconds_since_user_state_change=%.3f seconds_since_agent_state_change=%.3f user_state_is_speaking=%s agent_state_is_thinking=%s agent_state_is_speaking=%s agent_state_is_listening=%s",
                speech_id,
                active_ids_before_new,
                current_speech_id,
                current_speech_type,
                latest_user_state,
                latest_agent_state,
                user_state_age,
                agent_state_age,
                latest_user_state_normalized == "speaking",
                latest_agent_state_normalized == "thinking",
                latest_agent_state_normalized == "speaking",
                latest_agent_state_normalized == "listening",
            )
            previous_id = None
            previous_started_at = None
            for active_id in active_speech_handles.keys():
                if active_id == speech_id:
                    continue
                started_at = speech_start_times.get(active_id)
                if started_at is None:
                    continue
                if previous_started_at is None or started_at > previous_started_at:
                    previous_started_at = started_at
                    previous_id = active_id

            if previous_id is not None and previous_started_at is not None:
                age_seconds = now - previous_started_at
                if age_seconds <= overlap_suppress_window_seconds:
                    method_used = None
                    suppression_attempted = True
                    for method_name in ("cancel", "stop", "close"):
                        method = getattr(resolved_handle, method_name, None)
                        if callable(method):
                            try:
                                method()
                                method_used = method_name
                                break
                            except Exception as e:
                                logger.warning(
                                    "Duplicate speech suppression method failed: speech_id=%s method=%s err=%s",
                                    speech_id,
                                    method_name,
                                    e,
                                )

                    if method_used is not None:
                        suppressed = True
                        suppression_result = f"suppressed:{method_used}"
                        suppressed_speech_ids.add(speech_id)
                        logger.warning(
                            "Suppressing duplicate assistant speech overlap: kept_speech_id=%s suppressed_speech_id=%s active_count=%s method=%s",
                            previous_id,
                            speech_id,
                            len(active_speech_handles),
                            method_used,
                        )
                    else:
                        logger.warning(
                            "Possible assistant speech overlap not safely suppressed: previous_speech_id=%s new_speech_id=%s reason=%s active_count=%s",
                            previous_id,
                            speech_id,
                            "no_safe_suppression_method",
                            len(active_speech_handles),
                        )
                        suppression_result = "not_safely_suppressed:no_safe_suppression_method"
                else:
                    logger.warning(
                        "Possible assistant speech overlap not safely suppressed: previous_speech_id=%s new_speech_id=%s reason=%s active_count=%s",
                        previous_id,
                        speech_id,
                        "previous_speech_outside_suppress_window",
                        len(active_speech_handles),
                    )
                    suppression_result = "not_suppressed:previous_speech_outside_suppress_window"
            else:
                logger.warning(
                    "Possible assistant speech overlap not safely suppressed: previous_speech_id=%s new_speech_id=%s reason=%s active_count=%s",
                    "unknown",
                    speech_id,
                    "missing_previous_speech_start_time",
                    len(active_speech_handles),
                    )

        if active_speech_handles:
            logger.info(
                "Assistant overlap suppression outcome: new_speech_id=%s suppression_attempted=%s result=%s",
                speech_id,
                suppression_attempted,
                suppression_result,
            )

        if not suppressed:
            active_speech_handles[speech_id] = resolved_handle
            logger.info("Assistant speech started: speech_id=%s active_count=%s", speech_id, len(active_speech_handles))

        add_done_callback = getattr(resolved_handle, "add_done_callback", None)
        if callable(add_done_callback):
            def _on_done(done_event_or_handle: object) -> None:
                done_resolved_handle = _resolve_speech_handle(done_event_or_handle)
                done_id = _speech_id(done_resolved_handle)
                active_speech_handles.pop(done_id, None)
                speech_start_times.pop(done_id, None)
                was_suppressed = done_id in suppressed_speech_ids
                suppressed_speech_ids.discard(done_id)
                interrupted = _safe_attr(done_resolved_handle, "interrupted", "unknown")
                logger.info("Assistant speech finished: speech_id=%s interrupted=%s active_count=%s was_suppressed=%s", done_id, interrupted, len(active_speech_handles), was_suppressed)

            add_done_callback(_on_done)
        else:
            logger.warning("Resolved speech handle does not support add_done_callback")

    @session.on("overlapping_speech")
    def _on_overlapping_speech(*_: object) -> None:
        logger.warning("Session reported overlapping_speech event while assistant_active_count=%s", len(active_speech_handles))

    @session.on("agent_state_changed")
    def _on_agent_state_changed(state: object) -> None:
        nonlocal latest_agent_state, latest_agent_state_timestamp
        extracted_new_state = _extract_agent_new_state(state)
        latest_agent_state = extracted_new_state
        latest_agent_state_timestamp = time.monotonic()
        current_speech = getattr(session, "current_speech", None)
        has_current_speech = current_speech is not None
        logger.info(
            "Agent state changed: state=%s extracted_new_state=%s has_current_speech=%s assistant_active_count=%s",
            state,
            extracted_new_state,
            has_current_speech,
            len(active_speech_handles),
        )
        if extracted_new_state == "listening" and not has_current_speech:
            _clear_active_handles("agent_returned_to_listening")

    @session.on("user_state_changed")
    def _on_user_state_changed(state: object) -> None:
        nonlocal latest_user_state, latest_user_state_timestamp
        latest_user_state = _extract_user_new_state(state)
        latest_user_state_timestamp = time.monotonic()
        logger.info("User state changed: state=%s assistant_active_count=%s", state, len(active_speech_handles))

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
        safe_summary = _safe_error_summary(error)
        logger.error("Session error event summary: %s", safe_summary)

        searchable_safe_text = " ".join(str(v).lower() for v in safe_summary.values())
        if "tts" in searchable_safe_text:
            _clear_active_handles("tts_error")

    @session.on("close")
    def _on_close() -> None:
        _clear_active_handles("session_close")
        logger.info("Session close event: active_speeches_remaining=%s", len(active_speech_handles))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")



AI_COUSTICS_ENABLED = env_bool("AI_COUSTICS_ENABLED", True)


def _resolve_ai_coustics_model(model_name: str):
    normalized = (model_name or "").strip().upper()
    model_map = {
        "QUAIL_VF_S": ai_coustics.EnhancerModel.QUAIL_VF_S,
        "QUAIL_VF_L": ai_coustics.EnhancerModel.QUAIL_VF_L,
        "QUAIL_L": ai_coustics.EnhancerModel.QUAIL_L,
    }
    if normalized in model_map:
        return model_map[normalized], normalized

    logger.warning("Unknown AI_COUSTICS_MODEL provided: %s. Falling back to QUAIL_VF_S", model_name)
    return ai_coustics.EnhancerModel.QUAIL_VF_S, "QUAIL_VF_S"


def build_room_options() -> room_io.RoomOptions | None:
    if not AI_COUSTICS_ENABLED:
        logger.info("ai-coustics disabled: AI_COUSTICS_ENABLED=false")
        return None

    selected_model, selected_model_name = _resolve_ai_coustics_model(os.getenv("AI_COUSTICS_MODEL", "QUAIL_VF_S"))
    raw_level = os.getenv("AI_COUSTICS_ENHANCEMENT_LEVEL", "0.8")
    try:
        enhancement_level = float(raw_level)
    except ValueError:
        logger.warning("Invalid AI_COUSTICS_ENHANCEMENT_LEVEL=%s. Falling back to 0.8", raw_level)
        enhancement_level = 0.8
    enhancement_level = max(0.0, min(1.0, enhancement_level))

    logger.info(
        "ai-coustics configuration: enabled=%s model=%s enhancement_level=%s",
        True,
        selected_model_name,
        enhancement_level,
    )

    try:
        if hasattr(ai_coustics, "ModelParameters"):
            model_parameters = ai_coustics.ModelParameters(enhancement_level=enhancement_level)
            enhancer = ai_coustics.audio_enhancement(model=selected_model, model_parameters=model_parameters)
        else:
            logger.warning("ai-coustics ModelParameters unavailable; using model-only audio enhancement")
            enhancer = ai_coustics.audio_enhancement(model=selected_model)
    except TypeError as e:
        logger.warning("ai-coustics model_parameters unsupported in installed package, using model-only enhancement: %s", e)
        enhancer = ai_coustics.audio_enhancement(model=selected_model)

    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=enhancer,
        )
    )

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



def build_vad():
    if VAD_PROVIDER == "ai_coustics":
        logger.info("Using ai-coustics VAD provider")
        return ai_coustics.VAD()

    if VAD_PROVIDER == "silero":
        logger.info("Using Silero VAD provider")
        return silero.VAD.load()

    logger.warning("Unknown VAD_PROVIDER=%s. Falling back to ai-coustics VAD provider", VAD_PROVIDER)
    return ai_coustics.VAD()


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


def _attach_optional_interruption_diagnostics(session: AgentSession) -> None:
    def _register(event_name: str, handler):
        try:
            session.on(event_name)(handler)
            logger.info("Registered optional interruption diagnostic handler: event=%s", event_name)
            return True
        except Exception as e:
            logger.info("Optional interruption diagnostic event unavailable: event=%s reason=%s", event_name, e)
            return False

    def _on_user_interruption_detected(event: object) -> None:
        logger.info(
            "User interruption diagnostic event: event_type=%s interrupted=%s",
            type(event).__name__,
            _safe_attr(event, "interrupted", "unknown"),
        )

    _register("user_interruption_detected", _on_user_interruption_detected)



async def entrypoint(ctx: JobContext):
    llm = openai.LLM.with_openrouter(model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"))
    # TODO: Re-enable Tavily using LiveKit's supported function-tool pattern.
    logger.warning("Skipping Tavily tools for MVP voice path")

    logger.info("Startup provider config: STT_PROVIDER=%s VAD_PROVIDER(raw)=%s LIVEKIT_TURN_DETECTION_MODE(raw)=%s", STT_PROVIDER, os.getenv("VAD_PROVIDER", "ai_coustics"), os.getenv("LIVEKIT_TURN_DETECTION_MODE", "stt"))

    interruption_options: InterruptionOptions = {
        "enabled": True,
        "min_words": 1,
        "min_duration": 0.6,
        "resume_false_interruption": env_bool("LIVEKIT_RESUME_FALSE_INTERRUPTION", False),
        "false_interruption_timeout": 1.8,
    }

    session_kwargs: dict[str, Any] = {
        "stt": build_stt(),
        "llm": llm,
        "tts": build_tts(),
        "vad": build_vad(),
    }

    resolved_turn_detection_mode = "multilingual"
    if STT_PROVIDER == "deepgram_flux":
        if LIVEKIT_TURN_DETECTION_MODE == "stt":
            session_kwargs["turn_handling"] = TurnHandlingOptions(
                turn_detection="stt",
                interruption=interruption_options,
            )
            resolved_turn_detection_mode = "stt"
            logger.info("Using Flux STT-based turn detection")
        elif LIVEKIT_TURN_DETECTION_MODE == "vad":
            try:
                session_kwargs["turn_handling"] = TurnHandlingOptions(
                    turn_detection="vad",
                    interruption=interruption_options,
                )
                resolved_turn_detection_mode = "vad"
                logger.info("Using Flux VAD-based turn detection")
            except Exception as e:
                logger.warning("VAD turn_detection mode unavailable in this LiveKit version, falling back to stt: %s", e)
                session_kwargs["turn_handling"] = TurnHandlingOptions(
                    turn_detection="stt",
                    interruption=interruption_options,
                )
                resolved_turn_detection_mode = "stt"
        elif LIVEKIT_TURN_DETECTION_MODE == "default":
            resolved_turn_detection_mode = "default"
            logger.info("Using LiveKit default turn handling for Deepgram Flux")
        else:
            logger.warning(
                "Unknown LIVEKIT_TURN_DETECTION_MODE=%s. Falling back to stt.",
                LIVEKIT_TURN_DETECTION_MODE,
            )
            session_kwargs["turn_handling"] = TurnHandlingOptions(
                turn_detection="stt",
                interruption=interruption_options,
            )
            resolved_turn_detection_mode = "stt"

        logger.info(
            "Using Flux turn handling config: turn_detection_mode=%s interruption=%s resume_false_interruption=%s",
            resolved_turn_detection_mode,
            interruption_options,
            interruption_options.get("resume_false_interruption"),
        )
    else:
        session_kwargs["turn_detection"] = MultilingualModel()
        logger.info("Using non-Flux turn handling config: turn_detection=%s", "multilingual")

    session = AgentSession(**session_kwargs)
    resolved_vad = session_kwargs.get("vad")
    logger.info("Resolved VAD provider: provider=%s vad_type=%s", VAD_PROVIDER, type(resolved_vad).__name__)
    logger.info("Resolved turn detection mode: %s", resolved_turn_detection_mode)

    attach_session_diagnostics(session)
    _attach_optional_interruption_diagnostics(session)

    room_options = build_room_options()
    if room_options is not None:
        logger.info("Starting session with ai-coustics room_options attached")
        await session.start(room=ctx.room, agent=LucyAgent(), room_options=room_options)
    else:
        logger.info("Starting session without ai-coustics room_options")
        await session.start(room=ctx.room, agent=LucyAgent())
    logger.info("About to generate greeting reply")
    greeting_handle = await session.generate_reply(
        instructions="Greet the user in one short casual sentence as Crash. Say: Yo. What’s going on?",
        allow_interruptions=False,
    )
    logger.info(
        "Greeting generate_reply completed: handle_type=%s handle_id=%s allow_interruptions=%s interrupted=%s",
        type(greeting_handle).__name__,
        _safe_attr(greeting_handle, "id"),
        _safe_attr(greeting_handle, "allow_interruptions"),
        _safe_attr(greeting_handle, "interrupted"),
    )

    wait_for_playout = getattr(greeting_handle, "wait_for_playout", None)
    if callable(wait_for_playout):
        try:
            await asyncio.wait_for(wait_for_playout(), timeout=8.0)
            logger.info("Greeting playout completed")
        except TimeoutError:
            logger.warning("Greeting playout wait timed out")
        except Exception as e:
            logger.warning("Greeting playout wait failed: %s", e)
    else:
        logger.warning("Greeting handle does not support wait_for_playout")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

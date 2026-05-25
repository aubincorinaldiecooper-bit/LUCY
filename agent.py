import os
import asyncio
import inspect
import logging
import time
import hashlib
import re
import contextvars
from typing import Any, AsyncIterable

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from livekit.agents import Agent, AgentSession, InterruptionOptions, JobContext, TurnHandlingOptions, WorkerOptions, cli, room_io
from livekit.plugins import ai_coustics, deepgram, hume, mistralai, openai, silero
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
STT_PROVIDER = os.getenv("STT_PROVIDER", "mistral").strip().lower()
VAD_PROVIDER = os.getenv("VAD_PROVIDER", "ai_coustics").strip().lower()
LIVEKIT_TURN_DETECTION_MODE = os.getenv("LIVEKIT_TURN_DETECTION_MODE", "vad").strip().lower()

_speech_counter = 0
_hume_tts_request_counter = 0
_latest_normalized_text_hash = "n/a"
_latest_agent_state_for_hume = "unknown"
_latest_active_assistant_count_for_hume = 0
_latest_current_speech_id_for_hume = "n/a"
_normalized_text_hash_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("normalized_text_hash", default="n/a")


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


def _extract_text_for_debug(obj: object) -> str:
    if obj is None:
        return ""
    item = getattr(obj, "item", None)
    target = item if item is not None else obj

    text_content = getattr(target, "text_content", None)
    if isinstance(text_content, str):
        return text_content

    text = getattr(target, "text", None)
    if isinstance(text, str):
        return text

    content = getattr(target, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            else:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    parts.append(part_text)
        return " ".join(p for p in parts if p).strip()

    return str(content or "")


def _safe_nested_error_details(error: object) -> dict[str, str]:
    details: dict[str, str] = {}
    current = error
    for idx in range(3):
        if current is None:
            break
        prefix = f"err{idx}"
        details[f"{prefix}_type"] = type(current).__name__
        for field in ("message", "detail", "status_code", "code", "retryable", "details"):
            value = getattr(current, field, None)
            if value is not None:
                details[f"{prefix}_{field}"] = _redact_sensitive_text(value)
        body = getattr(current, "body", None)
        if body is not None:
            body_message = getattr(body, "message", None)
            body_code = getattr(body, "code", None)
            if body_message is not None:
                details[f"{prefix}_body_message"] = _redact_sensitive_text(body_message)
            if body_code is not None:
                details[f"{prefix}_body_code"] = _redact_sensitive_text(body_code)
        current = getattr(current, "error", None)
    return details


def _sanitize_spoken_laughter(text: str) -> str:
    if not text:
        return text
    pattern = r"\b(lol|lmao|rofl|haha|hehe)\b"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


def attach_session_diagnostics(session: AgentSession) -> None:
    global _latest_agent_state_for_hume, _latest_active_assistant_count_for_hume, _latest_current_speech_id_for_hume
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
    speech_created_at: dict[str, float] = {}
    assistant_speech_started_at: dict[str, float] = {}
    agent_speaking_at: dict[str, float] = {}
    agent_listening_at: dict[str, float] = {}
    assistant_speech_finished_at: dict[str, float] = {}
    pending_user_handoff_speech_id: str | None = None
    stt_partial_count = 0
    stt_final_count = 0
    last_stt_any_at = 0.0
    last_stt_final_at = 0.0
    last_stt_preview = ""
    last_stt_final_preview = ""
    last_user_speaking_at = 0.0
    last_user_listening_at = 0.0
    hume_request_count_at_speech_start: dict[str, int] = {}
    hume_request_count_at_speech_finish: dict[str, int] = {}

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
        speech_created_at[speech_id] = now
        assistant_speech_started_at[speech_id] = now

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
            hume_request_count_at_speech_start[speech_id] = _hume_tts_request_counter
            _latest_current_speech_id_for_hume = speech_id
            _latest_active_assistant_count_for_hume = len(active_speech_handles)
            logger.info("Assistant speech started: speech_id=%s active_count=%s", speech_id, len(active_speech_handles))
            logger.info(
                "Assistant speech Hume request baseline: speech_id=%s hume_request_count_at_start=%s active_count=%s",
                speech_id,
                hume_request_count_at_speech_start[speech_id],
                len(active_speech_handles),
            )

        add_done_callback = getattr(resolved_handle, "add_done_callback", None)
        if callable(add_done_callback):
            def _on_done(done_event_or_handle: object) -> None:
                nonlocal pending_user_handoff_speech_id
                done_resolved_handle = _resolve_speech_handle(done_event_or_handle)
                done_id = _speech_id(done_resolved_handle)
                active_speech_handles.pop(done_id, None)
                _latest_active_assistant_count_for_hume = len(active_speech_handles)
                finished_at = time.monotonic()
                started_at = speech_start_times.pop(done_id, None)
                was_suppressed = done_id in suppressed_speech_ids
                suppressed_speech_ids.discard(done_id)
                interrupted = _safe_attr(done_resolved_handle, "interrupted", "unknown")
                speaking_at = agent_speaking_at.get(done_id)
                listening_at = agent_listening_at.get(done_id)
                speech_duration_seconds = (finished_at - started_at) if started_at is not None else -1.0
                agent_speaking_to_finished_seconds = (finished_at - speaking_at) if speaking_at is not None else -1.0
                finish_to_agent_listening_seconds = (
                    listening_at - finished_at if listening_at is not None else -1.0
                )
                assistant_speech_finished_at[done_id] = finished_at
                hume_request_count_at_speech_finish[done_id] = _hume_tts_request_counter
                pending_user_handoff_speech_id = done_id
                logger.info("Assistant speech finished: speech_id=%s interrupted=%s active_count=%s was_suppressed=%s", done_id, interrupted, len(active_speech_handles), was_suppressed)
                start_count = hume_request_count_at_speech_start.get(done_id, -1)
                finish_count = hume_request_count_at_speech_finish.get(done_id, -1)
                during_count = finish_count - start_count if start_count >= 0 and finish_count >= 0 else -1
                logger.info(
                    "Assistant speech Hume request summary: speech_id=%s interrupted=%s was_suppressed=%s hume_request_count_at_start=%s hume_request_count_at_finish=%s hume_requests_during_speech=%s speech_duration_seconds=%.3f latest_agent_state=%s latest_user_state=%s",
                    done_id,
                    interrupted,
                    was_suppressed,
                    start_count,
                    finish_count,
                    during_count,
                    speech_duration_seconds,
                    latest_agent_state,
                    latest_user_state,
                )
                logger.info(
                    "Assistant handoff timing: speech_id=%s speech_duration_seconds=%.3f agent_speaking_to_finished_seconds=%.3f finish_to_agent_listening_seconds=%.3f interrupted=%s was_suppressed=%s active_count=%s",
                    done_id,
                    speech_duration_seconds,
                    agent_speaking_to_finished_seconds,
                    finish_to_agent_listening_seconds,
                    interrupted,
                    was_suppressed,
                    len(active_speech_handles),
                )

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
        _latest_agent_state_for_hume = extracted_new_state
        latest_agent_state_timestamp = time.monotonic()
        current_speech = getattr(session, "current_speech", None)
        has_current_speech = current_speech is not None
        current_speech_id = _speech_id(current_speech) if has_current_speech else None
        if current_speech_id is not None:
            _latest_current_speech_id_for_hume = current_speech_id
        _latest_active_assistant_count_for_hume = len(active_speech_handles)
        if extracted_new_state == "speaking" and current_speech_id is not None:
            agent_speaking_at[current_speech_id] = latest_agent_state_timestamp
        old_state = getattr(state, "old_state", None)
        old_state_normalized = str(old_state).strip().lower() if old_state is not None else ""
        if (
            old_state_normalized == "speaking"
            and extracted_new_state == "listening"
            and current_speech_id is not None
        ):
            agent_listening_at[current_speech_id] = latest_agent_state_timestamp
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
        nonlocal latest_user_state, latest_user_state_timestamp, pending_user_handoff_speech_id, last_user_speaking_at, last_user_listening_at
        latest_user_state = _extract_user_new_state(state)
        latest_user_state_timestamp = time.monotonic()
        if latest_user_state == "speaking":
            last_user_speaking_at = latest_user_state_timestamp
        if latest_user_state == "listening":
            last_user_listening_at = latest_user_state_timestamp
        logger.info("User state changed: state=%s assistant_active_count=%s", state, len(active_speech_handles))
        if latest_user_state == "speaking" and pending_user_handoff_speech_id is not None:
            previous_speech_id = pending_user_handoff_speech_id
            finished_at = assistant_speech_finished_at.get(previous_speech_id)
            if finished_at is not None and latest_user_state_timestamp >= finished_at:
                logger.info(
                    "Assistant handoff to user: previous_speech_id=%s finish_to_user_speaking_seconds=%.3f latest_agent_state=%s active_count=%s",
                    previous_speech_id,
                    latest_user_state_timestamp - finished_at,
                    latest_agent_state,
                    len(active_speech_handles),
                )
                pending_user_handoff_speech_id = None

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(*_: object) -> None:
        logger.warning("Agent false interruption detected")

    @session.on("conversation_item_added")
    def _on_conversation_item_added(item: object) -> None:
        event_item = getattr(item, "item", None)
        target = event_item if event_item is not None else item
        role = _safe_attr(target, "role")
        interrupted = _safe_attr(target, "interrupted")
        if PIPELINE_TEXT_DEBUG:
            text_str = _extract_text_for_debug(target)
            logger.info(
                "Conversation item added: role=%s interrupted=%s text_length=%s preview=%s",
                role,
                interrupted,
                len(text_str),
                _redact_sensitive_text(text_str)[:200],
            )
            return
        logger.info("Conversation item added: role=%s interrupted=%s", role, interrupted)

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event: object) -> None:
        nonlocal stt_partial_count, stt_final_count, last_stt_any_at, last_stt_final_at, last_stt_preview, last_stt_final_preview
        final = getattr(event, "final", getattr(event, "is_final", "n/a"))
        language = _safe_attr(event, "language", "n/a")
        speaker_id = getattr(event, "speaker_id", None)
        transcript = getattr(event, "transcript", None)
        if transcript is None:
            transcript = getattr(event, "text", "")
        transcript_str = str(transcript or "")
        last_stt_any_at = time.monotonic()
        last_stt_preview = _redact_sensitive_text(transcript_str)[:200]
        is_final = str(final).strip().lower() in {"true", "1", "yes"}
        if is_final:
            stt_final_count += 1
            last_stt_final_at = last_stt_any_at
            last_stt_final_preview = last_stt_preview
        else:
            stt_partial_count += 1
        if not PIPELINE_TEXT_DEBUG:
            return
        logger.info(
            "STT debug: final=%s language=%s speaker_id_present=%s transcript_length=%s preview=%s",
            final,
            language,
            speaker_id is not None,
            len(transcript_str),
            last_stt_preview,
        )

    @session.on("error")
    def _on_error(error: object) -> None:
        safe_summary = _safe_error_summary(error)
        logger.error("Session error event summary: %s", safe_summary)

        searchable_safe_text = " ".join(str(v).lower() for v in safe_summary.values())
        if "tts" in searchable_safe_text:
            _clear_active_handles("tts_error")
        if MISTRAL_STT_DIAGNOSTICS:
            stt_error_markers = ("stt", "mistral", "grpc", "http/2", "unavailable", "failed parsing", "3803")
            if any(marker in searchable_safe_text for marker in stt_error_markers):
                now = time.monotonic()
                stt_since_any = now - last_stt_any_at if last_stt_any_at > 0 else -1.0
                stt_since_final = now - last_stt_final_at if last_stt_final_at > 0 else -1.0
                logger.warning(
                    "Mistral STT diagnostic snapshot: error_chain=%s stt_partial_count=%s stt_final_count=%s seconds_since_last_stt=%s seconds_since_last_final_stt=%s last_stt_preview=%s last_final_stt_preview=%s latest_user_state=%s latest_agent_state=%s active_assistant_speech_count=%s STT_PROVIDER=%s MISTRAL_STT_MODEL=%s MISTRAL_TARGET_STREAMING_DELAY_MS=%s VAD_PROVIDER=%s AI_COUSTICS_ENABLED=%s AI_COUSTICS_MODEL=%s AI_COUSTICS_LEVEL=%s endpointing_mode=%s endpointing_min_delay=%s endpointing_max_delay=%s",
                    _safe_nested_error_details(error),
                    stt_partial_count,
                    stt_final_count,
                    stt_since_any,
                    stt_since_final,
                    last_stt_preview,
                    last_stt_final_preview,
                    latest_user_state,
                    latest_agent_state,
                    len(active_speech_handles),
                    STT_PROVIDER,
                    os.getenv("MISTRAL_STT_MODEL", "voxtral-mini-transcribe-realtime-2602"),
                    os.getenv("MISTRAL_TARGET_STREAMING_DELAY_MS", "160"),
                    VAD_PROVIDER,
                    AI_COUSTICS_ENABLED,
                    os.getenv("AI_COUSTICS_MODEL", "QUAIL_L"),
                    os.getenv("AI_COUSTICS_LEVEL", "0.7"),
                    os.getenv("LIVEKIT_ENDPOINTING_MODE", "fixed"),
                    os.getenv("LIVEKIT_ENDPOINTING_MIN_DELAY", "0.4"),
                    os.getenv("LIVEKIT_ENDPOINTING_MAX_DELAY", "1.5"),
                )

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
SPOKEN_TEXT_NORMALIZATION = env_bool("SPOKEN_TEXT_NORMALIZATION", False)
TTS_TEXT_DEBUG = env_bool("TTS_TEXT_DEBUG", False)
PIPELINE_TEXT_DEBUG = env_bool("PIPELINE_TEXT_DEBUG", False)
MISTRAL_STT_DIAGNOSTICS = env_bool("MISTRAL_STT_DIAGNOSTICS", True)
HUME_FULL_UTTERANCE_TTS = env_bool("HUME_FULL_UTTERANCE_TTS", False)
LIVEKIT_TTS_SOURCE_INSPECTION = env_bool("LIVEKIT_TTS_SOURCE_INSPECTION", False)


def _safe_source_excerpt(obj: object, max_chars: int) -> str:
    try:
        src = inspect.getsource(obj)
    except Exception as e:
        return f"<unavailable: {_redact_sensitive_text(e)}>"
    sanitized = src.replace("\r", "")
    return sanitized[:max_chars]


def _log_livekit_tts_source_inspection() -> None:
    if not LIVEKIT_TTS_SOURCE_INSPECTION:
        return
    try:
        import livekit.agents as lk_agents  # type: ignore
        import livekit.plugins.hume as lk_hume  # type: ignore
        from livekit.agents import Agent as LKAgent  # type: ignore
    except Exception as e:
        logger.warning("LiveKit TTS source inspection unavailable: reason=%s", _redact_sensitive_text(e))
        return

    inspect_terms = ("sentence", "tokenizer", "tokenize", "segment", "chunk", "synthesize", "stream", "capabilities")
    agents_version = getattr(lk_agents, "__version__", "unknown")
    agents_path = getattr(lk_agents, "__file__", "unknown")
    hume_module_path = getattr(lk_hume, "__file__", "unknown")
    default_tts_node = getattr(LKAgent.default, "tts_node", None)
    tts_node_signature = str(inspect.signature(default_tts_node)) if callable(default_tts_node) else "unavailable"
    tts_node_file = inspect.getsourcefile(default_tts_node) if callable(default_tts_node) else "unavailable"
    tts_node_src = _safe_source_excerpt(default_tts_node, 5000) if callable(default_tts_node) else "<unavailable>"

    hume_tts_cls = getattr(lk_hume, "TTS", None)
    hume_init_sig = "unavailable"
    hume_src = "<unavailable>"
    hume_synthesize_sig = "unavailable"
    hume_synthesize_src = "<unavailable>"
    hume_stream_sig = "unavailable"
    hume_stream_src = "<unavailable>"
    hume_caps = "unavailable"
    if hume_tts_cls is not None:
        hume_src = _safe_source_excerpt(hume_tts_cls, 8000)
        init_fn = getattr(hume_tts_cls, "__init__", None)
        if callable(init_fn):
            hume_init_sig = str(inspect.signature(init_fn))
        synth_fn = getattr(hume_tts_cls, "synthesize", None)
        if callable(synth_fn):
            hume_synthesize_sig = str(inspect.signature(synth_fn))
            hume_synthesize_src = _safe_source_excerpt(synth_fn, 4000)
        stream_fn = getattr(hume_tts_cls, "stream", None)
        if callable(stream_fn):
            hume_stream_sig = str(inspect.signature(stream_fn))
            hume_stream_src = _safe_source_excerpt(stream_fn, 4000)
        caps = getattr(hume_tts_cls, "capabilities", None)
        if caps is not None:
            hume_caps = _redact_sensitive_text(caps)

    combined = "\n".join([tts_node_src, hume_src, hume_synthesize_src, hume_stream_src]).lower()
    term_presence = {term: (term in combined) for term in inspect_terms}
    logger.info(
        "LiveKit TTS source inspection summary: agents_version=%s agents_module=%s agent_default_tts_node_file=%s agent_default_tts_node_signature=%s hume_module=%s hume_tts_init_signature=%s hume_tts_capabilities=%s term_presence=%s",
        agents_version,
        agents_path,
        tts_node_file or "unknown",
        tts_node_signature,
        hume_module_path,
        hume_init_sig,
        hume_caps,
        term_presence,
    )
    logger.info("LiveKit Agent.default.tts_node source excerpt (max_5000): %s", tts_node_src)
    logger.info("LiveKit Hume TTS class source excerpt (max_8000): %s", hume_src)
    logger.info("LiveKit Hume TTS.synthesize signature=%s source_excerpt(max_4000): %s", hume_synthesize_sig, hume_synthesize_src)
    logger.info("LiveKit Hume TTS.stream signature=%s source_excerpt(max_4000): %s", hume_stream_sig, hume_stream_src)


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

        hume_speed = float(os.getenv("HUME_SPEED", "0.9"))
        hume_tts_debug_http = env_bool("HUME_TTS_DEBUG_HTTP", False)
        hume_description = os.getenv("HUME_DESCRIPTION") or (
            "A warm, calm, natural companion voice. Speak with relaxed pacing, soft sentence endings, "
            "and brief natural pauses between thoughts. Do not sound rushed, clipped, or abrupt at the end of sentences."
        )
        hume_description_present = bool(hume_description)
        hume_description_length = len(hume_description)
        hume_trailing_silence = float(os.getenv("HUME_TRAILING_SILENCE", "0.25"))
        hume_model_version = _resolve_hume_model_version()
        hume_tts_signature = inspect.signature(hume.TTS)
        hume_tts_kwargs: dict[str, Any] = {
            "voice": voice,
            "model_version": hume_model_version,
            "speed": hume_speed,
            "instant_mode": instant_mode,
        }
        description_applied = hume_model_version != "2"
        if description_applied:
            hume_tts_kwargs["description"] = hume_description
        else:
            logger.info(
                "Hume description skipped: model_version=2 reason=octave2_unsupported description_present=%s description_length=%s",
                hume_description_present,
                hume_description_length,
            )
        trailing_silence_applied = False
        trailing_silence_supported = "trailing_silence" in hume_tts_signature.parameters
        if trailing_silence_supported:
            hume_tts_kwargs["trailing_silence"] = hume_trailing_silence
            trailing_silence_applied = True
            logger.info(
                "trailing_silence_supported=true trailing_silence_applied=true value=%s",
                hume_trailing_silence,
            )
        else:
            logger.info("trailing_silence_supported=false trailing_silence_applied=false")

        logger.info(
            "Hume TTS config: speed=%s description_present=%s trailing_silence_value=%s trailing_silence_supported=%s trailing_silence_applied=%s",
            hume_speed,
            hume_description_present,
            hume_trailing_silence if trailing_silence_applied else "n/a",
            trailing_silence_supported,
            trailing_silence_applied,
        )
        voice_kind = "None"
        voice_provider_effective = "n/a"
        if hume_voice_id:
            voice_kind = "VoiceById"
        elif hume_voice_name:
            voice_kind = "VoiceByName"
            voice_provider_effective = hume_voice_provider or "hume"
        logger.info(
            "Hume TTS effective config: model_version=%s voice_kind=%s voice_present=%s voice_provider=%s instant_mode=%s speed=%s description_present=%s description_applied=%s description_length=%s trailing_silence_supported=%s trailing_silence_applied=%s trailing_silence_value=%s debug_http=%s",
            hume_tts_kwargs.get("model_version"),
            voice_kind,
            bool(voice),
            voice_provider_effective,
            instant_mode,
            hume_speed,
            hume_description_present,
            description_applied,
            hume_description_length,
            trailing_silence_supported,
            trailing_silence_applied,
            hume_trailing_silence if trailing_silence_applied else "n/a",
            hume_tts_debug_http,
        )

        if hume_tts_debug_http:
            trace_config = aiohttp.TraceConfig()

            async def _log_hume_tts_error_detail(
                response: aiohttp.ClientResponse | None = None,
                response_url: Any = None,
                body_read_error: object | None = None,
            ) -> None:
                status = getattr(response, "status", "n/a")
                reason = _redact_sensitive_text(getattr(response, "reason", "n/a"))
                path = _redact_sensitive_text(getattr(response_url, "path", "n/a"))
                if body_read_error is not None:
                    logger.warning(
                        "Hume TTS HTTP error detail unavailable: status=%s reason=%s body_read_error=%s",
                        status,
                        reason,
                        _redact_sensitive_text(body_read_error),
                    )
                    return
                if response is None:
                    return
                body_text = await response.text()
                redacted_body = _redact_sensitive_text(body_text)[:2000]
                logger.warning(
                    "Hume TTS HTTP error detail: status=%s reason=%s path=%s body=%s",
                    status,
                    reason,
                    path,
                    redacted_body,
                )

            async def _on_request_start(session, trace_config_ctx, params):
                global _hume_tts_request_counter
                _hume_tts_request_counter += 1
                ctx_hash = _normalized_text_hash_ctx.get()
                logger.info(
                    "Hume TTS HTTP request: hume_request_index=%s method=%s path=%s latest_agent_state=%s active_assistant_count=%s current_speech_id=%s instant_mode=%s speed=%s trailing_silence=%s normalized_text_hash=%s debug=true",
                    _hume_tts_request_counter,
                    params.method,
                    _redact_sensitive_text(params.url.path),
                    _latest_agent_state_for_hume,
                    _latest_active_assistant_count_for_hume,
                    _latest_current_speech_id_for_hume,
                    instant_mode,
                    hume_speed,
                    hume_trailing_silence,
                    ctx_hash if ctx_hash != "n/a" else _latest_normalized_text_hash,
                )

            async def _on_request_end(session, trace_config_ctx, params):
                if params.response.status >= 400:
                    try:
                        await _log_hume_tts_error_detail(response=params.response, response_url=params.url)
                    except Exception as body_error:
                        await _log_hume_tts_error_detail(
                            response=params.response,
                            response_url=params.url,
                            body_read_error=body_error,
                        )

            async def _on_request_exception(session, trace_config_ctx, params):
                response = getattr(params, "response", None)
                if response is not None and getattr(response, "status", 0) >= 400:
                    try:
                        await _log_hume_tts_error_detail(response=response, response_url=params.url)
                    except Exception as body_error:
                        await _log_hume_tts_error_detail(
                            response=response,
                            response_url=params.url,
                            body_read_error=body_error,
                        )

            trace_config.on_request_start.append(_on_request_start)
            trace_config.on_request_end.append(_on_request_end)
            trace_config.on_request_exception.append(_on_request_exception)
            hume_tts_kwargs["http_session"] = aiohttp.ClientSession(trace_configs=[trace_config])

        return hume.TTS(**hume_tts_kwargs)

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

    def _normalize_spoken_text(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return normalized
        if "```" in normalized:
            return normalized
        if normalized[-1] not in {".", "?", "!", "…"}:
            return normalized + "."
        return normalized

    def tts_node(self, text: AsyncIterable[str], model_settings):
        global _latest_normalized_text_hash
        if not SPOKEN_TEXT_NORMALIZATION:
            logger.info("Spoken text normalization enabled=false")
            if not TTS_TEXT_DEBUG:
                return Agent.default.tts_node(self, text, model_settings)

            async def _passthrough_debug_stream() -> AsyncIterable[str]:
                chunks: list[str] = []
                count = 0
                async for chunk in text:
                    count += 1
                    chunks.append(chunk)
                    yield chunk
                raw_text = "".join(chunks)
                preview = _redact_sensitive_text(raw_text)[:200]
                logger.info(
                    "TTS text debug: raw_chunk_count=%s raw_total_length=%s raw_preview=%s final_preview=%s",
                    count,
                    len(raw_text),
                    preview,
                    preview,
                )

            return Agent.default.tts_node(self, _passthrough_debug_stream(), model_settings)

        logger.info("Spoken text normalization enabled=true mode=buffered_full_segment")

        async def _normalized_text_stream() -> AsyncIterable[str]:
            global _latest_normalized_text_hash
            chunks: list[str] = []
            chunk_count = 0
            async for chunk in text:
                chunk_count += 1
                chunks.append(chunk)
            raw_text = "".join(chunks)
            sanitized = _sanitize_spoken_laughter(raw_text)
            normalized = self._normalize_spoken_text(sanitized)
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12] if normalized else "empty"
            _latest_normalized_text_hash = normalized_hash
            _normalized_text_hash_ctx.set(normalized_hash)
            sentence_end_count = sum(normalized.count(mark) for mark in (".", "?", "!", "…"))
            newline_count = normalized.count("\n")
            logger.info(
                "TTS normalized yield diagnostics: tts_normalized_yield_count=%s raw_chunk_count=%s raw_total_length=%s normalized_text_length=%s normalized_text_preview=%s normalized_text_hash=%s sentence_end_count=%s newline_count=%s SPOKEN_TEXT_NORMALIZATION=%s TTS_PROVIDER=%s HUME_INSTANT_MODE=%s HUME_SPEED=%s HUME_TRAILING_SILENCE=%s",
                1,
                chunk_count,
                len(raw_text),
                len(normalized),
                _redact_sensitive_text(normalized)[:200],
                normalized_hash,
                sentence_end_count,
                newline_count,
                SPOKEN_TEXT_NORMALIZATION,
                TTS_PROVIDER,
                env_bool("HUME_INSTANT_MODE", True),
                os.getenv("HUME_SPEED", "0.9"),
                os.getenv("HUME_TRAILING_SILENCE", "0.25"),
            )
            if TTS_TEXT_DEBUG:
                logger.info(
                    "TTS text debug: raw_chunk_count=%s raw_total_length=%s raw_preview=%s final_preview=%s",
                    chunk_count,
                    len(raw_text),
                    _redact_sensitive_text(raw_text)[:200],
                    _redact_sensitive_text(normalized)[:200],
                )
            if normalized:
                yield normalized

        if TTS_PROVIDER == "hume":
            full_utterance_requested = bool(HUME_FULL_UTTERANCE_TTS)
            full_utterance_supported = False
            full_utterance_used = False
            fallback_reason = "not_requested"
            if full_utterance_requested:
                fallback_reason = "no_verified_supported_livekit_or_hume_one_shot_path_in_current_environment"
            logger.info(
                "Hume full-utterance mode: full_utterance_requested=%s full_utterance_supported=%s full_utterance_used=%s path=%s fallback_reason=%s",
                full_utterance_requested,
                full_utterance_supported,
                full_utterance_used,
                "default_livekit_tts_node_single_segment",
                fallback_reason,
            )
            if full_utterance_requested:
                logger.warning(
                    "Hume full-utterance mode fallback: installed LiveKit source could not be inspected in this build environment; no documented runtime sentence-tokenizer override detected here, using default LiveKit tts_node with single normalized segment."
                )

        return Agent.default.tts_node(self, _normalized_text_stream(), model_settings)

    def llm_node(self, chat_ctx, tools, model_settings):
        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)

        async def _llm_stream():
            assistant_fragments: list[str] = []
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                if PIPELINE_TEXT_DEBUG:
                    text_delta: object = None
                    delta = getattr(chunk, "delta", None)
                    if delta is not None:
                        delta_content = getattr(delta, "content", None)
                        if isinstance(delta_content, str):
                            text_delta = delta_content
                        elif isinstance(delta_content, list):
                            parts: list[str] = []
                            for part in delta_content:
                                if isinstance(part, str):
                                    parts.append(part)
                                else:
                                    part_text = getattr(part, "text", None)
                                    if isinstance(part_text, str):
                                        parts.append(part_text)
                            if parts:
                                text_delta = "".join(parts)
                        if text_delta is None:
                            delta_text = getattr(delta, "text", None)
                            if isinstance(delta_text, str):
                                text_delta = delta_text
                    if text_delta is None:
                        chunk_text = getattr(chunk, "text", None)
                        if isinstance(chunk_text, str):
                            text_delta = chunk_text
                    if text_delta is None:
                        chunk_content = getattr(chunk, "content", None)
                        if isinstance(chunk_content, str):
                            text_delta = chunk_content
                    if text_delta is None and isinstance(chunk, str):
                        text_delta = chunk
                    if isinstance(text_delta, str):
                        assistant_fragments.append(text_delta)
                yield chunk
            if PIPELINE_TEXT_DEBUG:
                combined = "".join(assistant_fragments)
                logger.info(
                    "LLM output debug: chunk_count=%s text_length=%s preview=%s",
                    chunk_count,
                    len(combined),
                    _redact_sensitive_text(combined)[:200],
                )

        return _llm_stream()

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        if PIPELINE_TEXT_DEBUG:
            msg_str = _extract_text_for_debug(new_message)
            messages = getattr(turn_ctx, "messages", None)
            message_count = "n/a"
            if messages is not None:
                try:
                    message_count = len(messages)
                except Exception:
                    message_count = "n/a"
            logger.info(
                "User turn debug: new_message_length=%s preview=%s turn_ctx_message_count=%s",
                len(msg_str),
                _redact_sensitive_text(msg_str)[:200],
                message_count,
            )



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
        mistral_stt_model = os.getenv("MISTRAL_STT_MODEL", "voxtral-mini-transcribe-realtime-2602")
        mistral_target_streaming_delay_ms = int(os.getenv("MISTRAL_TARGET_STREAMING_DELAY_MS", "160"))
        logger.info("MISTRAL_STT_MODEL=%s", mistral_stt_model)
        mistral_stt_signature = inspect.signature(mistralai.STT)
        if "target_streaming_delay_ms" in mistral_stt_signature.parameters:
            logger.info("MISTRAL_TARGET_STREAMING_DELAY_MS applied=true value=%s", mistral_target_streaming_delay_ms)
            return mistralai.STT(
                model=mistral_stt_model,
                target_streaming_delay_ms=mistral_target_streaming_delay_ms,
            )
        logger.info("MISTRAL_TARGET_STREAMING_DELAY_MS applied=false reason=unsupported_constructor")
        return mistralai.STT(model=mistral_stt_model)

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
    _log_livekit_tts_source_inspection()
    llm = openai.LLM.with_openrouter(model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"))
    # TODO: Re-enable Tavily using LiveKit's supported function-tool pattern.
    logger.warning("Skipping Tavily tools for MVP voice path")

    livekit_turn_detection_mode_present = "LIVEKIT_TURN_DETECTION_MODE" in os.environ
    livekit_turn_detection_mode_raw = os.getenv("LIVEKIT_TURN_DETECTION_MODE")
    livekit_turn_detection_mode = (
        livekit_turn_detection_mode_raw.strip().lower()
        if isinstance(livekit_turn_detection_mode_raw, str)
        else "vad"
    )

    if not livekit_turn_detection_mode_present:
        logger.info("LIVEKIT_TURN_DETECTION_MODE missing; defaulting to vad")

    if livekit_turn_detection_mode in {"vad", "stt", "default"}:
        resolved_livekit_turn_detection_mode = livekit_turn_detection_mode
    else:
        logger.warning(
            "Unknown LIVEKIT_TURN_DETECTION_MODE=%s. Falling back to vad.",
            livekit_turn_detection_mode,
        )
        resolved_livekit_turn_detection_mode = "vad"

    logger.info(
        "Startup provider config: STT_PROVIDER=%s VAD_PROVIDER(raw)=%s",
        STT_PROVIDER,
        os.getenv("VAD_PROVIDER", "ai_coustics"),
    )
    logger.info(
        "LIVEKIT_TURN_DETECTION_MODE present=%s raw=%s resolved=%s",
        livekit_turn_detection_mode_present,
        livekit_turn_detection_mode_raw if livekit_turn_detection_mode_present else "missing",
        resolved_livekit_turn_detection_mode,
    )
    logger.info(
        "Railway context: service=%s environment=%s deployment=%s",
        os.getenv("RAILWAY_SERVICE_NAME", "n/a"),
        os.getenv("RAILWAY_ENVIRONMENT_NAME", "n/a"),
        os.getenv("RAILWAY_DEPLOYMENT_ID", "n/a"),
    )

    interruption_options: InterruptionOptions = {
        "enabled": env_bool("LIVEKIT_INTERRUPTION_ENABLED", True),
        "min_words": int(os.getenv("LIVEKIT_INTERRUPTION_MIN_WORDS", "1")),
        "min_duration": float(os.getenv("LIVEKIT_INTERRUPTION_MIN_DURATION", "0.6")),
        "resume_false_interruption": env_bool("LIVEKIT_RESUME_FALSE_INTERRUPTION", False),
        "false_interruption_timeout": float(os.getenv("LIVEKIT_FALSE_INTERRUPTION_TIMEOUT", "1.8")),
    }
    logger.info("Resolved interruption config: %s", interruption_options)
    endpointing_mode = os.getenv("LIVEKIT_ENDPOINTING_MODE", "fixed")
    endpointing_min_delay = float(os.getenv("LIVEKIT_ENDPOINTING_MIN_DELAY", "0.4"))
    endpointing_max_delay = float(os.getenv("LIVEKIT_ENDPOINTING_MAX_DELAY", "1.5"))
    logger.info(
        "Endpointing config: mode=%s min_delay=%s max_delay=%s",
        endpointing_mode,
        endpointing_min_delay,
        endpointing_max_delay,
    )
    mistral_stt_model = os.getenv("MISTRAL_STT_MODEL", "voxtral-mini-transcribe-realtime-2602")
    mistral_target_streaming_delay_ms = int(os.getenv("MISTRAL_TARGET_STREAMING_DELAY_MS", "160"))
    mistral_target_delay_supported = "target_streaming_delay_ms" in inspect.signature(mistralai.STT).parameters
    logger.info("Startup Mistral STT config: model=%s target_streaming_delay_ms=%s applied=%s", mistral_stt_model, mistral_target_streaming_delay_ms, mistral_target_delay_supported)
    logger.info("MISTRAL_STT_DIAGNOSTICS enabled=%s", MISTRAL_STT_DIAGNOSTICS)
    logger.info("HUME_FULL_UTTERANCE_TTS enabled=%s", HUME_FULL_UTTERANCE_TTS)
    try:
        from livekit.agents import Agent as _AgentInspect

        tts_node_src = inspect.getsourcefile(_AgentInspect)
        logger.info(
            "LiveKit source inspection summary: Agent.default.tts_node source_file=%s note=runtime may sentence-split non-streaming TTS implementations",
            tts_node_src or "unknown",
        )
    except Exception as e:
        logger.warning(
            "LiveKit source inspection unavailable in current build environment; keeping default behavior and diagnostics only: reason=%s",
            _redact_sensitive_text(e),
        )
    if os.getenv("GRPC_TRACE") is not None or os.getenv("GRPC_VERBOSITY") is not None:
        logger.warning("Low-level gRPC tracing env vars are enabled and may be noisy")

    session_kwargs: dict[str, Any] = {
        "stt": build_stt(),
        "llm": llm,
        "tts": build_tts(),
        "vad": build_vad(),
    }

    resolved_turn_detection_mode = "unknown"
    if STT_PROVIDER == "deepgram_flux":
        if resolved_livekit_turn_detection_mode == "stt":
            session_kwargs["turn_handling"] = TurnHandlingOptions(
                turn_detection="stt",
                interruption=interruption_options,
            )
            resolved_turn_detection_mode = "stt"
            logger.info("Using Flux STT-based turn detection")
        elif resolved_livekit_turn_detection_mode == "vad":
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
        elif resolved_livekit_turn_detection_mode == "default":
            resolved_turn_detection_mode = "default"
            logger.info("Using LiveKit default turn handling for Deepgram Flux")

        logger.info(
            "Using Flux turn handling config: turn_detection_mode=%s interruption=%s resume_false_interruption=%s",
            resolved_turn_detection_mode,
            interruption_options,
            interruption_options.get("resume_false_interruption"),
        )
    elif STT_PROVIDER == "mistral":
        session_kwargs["turn_handling"] = TurnHandlingOptions(
            turn_detection="vad",
            interruption=interruption_options,
            endpointing={
                "mode": endpointing_mode,
                "min_delay": endpointing_min_delay,
                "max_delay": endpointing_max_delay,
            },
        )
        resolved_turn_detection_mode = "vad"
        logger.info("Using Mistral VAD-only turn handling")
    else:
        session_kwargs["turn_handling"] = TurnHandlingOptions(
            turn_detection="vad",
            interruption=interruption_options,
        )
        resolved_turn_detection_mode = "vad"
        logger.info("Using non-Flux VAD turn handling")

    session = AgentSession(**session_kwargs)
    resolved_stt = session_kwargs.get("stt")
    logger.info("Resolved STT type: %s", type(resolved_stt).__name__)
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
    logger.info("About to say fixed greeting")
    greeting_handle = await session.say(
        "Yo. What’s going on?",
        allow_interruptions=False,
    )
    logger.info(
        "Fixed greeting say completed: handle_type=%s handle_id=%s interrupted=%s",
        type(greeting_handle).__name__,
        _safe_attr(greeting_handle, "id"),
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

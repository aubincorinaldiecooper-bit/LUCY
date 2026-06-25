from __future__ import annotations

import os
import asyncio
import inspect
import json
import logging
import time
import hashlib
import re
import contextvars
import struct
import wave
import io
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, AsyncIterable

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from livekit.agents import Agent, AgentSession, InterruptionOptions, JobContext, StopResponse, TurnHandlingOptions, WorkerOptions, cli, function_tool, room_io, tts
try:  # livekit-agents >= 1.6.1 exposes the audio end-of-turn detector here
    from livekit.agents import inference as _lk_inference
except Exception:  # pragma: no cover - older SDKs
    _lk_inference = None
from livekit import api, rtc
from livekit.plugins import ai_coustics, deepgram, hume, mistralai, openai, silero
from internet_search import (
    SEARCH_DISABLED_MESSAGE,
    SEARCH_TOOL_DESCRIPTION,
    build_effective_search_query,
    format_search_results_for_voice,
    internet_search,
    search_disabled_reason,
    search_enabled,
    search_failure_response,
    search_max_results,
    search_provider,
    search_timeout_seconds,
)
from audiointeraction_shadow import AudioInteractionShadow, audiointeraction_mode, build_shadow_from_env
from inworld_voice_profile import InworldVoiceProfileShadow, build_inworld_shadow_from_env
from hume_evi_bridge import run_hume_evi_bridge, voice_engine
from interaction_state import (
    ASSISTANT_SPEAKING,
    ASSISTANT_THINKING,
    LISTENING,
    TOOL_CALL_PENDING,
    TURN_KIND_ACTION,
    USER_INTERRUPTING,
    USER_SPEAKING,
    USER_TURN_CANDIDATE,
    USER_SPEECH_OBSERVATION_WINDOW_SECONDS as INTERACTION_USER_SPEECH_OBSERVATION_WINDOW_SECONDS,
    AudioEnvironmentDecision,
    InteractionStateMachine,
    build_audio_environment_decision,
    classify_turn_kind,
)
from memory_layer import (
    EMOTIONAL_PATTERN_PREFIX,
    MemoryLayer,
    emotional_pattern_preload_note,
    identity_from_metadata,
    memory_enabled,
    partition_emotional_patterns,
)
from runtime_context import (
    RuntimeContext,
    answer_datetime_intent,
    current_datetime_snapshot,
    detect_datetime_intent,
    runtime_context_from_metadata,
)
from transcript_context import (
    ADDITIVE_FAMILY,
    ContextDecision,
    ContextResolution,
    TranscriptContext,
    build_context_decision,
    call_transcript_context_llm,
    classify_tool_revalidation_relationship,
    clean_transcript,
    decide_tool_result_resume,
    detect_transcript_context,
    interpret_transcript_context,
    normal_context_classifier_timeout,
    query_materially_changed,
    require_context_resolution_for_tool_authority,
    resolve_transcript_context,
    tool_revalidation_additive_min_dependency,
    tool_revalidation_context_classifier_max_wait_ms,
    transcript_context_debug,
    transcript_context_layer_enabled,
    transcript_context_llm_enabled,
    transcript_context_llm_model,
    transcript_context_llm_timeout_ms,
)
from voice_interruption import classify_tail_outcome, is_audible_cutoff
from omnivoice_tts import OmniVoiceConfig, OmniVoiceTTS, find_omnivoice_tts
from omnivoice_voice_pool import get_session_selector
from omnivoice_language import detect_language_request, language_name


load_dotenv()
logger = logging.getLogger(__name__)

OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"

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
- Write replies as spoken lines with emotion carried by wording itself; use natural pauses, simple phrasing, and short reactions.
- Do not use bracketed stage directions like “[warmly]” or “[laughs]”.

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


RUNTIME_CAPABILITY_CONTRACT = """Runtime capability contract:
- General honesty: Never claim a capability succeeded unless the runtime/tool actually performed it. If a capability is unavailable in this session, say so plainly and offer the closest available alternative. Do not over-apologize or over-explain. Keep capability answers short, natural, and conversational.
- Human language: Arche can speak and understand human languages. Arche is currently speaking English. If asked about languages, answer directly and truthfully. If the user asks for a known language, respond briefly in that language and explain the meaning in English if useful. If a requested language or dialect is ambiguous, clarify naturally; for Jamaican, you can say you can try a little Jamaican Patois. Never say you do not speak human languages, only speak companion language, cannot speak any language, or that Sri Lankan is a language.
- Voice switching: You can change the words, style, or language you say, but you cannot claim to switch the actual TTS voice in this session unless a voice-switching runtime feature succeeds. If asked for another voice, say: “I can speak differently, but I can’t switch the actual voice inside this session yet.”
- Internet search: You can use internet search only through the configured Exa search tool. For current or external facts, say a short bridge like “I’ll look it up — give me a second,” then use search. After useful results, say “This is what I found…” If search fails or is stale or unclear, say you could not get a clear result and ask what to search for. Do not guess current facts from model memory.
- Date and time: Answer date/time questions only from runtime context or the date-time guard. Do not use model memory. Do not use internet search for basic date/time.
- Email: Discuss sending email only if the AgentMail voice path/tool is available. Never claim an email was sent unless the send tool succeeds. If the recipient is missing, ask what email to use. Before sending, confirm the recipient. After success only, say it was sent. On failure, say you could not send it and ask whether to try another email.
- Documents and files: Do not claim you created a Word doc, PDF, or file unless that runtime capability exists and succeeds. If file creation is not wired in this voice session, say: “I can draft the content, but I can’t create the file from this voice session yet.”
- Math and calculations: You can do basic arithmetic, counting, comparisons, estimation, and simple mathematical reasoning conversationally. Answer simple calculations directly. For ambiguous numeric fragments, do not assume the operation; ask what the numbers refer to. Do not pretend a calculator/tool was used unless one actually was.
- Timers and reminders: Do not claim to set timers, alarms, reminders, calendar events, or scheduled tasks unless a real timer/reminder/calendar tool exists in this runtime and succeeds. If unavailable, say: “I can’t set an actual timer from here yet, but I can stay with you while you time it.” Never say “timer set” unless the runtime actually set one.
- Counting aloud: You can count out loud if asked. For short counts, count directly. For long counts like 100, ask if the user wants the full count out loud.
- Memory and privacy: If asked what you remember, say conversations are remembered for about a day, then cleared, and conversations are kept private. Do not invent long-term memory unless it is implemented.
- Tool-action honesty: Never say an action is complete unless the tool/runtime confirms success. Search requires Exa results; email requires AgentMail send success; document creation requires file generation success; timer/reminder scheduling requires an actual scheduling tool success; voice switching requires actual TTS voice switch success. For unavailable capabilities, say so plainly and offer the closest available alternative. Keep responses short and conversational.
- Corrections: When corrected by the user, acknowledge it directly. Do not drift into philosophical, semantic, or taxonomic debates unless asked. If the user says “You’re speaking one right now,” say: “You’re right — English is a human language.”
""".strip()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
if "SYSTEM_PROMPT" in os.environ:
    logger.warning("SYSTEM_PROMPT env override detected; code-level prompt edits may not affect production unless Railway SYSTEM_PROMPT is updated; mirror the runtime capability contract in Railway SYSTEM_PROMPT for consistency")

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "hume").strip().lower()
# When the primary provider fails/times out/returns invalid audio, LiveKit's
# tts.FallbackAdapter moves on to this provider so the session never goes silent.
# Currently wired for the omnivoice -> hume path; empty/"none" disables fallback.
TTS_FALLBACK_PROVIDER = (os.getenv("TTS_FALLBACK_PROVIDER", "hume") or "").strip().lower()
STT_PROVIDER = os.getenv("STT_PROVIDER", "mistral").strip().lower()
# Active/session language Arche is configured to operate in. Logged at turn commit
# so a transcript-language candidate can be compared against the intended language.
SESSION_LANGUAGE = (os.getenv("SESSION_LANGUAGE") or os.getenv("DEEPGRAM_STT_LANGUAGE") or "en").strip() or "en"
# Mutable active language for the current session: starts at SESSION_LANGUAGE and
# changes when the user asks Arche to switch (e.g. "speak French"). Drives the
# OmniVoice synthesis language and an LLM directive to reply in that language.
_active_session_language = SESSION_LANGUAGE
# The OmniVoiceTTS for this session (bare or inside the FallbackAdapter), so a
# runtime voice/language switch can reach it. None when OmniVoice isn't active.
_session_omnivoice_tts: "OmniVoiceTTS | None" = None
VAD_PROVIDER = os.getenv("VAD_PROVIDER", "ai_coustics").strip().lower()
LIVEKIT_TURN_DETECTION_MODE = os.getenv("LIVEKIT_TURN_DETECTION_MODE", "vad").strip().lower()
# LiveKit audio end-of-turn detector (livekit-agents >= 1.6.1: inference.TurnDetector).
# Feature-detected at build time; if unavailable we keep the existing vad/stt mode.
LIVEKIT_TURN_DETECTOR_ENABLED = os.getenv("LIVEKIT_TURN_DETECTOR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
LIVEKIT_TURN_DETECTOR_VERSION = (os.getenv("LIVEKIT_TURN_DETECTOR_VERSION", "auto") or "auto").strip().lower()
LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD = (os.getenv("LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD") or "").strip()
LIVEKIT_ENDPOINTING_DYNAMIC_ENABLED = os.getenv("LIVEKIT_ENDPOINTING_DYNAMIC_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
# Inbound audio enhancement (ai-coustics / QUAIL) applied before STT/VAD/turn detection.
LIVEKIT_AUDIO_ENHANCEMENT_ENABLED = os.getenv("LIVEKIT_AUDIO_ENHANCEMENT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
LIVEKIT_AUDIO_ENHANCEMENT_PROVIDER = (os.getenv("LIVEKIT_AUDIO_ENHANCEMENT_PROVIDER", "ai_coustics") or "ai_coustics").strip().lower()
LIVEKIT_AUDIO_ENHANCEMENT_MODEL = (os.getenv("LIVEKIT_AUDIO_ENHANCEMENT_MODEL", "QUAIL_VF_S") or "QUAIL_VF_S").strip()

_speech_counter = 0
_hume_tts_request_counter = 0
_latest_normalized_text_hash = "n/a"
_latest_agent_state_for_hume = "unknown"
_latest_active_assistant_count_for_hume = 0
_latest_current_speech_id_for_hume = "n/a"
_normalized_text_hash_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("normalized_text_hash", default="n/a")
_direct_hume_request_counter = 0
_hume_recent_request_keys: dict[str, int] = {}
_hume_recent_request_order: list[str] = []
_HUME_RECENT_REQUEST_KEY_LIMIT = 80
_hume_speech_audio_coverages: dict[str, "HumeSpeechAudioCoverage"] = {}
_last_llm_start_at = 0.0
_last_llm_first_token_at: float | None = None
_last_llm_complete_at = 0.0
_last_turn_committed_at = 0.0
_last_tts_request_start_at = 0.0
_last_tts_first_audio_at: float | None = None
_last_tts_text_length = 0
_last_tts_sentence_end_count = 0
_last_tts_path: str | None = None
_last_hume_model_version = "n/a"
_last_hume_description_applied = "n/a"
_last_hume_voice_present = False
_last_hume_voice_kind = "n/a"
_last_hume_instant_mode = "n/a"
_last_hume_speed = "n/a"
_last_hume_trailing_silence = "n/a"
_last_hume_style_context_applied = "n/a"
_last_hume_tts_build_started_at = 0.0
_last_hume_tts_build_completed_at = 0.0
_last_hume_tts_debug_http = False
# In-memory pre-rendered greeting WAV bytes (populated once at worker startup by
# _prerender_greeting). None until rendered or if pre-render is disabled/failed.
_prerendered_greeting_wav: bytes | None = None
# Guards against concurrent jobs each firing a duplicate render before the first
# one populates the buffer.
_prerender_greeting_in_flight = False
_silero_initialized = False
_last_llm_stream_status = "n/a"
_last_llm_timeout_stage = "n/a"
_last_llm_fallback_response_used = False
_pending_llm_fallback_text: str | None = None
_last_llm_completed_text = ""
_last_llm_completed_text_hash = "empty"
_last_llm_completed_at = 0.0
_last_user_message_text = ""
# Immutable per-final-STT candidates so every consumer of a turn (commit, debug,
# turn policy, pruning, generation) references the same transcript segment instead
# of a lagging global. Detects transcript drift under barge-in / rapid follow-up.
_stt_candidates: list[dict] = []
_stt_segment_counter = 0
_current_candidate_id: str | None = None
_last_tts_node_entered_at = 0.0
_last_tts_received_text_hash = "empty"
_last_hume_request_start_at = 0.0
_last_tts_completed_at = 0.0
_last_tts_raw_chunk_count = 0
_last_tts_normalized_yield_count = 0
_last_tts_first_input_at: float | None = None
_latest_user_state_for_greeting = "unknown"
_latest_user_state_changed_at = 0.0
_latest_user_speaking_at = 0.0
_latest_stt_partial_at = 0.0
_latest_stt_partial_text_hash = "empty"
_latest_stt_final_at = 0.0
# Latest STT-reported language candidate + confidence for the most recent final
# transcript, carried into the turn-commit log so a committed turn can be proven
# to have been (e.g.) Portuguese. "n/a" when the STT engine does not report it.
_latest_stt_language = "n/a"
_latest_stt_language_confidence = "n/a"
_search_tool_called = False
_search_in_progress = False
_search_started_at = 0.0
_search_completed_at = 0.0
_search_failed = False
_search_pre_ack_spoken = False
_search_specific_response_produced = False
_search_result_handoff_spoken = False
_last_search_tool_output = ""
_last_search_tool_output_hash = "empty"
_last_generic_llm_fallback_used = False
# Original query that initiated the in-flight search, captured at search start so
# a barge-in turn can be classified against it by the tool-result composer.
_pending_search_query = ""
_pending_search_result_available = False
# Set False when a turn commits with no valid speech/candidate owner; gates
# canonical writes and tool authority for that turn (runtime enforcement).
_current_turn_owner_valid = True
_current_turn_id = 0
_turn_id_counter = 0
_search_turn_id = 0
_llm_turn_id = 0
_current_turn_transcript_intent = "unknown"
_current_turn_search_allowed = False
_current_turn_search_allowed_reason = "not_evaluated"
_current_turn_policy_classification = "UNKNOWN"
_current_turn_policy_decision = "COMMIT_NOW"
_current_turn_audio_unclear = False
# Handoff guard: tracks a user barge-in that begins while the assistant is still
# thinking (no assistant audio yet), so a stale pending reply can be suppressed
# before it reaches TTS. Only a sustained "real" utterance latches confirmed.
_barge_in_during_thinking_turn_id = 0
_barge_in_started_at = 0.0
_barge_in_confirmed_real = False
# Context-policy state: a thin in-memory conversation ledger (not a memory system)
# used only to inject recent VISIBLE turns and to carry a contextual intent
# forward 1-2 turns. Suppressed/zero-audio/interrupted assistant turns are kept
# but marked non-canonical so they never pollute future prompt context.
_conversation_ledger: list[dict] = []
_prior_context_decision: ContextDecision | None = None
_current_context_dependency = "none"
# Lightweight rolling audio-environment signals (monotonic timestamps), pruned to
# a short window. Used only for AudioEnvironmentDecision / audio-status answers.
_recent_user_speech_start_times: list[float] = []
_recent_turn_candidate_times: list[float] = []
_AUDIO_ENV_WINDOW_SECONDS = 12.0
_interaction_state = InteractionStateMachine()
_active_memory_layer: MemoryLayer | None = None
_audiointeraction_shadow: AudioInteractionShadow | None = None
_inworld_voice_profile_shadow: InworldVoiceProfileShadow | None = None
_held_turn_fragment_text = ""
_held_turn_fragment_created_at = 0.0
_held_turn_fragment_classification = ""
_held_turn_fragment_incomplete = False
_calibration_session_id = "unknown"
_pending_calibration_moment: dict[str, Any] | None = None
_calibration_moments: list[dict[str, Any]] = []
_last_calibration_question_turn_id = -1000
CALIBRATION_MOMENTS_PATH = (os.getenv("CALIBRATION_MOMENTS_PATH") or "logs/calibration_moments.jsonl").strip()


def _next_turn_id() -> int:
    global _turn_id_counter
    _turn_id_counter += 1
    return _turn_id_counter


def _reset_search_state_for_turn(turn_id: int | None = None) -> None:
    global _search_tool_called, _search_in_progress, _search_started_at, _search_completed_at, _search_failed, _search_pre_ack_spoken, _search_specific_response_produced, _search_result_handoff_spoken, _last_search_tool_output, _last_search_tool_output_hash, _last_generic_llm_fallback_used, _search_turn_id
    _search_tool_called = False
    _search_in_progress = False
    _search_started_at = 0.0
    _search_completed_at = 0.0
    _search_failed = False
    _search_pre_ack_spoken = False
    _search_specific_response_produced = False
    _search_result_handoff_spoken = False
    _last_search_tool_output = ""
    _last_search_tool_output_hash = "empty"
    _last_generic_llm_fallback_used = False
    _search_turn_id = turn_id or _current_turn_id


def _mark_search_wait_started(pre_ack_spoken: bool = False, turn_id: int | None = None, query: str = "") -> None:
    global _search_tool_called, _search_in_progress, _search_started_at, _search_completed_at, _search_failed, _search_pre_ack_spoken, _search_specific_response_produced, _search_result_handoff_spoken, _last_search_tool_output, _last_search_tool_output_hash, _search_turn_id, _pending_search_query, _pending_search_result_available
    _search_turn_id = turn_id or _current_turn_id
    _search_tool_called = True
    _search_in_progress = True
    _search_started_at = time.monotonic()
    _search_completed_at = 0.0
    _search_failed = False
    _search_pre_ack_spoken = pre_ack_spoken
    _search_specific_response_produced = False
    _search_result_handoff_spoken = False
    _last_search_tool_output = ""
    _last_search_tool_output_hash = "empty"
    # Snapshot the query that initiated this search so a later barge-in can be
    # classified against it; the result is not yet available.
    _pending_search_query = query or _last_user_message_text
    _pending_search_result_available = False
    _interaction_state.on_tool_call_started("internet_search")


def _mark_search_wait_completed(failed: bool, output: str, result_handoff_spoken: bool = False, turn_id: int | None = None) -> bool:
    global _search_in_progress, _search_completed_at, _search_failed, _search_specific_response_produced, _search_result_handoff_spoken, _last_search_tool_output, _last_search_tool_output_hash, _pending_search_result_available
    completion_turn_id = turn_id or _search_turn_id
    if completion_turn_id != _current_turn_id or _search_turn_id != _current_turn_id:
        logger.warning(
            "search_wait_completed stale_search_result_ignored=true search_turn_id=%s current_turn_id=%s completion_turn_id=%s",
            _search_turn_id,
            _current_turn_id,
            completion_turn_id,
        )
        return False
    _search_in_progress = False
    _search_completed_at = time.monotonic()
    _search_failed = failed
    _search_specific_response_produced = bool(output.strip())
    _search_result_handoff_spoken = result_handoff_spoken
    _last_search_tool_output = output
    _last_search_tool_output_hash = _text_hash(output)
    _pending_search_result_available = True
    _interaction_state.on_tool_call_finished("internet_search")
    return True


def _search_turn_matches_current() -> bool:
    return _search_turn_id == _current_turn_id


def _search_active_for_current_turn() -> bool:
    return _search_in_progress and _search_turn_matches_current()


def _search_specific_response_for_current_turn() -> bool:
    return _search_tool_called and _search_specific_response_produced and _search_turn_matches_current()


def _tool_revalidation_class(classification: str | None, intent: str | None, context_dependency: str | None) -> str:
    """Map turn signals to the tool-result revalidation taxonomy.

    Conservative by design: only a high-dependency continuation is treated as
    additive context that lets an in-flight result keep its right to speak.
    Everything else defaults to a class that withholds that authority.
    """
    cls = (classification or "").strip().upper()
    if cls == "META_COMPLAINT":
        return "meta_complaint"
    if (context_dependency or "").strip().lower() == "high":
        return "additive_context"
    return "unrelated"


def _search_result_authority_gate(output: str, turn_id: int) -> str:
    """Withhold a search result that lost its right to speak after a barge-in.

    The user may speak during a search (TOOL_CALL_PENDING) to add context; that
    pauses the result's authority until the new utterance is classified. If
    revalidation revoked authority, the stale result must not auto-speak. If
    revalidation has not completed yet (rare race), allow but flag it.
    """
    if _interaction_state.tool_result_pending_revalidation:
        logger.warning(
            "search_result_authority_pending_revalidation=true stale_tool_result_blocked=false turn_id=%s note=allowing_result_revalidation_incomplete",
            turn_id,
        )
        return output
    if not _interaction_state.tool_result_speak_authority:
        logger.info(
            "search_result_authority_revoked=true stale_tool_result_blocked=true tool_result_resume_decision=%s tool_revalidation_class=%s turn_id=%s tool_result_paused_reason=%s note=stale_result_must_not_speak",
            _interaction_state.tool_result_resume_decision or "n/a",
            _interaction_state.tool_result_relationship or "n/a",
            turn_id,
            _interaction_state.tool_result_paused_reason or "n/a",
        )
        return (
            "Search result withheld: the user spoke again during the lookup and the new "
            "message was not additive context. Do not speak this result; address the user's "
            "newest message first and re-run the search only if it is still what they want."
        )
    return output


def _inject_tool_result_composition_note(
    turn_ctx: object, original_query: str, pending_result: str, newer_utterance: str
) -> bool:
    """Compose: keep the pending result but fold in the newer user utterance.

    Injects a developer note so Lucy answers using the in-flight result together
    with the user's mid-search addition, instead of discarding either.
    """
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        return False
    note = (
        "Mid-search addition (compose):\n"
        "The user added context while a lookup was in progress. Incorporate it into the answer; "
        "do not start over or ignore the earlier request.\n"
        f"- Original request: {_redact_sensitive_text(original_query)[:240]}\n"
        f"- User's addition: {_redact_sensitive_text(newer_utterance)[:240]}\n"
        f"- Lookup result so far: {_redact_sensitive_text(pending_result)[:600]}\n"
        "Answer the combined request naturally in one turn."
    )
    try:
        add_message(role="developer", content=note)
        return True
    except Exception as exc:
        logger.warning(
            "tool_result_composition_note_injection_failed=true error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )
        return False


async def _revalidate_pending_tool_result(
    turn_ctx: object,
    *,
    newer_utterance: str,
    classification: str | None,
    recent_turns: list[str] | None,
) -> str:
    """High-risk tool/search handoff composer.

    Classifies how the barge-in utterance relates to the pending query, resolves
    context with the longer tool-revalidation wait, and decides compose / rerun /
    withhold / discard / defer / clarify. Returns the resume decision.
    """
    original_query = _pending_search_query
    result_available = _pending_search_result_available

    resolution_info = await resolve_transcript_context(
        newer_utterance,
        recent_turns=recent_turns,
        runtime_context=None,
        path="tool_revalidation",
        high_risk=True,
    )
    resolution = resolution_info.resolution
    timed_out = resolution_info.timed_out
    classifier_path = resolution_info.classifier_path
    require_resolution = require_context_resolution_for_tool_authority()
    additive_min = tool_revalidation_additive_min_dependency()
    # Timeout + its winning env source are reported by the resolver so the log is
    # unambiguous about which knob applied.
    revalidation_timeout_ms = resolution_info.timeout_ms
    timeout_source = resolution_info.timeout_source
    context_classifier_path = resolution_info.path
    layer_enabled = transcript_context_layer_enabled()
    llm_enabled = transcript_context_llm_enabled()
    classifier_model = transcript_context_llm_model() if llm_enabled else "deterministic"

    relationship, dependency, rel_conf = classify_tool_revalidation_relationship(
        original_query=original_query,
        newer_utterance=newer_utterance,
        base_intent=_current_turn_transcript_intent,
        classification=classification,
    )
    materially_changed = query_materially_changed(original_query, newer_utterance)
    decision, additive_allowed = decide_tool_result_resume(
        relationship=relationship,
        resolution=resolution,
        dependency_level=dependency,
        additive_min_dependency=additive_min,
        require_resolution=require_resolution,
        materially_changed=materially_changed,
        result_available=result_available,
    )

    handoff_allowed = decision == "compose"
    if handoff_allowed:
        blocked_reason = "none"
    elif resolution == "unresolved" and require_resolution:
        blocked_reason = "timeout" if timed_out else "ambiguous"
    elif relationship in {"major_correction", "minor_correction"}:
        blocked_reason = "newer_user_turn"
    elif relationship in {"pivot", "unrelated", "meta_complaint"}:
        blocked_reason = "stale_origin"
    else:
        blocked_reason = "ambiguous"

    logger.info(
        "context_resolution=%s context_resolution_source=%s transcript_context_llm_timed_out=%s "
        "context_handoff_allowed=%s context_handoff_blocked_reason=%s tool_revalidation_class=%s "
        "tool_revalidation_dependency=%s tool_revalidation_confidence=%.2f "
        "tool_revalidation_additive_min_dependency=%s tool_result_resume_decision=%s "
        "query_materially_changed=%s result_available=%s "
        "transcript_context_layer_enabled=%s transcript_context_llm_enabled=%s "
        "tool_revalidation_classifier_path=%s tool_revalidation_classifier_model=%s "
        "tool_revalidation_timeout_ms=%s tool_revalidation_context_resolution=%s "
        "tool_revalidation_resume_decision=%s "
        "context_classifier_path=%s context_classifier_timeout_ms=%s "
        "context_classifier_timeout_source=%s context_classifier_model=%s turn_id=%s",
        resolution,
        resolution_info.resolution_source,
        timed_out,
        handoff_allowed,
        blocked_reason,
        relationship,
        dependency,
        rel_conf,
        additive_min,
        decision,
        materially_changed,
        result_available,
        layer_enabled,
        llm_enabled,
        classifier_path,
        classifier_model,
        revalidation_timeout_ms,
        resolution,
        decision,
        context_classifier_path,
        revalidation_timeout_ms,
        timeout_source,
        classifier_model,
        _current_turn_id,
    )
    # Enforce, not just observe: only a compose decision grants the in-flight
    # result the right to speak.
    _interaction_state.runtime_gate(
        "stale_tool_result_speak",
        handoff_allowed,
        reason=f"resume_decision={decision};resolution={resolution};class={relationship}",
    )
    _interaction_state.apply_tool_resume_decision(
        relationship=relationship,
        decision=decision,
        resolution=resolution,
        additive_allowed=additive_allowed,
    )
    if decision == "compose" and result_available and _last_search_tool_output.strip():
        composed = _inject_tool_result_composition_note(
            turn_ctx, original_query, _last_search_tool_output, newer_utterance
        )
        logger.info(
            "tool_result_composed_with_newer_user_utterance=%s turn_id=%s",
            composed,
            _current_turn_id,
        )
    return decision


def _search_wait_elapsed_seconds() -> float:
    if _search_started_at <= 0.0:
        return 0.0
    end = time.monotonic() if _search_in_progress else (_search_completed_at or time.monotonic())
    return max(0.0, end - _search_started_at)


async def _maybe_await(value: object) -> None:
    if inspect.isawaitable(value):
        await value


def _test_invoke_cleanup_method(target: object, method_name: str, speech_id: str, reason: str) -> tuple[bool, str]:
    """Testable cleanup scheduler helper; production cleanup uses the same safe wrapping pattern."""
    method = getattr(target, method_name, None)
    if not callable(method):
        return False, "missing"
    result = method()
    if inspect.isawaitable(result):
        loop = asyncio.get_running_loop()
        loop.create_task(_maybe_await(result))
        logger.info(
            "Assistant speech cleanup method succeeded: speech_id=%s method=%s reason=%s result=scheduled_awaitable",
            speech_id,
            method_name,
            reason,
        )
        return True, "scheduled_awaitable"
    logger.info(
        "Assistant speech cleanup method succeeded: speech_id=%s method=%s reason=%s result=called",
        speech_id,
        method_name,
        reason,
    )
    return True, "called"


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


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    return f"{seconds:.1f}s"


def _valid_timestamp(value: float | int | None, *, after: float | None = None, before: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    if after is not None and timestamp < after:
        return None
    if before is not None and timestamp > before:
        return None
    return timestamp


def _duration_between(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return end - start


def _cap_recent_ids(id_set: set[str], id_order: list[str], max_size: int = 20) -> int:
    if max_size < 1:
        max_size = 1
    pruned = 0
    seen: set[str] = set()
    deduped_order: list[str] = []
    for item in id_order:
        if item in id_set and item not in seen:
            seen.add(item)
            deduped_order.append(item)
    for item in sorted(id_set):
        if item not in seen:
            seen.add(item)
            deduped_order.append(item)
    while len(deduped_order) > max_size:
        old = deduped_order.pop(0)
        if old in id_set:
            id_set.discard(old)
            pruned += 1
    id_order[:] = deduped_order
    return pruned


def _hume_request_dedupe_key(path: str, speech_id: str, normalized_text_hash: str) -> str:
    return f"{speech_id or 'n/a'}:{normalized_text_hash or 'n/a'}:{path or 'n/a'}"


def _record_hume_request_metadata(
    *,
    path: str,
    speech_id: str,
    normalized_text_hash: str,
    feeds_playout: bool,
    register: bool = True,
) -> tuple[str, bool, int]:
    key = _hume_request_dedupe_key(path, speech_id, normalized_text_hash)
    previous_count = _hume_recent_request_keys.get(key, 0)
    duplicate = previous_count > 0
    if register:
        _hume_recent_request_keys[key] = previous_count + 1
        if key not in _hume_recent_request_order:
            _hume_recent_request_order.append(key)
        while len(_hume_recent_request_order) > _HUME_RECENT_REQUEST_KEY_LIMIT:
            old = _hume_recent_request_order.pop(0)
            _hume_recent_request_keys.pop(old, None)
    return key, duplicate, previous_count + (1 if register else 0)


def _hume_wav_artifact_capture_enabled() -> bool:
    return env_bool("HUME_TTS_CAPTURE_WAV_DEBUG", False) or env_bool("HUME_TTS_WAV_ARTIFACT_CAPTURE_ENABLED", False)


def _hume_wav_artifact_max_bytes() -> int:
    raw = os.getenv("HUME_TTS_WAV_ARTIFACT_MAX_BYTES", "12000000")
    try:
        return max(1024, min(50_000_000, int(raw)))
    except Exception:
        return 12_000_000


def _safe_artifact_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown").strip("._")
    return token[:80] or "unknown"


def _audio_frame_for_coverage(value: object) -> object | None:
    frame = getattr(value, "frame", None)
    return frame if frame is not None else value


def _audio_frame_pcm_and_metadata(value: object) -> tuple[bytes, int, int, int]:
    frame = _audio_frame_for_coverage(value)
    if frame is None:
        return b"", 0, 0, 0
    data = getattr(frame, "data", b"")
    try:
        pcm = bytes(data)
    except Exception:
        return b"", 0, 0, 0
    try:
        sample_rate = int(getattr(frame, "sample_rate", 0) or 0)
    except Exception:
        sample_rate = 0
    try:
        num_channels = int(getattr(frame, "num_channels", 0) or getattr(frame, "channels", 0) or 0)
    except Exception:
        num_channels = 0
    try:
        samples_per_channel = int(getattr(frame, "samples_per_channel", 0) or 0)
    except Exception:
        samples_per_channel = 0
    if samples_per_channel <= 0 and num_channels > 0:
        samples_per_channel = len(pcm) // max(1, 2 * num_channels)
    return pcm, sample_rate, num_channels, samples_per_channel


def _start_hume_speech_audio_coverage(*, speech_id: str, turn_id: int, path: str, normalized_text_hash: str) -> HumeSpeechAudioCoverage:
    coverage = HumeSpeechAudioCoverage(
        speech_id=speech_id or "n/a",
        turn_id=turn_id,
        path=path,
        normalized_text_hash=normalized_text_hash or "n/a",
        started_at=time.monotonic(),
        capture_chunks=[] if _hume_wav_artifact_capture_enabled() else None,
    )
    logger.info(
        "hume_speech_coverage_started=true turn_id=%s speech_id=%s tts_path=%s normalized_text_hash=%s wav_artifact_capture_enabled=%s",
        coverage.turn_id,
        coverage.speech_id,
        coverage.path,
        coverage.normalized_text_hash,
        coverage.capture_chunks is not None,
    )
    return coverage


def _record_hume_speech_audio_frame(coverage: HumeSpeechAudioCoverage | None, value: object) -> None:
    if coverage is None:
        return
    pcm, sample_rate, num_channels, samples_per_channel = _audio_frame_pcm_and_metadata(value)
    if not pcm:
        return
    now = time.monotonic()
    coverage.frame_count += 1
    coverage.byte_count += len(pcm)
    coverage.sample_count += max(0, samples_per_channel)
    if sample_rate > 0:
        coverage.sample_rate = sample_rate
    if num_channels > 0:
        coverage.num_channels = num_channels
    if coverage.first_frame_at is None:
        coverage.first_frame_at = now
    coverage.last_frame_at = now
    if coverage.capture_chunks is not None and not coverage.capture_truncated:
        max_bytes = _hume_wav_artifact_max_bytes()
        captured_so_far = sum(len(chunk) for chunk in coverage.capture_chunks)
        if captured_so_far + len(pcm) <= max_bytes:
            coverage.capture_chunks.append(pcm)
        else:
            coverage.capture_truncated = True


def _hume_generated_audio_duration_seconds(coverage: HumeSpeechAudioCoverage | None) -> float | None:
    if coverage is None or coverage.sample_rate <= 0:
        return None
    return coverage.sample_count / coverage.sample_rate


def _write_hume_wav_artifact(coverage: HumeSpeechAudioCoverage) -> str | None:
    if not coverage.capture_chunks:
        return None
    sample_rate = coverage.sample_rate or 48000
    num_channels = coverage.num_channels or 1
    artifact_dir = os.getenv("HUME_TTS_WAV_ARTIFACT_DIR", "/tmp/lucy_hume_tts_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    filename = (
        f"{int(time.time())}_turn-{coverage.turn_id}_speech-{_safe_artifact_token(coverage.speech_id)}_"
        f"{_safe_artifact_token(coverage.path)}_{_safe_artifact_token(coverage.normalized_text_hash)}.wav"
    )
    artifact_path = os.path.join(artifact_dir, filename)
    with wave.open(artifact_path, "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(coverage.capture_chunks))
    coverage.artifact_path = artifact_path
    return artifact_path


def _finalize_hume_speech_audio_coverage(coverage: HumeSpeechAudioCoverage | None, *, error_type: str = "none") -> HumeSpeechAudioCoverage | None:
    if coverage is None:
        return None
    artifact_path = None
    if coverage.capture_chunks is not None:
        try:
            artifact_path = _write_hume_wav_artifact(coverage)
        except Exception as exc:
            logger.warning(
                "hume_wav_artifact_capture_failed=true speech_id=%s error_type=%s error=%s",
                coverage.speech_id,
                type(exc).__name__,
                _redact_sensitive_text(exc),
            )
    generated_duration = _hume_generated_audio_duration_seconds(coverage)
    logger.info(
        "hume_speech_coverage_completed=true turn_id=%s speech_id=%s tts_path=%s normalized_text_hash=%s generated_frame_count=%s generated_audio_bytes=%s generated_sample_rate=%s generated_channels=%s generated_audio_duration_seconds=%s first_generated_audio_seconds=%s last_generated_audio_seconds=%s wav_artifact_capture_enabled=%s wav_artifact_path=%s wav_artifact_truncated=%s error_type=%s",
        coverage.turn_id,
        coverage.speech_id,
        coverage.path,
        coverage.normalized_text_hash,
        coverage.frame_count,
        coverage.byte_count,
        coverage.sample_rate or "unknown",
        coverage.num_channels or "unknown",
        _fmt_seconds(generated_duration),
        _fmt_seconds((coverage.first_frame_at - coverage.started_at) if coverage.first_frame_at is not None else None),
        _fmt_seconds((coverage.last_frame_at - coverage.started_at) if coverage.last_frame_at is not None else None),
        coverage.capture_chunks is not None,
        artifact_path or "none",
        coverage.capture_truncated,
        error_type,
    )
    return coverage


def _assistant_cleanup_action(
    *,
    cleanup_reason: str,
    current_user_turn_id: int,
    speech_turn_id: int | None,
    latest_user_state: str,
) -> str:
    normalized_reason = (cleanup_reason or "").strip().lower()
    normalized_user_state = (latest_user_state or "").strip().lower()
    if normalized_reason in {"session_closing", "session_close", "disconnect", "shutdown"}:
        return "interrupt"
    if normalized_user_state == "speaking" or normalized_reason in {"user_speaking_before_assistant_start", "user_started_speaking"}:
        return "interrupt"
    if speech_turn_id is not None and speech_turn_id > 0 and speech_turn_id < current_user_turn_id:
        return "interrupt"
    return "skip"


def _build_voice_latency_audit(
    *,
    turn_id: int,
    speech_id: str,
    user_speech_started_at: float | None,
    user_speech_stopped_at: float | None,
    final_stt_received_at: float | None,
    user_turn_committed_at: float | None,
    llm_request_started_at: float | None,
    llm_first_token_at: float | None,
    llm_completed_at: float | None,
    tts_request_started_at: float | None,
    tts_first_audio_at: float | None,
    tts_completed_at: float | None,
    assistant_playout_started_at: float | None,
    assistant_playout_completed_at: float | None,
) -> dict[str, float | int | str | None]:
    user_start = _valid_timestamp(user_speech_started_at)
    user_stop = _valid_timestamp(user_speech_stopped_at, after=user_start) if user_start is not None else _valid_timestamp(user_speech_stopped_at)
    final_stt = _valid_timestamp(final_stt_received_at, after=user_start) if user_start is not None else _valid_timestamp(final_stt_received_at)
    turn_committed = _valid_timestamp(user_turn_committed_at, after=final_stt) if final_stt is not None else _valid_timestamp(user_turn_committed_at)
    llm_start = _valid_timestamp(llm_request_started_at, after=turn_committed) if turn_committed is not None else _valid_timestamp(llm_request_started_at)
    llm_first = _valid_timestamp(llm_first_token_at, after=llm_start) if llm_start is not None else _valid_timestamp(llm_first_token_at)
    llm_done = _valid_timestamp(llm_completed_at, after=llm_start) if llm_start is not None else _valid_timestamp(llm_completed_at)
    tts_start = _valid_timestamp(tts_request_started_at, after=turn_committed) if turn_committed is not None else _valid_timestamp(tts_request_started_at)
    # tts_first must be anchored to this turn. When tts_start is missing we cannot
    # accept a raw tts_first_audio_at: a stale global (e.g. the greeting's
    # first-audio timestamp captured at session start) would otherwise leak in and
    # make tts_first_audio_to_playout_start report the whole session gap (~11-15s)
    # instead of the real per-speech latency. Fall back to the turn-commit boundary
    # so any timestamp predating this turn is rejected.
    if tts_start is not None:
        tts_first = _valid_timestamp(tts_first_audio_at, after=tts_start)
    elif turn_committed is not None:
        tts_first = _valid_timestamp(tts_first_audio_at, after=turn_committed)
    else:
        tts_first = _valid_timestamp(tts_first_audio_at)
    tts_done = _valid_timestamp(tts_completed_at, after=tts_start) if tts_start is not None else _valid_timestamp(tts_completed_at)
    playout_start = _valid_timestamp(assistant_playout_started_at, after=tts_first) if tts_first is not None else _valid_timestamp(assistant_playout_started_at)
    playout_done = _valid_timestamp(assistant_playout_completed_at, after=playout_start) if playout_start is not None else _valid_timestamp(assistant_playout_completed_at)
    return {
        "turn_id": turn_id,
        "speech_id": speech_id,
        "user_speech_started_at": user_start,
        "user_speech_stopped_at": user_stop,
        "final_stt_received_at": final_stt,
        "user_turn_committed_at": turn_committed,
        "llm_request_started_at": llm_start,
        "llm_first_token_at": llm_first,
        "llm_completed_at": llm_done,
        "tts_request_started_at": tts_start,
        "tts_first_audio_at": tts_first,
        "tts_completed_at": tts_done,
        "assistant_playout_started_at": playout_start,
        "assistant_playout_completed_at": playout_done,
        "user_stopped_to_final_stt": _duration_between(user_stop, final_stt),
        "final_stt_to_turn_committed": _duration_between(final_stt, turn_committed),
        "turn_committed_to_llm_first_token": _duration_between(turn_committed, llm_first),
        "llm_first_token_to_llm_complete": _duration_between(llm_first, llm_done),
        "llm_complete_to_tts_request": _duration_between(llm_done, tts_start),
        "tts_request_to_first_audio": _duration_between(tts_start, tts_first),
        "tts_first_audio_to_playout_start": _duration_between(tts_first, playout_start),
        "user_stopped_to_first_audio": _duration_between(user_stop, tts_first),
        "user_stopped_to_assistant_complete": _duration_between(user_stop, playout_done),
    }


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else "empty"


def _record_stt_final_candidate(text: str, *, user_state: str, partial_count: int) -> str:
    """Record an immutable candidate for a final STT result and return its id."""
    global _stt_segment_counter
    _stt_segment_counter += 1
    candidate_id = f"stt-{_stt_segment_counter}"
    _stt_candidates.append(
        {
            "candidate_id": candidate_id,
            "text": text or "",
            "text_hash": _text_hash((text or "").strip()),
            "received_at": time.monotonic(),
            "user_state": user_state,
            "partial_count_at_final": partial_count,
        }
    )
    if len(_stt_candidates) > 40:
        del _stt_candidates[:-40]
    return candidate_id


def _bind_candidate_for_commit(committed_text: str) -> tuple[str, bool, str]:
    """Bind the committing turn to the STT candidate that produced it.

    Returns (candidate_id, drift_suspected, latest_final_hash). Prefers the most
    recent candidate whose text matches the committed text; if the newest final
    differs from what is being committed, transcript drift is flagged so the
    one-segment lag is visible in logs.
    """
    committed_hash = _text_hash((committed_text or "").strip())
    latest_final_hash = _stt_candidates[-1]["text_hash"] if _stt_candidates else "none"
    matched_id = "none"
    for candidate in reversed(_stt_candidates):
        if candidate["text_hash"] == committed_hash:
            matched_id = candidate["candidate_id"]
            break
    drift_suspected = bool(_stt_candidates) and latest_final_hash != committed_hash
    candidate_id = matched_id if matched_id != "none" else (f"commit-{_current_turn_id}" if committed_text else "empty")
    return candidate_id, drift_suspected, latest_final_hash


def _categorize_transcript_drift(committed_text: str) -> str:
    """Classify a suspected transcript drift so logs distinguish expected merges
    from real one-segment lag.

    Returns one of:
      none                       committed text equals the newest final
      expected_merge_or_superset committed already contains the newest final
                                 (a merged/superset turn — expected behavior)
      real_lag                   the newest final is a superset/continuation of
                                 the committed text (committed lags one segment)
      ambiguous_commit           newest final and committed text diverge in a way
                                 that cannot be classified as merge or lag
      no_candidates              no STT candidates to compare against
    """
    if not _stt_candidates:
        return "no_candidates"
    latest = (_stt_candidates[-1].get("text") or "").strip().lower()
    committed = (committed_text or "").strip().lower()
    if not latest or not committed:
        return "ambiguous_commit"
    if latest == committed:
        return "none"
    if latest in committed:
        # Committed text already contains the newest final: merged/superset turn.
        return "expected_merge_or_superset"
    if committed in latest:
        # Newest final is a superset of what committed: committed lags behind.
        return "real_lag"
    # Disjoint finals: treat a shared leading prefix as lag, otherwise ambiguous.
    prefix_len = 0
    for a, b in zip(latest, committed):
        if a != b:
            break
        prefix_len += 1
    shared_prefix = prefix_len >= max(6, min(len(latest), len(committed)) // 2)
    return "real_lag" if shared_prefix else "ambiguous_commit"


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


def _safe_llm_error_details(error: object) -> dict[str, str]:
    details: dict[str, str] = {"error_type": type(error).__name__}
    nested = getattr(error, "error", None)
    if nested is not None:
        details["nested_error_type"] = type(nested).__name__
        for field in ("message", "reason", "detail", "status", "status_code", "code", "provider"):
            value = getattr(nested, field, None)
            if value is not None:
                details[field] = _redact_sensitive_text(value)
    for field in ("message", "reason", "detail", "status", "status_code", "code", "provider"):
        value = getattr(error, field, None)
        if value is not None and field not in details:
            details[field] = _redact_sensitive_text(value)
    return details


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


def _deployment_git_commit_sha() -> str:
    for env_name in (
        "RAILWAY_GIT_COMMIT_SHA",
        "RAILWAY_GIT_COMMIT",
        "GIT_COMMIT_SHA",
        "GIT_SHA",
        "SOURCE_COMMIT",
        "VERCEL_GIT_COMMIT_SHA",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def attach_session_diagnostics(session: AgentSession) -> None:
    global _latest_agent_state_for_hume, _latest_active_assistant_count_for_hume, _latest_current_speech_id_for_hume
    active_speech_handles: dict[str, object] = {}
    _local_speech_ids: dict[int, str] = {}
    payload_debug_logged = False
    speech_start_times: dict[str, float] = {}
    suppressed_speech_ids: set[str] = set()
    stale_speech_ids: set[str] = set()
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
    stale_speech_id_order: list[str] = []
    speech_latency_audits: dict[str, dict[str, float | int | str | None]] = {}
    assistant_speech_turn_ids: dict[str, int] = {}
    assistant_speech_llm_turn_ids: dict[str, int] = {}
    active_speech_interrupted_at: dict[str, float] = {}
    last_assistant_speech_outcome: dict[str, object] | None = None

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

    def _prune_stale_speech_ids(max_size: int = 20) -> int:
        pruned = _cap_recent_ids(stale_speech_ids, stale_speech_id_order, max_size=max_size)
        if pruned:
            logger.info(
                "stale_speech_ids_pruned_count=%s stale_speech_ids_count=%s stale_speech_ids=%s",
                pruned,
                len(stale_speech_ids),
                sorted(stale_speech_ids),
            )
        return pruned

    def _mark_speech_stale(speech_id: str) -> None:
        if speech_id not in stale_speech_ids:
            stale_speech_id_order.append(speech_id)
        stale_speech_ids.add(speech_id)
        _prune_stale_speech_ids()

    def _unmark_speech_stale(speech_id: str) -> None:
        stale_speech_ids.discard(speech_id)
        while speech_id in stale_speech_id_order:
            stale_speech_id_order.remove(speech_id)

    def _speech_produced_audio(speech_id: str, started_at: float | None = None, hume_coverage: object | None = None) -> bool:
        if started_at is not None:
            return True
        if speech_id in agent_speaking_at:
            return True
        coverage = hume_coverage if hume_coverage is not None else _hume_speech_audio_coverages.get(speech_id)
        return bool(getattr(coverage, "byte_count", 0) or getattr(coverage, "frame_count", 0))

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

    async def _maybe_await_cleanup_result(value: object, speech_id: str, method_name: str, reason: str) -> None:
        if inspect.isawaitable(value):
            try:
                await value
                logger.info(
                    "Assistant speech cleanup awaitable completed: speech_id=%s method=%s reason=%s",
                    speech_id,
                    method_name,
                    reason,
                )
            except Exception as exc:
                logger.warning(
                    "Assistant speech cleanup awaitable failed: speech_id=%s method=%s reason=%s error=%s",
                    speech_id,
                    method_name,
                    reason,
                    _redact_sensitive_text(exc),
                )

    def _invoke_cancel_method(target: object, method_name: str, speech_id: str, reason: str) -> tuple[bool, str]:
        method = getattr(target, method_name, None)
        if not callable(method):
            return False, "missing"
        try:
            result = method()
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_maybe_await_cleanup_result(result, speech_id, method_name, reason))
                    logger.info(
                        "Assistant speech cleanup method succeeded: speech_id=%s method=%s reason=%s result=scheduled_awaitable",
                        speech_id,
                        method_name,
                        reason,
                    )
                    return True, "scheduled_awaitable"
                except RuntimeError:
                    logger.warning(
                        "Assistant speech cleanup awaitable could not be scheduled: speech_id=%s method=%s reason=%s",
                        speech_id,
                        method_name,
                        reason,
                    )
                    close_awaitable = getattr(result, "close", None)
                    if callable(close_awaitable):
                        close_awaitable()
                    return False, "awaitable_without_running_loop"
            logger.info(
                "Assistant speech cleanup method succeeded: speech_id=%s method=%s reason=%s result=called",
                speech_id,
                method_name,
                reason,
            )
            return True, "called"
        except Exception as e:
            logger.warning(
                "Assistant speech cleanup method failed: speech_id=%s method=%s reason=%s error=%s",
                speech_id,
                method_name,
                reason,
                _redact_sensitive_text(e),
            )
            return False, "failed"

    def _cleanup_active_assistant_speeches(current_new_speech_id: str | None, cleanup_reason: str | None = None) -> None:
        nonlocal pending_user_handoff_speech_id
        global _latest_active_assistant_count_for_hume
        if cleanup_reason is None:
            cleanup_reason = "legacy_cleanup_call"
            logger.info(
                "assistant_speech_cleanup_started cleanup_reason=%s current_user_turn_id=%s current_new_speech_id=%s active_count_before=%s cleanup_action=skip reason=legacy_cleanup_disabled",
                cleanup_reason,
                _current_turn_id,
                current_new_speech_id or "none",
                len(active_speech_handles),
            )
            for speech_id in active_speech_handles:
                speech_turn_id = assistant_speech_turn_ids.get(speech_id, 0)
                speech_llm_turn_id = assistant_speech_llm_turn_ids.get(speech_id, 0)
                logger.info(
                    "assistant_speech_cleanup_decision cleanup_reason=%s current_user_turn_id=%s speech_id=%s speech_turn_id=%s llm_turn_id=%s is_current_turn_speech=%s cleanup_action=skip latest_user_state=%s",
                    cleanup_reason,
                    _current_turn_id,
                    speech_id,
                    speech_turn_id or "unknown",
                    speech_llm_turn_id or "unknown",
                    speech_turn_id == _current_turn_id,
                    latest_user_state,
                )
            return
        active_count_before = len(active_speech_handles)
        active_ids_before = list(active_speech_handles.keys())
        logger.info(
            "assistant_speech_cleanup_started turn_id=%s cleanup_reason=%s current_new_speech_id=%s active_count_before=%s active_speech_ids=%s stale_speech_ids=%s",
            _current_turn_id,
            cleanup_reason,
            current_new_speech_id or "none",
            active_count_before,
            active_ids_before,
            sorted(stale_speech_ids),
        )
        if not active_speech_handles:
            logger.info(
                "Assistant speech cleanup finished: turn_id=%s cleanup_reason=%s active_count_before=%s active_count_after=%s stale_speech_ids=%s new_speech_allowed=%s",
                _current_turn_id,
                cleanup_reason,
                active_count_before,
                0,
                sorted(stale_speech_ids),
                True,
            )
            return

        cleanup_ids = [speech_id for speech_id in active_speech_handles if speech_id != current_new_speech_id]
        if not cleanup_ids:
            logger.info(
                "assistant_speech_cleanup_skipped_current_speech cleanup_reason=%s current_new_speech_id=%s active_count_before=%s active_count_after=%s stale_speech_ids=%s",
                cleanup_reason,
                current_new_speech_id or "none",
                active_count_before,
                len(active_speech_handles),
                sorted(stale_speech_ids),
            )
            _latest_active_assistant_count_for_hume = len(active_speech_handles)
            return

        for speech_id in cleanup_ids:
            handle = active_speech_handles.get(speech_id)
            if handle is None:
                continue

            speech_turn_id = assistant_speech_turn_ids.get(speech_id, 0)
            speech_llm_turn_id = assistant_speech_llm_turn_ids.get(speech_id, 0)
            is_current_turn_speech = speech_turn_id == _current_turn_id
            cleanup_action = _assistant_cleanup_action(
                cleanup_reason=cleanup_reason,
                current_user_turn_id=_current_turn_id,
                speech_turn_id=speech_turn_id,
                latest_user_state=latest_user_state,
            )
            if not speech_turn_id or not speech_llm_turn_id:
                logger.warning(
                    "assistant_speech_ownership_unknown=true cleanup_reason=%s current_user_turn_id=%s speech_id=%s speech_turn_id=%s llm_turn_id=%s",
                    cleanup_reason,
                    _current_turn_id,
                    speech_id,
                    speech_turn_id or "unknown",
                    speech_llm_turn_id or "unknown",
                )
            logger.info(
                "assistant_speech_cleanup_decision cleanup_reason=%s current_user_turn_id=%s speech_id=%s speech_turn_id=%s llm_turn_id=%s is_current_turn_speech=%s cleanup_action=%s latest_user_state=%s",
                cleanup_reason,
                _current_turn_id,
                speech_id,
                speech_turn_id or "unknown",
                speech_llm_turn_id or "unknown",
                is_current_turn_speech,
                cleanup_action,
                latest_user_state,
            )
            if cleanup_action == "skip":
                continue

            attempted_method = "none"
            cleanup_result = "not_attempted"
            method_targets: list[tuple[str, object]] = [("", handle)]
            nested_handle = getattr(handle, "speech_handle", None)
            if nested_handle is not None and nested_handle is not handle:
                method_targets.append(("speech_handle.", nested_handle))

            for prefix, target in method_targets:
                for method_name in ("interrupt", "cancel", "stop", "close"):
                    attempted_method = f"{prefix}{method_name}"
                    ok, result = _invoke_cancel_method(target, method_name, speech_id, cleanup_reason)
                    if ok:
                        cleanup_result = f"cancel_requested:{attempted_method}:{result}"
                        break
                    if result != "missing":
                        cleanup_result = f"cancel_failed:{attempted_method}:{result}"
                if cleanup_result.startswith("cancel_requested"):
                    break

            if not cleanup_result.startswith("cancel_requested"):
                logger.warning(
                    "Assistant speech cleanup: no safe cancel method speech_id=%s cleanup_reason=%s attempted_method=%s",
                    speech_id,
                    cleanup_reason,
                    attempted_method,
                )
                cleanup_result = "marked_stale_no_safe_cancel_method"

            assistant_speech_turn_ids[speech_id] = speech_turn_id
            assistant_speech_llm_turn_ids[speech_id] = speech_llm_turn_id
            was_stale_before_cleanup = speech_id in stale_speech_ids
            produced_audio = _speech_produced_audio(speech_id, speech_start_times.get(speech_id))
            _mark_speech_stale(speech_id)
            suppressed_speech_ids.add(speech_id)
            active_speech_handles.pop(speech_id, None)
            speech_start_times.pop(speech_id, None)
            speech_latency_audits.pop(speech_id, None)
            _hume_speech_audio_coverages.pop(speech_id, None)
            assistant_speech_turn_ids.pop(speech_id, None)
            assistant_speech_llm_turn_ids.pop(speech_id, None)
            hume_request_count_at_speech_finish.setdefault(speech_id, _hume_tts_request_counter)
            assistant_speech_finished_at.setdefault(speech_id, time.monotonic())
            if pending_user_handoff_speech_id == speech_id:
                pending_user_handoff_speech_id = None
            logger.info(
                "speech_handle_cleanup reason=%s speech_id=%s was_stale=%s was_active=%s produced_audio=%s marked_interrupted=%s cleanup_action=%s",
                cleanup_reason,
                speech_id,
                was_stale_before_cleanup,
                True,
                produced_audio,
                cleanup_result.startswith("cancel_requested"),
                cleanup_result,
            )
            logger.info(
                "Assistant speech cleanup item: turn_id=%s cleanup_reason=%s speech_id=%s current_new_speech_id=%s attempted_method=%s cleanup_result=%s active_count_after_item=%s stale_speech_ids=%s",
                _current_turn_id,
                cleanup_reason,
                speech_id,
                current_new_speech_id or "none",
                attempted_method,
                cleanup_result,
                len(active_speech_handles),
                sorted(stale_speech_ids),
            )

        _latest_active_assistant_count_for_hume = len(active_speech_handles)
        new_speech_allowed = current_new_speech_id is None or len(active_speech_handles) <= 1
        logger.info(
            "Assistant speech cleanup finished: turn_id=%s cleanup_reason=%s current_new_speech_id=%s active_count_before=%s active_count_after=%s stale_speech_ids=%s new_speech_allowed=%s",
            _current_turn_id,
            cleanup_reason,
            current_new_speech_id or "none",
            active_count_before,
            len(active_speech_handles),
            sorted(stale_speech_ids),
            new_speech_allowed,
        )

    def _clear_active_handles(reason: str) -> None:
        global _latest_active_assistant_count_for_hume
        cleared_count = len(active_speech_handles)
        if cleared_count or speech_start_times or suppressed_speech_ids or stale_speech_ids:
            logger.warning(
                "Clearing stale active speech handles: reason=%s cleared_count=%s suppressed_count=%s stale_count=%s",
                reason,
                cleared_count,
                len(suppressed_speech_ids),
                len(stale_speech_ids),
            )
        for speech_id in list(active_speech_handles.keys()):
            was_active = speech_id in active_speech_handles
            was_stale = speech_id in stale_speech_ids
            produced_audio = _speech_produced_audio(speech_id, speech_start_times.get(speech_id))
            if reason == "agent_returned_to_listening" and produced_audio:
                logger.info(
                    "speech_handle_cleanup reason=%s speech_id=%s was_stale=%s was_active=%s produced_audio=%s marked_interrupted=%s cleanup_action=defer_until_done_callback",
                    reason,
                    speech_id,
                    was_stale,
                    was_active,
                    produced_audio,
                    False,
                )
                continue
            _mark_speech_stale(speech_id)
            suppressed_speech_ids.add(speech_id)
            active_speech_handles.pop(speech_id, None)
            speech_start_times.pop(speech_id, None)
            speech_latency_audits.pop(speech_id, None)
            _hume_speech_audio_coverages.pop(speech_id, None)
            assistant_speech_turn_ids.pop(speech_id, None)
            assistant_speech_llm_turn_ids.pop(speech_id, None)
            logger.info(
                "speech_handle_cleanup reason=%s speech_id=%s was_stale=%s was_active=%s produced_audio=%s marked_interrupted=%s cleanup_action=mark_stale_suppressed",
                reason,
                speech_id,
                was_stale,
                was_active,
                produced_audio,
                False,
            )
        _prune_stale_speech_ids()
        _latest_active_assistant_count_for_hume = len(active_speech_handles)


    @session.on("speech_created")
    def _on_speech_created(event_or_handle: object) -> None:
        nonlocal payload_debug_logged
        global _latest_current_speech_id_for_hume, _latest_active_assistant_count_for_hume

        if not payload_debug_logged:
            payload_debug_logged = True
            attrs = ("id", "speech_id", "handle", "speech", "speech_handle", "interrupted", "add_done_callback", "interrupt", "wait_for_playout", "cancel", "stop", "close")
            attr_presence = {name: hasattr(event_or_handle, name) for name in attrs}
            logger.info("speech_created payload debug: type=%s attrs=%s suppress_window_seconds=%s", type(event_or_handle).__name__, attr_presence, overlap_suppress_window_seconds)

        resolved_handle = _resolve_speech_handle(event_or_handle)
        speech_id = _speech_id(resolved_handle)
        speech_turn_id = _llm_turn_id if _llm_turn_id > 0 else _current_turn_id
        speech_llm_turn_id = _llm_turn_id if _llm_turn_id > 0 else speech_turn_id
        now = time.monotonic()
        speech_created_at[speech_id] = now
        pending_hume_coverage = _hume_speech_audio_coverages.pop("n/a", None)
        if pending_hume_coverage is not None and speech_id not in _hume_speech_audio_coverages:
            pending_hume_coverage.speech_id = speech_id
            _hume_speech_audio_coverages[speech_id] = pending_hume_coverage
            logger.info(
                "hume_speech_coverage_assigned_to_speech=true turn_id=%s speech_id=%s previous_speech_id=n/a tts_path=%s",
                speech_turn_id,
                speech_id,
                pending_hume_coverage.path,
            )

        if latest_user_state == "speaking":
            attempted_method = "none"
            cleanup_result = "not_attempted"
            for method_name in ("interrupt", "cancel", "stop", "close"):
                attempted_method = method_name
                ok, result = _invoke_cancel_method(resolved_handle, method_name, speech_id, "user_speaking_before_assistant_start")
                if ok:
                    cleanup_result = f"cancel_requested:{method_name}:{result}"
                    break
                if result != "missing":
                    cleanup_result = f"cancel_failed:{method_name}:{result}"
            assistant_speech_turn_ids[speech_id] = speech_turn_id
            assistant_speech_llm_turn_ids[speech_id] = speech_llm_turn_id
            was_stale_before_cleanup = speech_id in stale_speech_ids
            produced_audio = _speech_produced_audio(speech_id, speech_start_times.get(speech_id))
            _mark_speech_stale(speech_id)
            suppressed_speech_ids.add(speech_id)
            assistant_speech_finished_at.setdefault(speech_id, time.monotonic())
            logger.warning(
                "real_user_interruption=true speech_id=%s evidence=%s",
                speech_id,
                "user_speaking_before_assistant_start",
            )
            logger.info(
                "speech_handle_cleanup reason=%s speech_id=%s was_stale=%s was_active=%s produced_audio=%s marked_interrupted=%s cleanup_action=%s",
                "user_speaking_before_assistant_start",
                speech_id,
                was_stale_before_cleanup,
                False,
                produced_audio,
                cleanup_result.startswith("cancel_requested"),
                cleanup_result,
            )
            # Enforce: a speech object created while the user is speaking must not
            # reach audio playout (the FSM also only enters SPEAKING on real
            # playout). Block here and record it as a gated high-risk action.
            _interaction_state.runtime_gate(
                "assistant_speech_start_before_playout",
                False,
                reason="user_speaking_before_assistant_start",
            )
            logger.warning(
                "assistant_speech_start_allowed=false assistant_speech_start_blocked_reason=user_speaking current_user_turn_id=%s speech_id=%s speech_turn_id=%s llm_turn_id=%s cleanup_reason=%s cleanup_action=interrupt attempted_method=%s cleanup_result=%s",
                _current_turn_id,
                speech_id,
                speech_turn_id,
                speech_llm_turn_id,
                "user_speaking_before_assistant_start",
                attempted_method,
                cleanup_result,
            )
            return

        if speech_id in stale_speech_ids:
            logger.warning(
                "Assistant speech recreated with stale id; clearing stale marker before registration: speech_id=%s stale_speech_ids=%s",
                speech_id,
                sorted(stale_speech_ids),
            )
            _unmark_speech_stale(speech_id)
            suppressed_speech_ids.discard(speech_id)

        if active_speech_handles and set(active_speech_handles.keys()) == {speech_id}:
            _cleanup_active_assistant_speeches(speech_id, "current_speech_already_active")
            logger.info(
                "Assistant speech already active; skipping duplicate registration: speech_id=%s active_count=%s",
                speech_id,
                len(active_speech_handles),
            )
            return

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
            logger.warning(
                "Assistant overlap detected before new speech: new_speech_id=%s active_speech_ids_before_new=%s session_current_speech_id=%s session_current_speech_type=%s latest_user_state=%s latest_agent_state=%s seconds_since_user_state_change=%.3f seconds_since_agent_state_change=%.3f user_state_is_speaking=%s agent_state_is_thinking=%s agent_state_is_speaking=%s agent_state_is_listening=%s",
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
            _cleanup_active_assistant_speeches(speech_id, "before_new_assistant_speech")

        if active_speech_handles and set(active_speech_handles.keys()) == {speech_id}:
            logger.info(
                "Assistant speech already active after cleanup; skipping duplicate registration: speech_id=%s active_count=%s",
                speech_id,
                len(active_speech_handles),
            )
            return

        if active_speech_handles:
            logger.error(
                "Assistant speech active_count would exceed invariant before new speech; forcing stale cleanup of older speeches only: new_speech_id=%s active_speech_ids=%s",
                speech_id,
                list(active_speech_handles.keys()),
            )
            _cleanup_active_assistant_speeches(speech_id, "force_cleanup_before_new_registration")
            if active_speech_handles:
                logger.warning(
                    "assistant_speech_start_allowed=false assistant_speech_start_blocked_reason=active_speech_cleanup_skipped_current_turn current_user_turn_id=%s speech_id=%s speech_turn_id=%s active_speech_ids=%s",
                    _current_turn_id,
                    speech_id,
                    speech_turn_id,
                    list(active_speech_handles.keys()),
                )
                return

        logger.info(
            "assistant_speech_start_allowed=true assistant_speech_start_blocked_reason=none turn_id=%s speech_id=%s",
            _current_turn_id,
            speech_id,
        )
        assistant_speech_turn_ids[speech_id] = speech_turn_id
        assistant_speech_llm_turn_ids[speech_id] = speech_llm_turn_id
        active_speech_handles[speech_id] = resolved_handle
        speech_start_times[speech_id] = now
        assistant_speech_started_at[speech_id] = now
        speech_latency_audits[speech_id] = _build_voice_latency_audit(
            turn_id=_current_turn_id,
            speech_id=speech_id,
            user_speech_started_at=last_user_speaking_at,
            user_speech_stopped_at=last_user_listening_at,
            final_stt_received_at=last_stt_final_at,
            user_turn_committed_at=_last_turn_committed_at,
            llm_request_started_at=_last_llm_start_at,
            llm_first_token_at=_last_llm_first_token_at,
            llm_completed_at=_last_llm_complete_at,
            tts_request_started_at=_last_tts_request_start_at,
            tts_first_audio_at=_last_tts_first_audio_at,
            tts_completed_at=_last_tts_completed_at,
            assistant_playout_started_at=now,
            assistant_playout_completed_at=None,
        )
        hume_request_count_at_speech_start[speech_id] = _hume_tts_request_counter
        _latest_current_speech_id_for_hume = speech_id
        _latest_active_assistant_count_for_hume = len(active_speech_handles)
        if len(active_speech_handles) > 1:
            logger.error(
                "Assistant speech active_count invariant violated after registration: new_speech_id=%s active_count=%s active_speech_ids=%s",
                speech_id,
                len(active_speech_handles),
                list(active_speech_handles.keys()),
            )
            _cleanup_active_assistant_speeches(speech_id, "post_registration_active_count_gt_one")
            assistant_speech_turn_ids[speech_id] = speech_turn_id
            assistant_speech_llm_turn_ids[speech_id] = speech_llm_turn_id
            active_speech_handles[speech_id] = resolved_handle
            speech_start_times[speech_id] = now
            assistant_speech_started_at[speech_id] = now
            speech_latency_audits[speech_id] = _build_voice_latency_audit(
                turn_id=_current_turn_id,
                speech_id=speech_id,
                user_speech_started_at=last_user_speaking_at,
                user_speech_stopped_at=last_user_listening_at,
                final_stt_received_at=last_stt_final_at,
                user_turn_committed_at=_last_turn_committed_at,
                llm_request_started_at=_last_llm_start_at,
                llm_first_token_at=_last_llm_first_token_at,
                llm_completed_at=_last_llm_complete_at,
                tts_request_started_at=_last_tts_request_start_at,
                tts_first_audio_at=_last_tts_first_audio_at,
                tts_completed_at=_last_tts_completed_at,
                assistant_playout_started_at=now,
                assistant_playout_completed_at=None,
            )
            hume_request_count_at_speech_start[speech_id] = _hume_tts_request_counter
            _latest_current_speech_id_for_hume = speech_id
            _latest_active_assistant_count_for_hume = len(active_speech_handles)

        # Speech OBJECT created — do NOT mark the FSM as SPEAKING yet. Real audio
        # playout is signalled separately by agent_state_changed -> "speaking",
        # which is where the ASSISTANT_SPEAKING transition is driven.
        _interaction_state.on_assistant_speech_created(speech_id)
        logger.info(
            "Assistant speech started: current_user_turn_id=%s speech_id=%s speech_turn_id=%s llm_turn_id=%s active_count=%s",
            _current_turn_id,
            speech_id,
            speech_turn_id,
            speech_llm_turn_id,
            len(active_speech_handles),
        )
        logger.info(
            "Assistant speech Hume request baseline: speech_id=%s hume_request_count_at_start=%s active_count=%s",
            speech_id,
            hume_request_count_at_speech_start[speech_id],
            len(active_speech_handles),
        )

        add_done_callback = getattr(resolved_handle, "add_done_callback", None)
        if callable(add_done_callback):
            def _on_done(done_event_or_handle: object) -> None:
                nonlocal pending_user_handoff_speech_id, last_assistant_speech_outcome
                global _latest_active_assistant_count_for_hume
                done_resolved_handle = _resolve_speech_handle(done_event_or_handle)
                done_id = _speech_id(done_resolved_handle)
                was_active = done_id in active_speech_handles
                was_stale = done_id in stale_speech_ids
                active_speech_handles.pop(done_id, None)
                _latest_active_assistant_count_for_hume = len(active_speech_handles)
                finished_at = time.monotonic()
                started_at = speech_start_times.pop(done_id, None)
                was_suppressed = done_id in suppressed_speech_ids
                suppressed_speech_ids.discard(done_id)
                _unmark_speech_stale(done_id)
                interrupted = _safe_attr(done_resolved_handle, "interrupted", "unknown")
                hume_coverage_for_audio = _hume_speech_audio_coverages.get(done_id)
                produced_audio = _speech_produced_audio(done_id, started_at, hume_coverage_for_audio)
                # The TTS handle can report interrupted=False even when the FSM
                # observed the user barging in. The FSM observation is
                # authoritative for ledger ownership, so combine them, but stale
                # zero-audio cleanup is reconciliation rather than interruption.
                fsm_observed_interrupted = _interaction_state.was_speech_interrupted(done_id)
                effective_interrupted = _effective_interruption_for_speech(
                    interrupted,
                    fsm_observed_interrupted,
                    was_stale=was_stale,
                    produced_audio=produced_audio,
                )
                if effective_interrupted:
                    evidence = "fsm_observed_interrupted" if fsm_observed_interrupted else "handle_interrupted"
                    logger.warning("real_user_interruption=true speech_id=%s evidence=%s", done_id, evidence)
                elif was_stale and not produced_audio:
                    logger.info("stale_speech_finished_without_interruption=true speech_id=%s", done_id)
                logger.info(
                    "speech_handle_cleanup reason=%s speech_id=%s was_stale=%s was_active=%s produced_audio=%s marked_interrupted=%s",
                    "done_callback",
                    done_id,
                    was_stale,
                    was_active,
                    produced_audio,
                    effective_interrupted,
                )
                speaking_at = agent_speaking_at.get(done_id)
                listening_at = agent_listening_at.get(done_id)
                speech_duration_seconds = (finished_at - started_at) if started_at is not None else -1.0
                agent_speaking_to_finished_seconds = (finished_at - speaking_at) if speaking_at is not None else -1.0
                finish_to_agent_listening_seconds = (
                    listening_at - finished_at if listening_at is not None else -1.0
                )
                assistant_speech_finished_at[done_id] = finished_at
                hume_request_count_at_speech_finish[done_id] = _hume_tts_request_counter
                speech_latency_audit = speech_latency_audits.pop(done_id, None)
                done_speech_turn_id = assistant_speech_turn_ids.pop(done_id, 0)
                done_speech_llm_turn_id = assistant_speech_llm_turn_ids.pop(done_id, 0)
                if was_active and not was_stale:
                    pending_user_handoff_speech_id = done_id
                _interaction_state.on_assistant_speech_finished(interrupted=effective_interrupted, speech_id=done_id)
                logger.info(
                    "Assistant speech finished: current_user_turn_id=%s speech_id=%s speech_turn_id=%s llm_turn_id=%s interrupted=%s handle_interrupted=%s fsm_observed_interrupted=%s active_count=%s was_suppressed=%s was_stale=%s was_active=%s",
                    _current_turn_id,
                    done_id,
                    done_speech_turn_id or "unknown",
                    done_speech_llm_turn_id or "unknown",
                    effective_interrupted,
                    interrupted,
                    fsm_observed_interrupted,
                    len(active_speech_handles),
                    was_suppressed,
                    was_stale,
                    was_active,
                )
                interrupted_normalized = str(interrupted).strip().lower()
                if was_active and active_speech_handles:
                    logger.error(
                        "Assistant speech active_count invariant violation after finish: finished_speech_id=%s active_count=%s active_speech_ids=%s cleanup_skipped_reason=%s",
                        done_id,
                        len(active_speech_handles),
                        list(active_speech_handles.keys()),
                        "cleanup_only_allowed_before_new_assistant_speech",
                    )
                elif was_active and interrupted_normalized in {"true", "1", "yes"} and active_speech_handles:
                    logger.warning(
                        "Assistant speech interrupted while another speech is active; cleanup skipped outside new-speech path: speech_id=%s active_count=%s active_speech_ids=%s",
                        done_id,
                        len(active_speech_handles),
                        list(active_speech_handles.keys()),
                    )
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
                hume_coverage = _hume_speech_audio_coverages.pop(done_id, None)
                generated_duration = None
                generated_bytes = getattr(hume_coverage, "byte_count", None) if hume_coverage is not None else None
                if hume_coverage is not None:
                    generated_duration = _hume_generated_audio_duration_seconds(hume_coverage)
                    playout_duration = speech_duration_seconds if speech_duration_seconds >= 0 else None
                    logger.info(
                        "hume_speech_playout_coverage=true speech_id=%s turn_id=%s tts_path=%s generated_audio_duration_seconds=%s assistant_playout_duration_seconds=%s generated_minus_playout_seconds=%s generated_frame_count=%s generated_audio_bytes=%s interrupted=%s was_suppressed=%s wav_artifact_path=%s",
                        done_id,
                        done_speech_turn_id or "unknown",
                        hume_coverage.path,
                        _fmt_seconds(generated_duration),
                        _fmt_seconds(playout_duration),
                        _fmt_seconds((generated_duration - playout_duration) if generated_duration is not None and playout_duration is not None else None),
                        hume_coverage.frame_count,
                        hume_coverage.byte_count,
                        interrupted,
                        was_suppressed,
                        hume_coverage.artifact_path or "none",
                    )
                # Tail outcome: classify what actually happened from playout timing +
                # audio lifecycle, instead of treating every interrupted=True as a
                # cutoff. Audio fully played (playout >= generated) means an
                # interruption landed after the tail, not a real cut.
                _audio_fully_played = (
                    generated_duration is not None
                    and speech_duration_seconds is not None
                    and speech_duration_seconds >= 0
                    and (speech_duration_seconds + 0.1) >= generated_duration
                )
                _tail_outcome = classify_tail_outcome(
                    generated_audio_duration_s=generated_duration,
                    playout_started_at=speaking_at,
                    playout_completed_at=(
                        listening_at if (not effective_interrupted or _audio_fully_played) else None
                    ),
                    interrupted_at=(listening_at if effective_interrupted else None),
                    interrupted=bool(effective_interrupted),
                    was_stale=bool(was_stale),
                    was_active=bool(was_active),
                    hume_requests_during_speech=during_count if during_count and during_count > 0 else 0,
                )
                logger.info(
                    "assistant_speech_tail_outcome speech_id=%s generated_audio_duration_seconds=%s "
                    "playout_started_at=%s playout_completed_at=%s interrupted_at=%s interrupted=%s "
                    "fsm_observed_interrupted=%s handle_interrupted=%s was_stale=%s was_active=%s "
                    "hume_requests_during_speech=%s tail_outcome=%s audible_cutoff=%s",
                    done_id,
                    _fmt_seconds(generated_duration),
                    _fmt_seconds(speaking_at) if speaking_at is not None else "none",
                    _fmt_seconds(listening_at) if listening_at is not None else "none",
                    _fmt_seconds(listening_at) if effective_interrupted and listening_at is not None else "none",
                    effective_interrupted,
                    fsm_observed_interrupted,
                    interrupted,
                    was_stale,
                    was_active,
                    during_count,
                    _tail_outcome,
                    is_audible_cutoff(_tail_outcome),
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
                # Outcome-driven ledger reconciliation: if this assistant turn was
                # not actually audible, downgrade its provisional ledger entry so it
                # cannot pollute future prompt context.
                if CONTEXT_POLICY_ENABLED:
                    generated_bytes = getattr(hume_coverage, "byte_count", None) if hume_coverage is not None else None
                    downgrade_reason = _ledger_downgrade_reason_for_outcome(
                        was_suppressed=was_suppressed or was_stale,
                        interrupted=effective_interrupted,
                        generated_bytes=generated_bytes,
                        playout_seconds=speech_duration_seconds,
                    )
                    if downgrade_reason is not None:
                        _ledger_downgrade_for_outcome(
                            turn_id=done_speech_turn_id or done_speech_llm_turn_id or 0,
                            speech_id=done_id,
                            reason=downgrade_reason,
                        )
                    else:
                        logger.info(
                            "ledger_outcome_downgrade_applied=false speech_id=%s ledger_entry_turn_id=%s",
                            done_id,
                            done_speech_turn_id or "n/a",
                        )
                start_count = hume_request_count_at_speech_start.get(done_id, -1)
                finish_count = hume_request_count_at_speech_finish.get(done_id, _hume_tts_request_counter)
                hume_requests_during = finish_count - start_count if start_count >= 0 and finish_count >= 0 else -1
                interruption_at = active_speech_interrupted_at.pop(done_id, None)
                tail_outcome = _classify_assistant_tail_outcome(
                    interrupted=bool(effective_interrupted),
                    interruption_at=interruption_at,
                    playout_started_at=speaking_at,
                    playout_completed_at=finished_at if produced_audio else None,
                    generated_audio_duration_seconds=generated_duration,
                    hume_requests_during_speech=hume_requests_during,
                )
                playout_duration_for_report = speech_duration_seconds if speech_duration_seconds >= 0 else None
                last_assistant_speech_outcome = {
                    "previous_speech_id": done_id,
                    "generated_audio_duration_seconds": generated_duration,
                    "playout_duration_seconds": playout_duration_for_report,
                    "interrupted": bool(effective_interrupted),
                    **tail_outcome,
                }
                logger.info(
                    "assistant_tail_diagnostic speech_id=%s generated_audio_duration_seconds=%s playout_duration_seconds=%s interrupted=%s interruption_before_playout_complete=%s interruption_after_playout_complete=%s assistant_playout_completed_normally=%s assistant_tail_cut_likely=%s interruption_timing=%s suppressed_or_ghost_handle=%s",
                    done_id,
                    _fmt_seconds(generated_duration),
                    _fmt_seconds(playout_duration_for_report),
                    bool(effective_interrupted),
                    tail_outcome["interruption_before_playout_complete"],
                    tail_outcome["interruption_after_playout_complete"],
                    tail_outcome["assistant_playout_completed_normally"],
                    tail_outcome["assistant_tail_cut_likely"],
                    tail_outcome["interruption_timing"],
                    tail_outcome["suppressed_or_ghost_handle"],
                )
                base_audit = speech_latency_audit or {}
                latency_audit = _build_voice_latency_audit(
                    turn_id=int(base_audit.get("turn_id", _current_turn_id) or _current_turn_id),
                    speech_id=done_id,
                    user_speech_started_at=base_audit.get("user_speech_started_at", last_user_speaking_at),
                    user_speech_stopped_at=base_audit.get("user_speech_stopped_at", last_user_listening_at),
                    final_stt_received_at=base_audit.get("final_stt_received_at", last_stt_final_at),
                    user_turn_committed_at=base_audit.get("user_turn_committed_at", _last_turn_committed_at),
                    llm_request_started_at=base_audit.get("llm_request_started_at", _last_llm_start_at),
                    llm_first_token_at=base_audit.get("llm_first_token_at", _last_llm_first_token_at),
                    llm_completed_at=base_audit.get("llm_completed_at", _last_llm_complete_at),
                    tts_request_started_at=base_audit.get("tts_request_started_at", _last_tts_request_start_at),
                    tts_first_audio_at=base_audit.get("tts_first_audio_at", _last_tts_first_audio_at),
                    tts_completed_at=_last_tts_completed_at,
                    assistant_playout_started_at=speaking_at or base_audit.get("assistant_playout_started_at"),
                    assistant_playout_completed_at=finished_at,
                )
                logger.info(
                    "Voice latency audit: turn_id=%s speech_id=%s user_speech_started_at=%s user_speech_stopped_at=%s final_stt_received_at=%s user_turn_committed_at=%s llm_request_started_at=%s llm_first_token_at=%s llm_completed_at=%s tts_request_started_at=%s tts_first_audio_at=%s tts_completed_at=%s assistant_playout_started_at=%s assistant_playout_completed_at=%s user_stopped_to_final_stt=%s final_stt_to_turn_committed=%s turn_committed_to_llm_first_token=%s llm_first_token_to_llm_complete=%s llm_complete_to_tts_request=%s tts_request_to_first_audio=%s tts_first_audio_to_playout_start=%s user_stopped_to_first_audio=%s user_stopped_to_assistant_complete=%s endpointing_min_delay=%s endpointing_max_delay=%s mistral_target_streaming_delay_ms=%s spoken_text_normalization_enabled=%s spoken_text_normalization_mode=%s tts_input_buffering_mode=%s raw_chunk_count=%s normalized_yield_count=%s time_from_llm_first_token_to_first_tts_input=%s tts_provider=%s hume_instant_mode=%s hume_model_version=%s hume_speed=%s openrouter_model=%s llm_stream_status=%s llm_timeout_stage=%s llm_fallback_response_used=%s text_length=%s sentence_end_count=%s hume_requests_during_speech=%s tts_path=%s description_applied=%s",
                    latency_audit["turn_id"],
                    done_id,
                    _fmt_seconds(latency_audit.get("user_speech_started_at")),
                    _fmt_seconds(latency_audit.get("user_speech_stopped_at")),
                    _fmt_seconds(latency_audit.get("final_stt_received_at")),
                    _fmt_seconds(latency_audit.get("user_turn_committed_at")),
                    _fmt_seconds(latency_audit.get("llm_request_started_at")),
                    _fmt_seconds(latency_audit.get("llm_first_token_at")),
                    _fmt_seconds(latency_audit.get("llm_completed_at")),
                    _fmt_seconds(latency_audit.get("tts_request_started_at")),
                    _fmt_seconds(latency_audit.get("tts_first_audio_at")),
                    _fmt_seconds(latency_audit.get("tts_completed_at")),
                    _fmt_seconds(latency_audit.get("assistant_playout_started_at")),
                    _fmt_seconds(latency_audit.get("assistant_playout_completed_at")),
                    _fmt_seconds(latency_audit.get("user_stopped_to_final_stt")),
                    _fmt_seconds(latency_audit.get("final_stt_to_turn_committed")),
                    _fmt_seconds(latency_audit.get("turn_committed_to_llm_first_token")),
                    _fmt_seconds(latency_audit.get("llm_first_token_to_llm_complete")),
                    _fmt_seconds(latency_audit.get("llm_complete_to_tts_request")),
                    _fmt_seconds(latency_audit.get("tts_request_to_first_audio")),
                    _fmt_seconds(latency_audit.get("tts_first_audio_to_playout_start")),
                    _fmt_seconds(latency_audit.get("user_stopped_to_first_audio")),
                    _fmt_seconds(latency_audit.get("user_stopped_to_assistant_complete")),
                    os.getenv("ENDPOINTING_MIN_DELAY_SECONDS", os.getenv("LIVEKIT_ENDPOINTING_MIN_DELAY", "1.1")),
                    os.getenv("ENDPOINTING_MAX_DELAY_SECONDS", os.getenv("LIVEKIT_ENDPOINTING_MAX_DELAY", "2.4")),
                    os.getenv("MISTRAL_TARGET_STREAMING_DELAY_MS", "160"),
                    SPOKEN_TEXT_NORMALIZATION,
                    SPOKEN_TEXT_NORMALIZATION_MODE,
                    SPOKEN_TEXT_NORMALIZATION_MODE,
                    _last_tts_raw_chunk_count,
                    _last_tts_normalized_yield_count,
                    _fmt_seconds((_last_tts_first_input_at - _last_llm_first_token_at) if (_last_tts_first_input_at is not None and _last_llm_first_token_at is not None) else None),
                    TTS_PROVIDER,
                    env_bool("HUME_INSTANT_MODE", True),
                    _last_hume_model_version,
                    os.getenv("HUME_SPEED", "0.9"),
                    os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL),
                    _last_llm_stream_status,
                    _last_llm_timeout_stage,
                    _last_llm_fallback_response_used,
                    _last_tts_text_length,
                    _last_tts_sentence_end_count,
                    hume_requests_during,
                    _last_tts_path or "n/a",
                    _last_hume_description_applied,
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
        global _latest_agent_state_for_hume, _latest_current_speech_id_for_hume, _latest_active_assistant_count_for_hume
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
            # Real audio playout is starting now (not at speech-object creation):
            # this is where the FSM is allowed to enter ASSISTANT_SPEAKING.
            _interaction_state.on_assistant_speech_started(current_speech_id)
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
            # Reconcile the FSM: a speech object that was created but never reached
            # real audio playout (suppressed / zero-audio) leaves the FSM in an
            # assistant-active state. Now that the agent is idle with no speech,
            # return it to LISTENING so downstream logic does not believe the
            # assistant is still thinking/speaking.
            if _interaction_state.state in (ASSISTANT_THINKING, TOOL_CALL_PENDING, ASSISTANT_SPEAKING):
                _interaction_state.transition(
                    LISTENING, reason="agent_idle_no_active_speech_reconcile"
                )

    @session.on("user_state_changed")
    def _on_user_state_changed(state: object) -> None:
        nonlocal latest_user_state, latest_user_state_timestamp, pending_user_handoff_speech_id, last_user_speaking_at, last_user_listening_at
        global _latest_user_state_for_greeting, _latest_user_state_changed_at, _latest_user_speaking_at
        global _barge_in_during_thinking_turn_id, _barge_in_started_at, _barge_in_confirmed_real
        latest_user_state = _extract_user_new_state(state)
        latest_user_state_timestamp = time.monotonic()
        _latest_user_state_for_greeting = latest_user_state
        _latest_user_state_changed_at = latest_user_state_timestamp
        if latest_user_state == "speaking":
            last_user_speaking_at = latest_user_state_timestamp
            _latest_user_speaking_at = latest_user_state_timestamp
        if latest_user_state == "listening":
            last_user_listening_at = latest_user_state_timestamp
        if latest_user_state == "speaking":
            pre_state = _interaction_state.state
            interrupted_ids = set(active_speech_handles.keys())
            active_fsm_speech_id = getattr(_interaction_state, "active_speech_id", None)
            if active_fsm_speech_id:
                interrupted_ids.add(active_fsm_speech_id)
            current_speech = getattr(session, "current_speech", None)
            if current_speech is not None:
                interrupted_ids.add(_speech_id(current_speech))
            for interrupted_id in interrupted_ids:
                active_speech_interrupted_at.setdefault(interrupted_id, latest_user_state_timestamp)
            _record_audio_env_event("speech_start", latest_user_state_timestamp)
            _interaction_state.on_user_speech_started()
            # Latch a barge-in that begins while the assistant is still thinking
            # (no audio yet) so a stale pending reply can be suppressed before TTS.
            # Once audio has started (ASSISTANT_SPEAKING) the normal interruption
            # path owns it, so we do not latch then.
            if LLM_TO_TTS_HANDOFF_GUARD_ENABLED and pre_state in (ASSISTANT_THINKING, TOOL_CALL_PENDING):
                _barge_in_during_thinking_turn_id = _current_turn_id
                _barge_in_started_at = latest_user_state_timestamp
                _barge_in_confirmed_real = False
                logger.info(
                    "handoff_guard_barge_in_observed=true turn_id=%s pre_state=%s",
                    _current_turn_id,
                    pre_state,
                )
        elif latest_user_state == "listening":
            _record_audio_env_event("turn_candidate", latest_user_state_timestamp)
            _interaction_state.on_user_speech_stopped()
            # Confirm the barge-in as a real utterance only if it was sustained;
            # discard brief VAD blips so a cough never suppresses a wanted reply.
            if (
                LLM_TO_TTS_HANDOFF_GUARD_ENABLED
                and _barge_in_during_thinking_turn_id == _current_turn_id
                and _barge_in_started_at > 0
            ):
                speech_ms = (latest_user_state_timestamp - _barge_in_started_at) * 1000.0
                if speech_ms >= HANDOFF_GUARD_MIN_SPEECH_MS:
                    _barge_in_confirmed_real = True
                    logger.info(
                        "handoff_guard_barge_in_confirmed=true turn_id=%s speech_ms=%.0f min_ms=%s",
                        _current_turn_id,
                        speech_ms,
                        HANDOFF_GUARD_MIN_SPEECH_MS,
                    )
                else:
                    logger.info(
                        "handoff_guard_barge_in_discarded=true turn_id=%s speech_ms=%.0f min_ms=%s reason=below_min_speech",
                        _current_turn_id,
                        speech_ms,
                        HANDOFF_GUARD_MIN_SPEECH_MS,
                    )
                    _barge_in_during_thinking_turn_id = 0
                    _barge_in_started_at = 0.0
                    _barge_in_confirmed_real = False
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
        # Phantom-message guard + ledger: record user/assistant turns, marking an
        # interrupted or empty assistant turn as suppressed/non-canonical so it
        # never gets injected as future context.
        role_l = str(role).strip().lower()
        if CONTEXT_POLICY_ENABLED and role_l in ("user", "assistant"):
            ledger_text = _extract_text_for_debug(target)
            interrupted_bool = str(interrupted).strip().lower() in {"true", "1", "yes"}
            is_suppressed = role_l == "assistant" and (interrupted_bool or not ledger_text.strip())
            # speech_id is usually not exposed on the conversation item, which
            # broke outcome reconciliation (turn_id alone drifts when the speech's
            # turn differs from the live turn). Correlate the assistant entry with
            # the live speech so the later downgrade can match on speech_id — the
            # stable owner key — instead of a drifting turn_id.
            entry_speech_id = _safe_attr(target, "speech_id", None) or None
            if entry_speech_id is None and role_l == "assistant":
                current_speech = getattr(session, "current_speech", None)
                if current_speech is not None:
                    entry_speech_id = _speech_id(current_speech)
                elif active_speech_handles:
                    entry_speech_id = next(reversed(active_speech_handles), None)
                if entry_speech_id is not None:
                    logger.info(
                        "ledger_entry_speech_id_correlated=true ledger_entry_speech_id=%s ledger_entry_turn_id=%s source=%s",
                        entry_speech_id,
                        _current_turn_id,
                        "session_current_speech" if current_speech is not None else "active_speech_handles",
                    )
            # Runtime gate (enforce, not just observe): an assistant turn whose
            # owning user commit had no valid speech/candidate owner must not
            # become canonical context, even though it may still be shown/heard.
            canonical_block_reason = "none"
            if role_l == "assistant" and not is_suppressed and ledger_text.strip():
                if not _current_turn_owner_valid:
                    canonical_block_reason = "turn_commit_no_owner"
                if canonical_block_reason != "none":
                    _interaction_state.runtime_gate(
                        "assistant_canonical_write", False, reason=canonical_block_reason
                    )
                    logger.info(
                        "canonical_write_blocked_reason=%s ledger_entry_turn_id=%s ledger_entry_speech_id=%s",
                        canonical_block_reason,
                        _current_turn_id,
                        entry_speech_id or "n/a",
                    )
            _ledger_append(
                role_l,
                ledger_text,
                visible=not is_suppressed,
                suppressed=is_suppressed,
                turn_id=_current_turn_id,
                speech_id=entry_speech_id,
                provisional=role_l == "assistant",
                block_canonical=canonical_block_reason != "none",
                canonical_block_reason=canonical_block_reason,
            )
            if role_l == "assistant":
                logger.info(
                    "ledger_entry_provisional=true ledger_entry_speech_id=%s ledger_entry_turn_id=%s canonical_for_context=%s canonical_write_blocked_reason=%s",
                    entry_speech_id or "n/a",
                    _current_turn_id,
                    not is_suppressed and bool(ledger_text.strip()) and canonical_block_reason == "none",
                    canonical_block_reason,
                )
        if _active_memory_layer is not None and str(role).strip().lower() == "assistant":
            _active_memory_layer.schedule_remember(
                role="assistant",
                content=_extract_text_for_debug(target),
                turn_id=_current_turn_id,
            )
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
        if str(role).strip().lower() == "user":
            feedback = _user_feedback_marker(_extract_text_for_debug(target))
            if feedback and last_assistant_speech_outcome is not None:
                logger.info(
                    "assistant_tail_user_feedback_report user_feedback=%s previous_speech_id=%s generated_audio_duration_seconds=%s playout_duration_seconds=%s interrupted=%s interruption_timing=%s assistant_tail_cut_likely=%s",
                    feedback,
                    last_assistant_speech_outcome.get("previous_speech_id", "n/a"),
                    _fmt_seconds(last_assistant_speech_outcome.get("generated_audio_duration_seconds")),
                    _fmt_seconds(last_assistant_speech_outcome.get("playout_duration_seconds")),
                    last_assistant_speech_outcome.get("interrupted"),
                    last_assistant_speech_outcome.get("interruption_timing"),
                    last_assistant_speech_outcome.get("assistant_tail_cut_likely"),
                )
            _reset_search_state_for_turn()
            logger.info("Search state reset for new user turn: search_in_progress=%s search_tool_called=%s", _search_in_progress, _search_tool_called)

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event: object) -> None:
        nonlocal stt_partial_count, stt_final_count, last_stt_any_at, last_stt_final_at, last_stt_preview, last_stt_final_preview
        global _latest_stt_partial_at, _latest_stt_partial_text_hash, _latest_stt_final_at
        global _latest_stt_language, _latest_stt_language_confidence
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
            _latest_stt_final_at = last_stt_any_at
            last_stt_final_preview = last_stt_preview
            # Carry the engine-reported language candidate/confidence (if any) to
            # the turn-commit log. No new detection — just what STT already emits.
            _latest_stt_language = str(language) if language not in (None, "") else "n/a"
            stt_conf = _safe_attr(event, "language_confidence", _safe_attr(event, "confidence", None))
            _latest_stt_language_confidence = f"{float(stt_conf):.2f}" if isinstance(stt_conf, (int, float)) else "n/a"
            candidate_id = _record_stt_final_candidate(
                transcript_str,
                user_state=str(_latest_user_state_for_greeting or "unknown"),
                partial_count=stt_partial_count,
            )
            logger.info(
                "stt_final_candidate stt_segment_id=%s text_hash=%s user_state=%s transcript_length=%s",
                candidate_id,
                _text_hash(transcript_str.strip()),
                _latest_user_state_for_greeting or "unknown",
                len(transcript_str),
            )
        else:
            stt_partial_count += 1
            _latest_stt_partial_at = last_stt_any_at
            _latest_stt_partial_text_hash = _text_hash(transcript_str)
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
        if "llm" in searchable_safe_text:
            logger.error(
                "LLM error diagnostic: details=%s openrouter_api_key_present=%s openrouter_model=%s",
                _safe_llm_error_details(error),
                bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
                os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL),
            )
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
                    os.getenv("LIVEKIT_ENDPOINTING_MODE", "dynamic"),
                    os.getenv("ENDPOINTING_MIN_DELAY_SECONDS", os.getenv("LIVEKIT_ENDPOINTING_MIN_DELAY", "1.1")),
                    os.getenv("ENDPOINTING_MAX_DELAY_SECONDS", os.getenv("LIVEKIT_ENDPOINTING_MAX_DELAY", "2.4")),
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


def env_int_clamped(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except Exception:
        return default
    return max(min_value, min(max_value, value))



AI_COUSTICS_ENABLED = env_bool("AI_COUSTICS_ENABLED", True)
# Hold replies on clearly cut-off user fragments while the user is still talking,
# instead of answering mid-sentence (the "you're interrupting me" failure mode).
TURN_HOLD_INCOMPLETE_FRAGMENTS = env_bool("TURN_HOLD_INCOMPLETE_FRAGMENTS", True)
TURN_HOLD_MAX_CONSECUTIVE = env_int_clamped("TURN_HOLD_MAX_CONSECUTIVE", 2, 0, 5)
TURN_HOLD_FRAGMENT_TTL_SECONDS = float(os.getenv("TURN_HOLD_FRAGMENT_TTL_SECONDS", "60"))
SPEECH_TRACKING_TTL_SECONDS = float(os.getenv("SPEECH_TRACKING_TTL_SECONDS", "120"))
SPOKEN_TEXT_NORMALIZATION = env_bool("SPOKEN_TEXT_NORMALIZATION", False)
TTS_TEXT_DEBUG = env_bool("TTS_TEXT_DEBUG", False)
PIPELINE_TEXT_DEBUG = env_bool("PIPELINE_TEXT_DEBUG", False)
MISTRAL_STT_DIAGNOSTICS = env_bool("MISTRAL_STT_DIAGNOSTICS", True)
HUME_FULL_UTTERANCE_TTS = env_bool("HUME_FULL_UTTERANCE_TTS", False)
LIVEKIT_TTS_SOURCE_INSPECTION = env_bool("LIVEKIT_TTS_SOURCE_INSPECTION", False)
HUME_DIRECT_API_TTS = env_bool("HUME_DIRECT_API_TTS", False)
# Context-coherence layer (Phase 1, deterministic): judge whether the committed
# transcript plausibly fits the session before the main LLM responds. When low,
# the response is shaped to acknowledge-and-attempt (never silenced/blocked).
# Off by default; behavior is unchanged until enabled.
CONTEXT_COHERENCE_ENABLED = env_bool("CONTEXT_COHERENCE_ENABLED", False)
try:
    CONTEXT_COHERENCE_MIN_ASR_CONFIDENCE = max(0.0, min(1.0, float(os.getenv("CONTEXT_COHERENCE_MIN_ASR_CONFIDENCE", "0.45"))))
except Exception:
    CONTEXT_COHERENCE_MIN_ASR_CONFIDENCE = 0.45
# Context-policy layer: let the prior-context signals the system already detects
# govern prompt injection, response posture, and timeout fallback. Reuses the
# deterministic context classification + turn policy; adds no new model/memory.
CONTEXT_POLICY_ENABLED = env_bool("CONTEXT_POLICY_ENABLED", True)
CONTEXT_FORCE_LOCAL_INJECTION = env_bool("CONTEXT_FORCE_LOCAL_INJECTION", True)
CONTEXT_INJECTION_TURNS = env_int_clamped("CONTEXT_INJECTION_TURNS", 5, 1, 20)
CONTEXT_RESPONSE_POSTURE_ENABLED = env_bool("CONTEXT_RESPONSE_POSTURE_ENABLED", True)
CONTEXT_AWARE_FALLBACK_ENABLED = env_bool("CONTEXT_AWARE_FALLBACK_ENABLED", True)
CONTEXT_REFERENCE_CARRY_FORWARD_ENABLED = env_bool("CONTEXT_REFERENCE_CARRY_FORWARD_ENABLED", True)
CONVERSATION_LEDGER_EXCLUDE_SUPPRESSED = env_bool("CONVERSATION_LEDGER_EXCLUDE_SUPPRESSED", True)
LOG_PROMPT_CONTEXT_INJECTION = env_bool("LOG_PROMPT_CONTEXT_INJECTION", True)
LLM_RETRY_ON_FIRST_TOKEN_TIMEOUT = env_bool("LLM_RETRY_ON_FIRST_TOKEN_TIMEOUT", True)
LLM_FALLBACK_MODEL = (os.getenv("LLM_FALLBACK_MODEL") or "").strip()
CONTEXT_AWARE_FALLBACK_TEXT = "I know you’re pointing back to what just happened — give me a second."


def _context_aware_fallback_text(reason: str, context_dependency: str) -> str | None:
    """Context-preserving fallback for a timeout on a context-dependent turn."""
    if (
        CONTEXT_AWARE_FALLBACK_ENABLED
        and context_dependency == "high"
        and reason in {"first_token_timeout", "total_timeout_no_text"}
    ):
        return CONTEXT_AWARE_FALLBACK_TEXT
    return None


def _assess_context_coherence(stt_language: str, stt_language_confidence: str, session_language: str) -> tuple[str, str, float]:
    """Deterministic Phase-1 coherence check at turn commit.

    Returns (context_fit, reason, confidence): context_fit is "coherent",
    "low_coherence", or "unknown". Catches the cheap/obvious cases (a transcript
    in a different language than the session, or very low ASR confidence) with no
    added latency. Nuanced non-sequitur/topic-fit is deferred to the Tier-2 LLM
    interpreter (Phase 2). Returns "unknown" when disabled so nothing downstream
    changes.
    """
    if not CONTEXT_COHERENCE_ENABLED:
        return "unknown", "disabled", 0.0
    lang = (stt_language or "n/a").strip().lower()
    session = (session_language or "en").strip().lower()
    if lang not in ("", "n/a", "unknown") and not lang.startswith(session) and not session.startswith(lang):
        return "low_coherence", f"language_mismatch:{lang}", 0.8
    try:
        conf = float(stt_language_confidence)
    except (TypeError, ValueError):
        conf = -1.0
    if 0.0 <= conf < CONTEXT_COHERENCE_MIN_ASR_CONFIDENCE:
        return "low_coherence", f"asr_low_confidence:{conf:.2f}", 0.7
    return "coherent", "ok", 0.6


def _ledger_append(
    role: str,
    text: str,
    *,
    visible: bool,
    suppressed: bool,
    turn_id: int | None = None,
    speech_id: str | None = None,
    provisional: bool = False,
    block_canonical: bool = False,
    canonical_block_reason: str = "none",
) -> dict:
    """Record a turn in the thin conversation ledger.

    canonical_for_context gates prompt injection: a turn is canonical only when it
    is visible, not suppressed, has real text, and is not gated by a runtime
    canonical block (e.g. an owner-less commit). Assistant entries are added
    provisionally and may be downgraded later by the outcome reconciliation once
    the true audio result is known. Returns the appended entry.
    """
    canonical = bool(visible and not suppressed and (text or "").strip()) and not block_canonical
    entry = {
        "role": role,
        "text": text or "",
        "visible_to_user": bool(visible),
        "suppressed": bool(suppressed),
        "canonical_for_context": canonical,
        "canonical_block_reason": canonical_block_reason if block_canonical else "none",
        "turn_id": turn_id,
        "speech_id": speech_id,
        "provisional": bool(provisional),
    }
    _conversation_ledger.append(entry)
    if len(_conversation_ledger) > 60:
        del _conversation_ledger[:-60]
    return entry


def _ledger_recent_canonical(n: int) -> list[dict]:
    if CONVERSATION_LEDGER_EXCLUDE_SUPPRESSED:
        items = [
            entry
            for entry in _conversation_ledger
            if entry.get("canonical_for_context") and entry.get("visible_to_user") and not entry.get("suppressed")
        ]
    else:
        items = list(_conversation_ledger)
    return items[-n:] if n > 0 else []


def _effective_interruption(handle_interrupted: object, fsm_observed_interrupted: bool) -> bool:
    """Combine the TTS handle's interrupted flag with the FSM's observation.

    The LiveKit handle sometimes reports interrupted=False even when the user
    clearly barged in (the FSM logged active_speech_marked_interrupted=true). The
    FSM's observation is authoritative for ledger ownership, so an interruption
    from EITHER source counts.
    """
    return str(handle_interrupted).strip().lower() in {"true", "1", "yes"} or bool(fsm_observed_interrupted)


def _effective_interruption_for_speech(
    handle_interrupted: object,
    fsm_observed_interrupted: bool,
    *,
    was_stale: bool,
    produced_audio: bool,
) -> bool:
    """Return true only for real interruptions, not stale zero-audio reconciliation."""
    if was_stale and not produced_audio:
        return False
    return _effective_interruption(handle_interrupted, fsm_observed_interrupted)


def _classify_assistant_tail_outcome(
    *,
    interrupted: bool,
    interruption_at: float | None,
    playout_started_at: float | None,
    playout_completed_at: float | None,
    generated_audio_duration_seconds: float | None,
    hume_requests_during_speech: int | None,
) -> dict[str, object]:
    hume_request_count_allows_audio = hume_requests_during_speech is None or hume_requests_during_speech != 0
    produced_hume_audio = hume_request_count_allows_audio and bool(
        generated_audio_duration_seconds and generated_audio_duration_seconds > 0
    )
    if not produced_hume_audio:
        return {
            "interruption_before_playout_complete": False,
            "interruption_after_playout_complete": False,
            "assistant_playout_completed_normally": False,
            "assistant_tail_cut_likely": False,
            "interruption_timing": "none",
            "suppressed_or_ghost_handle": True,
        }
    estimated_generated_end = None
    if playout_started_at is not None and generated_audio_duration_seconds is not None:
        estimated_generated_end = playout_started_at + generated_audio_duration_seconds
    completion_boundary = playout_completed_at if playout_completed_at is not None else estimated_generated_end
    interruption_before = False
    interruption_after = False
    timing = "none"
    if interrupted and interruption_at is not None and completion_boundary is not None:
        # Small tolerance prevents a final VAD edge at the exact end from being
        # misread as a user-facing tail cut.
        if interruption_at < (completion_boundary - 0.05):
            interruption_before = True
            timing = "before_playout_complete"
        else:
            interruption_after = True
            timing = "after_playout_complete"
    elif interrupted:
        timing = "before_playout_complete" if playout_completed_at is None else "after_playout_complete"
        interruption_before = timing == "before_playout_complete"
        interruption_after = timing == "after_playout_complete"
    completed_normally = produced_hume_audio and not interruption_before
    return {
        "interruption_before_playout_complete": interruption_before,
        "interruption_after_playout_complete": interruption_after,
        "assistant_playout_completed_normally": completed_normally,
        "assistant_tail_cut_likely": interruption_before,
        "interruption_timing": timing,
        "suppressed_or_ghost_handle": False,
    }


def _user_feedback_marker(text: str) -> str | None:
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    clean_patterns = (
        "no cutoff",
        "no cut off",
        "ended clean",
        "ended cleanly",
        "didn't sound like a tail",
        "did not sound like a tail",
        "wasn't cut off",
        "was not cut off",
    )
    if any(pattern in lowered for pattern in clean_patterns):
        return "clean"
    cutoff_patterns = ("hard clip", "cutoff", "cut off", "tail response", "tail end")
    if any(pattern in lowered for pattern in cutoff_patterns):
        return "cutoff"
    return None


def _ledger_downgrade_reason_for_outcome(
    *,
    was_suppressed: bool,
    interrupted: object,
    generated_bytes: int | None,
    playout_seconds: float | None,
) -> str | None:
    """Pick a downgrade reason from the true audio outcome, or None if audible."""
    if was_suppressed:
        return "suppressed"
    if str(interrupted).strip().lower() in {"true", "1", "yes"}:
        return "interrupted"
    if generated_bytes == 0:
        return "zero_audio"
    if playout_seconds is not None and playout_seconds < 0:
        return "no_playout"
    return None


def _ledger_downgrade_for_outcome(*, turn_id: int | None, speech_id: str | None, reason: str) -> str:
    """Reconcile an assistant ledger entry against its true audio outcome.

    Matching: prefer speech_id, else the most recent assistant entry for the same
    turn_id. Never guesses across turns. Returns the match strategy used
    ("speech_id" | "turn_id_recent" | "failed").
    """
    target = None
    strategy = "failed"
    if speech_id:
        for entry in reversed(_conversation_ledger):
            if entry.get("role") == "assistant" and entry.get("speech_id") == speech_id:
                target, strategy = entry, "speech_id"
                break
    if target is None and turn_id:
        for entry in reversed(_conversation_ledger):
            if entry.get("role") == "assistant" and entry.get("turn_id") == turn_id:
                target, strategy = entry, "turn_id_recent"
                break
    if target is None:
        logger.warning(
            "ledger_downgrade_match_failed=true ledger_downgrade_match_strategy=failed ledger_downgrade_reason=%s ledger_entry_speech_id=%s ledger_entry_turn_id=%s",
            reason,
            speech_id or "n/a",
            turn_id if turn_id else "n/a",
        )
        return "failed"
    target["visible_to_user"] = False
    target["suppressed"] = True
    target["canonical_for_context"] = False
    target["provisional"] = False
    logger.info(
        "ledger_outcome_downgrade_applied=true ledger_downgrade_reason=%s ledger_downgrade_match_strategy=%s canonical_for_context=false ledger_entry_turn_id=%s ledger_entry_speech_id=%s",
        reason,
        strategy,
        target.get("turn_id") if target.get("turn_id") else "n/a",
        speech_id or "n/a",
    )
    return strategy


def _ledger_visible_canonical_count() -> int:
    return sum(1 for entry in _conversation_ledger if entry.get("canonical_for_context"))


def _ledger_suppressed_count() -> int:
    return sum(1 for entry in _conversation_ledger if entry.get("suppressed"))




def _inject_context_window_note(turn_ctx: object, response_posture: str, recent: list[dict]) -> int:
    """Force the recent visible exchange into the prompt for a context-dependent turn.

    Returns the number of turns injected (0 when nothing was injected).
    """
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message) or not recent:
        return 0
    lines = "\n".join(f"- {entry['role']}: {_redact_sensitive_text(entry['text'])[:200]}" for entry in recent)
    note = (
        "Recent conversation context:\n"
        "The user’s current message depends on the recent exchange.\n"
        "Use the recent visible turns below to understand what the user is pointing back to.\n"
        f"Response posture: {response_posture}.\n"
        "Do not answer as a generic standalone question.\n"
        "Do not ask the user to repeat unless the reference truly cannot be resolved.\n"
        f"Recent visible turns:\n{lines}"
    )
    try:
        add_message(role="developer", content=note)
        return len(recent)
    except Exception as exc:
        logger.warning(
            "Context window note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )
        return 0


_AUDIO_STATUS_CHECK_RE = re.compile(
    r"\b(can you (still )?hear me( now)?|do you hear me|are you (still )?there|"
    r"is it noisy|can you tell if it'?s noisy|how('?s| is) (my|the) audio|am i (coming through|breaking up))\b",
    re.IGNORECASE,
)


def _is_audio_status_check(text: str) -> bool:
    return bool(_AUDIO_STATUS_CHECK_RE.search((text or "").lower().replace("’", "'")))


def _audio_status_response_text(decision: AudioEnvironmentDecision) -> str:
    """Pick the audio-status reply that matches the measured environment."""
    if decision.noise_state == "noisy" and decision.transcript_stability == "unstable":
        return "I can hear parts of you, but the noise is making it harder to catch everything."
    if decision.noise_state == "noisy":
        return "I can hear you, but there’s background noise."
    if decision.noise_state == "uncertain":
        return "I can mostly hear you — if it’s noisy on your end, it might cut in and out."
    return "I can hear you clearly."


def _inject_audio_status_note(turn_ctx: object, status_line: str) -> bool:
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        return False
    note = (
        "Internal audio-status note. Do not reveal this note. "
        "The user is asking whether you can hear them / how the audio is. "
        f"Answer naturally and briefly with exactly this status: \"{status_line}\" "
        "Do not invent details beyond it."
    )
    try:
        add_message(role="developer", content=note)
        return True
    except Exception as exc:
        logger.warning("Audio-status note injection failed: error_type=%s error=%s", type(exc).__name__, _redact_sensitive_text(exc))
        return False


def _should_trigger_silence_recovery(classification: str | None, has_held_fragment: bool) -> bool:
    """A meta-complaint about silence/slowness with nothing held needs an explicit
    recovery note. (When a fragment IS held, the held-fragment recovery handles it.)
    """
    return (classification or "").strip().upper() == "META_COMPLAINT" and not has_held_fragment


def _inject_silence_recovery_note(turn_ctx: object) -> bool:
    """Tell Arche to acknowledge the silence/lag and re-engage, briefly and warmly.

    Used when the user complains that Arche went quiet (e.g. "you keep going
    silent") and there is no held fragment to point back to.
    """
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        return False
    note = (
        "Internal recovery note. Do not reveal this note. The user is frustrated "
        "that you went quiet or were slow to respond. Briefly and warmly acknowledge "
        "that you went quiet, do not over-explain or mention technical/latency reasons, "
        "and ask one short question to re-engage. Keep it to one or two sentences."
    )
    try:
        add_message(role="developer", content=note)
        return True
    except Exception as exc:
        logger.warning("Silence-recovery note injection failed: error_type=%s error=%s", type(exc).__name__, _redact_sensitive_text(exc))
        return False


def _record_audio_env_event(kind: str, now: float) -> None:
    bucket = _recent_user_speech_start_times if kind == "speech_start" else _recent_turn_candidate_times
    bucket.append(now)
    cutoff = now - _AUDIO_ENV_WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)


def _audio_env_recent_counts(now: float) -> tuple[int, int]:
    cutoff = now - _AUDIO_ENV_WINDOW_SECONDS
    starts = sum(1 for t in _recent_user_speech_start_times if t >= cutoff)
    candidates = sum(1 for t in _recent_turn_candidate_times if t >= cutoff)
    return starts, candidates


def _inject_coherence_note(turn_ctx: object, reason: str) -> bool:
    """Shape the next reply to acknowledge-and-attempt when the utterance may not fit.

    Never blocks or silences the turn; it only nudges the main LLM to flag a
    possible mishearing/language switch and continue.
    """
    note = (
        "Internal context note. Do not reveal this note or quote it. "
        f"The user's latest message may not fit the conversation so far (reason: {reason}). "
        "You may have misheard them, or they may have switched languages. "
        "Briefly acknowledge you might have misheard and offer your best understanding or a short check "
        "(for example: \"I might've misheard — did you mean…?\"), then continue naturally in English. "
        "Do not go silent and do not refuse to respond."
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("Coherence note could not be injected: turn_ctx_add_message_unavailable")
        return False
    try:
        add_message(role="developer", content=note)
        logger.info("coherence_note_injected=true reason=%s turn_id=%s", reason, _current_turn_id)
        return True
    except Exception as exc:
        logger.warning(
            "Coherence note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )
        return False


# Trailing silence (ms) appended after TTS audio so a client/WebRTC buffer drop at the
# speaking->listening transition trims silence instead of the final word/breath.
# 0 disables the hold (pure passthrough). Clamped to a sane ceiling.
TTS_POST_SPEECH_HOLD_MS = env_int_clamped("TTS_POST_SPEECH_HOLD_MS", 0, 0, 5000)
RUN_DB_MIGRATIONS_ON_STARTUP = env_bool("RUN_DB_MIGRATIONS_ON_STARTUP", False)

# --- Session time limit -------------------------------------------------------
# Hard cap on how long a single voice session runs. The worker speaks a short
# heads-up SESSION_ENDING_WARNING_SECONDS before the cap, then (optionally) says
# a brief goodbye and tears the room down so the client lands on the end screen.
# Set SESSION_TIME_LIMIT_ENABLED=false (or SESSION_MAX_DURATION_SECONDS=0) to
# disable. Times are seconds; default is a 7-minute session with a 30s warning.
SESSION_TIME_LIMIT_ENABLED = env_bool("SESSION_TIME_LIMIT_ENABLED", True)
SESSION_MAX_DURATION_SECONDS = float(os.getenv("SESSION_MAX_DURATION_SECONDS", "420") or "420")
SESSION_ENDING_WARNING_SECONDS = float(os.getenv("SESSION_ENDING_WARNING_SECONDS", "30") or "30")
SESSION_ENDING_WARNING_TEXT = (
    os.getenv("SESSION_ENDING_WARNING_TEXT")
    or "Hey, quick heads up — we've got about thirty seconds left for this one."
).strip()
SESSION_ENDING_GOODBYE_TEXT = (
    os.getenv("SESSION_ENDING_GOODBYE_TEXT")
    or "That's our time for now. Take care — talk soon."
).strip()
# The heads-up and goodbye are spoken through the same gate that drops any
# utterance created while the user is talking, so each waits out in-progress user
# speech before speaking. Reserve a little room before the hard cap for a
# late-spoken heads-up to finish, and cap how long the goodbye may wait for a
# pause (the room tears down right after it).
SESSION_ENDING_WARNING_PLAYBACK_RESERVE_SECONDS = float(
    os.getenv("SESSION_ENDING_WARNING_PLAYBACK_RESERVE_SECONDS", "6") or "6"
)
SESSION_ENDING_GOODBYE_CLEAR_WAIT_SECONDS = float(
    os.getenv("SESSION_ENDING_GOODBYE_CLEAR_WAIT_SECONDS", "3") or "3"
)

ENABLE_FIXED_GREETING = env_bool("ENABLE_FIXED_GREETING", True)
GREETING_TEXT = (os.getenv("GREETING_TEXT") or "Yo. What’s going on?").strip() or "Yo. What’s going on?"
GREETING_AUDIO_URL = (os.getenv("GREETING_AUDIO_URL") or "").strip()
GREETING_AUDIO_PATH = (os.getenv("GREETING_AUDIO_PATH") or "").strip()
GREETING_USE_CACHED_AUDIO = env_bool("GREETING_USE_CACHED_AUDIO", False)
# Pre-render the fixed greeting to WAV once at worker startup (in-memory) and
# serve every session's greeting from that buffer, so the first sound a user
# hears is instant and identical instead of paying a live Hume round-trip each
# session. The running worker holds the Hume key, so no asset needs committing.
# Skipped when a static cached source (GREETING_USE_CACHED_AUDIO + URL/path) is
# configured, since that path already serves cached audio.
GREETING_PRERENDER_AT_STARTUP = env_bool("GREETING_PRERENDER_AT_STARTUP", True)
SPOKEN_TEXT_NORMALIZATION_MODE = os.getenv("SPOKEN_TEXT_NORMALIZATION_MODE", "buffered_full_segment" if SPOKEN_TEXT_NORMALIZATION else "streaming_passthrough").strip() or ("buffered_full_segment" if SPOKEN_TEXT_NORMALIZATION else "streaming_passthrough")
def _env_int_clamped_with_source(name: str, default: int, minimum: int, maximum: int, *fallback_names: str) -> tuple[int, str]:
    for candidate in (name, *fallback_names):
        raw = os.getenv(candidate)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %s", candidate, raw, default)
            break
        return max(minimum, min(maximum, value)), candidate
    return default, "default"


LLM_FIRST_TOKEN_TIMEOUT_SECONDS, LLM_FIRST_TOKEN_TIMEOUT_SOURCE = _env_int_clamped_with_source("LLM_FIRST_TOKEN_TIMEOUT_SECONDS", 8, 1, 120)
LLM_TOTAL_TIMEOUT_SECONDS, LLM_TOTAL_TIMEOUT_SOURCE = _env_int_clamped_with_source("LLM_TOTAL_TIMEOUT_SECONDS", 20, 2, 300)
LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS = float(os.getenv("LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS", "1.5") or "1.5")
logger.info(
    "timeout_config_source first_token_timeout_source=%s total_timeout_source=%s first_token_timeout_seconds=%s total_timeout_seconds=%s first_token_extension_seconds=%s",
    LLM_FIRST_TOKEN_TIMEOUT_SOURCE,
    LLM_TOTAL_TIMEOUT_SOURCE,
    LLM_FIRST_TOKEN_TIMEOUT_SECONDS,
    LLM_TOTAL_TIMEOUT_SECONDS,
    LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS,
)
if LLM_TOTAL_TIMEOUT_SECONDS < LLM_FIRST_TOKEN_TIMEOUT_SECONDS:
    logger.warning(
        "LLM_TOTAL_TIMEOUT_SECONDS=%s is below LLM_FIRST_TOKEN_TIMEOUT_SECONDS=%s; raising total timeout to first-token timeout",
        LLM_TOTAL_TIMEOUT_SECONDS,
        LLM_FIRST_TOKEN_TIMEOUT_SECONDS,
    )
    LLM_TOTAL_TIMEOUT_SECONDS = LLM_FIRST_TOKEN_TIMEOUT_SECONDS
LLM_FALLBACK_RESPONSE = os.getenv(
    "LLM_FALLBACK_RESPONSE",
    "One second — I’m catching up.",
).strip() or "One second — I’m catching up."
if "zoned out" in LLM_FALLBACK_RESPONSE.lower():
    logger.warning("generic fallback uses forbidden wording; replacing configured fallback generic_fallback_forbidden_phrase=true")
    LLM_FALLBACK_RESPONSE = "One second — I’m catching up."
LLM_TO_TTS_HANDOFF_GUARD_ENABLED = env_bool("LLM_TO_TTS_HANDOFF_GUARD_ENABLED", False)
# Minimum sustained barge-in speech (ms) during ASSISTANT_THINKING before the
# handoff guard treats it as a real new utterance and suppresses the stale reply.
# Filters coughs/VAD blips. Only consulted when the guard is enabled.
HANDOFF_GUARD_MIN_SPEECH_MS = env_int_clamped("HANDOFF_GUARD_MIN_SPEECH_MS", 350, 0, 5000)


def _handoff_guard_should_suppress(turn_id: int) -> tuple[bool, str]:
    """Decide whether a pending reply for ``turn_id`` should be dropped before TTS.

    Returns (suppress, reason). True only when the handoff guard is enabled and a
    user barge-in began while the assistant was still thinking (no audio yet) for
    this same turn, and that barge-in qualifies as a real utterance — either
    already confirmed sustained on speech-stop, or still ongoing past the minimum
    speech threshold. Brief VAD blips never qualify.
    """
    if not LLM_TO_TTS_HANDOFF_GUARD_ENABLED:
        return False, "guard_disabled"
    if _barge_in_during_thinking_turn_id == 0 or _barge_in_during_thinking_turn_id != turn_id:
        return False, "no_barge_in_for_turn"
    elapsed_ms = (time.monotonic() - _barge_in_started_at) * 1000.0 if _barge_in_started_at > 0 else 0.0
    if _barge_in_confirmed_real or elapsed_ms >= HANDOFF_GUARD_MIN_SPEECH_MS:
        return True, f"real_barge_in_during_thinking:elapsed_ms={elapsed_ms:.0f}:min_ms={HANDOFF_GUARD_MIN_SPEECH_MS}"
    return False, f"barge_in_below_min_speech:elapsed_ms={elapsed_ms:.0f}:min_ms={HANDOFF_GUARD_MIN_SPEECH_MS}"


CONTEXT_WINDOW_TURNS = env_int_clamped("CONTEXT_WINDOW_TURNS", 10, 4, 100)
PREEMPTIVE_GENERATION_ENABLED = env_bool("PREEMPTIVE_GENERATION_ENABLED", False)
SEARCH_BRIDGE_MIN_DELAY_SECONDS = float(os.getenv("SEARCH_BRIDGE_MIN_DELAY_SECONDS", "0.75") or "0.75")
# When true (default), internet search only fires on an explicit lookup ask
# (intent=tool_request_search). Casual/conversational turns — including the
# catch-all "unknown" intent — no longer let the model auto-call search. This
# stops surprise searches on plain statements (the "I didn't ask you to look
# anything up" failure). Set false to restore the old permissive behavior where
# any non-blocked intent could trigger an LLM-initiated search.
SEARCH_REQUIRE_EXPLICIT_INTENT = env_bool("SEARCH_REQUIRE_EXPLICIT_INTENT", True)
ENDPOINTING_WAIT_EXTENSION_MIN_MS = env_int_clamped("ENDPOINTING_WAIT_EXTENSION_MIN_MS", 600, 100, 5000)
ENDPOINTING_WAIT_EXTENSION_MAX_MS = env_int_clamped("ENDPOINTING_WAIT_EXTENSION_MAX_MS", 1200, 100, 5000)
TURN_HOLD_FRAGMENT_REPLY_DEADLINE_SECONDS = float(os.getenv("TURN_HOLD_FRAGMENT_REPLY_DEADLINE_SECONDS", "2.5") or "2.5")
TURN_HOLD_FRAGMENT_MERGE_WINDOW_SECONDS = float(os.getenv("TURN_HOLD_FRAGMENT_MERGE_WINDOW_SECONDS", os.getenv("TURN_FRAGMENT_TTL_SECONDS", "7")) or "7")


def _pcm16_to_audio_frames(pcm_data: bytes, sample_rate: int, channels: int) -> list[rtc.AudioFrame]:
    bytes_per_sample = 2
    frame_samples_per_channel = max(1, int(sample_rate * 0.02))
    frame_bytes = frame_samples_per_channel * channels * bytes_per_sample
    frames: list[rtc.AudioFrame] = []
    cursor = 0
    while cursor < len(pcm_data):
        chunk = pcm_data[cursor : cursor + frame_bytes]
        cursor += frame_bytes
        if len(chunk) < frame_bytes:
            chunk = chunk + (b"\x00" * (frame_bytes - len(chunk)))
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=frame_samples_per_channel,
        )
        frames.append(frame)
    return frames


def _iter_post_speech_silence_frames(reference: "rtc.AudioFrame", hold_ms: int):
    """Yield ~20ms silent frames matching the reference frame's format, totaling hold_ms."""
    sample_rate = int(getattr(reference, "sample_rate", 0) or 24000)
    num_channels = int(getattr(reference, "num_channels", 0) or 1)
    frame_samples_per_channel = max(1, int(sample_rate * 0.02))
    silence = b"\x00" * (frame_samples_per_channel * num_channels * 2)
    total_samples = int(sample_rate * (hold_ms / 1000.0))
    emitted = 0
    while emitted < total_samples:
        yield rtc.AudioFrame(
            data=silence,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=frame_samples_per_channel,
        )
        emitted += frame_samples_per_channel


async def _with_post_speech_hold(source):
    """Pass through TTS audio frames, then append a trailing silence pad (config-gated).

    Keeps the published audio ending in silence so a client/WebRTC buffer drop at the
    speaking->listening transition trims silence instead of the final word/breath.
    """
    last_frame = None
    async for frame in source:
        if isinstance(frame, rtc.AudioFrame):
            last_frame = frame
        yield frame
    if TTS_POST_SPEECH_HOLD_MS > 0 and last_frame is not None:
        appended = 0
        for silence_frame in _iter_post_speech_silence_frames(last_frame, TTS_POST_SPEECH_HOLD_MS):
            appended += 1
            yield silence_frame
        logger.info(
            "Post-speech playout hold applied: tts_post_speech_hold_applied=true hold_ms=%s silent_frames_appended=%s tts_path=%s",
            TTS_POST_SPEECH_HOLD_MS,
            appended,
            _last_tts_path or "n/a",
        )


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


def _log_memory_identity_readiness() -> None:
    """One-line startup readiness check so the Railway config is verifiable at a
    glance: whether long-term memory, per-user identity, and the SimpleMem index
    are actually active. Logs presence only (never secret values)."""
    try:
        import importlib.util
        simplemem_installed = importlib.util.find_spec("simplemem") is not None
    except Exception:  # noqa: BLE001 - readiness check must never raise
        simplemem_installed = False
    logger.info(
        "memory_identity_readiness: memory_enabled=%s database_url_present=%s "
        "session_identity_shared_secret_present=%s simplemem_installed=%s "
        "simplemem_index_dir=%s memory_preload_limit=%s "
        "note=%s",
        memory_enabled(),
        bool(os.getenv("DATABASE_URL")),
        bool(os.getenv("SESSION_IDENTITY_SHARED_SECRET")),
        simplemem_installed,
        os.getenv("SIMPLEMEM_INDEX_DIR", "/data/simplemem"),
        os.getenv("MEMORY_PRELOAD_LIMIT", "10"),
        "set MEMORY_ENABLED=true and a shared SESSION_IDENTITY_SHARED_SECRET on both "
        "frontend+backend for per-user cross-session memory; simplemem optional "
        "(falls back to Postgres recency)",
    )
    # Semantic-memory / embedding readiness + the active fallback mode.
    try:
        from memory_layer import (
            memory_embedding_dimensions,
            memory_embedding_model,
            memory_vector_enabled,
        )
        semantic_on = memory_vector_enabled()
        embed_model = memory_embedding_model()
        embed_dim = memory_embedding_dimensions()
    except Exception:  # noqa: BLE001 - readiness check must never raise
        semantic_on, embed_model, embed_dim = False, "unknown", 0
    provider = (os.getenv("MEMORY_EMBEDDING_PROVIDER") or "openai").strip().lower()
    key_present = bool(
        (os.getenv("COHERE_API_KEY") if provider == "cohere" else os.getenv("OPENAI_API_KEY"))
    )
    if not memory_enabled():
        fallback_mode = "disabled"
    elif semantic_on and key_present:
        fallback_mode = "semantic_pgvector"
    elif simplemem_installed:
        fallback_mode = "simplemem"
    else:
        fallback_mode = "recency_text"
    logger.info(
        "memory_semantic_readiness: semantic_memory_enabled=%s embedding_provider=%s "
        "embedding_model=%s embedding_dim=%s embedding_key_present=%s "
        "simplemem_active=%s memory_fallback_mode=%s",
        semantic_on,
        provider,
        embed_model,
        embed_dim,
        key_present,
        simplemem_installed,
        fallback_mode,
    )


def _run_db_migrations_on_startup() -> None:
    logger.info("database_migrations_startup_enabled=%s", RUN_DB_MIGRATIONS_ON_STARTUP)
    if not RUN_DB_MIGRATIONS_ON_STARTUP:
        logger.info("database_migration_status=skipped")
        return

    if not os.getenv("DATABASE_URL"):
        logger.error("database_migration_status=failed reason=missing_DATABASE_URL")
        raise RuntimeError("DATABASE_URL is required when RUN_DB_MIGRATIONS_ON_STARTUP=true")

    script_path = os.path.join(os.path.dirname(__file__), "scripts", "apply_migrations.py")
    if not os.path.exists(script_path):
        logger.error("database_migration_status=failed reason=migration_runner_missing")
        raise RuntimeError("Migration runner not found at scripts/apply_migrations.py")

    logger.info("database_migration_status=running")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (result.stdout or "").splitlines():
        logger.info("migration_runner: %s", line)
    for line in (result.stderr or "").splitlines():
        logger.error("migration_runner: %s", line)

    if result.returncode != 0:
        logger.error("database_migration_status=failed return_code=%s", result.returncode)
        raise RuntimeError("Database migration failed during startup")

    logger.info("database_migration_status=success")


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


def _livekit_agents_version() -> str:
    try:
        import importlib.metadata as _md
        return _md.version("livekit-agents")
    except Exception:
        return "unknown"


def build_audio_turn_detector() -> tuple[object, dict]:
    """Build the LiveKit audio end-of-turn detector, feature-detected and flag-gated.

    Returns (detector_or_None, info). When the detector is unavailable or disabled
    we return None and the caller keeps the existing vad/stt turn_detection mode,
    so this is safe on SDKs without inference.TurnDetector.
    """
    info: dict = {
        "available": _lk_inference is not None and hasattr(_lk_inference, "TurnDetector"),
        "enabled": LIVEKIT_TURN_DETECTOR_ENABLED,
        "class": "none",
        "version": LIVEKIT_TURN_DETECTOR_VERSION,
        "default_selection": "auto(sdk: v1 via LiveKit Inference, else v1-mini local)"
        if LIVEKIT_TURN_DETECTOR_VERSION in ("", "auto")
        else LIVEKIT_TURN_DETECTOR_VERSION,
        "fallback_used": False,
        "error": "none",
    }
    if not LIVEKIT_TURN_DETECTOR_ENABLED:
        info["error"] = "disabled_by_config"
        return None, info
    if not info["available"]:
        info["fallback_used"] = True
        info["error"] = f"inference.TurnDetector unavailable in livekit-agents {_livekit_agents_version()} (needs >=1.6.1)"
        return None, info
    kwargs: dict = {}
    if LIVEKIT_TURN_DETECTOR_VERSION not in ("", "auto"):
        kwargs["version"] = LIVEKIT_TURN_DETECTOR_VERSION
    if LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD:
        try:
            kwargs["unlikely_threshold"] = float(LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD)
        except ValueError:
            logger.warning("Invalid LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD=%s; ignoring", LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD)
    try:
        detector = _lk_inference.TurnDetector(**kwargs)
        info["class"] = type(detector).__name__
        return detector, info
    except Exception as exc:
        info["fallback_used"] = True
        info["error"] = f"{type(exc).__name__}: {_redact_sensitive_text(exc)}"
        logger.warning("LiveKit audio TurnDetector init failed; keeping vad/stt mode: %s", info["error"])
        return None, info


def build_room_options() -> room_io.RoomOptions | None:
    # Inbound audio enhancement is applied at the room audio_input stage, i.e.
    # BEFORE STT / VAD / turn detection. AI_COUSTICS_ENABLED remains the master
    # switch; LIVEKIT_AUDIO_ENHANCEMENT_* are the canonical config names.
    enabled = LIVEKIT_AUDIO_ENHANCEMENT_ENABLED and AI_COUSTICS_ENABLED
    model_name_cfg = LIVEKIT_AUDIO_ENHANCEMENT_MODEL or os.getenv("AI_COUSTICS_MODEL", "QUAIL_VF_S")
    if not enabled:
        logger.info(
            "audio_enhancement_enabled=false audio_enhancement_provider=%s audio_enhancement_model=%s audio_enhancement_applied_stage=none audio_enhancement_error=disabled_by_config",
            LIVEKIT_AUDIO_ENHANCEMENT_PROVIDER,
            model_name_cfg,
        )
        return None
    if LIVEKIT_AUDIO_ENHANCEMENT_PROVIDER not in ("ai_coustics", "ai-coustics"):
        logger.warning(
            "audio_enhancement_enabled=true audio_enhancement_provider=%s audio_enhancement_model=%s audio_enhancement_applied_stage=none audio_enhancement_error=unsupported_provider_only_ai_coustics_supported",
            LIVEKIT_AUDIO_ENHANCEMENT_PROVIDER,
            model_name_cfg,
        )
        return None

    selected_model, selected_model_name = _resolve_ai_coustics_model(model_name_cfg)
    raw_level = os.getenv("AI_COUSTICS_ENHANCEMENT_LEVEL", "0.8")
    try:
        enhancement_level = float(raw_level)
    except ValueError:
        logger.warning("Invalid AI_COUSTICS_ENHANCEMENT_LEVEL=%s. Falling back to 0.8", raw_level)
        enhancement_level = 0.8
    enhancement_level = max(0.0, min(1.0, enhancement_level))

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
    except Exception as e:
        logger.error(
            "audio_enhancement_enabled=true audio_enhancement_provider=ai_coustics audio_enhancement_model=%s audio_enhancement_applied_stage=failed audio_enhancement_error=%s",
            selected_model_name,
            f"{type(e).__name__}: {_redact_sensitive_text(e)}",
        )
        return None

    logger.info(
        "audio_enhancement_enabled=true audio_enhancement_provider=ai_coustics audio_enhancement_model=%s audio_enhancement_applied_stage=inbound_pre_stt_vad_turndetection enhancement_level=%s audio_enhancement_error=none",
        selected_model_name,
        enhancement_level,
    )
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=enhancer,
        )
    )

def _resolve_hume_model_version() -> tuple[str, str]:
    hume_model_raw = (os.getenv("HUME_MODEL", "octave-2") or "octave-2").strip()
    hume_model = hume_model_raw.lower()
    if not hume_model:
        return "octave-2", "2"
    if hume_model in {"octave-2", "2", "v2"}:
        return hume_model_raw, "2"
    if hume_model in {"octave-1", "1", "v1"}:
        return hume_model_raw, "1"
    logger.warning("Unsupported HUME_MODEL=%s; falling back to octave-2 (model_version=2)", _redact_sensitive_text(hume_model_raw))
    return hume_model_raw, "2"



async def _load_cached_greeting_audio_bytes() -> tuple[bytes | None, str, str]:
    if GREETING_AUDIO_URL:
        try:
            timeout = aiohttp.ClientTimeout(total=float(os.getenv("GREETING_AUDIO_FETCH_TIMEOUT_SECONDS", "5")))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(GREETING_AUDIO_URL) as response:
                    if response.status >= 400:
                        return None, "url", f"http_status_{response.status}"
                    return await response.read(), "url", "none"
        except Exception as exc:
            return None, "url", f"fetch_error_{type(exc).__name__}"

    if GREETING_AUDIO_PATH:
        try:
            with open(GREETING_AUDIO_PATH, "rb") as audio_file:
                return audio_file.read(), "path", "none"
        except Exception as exc:
            return None, "path", f"path_error_{type(exc).__name__}"

    return None, "none", "cached_audio_missing"


def _validate_cached_wav_audio(audio_bytes: bytes) -> None:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            sample_width = wav.getsampwidth()
            if sample_width != 2:
                raise RuntimeError(f"cached_greeting_audio_unsupported_sample_width:{sample_width}")
            if wav.getnchannels() <= 0 or wav.getframerate() <= 0 or wav.getnframes() <= 0:
                raise RuntimeError("cached_greeting_audio_invalid_wav_metadata")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"cached_greeting_audio_not_wav:{type(exc).__name__}") from exc


def _cached_wav_audio_frames(audio_bytes: bytes, first_frame_marker: dict[str, float]) -> AsyncIterable[rtc.AudioFrame]:
    async def _frames() -> AsyncIterable[rtc.AudioFrame]:
        try:
            wav = wave.open(io.BytesIO(audio_bytes), "rb")
        except Exception as exc:
            raise RuntimeError(f"cached_greeting_audio_not_wav:{type(exc).__name__}") from exc
        with wav:
            sample_rate = wav.getframerate()
            num_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            if sample_width != 2:
                raise RuntimeError(f"cached_greeting_audio_unsupported_sample_width:{sample_width}")
            chunk_samples = max(1, int(sample_rate * 0.02))
            while True:
                data = wav.readframes(chunk_samples)
                if not data:
                    break
                samples_per_channel = len(data) // (sample_width * num_channels)
                if samples_per_channel <= 0:
                    continue
                if "at" not in first_frame_marker:
                    first_frame_marker["at"] = time.monotonic()
                yield rtc.AudioFrame(
                    data=data,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=samples_per_channel,
                )
                await asyncio.sleep(samples_per_channel / sample_rate)

    return _frames()


async def _prerender_greeting() -> None:
    """Render the fixed greeting to WAV once at startup and cache it in memory.

    The running worker holds the Hume key, so this renders the fixed greeting
    text through the same Hume TTS path the session uses, assembles the frames
    into a WAV buffer, and stores it in ``_prerendered_greeting_wav``. Every
    session then serves its greeting from that buffer for instant, identical
    first audio. Best-effort: any failure is logged and leaves the buffer None,
    in which case the greeting flow falls back to live TTS exactly as before.
    """
    global _prerendered_greeting_wav, _prerender_greeting_in_flight
    if _prerendered_greeting_wav is not None or _prerender_greeting_in_flight:
        return
    if TTS_PROVIDER != "hume":
        logger.info(
            "Greeting pre-render skipped: greeting_prerender_skipped_reason=tts_provider_not_hume tts_provider=%s",
            TTS_PROVIDER,
        )
        return
    _prerender_greeting_in_flight = True
    started_at = time.monotonic()
    try:
        wav_bytes = await _render_greeting_wav_bytes(started_at)
    finally:
        # Reset so a later job can retry if this render did not produce a buffer.
        _prerender_greeting_in_flight = False
    if wav_bytes is None:
        return
    _prerendered_greeting_wav = wav_bytes
    logger.info(
        "Greeting pre-render completed: greeting_prerender_ready=true wav_bytes=%s elapsed_seconds=%s",
        len(wav_bytes),
        _fmt_seconds(time.monotonic() - started_at),
    )


async def _render_greeting_wav_bytes(started_at: float) -> bytes | None:
    """Synthesize the greeting via Hume and assemble WAV bytes. None on failure."""
    try:
        tts = build_tts()
    except Exception as exc:  # noqa: BLE001 - pre-render must never break startup
        logger.warning(
            "Greeting pre-render skipped: greeting_prerender_skipped_reason=tts_build_failed error_type=%s",
            type(exc).__name__,
        )
        return None
    chunks: list[bytes] = []
    sample_rate = 0
    num_channels = 0
    frame_count = 0
    stream = tts.synthesize(GREETING_TEXT)
    try:
        async for event in stream:
            frame = getattr(event, "frame", None)
            if frame is None:
                continue
            data = getattr(frame, "data", None)
            if not data:
                continue
            chunks.append(bytes(data))
            if sample_rate == 0:
                sample_rate = int(getattr(frame, "sample_rate", 0) or 0)
                num_channels = int(getattr(frame, "num_channels", 0) or 0)
            frame_count += 1
    except Exception as exc:  # noqa: BLE001 - pre-render must never break startup
        logger.warning(
            "Greeting pre-render failed: greeting_prerender_failed_reason=synthesize_error error_type=%s elapsed_seconds=%s",
            type(exc).__name__,
            _fmt_seconds(time.monotonic() - started_at),
        )
        return None
    finally:
        for closeable in (stream, tts):
            aclose = getattr(closeable, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    pass
        # The throwaway TTS may own a debug http_session that aclose() doesn't
        # close — close it so the pre-render doesn't leak a session per job.
        await _close_tts_owned_session(tts)
    if not chunks or sample_rate <= 0 or num_channels <= 0:
        logger.warning(
            "Greeting pre-render produced no usable audio: frame_count=%s sample_rate=%s num_channels=%s elapsed_seconds=%s",
            frame_count,
            sample_rate,
            num_channels,
            _fmt_seconds(time.monotonic() - started_at),
        )
        return None
    try:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(num_channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"".join(chunks))
        wav_bytes = buffer.getvalue()
        _validate_cached_wav_audio(wav_bytes)
    except Exception as exc:  # noqa: BLE001 - pre-render must never break startup
        logger.warning(
            "Greeting pre-render WAV assembly failed: error_type=%s elapsed_seconds=%s",
            type(exc).__name__,
            _fmt_seconds(time.monotonic() - started_at),
        )
        return None
    logger.info(
        "Greeting pre-render synthesized: frame_count=%s sample_rate=%s num_channels=%s wav_bytes=%s elapsed_seconds=%s",
        frame_count,
        sample_rate,
        num_channels,
        len(wav_bytes),
        _fmt_seconds(time.monotonic() - started_at),
    )
    return wav_bytes


def _build_single_tts(provider: str):
    """Build a single TTS plugin instance for one provider name (no fallback)."""
    global _last_hume_model_version, _last_hume_description_applied, _last_hume_voice_present, _last_hume_voice_kind, _last_hume_instant_mode, _last_hume_speed, _last_hume_trailing_silence, _last_hume_style_context_applied, _last_hume_tts_build_started_at, _last_hume_tts_build_completed_at, _last_hume_tts_debug_http
    if provider == "deepgram":
        logger.info("Using Deepgram TTS provider")
        return deepgram.TTS(
            model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-asteria-en")
        )

    if provider == "omnivoice":
        omni_cfg = OmniVoiceConfig.from_env()
        usable, reason = omni_cfg.is_usable()
        logger.info(
            "Using OmniVoice TTS provider: omnivoice_enabled=%s omnivoice_url_present=%s device=%s default_language=%s expressive_tags_enabled=%s audio_format=%s sample_rate=%s usable=%s reason=%s",
            omni_cfg.enabled,
            bool(omni_cfg.base_url),
            omni_cfg.device,
            omni_cfg.default_language,
            omni_cfg.expressive_tags_enabled,
            omni_cfg.audio_format,
            omni_cfg.sample_rate,
            usable,
            reason,
        )
        if not usable:
            # Don't hand back a provider that can't synthesize; surface it so the
            # caller can decide (build_tts falls back to the configured fallback).
            raise RuntimeError(f"OmniVoice TTS not usable: {reason}")
        return OmniVoiceTTS(config=omni_cfg)

    if provider == "hume":
        _last_hume_tts_build_started_at = time.monotonic()
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
        hume_style_context = (os.getenv("HUME_STYLE_CONTEXT") or "").strip()
        hume_style_context_present = bool(hume_style_context)
        hume_style_context_length = len(hume_style_context)
        hume_context_source = "HUME_STYLE_CONTEXT" if hume_style_context_present else "none"
        hume_description = os.getenv("HUME_DESCRIPTION") or (
            "A warm, calm, natural companion voice. Speak with relaxed pacing, soft sentence endings, "
            "and brief natural pauses between thoughts. Do not sound rushed, clipped, or abrupt at the end of sentences."
        )
        hume_description_present = bool(hume_description)
        hume_description_length = len(hume_description)
        hume_trailing_silence = float(os.getenv("HUME_TRAILING_SILENCE", "0.25"))
        hume_model_raw, hume_model_version = _resolve_hume_model_version()
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
        hume_style_context_applied = False
        hume_style_context_skip_reason = "none"
        if hume_style_context_present:
            hume_style_context_skip_reason = "freeform_context_not_supported"
        logger.info(
            "Hume style context: hume_style_context_present=%s hume_style_context_length=%s hume_style_context_applied=%s hume_style_context_skip_reason=%s hume_context_source=%s description_applied=%s model_version=%s",
            hume_style_context_present,
            hume_style_context_length,
            hume_style_context_applied,
            hume_style_context_skip_reason,
            hume_context_source,
            description_applied,
            hume_model_version,
        )
        description_skip_reason = "none" if description_applied else "octave2_unsupported"
        logger.info(
            "Hume model/description summary: hume_model_raw=%s model_version=%s description_present=%s description_applied=%s description_skip_reason=%s hume_style_context_present=%s hume_style_context_applied=%s hume_style_context_skip_reason=%s",
            _redact_sensitive_text(hume_model_raw),
            hume_model_version,
            hume_description_present,
            description_applied,
            description_skip_reason,
            hume_style_context_present,
            hume_style_context_applied,
            hume_style_context_skip_reason,
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
        _last_hume_model_version = str(hume_model_version)
        _last_hume_description_applied = str(description_applied).lower()
        _last_hume_voice_present = bool(voice)
        _last_hume_voice_kind = voice_kind
        _last_hume_instant_mode = str(instant_mode).lower()
        _last_hume_speed = str(hume_speed)
        _last_hume_trailing_silence = str(hume_trailing_silence if trailing_silence_applied else "n/a")
        _last_hume_style_context_applied = str(hume_style_context_applied).lower()
        _last_hume_tts_debug_http = hume_tts_debug_http
        logger.info(
            "Hume TTS effective config: model_version=%s voice_kind=%s voice_present=%s voice_provider=%s instant_mode=%s speed=%s description_present=%s description_applied=%s description_length=%s trailing_silence_supported=%s trailing_silence_applied=%s trailing_silence_value=%s hume_style_context_present=%s hume_style_context_length=%s hume_style_context_applied=%s debug_http=%s",
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
            hume_style_context_present,
            hume_style_context_length,
            hume_style_context_applied,
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
                path = _redact_sensitive_text(params.url.path)
                text_hash = ctx_hash if ctx_hash != "n/a" else _latest_normalized_text_hash
                dedupe_key, duplicate, seen_count = _record_hume_request_metadata(
                    path=path,
                    speech_id=_latest_current_speech_id_for_hume,
                    normalized_text_hash=text_hash,
                    feeds_playout=True,
                    register=False,
                )
                logger.info(
                    "Hume TTS HTTP request: hume_request_index=%s method=%s path=%s latest_agent_state=%s active_assistant_count=%s current_speech_id=%s instant_mode=%s speed=%s trailing_silence=%s normalized_text_hash=%s hume_request_dedupe_key=%s hume_duplicate_request_detected=%s hume_request_seen_count=%s hume_request_feeds_playout=%s hume_retry=%s retry_reason=%s hume_request_observation=%s debug=true",
                    _hume_tts_request_counter,
                    params.method,
                    path,
                    _latest_agent_state_for_hume,
                    _latest_active_assistant_count_for_hume,
                    _latest_current_speech_id_for_hume,
                    instant_mode,
                    hume_speed,
                    hume_trailing_silence,
                    text_hash,
                    dedupe_key,
                    duplicate,
                    seen_count,
                    True,
                    False,
                    "debug_http_trace",
                    "debug_http_trace",
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

        tts_instance = hume.TTS(**hume_tts_kwargs)
        # Track a debug http_session WE created (HUME_TTS_DEBUG_HTTP): the plugin
        # never closes a caller-provided session, so without this it leaks as
        # "Unclosed client session"/"Unclosed connector". Callers close it via
        # _close_tts_owned_session(). None when we didn't create one.
        setattr(tts_instance, "_lucy_owned_http_session", hume_tts_kwargs.get("http_session"))
        _last_hume_tts_build_completed_at = time.monotonic()
        logger.info(
            "Hume TTS instance created: build_duration_seconds=%s lazy_http_session_expected=%s debug_http=%s",
            _fmt_seconds(_last_hume_tts_build_completed_at - _last_hume_tts_build_started_at),
            not hume_tts_debug_http,
            hume_tts_debug_http,
        )
        return tts_instance

    raise RuntimeError("Unsupported TTS_PROVIDER. Use 'deepgram', 'hume', or 'omnivoice'.")


def build_tts():
    """Build the session TTS, wrapping the primary in a FallbackAdapter when a
    distinct fallback provider is configured.

    Only the omnivoice path is wrapped today: a sidecar-backed OmniVoice can fail,
    time out, or return invalid audio, and we never want the session to go silent,
    so it degrades to Hume via LiveKit's tts.FallbackAdapter. Hume/Deepgram are
    returned bare, exactly as before (no behavior change for the existing path).
    If OmniVoice can't even be built (disabled / no URL), we fall straight to the
    fallback provider so a misconfigured worker still speaks.
    """
    fallback_enabled = (
        TTS_PROVIDER == "omnivoice"
        and TTS_FALLBACK_PROVIDER not in ("", "none", TTS_PROVIDER)
    )
    if not fallback_enabled:
        return _build_single_tts(TTS_PROVIDER)

    try:
        primary = _build_single_tts(TTS_PROVIDER)
    except Exception as exc:  # noqa: BLE001 - degrade to fallback instead of crashing
        logger.warning(
            "TTS primary build failed; using fallback provider only: tts_provider=%s tts_fallback_provider=%s error=%s",
            TTS_PROVIDER,
            TTS_FALLBACK_PROVIDER,
            _redact_sensitive_text(exc),
        )
        return _build_single_tts(TTS_FALLBACK_PROVIDER)

    fallback = _build_single_tts(TTS_FALLBACK_PROVIDER)
    logger.info(
        "TTS provider selected: tts_provider=%s tts_fallback_provider=%s fallback_adapter_enabled=true",
        TTS_PROVIDER,
        TTS_FALLBACK_PROVIDER,
    )
    return tts.FallbackAdapter([primary, fallback])


def _init_session_voice_and_language(session_tts: object) -> None:
    """Pick one stable voice preset for this session and prime the active language.

    Run once at session start. Finds the OmniVoiceTTS (bare or wrapped), selects a
    preset from the rotating pool (stable for the whole session), and points it at
    the active language. No-op when OmniVoice isn't the active provider. Best-effort
    — any failure leaves OmniVoice on its default voice rather than breaking start.
    """
    global _session_omnivoice_tts, _active_session_language
    _active_session_language = SESSION_LANGUAGE
    _session_omnivoice_tts = find_omnivoice_tts(session_tts)
    if _session_omnivoice_tts is None:
        return
    preset = None
    try:
        preset = get_session_selector().select()
    except Exception as exc:  # noqa: BLE001 - pool issues must not break the session
        logger.warning("omnivoice_voice_pool_select_failed=true error=%s", _redact_sensitive_text(exc))
    if preset is not None:
        _session_omnivoice_tts.update_options(voice=preset.id, language=_active_session_language)
    else:
        _session_omnivoice_tts.update_options(language=_active_session_language)
    logger.info(
        "omnivoice_session_voice_selected=true voice_preset_id=%s voice_preset_name=%s active_language=%s pool_used=%s",
        preset.id if preset else "default",
        preset.name if preset else "",
        _active_session_language,
        preset is not None,
    )


def _maybe_switch_language(user_text: str) -> tuple[str, str] | None:
    """If the user asked to switch language, update active state + OmniVoice.

    Returns (code, English name) on a switch (so the caller can nudge the LLM),
    else None. Skips no-op requests for the already-active language.
    """
    global _active_session_language
    detected = detect_language_request(user_text)
    if detected is None:
        return None
    code, name = detected
    if code == _active_session_language:
        return None
    previous = _active_session_language
    _active_session_language = code
    if _session_omnivoice_tts is not None:
        _session_omnivoice_tts.update_options(language=code)
    logger.info(
        "language_switch_request=true from_language=%s to_language=%s to_language_name=%s omnivoice_active=%s",
        previous,
        code,
        name,
        _session_omnivoice_tts is not None,
    )
    return code, name


async def _close_tts_owned_session(tts_obj: object) -> None:
    """Close http sessions/handles a build owns for this TTS instance.

    Three things to clean up, all best-effort:
      - the Hume debug http_session build_tts created (_lucy_owned_http_session),
      - an OmniVoiceTTS's own aiohttp session (via its aclose()),
      - children of a FallbackAdapter (recurse), since FallbackAdapter.aclose()
        does not close the wrapped providers' sessions.
    The Hume/Deepgram plugins won't close a caller-provided session, so without
    this every build leaks an aiohttp ClientSession/connector.
    """
    # Recurse into a FallbackAdapter's wrapped providers.
    children = getattr(tts_obj, "_tts_instances", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            await _close_tts_owned_session(child)

    if isinstance(tts_obj, OmniVoiceTTS):
        try:
            await tts_obj.aclose()
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass

    session = getattr(tts_obj, "_lucy_owned_http_session", None)
    if session is not None:
        try:
            if not getattr(session, "closed", True):
                await session.close()
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass

app = FastAPI()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})



def _safe_metadata_preview(value: Any) -> str:
    if value is None:
        return "none"
    return _redact_sensitive_text(value)[:500]


def _metadata_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value.keys())
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        if isinstance(parsed, dict):
            return sorted(str(key) for key in parsed.keys())
    return []


def _metadata_debug_entries_from_context(ctx: JobContext) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    for attr_name in ("job", "room"):
        obj = _safe_attr(ctx, attr_name)
        metadata = _safe_attr(obj, "metadata")
        entries.append((f"{attr_name}.metadata", metadata))

    job = _safe_attr(ctx, "job")
    for attr_name in ("participant", "participant_info"):
        participant = _safe_attr(job, attr_name)
        metadata = _safe_attr(participant, "metadata")
        entries.append((f"job.{attr_name}.metadata", metadata))

    room = _safe_attr(ctx, "room")
    remote_participants = _safe_attr(room, "remote_participants")
    if isinstance(remote_participants, dict):
        for participant_id, participant in remote_participants.items():
            metadata = _safe_attr(participant, "metadata")
            entries.append((f"room.remote_participants.{participant_id}.metadata", metadata))
    return entries


def _chat_ctx_items(chat_ctx: object) -> list[object]:
    """Return a safe, concrete message list from LiveKit chat context shapes."""
    if chat_ctx is None:
        return []
    messages = getattr(chat_ctx, "messages", None)
    if callable(messages):
        try:
            messages = messages()
        except Exception:
            messages = None
    if messages is None:
        messages = chat_ctx
    try:
        return list(messages)  # type: ignore[arg-type]
    except Exception:
        return []


def _extract_latest_user_text_from_chat_ctx(chat_ctx: object) -> str:
    iterable = _chat_ctx_items(chat_ctx)
    if not iterable:
        return _last_user_message_text
    for message in reversed(iterable):
        role = str(getattr(message, "role", "")).lower()
        if role and role != "user":
            continue
        text = _extract_text_for_debug(message).strip()
        if text:
            return text
    return _last_user_message_text



def _recent_turn_previews_from_chat_ctx(chat_ctx: object, limit: int = 5) -> list[str]:
    iterable = _chat_ctx_items(chat_ctx)
    if not iterable:
        return []
    previews: list[str] = []
    for message in iterable[-limit:]:
        role = str(getattr(message, "role", "unknown"))
        text = _extract_text_for_debug(message).strip()
        if not text:
            continue
        previews.append(f"{role}: {_redact_sensitive_text(text)[:240]}")
    return previews


@dataclass(frozen=True)
class TurnPolicyResult:
    decision: str
    classification: str
    confidence: float
    reason: str
    should_start_generation: bool
    should_merge_held_fragment: bool
    should_clear_held_fragment: bool


def _recent_user_messages_from_chat_ctx(chat_ctx: object, exclude_text: str = "", limit: int = 3) -> list[str]:
    """Return up to `limit` of the user's most recent prior messages (oldest→newest).

    Excludes the current recall question (`exclude_text`) so a "what did I just
    say / my last question" ask anchors on the user's actual previous turn, not
    on the question itself.
    """
    iterable = _chat_ctx_items(chat_ctx)
    if not iterable:
        return []
    exclude_norm = (exclude_text or "").strip()
    collected: list[str] = []
    for message in reversed(iterable):
        role = str(getattr(message, "role", "")).lower()
        if role != "user":
            continue
        text = _extract_text_for_debug(message).strip()
        if not text or text == exclude_norm:
            continue
        collected.append(text)
        if len(collected) >= max(1, limit):
            break
    collected.reverse()
    return collected


def _inject_recall_anchor_note(turn_ctx: object, recent_user_messages: list[str]) -> bool:
    """Anchor a recall ask on the user's verbatim most-recent prior message.

    "Do you remember what I just said / my last question?" is a transcript task,
    not a long-term-memory task. Without this, the model paraphrases and can grab
    an earlier turn. This note hands it the actual recent user messages and tells
    it to quote the most recent one rather than summarize an older one.
    """
    if not recent_user_messages:
        return False
    most_recent = recent_user_messages[-1]
    lines = "\n".join(f"- {m}" for m in recent_user_messages)
    note = (
        "Internal recall note. Do not reveal or quote this note itself. The user "
        "is asking you to recall what THEY just said or asked. Answer from the "
        "actual conversation transcript below, not from long-term memory or a "
        "loose paraphrase. Anchor on their MOST RECENT prior message and quote it "
        "closely; do not reach back to an earlier turn. If they ask for 'the last "
        "thing I said' or 'my last question', it is this: \"" + most_recent + "\". "
        "Recent user messages, oldest to newest:\n" + lines
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("Recall anchor note could not be injected: turn_ctx_add_message_unavailable")
        return False
    try:
        add_message(role="developer", content=note)
        logger.info(
            "recall_anchor_note_injected=true recent_user_message_count=%s turn_id=%s",
            len(recent_user_messages),
            _current_turn_id,
        )
        return True
    except Exception as exc:
        logger.warning(
            "Recall anchor note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )
        return False


@dataclass
class HumeSpeechAudioCoverage:
    speech_id: str
    turn_id: int
    path: str
    normalized_text_hash: str
    started_at: float
    frame_count: int = 0
    byte_count: int = 0
    sample_count: int = 0
    sample_rate: int = 0
    num_channels: int = 0
    first_frame_at: float | None = None
    last_frame_at: float | None = None
    capture_chunks: list[bytes] | None = None
    capture_truncated: bool = False
    artifact_path: str | None = None


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", clean_transcript(text or "").lower())


def _semantic_overlap(left: str, right: str) -> float:
    stop = {"the", "a", "an", "and", "or", "but", "so", "like", "i", "you", "it", "that", "this", "to", "of", "in", "on", "for", "with", "me", "my", "is", "was", "are", "were"}
    left_words = {word for word in _normalized_words(left) if word not in stop and len(word) > 2}
    right_words = {word for word in _normalized_words(right) if word not in stop and len(word) > 2}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(1, min(len(left_words), len(right_words)))


def _classify_turn_transcript(text: str, context: TranscriptContext | None = None) -> tuple[str, float, str]:
    cleaned = clean_transcript(text or "").strip()
    lowered = cleaned.lower().replace("’", "'")
    words = _normalized_words(cleaned)
    word_count = len(words)
    if not cleaned:
        return "UNCLEAR_AUDIO", 0.2, "empty_transcript"
    if context is not None and context.detected_intent == "unclear_fragment" and context.confidence < 0.5:
        return "UNCLEAR_AUDIO", 0.55, "transcript_context_unclear_low_confidence"
    if re.search(r"\b(interrupt|interrupted|not answering|didn't answer|did not answer|answer me|why aren't you|why are you|lag|lagged|slow|silence|silent|stuck|repeat|repeating|again|heard me|listening)\b", lowered):
        return "META_COMPLAINT", 0.86, "system_or_latency_complaint"
    if re.search(r"\b(i feel|i felt|i'm feeling|i am feeling|i hate|i love|i miss|i'm scared|i am scared|i'm worried|i am worried|it hurt|that hurt|got under my skin|annoyed|upset|sad|angry|afraid|lonely|tired|overwhelmed|stressed)\b", lowered):
        return "EMOTIONAL_STATEMENT", 0.88, "emotional_meaning"
    if context is not None and context.detected_intent in {"date_time_question", "tool_request_search", "tool_request_email", "tool_request_document", "calculation_request", "counting_request", "timer_request", "stop_or_cancel_request"}:
        return "COMPLETE_THOUGHT", 0.86, f"actionable_intent:{context.detected_intent}"
    if word_count <= 2 and lowered.rstrip(".!?") in {"yeah", "yes", "no", "okay", "ok", "right", "sure", "thanks", "thank you"}:
        return "LOW_INFORMATION_FILLER", 0.78, "backchannel_or_ack"
    if lowered.endswith(",") or re.search(r"\b(and|because|when|if|but|so|like|while|although|since|then)$", lowered.rstrip(".,!?")):
        return "INCOMPLETE_THOUGHT", 0.82, "dangling_clause_or_connector"
    if word_count < 4 and context is not None and context.ambiguity_detected and context.clarification_suggested:
        return "INCOMPLETE_THOUGHT", 0.72, "short_ambiguous_fragment"
    if word_count >= 4 or cleaned.endswith((".", "?", "!")):
        return "COMPLETE_THOUGHT", 0.8, "complete_shape"
    if word_count <= 3:
        return "INCOMPLETE_THOUGHT", 0.58, "short_without_clear_completion"
    return "COMPLETE_THOUGHT", 0.65, "default_complete"


def _make_turn_policy_decision(text: str, context: TranscriptContext | None = None, *, held_text: str = "", held_created_at: float = 0.0, now: float | None = None) -> TurnPolicyResult:
    now = now if now is not None else time.monotonic()
    classification, confidence, reason = _classify_turn_transcript(text, context)
    held_age = (now - held_created_at) if held_text and held_created_at > 0 else None
    has_mergeable_held = bool(held_text and held_age is not None and held_age <= TURN_HOLD_FRAGMENT_MERGE_WINDOW_SECONDS)
    if classification == "UNCLEAR_AUDIO":
        return TurnPolicyResult("ASK_FOR_AUDIO_CLARIFICATION", classification, confidence, reason, True, False, True)
    if classification == "META_COMPLAINT":
        return TurnPolicyResult("RECOVER_FROM_SILENCE", classification, confidence, reason, True, False, True)
    if classification == "LOW_INFORMATION_FILLER":
        return TurnPolicyResult("IGNORE_LOW_INFORMATION_FILLER", classification, confidence, reason, False, False, False)
    if has_mergeable_held:
        if _semantic_overlap(held_text, text) >= 0.2 or classification == "INCOMPLETE_THOUGHT":
            return TurnPolicyResult("MERGE_WITH_HELD_FRAGMENT", classification, confidence, "semantic_continuation", True, True, True)
        return TurnPolicyResult("FLUSH_HELD_AND_COMMIT_NEW", classification, confidence, "new_topic_or_unrelated_to_held_fragment", True, False, True)
    if classification == "INCOMPLETE_THOUGHT":
        return TurnPolicyResult("HOLD_FOR_CONTINUATION", classification, confidence, reason, False, False, False)
    return TurnPolicyResult("COMMIT_NOW", classification, confidence, reason, True, False, True)


def _fallback_requires_user_repeat(reason: str, classification: str) -> bool:
    return classification == "UNCLEAR_AUDIO" or reason in {"audio_unclear", "stt_unclear"}


def _fallback_text_for_reason(reason: str, classification: str) -> str:
    if _fallback_requires_user_repeat(reason, classification):
        return "I caught part of that, but not cleanly — could you say the last bit again?"
    if classification == "META_COMPLAINT":
        return "You’re right — I lagged there."
    if classification == "EMOTIONAL_STATEMENT":
        return "I’m with you."
    if reason in {"first_token_timeout", "total_timeout_no_text", "provider_error", "empty_stream"}:
        return "One second — I’m catching up."
    return LLM_FALLBACK_RESPONSE


def _is_transport_api_connection_error(error: object) -> bool:
    current: object | None = error
    for _ in range(4):
        if current is None:
            return False
        name = type(current).__name__.lower()
        if "apiconnectionerror" in name or "connection" in name and "error" in name:
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _should_retry_openrouter_connection_error(error: object, *, first_token_seen: bool, chunk_count: int, text_length: int, llm_turn_id: int, tts_started_for_turn: bool) -> tuple[bool, str]:
    if not _is_transport_api_connection_error(error):
        return False, "not_transport_api_connection_error"
    if first_token_seen:
        return False, "first_token_seen"
    if chunk_count > 0:
        return False, "chunks_already_received"
    if text_length > 0:
        return False, "text_already_received"
    if llm_turn_id != _current_turn_id:
        return False, "stale_turn"
    if tts_started_for_turn:
        return False, "tts_already_started"
    return True, "eligible"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def _generic_fallback_suppression_reason(fallback_turn_id: int, fallback_started_at: float) -> str | None:
    if (
        _latest_user_state_for_greeting == "speaking"
        or _current_turn_id != fallback_turn_id
        or _latest_user_speaking_at > fallback_started_at
        or (_latest_stt_partial_at > fallback_started_at and _latest_stt_partial_at > _latest_stt_final_at)
    ):
        return "user_speaking_or_newer_turn_pending"
    return None


def _endpointing_wait_extension_ms() -> int:
    lower = min(ENDPOINTING_WAIT_EXTENSION_MIN_MS, ENDPOINTING_WAIT_EXTENSION_MAX_MS)
    upper = max(ENDPOINTING_WAIT_EXTENSION_MIN_MS, ENDPOINTING_WAIT_EXTENSION_MAX_MS)
    return max(lower, min(upper, 900))


def _is_clear_short_commit(text: str) -> bool:
    cleaned = clean_transcript(text).strip().lower()
    normalized = cleaned.rstrip(".!?").strip()
    if not normalized:
        return False
    if cleaned.endswith("?"):
        return True
    if normalized in {"yes", "no", "yeah", "yep", "nope", "okay", "ok", "stop", "send it"}:
        return True
    if re.search(r"\b(what time is it|what day is it|what'?s today|count to|search that|look that up|stop|send it)\b", normalized):
        return True
    return False


def _is_stale_llm_turn(llm_turn_id: int) -> bool:
    return _current_turn_id != llm_turn_id


def _endpointing_decision_for_transcript(text: str, context: TranscriptContext | None = None) -> tuple[str, str, int]:
    cleaned = clean_transcript(text or "")
    normalized = cleaned.strip().lower().replace("’", "'")
    stripped = normalized.rstrip()
    if _is_clear_short_commit(cleaned):
        return "commit", "none", 0
    if stripped.endswith(","):
        return "extend_wait", "trailing_comma", _endpointing_wait_extension_ms()
    stripped_no_punct = stripped.rstrip(".!?").strip()
    filler_phrases = ("so", "like", "i mean", "you know", "because", "and")
    if stripped_no_punct in filler_phrases or any(stripped_no_punct.endswith(f" {phrase}") for phrase in filler_phrases):
        return "extend_wait", "filler_phrase", _endpointing_wait_extension_ms()
    if stripped_no_punct.startswith("now") and _word_count(stripped_no_punct) <= 3:
        return "extend_wait", "filler_phrase", _endpointing_wait_extension_ms()
    if context is not None:
        if context.detected_intent == "unclear_fragment":
            return "extend_wait", "unclear_fragment", _endpointing_wait_extension_ms()
        if context.ambiguity_detected and context.clarification_suggested:
            return "extend_wait", "unclear_fragment", _endpointing_wait_extension_ms()
    if _word_count(stripped_no_punct) < 4:
        return "extend_wait", "short_fragment", _endpointing_wait_extension_ms()
    return "commit", "none", 0


def _is_system_or_developer_message(message: object) -> bool:
    role = str(getattr(message, "role", "")).strip().lower()
    return role in {"system", "developer"}


def _prune_turn_context_messages(turn_ctx: object, turn_id: int | None = None) -> tuple[int, int, int]:
    messages_owner = turn_ctx
    messages = getattr(turn_ctx, "messages", None)
    if callable(messages):
        try:
            messages = messages()
        except Exception:
            messages = None
    if messages is None:
        if isinstance(turn_ctx, list):
            messages = turn_ctx
        else:
            try:
                messages = list(turn_ctx)  # type: ignore[arg-type]
            except Exception:
                logger.info(
                    "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=%s context_window_turns=%s reason=messages_unavailable",
                    turn_id or _current_turn_id,
                    0,
                    0,
                    0,
                    CONTEXT_WINDOW_TURNS,
                )
                return 0, 0, 0
            messages_owner = None
    try:
        message_list = list(messages)
    except Exception as exc:
        logger.info(
            "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=%s context_window_turns=%s reason=messages_unavailable error=%s",
            turn_id or _current_turn_id,
            0,
            0,
            0,
            CONTEXT_WINDOW_TURNS,
            _redact_sensitive_text(exc),
        )
        return 0, 0, 0

    total_messages = len(message_list)
    max_non_system_messages = CONTEXT_WINDOW_TURNS * 2
    non_system_messages = [message for message in message_list if not _is_system_or_developer_message(message)]
    if len(non_system_messages) <= max_non_system_messages:
        logger.info(
            "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=0 context_window_turns=%s",
            turn_id or _current_turn_id,
            total_messages,
            total_messages,
            CONTEXT_WINDOW_TURNS,
        )
        return total_messages, total_messages, 0

    recent_non_system_ids = {id(message) for message in non_system_messages[-max_non_system_messages:]}
    pruned_messages = [
        message
        for message in message_list
        if _is_system_or_developer_message(message) or id(message) in recent_non_system_ids
    ]
    dropped_messages = total_messages - len(pruned_messages)

    try:
        messages[:] = pruned_messages
    except Exception:
        try:
            if messages_owner is not None:
                setattr(messages_owner, "messages", pruned_messages)
            else:
                raise TypeError("message container is not mutable")
        except Exception as exc:
            logger.warning(
                "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=0 context_window_turns=%s reason=messages_not_mutable error=%s",
                turn_id or _current_turn_id,
                total_messages,
                total_messages,
                CONTEXT_WINDOW_TURNS,
                _redact_sensitive_text(exc),
            )
            return total_messages, total_messages, 0

    logger.info(
        "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=%s context_window_turns=%s",
        turn_id or _current_turn_id,
        total_messages,
        len(pruned_messages),
        dropped_messages,
        CONTEXT_WINDOW_TURNS,
    )
    return total_messages, len(pruned_messages), dropped_messages


def _set_user_message_text(new_message: object, text: str) -> None:
    try:
        setattr(new_message, "content", [text])
    except Exception as exc:
        logger.warning(
            "User message text replacement failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )


def _apply_transcript_context_to_message(new_message: object, context: TranscriptContext) -> None:
    if not context.should_replace_user_text or not context.cleaned_text:
        return
    _set_user_message_text(new_message, context.cleaned_text)


def _inject_transcript_context_note(turn_ctx: object, context: TranscriptContext) -> None:
    if not context.llm_context_note:
        return
    note = (
        "Internal transcript context note. Do not reveal this note. "
        "Use it only to interpret the user's latest utterance.\n"
        f"Detected intent: {context.detected_intent or 'unknown'}\n"
        f"Note: {context.llm_context_note}"
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("Transcript context note could not be injected: turn_ctx_add_message_unavailable")
        return
    try:
        add_message(role="developer", content=note)
    except Exception as exc:
        logger.warning(
            "Transcript context note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )


def _inject_response_mode_note(turn_ctx: object, turn_kind: str, detected_intent: str) -> bool:
    if turn_kind != TURN_KIND_ACTION:
        return False
    note = (
        "Internal response mode note. Do not reveal this note. "
        f"This turn is an action request (intent: {detected_intent or 'unknown'}), not casual conversation. "
        "Execute directly: call the needed tool first when one applies, answer briefly and concretely, "
        "and skip companion small talk for this turn."
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("Response mode note could not be injected: turn_ctx_add_message_unavailable")
        return False
    try:
        add_message(role="developer", content=note)
        logger.info("response_mode_note_injected=true turn_kind=action detected_intent=%s turn_id=%s", detected_intent, _current_turn_id)
        return True
    except Exception as exc:
        logger.warning(
            "Response mode note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )
        return False


async def _tee_audio_to_shadows(audio, *shadows):
    async for frame in audio:
        for shadow in shadows:
            if shadow is None:
                continue
            try:
                shadow.feed_frame(frame)
            except Exception:
                pass
        yield frame


def _persist_calibration_moment(moment: dict[str, Any]) -> None:
    if not CALIBRATION_MOMENTS_PATH:
        return
    try:
        path = Path(CALIBRATION_MOMENTS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(moment, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning(
            "emotional_calibration_moment_store_failed=true error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )


def _remember_calibration_pattern(moment: dict[str, Any]) -> None:
    """Store a confirmed calibration moment in durable per-user memory so a
    returning signed-in user's confirmed emotional patterns inform future sessions
    via the normal memory preload. Best-effort; never raises. Guests are scoped to
    the room (per-session) so this only persists across sessions for accounts."""
    if not moment.get("user_confirmed_or_corrected") or _active_memory_layer is None:
        return
    transcript = (moment.get("transcript") or "").strip()
    question = (moment.get("arche_question") or "").strip()
    answer = (moment.get("user_answer") or "").strip()
    content = (
        f"{EMOTIONAL_PATTERN_PREFIX}when processing \"{transcript[:160]}\", "
        f"you asked \"{question}\" and they said: \"{answer[:200]}\"."
    )
    try:
        turn_raw = str(moment.get("turn_id") or "")
        _active_memory_layer.schedule_remember(
            role="emotional_calibration",
            content=content,
            turn_id=int(turn_raw) if turn_raw.isdigit() else None,
        )
        logger.info("emotional_calibration_pattern_remembered=true turn_id=%s", moment.get("turn_id"))
    except Exception as exc:
        logger.warning(
            "emotional_calibration_pattern_remember_failed=true error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )


def _complete_pending_calibration_moment(user_answer: str) -> None:
    global _pending_calibration_moment
    if _pending_calibration_moment is None:
        return
    moment = dict(_pending_calibration_moment)
    moment["user_answer"] = user_answer
    moment["user_confirmed_or_corrected"] = bool(user_answer.strip())
    _calibration_moments.append(moment)
    _persist_calibration_moment(moment)
    _remember_calibration_pattern(moment)
    logger.info(
        "emotional_calibration_moment_stored=true turn_id=%s session_id=%s user_answer_present=%s total_moments=%s",
        moment.get("turn_id"),
        moment.get("session_id"),
        bool(user_answer.strip()),
        len(_calibration_moments),
    )
    _pending_calibration_moment = None


def _calibration_question_for_turn(transcript: str, profile, turn_id: int) -> tuple[str | None, str]:
    if profile is None:
        return None, "no_inworld_context"
    if turn_id - _last_calibration_question_turn_id < 3:
        return None, "cadence_limit"
    words = transcript.split()
    if len(words) < 4:
        return None, "transcript_too_short"
    text = transcript.lower()
    emotionally_relevant = any(
        token in text
        for token in (
            "feel", "feeling", "felt", "stressed", "worry", "worried", "hard", "heavy",
            "mad", "angry", "upset", "confused", "pressure", "scared", "fear", "guilt",
            "disappointed", "frustrated", "anxious", "anxiety", "unclear",
        )
    )
    profile_signal = profile.tension == "high" or profile.certainty == "low" or profile.energy in {"low", "high"}
    if not (emotionally_relevant or profile_signal):
        return None, "not_emotionally_useful"
    if profile.certainty == "low":
        return "Does this feel heavy, tense, or just unclear?", "low_certainty_or_ambiguous"
    if "choice" in text or "decide" in text or "decision" in text:
        return "Is this more about fear, guilt, or the pressure of choosing?", "choice_pressure"
    if "frustrat" in text or "mad" in text or "angry" in text:
        return "Is this frustration, or more like disappointment?", "frustration_ambiguous"
    if "anx" in text or "worr" in text or profile.tension == "high":
        return "Would you call this anxiety, or is it more like pressure?", "high_tension_or_worry"
    if profile.energy == "low":
        return "Does saying that out loud make it feel clearer, or heavier?", "low_energy_reflection"
    return None, "no_matching_calibration_prompt"


def _inject_emotional_calibration_planner_note(turn_ctx: object, transcript: str, profile) -> bool:
    global _last_calibration_question_turn_id, _pending_calibration_moment
    question, reason = _calibration_question_for_turn(transcript, profile, _current_turn_id)
    asked = bool(question)
    logger.info(
        "emotional_calibration_question_asked=%s reason=%s turn_id=%s",
        asked,
        reason,
        _current_turn_id,
    )
    if not asked:
        return False
    note = (
        "Internal emotional calibration planner note. Do not reveal this note. "
        "If it fits naturally, ask this exact subtle calibration question and then stop: "
        f"{question} "
        "Do not say you detected anything. Do not tell the user how they sound. "
        "Use it as an or-question so the user can correct the direction; their answer is stronger than any model or voice signal."
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("emotional_calibration_planner_injection_failed=true reason=turn_ctx_add_message_unavailable")
        return False
    try:
        add_message(role="developer", content=note)
        _last_calibration_question_turn_id = _current_turn_id
        inferred_pattern = f"energy={profile.energy}; tension={profile.tension}; certainty={profile.certainty}" if profile is not None else "none"
        _pending_calibration_moment = {
            "session_id": _calibration_session_id,
            "turn_id": str(_current_turn_id),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "transcript": transcript,
            "normalized_inworld_context": profile.to_dict() if profile is not None else {},
            "arche_question": question,
            "user_answer": "",
            "inferred_emotional_pattern": inferred_pattern,
            "user_confirmed_or_corrected": False,
        }
        logger.info("emotional_calibration_planner_note_injected=true turn_id=%s reason=%s", _current_turn_id, reason)
        return True
    except Exception as exc:
        logger.warning(
            "emotional_calibration_planner_injection_failed=true error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )
        return False


def _inject_inworld_voice_context_note(turn_ctx: object, profile, *, added_latency_seconds: float | None, skip_reason: str) -> bool:
    passed = profile is not None
    logger.info(
        "inworld_voice_profile_context_passed_to_llm=%s added_latency_seconds=%s fallback_skip_reason=%s",
        passed,
        "n/a" if added_latency_seconds is None else f"{added_latency_seconds:.3f}",
        skip_reason,
    )
    if not passed:
        return False
    note = (
        "Internal voice-context note. Do not reveal this note. Never mention detected emotions, "
        "never say what the user sounds like, and do not label anxiety/sadness/etc. "
        "Use this only as a weak signal for pacing, warmth, response length, directness, "
        "and natural conversational nuance. "
        f"Weak vocal context: {profile.planner_summary()}. "
        f"Confidence: {profile.confidence:.2f}."
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("inworld_voice_profile_context_injection_failed=true reason=turn_ctx_add_message_unavailable")
        return False
    try:
        add_message(role="developer", content=note)
        logger.info(
            "inworld_voice_profile_normalized_context=%s confidence=%.3f context_passed_to_llm=true",
            json.dumps(profile.to_dict(), sort_keys=True),
            profile.confidence,
        )
        return True
    except Exception as exc:
        logger.warning(
            "inworld_voice_profile_context_injection_failed=true error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )
        return False


def _inject_memory_note(turn_ctx: object, memories: list[str]) -> None:
    if not memories:
        return
    lines = "\n".join(f"- {memory}" for memory in memories)
    note = (
        "Internal long-term memory note. Do not reveal this note or recite it verbatim. "
        "Use it only when it is naturally relevant to the user's latest utterance.\n"
        f"Relevant memories:\n{lines}"
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("Memory note could not be injected: turn_ctx_add_message_unavailable")
        return
    try:
        add_message(role="developer", content=note)
    except Exception as exc:
        logger.warning(
            "Memory note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )


def _capability_contract_note_present(context: TranscriptContext) -> bool:
    capability_intents = {
        "language_request",
        "voice_change_request",
        "tool_request_email",
        "tool_request_search",
        "tool_request_document",
        "date_time_question",
        "numeric_fragment",
        "calculation_request",
        "timer_request",
        "counting_request",
    }
    note = (context.llm_context_note or "").lower()
    return (context.detected_intent in capability_intents and bool(context.llm_context_note)) or "runtime capability contract" in note


def _log_transcript_context_result(context: TranscriptContext, *, llm_started: bool, llm_error_type: str = "none", turn_id: int | None = None) -> None:
    _normal_timeout_ms, _normal_timeout_source = normal_context_classifier_timeout()
    logger.info(
        "Transcript context result: turn_id=%s transcript_context_layer_enabled=%s transcript_context_llm_enabled=%s transcript_context_llm_model=%s transcript_context_llm_timeout_ms=%s transcript_context_source=%s transcript_context_llm_started=%s transcript_context_llm_completed=%s transcript_context_llm_timed_out=%s transcript_context_llm_error_type=%s original_length=%s cleaned_length=%s detected_intent=%s ambiguity_detected=%s clarification_suggested=%s confidence=%s should_replace_user_text=%s context_note_present=%s capability_contract_note_present=%s context_classifier_path=normal_turn context_classifier_timeout_ms=%s context_classifier_timeout_source=%s context_classifier_model=%s",
        turn_id or _current_turn_id,
        transcript_context_layer_enabled(),
        transcript_context_llm_enabled(),
        transcript_context_llm_model(),
        transcript_context_llm_timeout_ms(),
        context.source,
        llm_started,
        context.source == "llm",
        context.source == "deterministic_timeout_fallback",
        llm_error_type,
        len(context.original_text),
        len(context.cleaned_text),
        context.detected_intent or "none",
        context.ambiguity_detected,
        context.clarification_suggested,
        f"{context.confidence:.2f}",
        context.should_replace_user_text,
        bool(context.llm_context_note),
        _capability_contract_note_present(context),
        _normal_timeout_ms,
        _normal_timeout_source,
        transcript_context_llm_model() if transcript_context_llm_enabled() else "deterministic",
    )
    if transcript_context_debug():
        logger.info(
            "Transcript context debug: original_preview=%s cleaned_preview=%s note_preview=%s",
            _redact_sensitive_text(context.original_text)[:200],
            _redact_sensitive_text(context.cleaned_text)[:200],
            _redact_sensitive_text(context.llm_context_note or "")[:200],
        )


def _search_policy_for_intent(intent: str | None, clarification_suggested: bool) -> tuple[bool, str]:
    normalized = (intent or "unknown").strip().lower()
    blocked_intents = {
        "unclear_fragment",
        "numeric_fragment",
        "language_request",
        "counting_request",
        "calculation_request",
        "pronunciation_correction",
        "voice_change_request",
        "date_time_question",
        "memory_recall_request",
    }
    if normalized == "tool_request_search":
        if clarification_suggested:
            return False, "blocked_unclear_fragment"
        return True, "clear_search_intent"
    if normalized in blocked_intents:
        return False, "blocked_non_lookup_intent" if normalized != "unclear_fragment" else "blocked_unclear_fragment"
    # Default: only an explicit search ask reaches the allow path above. Casual
    # turns and the catch-all "unknown" intent no longer auto-trigger search,
    # which is what produced surprise lookups on plain statements. The legacy
    # permissive behavior (any non-blocked intent may LLM-call search) is still
    # available behind SEARCH_REQUIRE_EXPLICIT_INTENT=false.
    if SEARCH_REQUIRE_EXPLICIT_INTENT:
        return False, "blocked_no_explicit_search_intent"
    return True, "llm_tool_call"


def _metadata_candidates_from_context(ctx: JobContext) -> list[Any]:
    candidates: list[Any] = []
    for attr_name in ("job", "room"):
        obj = _safe_attr(ctx, attr_name)
        metadata = _safe_attr(obj, "metadata")
        if metadata:
            candidates.append(metadata)

    job = _safe_attr(ctx, "job")
    for attr_name in ("participant", "participant_info"):  # wrapper versions differ
        participant = _safe_attr(job, attr_name)
        metadata = _safe_attr(participant, "metadata")
        if metadata:
            candidates.append(metadata)

    room = _safe_attr(ctx, "room")
    remote_participants = _safe_attr(room, "remote_participants")
    if isinstance(remote_participants, dict):
        for participant in remote_participants.values():
            metadata = _safe_attr(participant, "metadata")
            if metadata:
                candidates.append(metadata)

    return candidates


class LucyAgent(Agent):
    def __init__(
        self,
        runtime_context: RuntimeContext | None = None,
        memory_layer: MemoryLayer | None = None,
        memory_preload_note: str | None = None,
    ) -> None:
        self.runtime_context = runtime_context
        self.memory_layer = memory_layer
        instruction_parts = [SYSTEM_PROMPT, RUNTIME_CAPABILITY_CONTRACT]
        if runtime_context is not None:
            instruction_parts.append(runtime_context.system_message)
        if memory_preload_note:
            instruction_parts.append(memory_preload_note)
        instructions = "\n\n".join(part for part in instruction_parts if part)
        super().__init__(instructions=instructions)

    @function_tool(name="internet_search", description=SEARCH_TOOL_DESCRIPTION)
    async def internet_search_tool(self, query: str, max_results: int = 5) -> str:
        """Search the internet with Exa for current or external information."""
        provider = search_provider()
        disabled_reason = search_disabled_reason()
        runtime_context = self.runtime_context
        turn_id = _current_turn_id
        search_allowed = _current_turn_search_allowed
        search_allowed_reason = _current_turn_search_allowed_reason
        current_date = runtime_context.current_date if runtime_context else None
        current_datetime_iso = runtime_context.current_datetime_iso if runtime_context else None
        session_timezone = runtime_context.session_timezone if runtime_context else None
        _, freshness_applied = build_effective_search_query(query, current_date=current_date)
        try:
            requested_max_results = max(1, int(max_results or search_max_results()))
        except Exception:
            requested_max_results = search_max_results()
        capped_max_results = min(requested_max_results, search_max_results())
        logger.info(
            "search_tool_called turn_id=%s search_turn_id=%s search_provider=%s search_query_original=%s search_current_date=%s search_current_datetime_iso=%s search_timezone=%s search_freshness_applied=%s search_disabled_reason=%s search_pre_ack_spoken=%s requested_max_results=%s capped_max_results=%s search_allowed_for_turn=%s search_allowed_reason=%s detected_intent=%s",
            turn_id,
            turn_id,
            provider,
            _redact_sensitive_text(query),
            current_date or "unknown",
            current_datetime_iso or "unknown",
            session_timezone or "unknown",
            freshness_applied,
            disabled_reason or "none",
            False,
            max_results,
            capped_max_results,
            search_allowed,
            search_allowed_reason,
            _current_turn_transcript_intent,
        )
        if not search_allowed:
            output = "I need one more detail before searching. What exactly should I look up?"
            logger.warning(
                "search_blocked_by_turn_policy turn_id=%s search_allowed_for_turn=%s search_allowed_reason=%s detected_intent=%s search_query_original=%s",
                turn_id,
                search_allowed,
                search_allowed_reason,
                _current_turn_transcript_intent,
                _redact_sensitive_text(query),
            )
            return output
        if disabled_reason is not None:
            logger.warning(
                "search_disabled_reason=%s search_provider=%s search_query_original=%s search_current_date=%s search_current_datetime_iso=%s search_timezone=%s search_freshness_applied=%s search_specific_failure_response_used=%s",
                disabled_reason,
                provider,
                _redact_sensitive_text(query),
                current_date or "unknown",
                current_datetime_iso or "unknown",
                session_timezone or "unknown",
                freshness_applied,
                True,
            )
            output = f"{SEARCH_DISABLED_MESSAGE} Say: {search_failure_response()}"
            _mark_search_wait_completed(failed=True, output=output, result_handoff_spoken=False, turn_id=turn_id)
            return output

        _mark_search_wait_started(pre_ack_spoken=False, turn_id=turn_id, query=query)
        logger.info(
            "search_wait_started turn_id=%s search_turn_id=%s search_in_progress=%s search_started_at=%s search_pre_ack_spoken=%s search_query_original=%s search_allowed_for_turn=%s search_allowed_reason=%s",
            turn_id,
            _search_turn_id,
            _search_in_progress,
            _search_started_at,
            _search_pre_ack_spoken,
            _redact_sensitive_text(query),
            search_allowed,
            search_allowed_reason,
        )
        results = await internet_search(
            query=query,
            max_results=capped_max_results,
            current_date=current_date,
            current_datetime_iso=current_datetime_iso,
            session_timezone=session_timezone,
        )
        if not results:
            output = f"{SEARCH_DISABLED_MESSAGE} Say: {search_failure_response()}"
            completion_applied = _mark_search_wait_completed(failed=True, output=output, result_handoff_spoken=False, turn_id=turn_id)
            if not completion_applied:
                return "Search result ignored because a newer user turn started. Do not speak this stale result."
            logger.info(
                "search_result_count=0 turn_id=%s search_turn_id=%s search_provider=exa search_query_original=%s search_current_date=%s search_current_datetime_iso=%s search_timezone=%s search_freshness_applied=%s search_result_handoff_spoken=%s search_wait_completed search_in_progress=%s search_failed=%s search_specific_failure_response_used=%s search_latency_seconds=%.3f",
                turn_id,
                _search_turn_id,
                _redact_sensitive_text(query),
                current_date or "unknown",
                current_datetime_iso or "unknown",
                session_timezone or "unknown",
                freshness_applied,
                False,
                _search_in_progress,
                _search_failed,
                True,
                _search_wait_elapsed_seconds(),
            )
        else:
            output = format_search_results_for_voice(results, current_date=current_date, freshness_applied=freshness_applied)
            search_elapsed = _search_wait_elapsed_seconds()
            if search_elapsed < SEARCH_BRIDGE_MIN_DELAY_SECONDS:
                output = output.replace(
                    "If you did not already say a lookup bridge before the tool call, say:",
                    "Search returned quickly; do not say a separate lookup bridge. Start with the result handoff instead of:",
                    1,
                )
            completion_applied = _mark_search_wait_completed(failed=False, output=output, result_handoff_spoken=False, turn_id=turn_id)
            if not completion_applied:
                return "Search result ignored because a newer user turn started. Do not speak this stale result."
            logger.info(
                "search_result_count=%s turn_id=%s search_turn_id=%s search_provider=exa search_query_original=%s search_current_date=%s search_current_datetime_iso=%s search_timezone=%s search_freshness_applied=%s search_result_dates=%s search_result_handoff_spoken=%s search_wait_completed search_in_progress=%s search_failed=%s search_specific_failure_response_used=%s search_latency_seconds=%.3f search_bridge_min_delay_seconds=%.3f",
                len(results),
                turn_id,
                _search_turn_id,
                _redact_sensitive_text(query),
                current_date or "unknown",
                current_datetime_iso or "unknown",
                session_timezone or "unknown",
                freshness_applied,
                ",".join(result.published_date for result in results if result.published_date) or "none",
                False,
                _search_in_progress,
                _search_failed,
                False,
                _search_wait_elapsed_seconds(),
                SEARCH_BRIDGE_MIN_DELAY_SECONDS,
            )
        return _search_result_authority_gate(output, turn_id)

def _safe_metadata_preview(value: Any) -> str:
    if value is None:
        return "none"
    return _redact_sensitive_text(value)[:500]


def _metadata_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value.keys())
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        if isinstance(parsed, dict):
            return sorted(str(key) for key in parsed.keys())
    return []


def _metadata_debug_entries_from_context(ctx: JobContext) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    for attr_name in ("job", "room"):
        obj = _safe_attr(ctx, attr_name)
        metadata = _safe_attr(obj, "metadata")
        entries.append((f"{attr_name}.metadata", metadata))

    job = _safe_attr(ctx, "job")
    for attr_name in ("participant", "participant_info"):
        participant = _safe_attr(job, attr_name)
        metadata = _safe_attr(participant, "metadata")
        entries.append((f"job.{attr_name}.metadata", metadata))

    room = _safe_attr(ctx, "room")
    remote_participants = _safe_attr(room, "remote_participants")
    if isinstance(remote_participants, dict):
        for participant_id, participant in remote_participants.items():
            metadata = _safe_attr(participant, "metadata")
            entries.append((f"room.remote_participants.{participant_id}.metadata", metadata))
    return entries


def _chat_ctx_items(chat_ctx: object) -> list[object]:
    """Return a safe, concrete message list from LiveKit chat context shapes."""
    if chat_ctx is None:
        return []
    messages = getattr(chat_ctx, "messages", None)
    if callable(messages):
        try:
            messages = messages()
        except Exception:
            messages = None
    if messages is None:
        messages = chat_ctx
    try:
        return list(messages)  # type: ignore[arg-type]
    except Exception:
        return []


def _extract_latest_user_text_from_chat_ctx(chat_ctx: object) -> str:
    iterable = _chat_ctx_items(chat_ctx)
    if not iterable:
        return _last_user_message_text
    for message in reversed(iterable):
        role = str(getattr(message, "role", "")).lower()
        if role and role != "user":
            continue
        text = _extract_text_for_debug(message).strip()
        if text:
            return text
    return _last_user_message_text



def _recent_turn_previews_from_chat_ctx(chat_ctx: object, limit: int = 5) -> list[str]:
    iterable = _chat_ctx_items(chat_ctx)
    if not iterable:
        return []
    previews: list[str] = []
    for message in iterable[-limit:]:
        role = str(getattr(message, "role", "unknown"))
        text = _extract_text_for_debug(message).strip()
        if not text:
            continue
        previews.append(f"{role}: {_redact_sensitive_text(text)[:240]}")
    return previews


@dataclass(frozen=True)
class TurnPolicyResult:
    decision: str
    classification: str
    confidence: float
    reason: str
    should_start_generation: bool
    should_merge_held_fragment: bool
    should_clear_held_fragment: bool


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", clean_transcript(text or "").lower())


def _semantic_overlap(left: str, right: str) -> float:
    stop = {"the", "a", "an", "and", "or", "but", "so", "like", "i", "you", "it", "that", "this", "to", "of", "in", "on", "for", "with", "me", "my", "is", "was", "are", "were"}
    left_words = {word for word in _normalized_words(left) if word not in stop and len(word) > 2}
    right_words = {word for word in _normalized_words(right) if word not in stop and len(word) > 2}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(1, min(len(left_words), len(right_words)))


def _classify_turn_transcript(text: str, context: TranscriptContext | None = None) -> tuple[str, float, str]:
    cleaned = clean_transcript(text or "").strip()
    lowered = cleaned.lower().replace("’", "'")
    words = _normalized_words(cleaned)
    word_count = len(words)
    if not cleaned:
        return "UNCLEAR_AUDIO", 0.2, "empty_transcript"
    if context is not None and context.detected_intent == "unclear_fragment" and context.confidence < 0.5:
        return "UNCLEAR_AUDIO", 0.55, "transcript_context_unclear_low_confidence"
    if re.search(r"\b(interrupt|interrupted|not answering|didn't answer|did not answer|answer me|why aren't you|why are you|lag|lagged|slow|silence|silent|stuck|repeat|repeating|again|heard me|listening)\b", lowered):
        return "META_COMPLAINT", 0.86, "system_or_latency_complaint"
    if re.search(r"\b(i feel|i felt|i'm feeling|i am feeling|i hate|i love|i miss|i'm scared|i am scared|i'm worried|i am worried|it hurt|that hurt|got under my skin|annoyed|upset|sad|angry|afraid|lonely|tired|overwhelmed|stressed)\b", lowered):
        return "EMOTIONAL_STATEMENT", 0.88, "emotional_meaning"
    if context is not None and context.detected_intent in {"date_time_question", "tool_request_search", "tool_request_email", "tool_request_document", "calculation_request", "counting_request", "timer_request", "stop_or_cancel_request"}:
        return "COMPLETE_THOUGHT", 0.86, f"actionable_intent:{context.detected_intent}"
    if word_count <= 2 and lowered.rstrip(".!?") in {"yeah", "yes", "no", "okay", "ok", "right", "sure", "thanks", "thank you"}:
        return "LOW_INFORMATION_FILLER", 0.78, "backchannel_or_ack"
    if lowered.endswith(",") or re.search(r"\b(and|because|when|if|but|so|like|while|although|since|then)$", lowered.rstrip(".,!?")):
        return "INCOMPLETE_THOUGHT", 0.82, "dangling_clause_or_connector"
    if word_count < 4 and context is not None and context.ambiguity_detected and context.clarification_suggested:
        return "INCOMPLETE_THOUGHT", 0.72, "short_ambiguous_fragment"
    if word_count >= 4 or cleaned.endswith((".", "?", "!")):
        return "COMPLETE_THOUGHT", 0.8, "complete_shape"
    if word_count <= 3:
        return "INCOMPLETE_THOUGHT", 0.58, "short_without_clear_completion"
    return "COMPLETE_THOUGHT", 0.65, "default_complete"


def _make_turn_policy_decision(text: str, context: TranscriptContext | None = None, *, held_text: str = "", held_created_at: float = 0.0, now: float | None = None) -> TurnPolicyResult:
    now = now if now is not None else time.monotonic()
    classification, confidence, reason = _classify_turn_transcript(text, context)
    held_age = (now - held_created_at) if held_text and held_created_at > 0 else None
    has_mergeable_held = bool(held_text and held_age is not None and held_age <= TURN_HOLD_FRAGMENT_MERGE_WINDOW_SECONDS)
    if classification == "UNCLEAR_AUDIO":
        return TurnPolicyResult("ASK_FOR_AUDIO_CLARIFICATION", classification, confidence, reason, True, False, True)
    if classification == "META_COMPLAINT":
        return TurnPolicyResult("RECOVER_FROM_SILENCE", classification, confidence, reason, True, False, True)
    if classification == "LOW_INFORMATION_FILLER":
        return TurnPolicyResult("IGNORE_LOW_INFORMATION_FILLER", classification, confidence, reason, False, False, False)
    if has_mergeable_held:
        if _semantic_overlap(held_text, text) >= 0.2 or classification == "INCOMPLETE_THOUGHT":
            return TurnPolicyResult("MERGE_WITH_HELD_FRAGMENT", classification, confidence, "semantic_continuation", True, True, True)
        return TurnPolicyResult("FLUSH_HELD_AND_COMMIT_NEW", classification, confidence, "new_topic_or_unrelated_to_held_fragment", True, False, True)
    if classification == "INCOMPLETE_THOUGHT":
        return TurnPolicyResult("HOLD_FOR_CONTINUATION", classification, confidence, reason, False, False, False)
    return TurnPolicyResult("COMMIT_NOW", classification, confidence, reason, True, False, True)


def _fallback_requires_user_repeat(reason: str, classification: str) -> bool:
    return classification == "UNCLEAR_AUDIO" or reason in {"audio_unclear", "stt_unclear"}


def _fallback_text_for_reason(reason: str, classification: str) -> str:
    if _fallback_requires_user_repeat(reason, classification):
        return "I caught part of that, but not cleanly — could you say the last bit again?"
    if classification == "META_COMPLAINT":
        return "You’re right — I lagged there."
    if classification == "EMOTIONAL_STATEMENT":
        return "I’m with you."
    if reason in {"first_token_timeout", "total_timeout_no_text", "provider_error", "empty_stream"}:
        return "One second — I’m catching up."
    return LLM_FALLBACK_RESPONSE


def _is_transport_api_connection_error(error: object) -> bool:
    current: object | None = error
    for _ in range(4):
        if current is None:
            return False
        name = type(current).__name__.lower()
        if "apiconnectionerror" in name or "connection" in name and "error" in name:
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _should_retry_openrouter_connection_error(error: object, *, first_token_seen: bool, chunk_count: int, text_length: int, llm_turn_id: int, tts_started_for_turn: bool) -> tuple[bool, str]:
    if not _is_transport_api_connection_error(error):
        return False, "not_transport_api_connection_error"
    if first_token_seen:
        return False, "first_token_seen"
    if chunk_count > 0:
        return False, "chunks_already_received"
    if text_length > 0:
        return False, "text_already_received"
    if llm_turn_id != _current_turn_id:
        return False, "stale_turn"
    if tts_started_for_turn:
        return False, "tts_already_started"
    return True, "eligible"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def _generic_fallback_suppression_reason(fallback_turn_id: int, fallback_started_at: float) -> str | None:
    if (
        _latest_user_state_for_greeting == "speaking"
        or _current_turn_id != fallback_turn_id
        or _latest_user_speaking_at > fallback_started_at
        or (_latest_stt_partial_at > fallback_started_at and _latest_stt_partial_at > _latest_stt_final_at)
    ):
        return "user_speaking_or_newer_turn_pending"
    return None


def _endpointing_wait_extension_ms() -> int:
    lower = min(ENDPOINTING_WAIT_EXTENSION_MIN_MS, ENDPOINTING_WAIT_EXTENSION_MAX_MS)
    upper = max(ENDPOINTING_WAIT_EXTENSION_MIN_MS, ENDPOINTING_WAIT_EXTENSION_MAX_MS)
    return max(lower, min(upper, 900))


def _is_clear_short_commit(text: str) -> bool:
    cleaned = clean_transcript(text).strip().lower()
    normalized = cleaned.rstrip(".!?").strip()
    if not normalized:
        return False
    if cleaned.endswith("?"):
        return True
    if normalized in {"yes", "no", "yeah", "yep", "nope", "okay", "ok", "stop", "send it"}:
        return True
    if re.search(r"\b(what time is it|what day is it|what'?s today|count to|search that|look that up|stop|send it)\b", normalized):
        return True
    return False


def _is_stale_llm_turn(llm_turn_id: int) -> bool:
    return _current_turn_id != llm_turn_id


def _endpointing_decision_for_transcript(text: str, context: TranscriptContext | None = None) -> tuple[str, str, int]:
    cleaned = clean_transcript(text or "")
    normalized = cleaned.strip().lower().replace("’", "'")
    stripped = normalized.rstrip()
    if _is_clear_short_commit(cleaned):
        return "commit", "none", 0
    if stripped.endswith(","):
        return "extend_wait", "trailing_comma", _endpointing_wait_extension_ms()
    stripped_no_punct = stripped.rstrip(".!?").strip()
    filler_phrases = ("so", "like", "i mean", "you know", "because", "and")
    if stripped_no_punct in filler_phrases or any(stripped_no_punct.endswith(f" {phrase}") for phrase in filler_phrases):
        return "extend_wait", "filler_phrase", _endpointing_wait_extension_ms()
    if stripped_no_punct.startswith("now") and _word_count(stripped_no_punct) <= 3:
        return "extend_wait", "filler_phrase", _endpointing_wait_extension_ms()
    if context is not None:
        if context.detected_intent == "unclear_fragment":
            return "extend_wait", "unclear_fragment", _endpointing_wait_extension_ms()
        if context.ambiguity_detected and context.clarification_suggested:
            return "extend_wait", "unclear_fragment", _endpointing_wait_extension_ms()
    if _word_count(stripped_no_punct) < 4:
        return "extend_wait", "short_fragment", _endpointing_wait_extension_ms()
    return "commit", "none", 0


def _is_system_or_developer_message(message: object) -> bool:
    role = str(getattr(message, "role", "")).strip().lower()
    return role in {"system", "developer"}


def _prune_turn_context_messages(turn_ctx: object, turn_id: int | None = None) -> tuple[int, int, int]:
    messages_owner = turn_ctx
    messages = getattr(turn_ctx, "messages", None)
    if callable(messages):
        try:
            messages = messages()
        except Exception:
            messages = None
    if messages is None:
        if isinstance(turn_ctx, list):
            messages = turn_ctx
        else:
            try:
                messages = list(turn_ctx)  # type: ignore[arg-type]
            except Exception:
                logger.info(
                    "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=%s context_window_turns=%s reason=messages_unavailable",
                    turn_id or _current_turn_id,
                    0,
                    0,
                    0,
                    CONTEXT_WINDOW_TURNS,
                )
                return 0, 0, 0
            messages_owner = None
    try:
        message_list = list(messages)
    except Exception as exc:
        logger.info(
            "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=%s context_window_turns=%s reason=messages_unavailable error=%s",
            turn_id or _current_turn_id,
            0,
            0,
            0,
            CONTEXT_WINDOW_TURNS,
            _redact_sensitive_text(exc),
        )
        return 0, 0, 0

    total_messages = len(message_list)
    max_non_system_messages = CONTEXT_WINDOW_TURNS * 2
    non_system_messages = [message for message in message_list if not _is_system_or_developer_message(message)]
    if len(non_system_messages) <= max_non_system_messages:
        logger.info(
            "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=0 context_window_turns=%s",
            turn_id or _current_turn_id,
            total_messages,
            total_messages,
            CONTEXT_WINDOW_TURNS,
        )
        return total_messages, total_messages, 0

    recent_non_system_ids = {id(message) for message in non_system_messages[-max_non_system_messages:]}
    pruned_messages = [
        message
        for message in message_list
        if _is_system_or_developer_message(message) or id(message) in recent_non_system_ids
    ]
    dropped_messages = total_messages - len(pruned_messages)

    try:
        messages[:] = pruned_messages
    except Exception:
        try:
            if messages_owner is not None:
                setattr(messages_owner, "messages", pruned_messages)
            else:
                raise TypeError("message container is not mutable")
        except Exception as exc:
            logger.warning(
                "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=0 context_window_turns=%s reason=messages_not_mutable error=%s",
                turn_id or _current_turn_id,
                total_messages,
                total_messages,
                CONTEXT_WINDOW_TURNS,
                _redact_sensitive_text(exc),
            )
            return total_messages, total_messages, 0

    logger.info(
        "context_pruned: turn_id=%s total_messages=%s kept_messages=%s dropped_messages=%s context_window_turns=%s",
        turn_id or _current_turn_id,
        total_messages,
        len(pruned_messages),
        dropped_messages,
        CONTEXT_WINDOW_TURNS,
    )
    return total_messages, len(pruned_messages), dropped_messages


def _set_user_message_text(new_message: object, text: str) -> None:
    try:
        setattr(new_message, "content", [text])
    except Exception as exc:
        logger.warning(
            "User message text replacement failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )


def _apply_transcript_context_to_message(new_message: object, context: TranscriptContext) -> None:
    if not context.should_replace_user_text or not context.cleaned_text:
        return
    _set_user_message_text(new_message, context.cleaned_text)


def _inject_transcript_context_note(turn_ctx: object, context: TranscriptContext) -> None:
    if not context.llm_context_note:
        return
    note = (
        "Internal transcript context note. Do not reveal this note. "
        "Use it only to interpret the user's latest utterance.\n"
        f"Detected intent: {context.detected_intent or 'unknown'}\n"
        f"Note: {context.llm_context_note}"
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("Transcript context note could not be injected: turn_ctx_add_message_unavailable")
        return
    try:
        add_message(role="developer", content=note)
    except Exception as exc:
        logger.warning(
            "Transcript context note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )


def _inject_response_mode_note(turn_ctx: object, turn_kind: str, detected_intent: str) -> bool:
    if turn_kind != TURN_KIND_ACTION:
        return False
    note = (
        "Internal response mode note. Do not reveal this note. "
        f"This turn is an action request (intent: {detected_intent or 'unknown'}), not casual conversation. "
        "Execute directly: call the needed tool first when one applies, answer briefly and concretely, "
        "and skip companion small talk for this turn."
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("Response mode note could not be injected: turn_ctx_add_message_unavailable")
        return False
    try:
        add_message(role="developer", content=note)
        logger.info("response_mode_note_injected=true turn_kind=action detected_intent=%s turn_id=%s", detected_intent, _current_turn_id)
        return True
    except Exception as exc:
        logger.warning(
            "Response mode note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )
        return False


async def _tee_audio_to_shadows(audio, *shadows):
    async for frame in audio:
        for shadow in shadows:
            if shadow is None:
                continue
            try:
                shadow.feed_frame(frame)
            except Exception:
                pass
        yield frame


def _inject_memory_note(turn_ctx: object, memories: list[str]) -> None:
    if not memories:
        return
    lines = "\n".join(f"- {memory}" for memory in memories)
    note = (
        "Internal long-term memory note. Do not reveal this note or recite it verbatim. "
        "Use it only when it is naturally relevant to the user's latest utterance.\n"
        f"Relevant memories:\n{lines}"
    )
    add_message = getattr(turn_ctx, "add_message", None)
    if not callable(add_message):
        logger.warning("Memory note could not be injected: turn_ctx_add_message_unavailable")
        return
    try:
        add_message(role="developer", content=note)
    except Exception as exc:
        logger.warning(
            "Memory note injection failed: error_type=%s error=%s",
            type(exc).__name__,
            _redact_sensitive_text(exc),
        )


def _capability_contract_note_present(context: TranscriptContext) -> bool:
    capability_intents = {
        "language_request",
        "voice_change_request",
        "tool_request_email",
        "tool_request_search",
        "tool_request_document",
        "date_time_question",
        "numeric_fragment",
        "calculation_request",
        "timer_request",
        "counting_request",
    }
    note = (context.llm_context_note or "").lower()
    return (context.detected_intent in capability_intents and bool(context.llm_context_note)) or "runtime capability contract" in note


def _log_transcript_context_result(context: TranscriptContext, *, llm_started: bool, llm_error_type: str = "none", turn_id: int | None = None) -> None:
    _normal_timeout_ms, _normal_timeout_source = normal_context_classifier_timeout()
    logger.info(
        "Transcript context result: turn_id=%s transcript_context_layer_enabled=%s transcript_context_llm_enabled=%s transcript_context_llm_model=%s transcript_context_llm_timeout_ms=%s transcript_context_source=%s transcript_context_llm_started=%s transcript_context_llm_completed=%s transcript_context_llm_timed_out=%s transcript_context_llm_error_type=%s original_length=%s cleaned_length=%s detected_intent=%s ambiguity_detected=%s clarification_suggested=%s confidence=%s should_replace_user_text=%s context_note_present=%s capability_contract_note_present=%s context_classifier_path=normal_turn context_classifier_timeout_ms=%s context_classifier_timeout_source=%s context_classifier_model=%s",
        turn_id or _current_turn_id,
        transcript_context_layer_enabled(),
        transcript_context_llm_enabled(),
        transcript_context_llm_model(),
        transcript_context_llm_timeout_ms(),
        context.source,
        llm_started,
        context.source == "llm",
        context.source == "deterministic_timeout_fallback",
        llm_error_type,
        len(context.original_text),
        len(context.cleaned_text),
        context.detected_intent or "none",
        context.ambiguity_detected,
        context.clarification_suggested,
        f"{context.confidence:.2f}",
        context.should_replace_user_text,
        bool(context.llm_context_note),
        _capability_contract_note_present(context),
        _normal_timeout_ms,
        _normal_timeout_source,
        transcript_context_llm_model() if transcript_context_llm_enabled() else "deterministic",
    )
    if transcript_context_debug():
        logger.info(
            "Transcript context debug: original_preview=%s cleaned_preview=%s note_preview=%s",
            _redact_sensitive_text(context.original_text)[:200],
            _redact_sensitive_text(context.cleaned_text)[:200],
            _redact_sensitive_text(context.llm_context_note or "")[:200],
        )


def _search_policy_for_intent(intent: str | None, clarification_suggested: bool) -> tuple[bool, str]:
    normalized = (intent or "unknown").strip().lower()
    blocked_intents = {
        "unclear_fragment",
        "numeric_fragment",
        "language_request",
        "counting_request",
        "calculation_request",
        "pronunciation_correction",
        "voice_change_request",
        "date_time_question",
        "memory_recall_request",
    }
    if normalized == "tool_request_search":
        if clarification_suggested:
            return False, "blocked_unclear_fragment"
        return True, "clear_search_intent"
    if normalized in blocked_intents:
        return False, "blocked_non_lookup_intent" if normalized != "unclear_fragment" else "blocked_unclear_fragment"
    # Default: only an explicit search ask reaches the allow path above. Casual
    # turns and the catch-all "unknown" intent no longer auto-trigger search,
    # which is what produced surprise lookups on plain statements. The legacy
    # permissive behavior (any non-blocked intent may LLM-call search) is still
    # available behind SEARCH_REQUIRE_EXPLICIT_INTENT=false.
    if SEARCH_REQUIRE_EXPLICIT_INTENT:
        return False, "blocked_no_explicit_search_intent"
    return True, "llm_tool_call"


def _metadata_candidates_from_context(ctx: JobContext) -> list[Any]:
    candidates: list[Any] = []
    for attr_name in ("job", "room"):
        obj = _safe_attr(ctx, attr_name)
        metadata = _safe_attr(obj, "metadata")
        if metadata:
            candidates.append(metadata)

    job = _safe_attr(ctx, "job")
    for attr_name in ("participant", "participant_info"):  # wrapper versions differ
        participant = _safe_attr(job, attr_name)
        metadata = _safe_attr(participant, "metadata")
        if metadata:
            candidates.append(metadata)

    room = _safe_attr(ctx, "room")
    remote_participants = _safe_attr(room, "remote_participants")
    if isinstance(remote_participants, dict):
        for participant in remote_participants.values():
            metadata = _safe_attr(participant, "metadata")
            if metadata:
                candidates.append(metadata)

    return candidates


class LucyAgent(Agent):
    def __init__(
        self,
        runtime_context: RuntimeContext | None = None,
        memory_layer: MemoryLayer | None = None,
        memory_preload_note: str | None = None,
    ) -> None:
        self.runtime_context = runtime_context
        self.memory_layer = memory_layer
        instruction_parts = [SYSTEM_PROMPT, RUNTIME_CAPABILITY_CONTRACT]
        if runtime_context is not None:
            instruction_parts.append(runtime_context.system_message)
        if memory_preload_note:
            instruction_parts.append(memory_preload_note)
        instructions = "\n\n".join(part for part in instruction_parts if part)
        super().__init__(instructions=instructions)

    @function_tool(name="internet_search", description=SEARCH_TOOL_DESCRIPTION)
    async def internet_search_tool(self, query: str, max_results: int = 5) -> str:
        """Search the internet with Exa for current or external information."""
        provider = search_provider()
        disabled_reason = search_disabled_reason()
        runtime_context = self.runtime_context
        turn_id = _current_turn_id
        search_allowed = _current_turn_search_allowed
        search_allowed_reason = _current_turn_search_allowed_reason
        current_date = runtime_context.current_date if runtime_context else None
        current_datetime_iso = runtime_context.current_datetime_iso if runtime_context else None
        session_timezone = runtime_context.session_timezone if runtime_context else None
        _, freshness_applied = build_effective_search_query(query, current_date=current_date)
        try:
            requested_max_results = max(1, int(max_results or search_max_results()))
        except Exception:
            requested_max_results = search_max_results()
        capped_max_results = min(requested_max_results, search_max_results())
        logger.info(
            "search_tool_called turn_id=%s search_turn_id=%s search_provider=%s search_query_original=%s search_current_date=%s search_current_datetime_iso=%s search_timezone=%s search_freshness_applied=%s search_disabled_reason=%s search_pre_ack_spoken=%s requested_max_results=%s capped_max_results=%s search_allowed_for_turn=%s search_allowed_reason=%s detected_intent=%s",
            turn_id,
            turn_id,
            provider,
            _redact_sensitive_text(query),
            current_date or "unknown",
            current_datetime_iso or "unknown",
            session_timezone or "unknown",
            freshness_applied,
            disabled_reason or "none",
            False,
            max_results,
            capped_max_results,
            search_allowed,
            search_allowed_reason,
            _current_turn_transcript_intent,
        )
        if not search_allowed:
            output = "I need one more detail before searching. What exactly should I look up?"
            logger.warning(
                "search_blocked_by_turn_policy turn_id=%s search_allowed_for_turn=%s search_allowed_reason=%s detected_intent=%s search_query_original=%s",
                turn_id,
                search_allowed,
                search_allowed_reason,
                _current_turn_transcript_intent,
                _redact_sensitive_text(query),
            )
            return output
        if disabled_reason is not None:
            logger.warning(
                "search_disabled_reason=%s search_provider=%s search_query_original=%s search_current_date=%s search_current_datetime_iso=%s search_timezone=%s search_freshness_applied=%s search_specific_failure_response_used=%s",
                disabled_reason,
                provider,
                _redact_sensitive_text(query),
                current_date or "unknown",
                current_datetime_iso or "unknown",
                session_timezone or "unknown",
                freshness_applied,
                True,
            )
            output = f"{SEARCH_DISABLED_MESSAGE} Say: {search_failure_response()}"
            _mark_search_wait_completed(failed=True, output=output, result_handoff_spoken=False, turn_id=turn_id)
            return output

        _mark_search_wait_started(pre_ack_spoken=False, turn_id=turn_id, query=query)
        logger.info(
            "search_wait_started turn_id=%s search_turn_id=%s search_in_progress=%s search_started_at=%s search_pre_ack_spoken=%s search_query_original=%s search_allowed_for_turn=%s search_allowed_reason=%s",
            turn_id,
            _search_turn_id,
            _search_in_progress,
            _search_started_at,
            _search_pre_ack_spoken,
            _redact_sensitive_text(query),
            search_allowed,
            search_allowed_reason,
        )
        results = await internet_search(
            query=query,
            max_results=capped_max_results,
            current_date=current_date,
            current_datetime_iso=current_datetime_iso,
            session_timezone=session_timezone,
        )
        if not results:
            output = f"{SEARCH_DISABLED_MESSAGE} Say: {search_failure_response()}"
            completion_applied = _mark_search_wait_completed(failed=True, output=output, result_handoff_spoken=False, turn_id=turn_id)
            if not completion_applied:
                return "Search result ignored because a newer user turn started. Do not speak this stale result."
            logger.info(
                "search_result_count=0 turn_id=%s search_turn_id=%s search_provider=exa search_query_original=%s search_current_date=%s search_current_datetime_iso=%s search_timezone=%s search_freshness_applied=%s search_result_handoff_spoken=%s search_wait_completed search_in_progress=%s search_failed=%s search_specific_failure_response_used=%s search_latency_seconds=%.3f",
                turn_id,
                _search_turn_id,
                _redact_sensitive_text(query),
                current_date or "unknown",
                current_datetime_iso or "unknown",
                session_timezone or "unknown",
                freshness_applied,
                False,
                _search_in_progress,
                _search_failed,
                True,
                _search_wait_elapsed_seconds(),
            )
        else:
            output = format_search_results_for_voice(results, current_date=current_date, freshness_applied=freshness_applied)
            search_elapsed = _search_wait_elapsed_seconds()
            if search_elapsed < SEARCH_BRIDGE_MIN_DELAY_SECONDS:
                output = output.replace(
                    "If you did not already say a lookup bridge before the tool call, say:",
                    "Search returned quickly; do not say a separate lookup bridge. Start with the result handoff instead of:",
                    1,
                )
            completion_applied = _mark_search_wait_completed(failed=False, output=output, result_handoff_spoken=False, turn_id=turn_id)
            if not completion_applied:
                return "Search result ignored because a newer user turn started. Do not speak this stale result."
            logger.info(
                "search_result_count=%s turn_id=%s search_turn_id=%s search_provider=exa search_query_original=%s search_current_date=%s search_current_datetime_iso=%s search_timezone=%s search_freshness_applied=%s search_result_dates=%s search_result_handoff_spoken=%s search_wait_completed search_in_progress=%s search_failed=%s search_specific_failure_response_used=%s search_latency_seconds=%.3f search_bridge_min_delay_seconds=%.3f",
                len(results),
                turn_id,
                _search_turn_id,
                _redact_sensitive_text(query),
                current_date or "unknown",
                current_datetime_iso or "unknown",
                session_timezone or "unknown",
                freshness_applied,
                ",".join(result.published_date for result in results if result.published_date) or "none",
                False,
                _search_in_progress,
                _search_failed,
                False,
                _search_wait_elapsed_seconds(),
                SEARCH_BRIDGE_MIN_DELAY_SECONDS,
            )
        return _search_result_authority_gate(output, turn_id)

    def _normalize_spoken_text(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return normalized
        if "```" in normalized:
            return normalized
        if normalized[-1] not in {".", "?", "!", "…"}:
            return normalized + "."
        return normalized

    async def stt_node(self, audio, model_settings):
        # Observational AudioInteraction fork: tee user audio frames to the shadow
        # sidecar without altering the production STT stream. feed_frame never
        # blocks or raises; when shadow mode is off the stream passes through as-is.
        shadows = tuple(s for s in (_audiointeraction_shadow, _inworld_voice_profile_shadow) if s is not None)
        if shadows:
            audio = _tee_audio_to_shadows(audio, *shadows)
        async for event in Agent.default.stt_node(self, audio, model_settings):
            yield event

    def tts_node(self, text: AsyncIterable[str], model_settings):
        global _latest_normalized_text_hash, _last_tts_request_start_at, _last_tts_first_audio_at, _last_tts_text_length, _last_tts_sentence_end_count, _last_tts_path, _last_tts_node_entered_at, _last_tts_received_text_hash, _last_hume_request_start_at, _last_tts_completed_at, _last_tts_raw_chunk_count, _last_tts_normalized_yield_count, _last_tts_first_input_at
        _last_tts_node_entered_at = time.monotonic()
        _last_tts_completed_at = 0.0
        _last_tts_raw_chunk_count = 0
        _last_tts_normalized_yield_count = 0
        _last_tts_first_input_at = None
        logger.info(
            "TTS node entered: turn_id=%s TTS_PROVIDER=%s SPOKEN_TEXT_NORMALIZATION=%s spoken_text_normalization_mode=%s tts_input_buffering_mode=%s handoff_guard_enabled=%s llm_stream_status=%s",
            _current_turn_id,
            TTS_PROVIDER,
            SPOKEN_TEXT_NORMALIZATION,
            SPOKEN_TEXT_NORMALIZATION_MODE,
            SPOKEN_TEXT_NORMALIZATION_MODE,
            LLM_TO_TTS_HANDOFF_GUARD_ENABLED,
            _last_llm_stream_status,
        )

        if not SPOKEN_TEXT_NORMALIZATION:
            logger.info("Spoken text normalization enabled=false")

            async def _logging_passthrough_stream() -> AsyncIterable[str]:
                global _last_tts_received_text_hash, _last_tts_text_length, _last_tts_sentence_end_count, _last_tts_raw_chunk_count, _last_tts_normalized_yield_count, _last_tts_first_input_at
                chunks: list[str] = []
                count = 0
                yielded_any = False
                suppressed = False
                try:
                    async for chunk in text:
                        count += 1
                        if _last_tts_first_input_at is None:
                            _last_tts_first_input_at = time.monotonic()
                        # Before any audio is produced, drop the whole reply if the
                        # user resumed with a real utterance during thinking. Once a
                        # chunk has been yielded, the normal interruption path owns it.
                        if not yielded_any and not suppressed:
                            sup, sup_reason = _handoff_guard_should_suppress(_current_turn_id)
                            if sup:
                                suppressed = True
                                logger.warning(
                                    "stale_thinking_response_suppressed=true tts_path=passthrough turn_id=%s reason=%s",
                                    _current_turn_id,
                                    sup_reason,
                                )
                        if suppressed:
                            continue  # drain the source so upstream closes, yield nothing
                        if isinstance(chunk, str):
                            chunks.append(chunk)
                        yielded_any = True
                        yield chunk
                finally:
                    raw_text = "".join(chunks)
                    _last_tts_received_text_hash = _text_hash(raw_text)
                    _last_tts_text_length = len(raw_text)
                    _last_tts_sentence_end_count = sum(raw_text.count(mark) for mark in (".", "?", "!", "…"))
                    _last_tts_raw_chunk_count = count
                    _last_tts_normalized_yield_count = count
                    logger.info(
                        "TTS node received text chunk count: raw_chunk_count=%s raw_total_length=%s text_hash=%s handoff_guard_enabled=%s",
                        count,
                        len(raw_text),
                        _last_tts_received_text_hash,
                        LLM_TO_TTS_HANDOFF_GUARD_ENABLED,
                    )
                    if not raw_text.strip():
                        logger.warning(
                            "TTS node completed with empty text stream: llm_stream_status=%s fallback_used=%s",
                            _last_llm_stream_status,
                            _last_llm_fallback_response_used,
                        )
                    if TTS_TEXT_DEBUG:
                        preview = _redact_sensitive_text(raw_text)[:200]
                        logger.info(
                            "TTS text debug: raw_chunk_count=%s raw_total_length=%s raw_preview=%s final_preview=%s",
                            count,
                            len(raw_text),
                            preview,
                            preview,
                        )

            async def _default_tts_with_hume_logs() -> AsyncIterable[Any]:
                global _last_hume_request_start_at, _last_tts_path, _last_tts_first_audio_at, _last_tts_request_start_at, _last_tts_completed_at
                start = time.monotonic()
                _last_tts_request_start_at = start
                _last_tts_first_audio_at = None
                _last_tts_path = None
                frame_count = 0
                coverage: HumeSpeechAudioCoverage | None = None
                if TTS_PROVIDER == "hume":
                    _last_hume_request_start_at = start
                    coverage = _start_hume_speech_audio_coverage(
                        speech_id=_latest_current_speech_id_for_hume,
                        turn_id=_current_turn_id,
                        path="default_agent_tts_node_fallback",
                        normalized_text_hash="normalization_false",
                    )
                    dedupe_key, duplicate, seen_count = _record_hume_request_metadata(
                        path="default_agent_tts_node_fallback",
                        speech_id=_latest_current_speech_id_for_hume,
                        normalized_text_hash="normalization_false",
                        feeds_playout=True,
                    )
                    logger.info(
                        "Hume TTS HTTP request starting: path=default_agent_tts_node_fallback normalization=false latest_agent_state=%s current_speech_id=%s hume_request_dedupe_key=%s hume_duplicate_request_detected=%s hume_request_seen_count=%s hume_request_feeds_playout=%s hume_retry=%s retry_reason=%s",
                        _latest_agent_state_for_hume,
                        _latest_current_speech_id_for_hume,
                        dedupe_key,
                        duplicate,
                        seen_count,
                        True,
                        False,
                        "none",
                    )
                try:
                    async for out in Agent.default.tts_node(self, _logging_passthrough_stream(), model_settings):
                        if _last_tts_first_audio_at is None:
                            _last_tts_first_audio_at = time.monotonic()
                        if _last_tts_path is None:
                            _last_tts_path = "default_agent_tts_node_fallback"
                        frame_count += 1
                        _record_hume_speech_audio_frame(coverage, out)
                        yield out
                    _last_tts_completed_at = time.monotonic()
                    if TTS_PROVIDER == "hume":
                        finalized = _finalize_hume_speech_audio_coverage(coverage)
                        if finalized is not None:
                            _hume_speech_audio_coverages[finalized.speech_id] = finalized
                        logger.info(
                            "Hume TTS HTTP request completed: path=default_agent_tts_node_fallback frame_count_yielded=%s time_to_first_audio_seconds=%s total_tts_seconds=%.3f",
                            frame_count,
                            _fmt_seconds((_last_tts_first_audio_at - start) if _last_tts_first_audio_at is not None else None),
                            _last_tts_completed_at - start,
                        )
                except Exception as e:
                    if TTS_PROVIDER == "hume":
                        _finalize_hume_speech_audio_coverage(coverage, error_type=type(e).__name__)
                        logger.error(
                            "Hume TTS HTTP request error: path=default_agent_tts_node_fallback error_type=%s error=%s frame_count_yielded=%s total_tts_seconds=%.3f",
                            type(e).__name__,
                            _redact_sensitive_text(e),
                            frame_count,
                            time.monotonic() - start,
                        )
                    raise

            return _default_tts_with_hume_logs()

        logger.info("Spoken text normalization enabled=true mode=buffered_full_segment")

        async def _direct_or_plugin_or_default() -> AsyncIterable[Any]:
            global _latest_normalized_text_hash, _last_tts_request_start_at, _last_tts_first_audio_at, _last_tts_text_length, _last_tts_sentence_end_count, _last_tts_path, _last_hume_request_start_at, _last_tts_received_text_hash, _last_tts_completed_at, _last_tts_raw_chunk_count, _last_tts_normalized_yield_count, _last_tts_first_input_at
            chunks: list[str] = []
            chunk_count = 0
            async for chunk in text:
                chunk_count += 1
                if _last_tts_first_input_at is None:
                    _last_tts_first_input_at = time.monotonic()
                chunks.append(chunk if isinstance(chunk, str) else str(chunk))
            raw_text = "".join(chunks)
            _last_tts_received_text_hash = _text_hash(raw_text)
            _last_tts_raw_chunk_count = chunk_count
            logger.info(
                "TTS node received text chunk count: raw_chunk_count=%s raw_total_length=%s text_hash=%s handoff_guard_enabled=%s",
                chunk_count,
                len(raw_text),
                _last_tts_received_text_hash,
                LLM_TO_TTS_HANDOFF_GUARD_ENABLED,
            )
            sanitized = _sanitize_spoken_laughter(raw_text)
            if raw_text.strip() and _last_llm_stream_status == "stale_turn":
                # A stale stream may have emitted a chunk or two before
                # staleness was detected; never speak that fragment.
                logger.warning(
                    "TTS stale stream text dropped: raw_chunk_count=%s raw_total_length=%s llm_stream_status=%s preview=%s",
                    chunk_count,
                    len(raw_text),
                    _last_llm_stream_status,
                    _redact_sensitive_text(raw_text)[:120],
                )
                sanitized = ""
            handoff_suppress, handoff_reason = _handoff_guard_should_suppress(_current_turn_id)
            if raw_text.strip() and sanitized and handoff_suppress:
                # User resumed with a real utterance while Arche was still thinking
                # and no audio had started; drop the stale reply before TTS.
                logger.warning(
                    "stale_thinking_response_suppressed=true tts_path=normalized turn_id=%s raw_total_length=%s reason=%s",
                    _current_turn_id,
                    len(raw_text),
                    handoff_reason,
                )
                sanitized = ""
            normalized_text = self._normalize_spoken_text(sanitized)
            normalized_hash = _text_hash(normalized_text)
            _latest_normalized_text_hash = normalized_hash
            _normalized_text_hash_ctx.set(normalized_hash)
            sentence_end_count = sum(normalized_text.count(mark) for mark in (".", "?", "!", "…"))
            newline_count = normalized_text.count("\n")
            _last_tts_text_length = len(normalized_text)
            _last_tts_sentence_end_count = sentence_end_count
            _last_tts_normalized_yield_count = 1 if normalized_text else 0
            _last_tts_request_start_at = time.monotonic()
            _last_tts_first_audio_at = None
            _last_tts_path = None
            logger.info(
                "TTS normalized yield diagnostics: tts_normalized_yield_count=%s raw_chunk_count=%s raw_total_length=%s normalized_text_length=%s normalized_text_preview=%s normalized_text_hash=%s sentence_end_count=%s newline_count=%s SPOKEN_TEXT_NORMALIZATION=%s spoken_text_normalization_mode=%s tts_input_buffering_mode=%s time_from_llm_first_token_to_first_tts_input=%s TTS_PROVIDER=%s HUME_INSTANT_MODE=%s HUME_SPEED=%s HUME_TRAILING_SILENCE=%s",
                _last_tts_normalized_yield_count, chunk_count, len(raw_text), len(normalized_text), _redact_sensitive_text(normalized_text)[:200], normalized_hash,
                sentence_end_count, newline_count, SPOKEN_TEXT_NORMALIZATION, SPOKEN_TEXT_NORMALIZATION_MODE, SPOKEN_TEXT_NORMALIZATION_MODE, _fmt_seconds((_last_tts_first_input_at - _last_llm_first_token_at) if (_last_tts_first_input_at is not None and _last_llm_first_token_at is not None) else None), TTS_PROVIDER, env_bool("HUME_INSTANT_MODE", True), os.getenv("HUME_SPEED", "0.9"), os.getenv("HUME_TRAILING_SILENCE", "0.25"),
            )
            if not normalized_text.strip():
                logger.warning(
                    "TTS normalized text empty: raw_chunk_count=%s llm_stream_status=%s fallback_used=%s",
                    chunk_count,
                    _last_llm_stream_status,
                    _last_llm_fallback_response_used,
                )
            if TTS_TEXT_DEBUG:
                logger.info("TTS text debug: raw_chunk_count=%s raw_total_length=%s raw_preview=%s final_preview=%s", chunk_count, len(raw_text), _redact_sensitive_text(raw_text)[:200], _redact_sensitive_text(normalized_text)[:200])

            async def _single_text_stream() -> AsyncIterable[str]:
                if normalized_text:
                    yield normalized_text

            if TTS_PROVIDER == "hume" and HUME_DIRECT_API_TTS:
                # experimental Plan B direct path; fallback must reuse preserved normalized_text
                pass
            else:
                logger.info("Direct Hume TTS attempt: hume_direct_api_tts_requested=%s", False)

            if TTS_PROVIDER == "hume" and HUME_FULL_UTTERANCE_TTS:
                activity = getattr(self, "_activity", None)
                activity_tts = getattr(activity, "tts", None) if activity is not None else None
                synthesize_fn = getattr(activity_tts, "synthesize", None)
                if callable(synthesize_fn) and normalized_text:
                    start = time.monotonic()
                    _last_hume_request_start_at = start
                    yielded = 0
                    first_audio = None
                    coverage: HumeSpeechAudioCoverage | None = None
                    try:
                        sig = inspect.signature(synthesize_fn)
                        if "conn_options" in sig.parameters:
                            conn_opts = getattr(getattr(activity, "session", None), "conn_options", None)
                            tts_conn_options = getattr(conn_opts, "tts_conn_options", None)
                            chunked_stream = synthesize_fn(normalized_text, conn_options=tts_conn_options)
                        else:
                            chunked_stream = synthesize_fn(normalized_text)
                        dedupe_key, duplicate, seen_count = _record_hume_request_metadata(
                            path="livekit_hume_plugin_synthesize_full_text",
                            speech_id=_latest_current_speech_id_for_hume,
                            normalized_text_hash=normalized_hash,
                            feeds_playout=True,
                        )
                        coverage = _start_hume_speech_audio_coverage(
                            speech_id=_latest_current_speech_id_for_hume,
                            turn_id=_current_turn_id,
                            path="livekit_hume_plugin_synthesize_full_text",
                            normalized_text_hash=normalized_hash,
                        )
                        logger.info(
                            "Hume TTS HTTP request starting: path=livekit_hume_plugin_synthesize_full_text text_hash=%s text_length=%s latest_agent_state=%s current_speech_id=%s hume_request_dedupe_key=%s hume_duplicate_request_detected=%s hume_request_seen_count=%s hume_request_feeds_playout=%s hume_retry=%s retry_reason=%s",
                            normalized_hash,
                            len(normalized_text),
                            _latest_agent_state_for_hume,
                            _latest_current_speech_id_for_hume,
                            dedupe_key,
                            duplicate,
                            seen_count,
                            True,
                            False,
                            "none",
                        )
                        logger.info("Hume full-utterance mode: full_utterance_requested=%s full_utterance_supported=%s full_utterance_used=%s path=%s fallback_reason=%s", True, True, True, "livekit_hume_plugin_synthesize_full_text", "none")
                        logger.info(
                            "Hume full-utterance path engaged: hume_full_utterance_requested=true hume_full_utterance_supported=true hume_full_utterance_used=true hume_direct_api_tts_requested=%s path=livekit_hume_plugin_synthesize_full_text",
                            str(HUME_DIRECT_API_TTS).lower(),
                        )
                        _last_tts_path = "livekit_hume_plugin_synthesize_full_text"
                        async for event in chunked_stream:
                            frame = getattr(event, "frame", None)
                            if frame is None:
                                continue
                            if first_audio is None:
                                first_audio = time.monotonic()
                                _last_tts_first_audio_at = first_audio
                            yielded += 1
                            _record_hume_speech_audio_frame(coverage, frame)
                            yield frame
                        _last_tts_completed_at = time.monotonic()
                        finalized = _finalize_hume_speech_audio_coverage(coverage)
                        if finalized is not None:
                            _hume_speech_audio_coverages[finalized.speech_id] = finalized
                        logger.info("Hume TTS HTTP request completed: path=livekit_hume_plugin_synthesize_full_text frame_count_yielded=%s time_to_first_audio_seconds=%.3f total_tts_seconds=%.3f", yielded, (first_audio-start) if first_audio else -1.0, _last_tts_completed_at-start)
                        logger.info("Hume full-utterance plugin result: hume_full_utterance_plugin_requested=%s hume_full_utterance_plugin_used=%s hume_full_utterance_plugin_fallback_reason=%s path=%s normalized_text_hash=%s text_length=%s sentence_end_count=%s frame_count_yielded=%s time_to_first_audio_seconds=%.3f total_tts_seconds=%.3f", True, True, "none", "livekit_hume_plugin_synthesize_full_text", normalized_hash, len(normalized_text), sentence_end_count, yielded, (first_audio-start) if first_audio else -1.0, time.monotonic()-start)
                        return
                    except Exception as e:
                        _finalize_hume_speech_audio_coverage(coverage, error_type=type(e).__name__)
                        logger.error("Hume TTS HTTP request error: path=livekit_hume_plugin_synthesize_full_text error_type=%s error=%s frame_count_yielded=%s total_tts_seconds=%.3f", type(e).__name__, _redact_sensitive_text(e), yielded, time.monotonic()-start)
                        if yielded > 0:
                            logger.warning("Hume full-utterance plugin partial failure: hume_full_utterance_plugin_requested=%s hume_full_utterance_plugin_used=%s hume_full_utterance_plugin_fallback_reason=%s frame_count_yielded=%s", True, True, _redact_sensitive_text(e), yielded)
                            return
                        logger.warning("Hume full-utterance plugin fallback: hume_full_utterance_plugin_requested=%s hume_full_utterance_plugin_used=%s hume_full_utterance_plugin_fallback_reason=%s path=%s", True, False, _redact_sensitive_text(e), "default_agent_tts_node_fallback")
                else:
                    fu_reason = "empty_normalized_text" if not normalized_text else "activity_tts_synthesize_unavailable"
                    logger.info("Hume full-utterance mode: full_utterance_requested=%s full_utterance_supported=%s full_utterance_used=%s path=%s fallback_reason=%s", True, False, False, "default_agent_tts_node_fallback", fu_reason)
                    logger.warning(
                        "Hume full-utterance path unavailable: hume_full_utterance_requested=true hume_full_utterance_supported=false hume_full_utterance_used=false hume_full_utterance_unsupported_reason=%s path=default_agent_tts_node_fallback",
                        fu_reason,
                    )
            elif TTS_PROVIDER == "hume":
                logger.info("Hume full-utterance mode: full_utterance_requested=%s full_utterance_supported=%s full_utterance_used=%s path=%s fallback_reason=%s", False, False, False, "default_agent_tts_node_fallback", "not_requested")
                logger.info(
                    "Hume full-utterance path not requested: hume_full_utterance_requested=false hume_full_utterance_supported=unknown hume_full_utterance_used=false fallback_reason=not_requested path=default_agent_tts_node_fallback hint=set_HUME_FULL_UTTERANCE_TTS=true_to_bypass_default_tts_node_sentence_splitting",
                )

            try:
                hume_start = time.monotonic()
                frame_count = 0
                coverage: HumeSpeechAudioCoverage | None = None
                if TTS_PROVIDER == "hume":
                    _last_hume_request_start_at = hume_start
                    coverage = _start_hume_speech_audio_coverage(
                        speech_id=_latest_current_speech_id_for_hume,
                        turn_id=_current_turn_id,
                        path="default_agent_tts_node_fallback",
                        normalized_text_hash=normalized_hash,
                    )
                    dedupe_key, duplicate, seen_count = _record_hume_request_metadata(
                        path="default_agent_tts_node_fallback",
                        speech_id=_latest_current_speech_id_for_hume,
                        normalized_text_hash=normalized_hash,
                        feeds_playout=True,
                    )
                    logger.info(
                        "Hume TTS HTTP request starting: path=default_agent_tts_node_fallback text_hash=%s text_length=%s latest_agent_state=%s current_speech_id=%s hume_request_dedupe_key=%s hume_duplicate_request_detected=%s hume_request_seen_count=%s hume_request_feeds_playout=%s hume_retry=%s retry_reason=%s",
                        normalized_hash,
                        len(normalized_text),
                        _latest_agent_state_for_hume,
                        _latest_current_speech_id_for_hume,
                        dedupe_key,
                        duplicate,
                        seen_count,
                        True,
                        False,
                        "none",
                    )
                async for out in Agent.default.tts_node(self, _single_text_stream(), model_settings):
                    if _last_tts_first_audio_at is None:
                        _last_tts_first_audio_at = time.monotonic()
                    if _last_tts_path is None:
                        _last_tts_path = "default_agent_tts_node_fallback"
                    frame_count += 1
                    _record_hume_speech_audio_frame(coverage, out)
                    yield out
                _last_tts_completed_at = time.monotonic()
                if TTS_PROVIDER == "hume":
                    finalized = _finalize_hume_speech_audio_coverage(coverage)
                    if finalized is not None:
                        _hume_speech_audio_coverages[finalized.speech_id] = finalized
                    logger.info(
                        "Hume TTS HTTP request completed: path=default_agent_tts_node_fallback frame_count_yielded=%s time_to_first_audio_seconds=%s total_tts_seconds=%.3f",
                        frame_count,
                        _fmt_seconds((_last_tts_first_audio_at - hume_start) if _last_tts_first_audio_at is not None else None),
                        _last_tts_completed_at - hume_start,
                    )
            except Exception as e:
                if TTS_PROVIDER == "hume":
                    _finalize_hume_speech_audio_coverage(coverage, error_type=type(e).__name__)
                    logger.error(
                        "Hume TTS HTTP request error: path=default_agent_tts_node_fallback error_type=%s error=%s tts_path=%s total_tts_seconds=%.3f",
                        type(e).__name__,
                        _redact_sensitive_text(e),
                        _last_tts_path or "n/a",
                        time.monotonic() - _last_hume_request_start_at if _last_hume_request_start_at else -1.0,
                    )
                logger.error(
                    "tts_stream_status=error error_type=%s error=%s tts_path=%s",
                    type(e).__name__,
                    _redact_sensitive_text(e),
                    _last_tts_path or "n/a",
                )
                raise

        if TTS_PROVIDER == "hume" and SPOKEN_TEXT_NORMALIZATION and HUME_DIRECT_API_TTS:
            async def _direct_hume_or_fallback_stream() -> AsyncIterable[rtc.AudioFrame]:
                global _direct_hume_request_counter
                hume_api_key = os.getenv("HUME_API_KEY", "").strip()
                hume_voice_id = os.getenv("HUME_VOICE_ID", "").strip()
                if not hume_api_key or not hume_voice_id:
                    logger.warning("Direct Hume TTS fallback: hume_direct_api_tts_used=%s direct_api_fallback_reason=missing_required_hume_api_key_or_voice_id", False)
                    async for out in _direct_or_plugin_or_default():
                        yield out
                    return

                _direct_hume_request_counter += 1
                request_index = _direct_hume_request_counter
                endpoint_path = "/v0/tts/stream/file"
                yielded_count = 0
                start = time.monotonic()
                first_audio_at = None
                logger.info("Direct Hume TTS attempt: hume_direct_api_tts_requested=%s direct_hume_request_index=%s endpoint_used=%s", True, request_index, endpoint_path)
                try:
                    raise RuntimeError("direct_path_requires_preserved_normalized_text_buffer")
                except Exception as e:
                    if yielded_count > 0:
                        logger.warning("Direct Hume TTS ended after partial output: hume_direct_api_tts_used=%s direct_api_fallback_reason=%s frame_count_yielded=%s direct_hume_request_index=%s", True, _redact_sensitive_text(e), yielded_count, request_index)
                        return
                    logger.warning("Direct Hume TTS fallback: hume_direct_api_tts_used=%s direct_api_fallback_reason=%s endpoint_used=%s direct_hume_request_index=%s", False, _redact_sensitive_text(e), endpoint_path, request_index)
                    async for out in _direct_or_plugin_or_default():
                        if first_audio_at is None:
                            first_audio_at = time.monotonic()
                        yielded_count += 1
                        yield out
                    logger.info(
                        "Direct Hume TTS fallback completed via plugin/default: hume_direct_api_tts_used=%s frame_count_yielded=%s time_to_first_audio_seconds=%.3f total_tts_seconds=%.3f direct_hume_request_count=%s direct_hume_request_index=%s",
                        False,
                        yielded_count,
                        (first_audio_at - start) if first_audio_at is not None else -1.0,
                        time.monotonic() - start,
                        _direct_hume_request_counter,
                        request_index,
                    )

            return _with_post_speech_hold(_direct_hume_or_fallback_stream())
        return _with_post_speech_hold(_direct_or_plugin_or_default())

    def llm_node(self, chat_ctx, tools, model_settings):
        datetime_user_text = _extract_latest_user_text_from_chat_ctx(chat_ctx)
        # Honor a user request to switch languages: update the active language +
        # OmniVoice, and nudge the LLM to reply in it (skip if no real change).
        language_switch = _maybe_switch_language(datetime_user_text)
        if language_switch is not None:
            _lang_code, _lang_name = language_switch
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content=(
                    f"The user asked you to speak {_lang_name}. Respond only in {_lang_name} "
                    "from now on — including this reply — until they ask for another language. "
                    "Do not announce the switch or comment on their language."
                ),
            )
        datetime_intent = detect_datetime_intent(datetime_user_text)
        if datetime_intent and self.runtime_context is not None:
            async def _datetime_guard_stream():
                global _last_llm_start_at, _last_llm_first_token_at, _last_llm_complete_at, _last_llm_stream_status, _last_llm_timeout_stage, _last_llm_fallback_response_used, _pending_llm_fallback_text, _last_llm_completed_text, _last_llm_completed_text_hash, _last_llm_completed_at, _last_generic_llm_fallback_used, _llm_turn_id
                _llm_turn_id = _current_turn_id
                answer = answer_datetime_intent(self.runtime_context, datetime_intent)
                # Log the freshly-recomputed date/time (not the stale session-init
                # values) so observability matches what the user was actually told.
                fresh_date, fresh_time = current_datetime_snapshot(self.runtime_context)
                now = time.monotonic()
                _last_llm_start_at = now
                _last_llm_first_token_at = now
                _last_llm_complete_at = now
                _last_llm_completed_at = now
                _last_llm_stream_status = "datetime_guard"
                _last_llm_timeout_stage = "none"
                _last_llm_fallback_response_used = False
                _last_generic_llm_fallback_used = False
                _pending_llm_fallback_text = None
                _last_llm_completed_text = answer
                _last_llm_completed_text_hash = _text_hash(answer)
                logger.info(
                    "Date/time guard triggered: turn_id=%s datetime_guard_triggered=%s "
                    "datetime_intent=%s datetime_answer_source=%s search_called=%s "
                    "session_timezone=%s runtime_current_date=%s runtime_current_time=%s "
                    "text_length=%s",
                    _current_turn_id,
                    True,
                    datetime_intent,
                    "runtime_context",
                    False,
                    self.runtime_context.session_timezone,
                    fresh_date,
                    fresh_time,
                    len(answer),
                )
                yield answer
                logger.info(
                    "LLM stream ended: turn_id=%s llm_turn_id=%s search_turn_id=%s status=%s first_token_seen=%s chunk_count=%s text_length=%s fallback_used=%s timeout_stage=%s search_in_progress=%s search_tool_called=%s search_failed=%s generic_llm_fallback_used=%s",
                    _llm_turn_id,
                    _llm_turn_id,
                    _search_turn_id,
                    _last_llm_stream_status,
                    True,
                    1,
                    len(answer),
                    False,
                    "none",
                    _search_in_progress,
                    _search_tool_called,
                    _search_failed,
                    False,
                )
            return _datetime_guard_stream()

        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)

        async def _llm_stream():
            nonlocal stream
            global _last_llm_start_at, _last_llm_first_token_at, _last_llm_complete_at, _last_llm_stream_status, _last_llm_timeout_stage, _last_llm_fallback_response_used, _pending_llm_fallback_text, _last_llm_completed_text, _last_llm_completed_text_hash, _last_llm_completed_at, _last_generic_llm_fallback_used
            assistant_fragments: list[str] = []
            chunk_count = 0
            model_name = os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
            llm_turn_id = _current_turn_id
            _llm_turn_id = llm_turn_id
            _last_llm_stream_status = "started"
            _last_llm_timeout_stage = "none"
            _last_llm_fallback_response_used = False
            _last_generic_llm_fallback_used = False
            _pending_llm_fallback_text = None
            _last_llm_start_at = time.monotonic()
            _last_llm_first_token_at = None
            _last_llm_complete_at = 0.0
            _last_llm_completed_text = ""
            _last_llm_completed_text_hash = "empty"
            _last_llm_completed_at = 0.0
            start = _last_llm_start_at
            first_token_deadline = start + LLM_FIRST_TOKEN_TIMEOUT_SECONDS
            total_deadline = start + LLM_TOTAL_TIMEOUT_SECONDS
            it = stream.__aiter__()
            fallback_yielded = False
            first_token_timeout_extended_once = False
            openrouter_retry_attempted = False
            logger.info(
                "LLM stream starting: turn_id=%s llm_turn_id=%s openrouter_model=%s first_token_timeout_seconds=%s total_timeout_seconds=%s search_turn_id=%s timeout_config_source=first_token:%s,total:%s",
                llm_turn_id,
                llm_turn_id,
                model_name,
                LLM_FIRST_TOKEN_TIMEOUT_SECONDS,
                LLM_TOTAL_TIMEOUT_SECONDS,
                _search_turn_id,
                LLM_FIRST_TOKEN_TIMEOUT_SOURCE,
                LLM_TOTAL_TIMEOUT_SOURCE,
            )
            _interaction_state.on_llm_started()

            def _extract_text_delta(chunk: object) -> str:
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
                return text_delta if isinstance(text_delta, str) else ""

            async def _close_llm_stream(reason: str) -> None:
                close_fn = getattr(stream, "aclose", None)
                if not callable(close_fn):
                    close_fn = getattr(it, "aclose", None)
                if callable(close_fn):
                    try:
                        result = close_fn()
                        if inspect.isawaitable(result):
                            await result
                        logger.info("LLM stream close requested: reason=%s", reason)
                    except Exception as close_error:
                        logger.warning(
                            "LLM stream close failed: reason=%s error_type=%s error=%s",
                            reason,
                            type(close_error).__name__,
                            _redact_sensitive_text(close_error),
                        )

            async def _yield_fallback(reason: str) -> AsyncIterable[str]:
                nonlocal fallback_yielded
                global _pending_llm_fallback_text, _last_llm_fallback_response_used, _last_generic_llm_fallback_used
                search_turn_matches_current = _search_turn_matches_current() and _search_turn_id == llm_turn_id
                generic_suppression_reason = _generic_fallback_suppression_reason(llm_turn_id, start)
                if generic_suppression_reason == "user_speaking_or_newer_turn_pending":
                    _interaction_state.on_fallback_decision(allowed=False, reason="user_speaking_or_newer_turn_pending", requires_repeat=False)
                    logger.warning(
                        "fallback_decision_turn_id=%s fallback_suppressed=true fallback_suppressed_reason=user_speaking_or_newer_turn_pending latest_user_state=%s current_turn_id=%s user_speaking_after_fallback_turn_started=%s stt_partial_after_llm_start=%s generic_llm_fallback_used=%s",
                        llm_turn_id,
                        _latest_user_state_for_greeting,
                        _current_turn_id,
                        _latest_user_speaking_at > start,
                        _latest_stt_partial_at > start and _latest_stt_partial_at > _latest_stt_final_at,
                        False,
                    )
                    return
                if _search_in_progress and search_turn_matches_current:
                    logger.warning(
                        "fallback_decision_turn_id=%s fallback_suppressed=true fallback_suppressed_reason=search_in_progress search_in_progress=%s search_tool_called=%s search_turn_id=%s search_turn_matches_current=%s search_wait_elapsed_seconds=%.3f generic_llm_fallback_used=%s",
                        llm_turn_id,
                        _search_in_progress,
                        _search_tool_called,
                        _search_turn_id,
                        search_turn_matches_current,
                        _search_wait_elapsed_seconds(),
                        False,
                    )
                    return
                if _search_specific_response_produced and search_turn_matches_current:
                    logger.warning(
                        "fallback_decision_turn_id=%s fallback_suppressed=true fallback_suppressed_reason=search_specific_response_available search_in_progress=%s search_tool_called=%s search_failed=%s search_specific_failure_response_used=%s search_turn_id=%s search_turn_matches_current=%s generic_llm_fallback_used=%s",
                        llm_turn_id,
                        _search_in_progress,
                        _search_tool_called,
                        _search_failed,
                        _search_failed,
                        _search_turn_id,
                        search_turn_matches_current,
                        False,
                    )
                    return
                fallback_text = _fallback_text_for_reason(reason, _current_turn_policy_classification)
                requires_repeat = _fallback_requires_user_repeat(reason, _current_turn_policy_classification)
                if requires_repeat is False and "say that again" in fallback_text.lower():
                    logger.warning(
                        "repeat request emitted when STT transcript was good and failure was LLM timeout fallback_reason=%s transcript_classification=%s",
                        reason,
                        _current_turn_policy_classification,
                    )
                    fallback_text = "One second — I’m catching up."
                context_aware_text = _context_aware_fallback_text(reason, _current_context_dependency)
                fallback_context_aware = context_aware_text is not None
                if context_aware_text is not None:
                    fallback_text = context_aware_text
                    requires_repeat = False
                    logger.info(
                        "fallback_context_aware=true fallback_reason=%s context_dependency=%s turn_id=%s",
                        reason,
                        _current_context_dependency,
                        llm_turn_id,
                    )
                fallback_yielded = True
                _last_llm_fallback_response_used = True
                _last_generic_llm_fallback_used = True
                _pending_llm_fallback_text = None
                _interaction_state.on_fallback_decision(allowed=True, reason=reason, requires_repeat=bool(requires_repeat))
                logger.warning(
                    "LLM fallback yielded to TTS: fallback_decision_turn_id=%s fallback_suppressed=false fallback_suppressed_reason=none fallback_reason=%s fallback_requires_user_repeat=%s transcript_classification=%s fallback_text_length=%s generic_llm_fallback_used=%s search_in_progress=%s search_tool_called=%s search_turn_id=%s search_turn_matches_current=%s",
                    llm_turn_id,
                    reason,
                    requires_repeat,
                    _current_turn_policy_classification,
                    len(fallback_text),
                    True,
                    _search_in_progress,
                    _search_tool_called,
                    _search_turn_id,
                    _search_turn_matches_current() and _search_turn_id == llm_turn_id,
                )
                yield fallback_text

            pending_next_chunk_task: asyncio.Task[Any] | None = None
            search_fallback_suppressed_logged = False

            def _suppress_generic_fallback_for_search(reason: str) -> bool:
                nonlocal search_fallback_suppressed_logged
                search_turn_matches_current = _search_turn_matches_current() and _search_turn_id == llm_turn_id
                generic_suppression_reason = _generic_fallback_suppression_reason(llm_turn_id, start)
                if generic_suppression_reason == "user_speaking_or_newer_turn_pending":
                    if not search_fallback_suppressed_logged:
                        logger.warning(
                            "fallback_decision_turn_id=%s fallback_suppressed=true fallback_suppressed_reason=user_speaking_or_newer_turn_pending reason=%s latest_user_state=%s current_turn_id=%s user_speaking_after_fallback_turn_started=%s stt_partial_after_llm_start=%s generic_llm_fallback_used=%s",
                            llm_turn_id,
                            reason,
                            _latest_user_state_for_greeting,
                            _current_turn_id,
                            _latest_user_speaking_at > start,
                            _latest_stt_partial_at > start and _latest_stt_partial_at > _latest_stt_final_at,
                            False,
                        )
                        search_fallback_suppressed_logged = True
                    return True
                if _search_in_progress and search_turn_matches_current:
                    if not search_fallback_suppressed_logged:
                        logger.warning(
                            "fallback_decision_turn_id=%s fallback_suppressed=true fallback_suppressed_reason=search_in_progress reason=%s search_in_progress=%s search_tool_called=%s search_started_at=%s search_turn_id=%s search_turn_matches_current=%s search_wait_elapsed_seconds=%.3f generic_llm_fallback_used=%s",
                            llm_turn_id,
                            reason,
                            _search_in_progress,
                            _search_tool_called,
                            _search_started_at,
                            _search_turn_id,
                            search_turn_matches_current,
                            _search_wait_elapsed_seconds(),
                            False,
                        )
                        search_fallback_suppressed_logged = True
                    return True
                if _search_specific_response_produced and search_turn_matches_current:
                    logger.warning(
                        "fallback_decision_turn_id=%s fallback_suppressed=true fallback_suppressed_reason=search_specific_response_available reason=%s search_in_progress=%s search_tool_called=%s search_failed=%s search_specific_failure_response_used=%s search_turn_id=%s search_turn_matches_current=%s generic_llm_fallback_used=%s",
                        llm_turn_id,
                        reason,
                        _search_in_progress,
                        _search_tool_called,
                        _search_failed,
                        _search_failed,
                        _search_turn_id,
                        search_turn_matches_current,
                        False,
                    )
                    return True
                return False

            async def _cancel_pending_next_chunk(reason: str) -> None:
                nonlocal pending_next_chunk_task
                if pending_next_chunk_task is None or pending_next_chunk_task.done():
                    return
                pending_next_chunk_task.cancel()
                try:
                    await pending_next_chunk_task
                except BaseException:
                    pass
                finally:
                    pending_next_chunk_task = None
                    logger.info("LLM pending chunk task cancelled: reason=%s", reason)

            try:
                while True:
                    now = time.monotonic()
                    timeout_stage = "first_token" if _last_llm_first_token_at is None else "completion"
                    stage_deadline = first_token_deadline if _last_llm_first_token_at is None else total_deadline
                    timeout_seconds = min(stage_deadline, total_deadline) - now
                    if timeout_seconds <= 0:
                        if _search_in_progress:
                            _suppress_generic_fallback_for_search(f"{timeout_stage}_timeout")
                            timeout_seconds = min(max(search_timeout_seconds(), 0.25), 1.0)
                        else:
                            _last_llm_complete_at = time.monotonic()
                            if _last_llm_first_token_at is None:
                                if (
                                    not first_token_timeout_extended_once
                                    and LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS > 0
                                    and _current_turn_id == llm_turn_id
                                ):
                                    first_token_timeout_extended_once = True
                                    first_token_deadline = time.monotonic() + LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS
                                    logger.warning(
                                        "LLM first-token timeout extended silently: turn_id=%s llm_timeout_extended_once=%s extension_seconds=%s fallback_reason=%s",
                                        llm_turn_id,
                                        True,
                                        LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS,
                                        "first_token_timeout",
                                    )
                                    continue
                                _last_llm_stream_status = "first_token_timeout"
                                _last_llm_timeout_stage = "first_token"
                                # Optional one-shot re-request before speaking a fallback, so a
                                # slow first token does not immediately surface as a stall. Reuses
                                # the transport-retry recreation; only when nothing was emitted yet.
                                if (
                                    LLM_RETRY_ON_FIRST_TOKEN_TIMEOUT
                                    and not openrouter_retry_attempted
                                    and chunk_count == 0
                                    and _current_turn_id == llm_turn_id
                                ):
                                    openrouter_retry_attempted = True
                                    logger.warning(
                                        "fallback_retry_attempted=true fallback_reason=first_token_timeout fallback_model_used=%s turn_id=%s",
                                        LLM_FALLBACK_MODEL or "none",
                                        llm_turn_id,
                                    )
                                    await _cancel_pending_next_chunk("first_token_timeout_retry")
                                    await _close_llm_stream("first_token_timeout_retry")
                                    stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
                                    it = stream.__aiter__()
                                    pending_next_chunk_task = None
                                    first_token_deadline = time.monotonic() + LLM_FIRST_TOKEN_TIMEOUT_SECONDS
                                    total_deadline = time.monotonic() + LLM_TOTAL_TIMEOUT_SECONDS
                                    _last_llm_stream_status = "started"
                                    _last_llm_complete_at = 0.0
                                    continue
                                logger.error(
                                    "LLM first-token timeout: elapsed_seconds=%s openrouter_model=%s chunk_count=%s text_length=%s search_in_progress=%s search_tool_called=%s",
                                    _fmt_seconds(_last_llm_complete_at - start),
                                    model_name,
                                    chunk_count,
                                    len("".join(assistant_fragments)),
                                    _search_in_progress,
                                    _search_tool_called,
                                )
                                await _cancel_pending_next_chunk("first_token_timeout")
                                await _close_llm_stream("first_token_timeout")
                                async for fallback in _yield_fallback("first_token_timeout"):
                                    logger.warning("LLM first-token timeout: fallback yielded to TTS")
                                    yield fallback
                            else:
                                _last_llm_stream_status = "total_timeout"
                                _last_llm_timeout_stage = "completion"
                                logger.error(
                                    "LLM total timeout: elapsed_seconds=%s openrouter_model=%s chunk_count=%s text_length=%s fallback_yielded=%s search_in_progress=%s search_tool_called=%s",
                                    _fmt_seconds(_last_llm_complete_at - start),
                                    model_name,
                                    chunk_count,
                                    len("".join(assistant_fragments)),
                                    fallback_yielded,
                                    _search_in_progress,
                                    _search_tool_called,
                                )
                                await _cancel_pending_next_chunk("total_timeout")
                                await _close_llm_stream("total_timeout")
                                if not "".join(assistant_fragments).strip():
                                    async for fallback in _yield_fallback("total_timeout_no_text"):
                                        logger.warning("LLM total timeout before usable text: fallback yielded to TTS")
                                        yield fallback
                            break

                    if pending_next_chunk_task is None:
                        pending_next_chunk_task = asyncio.create_task(it.__anext__())

                    done, _ = await asyncio.wait({pending_next_chunk_task}, timeout=max(timeout_seconds, 0.001))
                    if not done:
                        if _search_in_progress:
                            _suppress_generic_fallback_for_search(f"{timeout_stage}_timeout")
                            continue

                        _last_llm_complete_at = time.monotonic()
                        if _last_llm_first_token_at is None:
                            if (
                                not first_token_timeout_extended_once
                                and LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS > 0
                                and _current_turn_id == llm_turn_id
                            ):
                                first_token_timeout_extended_once = True
                                first_token_deadline = time.monotonic() + LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS
                                logger.warning(
                                    "LLM first-token timeout extended silently: turn_id=%s llm_timeout_extended_once=%s extension_seconds=%s fallback_reason=%s",
                                    llm_turn_id,
                                    True,
                                    LLM_FIRST_TOKEN_TIMEOUT_EXTENSION_SECONDS,
                                    "first_token_timeout",
                                )
                                continue
                            _last_llm_stream_status = "first_token_timeout"
                            _last_llm_timeout_stage = "first_token"
                            logger.error(
                                "LLM first-token timeout: elapsed_seconds=%s openrouter_model=%s chunk_count=%s text_length=%s search_in_progress=%s search_tool_called=%s",
                                _fmt_seconds(_last_llm_complete_at - start),
                                model_name,
                                chunk_count,
                                len("".join(assistant_fragments)),
                                _search_in_progress,
                                _search_tool_called,
                            )
                            await _cancel_pending_next_chunk("first_token_timeout")
                            await _close_llm_stream("first_token_timeout")
                            async for fallback in _yield_fallback("first_token_timeout"):
                                logger.warning("LLM first-token timeout: fallback yielded to TTS")
                                yield fallback
                        else:
                            _last_llm_stream_status = "total_timeout"
                            _last_llm_timeout_stage = "completion"
                            logger.error(
                                "LLM total timeout: elapsed_seconds=%s openrouter_model=%s chunk_count=%s text_length=%s search_in_progress=%s search_tool_called=%s",
                                _fmt_seconds(_last_llm_complete_at - start),
                                model_name,
                                chunk_count,
                                len("".join(assistant_fragments)),
                                _search_in_progress,
                                _search_tool_called,
                            )
                            await _cancel_pending_next_chunk("total_timeout")
                            await _close_llm_stream("total_timeout")
                            if not "".join(assistant_fragments).strip():
                                async for fallback in _yield_fallback("total_timeout_no_text"):
                                    logger.warning("LLM total timeout before usable text: fallback yielded to TTS")
                                    yield fallback
                        break

                    task = pending_next_chunk_task
                    pending_next_chunk_task = None
                    try:
                        chunk = task.result()
                    except StopAsyncIteration:
                        _last_llm_complete_at = time.monotonic()
                        completed_text = "".join(assistant_fragments)
                        _last_llm_completed_at = _last_llm_complete_at
                        _last_llm_completed_text = completed_text
                        _last_llm_completed_text_hash = _text_hash(completed_text)
                        if completed_text.strip():
                            _last_llm_stream_status = "ok"
                            logger.info(
                                "LLM stream completed: openrouter_model=%s chunk_count=%s text_length=%s stream_duration_seconds=%s first_token_latency_seconds=%s",
                                model_name,
                                chunk_count,
                                len(completed_text),
                                _fmt_seconds(_last_llm_complete_at - start),
                                _fmt_seconds((_last_llm_first_token_at - start) if _last_llm_first_token_at is not None else None),
                            )
                        else:
                            _last_llm_stream_status = "empty_stream"
                            _last_llm_timeout_stage = "first_token"
                            logger.warning(
                                "LLM stream ended empty: chunk_count=%s stream_duration_seconds=%s search_in_progress=%s search_tool_called=%s",
                                chunk_count,
                                _fmt_seconds(_last_llm_complete_at - start),
                                _search_in_progress,
                                _search_tool_called,
                            )
                            async for fallback in _yield_fallback("empty_stream"):
                                logger.warning("LLM stream ended empty: fallback yielded to TTS")
                                yield fallback
                        break
                    except asyncio.CancelledError:
                        _last_llm_stream_status = "cancelled"
                        _last_llm_timeout_stage = "none"
                        _last_llm_complete_at = time.monotonic()
                        logger.warning(
                            "LLM stream cancelled/interrupted: openrouter_model=%s chunk_count=%s text_length=%s elapsed_seconds=%s",
                            model_name,
                            chunk_count,
                            len("".join(assistant_fragments)),
                            _fmt_seconds(_last_llm_complete_at - start),
                        )
                        raise
                    except Exception as e:
                        text_length = len("".join(assistant_fragments))
                        retry_allowed, retry_skip_reason = _should_retry_openrouter_connection_error(
                            e,
                            first_token_seen=_last_llm_first_token_at is not None,
                            chunk_count=chunk_count,
                            text_length=text_length,
                            llm_turn_id=llm_turn_id,
                            tts_started_for_turn=_last_tts_request_start_at >= start,
                        )
                        if retry_allowed and not openrouter_retry_attempted:
                            openrouter_retry_attempted = True
                            logger.warning(
                                "OpenRouter transport error retrying once: turn_id=%s openrouter_retry_attempted=%s openrouter_retry_skipped_reason=%s error_type=%s",
                                llm_turn_id,
                                True,
                                "none",
                                type(e).__name__,
                            )
                            await _cancel_pending_next_chunk("openrouter_api_connection_retry")
                            await _close_llm_stream("openrouter_api_connection_retry")
                            stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
                            it = stream.__aiter__()
                            pending_next_chunk_task = None
                            first_token_deadline = time.monotonic() + LLM_FIRST_TOKEN_TIMEOUT_SECONDS
                            total_deadline = time.monotonic() + LLM_TOTAL_TIMEOUT_SECONDS
                            continue
                        logger.warning(
                            "OpenRouter transport retry skipped: turn_id=%s openrouter_retry_attempted=%s openrouter_retry_skipped_reason=%s error_type=%s",
                            llm_turn_id,
                            openrouter_retry_attempted,
                            "already_attempted" if openrouter_retry_attempted and retry_allowed else retry_skip_reason,
                            type(e).__name__,
                        )
                        _last_llm_complete_at = time.monotonic()
                        _last_llm_stream_status = "provider_error"
                        _last_llm_timeout_stage = "none"
                        logger.error(
                            "LLM provider error: error_type=%s error=%s error_details=%s openrouter_model=%s chunk_count=%s text_length=%s first_token_seen=%s stream_duration_seconds=%s search_in_progress=%s search_tool_called=%s",
                            type(e).__name__,
                            _redact_sensitive_text(e),
                            _safe_llm_error_details(e),
                            model_name,
                            chunk_count,
                            text_length,
                            _last_llm_first_token_at is not None,
                            _fmt_seconds(_last_llm_complete_at - start),
                            _search_in_progress,
                            _search_tool_called,
                        )
                        await _close_llm_stream("provider_error")
                        if not "".join(assistant_fragments).strip():
                            async for fallback in _yield_fallback("provider_error"):
                                logger.warning("LLM provider error before usable text: fallback yielded to TTS")
                                yield fallback
                        break

                    if _is_stale_llm_turn(llm_turn_id):
                        _last_llm_stream_status = "stale_turn"
                        _last_llm_complete_at = time.monotonic()
                        logger.warning(
                            "stale_llm_output_ignored=true original_turn_id=%s current_turn_id=%s llm_turn_id=%s chunk_count=%s",
                            llm_turn_id,
                            _current_turn_id,
                            llm_turn_id,
                            chunk_count,
                        )
                        await _cancel_pending_next_chunk("stale_turn")
                        await _close_llm_stream("stale_turn")
                        break

                    chunk_count += 1
                    text_delta = _extract_text_delta(chunk)
                    if text_delta:
                        assistant_fragments.append(text_delta)
                        if text_delta.strip() and _last_llm_first_token_at is None:
                            _last_llm_first_token_at = time.monotonic()
                            logger.info(
                                "LLM first token received: openrouter_model=%s first_token_latency_seconds=%s chunk_count=%s",
                                model_name,
                                _fmt_seconds(_last_llm_first_token_at - start),
                                chunk_count,
                            )
                    yield chunk
            finally:
                if _last_llm_complete_at <= 0.0:
                    _last_llm_complete_at = time.monotonic()
                completed_text = "".join(assistant_fragments)
                _last_llm_completed_text = completed_text
                _last_llm_completed_text_hash = _text_hash(completed_text)
                if _last_llm_completed_at <= 0.0 and _last_llm_stream_status in {"ok", "empty_stream", "provider_error", "first_token_timeout", "total_timeout"}:
                    _last_llm_completed_at = _last_llm_complete_at
                logger.info(
                    "LLM stream ended: turn_id=%s llm_turn_id=%s search_turn_id=%s status=%s first_token_seen=%s chunk_count=%s text_length=%s fallback_used=%s timeout_stage=%s search_in_progress=%s search_tool_called=%s search_failed=%s generic_llm_fallback_used=%s",
                    llm_turn_id,
                    llm_turn_id,
                    _search_turn_id,
                    _last_llm_stream_status,
                    _last_llm_first_token_at is not None,
                    chunk_count,
                    len(completed_text),
                    _last_llm_fallback_response_used,
                    _last_llm_timeout_stage,
                    _search_in_progress,
                    _search_tool_called,
                    _search_failed,
                    _last_generic_llm_fallback_used,
                )

        return _llm_stream()

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        global _last_turn_committed_at, _last_llm_start_at, _last_llm_first_token_at, _last_llm_complete_at, _last_llm_stream_status, _last_llm_timeout_stage, _last_llm_fallback_response_used, _last_llm_completed_text, _last_llm_completed_text_hash, _last_llm_completed_at, _last_tts_received_text_hash, _last_user_message_text, _current_turn_id, _current_turn_transcript_intent, _current_turn_search_allowed, _current_turn_search_allowed_reason, _current_turn_policy_classification, _current_turn_policy_decision, _current_turn_audio_unclear, _held_turn_fragment_text, _held_turn_fragment_created_at, _held_turn_fragment_classification, _held_turn_fragment_incomplete
        global _barge_in_during_thinking_turn_id, _barge_in_started_at, _barge_in_confirmed_real
        global _prior_context_decision, _current_context_dependency, _current_candidate_id, _current_turn_owner_valid
        _current_turn_id = _next_turn_id()
        # New turn committed: clear any barge-in latch from the prior turn. A stale
        # prior reply is now covered by the existing _is_stale_llm_turn path.
        _barge_in_during_thinking_turn_id = 0
        _barge_in_started_at = 0.0
        _barge_in_confirmed_real = False
        # Attribute this commit against real user-speech signals (not only FSM
        # events, which can be missed): a recent user-speaking edge or a recent
        # STT final means the user genuinely spoke for this turn even if the FSM
        # state already advanced. Only a commit with neither is a true anomaly.
        _commit_now = time.monotonic()
        _user_speech_observed_for_commit = (
            (_latest_user_speaking_at > 0.0 and (_commit_now - _latest_user_speaking_at) <= INTERACTION_USER_SPEECH_OBSERVATION_WINDOW_SECONDS)
            or (_latest_stt_final_at > 0.0 and (_commit_now - _latest_stt_final_at) <= INTERACTION_USER_SPEECH_OBSERVATION_WINDOW_SECONDS)
        )
        # Owner validity: a turn has a valid owner when real user speech was
        # observed or the FSM was already in a user-speech state. An owner-less
        # commit is gated so it cannot produce canonical context (enforce).
        _pre_begin_state = _interaction_state.state
        _current_turn_owner_valid = bool(_user_speech_observed_for_commit) or _pre_begin_state in (
            USER_TURN_CANDIDATE,
            USER_SPEAKING,
            USER_INTERRUPTING,
        )
        _interaction_state.begin_turn(_current_turn_id, user_speech_observed=_user_speech_observed_for_commit)
        _interaction_state.runtime_gate(
            "turn_commit_owner",
            _current_turn_owner_valid,
            reason="valid_owner" if _current_turn_owner_valid else "turn_commit_no_owner",
        )
        _reset_search_state_for_turn(_current_turn_id)
        _last_turn_committed_at = time.monotonic()
        _last_llm_start_at = 0.0
        _last_llm_first_token_at = None
        _last_llm_complete_at = 0.0
        _last_llm_stream_status = "n/a"
        _last_llm_timeout_stage = "n/a"
        _last_llm_fallback_response_used = False
        _last_llm_completed_text = ""
        _last_llm_completed_text_hash = "empty"
        _last_llm_completed_at = 0.0
        _last_user_message_text = _extract_text_for_debug(new_message).strip()
        _complete_pending_calibration_moment(_last_user_message_text)
        _last_tts_received_text_hash = "empty"
        _current_candidate_id, _candidate_drift_suspected, _candidate_latest_final_hash = _bind_candidate_for_commit(_last_user_message_text)
        logger.info(
            "User turn committed: turn_id=%s candidate_id=%s search_state_reset=true search_turn_id=%s search_in_progress=%s search_tool_called=%s",
            _current_turn_id,
            _current_candidate_id,
            _search_turn_id,
            _search_in_progress,
            _search_tool_called,
        )
        if _candidate_drift_suspected:
            _drift_category = _categorize_transcript_drift(_last_user_message_text)
            logger.warning(
                "transcript_drift_suspected=true transcript_drift_category=%s turn_id=%s candidate_id=%s committed_text_hash=%s latest_stt_final_text_hash=%s note=committed_turn_text_differs_from_newest_stt_final",
                _drift_category,
                _current_turn_id,
                _current_candidate_id,
                _text_hash(_last_user_message_text),
                _candidate_latest_final_hash,
            )
        if transcript_context_layer_enabled():
            transcript_context_llm_started = transcript_context_llm_enabled()
            context_error_type = "none"
            try:
                transcript_context = await interpret_transcript_context(
                    _last_user_message_text,
                    recent_turns=_recent_turn_previews_from_chat_ctx(turn_ctx),
                    runtime_context=self.runtime_context.system_message if self.runtime_context is not None else None,
                )
            except Exception as exc:
                context_error_type = type(exc).__name__
                logger.warning(
                    "Transcript context layer failed; using deterministic fallback: error_type=%s error=%s",
                    context_error_type,
                    _redact_sensitive_text(exc),
                )
                transcript_context = detect_transcript_context(_last_user_message_text)
            _current_turn_transcript_intent = transcript_context.detected_intent or "unknown"
            _current_turn_search_allowed, _current_turn_search_allowed_reason = _search_policy_for_intent(
                _current_turn_transcript_intent,
                transcript_context.clarification_suggested,
            )
            logger.info(
                "search_allowed_for_turn=%s search_allowed_reason=%s turn_id=%s detected_intent=%s clarification_suggested=%s",
                _current_turn_search_allowed,
                _current_turn_search_allowed_reason,
                _current_turn_id,
                _current_turn_transcript_intent,
                transcript_context.clarification_suggested,
            )
            _apply_transcript_context_to_message(new_message, transcript_context)
            if transcript_context.should_replace_user_text and transcript_context.cleaned_text:
                _last_user_message_text = transcript_context.cleaned_text
            _inject_transcript_context_note(turn_ctx, transcript_context)
            if transcript_context.source == "deterministic_llm_error_fallback" and context_error_type == "none":
                context_error_type = "llm_error_or_invalid_json"
            _log_transcript_context_result(transcript_context, llm_started=transcript_context_llm_started, llm_error_type=context_error_type, turn_id=_current_turn_id)
        else:
            _current_turn_transcript_intent = "disabled"
            _current_turn_search_allowed = True
            _current_turn_search_allowed_reason = "llm_tool_call"
            logger.info(
                "Transcript context result: turn_id=%s transcript_context_layer_enabled=%s transcript_context_llm_enabled=%s transcript_context_llm_model=%s transcript_context_llm_timeout_ms=%s transcript_context_source=%s original_length=%s cleaned_length=%s context_note_present=%s capability_contract_note_present=%s",
                _current_turn_id,
                False,
                transcript_context_llm_enabled(),
                transcript_context_llm_model(),
                transcript_context_llm_timeout_ms(),
                "disabled",
                len(_last_user_message_text),
                len(_last_user_message_text),
                False,
                False,
            )
        inworld_profile = None
        if _inworld_voice_profile_shadow is not None:
            try:
                inworld_profile, skip_reason, added_latency_seconds = _inworld_voice_profile_shadow.context_for_turn(_last_turn_committed_at)
                _inject_inworld_voice_context_note(
                    turn_ctx,
                    inworld_profile,
                    added_latency_seconds=added_latency_seconds,
                    skip_reason=skip_reason,
                )
            except Exception as exc:
                logger.warning(
                    "inworld_voice_profile_context_failed=true fallback_skip_reason=%s error=%s",
                    type(exc).__name__,
                    _redact_sensitive_text(exc),
                )
        else:
            logger.info("inworld_voice_profile_context_passed_to_llm=false fallback_skip_reason=disabled")
        _inject_emotional_calibration_planner_note(turn_ctx, _last_user_message_text, inworld_profile)
        policy_context = transcript_context if transcript_context_layer_enabled() else detect_transcript_context(_last_user_message_text)
        turn_policy = _make_turn_policy_decision(
            _last_user_message_text,
            policy_context,
            held_text=_held_turn_fragment_text,
            held_created_at=_held_turn_fragment_created_at,
        )
        _current_turn_policy_classification = turn_policy.classification
        _current_turn_policy_decision = turn_policy.decision
        _current_turn_audio_unclear = turn_policy.classification == "UNCLEAR_AUDIO"
        context_fit, coherence_reason, coherence_confidence = _assess_context_coherence(
            _latest_stt_language, _latest_stt_language_confidence, SESSION_LANGUAGE
        )
        response_posture = "acknowledge_and_attempt" if context_fit == "low_coherence" else "answer"
        logger.info(
            "turn_policy_decision=%s candidate_id=%s transcript_classification=%s classification_confidence=%.2f classification_reason=%s should_start_generation=%s should_merge_held_fragment=%s should_clear_held_fragment=%s commit_allowed=%s commit_block_reason=%s active_language=%s stt_language_candidate=%s stt_language_confidence=%s normalized_user_text=%s context_fit=%s coherence_reason=%s coherence_confidence=%.2f response_posture=%s",
            turn_policy.decision,
            _current_candidate_id,
            turn_policy.classification,
            turn_policy.confidence,
            turn_policy.reason,
            turn_policy.should_start_generation,
            turn_policy.should_merge_held_fragment,
            turn_policy.should_clear_held_fragment,
            turn_policy.should_start_generation,
            "none" if turn_policy.should_start_generation else f"{turn_policy.decision}:{turn_policy.reason}",
            SESSION_LANGUAGE,
            _latest_stt_language,
            _latest_stt_language_confidence,
            bool(getattr(policy_context, "should_replace_user_text", False)),
            context_fit,
            coherence_reason,
            coherence_confidence,
            response_posture,
        )
        context_decision: ContextDecision | None = None
        if CONTEXT_POLICY_ENABLED:
            context_decision = build_context_decision(
                text=_last_user_message_text,
                base_intent=_current_turn_transcript_intent,
                classification=turn_policy.classification,
                ambiguity_detected=bool(getattr(policy_context, "ambiguity_detected", False)),
                clarification_suggested=bool(getattr(policy_context, "clarification_suggested", False)),
                confidence=float(getattr(policy_context, "confidence", 0.0) or 0.0),
                prior_decision=_prior_context_decision if CONTEXT_REFERENCE_CARRY_FORWARD_ENABLED else None,
            )
            _current_context_dependency = context_decision.context_dependency
            logger.info(
                "context_decision_final_intent=%s context_dependency=%s response_posture=%s force_context_injection=%s context_decision_source=%s confidence=%.2f ambiguity_detected=%s clarification_suggested=%s reference_carry_forward_applied=%s turn_id=%s",
                context_decision.final_intent,
                context_decision.context_dependency,
                context_decision.response_posture,
                context_decision.force_context_injection,
                context_decision.decision_source,
                context_decision.confidence,
                context_decision.ambiguity_detected,
                context_decision.clarification_suggested,
                context_decision.decision_source == "carry_forward",
                _current_turn_id,
            )
            # Carry the contextual intent forward 1-2 turns; a clearly standalone
            # turn (no dependency, not a carry-forward trigger) clears it.
            if context_decision.context_dependency == "high":
                _prior_context_decision = context_decision
            else:
                _prior_context_decision = None
        else:
            _current_context_dependency = "none"
        _interaction_state.set_turn_kind(
            classify_turn_kind(_current_turn_transcript_intent, turn_policy.classification, turn_policy.decision),
            detected_intent=_current_turn_transcript_intent,
        )
        _interaction_state.on_turn_policy(turn_policy.decision, turn_policy.classification, turn_policy.reason)
        # If a tool/search result's authority to speak was paused by user speech
        # mid tool call, run the high-risk composer: classify how this newer
        # utterance relates to the pending query and decide compose / rerun /
        # withhold. A timeout protects latency but never grants authority.
        if _interaction_state.tool_result_pending_revalidation:
            await _revalidate_pending_tool_result(
                turn_ctx,
                newer_utterance=_last_user_message_text,
                classification=turn_policy.classification,
                recent_turns=_recent_turn_previews_from_chat_ctx(turn_ctx),
            )
        if _audiointeraction_shadow is not None:
            try:
                _audiointeraction_shadow.compare_at_turn_commit(
                    _current_turn_id,
                    turn_policy.decision,
                    turn_policy.should_start_generation,
                )
            except Exception as exc:
                logger.warning(
                    "audiointeraction_shadow_comparison_failed=true error_type=%s error=%s",
                    type(exc).__name__,
                    _redact_sensitive_text(exc),
                )
        if turn_policy.decision == "IGNORE_LOW_INFORMATION_FILLER" and not turn_policy.should_start_generation:
            logger.info(
                "turn_generation_skipped=true skip_reason=low_information_filler turn_id=%s transcript_classification=%s classification_reason=%s transcript_length=%s",
                _current_turn_id,
                turn_policy.classification,
                turn_policy.reason,
                len(_last_user_message_text),
            )
            raise StopResponse()
        _inject_response_mode_note(turn_ctx, _interaction_state.turn_kind, _current_turn_transcript_intent)
        if context_fit == "low_coherence":
            _inject_coherence_note(turn_ctx, coherence_reason)
        if _is_audio_status_check(_last_user_message_text):
            _audio_now = time.monotonic()
            _starts, _candidates = _audio_env_recent_counts(_audio_now)
            audio_env = build_audio_environment_decision(
                false_speech_start_count_recent=_starts,
                candidate_turn_count_recent=_candidates,
                short_noisy_fragment_detected=len(_last_user_message_text.split()) <= 3 and _current_turn_audio_unclear,
                is_audio_status_check=True,
            )
            logger.info(
                "audio_environment_decision noise_state=%s noise_confidence=%.2f speech_stability=%s transcript_stability=%s false_speech_start_count_recent=%s candidate_turn_count_recent=%s short_noisy_fragment_detected=%s action_hint=%s reason=%s turn_id=%s",
                audio_env.noise_state,
                audio_env.noise_confidence,
                audio_env.speech_stability,
                audio_env.transcript_stability,
                audio_env.false_speech_start_count_recent,
                audio_env.candidate_turn_count_recent,
                audio_env.short_noisy_fragment_detected,
                audio_env.action_hint,
                audio_env.reason,
                _current_turn_id,
            )
            _inject_audio_status_note(turn_ctx, _audio_status_response_text(audio_env))
        if (
            context_decision is not None
            and CONTEXT_FORCE_LOCAL_INJECTION
            and context_decision.force_context_injection
        ):
            recent = _ledger_recent_canonical(CONTEXT_INJECTION_TURNS)
            posture = context_decision.response_posture if CONTEXT_RESPONSE_POSTURE_ENABLED else "answer"
            injected_count = _inject_context_window_note(turn_ctx, posture, recent)
            if LOG_PROMPT_CONTEXT_INJECTION:
                logger.info(
                    "prompt_context_injected=%s prompt_context_turn_count=%s suppressed_messages_excluded_count=%s reference_carry_forward_applied=%s canonical_history_visible_messages_count=%s response_posture=%s turn_id=%s",
                    injected_count > 0,
                    injected_count,
                    _ledger_suppressed_count(),
                    context_decision.decision_source == "carry_forward",
                    _ledger_visible_canonical_count(),
                    posture,
                    _current_turn_id,
                )
        merged_text: str | None = None
        if turn_policy.classification == "META_COMPLAINT":
            has_held_fragment = bool(_held_turn_fragment_text)
            silence_recovery_triggered = False
            if has_held_fragment:
                recovery_text = (
                    "You’re right — I held that too long. "
                    f"You were talking about: {_redact_sensitive_text(_held_turn_fragment_text)[:160]}"
                )
                _set_user_message_text(new_message, recovery_text)
                _last_user_message_text = recovery_text
            elif _should_trigger_silence_recovery(turn_policy.classification, has_held_fragment):
                # No fragment to point back to (e.g. "you keep going silent"):
                # inject a recovery note so Arche acknowledges the silence and
                # re-engages instead of just answering the complaint literally.
                silence_recovery_triggered = _inject_silence_recovery_note(turn_ctx)
            logger.warning(
                "meta_complaint_detected=true held_fragment_present=%s recovery_from_silence_triggered=%s recovery_path=%s",
                has_held_fragment,
                has_held_fragment or silence_recovery_triggered,
                "held_fragment" if has_held_fragment else ("silence_recovery_note" if silence_recovery_triggered else "none"),
            )
        elif turn_policy.should_merge_held_fragment and turn_policy.decision == "MERGE_WITH_HELD_FRAGMENT" and _held_turn_fragment_text:
            merged_text = f"{_held_turn_fragment_text.rstrip()} {_last_user_message_text.lstrip()}".strip()
            _set_user_message_text(new_message, merged_text)
            _last_user_message_text = merged_text
            logger.info("held_fragment_merged=true held_fragment_age_seconds=%.3f", time.monotonic() - _held_turn_fragment_created_at)
        elif _held_turn_fragment_text and not turn_policy.should_merge_held_fragment:
            logger.info(
                "held_fragment_not_merged_reason=%s held_fragment_age_seconds=%s",
                turn_policy.reason,
                _fmt_seconds(time.monotonic() - _held_turn_fragment_created_at if _held_turn_fragment_created_at else None),
            )

        if turn_policy.decision == "HOLD_FOR_CONTINUATION":
            _held_turn_fragment_text = _last_user_message_text
            _held_turn_fragment_created_at = time.monotonic()
            _held_turn_fragment_classification = turn_policy.classification
            _held_turn_fragment_incomplete = True
            logger.info(
                "held_fragment_created=true transcript_classification=%s classification_confidence=%.2f classification_reason=%s reply_deadline_seconds=%s merge_window_seconds=%s",
                turn_policy.classification,
                turn_policy.confidence,
                turn_policy.reason,
                TURN_HOLD_FRAGMENT_REPLY_DEADLINE_SECONDS,
                TURN_HOLD_FRAGMENT_MERGE_WINDOW_SECONDS,
            )
            hold_turn_id = _current_turn_id
            hold_started_at = time.monotonic()
            hold_deadline_seconds = max(0.0, TURN_HOLD_FRAGMENT_REPLY_DEADLINE_SECONDS)
            hold_yield_reason = None
            while time.monotonic() - hold_started_at < hold_deadline_seconds:
                await asyncio.sleep(0.1)
                if _latest_user_state_for_greeting == "speaking":
                    hold_yield_reason = "user_resumed_speaking"
                    break
                if _latest_stt_partial_at > hold_started_at or _latest_stt_final_at > hold_started_at:
                    hold_yield_reason = "new_transcript_activity"
                    break
                if _current_turn_id != hold_turn_id:
                    hold_yield_reason = "newer_turn_started"
                    break
            if hold_yield_reason is not None:
                logger.info(
                    "held_fragment_wait_yielded=true yield_reason=%s turn_id=%s held_fragment_age_seconds=%.3f held_fragment_retained_for_merge=true",
                    hold_yield_reason,
                    hold_turn_id,
                    time.monotonic() - _held_turn_fragment_created_at,
                )
                raise StopResponse()
            # Final conservative guard: the loop above can exit on the deadline
            # in the same tick that newer speech/continuation arrives. Re-check
            # right before committing so the deadline never fires while newer
            # speech, active speech, or fresh continuation candidates exist —
            # the held fragment is retained for merge instead.
            _deadline_now = time.monotonic()
            _active_continuation = (
                _latest_user_state_for_greeting == "speaking"
                or _latest_stt_partial_at > hold_started_at
                or _latest_stt_final_at > hold_started_at
                or _interaction_state.state in (USER_SPEAKING, USER_INTERRUPTING)
                or _current_turn_id != hold_turn_id
            )
            # Enforce: the deadline commit is gated, not merely observed.
            _deadline_commit_allowed = _interaction_state.runtime_gate(
                "held_fragment_deadline_commit",
                not _active_continuation,
                reason="active_continuation" if _active_continuation else "no_active_continuation",
            )
            if not _deadline_commit_allowed:
                logger.info(
                    "held_fragment_deadline_suppressed_active_continuation=true turn_id=%s user_state=%s interaction_state=%s held_fragment_age_seconds=%.3f held_fragment_retained_for_merge=true",
                    hold_turn_id,
                    _latest_user_state_for_greeting,
                    _interaction_state.state,
                    _deadline_now - _held_turn_fragment_created_at,
                )
                raise StopResponse()
            logger.info(
                "held_fragment_committed_due_to_reply_deadline=true held_fragment_age_seconds=%.3f",
                time.monotonic() - _held_turn_fragment_created_at,
            )
            _interaction_state.on_hold_deadline_commit()
            _held_turn_fragment_text = ""
            _held_turn_fragment_created_at = 0.0
            _held_turn_fragment_classification = ""
            _held_turn_fragment_incomplete = False
        if turn_policy.should_clear_held_fragment:
            _held_turn_fragment_text = ""
            _held_turn_fragment_created_at = 0.0
            _held_turn_fragment_classification = ""
            _held_turn_fragment_incomplete = False

        endpointing_decision_context = transcript_context if transcript_context_layer_enabled() else detect_transcript_context(_last_user_message_text)
        endpointing_decision, endpointing_extend_reason, endpointing_wait_extension_ms = _endpointing_decision_for_transcript(
            _last_user_message_text,
            endpointing_decision_context,
        )
        logger.info(
            "endpointing_decision=%s endpointing_extend_reason=%s endpointing_wait_extension_ms=%s turn_id=%s",
            endpointing_decision,
            endpointing_extend_reason,
            endpointing_wait_extension_ms,
            _current_turn_id,
        )
        if endpointing_decision == "extend_wait" and endpointing_wait_extension_ms > 0:
            logger.info(
                "endpointing_extension_skipped=true skip_reason=intent_based_turn_policy turn_policy_decision=%s endpointing_extend_reason=%s endpointing_wait_extension_ms=%s turn_id=%s",
                turn_policy.decision,
                endpointing_extend_reason,
                endpointing_wait_extension_ms,
                _current_turn_id,
            )

        memory_layer = getattr(self, "memory_layer", None)
        if memory_layer is not None:
            memory_intent = getattr(endpointing_decision_context, "detected_intent", None)
            retrieval_allowed, retrieval_gate_reason = _memory_retrieval_policy_for_intent(memory_intent)
            logger.info(
                "memory_retrieval_gate=%s memory_retrieval_gate_reason=%s detected_intent=%s turn_id=%s",
                retrieval_allowed,
                retrieval_gate_reason,
                memory_intent,
                _current_turn_id,
            )
            if retrieval_allowed:
                retrieved_memories = await memory_layer.retrieve(_last_user_message_text)
                if retrieved_memories:
                    _inject_memory_note(turn_ctx, retrieved_memories)
                # A recall ask ("what did I just say / my last question") is about
                # the conversation transcript, not long-term memory. Anchor the
                # model on the user's actual most-recent prior message so it does
                # not grab an earlier turn.
                if (memory_intent or "").strip().lower() == "memory_recall_request":
                    _inject_recall_anchor_note(
                        turn_ctx,
                        _recent_user_messages_from_chat_ctx(
                            turn_ctx, exclude_text=_last_user_message_text, limit=3
                        ),
                    )
            # The write path is never gated: Lucy keeps remembering every turn so
            # there is material to recall on a later "do you remember..." turn.
            memory_layer.schedule_remember(role="user", content=_last_user_message_text, turn_id=_current_turn_id)

        _prune_turn_context_messages(turn_ctx, _current_turn_id)
        if PIPELINE_TEXT_DEBUG:
            msg_str = _last_user_message_text
            turn_ctx_items = _chat_ctx_items(turn_ctx)
            message_count = len(turn_ctx_items) if turn_ctx_items is not None else "n/a"
            held_fragment_count = 1 if _held_turn_fragment_text.strip() else 0
            logger.info(
                "User turn debug: candidate_id=%s new_message_length=%s preview=%s turn_ctx_message_count=%s held_fragment_count=%s held_fragment_merged=%s",
                _current_candidate_id,
                len(msg_str),
                _redact_sensitive_text(msg_str)[:200],
                message_count,
                held_fragment_count,
                merged_text is not None,
            )



def build_vad():
    global _silero_initialized
    if VAD_PROVIDER == "ai_coustics":
        logger.info("Using ai-coustics VAD provider")
        return ai_coustics.VAD()

    if VAD_PROVIDER == "silero":
        logger.info("Using Silero VAD provider")
        _silero_initialized = True
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



async def _run_hume_evi_voice_engine(ctx: JobContext) -> None:
    """Connect the LiveKit room, then hand room audio to the Hume EVI bridge.

    The bridge publishes an output track and subscribes to remote audio, both of
    which require a connected room/local participant. The cascaded pipeline gets
    this connect for free via ``AgentSession.start()``; the EVI path has no
    session, so it must connect explicitly before bridging. Bootstrap failures
    are logged with a clear, parseable line (no secrets) before re-raising so the
    job fails visibly rather than via an opaque traceback.
    """
    logger.info("voice_engine_selected=hume_evi current_pipeline_disabled=true livekit_room_layer=true")
    try:
        await ctx.connect()
        logger.info("voice_engine_room_connected=true engine=hume_evi")
        await run_hume_evi_bridge(ctx.room)
    except Exception as exc:
        logger.error(
            "voice_engine_bootstrap_failed=true engine=hume_evi error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        raise


async def _prewarm_models(llm_obj: Any) -> None:
    """Fire tiny throwaway requests so the first real turn isn't paying
    connection/provider cold-start cost.

    Best-effort: every failure is swallowed and logged. This never blocks the
    live path — it is launched as a background task and bounded by short
    timeouts. Warms two distinct paths: the main generation LLM (OpenRouter via
    the LiveKit openai client, which is where the turn-1 first-token cold start
    was observed) and, when enabled, the transcript-context classifier model
    (raw OpenRouter POST) with a generous timeout so the warmup itself can
    actually complete instead of timing out at the normal per-turn budget.
    """
    from livekit.agents import APIConnectOptions
    from livekit.agents import llm as _lk_llm

    gen_started_at = time.monotonic()
    try:
        warm_ctx = _lk_llm.ChatContext.empty()
        warm_ctx.add_message(role="user", content="ping")
        stream = llm_obj.chat(
            chat_ctx=warm_ctx,
            conn_options=APIConnectOptions(max_retry=0, timeout=10.0),
        )
        try:
            async for _chunk in stream:
                break
        finally:
            await stream.aclose()
        logger.info(
            "LLM warmup completed: target=generation elapsed_seconds=%s",
            _fmt_seconds(time.monotonic() - gen_started_at),
        )
    except Exception as exc:  # noqa: BLE001 - warmup must never break startup
        logger.info(
            "LLM warmup skipped/failed: target=generation error_type=%s elapsed_seconds=%s",
            type(exc).__name__,
            _fmt_seconds(time.monotonic() - gen_started_at),
        )

    if transcript_context_llm_enabled():
        tc_started_at = time.monotonic()
        try:
            await call_transcript_context_llm(
                detect_transcript_context("ok"),
                timeout_ms=10000,
            )
            logger.info(
                "LLM warmup completed: target=transcript_context elapsed_seconds=%s",
                _fmt_seconds(time.monotonic() - tc_started_at),
            )
        except Exception as exc:  # noqa: BLE001 - warmup must never break startup
            logger.info(
                "LLM warmup skipped/failed: target=transcript_context error_type=%s elapsed_seconds=%s",
                type(exc).__name__,
                _fmt_seconds(time.monotonic() - tc_started_at),
            )


def _memory_retrieval_policy_for_intent(intent: str | None) -> tuple[bool, str]:
    """Gate semantic memory retrieval to explicit recall asks.

    Mirrors the search-intent gate: the ~300ms SimpleMem lookup is only paid when
    the user is actually asking Lucy to remember something ("do you remember...",
    "what did I..."). Every other turn skips retrieval entirely and pays no
    latency. The memory *write* path is never gated — Lucy keeps remembering
    everything so there is material to recall later.
    """
    normalized = (intent or "unknown").strip().lower()
    if normalized == "memory_recall_request":
        return True, "memory_recall_intent"
    return False, "no_recall_intent"


async def _terminate_room(ctx: JobContext) -> str:
    """End the call for everyone so the client lands on the end screen.

    Prefer deleting the room (disconnects ALL participants, including the user,
    which fires RoomEvent.Disconnected on the client). Fall back to disconnecting
    the agent's own participant if the server API isn't reachable.
    """
    room = getattr(ctx, "room", None)
    room_name = getattr(room, "name", None)

    # Newer livekit-agents expose a JobContext.delete_room() convenience.
    ctx_delete_room = getattr(ctx, "delete_room", None)
    if callable(ctx_delete_room):
        try:
            await ctx_delete_room()
            logger.info("session_room_terminated=true strategy=ctx_delete_room room=%s", room_name)
            return "ctx_delete_room"
        except Exception as exc:
            logger.warning(
                "session_room_terminate_failed strategy=ctx_delete_room error=%s",
                _redact_sensitive_text(exc),
            )

    # JobContext.api when present.
    ctx_api = getattr(ctx, "api", None)
    if ctx_api is not None and room_name:
        try:
            await ctx_api.room.delete_room(api.DeleteRoomRequest(room=room_name))
            logger.info("session_room_terminated=true strategy=ctx_api_delete_room room=%s", room_name)
            return "ctx_api_delete_room"
        except Exception as exc:
            logger.warning(
                "session_room_terminate_failed strategy=ctx_api_delete_room error=%s",
                _redact_sensitive_text(exc),
            )

    # Reliable path: build a server API client from the worker's LiveKit creds
    # (the same creds server.py uses to create rooms) and delete the room. This
    # forcibly disconnects the USER, which is what actually ends the call.
    lk_url = os.getenv("LIVEKIT_URL")
    lk_key = os.getenv("LIVEKIT_API_KEY")
    lk_secret = os.getenv("LIVEKIT_API_SECRET")
    if room_name and lk_url and lk_key and lk_secret:
        lkapi = api.LiveKitAPI(url=lk_url, api_key=lk_key, api_secret=lk_secret)
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
            logger.info("session_room_terminated=true strategy=env_api_delete_room room=%s", room_name)
            return "env_api_delete_room"
        except Exception as exc:
            logger.warning(
                "session_room_terminate_failed strategy=env_api_delete_room error=%s",
                _redact_sensitive_text(exc),
            )
        finally:
            try:
                await lkapi.aclose()
            except Exception:
                pass
    else:
        logger.warning(
            "session_room_terminate_env_api_unavailable room_name_present=%s livekit_url_present=%s "
            "livekit_key_present=%s livekit_secret_present=%s",
            bool(room_name),
            bool(lk_url),
            bool(lk_key),
            bool(lk_secret),
        )

    # Last resort: disconnecting our own participant removes the AGENT but leaves
    # the user in the room, so the call may not actually end. Logged as degraded.
    try:
        if room is not None:
            await room.disconnect()
            logger.warning(
                "session_room_terminated=degraded strategy=room_disconnect_agent_only room=%s "
                "(user may remain connected)",
                room_name,
            )
            return "room_disconnect"
    except Exception as exc:
        logger.warning(
            "session_room_terminate_failed strategy=room_disconnect error=%s",
            _redact_sensitive_text(exc),
        )
    return "none"


async def _say_session_line_when_clear(
    session: AgentSession,
    text: str,
    *,
    deadline: float,
    label: str,
    started: float,
) -> bool:
    """Speak a session-management line, waiting out any in-progress user speech.

    The assistant-speech gate drops any utterance created while the user is
    talking (assistant_speech_start_blocked_reason=user_speaking), which
    previously swallowed the time-limit heads-up whenever the user happened to be
    mid-sentence — the user literally would not hear it. Poll the live user state
    and only say the line once they pause, then say it non-interruptibly so the
    heads-up can't be cut off. Bounded by `deadline` (monotonic) so retrying can
    never push past the hard cap; if the user never pauses in time we log and skip
    rather than block the end.
    """
    poll_seconds = 0.25
    while _latest_user_state_for_greeting == "speaking":
        now = time.monotonic()
        if now >= deadline:
            logger.warning(
                "session_time_%s_skipped=true reason=user_speaking_through_deadline elapsed_seconds=%s",
                label,
                _fmt_seconds(now - started),
            )
            return False
        await asyncio.sleep(min(poll_seconds, max(0.0, deadline - now)))
    try:
        handle = await session.say(text, allow_interruptions=False)
        wait_for_playout = getattr(handle, "wait_for_playout", None)
        if callable(wait_for_playout):
            await asyncio.wait_for(wait_for_playout(), timeout=8.0)
        logger.info(
            "session_time_%s_spoken=true elapsed_seconds=%s",
            label,
            _fmt_seconds(time.monotonic() - started),
        )
        return True
    except Exception as exc:
        logger.warning(
            "session_time_%s_say_failed=true error=%s",
            label,
            _redact_sensitive_text(exc),
        )
        return False


async def _run_session_time_limit(session: AgentSession, ctx: JobContext) -> None:
    """Enforce the hard session cap: speak a heads-up, then end the room.

    Anchored to a monotonic start so the hard cut lands at SESSION_MAX_DURATION_SECONDS
    regardless of how long the spoken lines take. Cancelled on shutdown.
    """
    if not SESSION_TIME_LIMIT_ENABLED or SESSION_MAX_DURATION_SECONDS <= 0:
        return
    limit = SESSION_MAX_DURATION_SECONDS
    warn_at = limit - SESSION_ENDING_WARNING_SECONDS
    started = time.monotonic()
    logger.info(
        "session_time_limit_armed=true limit_seconds=%s warning_seconds=%s warning_at_seconds=%s",
        limit,
        SESSION_ENDING_WARNING_SECONDS,
        max(0.0, warn_at),
    )
    try:
        if SESSION_ENDING_WARNING_TEXT and warn_at > 0:
            await asyncio.sleep(warn_at)
            logger.info(
                "session_time_warning_speaking=true elapsed_seconds=%s remaining_seconds=%s",
                _fmt_seconds(time.monotonic() - started),
                _fmt_seconds(limit - (time.monotonic() - started)),
            )
            # Hold a few seconds back from the hard cap so a heads-up spoken late
            # (after waiting out the user) still finishes before the goodbye.
            warning_deadline = started + limit - SESSION_ENDING_WARNING_PLAYBACK_RESERVE_SECONDS
            await _say_session_line_when_clear(
                session,
                SESSION_ENDING_WARNING_TEXT,
                deadline=warning_deadline,
                label="warning",
                started=started,
            )

        remaining = limit - (time.monotonic() - started)
        if remaining > 0:
            await asyncio.sleep(remaining)

        logger.info(
            "session_time_limit_reached=true elapsed_seconds=%s; ending session",
            _fmt_seconds(time.monotonic() - started),
        )
        if SESSION_ENDING_GOODBYE_TEXT:
            # The goodbye runs through the same speech gate, so briefly wait out
            # any in-progress user speech before saying it; bounded tightly since
            # the room tears down right after.
            await _say_session_line_when_clear(
                session,
                SESSION_ENDING_GOODBYE_TEXT,
                deadline=time.monotonic() + SESSION_ENDING_GOODBYE_CLEAR_WAIT_SECONDS,
                label="goodbye",
                started=started,
            )

        await _terminate_room(ctx)
    except asyncio.CancelledError:
        logger.info("session_time_limit_cancelled=true (session ended before limit)")
        raise


async def entrypoint(ctx: JobContext):
    job_started_at = time.monotonic()
    logger.info(
        "Startup deployment version: git_commit_sha=%s railway_deployment_id=%s railway_service=%s railway_environment=%s",
        _deployment_git_commit_sha(),
        os.getenv("RAILWAY_DEPLOYMENT_ID", "n/a"),
        os.getenv("RAILWAY_SERVICE_NAME", "n/a"),
        os.getenv("RAILWAY_ENVIRONMENT_NAME", "n/a"),
    )
    _run_db_migrations_on_startup()
    _log_memory_identity_readiness()
    _log_livekit_tts_source_inspection()
    openrouter_model = os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
    openrouter_api_key_present = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    provider_order_raw = (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").strip()
    provider_order = [p.strip() for p in provider_order_raw.split(",") if p.strip()]
    openrouter_allow_fallbacks_requested = env_bool("OPENROUTER_ALLOW_FALLBACKS", True)
    with_openrouter_sig = inspect.signature(openai.LLM.with_openrouter)
    provider_routing_applied = False
    provider_routing_skip_reason = "provider_order_not_set"
    llm: Any
    if provider_order:
        provider_payload = {"order": provider_order, "allow_fallbacks": openrouter_allow_fallbacks_requested}
        if "provider" in with_openrouter_sig.parameters:
            llm = openai.LLM.with_openrouter(model=openrouter_model, provider=provider_payload)
            provider_routing_applied = True
            provider_routing_skip_reason = "none"
        elif "extra_body" in with_openrouter_sig.parameters:
            try:
                llm = openai.LLM.with_openrouter(model=openrouter_model, extra_body={"provider": provider_payload})
                provider_routing_applied = True
                provider_routing_skip_reason = "none"
            except TypeError:
                llm = openai.LLM.with_openrouter(model=openrouter_model)
                provider_routing_skip_reason = "with_openrouter_rejected_extra_body_at_runtime"
                logger.warning(
                    "OpenRouter provider routing not applied: wrapper rejected extra_body at runtime; provider fallback routing is not active"
                )
        else:
            llm = openai.LLM.with_openrouter(model=openrouter_model)
            provider_routing_skip_reason = "with_openrouter_missing_provider_or_extra_body_parameter"
            logger.warning(
                "OpenRouter provider routing not applied: current LiveKit OpenRouter wrapper exposes neither provider nor extra_body; provider fallback routing is not active"
            )
    else:
        llm = openai.LLM.with_openrouter(model=openrouter_model)
    openrouter_allow_fallbacks_effective = openrouter_allow_fallbacks_requested if provider_routing_applied else False
    logger.info(
        "LLM provider config: openrouter_api_key_present=%s openrouter_model_present=%s openrouter_model=%s openrouter_provider_order=%s openrouter_allow_fallbacks_requested=%s openrouter_allow_fallbacks_effective=%s provider_routing_applied=%s provider_routing_skip_reason=%s",
        openrouter_api_key_present,
        bool(openrouter_model),
        openrouter_model,
        ",".join(provider_order) if provider_order else "none",
        openrouter_allow_fallbacks_requested,
        openrouter_allow_fallbacks_effective,
        provider_routing_applied,
        provider_routing_skip_reason,
    )
    logger.info(
        "Transcript context startup config: transcript_context_layer_enabled=%s transcript_context_llm_enabled=%s transcript_context_llm_model=%s transcript_context_llm_timeout_ms=%s transcript_context_debug=%s",
        transcript_context_layer_enabled(),
        transcript_context_llm_enabled(),
        transcript_context_llm_model(),
        transcript_context_llm_timeout_ms(),
        transcript_context_debug(),
    )
    # Warm the OpenRouter generation model (and the transcript-context model when
    # enabled) with tiny throwaway requests so the first real turn isn't paying
    # connection/provider cold-start cost. Runs concurrently with session setup
    # and the greeting; best-effort and never blocks the live path.
    if env_bool("LLM_WARMUP_ENABLED", True):
        logger.info("LLM warmup starting: target=generation+transcript_context")
        asyncio.create_task(_prewarm_models(llm))
    # Pre-render the fixed greeting to an in-memory WAV once at startup so the
    # first audio a user hears is instant and identical, instead of paying a live
    # Hume round-trip each session. Runs concurrently with session setup; if it
    # is not ready by the time the greeting plays, the greeting falls back to
    # live TTS. Skipped when a static cached source is already configured.
    _prerender_static_source_configured = GREETING_USE_CACHED_AUDIO and bool(GREETING_AUDIO_URL or GREETING_AUDIO_PATH)
    if ENABLE_FIXED_GREETING and GREETING_PRERENDER_AT_STARTUP and not _prerender_static_source_configured:
        logger.info("Greeting pre-render starting: greeting_prerender_enabled=true greeting_text_length=%s", len(GREETING_TEXT))
        asyncio.create_task(_prerender_greeting())
    else:
        logger.info(
            "Greeting pre-render skipped: greeting_prerender_enabled=%s fixed_greeting_enabled=%s static_cached_source_configured=%s",
            GREETING_PRERENDER_AT_STARTUP,
            ENABLE_FIXED_GREETING,
            _prerender_static_source_configured,
        )
    # TODO: Re-enable Tavily using LiveKit's supported function-tool pattern.
    logger.warning("Skipping Tavily tools for MVP voice path; Exa is the only active search provider when enabled")

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
            "Unknown LIVEKIT_TURN_DETECTION_MODE=%s. Valid values: vad | stt | default. "
            "Falling back to vad. (The audio end-of-turn detector is configured "
            "separately via LIVEKIT_TURN_DETECTOR_ENABLED; 'audio' is not a valid mode here.)",
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
        "min_words": int(os.getenv("LIVEKIT_INTERRUPTION_MIN_WORDS", "2")),
        "min_duration": float(os.getenv("LIVEKIT_INTERRUPTION_MIN_DURATION", "0.65")),
        "resume_false_interruption": env_bool("LIVEKIT_RESUME_FALSE_INTERRUPTION", True),
        "false_interruption_timeout": float(os.getenv("LIVEKIT_FALSE_INTERRUPTION_TIMEOUT", "1.0")),
    }
    logger.info(
        "Resolved interruption config: %s resume_false_interruption_active=%s",
        interruption_options,
        interruption_options.get("resume_false_interruption") and interruption_options.get("enabled"),
    )
    endpointing_mode = os.getenv("LIVEKIT_ENDPOINTING_MODE", "dynamic")
    endpointing_min_delay = float(os.getenv("ENDPOINTING_MIN_DELAY_SECONDS", os.getenv("LIVEKIT_ENDPOINTING_MIN_DELAY", "1.1")))
    endpointing_max_delay = float(os.getenv("ENDPOINTING_MAX_DELAY_SECONDS", os.getenv("LIVEKIT_ENDPOINTING_MAX_DELAY", "2.4")))
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
    logger.info("HUME_DIRECT_API_TTS enabled=%s", HUME_DIRECT_API_TTS)
    logger.info(
        "Spoken text normalization latency guidance: spoken_text_normalization_enabled=%s spoken_text_normalization_mode=%s tts_input_buffering_mode=%s lowest_latency_setting=SPOKEN_TEXT_NORMALIZATION=false best_sentence_quality_setting=SPOKEN_TEXT_NORMALIZATION=true,mode=buffered_full_segment",
        SPOKEN_TEXT_NORMALIZATION,
        SPOKEN_TEXT_NORMALIZATION_MODE,
        SPOKEN_TEXT_NORMALIZATION_MODE,
    )
    logger.info(
        "Search startup config: search_enabled=%s search_provider=%s search_max_results=%s search_timeout_seconds=%s search_disabled_reason=%s",
        search_enabled(),
        search_provider(),
        search_max_results(),
        search_timeout_seconds(),
        search_disabled_reason() or "none",
    )
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

    metadata_debug_entries = _metadata_debug_entries_from_context(ctx)
    for metadata_label, metadata_value in metadata_debug_entries:
        logger.info(
            "Runtime metadata candidate: metadata_source=%s metadata_present=%s metadata_keys=%s raw_metadata=%s",
            metadata_label,
            bool(metadata_value),
            _metadata_keys(metadata_value),
            _safe_metadata_preview(metadata_value),
        )
    metadata_candidates = [metadata_value for _, metadata_value in metadata_debug_entries if metadata_value]
    runtime_context = runtime_context_from_metadata(*metadata_candidates)
    logger.info(
        "Runtime context resolved: client_timezone_present=%s client_timezone_value=%s extracted_client_timezone=%s timezone_resolution_source=%s session_timezone=%s runtime_current_date=%s runtime_current_time=%s runtime_datetime_iso=%s runtime_context_injected=%s",
        runtime_context.client_timezone_present,
        _redact_sensitive_text(runtime_context.client_timezone_value or "none"),
        _redact_sensitive_text(runtime_context.client_timezone_value or "none"),
        runtime_context.timezone_resolution_source,
        runtime_context.session_timezone,
        runtime_context.current_date,
        runtime_context.current_time,
        runtime_context.current_datetime_iso,
        True,
    )

    global _active_memory_layer, _audiointeraction_shadow, _inworld_voice_profile_shadow, _calibration_session_id
    _calibration_session_id = str(_safe_attr(_safe_attr(ctx, "room"), "name") or "unknown")
    logger.info(
        "tts_runtime_selection hume_active=%s tts_provider=%s tts_fallback_provider=%s omnivoice_inactive=%s omnivoice_enabled=%s omnivoice_expressive_planner_enabled=%s",
        TTS_PROVIDER == "hume",
        TTS_PROVIDER,
        TTS_FALLBACK_PROVIDER,
        TTS_PROVIDER != "omnivoice",
        env_bool("OMNIVOICE_ENABLED", False),
        env_bool("OMNIVOICE_EXPRESSIVE_PLANNER_ENABLED", False),
    )
    _audiointeraction_shadow = build_shadow_from_env()
    if _audiointeraction_shadow is not None:
        _audiointeraction_shadow.start()
        logger.info(
            "AudioInteraction shadow startup: audiointeraction_mode=shadow endpoint_present=true timeout_ms=%s debug_text=%s",
            _audiointeraction_shadow.timeout_ms,
            _audiointeraction_shadow.debug_text,
        )
        try:
            ctx.add_shutdown_callback(_audiointeraction_shadow.aclose)
        except Exception as exc:
            logger.warning("audiointeraction_shutdown_callback_unavailable=true error_type=%s error=%s", type(exc).__name__, exc)
    else:
        logger.info("AudioInteraction shadow startup: audiointeraction_mode=%s shadow_active=false", audiointeraction_mode())

    _inworld_voice_profile_shadow = build_inworld_shadow_from_env()
    if _inworld_voice_profile_shadow is not None:
        _inworld_voice_profile_shadow.start()
        try:
            ctx.add_shutdown_callback(_inworld_voice_profile_shadow.aclose)
        except Exception as exc:
            logger.warning("inworld_voice_profile_shutdown_callback_unavailable=true error_type=%s error=%s", type(exc).__name__, exc)
    else:
        logger.info("inworld_voice_profile_startup shadow_active=false")

    memory_layer_instance: MemoryLayer | None = None
    memory_preload_note: str | None = None
    if memory_enabled():
        room_name = _safe_attr(_safe_attr(ctx, "room"), "name") or None
        memory_identity = identity_from_metadata(metadata_candidates, fallback_guest_id=room_name)
        memory_layer_instance = MemoryLayer(memory_identity)
        try:
            preloaded_memories = await asyncio.wait_for(memory_layer_instance.preload(), timeout=2.0)
        except asyncio.TimeoutError:
            preloaded_memories = []
            logger.warning("memory_preload status=timeout timeout_seconds=2.0")
        # Split confirmed emotional-calibration patterns out of the general
        # memories into their own private "what we've learned" note, and combine.
        general_memories, emotional_patterns = partition_emotional_patterns(preloaded_memories)
        general_note = MemoryLayer.preload_note(general_memories)
        emotional_note = emotional_pattern_preload_note(emotional_patterns)
        memory_preload_note = "\n\n".join(n for n in (general_note, emotional_note) if n) or None
        logger.info(
            "Memory layer startup: memory_enabled=true memory_scope=%s memory_identity_present=%s preloaded_memory_count=%s emotional_pattern_count=%s preload_note_present=%s",
            memory_identity.scope,
            memory_identity.present,
            len(general_memories),
            len(emotional_patterns),
            bool(memory_preload_note),
        )
        try:
            ctx.add_shutdown_callback(memory_layer_instance.aclose)
        except Exception as exc:
            logger.warning("memory_shutdown_callback_unavailable=true error_type=%s error=%s", type(exc).__name__, exc)
        asyncio.create_task(memory_layer_instance.rebuild_index_if_empty())
    else:
        logger.info("Memory layer startup: memory_enabled=false")
    _active_memory_layer = memory_layer_instance
    selected_voice_engine = voice_engine()
    logger.info("voice_engine_selected=%s", selected_voice_engine)
    if selected_voice_engine == "hume_evi":
        await _run_hume_evi_voice_engine(ctx)
        return

    lucy_agent = LucyAgent(
        runtime_context=runtime_context,
        memory_layer=memory_layer_instance,
        memory_preload_note=memory_preload_note,
    )

    session_tts = build_tts()
    # Pick this session's stable voice preset and prime the active language.
    _init_session_voice_and_language(session_tts)
    # If this TTS owns a debug http_session, close it on shutdown so the
    # long-lived session doesn't leak it the way the pre-render did.
    try:
        ctx.add_shutdown_callback(lambda: _close_tts_owned_session(session_tts))
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts_shutdown_callback_unavailable=true error_type=%s error=%s", type(exc).__name__, exc)
    session_kwargs: dict[str, Any] = {
        "stt": build_stt(),
        "llm": llm,
        "tts": session_tts,
        "vad": build_vad(),
    }

    preemptive_generation_options = {"enabled": PREEMPTIVE_GENERATION_ENABLED}
    _audio_turn_detector, _audio_turn_detector_info = build_audio_turn_detector()
    logger.info(
        "livekit_agents_version=%s livekit_turn_detector_available=%s livekit_turn_detector_enabled=%s turn_detector_class=%s turn_detector_version=%s turn_detector_default_selection=%s turn_detector_fallback_used=%s turn_detector_error=%s",
        _livekit_agents_version(),
        _audio_turn_detector_info["available"],
        _audio_turn_detector_info["enabled"],
        _audio_turn_detector_info["class"],
        _audio_turn_detector_info["version"],
        _audio_turn_detector_info["default_selection"],
        _audio_turn_detector_info["fallback_used"],
        _audio_turn_detector_info["error"],
    )

    def _td(default_mode: str):
        return _audio_turn_detector if _audio_turn_detector is not None else default_mode

    _td_active = _audio_turn_detector is not None
    resolved_turn_detection_mode = "unknown"
    if STT_PROVIDER == "deepgram_flux":
        if resolved_livekit_turn_detection_mode == "stt":
            session_kwargs["turn_handling"] = TurnHandlingOptions(
                turn_detection=_td("stt"),
                interruption=interruption_options,
                preemptive_generation=preemptive_generation_options,
            )
            resolved_turn_detection_mode = "stt"
            logger.info("Using Flux STT-based turn detection")
        elif resolved_livekit_turn_detection_mode == "vad":
            try:
                session_kwargs["turn_handling"] = TurnHandlingOptions(
                    turn_detection=_td("vad"),
                    interruption=interruption_options,
                    preemptive_generation=preemptive_generation_options,
                )
                resolved_turn_detection_mode = "vad"
                logger.info("Using Flux VAD-based turn detection")
            except Exception as e:
                logger.warning("VAD turn_detection mode unavailable in this LiveKit version, falling back to stt: %s", e)
                session_kwargs["turn_handling"] = TurnHandlingOptions(
                    turn_detection=_td("stt"),
                    interruption=interruption_options,
                    preemptive_generation=preemptive_generation_options,
                )
                resolved_turn_detection_mode = "stt"
        elif resolved_livekit_turn_detection_mode == "default":
            session_kwargs["turn_handling"] = TurnHandlingOptions(
                preemptive_generation=preemptive_generation_options,
            )
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
            turn_detection=_td("vad"),
            interruption=interruption_options,
            endpointing={
                "mode": endpointing_mode,
                "min_delay": endpointing_min_delay,
                "max_delay": endpointing_max_delay,
            },
            preemptive_generation=preemptive_generation_options,
        )
        resolved_turn_detection_mode = "vad"
        logger.info("Using Mistral VAD-only turn handling")
    else:
        session_kwargs["turn_handling"] = TurnHandlingOptions(
            turn_detection=_td("vad"),
            interruption=interruption_options,
            preemptive_generation=preemptive_generation_options,
        )
        resolved_turn_detection_mode = "vad"
        logger.info("Using non-Flux VAD turn handling")

    if _td_active:
        resolved_turn_detection_mode = "livekit_audio_turn_detector"
    logger.info(
        "turn_detection_resolved=%s livekit_turn_detector_active=%s turn_detector_version_active=%s turn_detector_fallback_used=%s endpointing_dynamic_enabled=%s text_turn_detector_used=false",
        resolved_turn_detection_mode,
        _td_active,
        _audio_turn_detector_info["version"] if _td_active else "n/a",
        _audio_turn_detector_info["fallback_used"],
        LIVEKIT_ENDPOINTING_DYNAMIC_ENABLED,
    )
    session_kwargs["preemptive_generation"] = PREEMPTIVE_GENERATION_ENABLED
    logger.info(
        "Preemptive generation config: preemptive_generation_enabled=%s context_or_tools_mutate=%s",
        PREEMPTIVE_GENERATION_ENABLED,
        True,
    )
    session = AgentSession(**session_kwargs)
    resolved_stt = session_kwargs.get("stt")
    logger.info("Resolved STT type: %s", type(resolved_stt).__name__)
    resolved_vad = session_kwargs.get("vad")
    logger.info("Resolved VAD provider: provider=%s vad_type=%s", VAD_PROVIDER, type(resolved_vad).__name__)
    silero_imported = "silero" in globals()
    silero_passed_to_session = "silero" in type(resolved_vad).__module__.lower() or "silero" in type(resolved_vad).__name__.lower()
    if silero_passed_to_session:
        silero_reason = "silero_selected_or_fallback_runtime_vad"
    elif silero_imported:
        silero_reason = "silero_imported_or_preloaded_by_runtime_not_active_vad"
    else:
        silero_reason = "silero_not_imported"
    logger.info(
        "VAD runtime diagnostic: active_vad_provider=%s silero_imported=%s silero_initialized=%s silero_passed_to_session=%s silero_runtime_warning_possible_reason=%s",
        VAD_PROVIDER,
        silero_imported,
        _silero_initialized,
        silero_passed_to_session,
        silero_reason,
    )
    logger.info("Resolved turn detection mode: %s", resolved_turn_detection_mode)

    attach_session_diagnostics(session)
    _attach_optional_interruption_diagnostics(session)

    room_options = build_room_options()
    session_started_at = 0.0
    if room_options is not None:
        logger.info("Starting session with ai-coustics room_options attached")
        await session.start(room=ctx.room, agent=lucy_agent, room_options=room_options)
        session_started_at = time.monotonic()
    else:
        logger.info("Starting session without ai-coustics room_options")
        await session.start(room=ctx.room, agent=lucy_agent)
        session_started_at = time.monotonic()

    # Arm the session time limit on a background clock: it speaks a heads-up
    # ~SESSION_ENDING_WARNING_SECONDS before the cap, then ends the room.
    if SESSION_TIME_LIMIT_ENABLED and SESSION_MAX_DURATION_SECONDS > 0:
        session_timer_task = asyncio.create_task(_run_session_time_limit(session, ctx))

        async def _cancel_session_timer() -> None:
            if not session_timer_task.done():
                session_timer_task.cancel()

        ctx.add_shutdown_callback(_cancel_session_timer)

    greeting_agent_listening_at = 0.0
    for _ in range(50):
        state = _safe_attr(session, "agent_state", "").strip().lower()
        if state == "listening":
            greeting_agent_listening_at = time.monotonic()
            break
        await asyncio.sleep(0.1)

    greeting_audio_source = "url" if GREETING_AUDIO_URL else ("path" if GREETING_AUDIO_PATH else "none")
    logger.info(
        "Fixed greeting config: fixed_greeting_enabled=%s greeting_use_cached_audio=%s greeting_audio_source=%s greeting_text_length=%s live_hume_primary_path=%s",
        ENABLE_FIXED_GREETING,
        GREETING_USE_CACHED_AUDIO,
        greeting_audio_source,
        len(GREETING_TEXT),
        not GREETING_USE_CACHED_AUDIO,
    )
    if not GREETING_USE_CACHED_AUDIO:
        logger.info(
            "Fixed greeting cached audio disabled: greeting_path=hume_live_tts cached_audio_load_attempted=false greeting_audio_source=%s",
            greeting_audio_source,
        )

    if not ENABLE_FIXED_GREETING:
        logger.info(
            "Fixed greeting skipped: fixed_greeting_enabled=%s greeting_path=skipped fixed_greeting_skipped_reason=disabled greeting_cancelled_due_to_user_speech=%s",
            False,
            False,
        )
        return

    greeting_cancelled_due_to_user_speech = False
    if _latest_user_speaking_at > session_started_at or _latest_user_state_for_greeting == "speaking":
        greeting_cancelled_due_to_user_speech = True
        logger.warning(
            "Fixed greeting skipped: greeting_path=skipped fixed_greeting_skipped_reason=user_started_speaking_before_greeting greeting_cancelled_due_to_user_speech=%s latest_user_state=%s latest_user_speaking_at=%s session_started_at=%s",
            True,
            _latest_user_state_for_greeting,
            _latest_user_speaking_at,
            session_started_at,
        )
        return

    logger.info("About to play fixed greeting")
    greeting_tts_request_at = time.monotonic()
    greeting_tts_completed_at = 0.0
    greeting_handle = None
    greeting_path = "skipped"
    greeting_fallback_reason = "none"
    greeting_hume_error = "none"
    greeting_first_audio_marker: dict[str, float] = {}
    greeting_hume_request_index_before = _hume_tts_request_counter
    greeting_hume_request_index_after = _hume_tts_request_counter

    if GREETING_USE_CACHED_AUDIO and greeting_audio_source != "none":
        audio_bytes, loaded_source, load_error = await _load_cached_greeting_audio_bytes()
        greeting_audio_source = loaded_source
        if audio_bytes:
            if _latest_user_speaking_at > greeting_tts_request_at or _latest_user_state_for_greeting == "speaking":
                greeting_cancelled_due_to_user_speech = True
                logger.warning(
                    "Fixed greeting skipped: greeting_path=skipped fixed_greeting_skipped_reason=user_started_speaking_before_greeting greeting_cancelled_due_to_user_speech=%s latest_user_state=%s latest_user_speaking_at=%s greeting_playout_started_at=%s",
                    True,
                    _latest_user_state_for_greeting,
                    _latest_user_speaking_at,
                    greeting_tts_request_at,
                )
                return
            try:
                _validate_cached_wav_audio(audio_bytes)
                greeting_path = "cached_audio"
                logger.info(
                    "Fixed greeting cached audio starting: greeting_path=%s greeting_audio_source=%s greeting_playout_started_at=%s greeting_cancelled_due_to_user_speech=%s",
                    greeting_path,
                    greeting_audio_source,
                    greeting_tts_request_at,
                    False,
                )
                greeting_handle = await session.say(
                    GREETING_TEXT,
                    audio=_cached_wav_audio_frames(audio_bytes, greeting_first_audio_marker),
                    allow_interruptions=False,
                )
            except Exception as exc:
                greeting_path = "hume_live_tts"
                greeting_fallback_reason = f"cached_audio_error_{type(exc).__name__}"
                logger.warning(
                    "Fixed greeting cached audio unavailable; falling back to live TTS: greeting_path=%s fallback_reason=%s greeting_audio_source=%s error=%s",
                    greeting_path,
                    greeting_fallback_reason,
                    greeting_audio_source,
                    _redact_sensitive_text(exc),
                )
        else:
            greeting_path = "hume_live_tts"
            greeting_fallback_reason = load_error or "cached_audio_missing"
            logger.warning(
                "Fixed greeting cached audio missing; falling back to live TTS: greeting_path=%s fallback_reason=%s greeting_audio_source=%s",
                greeting_path,
                greeting_fallback_reason,
                greeting_audio_source,
            )
    elif GREETING_USE_CACHED_AUDIO:
        greeting_path = "hume_live_tts"
        greeting_fallback_reason = "cached_audio_missing"
        logger.warning(
            "Fixed greeting cached audio requested without source; falling back to live TTS: greeting_path=%s fallback_reason=%s greeting_audio_source=%s",
            greeting_path,
            greeting_fallback_reason,
            greeting_audio_source,
        )

    # Serve from the in-memory pre-rendered greeting when no static cached source
    # produced a handle. This is the instant, identical-every-session path.
    if greeting_handle is None and _prerendered_greeting_wav is not None:
        if _latest_user_speaking_at > greeting_tts_request_at or _latest_user_state_for_greeting == "speaking":
            greeting_cancelled_due_to_user_speech = True
            logger.warning(
                "Fixed greeting skipped: greeting_path=skipped fixed_greeting_skipped_reason=user_started_speaking_before_greeting greeting_cancelled_due_to_user_speech=%s latest_user_state=%s latest_user_speaking_at=%s greeting_playout_started_at=%s",
                True,
                _latest_user_state_for_greeting,
                _latest_user_speaking_at,
                greeting_tts_request_at,
            )
            return
        try:
            _validate_cached_wav_audio(_prerendered_greeting_wav)
            greeting_path = "prerendered_cached_audio"
            greeting_audio_source = "startup_prerender"
            logger.info(
                "Fixed greeting pre-rendered audio starting: greeting_path=%s greeting_audio_source=%s greeting_playout_started_at=%s greeting_cancelled_due_to_user_speech=%s prerendered_wav_bytes=%s",
                greeting_path,
                greeting_audio_source,
                greeting_tts_request_at,
                False,
                len(_prerendered_greeting_wav),
            )
            greeting_handle = await session.say(
                GREETING_TEXT,
                audio=_cached_wav_audio_frames(_prerendered_greeting_wav, greeting_first_audio_marker),
                allow_interruptions=False,
            )
        except Exception as exc:
            greeting_path = "hume_live_tts"
            greeting_fallback_reason = f"prerendered_audio_error_{type(exc).__name__}"
            logger.warning(
                "Fixed greeting pre-rendered audio unavailable; falling back to live TTS: greeting_path=%s fallback_reason=%s error=%s",
                greeting_path,
                greeting_fallback_reason,
                _redact_sensitive_text(exc),
            )

    if greeting_handle is None:
        if _latest_user_speaking_at > greeting_tts_request_at or _latest_user_state_for_greeting == "speaking":
            greeting_cancelled_due_to_user_speech = True
            logger.warning(
                "Fixed greeting skipped: greeting_path=skipped fixed_greeting_skipped_reason=user_started_speaking_before_greeting greeting_cancelled_due_to_user_speech=%s latest_user_state=%s latest_user_speaking_at=%s greeting_playout_started_at=%s",
                True,
                _latest_user_state_for_greeting,
                _latest_user_speaking_at,
                greeting_tts_request_at,
            )
            return
        greeting_path = "hume_live_tts"
        logger.info(
            "Fixed greeting live TTS starting: fixed_greeting_enabled=%s greeting_use_cached_audio=%s greeting_path=%s fallback_reason=%s greeting_text_length=%s greeting_tts_request_started_at=%s greeting_cancelled_due_to_user_speech=%s hume_request_index_before=%s hume_model_version=%s hume_instant_mode=%s hume_voice_present=%s hume_voice_kind=%s hume_speed=%s hume_trailing_silence=%s hume_description_applied=%s hume_style_context_applied=%s hume_tts_debug_http=%s hume_tts_build_completed_at=%s hume_tts_build_to_greeting_seconds=%s spoken_text_normalization_enabled=%s spoken_text_normalization_mode=%s",
            ENABLE_FIXED_GREETING,
            GREETING_USE_CACHED_AUDIO,
            greeting_path,
            greeting_fallback_reason,
            len(GREETING_TEXT),
            greeting_tts_request_at,
            False,
            greeting_hume_request_index_before,
            _last_hume_model_version,
            _last_hume_instant_mode,
            _last_hume_voice_present,
            _last_hume_voice_kind,
            _last_hume_speed,
            _last_hume_trailing_silence,
            _last_hume_description_applied,
            _last_hume_style_context_applied,
            _last_hume_tts_debug_http,
            _last_hume_tts_build_completed_at,
            _fmt_seconds(greeting_tts_request_at - _last_hume_tts_build_completed_at if _last_hume_tts_build_completed_at > 0 else None),
            SPOKEN_TEXT_NORMALIZATION,
            SPOKEN_TEXT_NORMALIZATION_MODE,
        )
        try:
            greeting_handle = await session.say(
                GREETING_TEXT,
                allow_interruptions=False,
            )
        except Exception as exc:
            greeting_hume_error = f"{type(exc).__name__}:{_redact_sensitive_text(exc)}"
            logger.error(
                "Fixed greeting live TTS error: greeting_path=%s hume_error=%s greeting_tts_request_started_at=%s elapsed_seconds=%s hume_request_index_before=%s hume_request_index_after=%s",
                greeting_path,
                greeting_hume_error,
                greeting_tts_request_at,
                _fmt_seconds(time.monotonic() - greeting_tts_request_at),
                greeting_hume_request_index_before,
                _hume_tts_request_counter,
            )
            raise

    greeting_after_say_at = time.monotonic()
    greeting_tts_completed_at = greeting_after_say_at
    greeting_hume_request_index_after = _hume_tts_request_counter
    logger.info(
        "Fixed greeting say completed: handle_type=%s handle_id=%s interrupted=%s greeting_path=%s greeting_tts_request_completed_at=%s greeting_tts_session_say_seconds=%s hume_request_index_before=%s hume_request_index_after=%s hume_request_index=%s",
        type(greeting_handle).__name__,
        _safe_attr(greeting_handle, "id"),
        _safe_attr(greeting_handle, "interrupted"),
        greeting_path,
        greeting_tts_completed_at,
        _fmt_seconds(greeting_tts_completed_at - greeting_tts_request_at),
        greeting_hume_request_index_before,
        greeting_hume_request_index_after,
        greeting_hume_request_index_after if greeting_hume_request_index_after != greeting_hume_request_index_before else "not_observed_without_hume_http_debug",
    )

    wait_for_playout = getattr(greeting_handle, "wait_for_playout", None)
    greeting_playout_done_at = 0.0
    if callable(wait_for_playout):
        try:
            await asyncio.wait_for(wait_for_playout(), timeout=8.0)
            greeting_playout_done_at = time.monotonic()
            logger.info("Greeting playout completed")
        except TimeoutError:
            logger.warning("Greeting playout wait timed out")
        except Exception as e:
            logger.warning("Greeting playout wait failed: %s", e)
    else:
        logger.warning("Greeting handle does not support wait_for_playout")

    greeting_first_audio_at = greeting_first_audio_marker.get("at")
    if greeting_first_audio_at is None and greeting_path == "hume_live_tts":
        greeting_first_audio_at = greeting_after_say_at
    logger.info(
        "Greeting latency summary: fixed_greeting_enabled=%s greeting_use_cached_audio=%s greeting_audio_source=%s greeting_path=%s fallback_reason=%s greeting_tts_request_started_at=%s greeting_tts_request_completed_at=%s greeting_playout_started_at=%s greeting_time_to_first_audio_seconds=%s greeting_first_audio_seconds=%s greeting_first_audio_source=%s greeting_total_tts_seconds=%s greeting_total_playout_seconds=%s greeting_cancelled_due_to_user_speech=%s greeting_job_to_session_start=%s greeting_session_start_to_agent_listening=%s greeting_text_length=%s hume_request_index=%s hume_request_index_before=%s hume_request_index_after=%s hume_model_version=%s hume_instant_mode=%s hume_voice_present=%s hume_voice_kind=%s hume_speed=%s hume_trailing_silence=%s hume_description_applied=%s hume_style_context_applied=%s hume_error=%s",
        ENABLE_FIXED_GREETING,
        GREETING_USE_CACHED_AUDIO,
        greeting_audio_source,
        greeting_path,
        greeting_fallback_reason,
        greeting_tts_request_at,
        greeting_tts_completed_at,
        greeting_tts_request_at,
        _fmt_seconds((greeting_first_audio_at - greeting_tts_request_at) if greeting_first_audio_at is not None else None),
        _fmt_seconds((greeting_first_audio_at - greeting_tts_request_at) if greeting_first_audio_at is not None else None),
        "cached_audio_frame_marker" if greeting_first_audio_marker.get("at") is not None else ("session_say_return_estimate" if greeting_path == "hume_live_tts" else "unavailable"),
        _fmt_seconds(greeting_after_say_at - greeting_tts_request_at if greeting_after_say_at > 0 else -1.0),
        _fmt_seconds(greeting_playout_done_at - greeting_tts_request_at if greeting_playout_done_at > 0 else -1.0),
        greeting_cancelled_due_to_user_speech,
        _fmt_seconds(session_started_at - job_started_at if session_started_at > 0 else -1.0),
        _fmt_seconds(greeting_agent_listening_at - session_started_at if greeting_agent_listening_at > 0 and session_started_at > 0 else -1.0),
        len(GREETING_TEXT),
        greeting_hume_request_index_after if greeting_hume_request_index_after != greeting_hume_request_index_before else "not_observed_without_hume_http_debug",
        greeting_hume_request_index_before,
        greeting_hume_request_index_after,
        _last_hume_model_version,
        _last_hume_instant_mode,
        _last_hume_voice_present,
        _last_hume_voice_kind,
        _last_hume_speed,
        _last_hume_trailing_silence,
        _last_hume_description_applied,
        _last_hume_style_context_applied,
        greeting_hume_error,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

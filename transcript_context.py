import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Sequence

import aiohttp

logger = logging.getLogger(__name__)

ALLOWED_INTENTS = {
    "numeric_fragment",
    "language_request",
    "voice_change_request",
    "tool_request_email",
    "tool_request_search",
    "tool_request_document",
    "frustration_fragment",
    "unclear_fragment",
    "choice_delegation",
    "greeting_or_backchannel",
    "stop_or_cancel_request",
    "date_time_question",
    "profanity_reaction",
    "reference_to_prior_context",
    "unknown",
}

OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


@dataclass(slots=True)
class TranscriptContext:
    original_text: str
    cleaned_text: str
    should_replace_user_text: bool
    llm_context_note: str | None
    ambiguity_detected: bool
    clarification_suggested: bool
    detected_intent: str | None
    confidence: float
    source: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def transcript_context_layer_enabled() -> bool:
    return _env_bool("TRANSCRIPT_CONTEXT_LAYER_ENABLED", True)


def transcript_context_llm_enabled() -> bool:
    return _env_bool("TRANSCRIPT_CONTEXT_LLM_ENABLED", False)


def transcript_context_debug() -> bool:
    return _env_bool("TRANSCRIPT_CONTEXT_DEBUG", False)


def transcript_context_llm_timeout_ms() -> int:
    raw = os.getenv("TRANSCRIPT_CONTEXT_LLM_TIMEOUT_MS", "350")
    try:
        return max(1, min(int(raw), 5000))
    except Exception:
        return 350


def transcript_context_llm_model() -> str:
    return (
        os.getenv("TRANSCRIPT_CONTEXT_LLM_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or OPENROUTER_DEFAULT_MODEL
    ).strip() or OPENROUTER_DEFAULT_MODEL


def _provider_payload() -> dict[str, Any] | None:
    provider_order_raw = (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").strip()
    provider_order = [item.strip() for item in provider_order_raw.split(",") if item.strip()]
    if not provider_order:
        return None
    return {
        "order": provider_order,
        "allow_fallbacks": _env_bool("OPENROUTER_ALLOW_FALLBACKS", True),
    }


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_transcript(text: str) -> str:
    cleaned = _clean_spaces(text or "")
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    cleaned = re.sub(r"([?.!,]){3,}", r"\1", cleaned)
    return cleaned


def _normalized(text: str) -> str:
    return clean_transcript(text).lower().replace("’", "'")


def _is_backchannel(text: str) -> bool:
    return bool(re.fullmatch(r"(yeah|yep|yes|no|nope|okay|ok|mm|mhm|uh huh|right|sure|cool|alright|all right)[.!?]*", text))


def build_llm_context_note(context: TranscriptContext) -> str | None:
    intent = context.detected_intent
    if intent == "numeric_fragment":
        return "The user gave a numeric fragment. Do not assume what it refers to. Ask a short clarification question."
    if intent == "language_request":
        return "The user likely means a language spoken in Sri Lanka, such as Sinhala or Tamil. Do not say Sri Lankan is a language. Ask whether they mean Sinhala or Tamil."
    if intent == "voice_change_request":
        return "The user is reacting to or requesting a different voice. Acknowledge briefly; do not claim the voice changed unless a voice-change tool succeeds."
    if intent == "tool_request_email":
        return "The user may want something sent by email. Do not claim an email was sent unless an email tool succeeds. Ask for or confirm missing recipient/content details."
    if intent == "tool_request_search":
        return "The user is asking for lookup/search. If the target is unclear, ask what to search for; otherwise use the existing search tool flow."
    if intent == "tool_request_document":
        return "The user may want a document created. Do not claim a document was created unless a document tool succeeds. Clarify content if needed."
    if intent == "frustration_fragment":
        return "The user sounds frustrated or fragmented. Respond calmly and ask one short grounding question instead of assuming the missing context."
    if intent == "profanity_reaction":
        return "The user made a profanity-heavy reaction. Treat it as emotional emphasis, not a literal request, unless more context is available."
    if intent == "choice_delegation":
        return "The user is delegating a choice. Offer a simple recommendation or ask for one constraint if the choice is unclear."
    if intent == "reference_to_prior_context":
        return "The user referred to prior context with an ambiguous word like 'that' or 'it'. Use recent context if clear; otherwise ask what they mean."
    if intent == "unclear_fragment":
        return "The transcript is fragmentary or unclear. Do not over-interpret; ask a short clarification question."
    return context.llm_context_note


def detect_transcript_context(text: str) -> TranscriptContext:
    original = text or ""
    cleaned = clean_transcript(original)
    lower = _normalized(cleaned)
    intent: str = "unknown"
    ambiguity = False
    clarification = False
    should_replace = cleaned != original.strip() and bool(cleaned)
    confidence = 0.72

    if not cleaned:
        intent = "unclear_fragment"
        ambiguity = True
        clarification = True
        confidence = 0.3
    elif re.fullmatch(r"[\d\s,.-]+(?:and\s+)?[\d\s,.-]+", lower) and any(ch.isdigit() for ch in lower):
        intent = "numeric_fragment"
        ambiguity = True
        clarification = True
        should_replace = False
        confidence = 0.88
    elif "sri lankan" in lower or "sinhala" in lower or "tamil" in lower and "speak" in lower:
        intent = "language_request"
        ambiguity = "sri lankan" in lower
        clarification = ambiguity
        should_replace = False
        confidence = 0.9
    elif re.search(r"\b(another|different|new) voice\b|\bchange (your )?voice\b|\bdon't want to speak to this voice\b", lower):
        intent = "voice_change_request"
        ambiguity = False
        clarification = False
        should_replace = False
        confidence = 0.86
    elif re.search(r"\b(email|e-mail|send (that|this|it|me)|send .* to me)\b", lower):
        intent = "tool_request_email"
        ambiguity = bool(re.search(r"\b(that|this|it)\b", lower)) or "@" not in lower
        clarification = ambiguity
        should_replace = False
        confidence = 0.82
    elif re.search(r"\b(look that up|look it up|search|google|find out|check online|look up)\b", lower):
        intent = "tool_request_search"
        ambiguity = bool(re.search(r"\b(that|it)\b", lower)) and len(lower.split()) <= 5
        clarification = ambiguity
        should_replace = False
        confidence = 0.84
    elif re.search(r"\b(word doc|document|docx|make a doc|make .* document)\b", lower):
        intent = "tool_request_document"
        ambiguity = bool(re.search(r"\b(that|this|it)\b", lower))
        clarification = ambiguity
        should_replace = False
        confidence = 0.84
    elif re.search(r"\bwhat (time|day|date|month|year)\b|\btime is it\b|\btoday'?s date\b|\bis it (morning|afternoon|evening|night)\b", lower):
        intent = "date_time_question"
        confidence = 0.92
        should_replace = False
    elif _is_backchannel(lower):
        intent = "greeting_or_backchannel"
        confidence = 0.9
        should_replace = False
    elif re.search(r"\b(stop|cancel|nevermind|never mind|quit|end this)\b", lower):
        intent = "stop_or_cancel_request"
        confidence = 0.84
        should_replace = False
    elif re.search(r"\b(anyone pick anyone|whatever works|you choose|pick for me|up to you)\b", lower):
        intent = "choice_delegation"
        ambiguity = True
        clarification = False
        should_replace = False
        confidence = 0.78
    elif re.search(r"\b(fuck|fucking|shit|damn|uber)\b", lower) and len(lower.split()) <= 5:
        intent = "profanity_reaction" if re.search(r"\b(fuck|fucking|shit|damn)\b", lower) else "frustration_fragment"
        ambiguity = True
        clarification = False
        should_replace = False
        confidence = 0.78
    elif re.search(r"\b(that|this|it|the thing|what we said)\b", lower) and len(lower.split()) <= 6:
        intent = "reference_to_prior_context"
        ambiguity = True
        clarification = True
        should_replace = False
        confidence = 0.74
    elif len(lower.split()) <= 2 and not re.search(r"[?.!]$", cleaned):
        intent = "unclear_fragment"
        ambiguity = True
        clarification = True
        should_replace = False
        confidence = 0.62

    context = TranscriptContext(
        original_text=original,
        cleaned_text=cleaned,
        should_replace_user_text=should_replace,
        llm_context_note=None,
        ambiguity_detected=ambiguity,
        clarification_suggested=clarification,
        detected_intent=intent,
        confidence=confidence,
        source="deterministic",
    )
    return replace(context, llm_context_note=build_llm_context_note(context))


def _extract_text_from_openrouter(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "".join(parts).strip()
    text = choices[0].get("text") if isinstance(choices[0], dict) else None
    if isinstance(text, str):
        return text.strip()
    raise ValueError("missing text")


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("json root was not object")
    return parsed


def _meaningful_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-zA-Z]{4,}", text.lower())}


def _context_from_llm_payload(payload: dict[str, Any], deterministic: TranscriptContext) -> TranscriptContext:
    cleaned = clean_transcript(str(payload.get("cleaned_text") or deterministic.cleaned_text))
    intent = payload.get("detected_intent")
    if intent is not None:
        intent = str(intent)
    if intent not in ALLOWED_INTENTS:
        raise ValueError("unsupported detected_intent")
    confidence = float(payload.get("confidence", 0.0))
    if confidence < 0.55:
        raise ValueError("low confidence")
    if not cleaned:
        raise ValueError("empty cleaned_text")

    should_replace = bool(payload.get("should_replace_user_text", False))
    protected_intents = {"numeric_fragment", "language_request", "date_time_question", "profanity_reaction", "frustration_fragment"}
    if deterministic.detected_intent in protected_intents and intent not in {deterministic.detected_intent, "unknown"}:
        raise ValueError("unsafe intent change")
    if should_replace:
        original_words = _meaningful_words(deterministic.original_text)
        cleaned_words = _meaningful_words(cleaned)
        added_words = cleaned_words - original_words
        if len(added_words) > 3:
            raise ValueError("cleaned_text appears to add unsupported meaning")

    note = payload.get("llm_context_note")
    if note is not None:
        note = clean_transcript(str(note))[:500] or None

    return TranscriptContext(
        original_text=deterministic.original_text,
        cleaned_text=cleaned,
        should_replace_user_text=should_replace,
        llm_context_note=note,
        ambiguity_detected=bool(payload.get("ambiguity_detected", deterministic.ambiguity_detected)),
        clarification_suggested=bool(payload.get("clarification_suggested", deterministic.clarification_suggested)),
        detected_intent=intent,
        confidence=max(0.0, min(confidence, 1.0)),
        source="llm",
    )


def _redacted_preview(text: str, limit: int = 160) -> str:
    return clean_transcript(text)[:limit]


def _build_interpreter_messages(
    deterministic: TranscriptContext,
    recent_turns: Sequence[str] | None = None,
    runtime_context: str | None = None,
) -> list[dict[str, str]]:
    recent = "\n".join(f"- {_redacted_preview(turn, 180)}" for turn in (recent_turns or [])[-5:]) or "none"
    runtime = _redacted_preview(runtime_context or "none", 240)
    user_content = (
        f"Current transcript: {deterministic.original_text[:1200]}\n"
        f"Cleaned deterministic transcript: {deterministic.cleaned_text[:1200]}\n"
        f"Deterministic detected intent: {deterministic.detected_intent}\n"
        f"Recent short turns:\n{recent}\n"
        f"Runtime context: {runtime}\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an invisible transcript context interpreter. You do not answer the user. "
                "You only produce structured JSON to help the main voice companion understand the user's utterance. "
                "Preserve the user's meaning. Do not hallucinate. Do not invent missing facts. "
                "Do not rewrite emotional or casual language into polished language. If the utterance is ambiguous, "
                "mark ambiguity and suggest clarification. Keep llm_context_note short and practical. Return JSON only."
            ),
        },
        {"role": "user", "content": user_content},
    ]


async def call_transcript_context_llm(
    deterministic: TranscriptContext,
    *,
    recent_turns: Sequence[str] | None = None,
    runtime_context: str | None = None,
) -> TranscriptContext:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing OPENROUTER_API_KEY")
    model = transcript_context_llm_model()
    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_interpreter_messages(deterministic, recent_turns, runtime_context),
        "temperature": 0,
        "max_tokens": 260,
        "response_format": {"type": "json_object"},
    }
    provider = _provider_payload()
    if provider:
        payload["provider"] = provider

    timeout = aiohttp.ClientTimeout(total=max(transcript_context_llm_timeout_ms() / 1000, 0.05))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "truthful-abundance/transcript-context",
            },
            json=payload,
        ) as response:
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"openrouter_status_{response.status}_body_length_{len(body)}")
            response_payload = await response.json()
    text = _extract_text_from_openrouter(response_payload)
    return _context_from_llm_payload(_json_from_text(text), deterministic)


def with_source(context: TranscriptContext, source: str) -> TranscriptContext:
    return replace(context, source=source)


async def interpret_transcript_context(
    text: str,
    *,
    recent_turns: Sequence[str] | None = None,
    runtime_context: str | None = None,
    llm_caller: Callable[[TranscriptContext], Awaitable[TranscriptContext]] | None = None,
) -> TranscriptContext:
    deterministic = detect_transcript_context(text)
    if not transcript_context_llm_enabled():
        return deterministic

    caller = llm_caller
    if caller is None:
        async def _default_caller(ctx: TranscriptContext) -> TranscriptContext:
            return await call_transcript_context_llm(ctx, recent_turns=recent_turns, runtime_context=runtime_context)
        caller = _default_caller

    task = asyncio.create_task(caller(deterministic))
    try:
        result = await asyncio.wait_for(task, timeout=transcript_context_llm_timeout_ms() / 1000)
        return result
    except TimeoutError:
        task.cancel()
        return with_source(deterministic, "deterministic_timeout_fallback")
    except Exception:
        return with_source(deterministic, "deterministic_llm_error_fallback")

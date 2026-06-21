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
    "calculation_request",
    "timer_request",
    "counting_request",
    "profanity_reaction",
    "reference_to_prior_context",
    "memory_recall_request",
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


def _int_env_ms(name: str, default: int, *, fallback_env: str | None = None) -> int:
    raw = os.getenv(name)
    if raw is None and fallback_env is not None:
        raw = os.getenv(fallback_env)
    if raw is None:
        return default
    try:
        return max(1, min(int(raw), 5000))
    except Exception:
        return default


def _timeout_with_source(candidates: list[tuple[str, str]], default_ms: int) -> tuple[int, str]:
    """Resolve a classifier timeout by precedence, reporting which knob won.

    candidates is an ordered list of (env_var_name, source_label); the first one
    that is set to a valid value wins. Falls back to (default_ms, "default").
    """
    for env_name, source in candidates:
        raw = os.getenv(env_name)
        if raw is None:
            continue
        try:
            return max(1, min(int(raw), 5000)), source
        except Exception:
            continue
    return default_ms, "default"


# Precedence for the normal-turn classifier timeout: the clear new var first,
# then the legacy var, then the code default.
def normal_context_classifier_timeout() -> tuple[int, str]:
    return _timeout_with_source(
        [
            ("NORMAL_CONTEXT_CLASSIFIER_MAX_WAIT_MS", "normal_env"),
            ("TRANSCRIPT_CONTEXT_LLM_TIMEOUT_MS", "legacy_env"),
        ],
        500,
    )


# Precedence for the tool-revalidation classifier timeout: the tool var first,
# then the normal var, then the legacy var, then the code default.
def tool_revalidation_context_classifier_timeout() -> tuple[int, str]:
    return _timeout_with_source(
        [
            ("TOOL_REVALIDATION_CONTEXT_CLASSIFIER_MAX_WAIT_MS", "tool_env"),
            ("NORMAL_CONTEXT_CLASSIFIER_MAX_WAIT_MS", "normal_env"),
            ("TRANSCRIPT_CONTEXT_LLM_TIMEOUT_MS", "legacy_env"),
        ],
        1000,
    )


def normal_context_classifier_max_wait_ms() -> int:
    """How long to wait for the classifier on a normal (low-risk) turn.

    Primary knob: NORMAL_CONTEXT_CLASSIFIER_MAX_WAIT_MS; legacy fallback:
    TRANSCRIPT_CONTEXT_LLM_TIMEOUT_MS; then the code default. A timeout here only
    loses optional enrichment, so the turn proceeds on the deterministic result.
    """
    return normal_context_classifier_timeout()[0]


def tool_revalidation_context_classifier_max_wait_ms() -> int:
    """How long to wait for the classifier on a high-risk tool/search handoff.

    Primary knob: TOOL_REVALIDATION_CONTEXT_CLASSIFIER_MAX_WAIT_MS; falls back to
    the normal var, then the legacy var, then the code default. The user is
    already waiting on a tool result, so we can afford a longer wait.
    """
    return tool_revalidation_context_classifier_timeout()[0]


def require_context_resolution_for_tool_authority() -> bool:
    """When true, a stale tool result may not regain authority on unresolved context."""
    return _env_bool("REQUIRE_CONTEXT_RESOLUTION_FOR_TOOL_AUTHORITY", True)


# Dependency ranking so a configurable minimum ("high"/"medium"/"low") can gate
# whether a newer utterance counts as additive enough to compose with a result.
DEPENDENCY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def tool_revalidation_additive_min_dependency() -> str:
    raw = (os.getenv("TOOL_REVALIDATION_ADDITIVE_MIN_DEPENDENCY", "high") or "high").strip().lower()
    return raw if raw in {"high", "medium", "low"} else "high"


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
        return "The user gave a numeric fragment. Follow the runtime capability contract: do not assume what it means or invent an operation. Ask a short clarification question."
    if intent == "language_request":
        return "The user is asking about language capability. Follow the runtime capability contract: Arche can speak human languages and is currently speaking English. If the requested language is ambiguous, clarify naturally; for Sri Lankan, ask whether they mean Sinhala or Tamil; for Jamaican, offer Jamaican Patois briefly."
    if intent == "voice_change_request":
        return "The user wants a different voice. Follow the runtime capability contract: Arche can change wording/language/style, but must not claim the actual TTS voice changed unless voice switching succeeded."
    if intent == "tool_request_email":
        return "The user may want something sent by email. Follow the runtime capability contract: do not claim an email was sent unless the email tool succeeds. Ask for missing recipient/confirmation."
    if intent == "tool_request_search":
        return "The user is asking for lookup/search. Follow the runtime capability contract: use the existing Exa search tool for current/external facts, avoid guessing, and clarify the search target if unclear."
    if intent == "tool_request_document":
        return "The user may want a document created. Follow the runtime capability contract: do not claim a document/file was created unless file creation exists in this runtime and succeeds. Offer to draft content if file creation is unavailable."
    if intent == "date_time_question":
        return "The user is asking for date or time. Follow the runtime capability contract: answer from runtime context/date-time guard only, do not search, and do not guess from model memory."
    if intent == "calculation_request":
        return "The user is asking for a calculation. Follow the runtime capability contract: answer directly if it is simple. If ambiguous, ask what operation they want. Do not pretend a calculator/tool was used unless one was."
    if intent == "timer_request":
        return "The user wants a timer or reminder. Follow the runtime capability contract: do not claim a timer/reminder was set unless a real timer/reminder tool succeeds. If unavailable, say you cannot set an actual timer from here yet."
    if intent == "counting_request":
        return "The user asked for counting. Follow the runtime capability contract: for short counts, count directly; for long counts, ask if they want the full count out loud."
    if intent == "frustration_fragment":
        return "The user sounds frustrated or fragmented. Respond calmly and ask one short grounding question instead of assuming the missing context."
    if intent == "profanity_reaction":
        return "The user made a profanity-heavy reaction. Treat it as emotional emphasis, not a literal request, unless more context is available."
    if intent == "choice_delegation":
        return "The user is delegating a choice. Offer a simple recommendation or ask for one constraint if the choice is unclear."
    if intent == "reference_to_prior_context":
        return "The user referred to prior context with an ambiguous word like 'that' or 'it'. Use recent context if clear; otherwise ask what they mean."
    if intent == "memory_recall_request":
        return "The user is asking you to recall something from earlier conversations. Use any provided long-term memory context naturally, the way a friend remembers. If no relevant memory is available, say you don't remember rather than inventing details."
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
    elif re.search(r"\b(what is|what's|calculate|add|plus|minus|subtract|times|multiply|multiplied by|divide|divided by|percent of|percentage of)\b", lower) and any(ch.isdigit() for ch in lower):
        intent = "calculation_request"
        ambiguity = False
        clarification = False
        should_replace = False
        confidence = 0.9
    elif re.fullmatch(r"[\d\s,.-]+(?:and\s+)?[\d\s,.-]+", lower) and any(ch.isdigit() for ch in lower):
        intent = "numeric_fragment"
        ambiguity = True
        clarification = True
        should_replace = False
        confidence = 0.88
    elif (
        "sri lankan" in lower
        or re.search(r"\b(speak|understand|talk in|use) (other )?(human )?(languages?|german|french|spanish|sinhala|tamil|english|jamaican|patois)\b", lower)
        or re.search(r"\bdo you speak (other )?(human )?languages?\b", lower)
    ):
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
    elif re.search(
        r"\b(do you remember|do you recall|you remember|remember when|remember that time|"
        r"what did i (say|tell|mention|call|name|ask|talk)|did i (ever )?(tell|mention|say) you|"
        r"what (was|were) (that|the) .* i (said|told|mentioned)|what do you (remember|know) about me|"
        r"what do you remember|last time we (talked|spoke)|earlier i (said|told|mentioned)|"
        r"i told you (about|that)|i (already )?told you|we (talked|spoke) about)\b",
        lower,
    ):
        intent = "memory_recall_request"
        ambiguity = False
        clarification = False
        should_replace = False
        confidence = 0.85
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
    elif re.search(r"\b(set|start) (a )?(timer|alarm)\b|\bremind me\b|\bset (a )?reminder\b|\bcalendar event\b", lower):
        intent = "timer_request"
        ambiguity = not any(ch.isdigit() for ch in lower)
        clarification = ambiguity
        should_replace = False
        confidence = 0.88
    elif re.search(r"\bcount (to|up to|down from)\b", lower):
        intent = "counting_request"
        ambiguity = False
        clarification = False
        should_replace = False
        confidence = 0.88
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
    timeout_ms: int | None = None,
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

    effective_timeout_ms = timeout_ms if timeout_ms is not None else transcript_context_llm_timeout_ms()
    timeout = aiohttp.ClientTimeout(total=max(effective_timeout_ms / 1000, 0.05))
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
    max_wait_ms: int | None = None,
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

    # Normal-turn path: NORMAL_CONTEXT_CLASSIFIER_MAX_WAIT_MS is the primary knob
    # (legacy TRANSCRIPT_CONTEXT_LLM_TIMEOUT_MS remains a backward-compatible
    # fallback inside normal_context_classifier_max_wait_ms()).
    wait_ms = max_wait_ms if max_wait_ms is not None else normal_context_classifier_max_wait_ms()
    task = asyncio.create_task(caller(deterministic))
    try:
        result = await asyncio.wait_for(task, timeout=max(wait_ms / 1000, 0.01))
        return result
    except TimeoutError:
        task.cancel()
        return with_source(deterministic, "deterministic_timeout_fallback")
    except Exception:
        return with_source(deterministic, "deterministic_llm_error_fallback")


# ---------------------------------------------------------------------------
# Context resolution: a timeout protects latency but must not grant authority.
# A handoff is "resolved" only when the classifier actually answered, or
# "deterministic_safe" when the deterministic evidence is clearly safe on its
# own. A high-risk handoff whose classifier timed out without clearly-safe
# deterministic evidence is "unresolved" and must fail safe.
# ---------------------------------------------------------------------------

CONTEXT_RESOLUTIONS = {"resolved", "unresolved", "deterministic_safe"}
CONTEXT_RESOLUTION_SOURCES = {"llm", "deterministic", "timeout"}

_DETERMINISTIC_UNSAFE_INTENTS = {None, "unknown", "unclear_fragment", "numeric_fragment"}


CONTEXT_CLASSIFIER_PATHS = {"normal_turn", "tool_revalidation"}


@dataclass(slots=True)
class ContextResolution:
    context: TranscriptContext
    resolution: str          # resolved | unresolved | deterministic_safe
    resolution_source: str   # llm | deterministic | timeout
    timed_out: bool
    classifier_path: str = "deterministic"  # llm | deterministic | timeout
    timeout_ms: int = 0      # the wait budget actually applied
    timeout_source: str = "default"  # normal_env | tool_env | legacy_env | default | explicit
    path: str = "normal_turn"  # normal_turn | tool_revalidation


def deterministic_is_clearly_safe(context: TranscriptContext) -> bool:
    """A deterministic classification is safe to act on without the LLM when it is
    confident and unambiguous and lands on a concrete intent."""
    return (
        context.confidence >= 0.8
        and not context.ambiguity_detected
        and context.detected_intent not in _DETERMINISTIC_UNSAFE_INTENTS
    )


async def resolve_transcript_context(
    text: str,
    *,
    recent_turns: Sequence[str] | None = None,
    runtime_context: str | None = None,
    path: str = "normal_turn",
    max_wait_ms: int | None = None,
    high_risk: bool = False,
    llm_caller: Callable[[TranscriptContext], Awaitable[TranscriptContext]] | None = None,
) -> ContextResolution:
    """Classify a turn and report whether the context was actually resolved.

    For a high-risk handoff, a classifier timeout yields `unresolved` unless the
    deterministic evidence is clearly safe; the caller is expected to fail safe.
    For a normal turn, a timeout degrades to the deterministic result and the
    turn proceeds (timeouts protect latency, they do not grant authority).

    The timeout is resolved by `path` precedence (normal vs tool revalidation) and
    its source is reported so logs show which env knob won. An explicit
    `max_wait_ms` overrides the precedence and is reported as source "explicit".
    """
    deterministic = detect_transcript_context(text)
    clearly_safe = deterministic_is_clearly_safe(deterministic)

    if max_wait_ms is not None:
        wait_ms, timeout_source = max_wait_ms, "explicit"
    elif path == "tool_revalidation":
        wait_ms, timeout_source = tool_revalidation_context_classifier_timeout()
    else:
        wait_ms, timeout_source = normal_context_classifier_timeout()

    def _mk(ctx_out: TranscriptContext, resolution: str, source: str, timed_out: bool, classifier_path: str) -> ContextResolution:
        return ContextResolution(
            ctx_out, resolution, source, timed_out, classifier_path, wait_ms, timeout_source, path
        )

    if not transcript_context_llm_enabled():
        if clearly_safe:
            return _mk(deterministic, "deterministic_safe", "deterministic", False, "deterministic")
        return _mk(
            deterministic,
            "unresolved" if high_risk else "deterministic_safe",
            "deterministic",
            False,
            "deterministic",
        )

    caller = llm_caller
    if caller is None:
        async def _default_caller(ctx: TranscriptContext) -> TranscriptContext:
            return await call_transcript_context_llm(ctx, recent_turns=recent_turns, runtime_context=runtime_context)
        caller = _default_caller

    task = asyncio.create_task(caller(deterministic))
    try:
        result = await asyncio.wait_for(task, timeout=max(wait_ms / 1000, 0.01))
        return _mk(result, "resolved", "llm", False, "llm")
    except TimeoutError:
        task.cancel()
        ctx = with_source(deterministic, "deterministic_timeout_fallback")
        if clearly_safe:
            return _mk(ctx, "deterministic_safe", "timeout", True, "timeout")
        return _mk(ctx, "unresolved", "timeout", True, "timeout")
    except Exception:
        ctx = with_source(deterministic, "deterministic_llm_error_fallback")
        if clearly_safe:
            return _mk(ctx, "deterministic_safe", "deterministic", False, "deterministic")
        return _mk(
            ctx,
            "unresolved" if high_risk else "deterministic_safe",
            "deterministic",
            False,
            "deterministic",
        )


# ---------------------------------------------------------------------------
# Tool-result composer: classify how a newer utterance during TOOL_CALL_PENDING
# relates to the in-flight query/result, then choose how to resume. The goal is
# not to cancel every interrupted lookup but to compose when safe, rerun when the
# query materially changed, and withhold when the old result is stale.
# ---------------------------------------------------------------------------

TOOL_REVALIDATION_CLASSES = (
    "additive_context",
    "constraint",
    "preference",
    "narrowing",
    "minor_correction",
    "major_correction",
    "pivot",
    "meta_complaint",
    "unrelated",
)

# Classes that augment the existing query rather than replace/abandon it.
ADDITIVE_FAMILY = {"additive_context", "constraint", "preference", "narrowing"}

TOOL_RESULT_RESUME_DECISIONS = {"compose", "rerun", "withhold", "discard", "defer", "clarify"}

_RELATIONSHIP_DEPENDENCY = {
    "additive_context": "high",
    "constraint": "high",
    "preference": "high",
    "narrowing": "high",
    "minor_correction": "medium",
    "major_correction": "low",
    "pivot": "low",
    "meta_complaint": "low",
    "unrelated": "low",
}

_CONSTRAINT_RE = re.compile(
    r"\b(only|just|make sure|as long as|but |without |with no |no more than|"
    r"under |over |less than |more than |before |after |by tomorrow|by tonight|"
    r"cheaper|cheapest|in (english|spanish|french|german|sinhala|tamil))\b"
)
_PREFERENCE_RE = re.compile(r"\b(i'?d (prefer|rather)|i prefer|rather|i'?d like .* instead|prefer the)\b")
_NARROWING_RE = re.compile(r"\b(specifically|more specifically|focus on|just the|narrow (it )?(down|to)|only the|the part about)\b")
_MAJOR_CORRECTION_RE = re.compile(
    r"\b(no,|that'?s not (it|what)|i didn'?t (say|mean)|i did not (say|mean)|not what i (said|meant)|"
    r"that'?s wrong|wrong (one|thing)|forget (that|it)|scratch that)\b"
)
_MINOR_CORRECTION_RE = re.compile(r"\b(actually|i mean|i meant|sorry,|well,|to be clear|rather than)\b")
_PIVOT_RE = re.compile(r"\b(stop|cancel|nevermind|never mind|forget it|different (question|topic)|change of topic|new question|instead, )\b")


def _overlap_ratio(a: str, b: str) -> float:
    sa, sb = _meaningful_words(a), _meaningful_words(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def classify_tool_revalidation_relationship(
    *,
    original_query: str,
    newer_utterance: str,
    base_intent: str | None = None,
    classification: str | None = None,
) -> tuple[str, str, float]:
    """Deterministically classify how a barge-in utterance relates to a pending
    tool/search query. Returns (relationship, dependency_level, confidence).

    Order matters: explicit meta-complaints, pivots, and corrections win over
    generic additive phrasing; only then do constraint/preference/narrowing cues
    and transcript overlap decide additive vs unrelated.
    """
    text = _normalize_for_match(newer_utterance)
    overlap = _overlap_ratio(original_query, newer_utterance)
    cls = (classification or "").strip().upper()

    if cls == "META_COMPLAINT" or re.search(r"\b(you missed|did you miss|you got it wrong|not listening)\b", text):
        relationship = "meta_complaint"
    elif _PIVOT_RE.search(text):
        relationship = "pivot"
    elif _MAJOR_CORRECTION_RE.search(text):
        relationship = "major_correction"
    elif _MINOR_CORRECTION_RE.search(text):
        relationship = "minor_correction"
    elif _NARROWING_RE.search(text):
        relationship = "narrowing"
    elif _PREFERENCE_RE.search(text):
        relationship = "preference"
    elif _CONSTRAINT_RE.search(text):
        relationship = "constraint"
    elif overlap >= 0.3:
        relationship = "additive_context"
    elif overlap <= 0.05 and len(text.split()) >= 3:
        relationship = "unrelated"
    else:
        relationship = "additive_context"

    dependency = _RELATIONSHIP_DEPENDENCY[relationship]
    # An additive_context inferred purely from overlap is "high" only when the
    # overlap is strong; a weak overlap is "medium" so the configurable minimum
    # dependency can keep weak additions conservative by default. Explicit
    # constraint/preference/narrowing cues stay "high" (a strong signal).
    if relationship == "additive_context":
        dependency = "high" if overlap >= 0.5 else "medium"
    # Confidence rises with how strongly the cue/overlap supports the class.
    if relationship in ADDITIVE_FAMILY:
        confidence = min(0.6 + overlap, 0.92)
    else:
        confidence = 0.8
    return relationship, dependency, round(confidence, 2)


def query_materially_changed(original_query: str, newer_utterance: str) -> bool:
    """A barge-in materially changes the query when it shares little with it."""
    return _overlap_ratio(original_query, newer_utterance) < 0.2


def decide_tool_result_resume(
    *,
    relationship: str,
    resolution: str,
    dependency_level: str,
    additive_min_dependency: str = "high",
    require_resolution: bool = True,
    materially_changed: bool = False,
    result_available: bool = True,
) -> tuple[str, bool]:
    """Choose how to resume a pending tool result after a barge-in.

    Returns (decision, additive_allowed). `additive_allowed` is True only when
    the in-flight result keeps its right to speak (composed). Fails safe when the
    context is unresolved and resolution is required for tool authority.
    """
    # Unresolved context must not grant authority to a stale result.
    if resolution == "unresolved" and require_resolution:
        # If the result has not even arrived yet, defer rather than discard;
        # otherwise withhold so it cannot auto-speak.
        return ("defer" if not result_available else "withhold"), False

    if relationship == "meta_complaint":
        return "withhold", False
    if relationship == "unrelated":
        return "discard", False
    if relationship == "pivot":
        return "discard", False
    if relationship == "major_correction":
        return "rerun", False
    if relationship == "minor_correction":
        if materially_changed:
            return "rerun", False
        return "compose", True
    if relationship in ADDITIVE_FAMILY:
        if DEPENDENCY_RANK.get(dependency_level, 0) >= DEPENDENCY_RANK.get(additive_min_dependency, 3):
            return "compose", True
        return "withhold", False
    return "withhold", False


# ---------------------------------------------------------------------------
# ContextDecision: a thin policy object built from existing context outputs.
# It does not classify from scratch or add a new model; it maps the signals the
# deterministic context layer + turn policy already produce into a single
# decision that governs prompt injection, response posture, and fallback.
# ---------------------------------------------------------------------------

CONTEXT_DECISION_INTENTS = {
    "reference_to_prior_context",
    "meta_complaint",
    "correction",
    "user_evaluating_assistant",
    "unknown",
}

RESPONSE_POSTURES = {
    "answer",
    "contextual_acknowledgment",
    "missed_context_recovery",
    "correction_received",
    "unclear_reference_clarification",
}

_HIGH_DEPENDENCY_INTENTS = {
    "reference_to_prior_context",
    "meta_complaint",
    "correction",
    "user_evaluating_assistant",
}

_POSTURE_BY_INTENT = {
    "reference_to_prior_context": "contextual_acknowledgment",
    "user_evaluating_assistant": "contextual_acknowledgment",
    "meta_complaint": "missed_context_recovery",
    "correction": "correction_received",
}

# Short/vague/evaluative utterances that depend on the prior 1-2 turns to resolve.
_CARRY_FORWARD_TRIGGERS = (
    "why",
    "do you know why",
    "see",
    "see what i mean",
    "right",
    "this time",
    "that time",
    "the last thing",
    "you got it",
    "you missed it",
)


@dataclass(slots=True)
class ContextDecision:
    final_intent: str
    context_dependency: str  # "high" | "none"
    response_posture: str
    force_context_injection: bool
    decision_source: str  # "deterministic" | "carry_forward" | "disabled"
    confidence: float
    ambiguity_detected: bool
    clarification_suggested: bool


def _normalize_for_match(text: str) -> str:
    return (text or "").strip().lower().replace("’", "'")


def is_carry_forward_trigger(text: str) -> bool:
    """True for short/vague/evaluative utterances that lean on the recent exchange."""
    normalized = _normalize_for_match(text).rstrip("?.! ")
    if not normalized or len(normalized.split()) > 6:
        return False
    return any(normalized == trigger or normalized.startswith(trigger + " ") for trigger in _CARRY_FORWARD_TRIGGERS)


def classify_context_intent(text: str, base_intent: str | None = None, classification: str | None = None) -> str:
    """Map existing signals + light phrase cues onto a ContextDecision intent.

    Order matters: a turn-policy META_COMPLAINT and explicit corrections win over
    generic evaluative phrasing.
    """
    normalized = _normalize_for_match(text)
    cls = (classification or "").strip().upper()
    base = (base_intent or "").strip().lower()
    if cls == "META_COMPLAINT":
        return "meta_complaint"
    if re.search(
        r"\bno,|\bthat'?s not\b|\bi didn'?t say\b|\bi did not say\b|\bnot what i (said|meant)\b|"
        r"\bthat'?s wrong\b|\bi meant\b|\bthe last thing\b|\bwrong\b",
        normalized,
    ):
        return "correction"
    if re.search(r"\b(you missed|did you miss|why did you miss|missed it|you got it wrong)\b", normalized):
        return "meta_complaint"
    if re.search(
        r"\b(you got it|you'?re getting it|you are getting it|exactly|correct|nailed it|you understand|"
        r"see what i mean|that'?s what i was testing)\b",
        normalized,
    ) or re.fullmatch(r"see\s*[?.!]*", normalized):
        return "user_evaluating_assistant"
    if base == "reference_to_prior_context":
        return "reference_to_prior_context"
    return "unknown"


def build_context_decision(
    *,
    text: str,
    base_intent: str | None = None,
    classification: str | None = None,
    ambiguity_detected: bool = False,
    clarification_suggested: bool = False,
    confidence: float = 0.0,
    prior_decision: "ContextDecision | None" = None,
) -> ContextDecision:
    """Build the per-turn context decision from already-computed context signals.

    A turn can be COMPLETE_THOUGHT and still depend on recent context: dependency
    is derived from intent/reference cues, never downgraded by the turn-shape
    classification.
    """
    intent = classify_context_intent(text, base_intent, classification)
    source = "deterministic"

    # Carry-forward: a vague/evaluative utterance inherits the prior contextual
    # intent when the previous turn was itself context-dependent.
    if (
        intent == "unknown"
        and prior_decision is not None
        and prior_decision.context_dependency == "high"
        and is_carry_forward_trigger(text)
    ):
        inherited = prior_decision.final_intent
        intent = inherited if inherited in _HIGH_DEPENDENCY_INTENTS else "reference_to_prior_context"
        source = "carry_forward"

    if intent in _HIGH_DEPENDENCY_INTENTS:
        dependency = "high"
        force = True
        posture = _POSTURE_BY_INTENT[intent]
    elif ambiguity_detected and clarification_suggested:
        # Do not answer as a generic standalone turn.
        dependency = "high"
        force = True
        posture = "unclear_reference_clarification"
    else:
        dependency = "none"
        force = False
        posture = "answer"

    return ContextDecision(
        final_intent=intent,
        context_dependency=dependency,
        response_posture=posture,
        force_context_injection=force,
        decision_source=source,
        confidence=float(confidence or 0.0),
        ambiguity_detected=bool(ambiguity_detected),
        clarification_suggested=bool(clarification_suggested),
    )


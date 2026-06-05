import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

EXA_SEARCH_URL = "https://api.exa.ai/search"
SEARCH_DISABLED_MESSAGE = "I couldn't reach search right now."


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published_date: str | None = None
    provider: str = "exa"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_clamped(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except Exception:
        return default
    return max(min_value, min(max_value, value))


def _env_float_clamped(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except Exception:
        return default
    return max(min_value, min(max_value, value))


def search_enabled() -> bool:
    return _env_bool("SEARCH_ENABLED", False)


def search_provider() -> str:
    return os.getenv("SEARCH_PROVIDER", "exa").strip().lower() or "exa"


def search_max_results() -> int:
    return _env_int_clamped("SEARCH_MAX_RESULTS", 5, 1, 10)


def search_timeout_seconds() -> float:
    return _env_float_clamped("SEARCH_TIMEOUT_SECONDS", 5.0, 0.5, 30.0)


def search_disabled_reason() -> str | None:
    if not search_enabled():
        return "search_disabled"
    provider = search_provider()
    if provider != "exa":
        return f"unsupported_provider:{provider}"
    if not os.getenv("EXA_API_KEY", "").strip():
        return "missing_exa_api_key"
    return None


def _clean_text(value: object, max_chars: int = 280) -> str:
    text = str(value or "")
    text = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    return text[:max_chars].strip()


def _result_snippet(item: dict[str, Any]) -> str:
    summary = item.get("summary")
    if isinstance(summary, str) and summary.strip():
        return _clean_text(summary)

    highlights = item.get("highlights")
    if isinstance(highlights, list):
        joined = " ".join(str(part) for part in highlights if part)
        if joined.strip():
            return _clean_text(joined)

    text = item.get("text")
    if isinstance(text, str) and text.strip():
        return _clean_text(text)

    context = item.get("context")
    if isinstance(context, str) and context.strip():
        return _clean_text(context)

    return ""


def normalize_exa_results(payload: dict[str, Any], max_results: int | None = None) -> list[SearchResult]:
    limit = min(max_results or search_max_results(), search_max_results())
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        logger.warning("search_error=malformed_exa_response reason=missing_results_list")
        return []

    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = _clean_text(item.get("url"), 500)
        if not url:
            continue
        title = _clean_text(item.get("title"), 160) or url
        published_date = item.get("publishedDate") or item.get("published_date")
        published_date_str = _clean_text(published_date, 80) if published_date else None
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=_result_snippet(item),
                published_date=published_date_str,
            )
        )
        if len(results) >= limit:
            break
    return results


async def internet_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the internet with the configured provider and never raise into the voice session."""
    started_at = time.monotonic()
    query = _clean_text(query, 500)
    try:
        requested_max = max(1, int(max_results or search_max_results()))
    except Exception:
        requested_max = search_max_results()
    limit = min(requested_max, search_max_results())
    provider = search_provider()

    disabled_reason = search_disabled_reason()
    if disabled_reason is not None:
        logger.warning(
            "search_tool_called search_provider=%s search_query=%s search_disabled_reason=%s search_result_count=%s search_latency_seconds=%.3f",
            provider,
            query,
            disabled_reason,
            0,
            time.monotonic() - started_at,
        )
        return []

    if not query:
        logger.warning(
            "search_tool_called search_provider=exa search_query=%s search_disabled_reason=%s search_result_count=%s search_latency_seconds=%.3f",
            query,
            "empty_query",
            0,
            time.monotonic() - started_at,
        )
        return []

    api_key = os.getenv("EXA_API_KEY", "").strip()
    timeout_seconds = search_timeout_seconds()
    payload = {
        "query": query,
        "numResults": limit,
        "type": "auto",
        "text": True,
    }
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "truthful-abundance/livekit-exa-search",
    }

    logger.info(
        "search_tool_called search_provider=exa search_query=%s search_pre_ack_spoken=%s max_results=%s timeout_seconds=%s",
        query,
        False,
        limit,
        timeout_seconds,
    )

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(EXA_SEARCH_URL, json=payload, headers=headers) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.warning(
                        "search_error=http_status search_provider=exa status=%s body_length=%s search_query=%s search_latency_seconds=%.3f",
                        response.status,
                        len(body),
                        query,
                        time.monotonic() - started_at,
                    )
                    return []
                response_payload = await response.json(content_type=None)
    except asyncio.TimeoutError:
        logger.warning(
            "search_error=timeout search_provider=exa search_query=%s search_latency_seconds=%.3f",
            query,
            time.monotonic() - started_at,
        )
        return []
    except Exception as exc:
        logger.warning(
            "search_error=%s search_provider=exa search_query=%s search_latency_seconds=%.3f",
            type(exc).__name__,
            query,
            time.monotonic() - started_at,
        )
        return []

    if not isinstance(response_payload, dict):
        logger.warning(
            "search_error=malformed_exa_response search_provider=exa search_query=%s search_latency_seconds=%.3f",
            query,
            time.monotonic() - started_at,
        )
        return []

    results = normalize_exa_results(response_payload, max_results=limit)
    logger.info(
        "search_provider=exa search_query=%s search_result_count=%s search_latency_seconds=%.3f search_result_handoff_spoken=%s",
        query,
        len(results),
        time.monotonic() - started_at,
        False,
    )
    return results


def format_search_results_for_voice(results: list[SearchResult]) -> str:
    if not results:
        return (
            f"{SEARCH_DISABLED_MESSAGE} "
            f"Say: {search_failure_response()}"
        )

    lines = [
        f"Search succeeded. Before summarizing aloud, say: {search_result_handoff()}",
        "Keep the spoken summary to one or two short sentences. Do not read URLs aloud unless asked.",
    ]
    for index, result in enumerate(results, start=1):
        parts = [f"{index}. {result.title}", f"URL: {result.url}"]
        if result.published_date:
            parts.append(f"Published: {result.published_date}")
        if result.snippet:
            parts.append(f"Snippet: {result.snippet}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


SEARCH_PRE_ACK_PHRASES = (
    "Yeah, I’ll look it up. Give me a second.",
    "Let me check that real quick.",
    "I don’t want to guess. I’ll look.",
    "Hold on, I’ll check online.",
    "I can check that.",
)
SEARCH_RESULT_HANDOFF_PHRASES = (
    "Okay, this is what I found.",
    "Alright, I found a few things.",
    "Here’s what came up.",
    "Looks like this is what’s going on.",
    "From what I found…",
)
SEARCH_FAILURE_SPOKEN_RESPONSE = "I couldn’t get a clean result. What exactly should I search for?"


def search_spoken_bridge() -> str:
    return SEARCH_PRE_ACK_PHRASES[1]


def search_result_handoff() -> str:
    return SEARCH_RESULT_HANDOFF_PHRASES[0]


def search_failure_response() -> str:
    return SEARCH_FAILURE_SPOKEN_RESPONSE


def search_intent_requires_external_info(user_text: str) -> bool:
    """Heuristic used for local validation of intent guidance; search remains tool-triggered by the LLM."""
    text = f" {user_text.strip().lower()} "
    if not text.strip():
        return False

    emotional_markers = (
        "i feel",
        "i'm feeling",
        "i am feeling",
        "i feel lonely",
        "i feel stuck",
        "i don't know",
        "i dont know",
        "what do you mean",
        "i'm just thinking",
        "i am just thinking",
    )
    if any(marker in text for marker in emotional_markers):
        return False

    current_markers = (
        "latest",
        "current",
        "currently",
        "right now",
        "recent",
        "changed recently",
        "still accurate",
        "still active",
        "still available",
        "pricing",
        "price",
        "schedule",
        "availability",
        "who runs",
        "who currently",
        "what changed",
        "people saying",
        "look up",
        "check",
        "search",
        "find me",
        "verify",
        "confirm",
        "documentation",
        "docs",
        "api support",
        "support streaming",
        "has an api",
        "have an api",
        "integrate",
        "in canada",
    )
    if any(marker in text for marker in current_markers):
        return True

    if " does " in text and any(noun in text for noun in (" api ", " platform ", " service ", " company ", " product ", " library ")):
        return True
    if " is " in text and any(noun in text for noun in (" company ", " service ", " platform ", " product ")):
        return True
    return False


SEARCH_TOOL_DESCRIPTION = """Search the web with Exa only when the user clearly needs current, external, specific, or verifiable information. Use intent, not exact phrases. Use this for current/recent facts; changed facts; people, companies, products, events, laws, prices, releases, schedules, availability, API/library/platform docs, verification, recommendations depending on what exists now, or anything the user asks to look up/check/find/research/confirm. Do not use for emotional support, casual conversation, personal brainstorming, or follow-ups answerable from the conversation. Before calling this tool in voice, say a short bridge like: 'Let me check that real quick.' After useful results, say: 'Okay, this is what I found.' If results are weak or unavailable, say so and ask one short clarifying question. Do not read long URLs aloud."""

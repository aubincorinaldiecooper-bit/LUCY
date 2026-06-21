import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TIMEZONE = "America/Toronto"


@dataclass(slots=True)
class RuntimeContext:
    session_timezone: str
    timezone_resolution_source: str
    client_timezone_present: bool
    client_timezone_value: str | None
    current_date: str
    current_time: str
    current_datetime_iso: str
    weekday: str
    human_readable_datetime: str
    system_message: str

    def to_search_context(self) -> dict[str, str]:
        return {
            "current_date": self.current_date,
            "current_datetime_iso": self.current_datetime_iso,
            "session_timezone": self.session_timezone,
        }


def _as_metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _valid_timezone(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    timezone = value.strip()
    if not timezone:
        return None
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return None
    except Exception:
        return None
    return timezone


def extract_client_timezone_from_metadata(*metadata_values: Any) -> str | None:
    for metadata in metadata_values:
        data = _as_metadata_dict(metadata)
        if "client_timezone" not in data:
            continue
        value = data.get("client_timezone")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_session_timezone(client_timezone: str | None = None, app_timezone: str | None = None) -> tuple[str, str]:
    valid_client_timezone = _valid_timezone(client_timezone)
    if valid_client_timezone:
        return valid_client_timezone, "client"

    valid_app_timezone = _valid_timezone(app_timezone if app_timezone is not None else os.getenv("APP_TIMEZONE"))
    if valid_app_timezone:
        return valid_app_timezone, "env"

    return DEFAULT_SESSION_TIMEZONE, "default"


def build_runtime_context(client_timezone: str | None = None, now: datetime | None = None) -> RuntimeContext:
    session_timezone, source = resolve_session_timezone(client_timezone)
    local_now = now.astimezone(ZoneInfo(session_timezone)) if now else datetime.now(ZoneInfo(session_timezone))
    current_date = local_now.strftime("%Y-%m-%d")
    current_time = local_now.strftime("%I:%M %p").lstrip("0")
    weekday = local_now.strftime("%A")
    human_readable_datetime = f"{local_now.strftime('%A, %B')} {local_now.day}, {local_now.year} at {current_time} {local_now.tzname()}"
    system_message = (
        "Runtime context:\n"
        "Runtime context overrides any model knowledge about the current date or time. "
        "For date/time questions, answer from this runtime context only.\n"
        f"Today is {weekday}, {local_now.strftime('%B')} {local_now.day}, {local_now.year}.\n"
        f"The current local time is {current_time} in {session_timezone}.\n"
        "Use this runtime context for questions about today’s date, current time, month, weekday, or whether it is morning/afternoon/evening.\n"
        "Do not use internet search for basic date/time questions.\n"
        "For current, external, or verifiable facts, use the internet_search tool instead of guessing.\n"
        "If search results are old, weak, or unclear, say that plainly and ask one short clarifying question.\n"
        "Do not say you do not know and then guess anyway."
    )
    return RuntimeContext(
        session_timezone=session_timezone,
        timezone_resolution_source=source,
        client_timezone_present=bool(client_timezone),
        client_timezone_value=client_timezone,
        current_date=current_date,
        current_time=current_time,
        current_datetime_iso=local_now.isoformat(),
        weekday=weekday,
        human_readable_datetime=human_readable_datetime,
        system_message=system_message,
    )


def runtime_context_from_metadata(*metadata_values: Any) -> RuntimeContext:
    client_timezone = extract_client_timezone_from_metadata(*metadata_values)
    return build_runtime_context(client_timezone)


def _ordinal_day(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def detect_datetime_intent(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower().replace("’", "'"))
    if not normalized:
        return None

    if re.search(r"\b(what|which)\s+(year)\b|\bcurrent\s+year\b", normalized):
        return "year"
    if re.search(r"\b(what|which)\s+(month)\b|\bcurrent\s+month\b", normalized):
        return "month"
    if re.search(r"\b(what|which)\s+(time)\b|\bcurrent\s+time\b|\btime\s+is\s+it\b", normalized):
        return "time"
    if re.search(r"\b(today'?s?\s+date|date\s+today|current\s+date|what'?s?\s+(today'?s?\s+)?date|what\s+date)\b", normalized):
        return "date"
    if re.search(r"\b(what|which)\s+day\b|\bday\s+of\s+the\s+week\b|\bweekday\b", normalized):
        return "weekday"
    if re.search(r"\b(is\s+it\s+)?(morning|afternoon|evening|night)\b", normalized) and re.search(r"\b(is it|right now|now|currently|there|here)\b", normalized):
        return "part_of_day"
    return None


def current_datetime_snapshot(
    runtime_context: RuntimeContext, now: datetime | None = None
) -> tuple[str, str]:
    """Fresh (current_date, current_time) in the session timezone, recomputed now.

    Never reuses runtime_context.current_* (which is frozen at session init). Pass
    `now` in tests to advance the clock deterministically.
    """
    tz = ZoneInfo(runtime_context.session_timezone)
    local_now = now.astimezone(tz) if now else datetime.now(tz)
    return local_now.strftime("%Y-%m-%d"), local_now.strftime("%I:%M %p").lstrip("0")


def answer_datetime_intent(
    runtime_context: RuntimeContext, intent: str, now: datetime | None = None
) -> str:
    # Recompute the current moment in the session timezone on every call. The
    # values stored on runtime_context are captured at session start and go stale
    # as the conversation continues, so a later "what time is it?" must not reuse
    # them. `now` is injectable for deterministic tests.
    tz = ZoneInfo(runtime_context.session_timezone)
    local_now = now.astimezone(tz) if now else datetime.now(tz)
    current_time = local_now.strftime("%I:%M %p").lstrip("0")
    weekday = local_now.strftime("%A")
    date_phrase = f"{weekday}, {local_now.strftime('%B')} {_ordinal_day(local_now.day)}, {local_now.year}"
    if intent == "time":
        return f"It’s {current_time} in {runtime_context.session_timezone}."
    if intent == "month":
        return f"It’s {local_now.strftime('%B')} in {runtime_context.session_timezone}."
    if intent == "year":
        return f"It’s {local_now.year}."
    if intent == "part_of_day":
        return f"It’s {_part_of_day(local_now.hour)} in {runtime_context.session_timezone}."
    return f"It’s {date_phrase}."

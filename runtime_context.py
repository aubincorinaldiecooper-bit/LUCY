import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
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

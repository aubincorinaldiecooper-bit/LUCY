import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).rstrip("/")
COMPANION_EMAIL_TIMEOUT_SECONDS = float(os.getenv("COMPANION_EMAIL_TIMEOUT_SECONDS", "30"))
COMPANION_EMAIL_MAX_BODY_CHARS = int(os.getenv("COMPANION_EMAIL_MAX_BODY_CHARS", "8000"))
COMPANION_EMAIL_MAX_TOKENS = int(os.getenv("COMPANION_EMAIL_MAX_TOKENS", "500"))


class CompanionEmailError(RuntimeError):
    """Raised when the companion email response cannot be generated."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise CompanionEmailError(f"Missing required environment variable: {name}")
    return value


def _openrouter_provider_payload() -> dict[str, Any] | None:
    provider_order_raw = (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").strip()
    provider_order = [p.strip() for p in provider_order_raw.split(",") if p.strip()]
    if not provider_order:
        return None

    allow_fallbacks = (os.getenv("OPENROUTER_ALLOW_FALLBACKS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    })
    return {"order": provider_order, "allow_fallbacks": allow_fallbacks}


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CompanionEmailError("OpenRouter response did not include choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise CompanionEmailError("OpenRouter response choice was malformed")

    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "".join(parts).strip()

    text = first_choice.get("text")
    if isinstance(text, str):
        return text.strip()

    raise CompanionEmailError("OpenRouter response did not include text content")


def generate_companion_email_response(companion_input: dict[str, Any]) -> str:
    """Generate an Arche companion reply for a normalized email input."""
    api_key = _required_env("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o").strip() or "openai/gpt-4o"
    body = str(companion_input.get("body") or "")[:COMPANION_EMAIL_MAX_BODY_CHARS]
    user_email = str(companion_input.get("userEmail") or "unknown")
    subject = str(companion_input.get("subject") or "(no subject)")

    user_content = (
        "Inbound email from an Arche user. Respond as the companion directly to the sender.\n"
        "Do not mention backend systems, webhooks, AgentMail, OpenRouter, "
        "or implementation details.\n"
        "Do not include markdown headings.\n\n"
        f"Sender: {user_email}\n"
        f"Subject: {subject}\n"
        f"Body:\n{body}"
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": COMPANION_EMAIL_MAX_TOKENS,
    }
    provider = _openrouter_provider_payload()
    if provider:
        payload["provider"] = provider

    request = Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "truthful-abundance/agentmail-companion",
        },
    )

    try:
        with urlopen(request, timeout=COMPANION_EMAIL_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise CompanionEmailError(
            f"OpenRouter request failed with status {exc.code}; "
            f"error_body_length={len(error_body)}"
        ) from exc
    except URLError as exc:
        raise CompanionEmailError(f"OpenRouter request failed: {exc.reason}") from exc

    try:
        response_payload = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise CompanionEmailError("OpenRouter returned invalid JSON") from exc

    response_text = _extract_response_text(response_payload)
    if not response_text:
        raise CompanionEmailError("OpenRouter returned an empty companion response")

    logger.info(
        "Companion email response generated model=%s response_length=%s",
        model,
        len(response_text),
    )
    return response_text

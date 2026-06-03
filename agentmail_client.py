import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

AGENTMAIL_BASE_URL = os.getenv(
    "AGENTMAIL_BASE_URL",
    "https://api.agentmail.to/v0",
).rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("AGENTMAIL_TIMEOUT_SECONDS", "10"))


class AgentMailError(RuntimeError):
    """Raised when AgentMail cannot complete an outbound API request."""


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise AgentMailError(f"Missing required environment variable: {name}")
    return value


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = _required_env("AGENTMAIL_API_KEY")
    url = f"{AGENTMAIL_BASE_URL}{path}"
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "truthful-abundance/agentmail",
        },
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
            if not response_body:
                return {}
            return json.loads(response_body)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise AgentMailError(
            f"AgentMail API request failed with status {exc.code}: {error_body[:500]}"
        ) from exc
    except URLError as exc:
        raise AgentMailError(f"AgentMail API request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise AgentMailError("AgentMail API returned invalid JSON") from exc


def send_email(
    *,
    to: str | list[str],
    subject: str,
    text: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Send a new email from the configured AgentMail inbox."""
    inbox_id = quote(_required_env("AGENTMAIL_INBOX_ID"), safe="")
    from_email = _required_env("AGENTMAIL_FROM_EMAIL")

    if not text and not html:
        raise AgentMailError("send_email requires text or html")

    payload: dict[str, Any] = {
        "to": to,
        "subject": subject,
        "reply_to": from_email,
    }
    if text:
        payload["text"] = text
    if html:
        payload["html"] = html

    return _post_json(f"/inboxes/{inbox_id}/messages/send", payload)


def reply_to_email(
    *,
    message_id: str,
    text: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Reply to an existing AgentMail message from the configured inbox."""
    inbox_id = quote(_required_env("AGENTMAIL_INBOX_ID"), safe="")
    quoted_message_id = quote(message_id, safe="")
    from_email = _required_env("AGENTMAIL_FROM_EMAIL")

    if not message_id:
        raise AgentMailError("reply_to_email requires message_id")
    if not text and not html:
        raise AgentMailError("reply_to_email requires text or html")

    payload: dict[str, Any] = {"reply_to": from_email}
    if text:
        payload["text"] = text
    if html:
        payload["html"] = html

    return _post_json(f"/inboxes/{inbox_id}/messages/{quoted_message_id}/reply", payload)


# Keep camelCase aliases because the integration task explicitly named this helper API.
def sendEmail(
    *,
    to: str | list[str],
    subject: str,
    text: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    return send_email(to=to, subject=subject, text=text, html=html)


def replyToEmail(
    *,
    messageId: str,
    text: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    return reply_to_email(message_id=messageId, text=text, html=html)

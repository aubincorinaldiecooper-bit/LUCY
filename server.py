import base64
import hashlib
import hmac
import json
import logging
import os
import time
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from livekit import api
from pydantic import BaseModel

from agentmail_client import AgentMailError, reply_to_email
from companion_email import CompanionEmailError, generate_companion_email_response

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Lucy LiveKit Session API")

default_origins = "http://localhost:3000,https://vigilant-youth-production-452c.up.railway.app"

allowed_origins = (
    os.getenv("CORS_ORIGINS")
    or os.getenv("ALLOWED_ORIGINS")
    or default_origins
)

origins = [
    origin.strip().rstrip("/")
    for origin in allowed_origins.split(",")

    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


SUPPORTED_AGENTMAIL_EVENTS = {
    "message.received",
    "message.sent",
    "message.delivered",
    "message.bounced",
    "message.rejected",
    "message.complained",
    "message.received.blocked",
    "message.received.spam",
    "message.received.unauthenticated",
}
SVIX_TOLERANCE_SECONDS = 5 * 60


def _get_header(request: Request, name: str) -> str | None:
    value = request.headers.get(name)
    return value.strip() if value else None


def _svix_header_values(request: Request) -> tuple[str | None, str | None, str | None]:
    return (
        _get_header(request, "svix-id"),
        _get_header(request, "svix-timestamp"),
        _get_header(request, "svix-signature"),
    )


def _decode_svix_secret(secret: str) -> bytes:
    encoded_secret = secret.split("_", 1)[1] if secret.startswith("whsec_") else secret
    padded_secret = encoded_secret + "=" * (-len(encoded_secret) % 4)
    try:
        return base64.urlsafe_b64decode(padded_secret)
    except Exception:
        return secret.encode("utf-8")


def _verify_agentmail_signature(request: Request, raw_body: bytes) -> bool:
    secret = os.getenv("AGENTMAIL_WEBHOOK_SECRET")
    svix_id, svix_timestamp, svix_signature = _svix_header_values(request)
    if not any((svix_id, svix_timestamp, svix_signature)):
        logger.warning(
            "AgentMail webhook received without Svix headers; "
            "skipping signature verification"
        )
        return True
    if not all((svix_id, svix_timestamp, svix_signature)):
        logger.warning("AgentMail webhook rejected: incomplete Svix headers")
        return False
    if not secret:
        logger.error(
            "AgentMail webhook Svix headers present but "
            "AGENTMAIL_WEBHOOK_SECRET is not configured"
        )
        return False

    try:
        timestamp = int(svix_timestamp)
    except ValueError:
        logger.warning("AgentMail webhook rejected: invalid Svix timestamp")
        return False

    if abs(time.time() - timestamp) > SVIX_TOLERANCE_SECONDS:
        logger.warning("AgentMail webhook rejected: Svix timestamp outside tolerance")
        return False

    signed_content = b".".join(
        [svix_id.encode("utf-8"), svix_timestamp.encode("utf-8"), raw_body]
    )
    expected_signature = base64.b64encode(
        hmac.new(_decode_svix_secret(secret), signed_content, hashlib.sha256).digest()
    ).decode("utf-8")

    for signature in svix_signature.split():
        try:
            version, signature_value = signature.split(",", 1)
        except ValueError:
            continue
        if version == "v1" and hmac.compare_digest(signature_value, expected_signature):
            return True

    logger.warning("AgentMail webhook rejected: signature mismatch")
    return False


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _email_address(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = parseaddr(value)[1] or value
    parsed = parsed.strip().lower()
    return parsed or None


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip_depth += 1
        elif tag.lower() in {"br", "p", "div", "li", "tr"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join("".join(self._parts).split())


def _strip_html(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return " ".join(value.split())
    return parser.text()


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_message_body(message: dict[str, Any]) -> str:
    plain_text = _first_string(
        message.get("text"),
        message.get("text_body"),
        message.get("body_text"),
        message.get("extracted_text"),
        message.get("plain"),
        message.get("plain_text"),
    )
    if plain_text:
        return plain_text

    html_body = _first_string(
        message.get("html"),
        message.get("html_body"),
        message.get("body_html"),
        message.get("extracted_html"),
    )
    if html_body:
        return _strip_html(html_body)

    return _first_string(message.get("body"), message.get("preview"))


def _normalize_companion_email_input(
    *,
    sender_email: str,
    subject: str | None,
    body: str,
    message_id: str,
    thread_id: str | None,
) -> dict[str, Any]:
    return {
        "channel": "email",
        "userEmail": sender_email,
        "subject": subject,
        "body": body,
        "messageId": message_id,
        "threadId": thread_id,
        "source": "agentmail",
    }


def _message_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("message", "send", "delivery", "bounce", "rejection", "complaint"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _extract_agentmail_log_fields(payload: dict[str, Any]) -> dict[str, Any]:
    message = _message_payload(payload)
    thread = payload.get("thread") if isinstance(payload.get("thread"), dict) else {}
    sender = message.get("from") or next(iter(_as_list(message.get("senders"))), None)
    recipient = (
        message.get("to") or message.get("recipients") or thread.get("recipients")
    )
    subject = message.get("subject") or thread.get("subject")

    return {
        "event_type": payload.get("event_type"),
        "message_id": message.get("message_id") or message.get("last_message_id"),
        "thread_id": message.get("thread_id") or thread.get("thread_id"),
        "sender": sender,
        "recipient": recipient,
        "subject": subject,
        "preview_present": bool(
            message.get("preview")
            or message.get("text")
            or message.get("html")
            or message.get("extracted_text")
            or message.get("extracted_html")
        ),
    }


def _log_agentmail_event(payload: dict[str, Any]) -> None:
    fields = _extract_agentmail_log_fields(payload)
    logger.info(
        "AgentMail webhook event_type=%s message_id=%s thread_id=%s "
        "sender=%s recipient=%s subject=%s preview_present=%s",
        fields["event_type"],
        fields["message_id"],
        fields["thread_id"],
        fields["sender"],
        fields["recipient"],
        fields["subject"],
        fields["preview_present"],
    )


def _reply_to_agentmail_message(companion_input: dict[str, Any]) -> None:
    message_id = str(companion_input.get("messageId") or "")
    logger.info(
        "AgentMail companion reply attempted message_id=%s thread_id=%s",
        message_id,
        companion_input.get("threadId"),
    )

    try:
        reply_text = generate_companion_email_response(companion_input)
    except CompanionEmailError:
        logger.exception("AgentMail companion response failed message_id=%s", message_id)
        return

    logger.info(
        "AgentMail reply attempted message_id=%s reply_text_length=%s",
        message_id,
        len(reply_text),
    )
    try:
        result = reply_to_email(message_id=message_id, text=reply_text)
    except AgentMailError:
        logger.exception("AgentMail reply failed message_id=%s", message_id)
        return
    logger.info(
        "AgentMail reply succeeded original_message_id=%s reply_message_id=%s thread_id=%s",
        message_id,
        result.get("message_id"),
        result.get("thread_id"),
    )


def _handle_message_received(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> None:
    message = payload.get("message")
    if not isinstance(message, dict):
        logger.warning("AgentMail message.received payload missing message object")
        return

    thread = payload.get("thread") if isinstance(payload.get("thread"), dict) else {}
    sender = _email_address(message.get("from")) or _email_address(
        next(iter(_as_list(message.get("senders"))), None)
    )
    from_email = _email_address(os.getenv("AGENTMAIL_FROM_EMAIL"))
    message_id = message.get("message_id") or message.get("last_message_id")
    thread_id = message.get("thread_id") or thread.get("thread_id")
    subject = message.get("subject") or thread.get("subject")
    body = _extract_message_body(message)
    logger.info(
        "AgentMail inbound sender=%s subject=%s message_id=%s thread_id=%s body_present=%s",
        sender,
        subject,
        message_id,
        thread_id,
        bool(body),
    )

    if not sender or not message_id:
        logger.warning("AgentMail inbound message missing sender or message_id")
        return
    if from_email and sender == from_email:
        logger.info(
            "AgentMail inbound ignored self-sent email sender=%s message_id=%s",
            sender,
            message_id,
        )
        return

    # Do not log or store private email bodies. Body is used only to generate
    # the companion response for this email.
    companion_input = _normalize_companion_email_input(
        sender_email=sender,
        subject=subject if isinstance(subject, str) else None,
        body=body,
        message_id=str(message_id),
        thread_id=str(thread_id) if thread_id else None,
    )
    background_tasks.add_task(_reply_to_agentmail_message, companion_input)


@app.post("/agentmail/webhook")
async def agentmail_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    logger.info("AgentMail webhook received")
    raw_body = await request.body()

    if not _verify_agentmail_signature(request, raw_body):
        return JSONResponse({"ok": False, "error": "invalid signature"}, status_code=401)

    try:
        payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except json.JSONDecodeError:
        logger.warning("AgentMail webhook malformed JSON payload")
        return JSONResponse({"ok": True})

    if not isinstance(payload, dict):
        logger.warning("AgentMail webhook payload was not an object")
        return JSONResponse({"ok": True})

    event_type = payload.get("event_type")
    if event_type not in SUPPORTED_AGENTMAIL_EVENTS:
        logger.info("AgentMail webhook unsupported event_type=%s", event_type)
        return JSONResponse({"ok": True})

    try:
        _log_agentmail_event(payload)
        if event_type == "message.received":
            _handle_message_received(payload, background_tasks)
    except Exception:
        logger.exception("AgentMail webhook handler failed event_type=%s", event_type)

    return JSONResponse({"ok": True})


class SessionRequest(BaseModel):
    model: str | None = None


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/livekit/session")
async def create_livekit_session(payload: SessionRequest):
    room_name = f"lucy-{uuid4().hex[:10]}"
    identity = f"web-{uuid4().hex[:8]}"

    lkapi = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    await lkapi.room.create_room(api.CreateRoomRequest(name=room_name, empty_timeout=600))

    grants = api.VideoGrants(room_join=True, room=room_name)
    metadata = "{}" if payload.model is None else f'{{"model":"{payload.model}"}}'
    token = (
        api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
        .with_identity(identity)
        .with_name("Lucy User")
        .with_grants(grants)
        .with_metadata(metadata)
        .to_jwt()
    )

    await lkapi.aclose()
    return {"room_url": os.getenv("LIVEKIT_URL"), "token": token}

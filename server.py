import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Any, AsyncIterable
from uuid import uuid4

import aiohttp
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from livekit import api
from pydantic import BaseModel

from agentmail_client import AgentMailError, reply_to_email, send_email
from companion_email import CompanionEmailError, generate_companion_email_response
from memory_layer import MemoryIdentity, MemoryLayer, memory_enabled

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

OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"
HUME_CLM_MAX_MESSAGES = 24


def _safe_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clm_authorize(request: Request) -> None:
    expected = (os.getenv("HUME_CLM_BEARER_TOKEN") or "").strip()
    if not expected:
        logger.error("hume_clm_request_rejected=true reason=missing_hume_clm_bearer_token")
        raise HTTPException(status_code=503, detail="HUME_CLM_BEARER_TOKEN is not configured")
    authorization = request.headers.get("authorization") or ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), expected):
        logger.warning("hume_clm_request_rejected=true reason=invalid_bearer_token")
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _extract_hume_message_content(message: dict[str, Any]) -> tuple[str, str] | None:
    nested = message.get("message") if isinstance(message.get("message"), dict) else message
    role = str(nested.get("role") or message.get("role") or "").strip().lower()
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        msg_type = str(message.get("type") or "").lower()
        if msg_type.startswith("user"):
            role = "user"
        elif msg_type.startswith("assistant"):
            role = "assistant"
    content = nested.get("content") if isinstance(nested, dict) else None
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        content = " ".join(text_parts)
    if not role or not isinstance(content, str) or not content.strip():
        return None
    return role, content.strip()


def _extract_hume_clm_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []
    messages: list[dict[str, str]] = []
    for raw in raw_messages[-HUME_CLM_MAX_MESSAGES:]:
        if not isinstance(raw, dict):
            continue
        extracted = _extract_hume_message_content(raw)
        if extracted is None:
            continue
        role, content = extracted
        if role == "developer":
            role = "system"
        messages.append({"role": role, "content": content})
    return messages


def _latest_user_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _openrouter_model_for_hume(payload: dict[str, Any]) -> str:
    configured = os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
    hint = str(payload.get("model") or payload.get("custom_language_model_id") or "").strip()
    if hint and _safe_bool_env("HUME_CLM_HONOR_MODEL_HINT", False):
        return hint
    return configured


async def _hume_clm_memory_note(custom_session_id: str | None, query: str) -> str | None:
    if not memory_enabled() or not custom_session_id or not query.strip():
        return None
    layer = MemoryLayer(MemoryIdentity(guest_id=custom_session_id), session_id=custom_session_id)
    try:
        memories = await layer.retrieve(query)
        return MemoryLayer.preload_note(memories)
    except Exception as exc:
        logger.warning("hume_clm_memory_retrieval_failed=true error_type=%s", type(exc).__name__)
        return None
    finally:
        await layer.aclose()


def _sse_error_chunk(request_id: str, message: str) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "error",
        "choices": [{"index": 0, "delta": {"content": message}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_openrouter_for_hume(payload: dict[str, Any], custom_session_id: str | None) -> AsyncIterable[str]:
    started_at = time.monotonic()
    request_id = f"hume-clm-{uuid.uuid4().hex[:12]}"
    messages = _extract_hume_clm_messages(payload)
    latest_user = _latest_user_message(messages)
    memory_note = await _hume_clm_memory_note(custom_session_id, latest_user)
    system_prompt = (os.getenv("SYSTEM_PROMPT") or "You are Arche, a concise, calm voice companion.").strip()
    upstream_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if memory_note:
        upstream_messages.append({"role": "system", "content": memory_note})
    upstream_messages.extend(messages)
    model = _openrouter_model_for_hume(payload)
    logger.info(
        "hume_clm_response_started=true request_id=%s model=%s message_count=%s memory_note_present=%s custom_session_id_present=%s",
        request_id,
        model,
        len(messages),
        bool(memory_note),
        bool(custom_session_id),
    )
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        logger.error("hume_clm_response_error=true request_id=%s reason=missing_openrouter_api_key", request_id)
        yield _sse_error_chunk(request_id, "I can't reach my language model configuration yet.")
        yield "data: [DONE]\n\n"
        return
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://lucy-production-c960.up.railway.app"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "LUCY Hume CLM"),
    }
    body = {"model": model, "messages": upstream_messages, "stream": True}
    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body) as response:
                if response.status >= 400:
                    error_preview = (await response.text())[:300]
                    logger.error("hume_clm_response_error=true request_id=%s status=%s error_preview=%s", request_id, response.status, error_preview)
                    yield _sse_error_chunk(request_id, "I hit a language-model error there.")
                    yield "data: [DONE]\n\n"
                    return
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.error("hume_clm_response_error=true request_id=%s error_type=%s", request_id, type(exc).__name__)
        yield _sse_error_chunk(request_id, "I hit a connection issue there.")
        yield "data: [DONE]\n\n"
        return
    logger.info(
        "hume_clm_response_completed=true request_id=%s duration_seconds=%.3f",
        request_id,
        time.monotonic() - started_at,
    )



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


@app.post("/hume/clm/chat/completions")
async def hume_clm_chat_completions(request: Request):
    _clm_authorize(request)
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object")
    custom_session_id = request.query_params.get("custom_session_id") or payload.get("custom_session_id")
    if not isinstance(custom_session_id, str):
        custom_session_id = None
    messages = _extract_hume_clm_messages(payload)
    logger.info(
        "hume_clm_request_received=true message_count=%s custom_session_id_present=%s model_hint_present=%s",
        len(messages),
        bool(custom_session_id),
        bool(payload.get("model") or payload.get("custom_language_model_id")),
    )
    return StreamingResponse(
        _stream_openrouter_for_hume(payload, custom_session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SessionRequest(BaseModel):
    model: str | None = None
    client_timezone: str | None = None
    # Verified user id, only trusted when the request carries the matching
    # SESSION_IDENTITY_SHARED_SECRET (i.e. it came from our own Next.js BFF route
    # after Better Auth validated the session). Never trust this from the browser.
    user_id: str | None = None


def _trusted_user_id_from_request(request: Request, payload_user_id: str | None) -> str | None:
    """Return payload.user_id only if the caller proved it's our trusted server.

    Fails closed: if the secret is unset or the header doesn't match, the id is
    ignored and the session is treated as anonymous (guest), exactly as before.
    """
    if not payload_user_id:
        return None
    expected = os.getenv("SESSION_IDENTITY_SHARED_SECRET", "")
    provided = request.headers.get("x-internal-auth", "")
    if expected and provided and hmac.compare_digest(provided, expected):
        return payload_user_id
    logger.warning(
        "LiveKit session ignored untrusted user_id: secret_configured=%s header_present=%s",
        bool(expected),
        bool(provided),
    )
    return None


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/livekit/session")
async def create_livekit_session(payload: SessionRequest, request: Request):
    room_name = f"lucy-{uuid4().hex[:10]}"
    identity = f"web-{uuid4().hex[:8]}"
    trusted_user_id = _trusted_user_id_from_request(request, payload.user_id)

    lkapi = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    metadata_payload = {
        key: value
        for key, value in {
            "model": payload.model,
            "client_timezone": payload.client_timezone,
            # The agent's memory_layer reads user_id to scope long-term memory to
            # a real signed-in user; absent -> anonymous/guest scope.
            "user_id": trusted_user_id,
        }.items()
        if value is not None
    }
    metadata = json.dumps(metadata_payload)
    metadata_keys = sorted(metadata_payload.keys())
    logger.info(
        "LiveKit session metadata prepared: client_timezone_present=%s client_timezone_value=%s metadata_payload_keys=%s room_metadata_includes_client_timezone=%s token_metadata_includes_client_timezone=%s session_user_id_attached=%s",
        bool(payload.client_timezone),
        payload.client_timezone or "none",
        metadata_keys,
        "client_timezone" in metadata_payload,
        "client_timezone" in metadata_payload,
        bool(trusted_user_id),
    )
    room_request = api.CreateRoomRequest(name=room_name, empty_timeout=600)
    room_request.metadata = metadata
    await lkapi.room.create_room(room_request)

    grants = api.VideoGrants(room_join=True, room=room_name)
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


class FeedbackRequest(BaseModel):
    email: str | None = None
    message: str | None = None


def _respond_to_feedback(companion_input: dict[str, Any]) -> None:
    """Generate Arche's reply to a user's feedback and email it back to them.

    Runs in the background so the request returns immediately, mirroring the
    inbound-email companion flow. The user's address is the reply target, which
    is why this is an autonomous loop (Arche writes back to the person).
    """
    to_email = str(companion_input.get("userEmail") or "").strip()
    if not to_email:
        return
    try:
        reply_text = generate_companion_email_response(companion_input)
    except CompanionEmailError:
        logger.exception("feedback companion response failed")
        return
    try:
        result = send_email(
            to=to_email,
            subject="Re: your note to Arche",
            text=reply_text,
        )
    except AgentMailError:
        logger.exception("feedback reply send failed")
        return
    logger.info(
        "feedback reply sent reply_message_id=%s reply_text_length=%s",
        result.get("message_id"),
        len(reply_text),
    )


@app.post("/api/feedback")
async def submit_feedback(
    payload: FeedbackRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    # Trust comes from the frontend BFF route, which verified the Better Auth
    # session before forwarding. Require the shared secret so the public endpoint
    # can't be driven directly.
    expected = os.getenv("SESSION_IDENTITY_SHARED_SECRET", "")
    provided = request.headers.get("x-internal-auth", "")
    if not (expected and provided and hmac.compare_digest(provided, expected)):
        logger.warning("feedback rejected: secret_configured=%s header_present=%s", bool(expected), bool(provided))
        raise HTTPException(status_code=401, detail="unauthorized")

    email = (payload.email or "").strip()
    message = (payload.message or "").strip()
    if not email or not message:
        raise HTTPException(status_code=400, detail="email and message are required")
    if len(message) > 5000:
        raise HTTPException(status_code=400, detail="message too long")

    companion_input = {
        "userEmail": email,
        "subject": "Your note to Arche",
        "body": message,
    }
    background_tasks.add_task(_respond_to_feedback, companion_input)
    logger.info("feedback accepted message_length=%s", len(message))
    return JSONResponse({"ok": True})

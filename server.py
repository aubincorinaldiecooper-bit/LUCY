import base64
import hashlib
import hmac
import json
import logging
import os
import time

import asyncio
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Any
from uuid import uuid4

import aiohttp
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from livekit import api
from pydantic import BaseModel

from agentmail_client import AgentMailError, reply_to_email, send_email
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

# --- MiniCPM-o live vision: startup probe + cached provider snapshot ----------
# Verifying that the deployed backend can actually read its vision env vars and
# reach the Modal worker is otherwise guesswork, so the probe runs once at
# startup and its result is cached here. /health then reports it without ever
# making a network call — the Railway healthcheck must stay instant, and Modal
# cold-starting from scale-to-zero can take minutes.

_vision_provider_snapshot: dict[str, Any] = {"state": "unknown", "checked": False}
_vision_probe_tasks: set[Any] = set()


def _vision_provider_public_snapshot() -> dict[str, Any]:
    """Non-secret provider view for /health.

    Deliberately booleans and state names only — never MINICPMO_WORKER_URL or
    MINICPMO_REALTIME_URL. /health is unauthenticated, and infrastructure
    endpoints are server-side configuration, not something to publish.
    """
    return dict(_vision_provider_snapshot)


async def _probe_vision_provider() -> None:
    """One-shot startup connectivity probe. Never raises, never blocks startup.

    Runs as a detached task so a slow or unreachable Modal endpoint delays
    nothing: not app startup, not the Railway healthcheck, not a voice session.
    """
    global _vision_provider_snapshot
    try:
        from minicpmo_provider import (
            GatewayState,
            _host_of,
            check_worker_health,
            gateway_reason,
            load_minicpmo_config,
        )

        config = load_minicpmo_config()
        gateway = config.gateway_state
        snapshot: dict[str, Any] = {
            "checked": True,
            "enabled": config.enabled,
            "gateway": gateway.value,
            "gateway_reason": gateway_reason(gateway),
            "worker_configured": bool(config.worker_url),
            "target_fps": config.target_fps,
            "max_queue_size": config.max_queue_size,
        }

        # Log the parsed config so the deployed service's own view of its env is
        # visible in Railway logs. Host only — never the full URL.
        worker_host = _host_of(config.worker_url)
        logger.info(
            "minicpmo_config_loaded=true enabled=%s worker_configured=%s worker_host=%s "
            "gateway=%s gateway_reason=%s connect_timeout_seconds=%s "
            "session_timeout_seconds=%s target_fps=%s max_queue_size=%s",
            config.enabled, bool(config.worker_url), worker_host, gateway.value,
            gateway_reason(gateway), config.connect_timeout_seconds,
            config.session_timeout_seconds, config.target_fps, config.max_queue_size,
        )

        if not config.enabled or not config.worker_url:
            snapshot["state"] = "disabled" if not config.enabled else "unavailable"
            snapshot["worker_reachable"] = None
            _vision_provider_snapshot = snapshot
            return

        result = await check_worker_health(config)
        snapshot["worker_reachable"] = result.ok
        # Worker reachable is NOT the same as live vision working: without the
        # realtime Gateway there is no video path at all. Say so explicitly so a
        # green worker probe can never be misread as a working integration.
        if result.ok and gateway is GatewayState.CONFIGURED:
            snapshot["state"] = "ready"
        elif result.ok:
            snapshot["state"] = "unavailable"
        else:
            snapshot["state"] = "degraded"
        _vision_provider_snapshot = snapshot

        logger.info(
            "minicpmo_worker_health_probe=true %s gateway=%s provider_state=%s "
            "realtime_available=%s",
            result.log_fields(), gateway.value, snapshot["state"],
            gateway is GatewayState.CONFIGURED and result.ok,
        )
        if not result.ok:
            logger.warning(
                "minicpmo_worker_unreachable=true reason=%s detail=%s "
                "(vision degrades to voice-only; conversation unaffected)",
                result.reason, result.detail,
            )
    except Exception as exc:  # noqa: BLE001 - a probe fault must never affect the app
        logger.warning(
            "minicpmo_probe_failed=true error_type=%s error=%s", type(exc).__name__, exc
        )
        _vision_provider_snapshot = {"state": "unknown", "checked": True, "error": type(exc).__name__}


@app.on_event("startup")
async def _startup_vision_probe() -> None:
    # Detached on purpose: awaiting this would put a Modal round trip in front
    # of the app becoming ready, and Railway's healthcheck would fail the deploy
    # on a cold GPU. Held in a module-level set because asyncio keeps only a
    # weak reference — an unreferenced task can be collected mid-probe.
    task = asyncio.create_task(_probe_vision_provider())
    _vision_probe_tasks.add(task)
    task.add_done_callback(_vision_probe_tasks.discard)

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "vision": _vision_provider_public_snapshot()})

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
    user_id: str | None = None

def _insert_feedback_row(user_id: str | None, email: str | None, message: str) -> str | None:
    """Persist feedback to Postgres and return its id. None on failure.

    This is the durable record the team reviews; capturing it is the priority,
    so it runs before (and independently of) the autonomous Arche reply.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("feedback_not_persisted reason=missing_DATABASE_URL")
        return None
    try:
        import psycopg

        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO feedback (user_id, email, message) VALUES (%s, %s, %s) RETURNING id",
                    (user_id, email, message),
                )
                row = cur.fetchone()
            conn.commit()
        feedback_id = str(row[0]) if row else None
        logger.info("feedback_persisted feedback_id=%s", feedback_id)
        return feedback_id
    except Exception:
        logger.exception("feedback_persist_failed")
        return None

def _update_feedback_reply(feedback_id: str, reply: str) -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url or not feedback_id:
        return
    try:
        import psycopg

        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE feedback SET reply = %s, replied_at = now() WHERE id = %s",
                    (reply, feedback_id),
                )
            conn.commit()
    except Exception:
        logger.exception("feedback_reply_update_failed")

def _reply_to_feedback(feedback_id: str | None, email: str, message: str) -> None:
    """Generate Arche's reply to feedback and email it back to the user.

    Runs in the background so the request returns immediately. Best-effort: a
    failure here never loses the already-persisted feedback row.
    """
    companion_input = {
        "userEmail": email,
        "subject": "Your note to Arche",
        "body": message,
    }
    try:
        reply_text = generate_companion_email_response(companion_input)
    except CompanionEmailError:
        logger.exception("feedback companion response failed feedback_id=%s", feedback_id)
        return
    try:
        result = send_email(to=email, subject="Re: your note to Arche", text=reply_text)
    except AgentMailError:
        logger.exception("feedback reply send failed feedback_id=%s", feedback_id)
        return
    logger.info(
        "feedback reply sent feedback_id=%s reply_message_id=%s reply_text_length=%s",
        feedback_id,
        result.get("message_id"),
        len(reply_text),
    )
    if feedback_id:
        _update_feedback_reply(feedback_id, reply_text)

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
    user_id = (payload.user_id or "").strip() or None
    if not email or not message:
        raise HTTPException(status_code=400, detail="email and message are required")
    if len(message) > 5000:
        raise HTTPException(status_code=400, detail="message too long")

    # Persist first (authoritative) — the durable record matters more than the
    # auto-reply. Fail the request if we couldn't store it so the user can retry.
    feedback_id = await asyncio.to_thread(_insert_feedback_row, user_id, email, message)
    if feedback_id is None:
        raise HTTPException(status_code=502, detail="could not store feedback")

    # Then reply as Arche in the background (best-effort).
    background_tasks.add_task(_reply_to_feedback, feedback_id, email, message)
    logger.info("feedback accepted feedback_id=%s message_length=%s", feedback_id, len(message))
    return JSONResponse({"ok": True})

# 128x128 PNG, large red circle on white — unmistakable content for the image
# smoke test: if Inworld's Realtime session accepts input_image content parts
# and routes them to a vision-capable LLM, the reply must mention red/circle.
_WS_SMOKE_TEST_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAAB3ElEQVR42u3dzVFDMQxF4URDH1nQf0ks"
    "UsmjhTA8Wz/+zpoZdHUsGRjAz+u6HsgjtIAAAkAAASCAABBAAAggAAQQAAIIAAEEgAACQAABIIAA3M9X"
    "r3J/Xq9PPuz7/e6S6Fn/F7M+bHpTGXUF/LPvXUyUE3B734ubKCRgQ+sLaighYHPrS2mIk7uf/tmTJyA9"
    "fIVRCN3PrSd0P7eq3SuoZusT11Hofm6dofu51Ybu59bsx9HJhOOfW3nofm79ofu5KdwBc++AGcd/dZbQ"
    "/dxEVtDEFTTv+K/LZQLGTcDU478onQmY+2UoEgTM3j8rMpoAK4gA+ycxqQmwggjABAHnXAD35jUBVhAB"
    "IIAAEEAACCAABBAAAgj4M43+Q0ypvCbACiIAQwSccw3cmNQEWEEE2EKJGU2AFUSALZSYzgRMXEFTh2BF"
    "LhMw9BKeNwSLEkW7iid13wqa/n3AjCFYmiJaV9+9+5tWUF8HGyp3B4y+A1oPwZ6aY1iedtXGyFSN6sx5"
    "P6DsX/TtPyJxSM6yVcVRaQvWk/+IT/o6yj0KcezRKzKI3hF7ELBbg5f0ckx4SzLHhNdUE2R4Txh9vgwl"
    "AAQQAAIIAAEEgAACQAABIIAAEEAACCAABBAAAgjACn4BmM29w/o7Xr8AAAAASUVORK5CYII="
)

_WS_SMOKE_IMAGE_PROMPT = "In one short sentence, say the shape and the color you see in the image."


@app.post("/api/inworld/ws-smoke-test")
async def inworld_ws_smoke_test(request: Request) -> JSONResponse:
    """Backend-only direct Inworld Realtime WebSocket audio smoke test.

    This does not use LiveKit or WebRTC. It answers whether Inworld emits usable
    Luna audio as ``response.output_audio.delta`` events for the bridge config.

    Body (optional JSON): ``{"mode": "image"}`` switches to the image smoke
    test — the prompt item carries an ``input_image`` content part (red circle
    on white) and the reply transcript shows whether the model actually saw it.
    ``{"model": "..."}`` overrides the LLM for the run.
    """
    result: dict[str, Any] = {
        "mode": "text",
        "model": None,
        "connected": False,
        "session_updated": False,
        "response_created": False,
        "event_counts": {},
        "audio_delta_count": 0,
        "audio_delta_total_chars": 0,
        "saw_audio_done": False,
        "saw_audio_transcript": False,
        "saw_text_done": False,
        "saw_response_done": False,
        "response_transcript": "",
        "errors": [],
        "first_events": [],
        "last_events": [],
    }

    if (os.getenv("INWORLD_WS_SMOKE_ENABLED") or "").strip().lower() != "true":
        return JSONResponse({"error": "ws_smoke_disabled"}, status_code=404)

    expected_token = (os.getenv("INWORLD_WS_SMOKE_TOKEN") or "").strip()
    auth_header = request.headers.get("authorization") or ""
    scheme, _, provided_token = auth_header.partition(" ")
    if not expected_token or scheme.lower() != "bearer" or provided_token.strip() != expected_token:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    from dataclasses import replace

    from inworld_realtime_bridge import (
        build_conversation_item_create,
        build_image_conversation_item_create,
        build_response_create,
        build_session_update,
        load_inworld_realtime_settings,
    )

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    image_mode = str(body.get("mode") or "").strip().lower() == "image"
    model_override = str(body.get("model") or "").strip()
    result["mode"] = "image" if image_mode else "text"

    def bool_str(value: bool) -> str:
        return str(value).lower()

    def record_event(payload: dict[str, Any]) -> str:
        event_type = str(payload.get("type") or "unknown")
        event_counts = result["event_counts"]
        event_counts[event_type] = int(event_counts.get(event_type, 0)) + 1
        event_summary = {"type": event_type, "keys": sorted(str(key) for key in payload.keys())}
        if "error" in payload:
            event_summary["error"] = payload.get("error")
        if len(result["first_events"]) < 20:
            result["first_events"].append(event_summary)
        result["last_events"].append(event_summary)
        result["last_events"] = result["last_events"][-20:]
        print(f"inworld_ws_smoke_event type={event_type}", flush=True)
        return event_type

    print("inworld_ws_smoke_started=true", flush=True)

    prompt_text = _WS_SMOKE_IMAGE_PROMPT if image_mode else "Say hello in one short sentence."

    try:
        settings = load_inworld_realtime_settings(instructions=prompt_text)
        # Image mode keeps the env-configured model (vision-capable
        # openai/gpt-4o-mini by default): the text mode's groq/gpt-oss-120b
        # override is a text-only model that can never see the image.
        smoke_model = model_override or ("" if image_mode else "groq/openai/gpt-oss-120b") or settings.model
        settings = replace(
            settings,
            model=smoke_model,
            stt_model="assemblyai/u3-rt-pro",
            tts_model="inworld-tts-2",
            voice="Luna",
            input_format="pcm16",
            output_format="pcm16",
            turn_detection_type="semantic_vad",
            tts_delivery_mode="CREATIVE",
            tts_segmenter_strategy="full_turn",
            tts_steering_handling="emit_once",
        )
        result["model"] = settings.model
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(repr(exc))
        print(f"inworld_ws_smoke_exception error={exc!r}", flush=True)
        return JSONResponse(result, status_code=500)

    session_update_sent = False
    prompt_sent = False
    response_create_sent = False
    transcript_deltas: list[str] = []
    # Vision inference is slower than the text hello: give image runs longer.
    deadline = time.monotonic() + (25.0 if image_mode else 15.0)

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30 if image_mode else 20)) as session:
            async with session.ws_connect(
                settings.connection_url,
                headers=settings.auth_headers,
                heartbeat=20,
            ) as ws:
                result["connected"] = True
                print("inworld_ws_smoke_connected=true", flush=True)

                while time.monotonic() < deadline:
                    timeout = max(0.1, deadline - time.monotonic())
                    try:
                        msg = await ws.receive(timeout=timeout)
                    except asyncio.TimeoutError:
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            payload = json.loads(msg.data)
                        except Exception as exc:  # noqa: BLE001
                            result["errors"].append(f"json_parse_error: {exc!r}")
                            continue

                        event_type = record_event(payload)

                        if event_type == "session.created" and not session_update_sent:
                            await ws.send_json(build_session_update(settings))
                            session_update_sent = True
                            print("inworld_ws_smoke_session_update_sent=true", flush=True)

                        if event_type == "session.updated":
                            result["session_updated"] = True
                            if not prompt_sent:
                                if image_mode:
                                    await ws.send_json(
                                        build_image_conversation_item_create(
                                            _WS_SMOKE_TEST_IMAGE_DATA_URL, prompt_text
                                        )
                                    )
                                else:
                                    await ws.send_json(build_conversation_item_create(prompt_text))
                                prompt_sent = True

                        if event_type == "conversation.item.done" and prompt_sent and not response_create_sent:
                            await ws.send_json(build_response_create(prompt_text))
                            response_create_sent = True
                            result["response_created"] = True
                            print("inworld_ws_smoke_response_create_sent=true", flush=True)

                        if event_type == "response.created":
                            result["response_created"] = True
                        elif event_type == "response.output_audio.delta":
                            delta = payload.get("delta")
                            chars = len(delta) if isinstance(delta, str) else 0
                            result["audio_delta_count"] += 1
                            result["audio_delta_total_chars"] += chars
                            print(f"inworld_ws_smoke_audio_delta chars={chars}", flush=True)
                        elif event_type == "response.output_audio.done":
                            result["saw_audio_done"] = True
                        elif event_type == "response.output_audio_transcript.delta":
                            result["saw_audio_transcript"] = True
                            delta = payload.get("delta")
                            if isinstance(delta, str):
                                transcript_deltas.append(delta)
                        elif event_type == "response.output_audio_transcript.done":
                            result["saw_audio_transcript"] = True
                            transcript = payload.get("transcript")
                            if isinstance(transcript, str) and transcript.strip():
                                # The done event carries the authoritative full
                                # transcript; prefer it over accumulated deltas.
                                transcript_deltas = [transcript]
                        elif event_type in {"response.output_text.done", "response.text.done"}:
                            result["saw_text_done"] = True
                            text = payload.get("text")
                            if isinstance(text, str) and text.strip() and not transcript_deltas:
                                transcript_deltas = [text]
                        elif event_type == "response.done":
                            result["saw_response_done"] = True
                            break
                        elif event_type == "error":
                            result["errors"].append(payload.get("error") or payload)

                    elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        result["errors"].append(repr(ws.exception()))
                        break
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(repr(exc))
        print(f"inworld_ws_smoke_exception error={exc!r}", flush=True)

    result["response_transcript"] = "".join(transcript_deltas).strip()

    print(
        "inworld_ws_smoke_summary "
        f"mode={result['mode']} "
        f"audio_delta_count={result['audio_delta_count']} "
        f"audio_delta_total_chars={result['audio_delta_total_chars']} "
        f"saw_audio_done={bool_str(bool(result['saw_audio_done']))} "
        f"saw_response_done={bool_str(bool(result['saw_response_done']))} "
        f"response_transcript={result['response_transcript']!r}",
        flush=True,
    )
    return JSONResponse(result)


# --- Public Arche API (speech + vision) ---------------------------------------
# Off by default (ARCHE_API_ENABLED). See arche_api.py for the protocol.


@app.websocket("/api/v1/realtime")
async def arche_realtime_api(websocket: WebSocket) -> None:
    from arche_api import handle_realtime_api_websocket

    await handle_realtime_api_websocket(websocket)


@app.post("/api/v1/vision/describe")
async def arche_vision_describe(request: Request) -> JSONResponse:
    """One-shot image description — the standalone visual API.

    Same auth as the realtime API. The model is the env-configured VISION_MODEL
    unless the request overrides it; the call goes through OpenRouter with the
    same cost bounds as in-session vision (one sentence, max_tokens capped).
    """
    from arche_api import api_enabled, api_key_from_scope, api_key_valid
    from vision_context import describe_frame, load_vision_context_config

    if not api_enabled():
        return JSONResponse({"error": "api_disabled"}, status_code=404)
    if not api_key_valid(api_key_from_scope(request.headers, request.query_params)):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    image = body.get("image")
    if not isinstance(image, str) or not image.startswith("data:image/"):
        return JSONResponse(
            {"error": "invalid_image", "detail": "expected a data:image/... data URL"},
            status_code=400,
        )

    from dataclasses import replace as dc_replace

    config = load_vision_context_config()
    model_override = str(body.get("model") or "").strip()
    config = dc_replace(
        config,
        enabled=True,  # endpoint access is the opt-in; VISION_CONTEXT_ENABLED gates in-session use
        model=model_override or config.model,
    )
    description = await describe_frame(image, config)
    if not description:
        return JSONResponse({"error": "vision_call_failed"}, status_code=502)
    return JSONResponse({"description": description, "model": config.model})

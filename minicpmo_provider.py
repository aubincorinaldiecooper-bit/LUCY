"""MiniCPM-o 4.5 live-vision provider: config, health, and realtime transport.

This is the only module that knows MiniCPM-o or Modal exist. Everything above it
(vision_session.py, inworld_realtime_bridge.py) talks to the small surface at the
bottom of this file, so swapping providers — or pointing at a self-hosted
gateway instead of Modal — is an env change, not a code change.

ARCHITECTURE (OpenBMB's intended realtime shape):

    Lucy backend  ->  realtime WebSocket Gateway  ->  MiniCPM-o worker  ->  GPU

Two DIFFERENT endpoints, and we currently have only the second one:

  MINICPMO_WORKER_URL    https://...modal.run  — the deployed Modal worker.
                         Reachable today. Used for health/connectivity only.

  MINICPMO_REALTIME_URL  wss://.../v1/realtime?mode=video — the browser-facing
                         realtime Gateway. NOT DEPLOYED YET, so this is blank.

  MINICPMO_GATEWAY_TOKEN Bearer credential for the Gateway leg. Blank today.
                         Should be set when the Gateway is deployed: every
                         connect wakes an L40S, so an unauthenticated Gateway
                         is a GPU-cost DoS.

The worker is NOT a realtime endpoint and must never be used as one. Pointing
MINICPMO_REALTIME_URL at the worker would produce a provider that looks
configured, fails at connect time on every session, and buries the real cause
(no Gateway) under a generic socket error. ``load_minicpmo_config`` therefore
*rejects* a realtime URL whose host matches the worker's and reports
``realtime_url_is_worker_url`` — see resolve_gateway_state().

Until the Gateway is deployed the provider sits in a well-defined pending state:
health checks against the worker still run and are reported, session creation
refuses immediately with reason ``realtime_gateway_not_configured``, and — the
part that matters — Arche's voice conversation is completely unaffected.

FAILURE POLICY: conservative. Every Modal connect spins up an L40S, so this
never retries in a tight loop. One session gets at most
``MINICPMO_MAX_CONNECT_ATTEMPTS`` (default 2) attempts with a fixed backoff;
after that the session is degraded for good and stops touching the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from functools import cached_property
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

import aiohttp

logger = logging.getLogger(__name__)

# Cold-start on an L40S pulling MiniCPM-o 4.5 into GPU memory is minutes, not
# seconds, when Modal has scaled to zero (scaledown_window=300s). A connect
# slower than this is treated as unavailable rather than hung.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 120.0
DEFAULT_SESSION_TIMEOUT_SECONDS = 900.0
DEFAULT_TARGET_FPS = 2.0
DEFAULT_MAX_QUEUE_SIZE = 4
# Ceilings, not defaults. Each accepted frame costs a synchronous PIL encode on
# the event loop that also forwards microphone audio, and each queued frame is
# a retained JPEG. Both are clamped so a typo in an env var cannot turn a
# vision knob into a voice-latency or memory problem.
MAX_TARGET_FPS = 10.0
MAX_QUEUE_SIZE = 32
# Largest frame accepted into the queue (~1.3 MB of base64). Comfortably above
# a 512px JPEG data URL (~50 KB); well below what would let a client pin memory.
MAX_FRAME_CHARS = 1_400_000
# Health probes are a liveness signal, not a session — they must never sit
# behind a cold start, so they get their own short timeout.
DEFAULT_HEALTH_TIMEOUT_SECONDS = 10.0
# Above this, a connect is reported as cold=True: a warm Modal container answers
# in well under a second, a cold one has to pull weights onto the GPU.
COLD_START_THRESHOLD_MS = 5_000.0


class VisionProviderState(str, Enum):
    """Lifecycle of the vision provider for one Lucy session.

    ``degraded`` and ``unavailable`` both mean "no visual context right now" and
    both leave the voice conversation untouched. They are kept apart because
    they need different operator responses: degraded = it worked and then broke
    (retryable, look at logs), unavailable = it never could work (config or
    infrastructure, look at env).
    """

    DISABLED = "disabled"
    CONNECTING = "connecting"
    COLD_START = "cold_start"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class GatewayState(str, Enum):
    """Whether the realtime Gateway is usable, and if not, precisely why."""

    CONFIGURED = "configured"
    # MINICPMO_REALTIME_URL is blank: the official OpenBMB Gateway has not been
    # deployed yet. This is the expected state today.
    PENDING_DEPLOYMENT = "pending_deployment"
    # MINICPMO_REALTIME_URL points at the Modal worker. A misconfiguration we
    # refuse rather than silently attempt.
    INVALID_WORKER_URL = "invalid_worker_url"
    # Set to something that is not a ws:// or wss:// URL at all.
    INVALID_SCHEME = "invalid_scheme"


def _env_bool(name: str, default: bool = False) -> bool:
    """Match inworld_realtime_bridge._env_bool / vision_context semantics exactly."""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "minicpmo_config_invalid_float=true var=%s raw=%s fallback=%s", name, raw, default
        )
        return default
    # A non-positive timeout/FPS would mean "never" or a division by zero
    # downstream; treat it as a typo and keep the default rather than shipping a
    # provider that silently never runs.
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "minicpmo_config_invalid_int=true var=%s raw=%s fallback=%s", name, raw, default
        )
        return default
    return value if value > 0 else default


def _host_of(url: str) -> str:
    """Normalized hostname for comparison, or "" if unparseable.

    The trailing dot matters: "host.modal.run." is the same host to DNS but a
    different string, so without stripping it the worker-URL guard below can be
    walked straight past.
    """
    try:
        host = (urlsplit(url).hostname or "").strip().lower()
    except ValueError:
        return ""
    return host.rstrip(".")


@dataclass(frozen=True)
class MiniCPMOConfig:
    """Immutable per-process view of the MiniCPM-o env configuration.

    Frozen and loaded once per session (mirroring VisionContextConfig /
    MoodContextConfig), so a config read can't drift mid-conversation.
    """

    enabled: bool
    worker_url: str
    realtime_url: str
    # Bearer credential for the Gateway leg. Blank today because the Gateway is
    # not deployed; it matters as soon as it is, because every connect wakes an
    # L40S — an unauthenticated Gateway is a GPU-cost DoS, not just a data
    # exposure. Server-side only; never reaches a browser.
    gateway_token: str
    connect_timeout_seconds: float
    session_timeout_seconds: float
    health_timeout_seconds: float
    target_fps: float
    max_queue_size: int
    max_connect_attempts: int
    # Total Gateway connects allowed per CONVERSATION, across every camera
    # on/off cycle. max_connect_attempts bounds retries within one start;
    # this bounds how often a whole conversation can wake a GPU.
    max_total_connects: int
    reconnect_backoff_seconds: float
    min_state_update_interval_seconds: float

    @cached_property
    def frame_interval_seconds(self) -> float:
        """Minimum wall-clock gap between frames handed to the provider.

        cached: read on every offered frame and every sampler event, on the
        event loop shared with audio forwarding, yet constant for a frozen
        config.
        """
        return 1.0 / self.target_fps if self.target_fps > 0 else 0.5

    @cached_property
    def gateway_state(self) -> GatewayState:
        """Classification of MINICPMO_REALTIME_URL — see resolve_gateway_state.

        cached for the same reason as frame_interval_seconds: wants_frames
        reads it per frame, and re-running urlsplit + host normalization per
        camera frame is pure waste on a frozen config.
        """
        return resolve_gateway_state(self.realtime_url, self.worker_url)

    @property
    def realtime_ready(self) -> bool:
        """True only when a real Gateway URL is configured and usable."""
        return self.gateway_state is GatewayState.CONFIGURED


def resolve_gateway_state(realtime_url: str, worker_url: str) -> GatewayState:
    """Classify MINICPMO_REALTIME_URL, refusing the worker URL outright.

    Split out from config loading so the "is the Gateway real?" rule is one
    testable function rather than a condition duplicated at every call site.
    """
    realtime_url = (realtime_url or "").strip()
    if not realtime_url:
        return GatewayState.PENDING_DEPLOYMENT

    scheme = ""
    try:
        scheme = (urlsplit(realtime_url).scheme or "").lower()
    except ValueError:
        return GatewayState.INVALID_SCHEME
    if scheme not in {"ws", "wss"}:
        return GatewayState.INVALID_SCHEME

    # The load-bearing guard. The Modal worker answers on :22400 behind an HTTPS
    # endpoint and speaks the worker protocol, not the realtime Gateway
    # protocol. Someone copying MINICPMO_WORKER_URL into MINICPMO_REALTIME_URL
    # (and swapping https->wss) is the single most likely misconfiguration here,
    # so compare hosts, not raw strings.
    worker_host = _host_of(worker_url)
    realtime_host = _host_of(realtime_url)
    if worker_host and realtime_host == worker_host:
        return GatewayState.INVALID_WORKER_URL
    # Belt and braces: even with MINICPMO_WORKER_URL unset, a Modal *-worker
    # endpoint is never the realtime Gateway. Without this the guard would
    # quietly switch itself off exactly when the config is most confused.
    if realtime_host.endswith(".modal.run") and "-worker" in realtime_host:
        return GatewayState.INVALID_WORKER_URL

    return GatewayState.CONFIGURED


def load_minicpmo_config() -> MiniCPMOConfig:
    """Read MiniCPM-o settings from the environment.

    Never raises: a malformed number falls back to its default with a warning,
    because a bad vision env var must not stop the process that also serves the
    voice conversation from starting.
    """
    return MiniCPMOConfig(
        enabled=_env_bool("MINICPMO_ENABLED", False),
        worker_url=(os.getenv("MINICPMO_WORKER_URL") or "").strip().rstrip("/"),
        # Intentionally blank until the official OpenBMB realtime Gateway is
        # deployed. Never defaulted to the worker URL — see module docstring.
        realtime_url=(os.getenv("MINICPMO_REALTIME_URL") or "").strip(),
        gateway_token=(os.getenv("MINICPMO_GATEWAY_TOKEN") or "").strip(),
        connect_timeout_seconds=_env_float(
            "MINICPMO_CONNECT_TIMEOUT_SECONDS", DEFAULT_CONNECT_TIMEOUT_SECONDS
        ),
        session_timeout_seconds=_env_float(
            "MINICPMO_SESSION_TIMEOUT_SECONDS", DEFAULT_SESSION_TIMEOUT_SECONDS
        ),
        health_timeout_seconds=_env_float(
            "MINICPMO_HEALTH_TIMEOUT_SECONDS", DEFAULT_HEALTH_TIMEOUT_SECONDS
        ),
        target_fps=min(_env_float("VISION_TARGET_FPS", DEFAULT_TARGET_FPS), MAX_TARGET_FPS),
        max_queue_size=min(
            _env_int("VISION_MAX_QUEUE_SIZE", DEFAULT_MAX_QUEUE_SIZE), MAX_QUEUE_SIZE
        ),
        # Conservative by default: one retry, then stop. Each attempt can wake
        # an L40S.
        max_connect_attempts=_env_int("MINICPMO_MAX_CONNECT_ATTEMPTS", 2),
        max_total_connects=_env_int("MINICPMO_MAX_SESSION_CONNECTS", 6),
        reconnect_backoff_seconds=_env_float("MINICPMO_RECONNECT_BACKOFF_SECONDS", 5.0),
        min_state_update_interval_seconds=_env_float(
            "VISION_STATE_MIN_UPDATE_INTERVAL_SECONDS", 4.0
        ),
    )


@dataclass(frozen=True)
class HealthResult:
    """Outcome of one worker connectivity probe."""

    ok: bool
    status: int | None
    latency_ms: float
    reason: str
    detail: str = ""

    def log_fields(self) -> str:
        return (
            f"ok={self.ok} status={self.status} latency_ms={self.latency_ms:.1f} "
            f"reason={self.reason}"
        )


async def check_worker_health(config: MiniCPMOConfig) -> HealthResult:
    """Probe MINICPMO_WORKER_URL for reachability. Never raises.

    Deliberately permissive about the response: this validates that Lucy's
    network can reach the Modal endpoint and that Modal is answering for this
    app — it is NOT a claim that realtime video works. Any HTTP response below
    500 counts as reachable, including 404/405, because the worker's root path
    is not guaranteed to be a health route and a 404 from Modal still proves the
    container is up and routing.

    A short-lived aiohttp session per probe is deliberate: this runs once per
    process at startup, and tests fake it by patching this module's ``aiohttp``
    binding.
    """
    if not config.worker_url:
        return HealthResult(
            ok=False, status=None, latency_ms=0.0, reason="worker_url_not_configured"
        )

    started = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(total=config.health_timeout_seconds)
        session = aiohttp.ClientSession(timeout=timeout)
        try:
            async with session.get(config.worker_url) as response:
                status = getattr(response, "status", None)
                latency_ms = (time.monotonic() - started) * 1000.0
                if isinstance(status, int) and status >= 500:
                    detail = ""
                    try:
                        detail = (await response.text())[:200]
                    except Exception:  # noqa: BLE001 - body is a nicety, not required
                        pass
                    return HealthResult(
                        ok=False,
                        status=status,
                        latency_ms=latency_ms,
                        reason="worker_error_status",
                        detail=detail,
                    )
                return HealthResult(
                    ok=True, status=status, latency_ms=latency_ms, reason="reachable"
                )
        finally:
            await session.close()
    except asyncio.TimeoutError:
        return HealthResult(
            ok=False,
            status=None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            reason="timeout",
        )
    except asyncio.CancelledError:
        # Shutdown, not a health failure — never swallow this.
        raise
    except Exception as exc:  # noqa: BLE001 - health must never raise into a session
        return HealthResult(
            ok=False,
            status=None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            reason="unreachable",
            detail=f"{type(exc).__name__}: {exc}"[:200],
        )


_URL_IN_TEXT = re.compile(r"\b(?:wss?|https?)://\S+", re.IGNORECASE)


def _redact_urls(text: str) -> str:
    """Replace any URL with its host.

    aiohttp's handshake/connection errors stringify the full request URL. This
    module promises host-only logging (and a Gateway URL may one day carry a
    query credential), so the detail we surface must not smuggle one through.
    """
    def _host_only(match: "re.Match[str]") -> str:
        host = _host_of(match.group(0))
        return f"<{host or 'url'}>"

    return _URL_IN_TEXT.sub(_host_only, text)


class MiniCPMOConnectionError(Exception):
    """Realtime connection could not be established. Carries a stable reason code.

    ``config_error`` marks refusals caused by configuration (a missing or
    invalid Gateway URL) rather than the network. The session layer uses it to
    classify UNAVAILABLE ("never could work — check env") vs DEGRADED ("worked
    then broke — check logs") without re-encoding this module's reason strings;
    a new Gateway state added here is classified correctly there for free.
    """

    def __init__(self, reason: str, detail: str = "", *, config_error: bool = False) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.config_error = config_error


class MiniCPMORealtimeConnection:
    """Thin lifecycle wrapper around the Gateway WebSocket.

    Owns only the socket: framing in, messages out, clean close. Queueing,
    backpressure, rolling state, telemetry and degradation all live in
    vision_session.py, so this stays small enough to swap wholesale when the
    real Gateway protocol is pinned down.

    The frame/text envelopes below follow OpenBMB's realtime demo shape. They
    are the one part of this integration that CANNOT be verified until the
    Gateway is actually deployed — see MINICPMO_REALTIME_URL in the module
    docstring — and are isolated here for exactly that reason.
    """

    def auth_headers(self) -> dict[str, str]:
        """Credential for the Gateway leg, or {} when none is configured.

        Returned as headers rather than a query param on purpose: a token in a
        URL lands in proxy and access logs. Never logged, never included in a
        MiniCPMOConnectionError detail.
        """
        if not self.config.gateway_token:
            return {}
        return {"Authorization": f"Bearer {self.config.gateway_token}"}

    def __init__(self, config: MiniCPMOConfig, *, session_id: str) -> None:
        self.config = config
        self.session_id = session_id
        self._session: aiohttp.ClientSession | None = None
        self._ws: Any | None = None
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._closed

    async def connect(self) -> None:
        """Open the Gateway socket. Raises MiniCPMOConnectionError on any failure.

        Refuses before touching the network when the Gateway is not properly
        configured, so a pending deployment costs nothing and — importantly —
        never wakes a Modal GPU.
        """
        gateway_state = self.config.gateway_state
        if gateway_state is not GatewayState.CONFIGURED:
            raise MiniCPMOConnectionError(
                _GATEWAY_REASONS[gateway_state],
                "MINICPMO_REALTIME_URL must point at the deployed OpenBMB realtime "
                "Gateway (wss://.../v1/realtime?mode=video), never at MINICPMO_WORKER_URL",
                config_error=True,
            )

        started = time.monotonic()
        try:
            timeout = aiohttp.ClientTimeout(
                total=None, sock_connect=self.config.connect_timeout_seconds, sock_read=None
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self.config.realtime_url,
                    headers=self.auth_headers(),
                    heartbeat=20,
                ),
                timeout=self.config.connect_timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._cleanup()
            raise MiniCPMOConnectionError(
                "connect_timeout",
                f"no Gateway response within {self.config.connect_timeout_seconds:.0f}s "
                f"(elapsed={time.monotonic() - started:.1f}s; Modal cold start?)",
            ) from None
        except asyncio.CancelledError:
            await self._cleanup()
            raise
        except Exception as exc:  # noqa: BLE001 - any transport failure is a connect failure
            await self._cleanup()
            raise MiniCPMOConnectionError(
                "connect_failed", _redact_urls(f"{type(exc).__name__}: {exc}")[:200]
            ) from exc

    async def send_frame(self, image_data_url: str) -> None:
        """Push one camera frame toward the model. Raises if the socket is gone."""
        ws = self._ws
        if ws is None or self._closed:
            raise MiniCPMOConnectionError("not_connected", "send_frame before connect")
        await ws.send_json(
            {
                "type": "input_video_frame",
                "session_id": self.session_id,
                "image": image_data_url,
            }
        )

    async def receive(self) -> Any:
        """Yield decoded provider messages until the socket closes.

        Yields raw payloads; interpreting them is visual_state's job. A message
        that fails to decode is skipped, never raised — a single malformed frame
        must not end a live session.
        """
        ws = self._ws
        if ws is None:
            return
        async for msg in ws:
            msg_type = getattr(msg, "type", None)
            if msg_type == aiohttp.WSMsgType.TEXT:
                yield msg.data
            elif msg_type == aiohttp.WSMsgType.BINARY:
                yield msg.data
            elif msg_type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                              aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                break

    async def aclose(self) -> None:
        """Idempotent teardown — safe to call from any cleanup path."""
        self._closed = True
        await self._cleanup()

    async def _cleanup(self) -> None:
        ws, self._ws = self._ws, None
        session, self._session = self._session, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - already gone is fine
                pass
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001
                pass


_GATEWAY_REASONS: dict[GatewayState, str] = {
    GatewayState.PENDING_DEPLOYMENT: "realtime_gateway_not_configured",
    GatewayState.INVALID_WORKER_URL: "realtime_url_is_worker_url",
    GatewayState.INVALID_SCHEME: "realtime_url_invalid_scheme",
    GatewayState.CONFIGURED: "configured",
}


def gateway_reason(state: GatewayState) -> str:
    """Stable, greppable reason code for a Gateway state."""
    return _GATEWAY_REASONS.get(state, "unknown")

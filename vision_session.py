"""Per-Lucy-session live vision: bounded ingestion, lifecycle, telemetry.

One MiniCPMOVisionSession is bound to one ArcheRealtimeSession (same session id)
and exists ONLY while camera understanding is on. It owns the whole live-video
path so the bridge itself stays a voice bridge:

    camera frames -> offer_frame() -> FPS gate -> bounded queue -> Gateway
                                                       |
                                              (queue full: drop OLDEST)
                                                       v
                                     provider messages -> RollingVisualState
                                                       v
                                    notable change -> on_visual_update() -> Arche

THE CENTRAL RULE: vision failure degrades to voice-only, always. Every public
method here is non-raising. ``start()`` returns immediately and does its work in
a background task; a Gateway that is missing, cold, slow, broken, or sending
garbage moves this object to ``degraded``/``unavailable`` and stops — the
Inworld voice session that owns this object never sees an exception and never
drops a turn. That is why start() spawns rather than awaits, and why _run()
wraps everything in a bare except.

BACKPRESSURE: for live understanding a fresh frame beats a queued one. When the
model is slower than the camera (always, during a Modal cold start) the queue
fills and we discard the OLDEST pending frame instead of growing the queue.
Latency stays bounded at ``VISION_MAX_QUEUE_SIZE / VISION_TARGET_FPS`` seconds
instead of climbing forever, and Arche reasons about the room as it is now, not
as it was thirty seconds ago.

PRIVACY: frames are bytes we forward and immediately forget. No frame, and no
fragment of one, is ever written to a log line — only counts, timings, and the
model's own text description.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4

from minicpmo_provider import (
    COLD_START_THRESHOLD_MS,
    MAX_FRAME_CHARS,
    GatewayState,
    MiniCPMOConfig,
    MiniCPMOConnectionError,
    MiniCPMORealtimeConnection,
    VisionProviderState,
    gateway_reason,
    load_minicpmo_config,
)
from visual_state import RollingVisualState, VisualState

logger = logging.getLogger(__name__)


class BoundedFrameQueue:
    """Fixed-capacity frame buffer that drops the OLDEST frame when full.

    Not an asyncio.Queue: a standard queue either blocks the producer (which
    would stall the camera sampler, and on the LiveKit path the audio task
    alongside it) or rejects the newest frame (which is exactly backwards for
    live understanding — the newest frame is the valuable one).
    """

    def __init__(self, max_size: int) -> None:
        # Guard the degenerate case: a 0/negative cap would silently discard
        # every frame and look identical to a dead provider.
        self.max_size = max(1, int(max_size))
        self._items: deque[str] = deque()
        self._available = asyncio.Event()
        self.offered = 0
        self.dropped = 0
        self.delivered = 0

    def __len__(self) -> int:
        return len(self._items)

    def put(self, frame: str) -> bool:
        """Enqueue a frame. Returns False if it displaced an older one."""
        self.offered += 1
        displaced = False
        if len(self._items) >= self.max_size:
            self._items.popleft()
            self.dropped += 1
            displaced = True
        self._items.append(frame)
        self._available.set()
        return not displaced

    async def get(self) -> str:
        """Await the next frame, newest-first is NOT implied — FIFO within the cap."""
        while not self._items:
            self._available.clear()
            await self._available.wait()
        frame = self._items.popleft()
        if not self._items:
            self._available.clear()
        self.delivered += 1
        return frame

    def clear(self) -> None:
        """Drop everything pending (camera off / teardown) without unblocking readers.

        Discarded frames count as dropped: otherwise offered != delivered +
        dropped + pending and the backpressure numbers quietly understate how
        much the camera outran the model.
        """
        self.dropped += len(self._items)
        self._items.clear()
        self._available.clear()

    def stats(self) -> dict[str, int]:
        return {
            "offered": self.offered,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "pending": len(self._items),
            "capacity": self.max_size,
        }


@dataclass
class VisionLatencyTelemetry:
    """Cold-start / first-context instrumentation for one vision session.

    Modal scales to zero (scaledown_window=300s), so the honest question is not
    "how fast is the model" but "how long does a real user wait for the FIRST
    piece of visual context after a scale-to-zero". These marks answer it, and
    ``cold`` records whether that session paid the cold-start price at all —
    without which an average latency across warm and cold sessions is
    meaningless.

    Monotonic clocks only: wall-clock is not safe for durations.
    """

    session_requested_at: float | None = None
    modal_request_started_at: float | None = None
    modal_connected_at: float | None = None
    model_ready_at: float | None = None
    first_frame_sent_at: float | None = None
    first_visual_event_at: float | None = None
    cold: bool | None = None
    # Duration of the attempt that actually succeeded. modal_connect_ms spans
    # every attempt (the honest user-facing wait); this one is what cold/warm
    # is judged on, so a retried connect can't misreport a warm container.
    last_attempt_connect_ms: float | None = None

    def mark(self, name: str, *, now: float) -> None:
        attr = f"{name}_at"
        if hasattr(self, attr) and getattr(self, attr) is None:
            setattr(self, attr, now)

    @staticmethod
    def _ms(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round((end - start) * 1000.0, 1)

    @property
    def modal_connect_ms(self) -> float | None:
        return self._ms(self.modal_request_started_at, self.modal_connected_at)

    @property
    def model_ready_ms(self) -> float | None:
        return self._ms(self.modal_request_started_at, self.model_ready_at)

    @property
    def first_visual_context_ms(self) -> float | None:
        """The number that actually matters: request -> first usable visual context."""
        return self._ms(self.session_requested_at, self.first_visual_event_at)

    @property
    def first_frame_ms(self) -> float | None:
        return self._ms(self.session_requested_at, self.first_frame_sent_at)

    def classify(self) -> str:
        """'cold' | 'warm' | 'unknown' — never guessed from a missing connect."""
        if self.cold is None:
            return "unknown"
        return "cold" if self.cold else "warm"

    def as_dict(self) -> dict[str, Any]:
        return {
            "modal_connect_ms": self.modal_connect_ms,
            "model_ready_ms": self.model_ready_ms,
            "first_frame_ms": self.first_frame_ms,
            "first_visual_context_ms": self.first_visual_context_ms,
            "last_attempt_connect_ms": self.last_attempt_connect_ms,
            "start_class": self.classify(),
        }

    def log_fields(self) -> str:
        """Single structured line matching the repo's key=value log convention."""
        return (
            f"modal_connect_ms={self.modal_connect_ms} "
            f"model_ready_ms={self.model_ready_ms} "
            f"first_frame_ms={self.first_frame_ms} "
            f"first_visual_context_ms={self.first_visual_context_ms} "
            f"start_class={self.classify()}"
        )


@dataclass(frozen=True)
class VisionSessionStatus:
    """Externally observable provider state for one session (health endpoints, logs)."""

    state: VisionProviderState
    reason: str
    gateway: GatewayState
    frames: dict[str, int]
    telemetry: dict[str, Any]
    updates_ingested: int = 0
    updates_published: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "gateway": self.gateway.value,
            "frames": self.frames,
            "telemetry": self.telemetry,
            "updates_ingested": self.updates_ingested,
            "updates_published": self.updates_published,
        }


class MiniCPMOVisionSession:
    """Live-vision sidecar for one ArcheRealtimeSession.

    Bound to ``session_id`` so visual context can never cross sessions: the
    bridge constructs exactly one of these per conversation and the frames it
    receives come from that conversation's own transport, which has already
    authenticated the user (LiveKit room token, or the Arche API bearer key).
    There is no path by which a caller reaches another user's vision session,
    because there is no lookup by id — the object is owned, not registered.
    """

    def __init__(
        self,
        session_id: str,
        *,
        config: MiniCPMOConfig | None = None,
        on_visual_update: Callable[[VisualState], Awaitable[None]] | None = None,
        connection_factory: Callable[[MiniCPMOConfig, str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_id = session_id
        # Identity sent to the Gateway. NOT session_id: Lucy derives that from
        # INWORLD_REALTIME_SESSION_ID / LIVEKIT_ROOM_NAME / a millisecond
        # timestamp (inworld_realtime_bridge.load_inworld_realtime_settings), so
        # two conversations can share one. The Gateway is shared infrastructure
        # keyed on whatever we send it, so a collision there could cross two
        # users' video. A per-instance suffix makes that impossible while
        # session_id stays the Lucy-side association used for logs and status.
        self.stream_id = f"{session_id}-{uuid4().hex[:8]}"
        self.config = config if config is not None else load_minicpmo_config()
        self._on_visual_update = on_visual_update
        # Injectable so tests exercise the full lifecycle without a GPU, a
        # Gateway, or a socket. Production callers leave it None.
        self._connection_factory = connection_factory or (
            lambda cfg, sid: MiniCPMORealtimeConnection(cfg, session_id=sid)
        )
        # Every connection this session opens is created with stream_id, never
        # the (possibly shared) Lucy session id.
        self._clock = clock

        self.queue = BoundedFrameQueue(self.config.max_queue_size)
        self.telemetry = VisionLatencyTelemetry()
        self.rolling = RollingVisualState(
            min_update_interval_seconds=self.config.min_state_update_interval_seconds
        )

        self._state = VisionProviderState.DISABLED
        self._reason = "not_started"
        self._connection: Any | None = None
        # Total Gateway connects attempted for this CONVERSATION, across every
        # camera on/off cycle. Survives stop()'s re-arm on purpose: without it,
        # toggling the camera would reset the retry budget and let a client
        # wake an L40S an unbounded number of times.
        self._connects_used = 0
        self._oversized_frames = 0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._run_task: asyncio.Task[Any] | None = None
        self._stopped = asyncio.Event()
        self._last_frame_accepted_at = 0.0
        self._started = False
        # Reentrancy guard for stop(); distinct from _stopped, which also acts
        # as the send loop's exit signal and is cleared again on re-arm.
        self._stopping = False
        # Set only by aclose(): the conversation is over, so unlike a
        # camera-off this must NOT re-arm.
        self._terminal = False

    # --- observable state -------------------------------------------------

    @property
    def state(self) -> VisionProviderState:
        return self._state

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def active(self) -> bool:
        """True while a connection is being established or is live."""
        return self._state in (
            VisionProviderState.CONNECTING,
            VisionProviderState.COLD_START,
            VisionProviderState.READY,
        )

    @property
    def wants_frames(self) -> bool:
        """True while capturing frames is worthwhile — including BEFORE connect.

        Deliberately wider than ``active``: the camera sampler is started from
        the transport, which may subscribe an already-published camera track
        before start() has run. Gating that on ``active`` would skip the sampler
        for a user who had their camera on at session start. The queue is
        bounded, so buffering a handful of frames across the connect window
        costs at most VISION_MAX_QUEUE_SIZE frames and makes the first frame
        available the moment the Gateway is ready.

        False once the provider has settled into a state where nothing will
        consume frames (unavailable/degraded/stopped), so a pending Gateway
        genuinely means no camera is touched.
        """
        if not self.config.enabled or self._terminal or self._stopped.is_set():
            return False
        if self._connects_used >= self.config.max_total_connects:
            return False
        return self._state not in (
            VisionProviderState.UNAVAILABLE,
            VisionProviderState.DEGRADED,
        )

    @property
    def latest_state(self) -> VisualState | None:
        return self.rolling.current

    def context_line(self) -> str | None:
        """Compact visual context for the session instructions, or None."""
        return self.rolling.context_line()

    def status(self) -> VisionSessionStatus:
        return VisionSessionStatus(
            state=self._state,
            reason=self._reason,
            gateway=self.config.gateway_state,
            frames=self.queue.stats(),
            telemetry=self.telemetry.as_dict(),
            updates_ingested=self.rolling.updates_ingested,
            updates_published=self.rolling.updates_published,
        )

    def _set_state(self, state: VisionProviderState, reason: str) -> None:
        if self._state is state and self._reason == reason:
            return
        self._state = state
        self._reason = reason
        if state in (VisionProviderState.DEGRADED, VisionProviderState.UNAVAILABLE):
            # A frozen last-known scene is worse than no scene: it would keep
            # describing a room that may have changed, AND it would shadow the
            # OpenRouter fallback (which _effective_vision_summary only reaches
            # when the live context line is empty). Dropping it is what makes
            # the documented degradation path actually work.
            self.rolling.clear()
        logger.info(
            "minicpmo_vision_state=%s reason=%s session_id=%s gateway=%s",
            state.value, reason, self.session_id, self.config.gateway_state.value,
        )

    # --- lifecycle --------------------------------------------------------

    def start(self) -> bool:
        """Begin the vision session. Non-blocking; never raises.

        Returns False when vision will not run (disabled, or the Gateway is not
        deployed) so the caller can log it once — but a False here is a normal,
        expected outcome today, not an error: the voice conversation proceeds
        identically either way.
        """
        if self._terminal:
            return False
        if self._started:
            return self.active
        self._started = True
        now = self._clock()
        self.telemetry.mark("session_requested", now=now)

        if not self.config.enabled:
            self._set_state(VisionProviderState.DISABLED, "minicpmo_disabled")
            return False

        gateway = self.config.gateway_state
        if gateway is not GatewayState.CONFIGURED:
            # The expected state until the OpenBMB Gateway is deployed. Not a
            # failure of this session — nothing is retried and no GPU is woken.
            self._set_state(VisionProviderState.UNAVAILABLE, gateway_reason(gateway))
            logger.info(
                "minicpmo_vision_unavailable=true session_id=%s reason=%s "
                "worker_url_configured=%s (deploy the OpenBMB realtime Gateway and set "
                "MINICPMO_REALTIME_URL; voice conversation is unaffected)",
                self.session_id, gateway_reason(gateway), bool(self.config.worker_url),
            )
            return False

        self._set_state(VisionProviderState.CONNECTING, "connect_requested")
        self._run_task = asyncio.create_task(self._run())
        return True

    async def _run(self) -> None:
        """Connect, then pump frames and messages until stopped. Never raises."""
        try:
            connection = await self._connect_with_retries()
            if connection is None:
                return
            self._connection = connection

            sender = asyncio.create_task(self._send_loop(connection))
            receiver = asyncio.create_task(self._receive_loop(connection))
            self._tasks.update({sender, receiver})

            try:
                # A session that outlives MINICPMO_SESSION_TIMEOUT_SECONDS is
                # almost certainly a leak (browser gone without a close); time
                # it out so a forgotten socket can't pin a GPU indefinitely.
                await asyncio.wait_for(
                    asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED),
                    timeout=self.config.session_timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._set_state(VisionProviderState.DEGRADED, "session_timeout")
                logger.info(
                    "minicpmo_vision_session_timeout=true session_id=%s timeout_seconds=%s %s",
                    self.session_id, self.config.session_timeout_seconds,
                    self.telemetry.log_fields(),
                )
            finally:
                # Whichever loop is still alive must go. asyncio.wait returns on
                # FIRST_COMPLETED, so a Gateway that closes cleanly would
                # otherwise leave _send_loop parked on queue.get() forever, and
                # a session_timeout would leave BOTH running past _run().
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                self._tasks.difference_update({sender, receiver})
                if self._state is VisionProviderState.READY:
                    # The socket ended while we still believed we were live.
                    # Say so, and drop the stale note (see _set_state).
                    self._set_state(VisionProviderState.DEGRADED, "gateway_closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - vision must never break the voice session
            self._set_state(VisionProviderState.DEGRADED, "run_failed")
            logger.warning(
                "minicpmo_vision_run_failed=true session_id=%s error_type=%s error=%s",
                self.session_id, type(exc).__name__, exc,
            )
        finally:
            await self._teardown_connection()

    async def _connect_with_retries(self) -> Any | None:
        """Connect conservatively. Returns the connection, or None if degraded.

        Bounded attempts with a fixed backoff — never an unbounded reconnect
        loop. Every attempt can wake an L40S from scale-to-zero, so an outage
        must not turn Lucy into a GPU-spinning retry storm.
        """
        attempts = max(1, self.config.max_connect_attempts)
        for attempt in range(1, attempts + 1):
            if self._connects_used >= self.config.max_total_connects:
                # Per-conversation GPU budget spent. Stop for good rather than
                # letting camera toggling keep waking L40S capacity.
                self._set_state(VisionProviderState.UNAVAILABLE, "connect_budget_exhausted")
                logger.warning(
                    "minicpmo_vision_connect_budget_exhausted=true session_id=%s "
                    "connects_used=%s budget=%s",
                    self.session_id, self._connects_used, self.config.max_total_connects,
                )
                return None
            self._connects_used += 1
            connection = self._connection_factory(self.config, self.stream_id)
            started = self._clock()
            # Session-level mark (first attempt only) drives the user-facing
            # "how long until visual context" number; `started` is per-attempt
            # and is what cold/warm is classified from, so a retry cannot make a
            # warm container look cold.
            self.telemetry.mark("modal_request_started", now=started)
            try:
                await connection.connect()
            except MiniCPMOConnectionError as exc:
                await _safe_aclose(connection)
                terminal = attempt >= attempts
                logger.warning(
                    "minicpmo_vision_connect_failed=true session_id=%s attempt=%s/%s "
                    "reason=%s detail=%s terminal=%s",
                    self.session_id, attempt, attempts, exc.reason, exc.detail, terminal,
                )
                if terminal:
                    # connect_timeout is the one failure that says "the
                    # infrastructure never answered" rather than "it broke" —
                    # keep them distinguishable for operators.
                    state = (
                        VisionProviderState.UNAVAILABLE
                        if exc.reason in ("connect_timeout", "realtime_gateway_not_configured",
                                          "realtime_url_is_worker_url", "realtime_url_invalid_scheme")
                        else VisionProviderState.DEGRADED
                    )
                    self._set_state(state, exc.reason)
                    return None
                await asyncio.sleep(self.config.reconnect_backoff_seconds)
                continue
            except asyncio.CancelledError:
                await _safe_aclose(connection)
                raise
            except Exception as exc:  # noqa: BLE001 - a factory/transport bug is still just degraded vision
                await _safe_aclose(connection)
                self._set_state(VisionProviderState.DEGRADED, "connect_error")
                logger.warning(
                    "minicpmo_vision_connect_error=true session_id=%s error_type=%s error=%s",
                    self.session_id, type(exc).__name__, exc,
                )
                return None

            connected_at = self._clock()
            self.telemetry.mark("modal_connected", now=connected_at)
            connect_ms = (connected_at - started) * 1000.0
            # Cold vs warm is inferred from how long Modal took to answer: a
            # warm container is sub-second, a cold one has to load MiniCPM-o
            # onto the GPU. Recorded per session so cold-start latency can be
            # reported separately instead of averaged away.
            self.telemetry.cold = connect_ms >= COLD_START_THRESHOLD_MS
            self.telemetry.last_attempt_connect_ms = round(connect_ms, 1)
            if self.telemetry.cold:
                self._set_state(VisionProviderState.COLD_START, "modal_cold_start")

            # The Gateway accepting the socket is what "model ready" means for
            # this provider — it fronts a worker that already holds the model.
            self.telemetry.mark("model_ready", now=connected_at)
            self._set_state(VisionProviderState.READY, "connected")
            logger.info(
                "minicpmo_vision_connected=true session_id=%s attempt=%s %s",
                self.session_id, attempt, self.telemetry.log_fields(),
            )
            return connection
        return None

    # --- frame ingestion --------------------------------------------------

    def offer_frame(self, image_data_url: str) -> bool:
        """Offer one camera frame. Returns True if accepted into the queue.

        Called from the camera path on every sampled frame, so it must stay
        cheap and synchronous: it rate-limits to VISION_TARGET_FPS and enqueues.
        Never raises and never blocks the caller — on the LiveKit path this runs
        on the same event loop as audio forwarding.
        """
        if not self.wants_frames:
            return False
        if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image/"):
            return False
        if len(image_data_url) > MAX_FRAME_CHARS:
            # An oversized frame is refused rather than queued: the queue holds
            # up to VISION_MAX_QUEUE_SIZE frames per session, so without a
            # per-frame ceiling a client on the API path could pin tens of
            # megabytes just by sending large images.
            self._oversized_frames += 1
            if self._oversized_frames % 20 == 1:
                logger.info(
                    "minicpmo_vision_frame_too_large=true session_id=%s count=%s limit=%s",
                    self.session_id, self._oversized_frames, MAX_FRAME_CHARS,
                )
            return False
        # A frame on an armed-but-idle session means the camera just came (back)
        # on. Self-starting here is what makes camera-off -> camera-on restore
        # live vision on BOTH transports, without either needing restart logic
        # of its own. start() is non-blocking and idempotent, so this is a
        # no-op on every subsequent frame.
        if not self._started:
            self.start()
        now = self._clock()
        # FPS gate first: pointless to enqueue faster than the configured target
        # only to drop it a moment later.
        if now - self._last_frame_accepted_at < self.config.frame_interval_seconds:
            return False
        self._last_frame_accepted_at = now
        fit = self.queue.put(image_data_url)
        if not fit and self.queue.dropped % 10 == 1:
            # Sampled logging: a sustained cold start drops steadily and we
            # don't want a log line per frame.
            logger.info(
                "minicpmo_vision_backpressure=true session_id=%s dropped=%s capacity=%s "
                "(oldest frame discarded; newest frame kept)",
                self.session_id, self.queue.dropped, self.queue.max_size,
            )
        return True

    async def _send_loop(self, connection: Any) -> None:
        """Drain the bounded queue into the Gateway until the socket dies."""
        try:
            while not self._stopped.is_set():
                frame = await self.queue.get()
                await connection.send_frame(frame)
                self.telemetry.mark("first_frame_sent", now=self._clock())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_state(VisionProviderState.DEGRADED, "send_failed")
            logger.warning(
                "minicpmo_vision_send_failed=true session_id=%s error_type=%s error=%s",
                self.session_id, type(exc).__name__, exc,
            )

    async def _receive_loop(self, connection: Any) -> None:
        """Fold provider messages into the rolling state; publish notable changes."""
        try:
            async for payload in connection.receive():
                now = self._clock()
                # Any decodable message proves the round trip works, even if it
                # carries no observation — that is the honest "first visual
                # event" mark for latency purposes.
                should_publish = self.rolling.ingest(payload, now=now)
                if self.rolling.current is not None:
                    self.telemetry.mark("first_visual_event", now=now)
                    if self.telemetry.first_visual_event_at == now:
                        logger.info(
                            "minicpmo_vision_first_context=true session_id=%s %s",
                            self.session_id, self.telemetry.log_fields(),
                        )
                if should_publish and self._on_visual_update is not None:
                    state = self.rolling.current
                    if state is not None:
                        try:
                            await self._on_visual_update(state)
                        except Exception as exc:  # noqa: BLE001 - a consumer fault is not a vision fault
                            logger.warning(
                                "minicpmo_vision_update_callback_failed=true session_id=%s "
                                "error_type=%s error=%s",
                                self.session_id, type(exc).__name__, exc,
                            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead socket degrades vision, nothing more
            self._set_state(VisionProviderState.DEGRADED, "gateway_disconnected")
            logger.warning(
                "minicpmo_vision_receive_failed=true session_id=%s error_type=%s error=%s",
                self.session_id, type(exc).__name__, exc,
            )

    # --- teardown ---------------------------------------------------------

    async def stop(self, reason: str = "camera_stopped") -> None:
        """Stop ingestion and release everything. Idempotent; never raises.

        This is the path taken when the user disables the camera, leaves, ends
        the conversation, or the session times out. All four must leave nothing
        behind: no pending frames, no visual state feeding the prompt, no
        socket, no task, no GPU session.

        Cancellation-safe: teardown is frequently awaited from a path that is
        itself being cancelled (session shutdown, a cancelled LiveKit callback).
        Without the try/finally below, a cancellation mid-teardown would leave
        _stopping latched True, and every later stop()/aclose() would silently
        no-op — leaking the Gateway socket and the Modal GPU session for the
        rest of the process's life.
        """
        if self._stopping:
            return
        self._stopping = True
        self._stopped.set()

        try:
            for task in list(self._tasks):
                if not task.done():
                    task.cancel()
            if self._run_task is not None and not self._run_task.done():
                self._run_task.cancel()

            pending = [t for t in (*self._tasks, self._run_task) if t is not None]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            self._tasks.clear()
            self._run_task = None
            # Shielded: the socket must be released even if this teardown is
            # itself cancelled — that is precisely the case where a leaked GPU
            # session would otherwise go unnoticed.
            try:
                await asyncio.shield(asyncio.ensure_future(self._teardown_connection()))
            except asyncio.CancelledError:
                # The shielded close still runs to completion in the background.
                pass
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass
            # Clearing the rolling state matters as much as closing the socket:
            # a stale visual note would have Arche talking about a room she can
            # no longer see.
            self.queue.clear()
            self.rolling.clear()
            self._finish_stop(reason)

    def _finish_stop(self, reason: str) -> None:
        """Settle state and re-arm. Runs from stop()'s finally, so it always runs."""
        if self._state is not VisionProviderState.DISABLED:
            self._set_state(VisionProviderState.DISABLED, reason)
        logger.info(
            "minicpmo_vision_stopped=true session_id=%s reason=%s frames=%s %s "
            "terminal=%s connects_used=%s",
            self.session_id, reason, self.queue.stats(), self.telemetry.log_fields(),
            self._terminal, self._connects_used,
        )

        # Re-arm unless the conversation itself ended. A user who turns the
        # camera off and back on mid-conversation must get live vision back;
        # without this, start() would short-circuit on _started forever and
        # MiniCPM-o would stay silently dark while the OpenRouter path resumed.
        # Telemetry resets so the next camera session's cold/warm and
        # first-visual-context numbers describe that session, not the old one.
        #
        # _connects_used is deliberately NOT reset: it is the per-conversation
        # GPU budget. Resetting it here would let repeated camera toggling wake
        # an L40S an unbounded number of times.
        self._stopping = False
        if not self._terminal:
            self._started = False
            self._stopped.clear()
            self.telemetry = VisionLatencyTelemetry()
            self._last_frame_accepted_at = 0.0


    async def aclose(self) -> None:
        """Terminal teardown for the end of the conversation.

        Unlike stop() (a camera-off, which re-arms), this marks the session
        spent: a later start() is refused and no frames are wanted, so nothing
        can resurrect a vision session after the conversation is over.
        """
        self._terminal = True
        await self.stop(reason="session_closed")
        self._stopped.set()

    async def _teardown_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await _safe_aclose(connection)


async def _safe_aclose(connection: Any) -> None:
    try:
        await connection.aclose()
    except Exception:  # noqa: BLE001 - teardown must never raise
        pass


def build_vision_session(
    session_id: str,
    *,
    on_visual_update: Callable[[VisualState], Awaitable[None]] | None = None,
    config: MiniCPMOConfig | None = None,
) -> MiniCPMOVisionSession | None:
    """Construct a vision session, or None when MiniCPM-o is switched off.

    Returning None rather than an inert object keeps the bridge's hot paths free
    of vision work entirely when the feature is off — the same shape as
    build_emotional_dataset_writer_from_env().
    """
    config = config if config is not None else load_minicpmo_config()
    if not config.enabled:
        return None
    return MiniCPMOVisionSession(
        session_id, config=config, on_visual_update=on_visual_update
    )

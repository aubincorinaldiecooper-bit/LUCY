"""minicpmo_provider.py: env parsing, Gateway classification, worker health.

Pins the rules that keep this integration honest:
  - MiniCPM-o is off unless MINICPMO_ENABLED=true (no GPU is woken by default),
  - a blank MINICPMO_REALTIME_URL is the *expected* pending-Gateway state, not
    an error, and it refuses to connect without touching the network,
  - MINICPMO_REALTIME_URL must NEVER be the Modal worker URL — that is the one
    misconfiguration that would look configured and fail every session,
  - worker health never raises and never blocks: every network failure mode
    degrades to a HealthResult,
  - reaching the worker is NOT evidence that live video works.

No GPU, no Modal, no socket: the aiohttp boundary is faked, exactly as
tests/test_vision_context.py fakes it for the OpenRouter path.
"""

import asyncio
import types
import unittest
from unittest import mock

import minicpmo_provider as mp

WORKER_URL = "https://aubincorinaldiecooper--minicpmo45-realtime-minicpmo-worker.modal.run"


def _env(**overrides):
    env = {"MINICPMO_ENABLED": "true", "MINICPMO_WORKER_URL": WORKER_URL}
    env.update({k: v for k, v in overrides.items() if v is not None})
    return env


def _config(**overrides):
    kwargs = dict(
        enabled=True,
        worker_url=WORKER_URL,
        realtime_url="",
        connect_timeout_seconds=120.0,
        session_timeout_seconds=900.0,
        health_timeout_seconds=10.0,
        target_fps=2.0,
        max_queue_size=4,
        max_connect_attempts=2,
        reconnect_backoff_seconds=5.0,
        min_state_update_interval_seconds=4.0,
    )
    kwargs.update(overrides)
    return mp.MiniCPMOConfig(**kwargs)


class ConfigParsingTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            cfg = mp.load_minicpmo_config()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.worker_url, "")
        self.assertEqual(cfg.realtime_url, "")

    def test_reads_documented_defaults(self):
        with mock.patch.dict("os.environ", _env(), clear=True):
            cfg = mp.load_minicpmo_config()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.worker_url, WORKER_URL)
        self.assertEqual(cfg.connect_timeout_seconds, 120.0)
        self.assertEqual(cfg.session_timeout_seconds, 900.0)
        self.assertEqual(cfg.target_fps, 2.0)
        self.assertEqual(cfg.max_queue_size, 4)

    def test_reads_overrides_from_env(self):
        env = _env(
            MINICPMO_CONNECT_TIMEOUT_SECONDS="45",
            MINICPMO_SESSION_TIMEOUT_SECONDS="600",
            VISION_TARGET_FPS="5",
            VISION_MAX_QUEUE_SIZE="12",
        )
        with mock.patch.dict("os.environ", env, clear=True):
            cfg = mp.load_minicpmo_config()
        self.assertEqual(cfg.connect_timeout_seconds, 45.0)
        self.assertEqual(cfg.session_timeout_seconds, 600.0)
        self.assertEqual(cfg.target_fps, 5.0)
        self.assertEqual(cfg.max_queue_size, 12)

    def test_target_fps_drives_frame_interval(self):
        # The FPS knob is a *starting value*, not a baked-in assumption: the
        # ingestion gate is derived from it, so retuning is an env change.
        self.assertAlmostEqual(_config(target_fps=2.0).frame_interval_seconds, 0.5)
        self.assertAlmostEqual(_config(target_fps=4.0).frame_interval_seconds, 0.25)
        self.assertAlmostEqual(_config(target_fps=1.0).frame_interval_seconds, 1.0)

    def test_malformed_numbers_fall_back_without_raising(self):
        # A typo'd vision var must never stop the process that also serves the
        # voice conversation from starting.
        env = _env(
            MINICPMO_CONNECT_TIMEOUT_SECONDS="not-a-number",
            VISION_MAX_QUEUE_SIZE="lots",
            VISION_TARGET_FPS="",
        )
        with mock.patch.dict("os.environ", env, clear=True):
            cfg = mp.load_minicpmo_config()
        self.assertEqual(cfg.connect_timeout_seconds, 120.0)
        self.assertEqual(cfg.max_queue_size, 4)
        self.assertEqual(cfg.target_fps, 2.0)

    def test_non_positive_values_fall_back(self):
        env = _env(VISION_TARGET_FPS="0", VISION_MAX_QUEUE_SIZE="-3",
                   MINICPMO_CONNECT_TIMEOUT_SECONDS="-1")
        with mock.patch.dict("os.environ", env, clear=True):
            cfg = mp.load_minicpmo_config()
        self.assertEqual(cfg.target_fps, 2.0)
        self.assertEqual(cfg.max_queue_size, 4)
        self.assertEqual(cfg.connect_timeout_seconds, 120.0)

    def test_enabled_accepts_the_repo_boolean_spellings(self):
        for raw in ("true", "TRUE", "  True  ", "1", "yes", "on"):
            with mock.patch.dict("os.environ", _env(MINICPMO_ENABLED=raw), clear=True):
                self.assertTrue(mp.load_minicpmo_config().enabled, raw)
        for raw in ("false", "0", "no", "off", ""):
            with mock.patch.dict("os.environ", _env(MINICPMO_ENABLED=raw), clear=True):
                self.assertFalse(mp.load_minicpmo_config().enabled, raw)

    def test_trailing_slash_stripped_from_worker_url(self):
        with mock.patch.dict("os.environ", _env(MINICPMO_WORKER_URL=WORKER_URL + "/"), clear=True):
            self.assertEqual(mp.load_minicpmo_config().worker_url, WORKER_URL)


class GatewayStateTests(unittest.TestCase):
    """The pending-Gateway rules. These are the load-bearing guards."""

    def test_blank_realtime_url_is_pending_deployment(self):
        # The expected state today: the OpenBMB realtime Gateway is not deployed.
        with mock.patch.dict("os.environ", _env(), clear=True):
            cfg = mp.load_minicpmo_config()
        self.assertEqual(cfg.realtime_url, "")
        self.assertIs(cfg.gateway_state, mp.GatewayState.PENDING_DEPLOYMENT)
        self.assertFalse(cfg.realtime_ready)
        self.assertEqual(
            mp.gateway_reason(cfg.gateway_state), "realtime_gateway_not_configured"
        )

    def test_realtime_url_is_never_defaulted_to_the_worker_url(self):
        # Nothing in config loading may quietly fill the Gateway slot from the
        # worker slot, however convenient that would look.
        with mock.patch.dict("os.environ", _env(), clear=True):
            cfg = mp.load_minicpmo_config()
        self.assertNotEqual(cfg.realtime_url, cfg.worker_url)
        self.assertEqual(cfg.realtime_url, "")

    def test_worker_url_as_realtime_url_is_rejected(self):
        # https -> wss on the same host is the most likely wrong turn here.
        for candidate in (
            WORKER_URL.replace("https://", "wss://"),
            WORKER_URL.replace("https://", "ws://"),
            WORKER_URL.replace("https://", "wss://") + "/v1/realtime?mode=video",
            WORKER_URL.replace("https://", "WSS://").upper(),
        ):
            state = mp.resolve_gateway_state(candidate, WORKER_URL)
            self.assertIs(state, mp.GatewayState.INVALID_WORKER_URL, candidate)
            self.assertEqual(mp.gateway_reason(state), "realtime_url_is_worker_url")

    def test_worker_url_as_realtime_url_rejected_through_env(self):
        env = _env(MINICPMO_REALTIME_URL=WORKER_URL.replace("https://", "wss://"))
        with mock.patch.dict("os.environ", env, clear=True):
            cfg = mp.load_minicpmo_config()
        self.assertIs(cfg.gateway_state, mp.GatewayState.INVALID_WORKER_URL)
        self.assertFalse(cfg.realtime_ready)

    def test_non_websocket_scheme_is_rejected(self):
        for candidate in ("https://gateway.example.com/v1/realtime",
                          "http://gateway.example.com",
                          "gateway.example.com",
                          "ftp://gateway.example.com"):
            self.assertIs(
                mp.resolve_gateway_state(candidate, WORKER_URL),
                mp.GatewayState.INVALID_SCHEME,
                candidate,
            )

    def test_a_real_gateway_url_is_accepted(self):
        # What MINICPMO_REALTIME_URL will hold once the Gateway is deployed.
        state = mp.resolve_gateway_state(
            "wss://minicpmo-gateway.example.com/v1/realtime?mode=video", WORKER_URL
        )
        self.assertIs(state, mp.GatewayState.CONFIGURED)
        self.assertTrue(_config(realtime_url="wss://gw.example.com/v1/realtime").realtime_ready)


# --- aiohttp boundary fakes (same shape as tests/test_vision_context.py) -------


class _FakeResponse:
    def __init__(self, status, text_body=""):
        self.status = status
        self._text_body = text_body

    async def text(self):
        return self._text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response=None, raise_on_get=None):
        self._response = response
        self._raise_on_get = raise_on_get
        self.closed = False
        self.last_url = None

    def get(self, url):
        self.last_url = url
        if self._raise_on_get is not None:
            raise self._raise_on_get
        return self._response

    async def close(self):
        self.closed = True


def _patch_aiohttp(fake_session):
    # minicpmo_provider imports aiohttp at module level, so the fake must
    # replace the name bound in that module's namespace.
    fake = types.SimpleNamespace(
        ClientSession=lambda timeout=None: fake_session,
        ClientTimeout=lambda **k: None,
    )
    return mock.patch.object(mp, "aiohttp", fake)


class WorkerHealthTests(unittest.TestCase):
    def test_health_success(self):
        fake = _FakeSession(_FakeResponse(200, "ok"))
        with _patch_aiohttp(fake):
            result = asyncio.run(mp.check_worker_health(_config()))
        self.assertTrue(result.ok)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.reason, "reachable")
        self.assertEqual(fake.last_url, WORKER_URL)
        self.assertTrue(fake.closed, "health probe must not leak its aiohttp session")

    def test_non_health_route_still_counts_as_reachable(self):
        # The worker's root path is not guaranteed to be a health route; a 404
        # from Modal still proves the container is up and routing.
        for status in (404, 405, 401, 403):
            fake = _FakeSession(_FakeResponse(status))
            with _patch_aiohttp(fake):
                result = asyncio.run(mp.check_worker_health(_config()))
            self.assertTrue(result.ok, status)

    def test_server_error_is_a_health_failure(self):
        fake = _FakeSession(_FakeResponse(503, "no GPU capacity"))
        with _patch_aiohttp(fake):
            result = asyncio.run(mp.check_worker_health(_config()))
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 503)
        self.assertEqual(result.reason, "worker_error_status")
        self.assertIn("no GPU capacity", result.detail)

    def test_modal_unreachable_degrades_without_raising(self):
        fake = _FakeSession(raise_on_get=OSError("Name or service not known"))
        with _patch_aiohttp(fake):
            result = asyncio.run(mp.check_worker_health(_config()))
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "unreachable")
        self.assertIn("OSError", result.detail)
        self.assertTrue(fake.closed)

    def test_health_timeout_degrades_without_raising(self):
        fake = _FakeSession(raise_on_get=asyncio.TimeoutError())
        with _patch_aiohttp(fake):
            result = asyncio.run(mp.check_worker_health(_config()))
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "timeout")

    def test_missing_worker_url_reports_config_not_network(self):
        with _patch_aiohttp(_FakeSession(_FakeResponse(200))):
            result = asyncio.run(mp.check_worker_health(_config(worker_url="")))
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "worker_url_not_configured")

    def test_health_records_latency(self):
        fake = _FakeSession(_FakeResponse(200))
        with _patch_aiohttp(fake):
            result = asyncio.run(mp.check_worker_health(_config()))
        self.assertGreaterEqual(result.latency_ms, 0.0)
        self.assertIn("latency_ms=", result.log_fields())

    def test_worker_reachable_does_not_imply_realtime_available(self):
        # The whole point of the two-URL split: a green worker probe says
        # nothing about live video, because the Gateway is a different endpoint.
        cfg = _config(realtime_url="")
        fake = _FakeSession(_FakeResponse(200))
        with _patch_aiohttp(fake):
            result = asyncio.run(mp.check_worker_health(cfg))
        self.assertTrue(result.ok)
        self.assertFalse(cfg.realtime_ready)
        self.assertIs(cfg.gateway_state, mp.GatewayState.PENDING_DEPLOYMENT)


class RealtimeConnectionTests(unittest.TestCase):
    def test_connect_refuses_when_gateway_pending(self):
        conn = mp.MiniCPMORealtimeConnection(_config(realtime_url=""), session_id="s-1")
        with self.assertRaises(mp.MiniCPMOConnectionError) as ctx:
            asyncio.run(conn.connect())
        self.assertEqual(ctx.exception.reason, "realtime_gateway_not_configured")
        self.assertFalse(conn.connected)

    def test_connect_refuses_worker_url_without_touching_the_network(self):
        cfg = _config(realtime_url=WORKER_URL.replace("https://", "wss://"))
        conn = mp.MiniCPMORealtimeConnection(cfg, session_id="s-1")

        def _explode(*a, **k):
            raise AssertionError("must not open a socket to the worker URL")

        with mock.patch.object(mp, "aiohttp", types.SimpleNamespace(
            ClientSession=_explode, ClientTimeout=lambda **k: None
        )):
            with self.assertRaises(mp.MiniCPMOConnectionError) as ctx:
                asyncio.run(conn.connect())
        self.assertEqual(ctx.exception.reason, "realtime_url_is_worker_url")

    def test_pending_gateway_never_wakes_a_gpu(self):
        # A blank Gateway must short-circuit before any ClientSession exists —
        # otherwise a disabled feature would still be spinning up L40S capacity.
        def _explode(*a, **k):
            raise AssertionError("must not construct an aiohttp session")

        conn = mp.MiniCPMORealtimeConnection(_config(realtime_url=""), session_id="s-1")
        with mock.patch.object(mp, "aiohttp", types.SimpleNamespace(
            ClientSession=_explode, ClientTimeout=lambda **k: None
        )):
            with self.assertRaises(mp.MiniCPMOConnectionError):
                asyncio.run(conn.connect())

    def test_send_frame_before_connect_raises_typed_error(self):
        conn = mp.MiniCPMORealtimeConnection(_config(), session_id="s-1")
        with self.assertRaises(mp.MiniCPMOConnectionError) as ctx:
            asyncio.run(conn.send_frame("data:image/jpeg;base64,AAAA"))
        self.assertEqual(ctx.exception.reason, "not_connected")

    def test_aclose_is_idempotent_on_an_unconnected_connection(self):
        conn = mp.MiniCPMORealtimeConnection(_config(), session_id="s-1")

        async def scenario():
            await conn.aclose()
            await conn.aclose()

        asyncio.run(scenario())
        self.assertFalse(conn.connected)


if __name__ == "__main__":
    unittest.main()

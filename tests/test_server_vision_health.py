"""server.py: the vision startup probe and the /health provider snapshot.

Pins the operational rules:
  - the startup probe is DETACHED — an unreachable or cold Modal endpoint must
    never delay app startup, and /health (Railway's healthcheck path) must never
    make a network call,
  - /health reports provider state without leaking any infrastructure URL,
  - reaching the Modal worker is NOT reported as "ready" while the realtime
    Gateway is still pending. This is the rule that stops a green worker probe
    being mistaken for working live video.

The aiohttp boundary is faked, so no GPU, Modal, or network is involved.
"""

import os
import time
import types
import unittest
from unittest import mock

from vision_fixtures import WORKER_URL, FakeResponse, FakeSession, HangingSession, patch_aiohttp

# The exact variable set configured on the Railway backend service.
RAILWAY_ENV = {
    "MINICPMO_ENABLED": "true",
    "MINICPMO_WORKER_URL": WORKER_URL,
    "MINICPMO_REALTIME_URL": "",
    "MINICPMO_CONNECT_TIMEOUT_SECONDS": "120",
    "MINICPMO_SESSION_TIMEOUT_SECONDS": "900",
    "VISION_TARGET_FPS": "2",
    "VISION_MAX_QUEUE_SIZE": "4",
}


def _probe(env, session):
    """Run the app through startup with a faked network, return (/health json, timings)."""
    import minicpmo_provider as mp
    import server
    from fastapi.testclient import TestClient

    with mock.patch.dict(os.environ, env, clear=False), \
            patch_aiohttp(mp, session), \
            mock.patch.object(server, "_vision_provider_snapshot",
                              {"state": "unknown", "checked": False}):
        started = time.monotonic()
        with TestClient(server.app) as client:
            startup_seconds = time.monotonic() - started
            health_started = time.monotonic()
            first = client.get("/health")
            health_seconds = time.monotonic() - health_started
            # Let the detached probe settle, then read the recorded snapshot.
            for _ in range(50):
                body = client.get("/health").json()
                if body.get("vision", {}).get("checked"):
                    break
                time.sleep(0.02)
            return body, first, startup_seconds, health_seconds


class StartupProbeTests(unittest.TestCase):
    def test_unreachable_modal_never_delays_startup_or_health(self):
        # Railway fails a deploy whose healthcheck is slow, and a cold L40S can
        # take minutes — so neither startup nor /health may wait on Modal.
        body, first, startup_seconds, health_seconds = _probe(RAILWAY_ENV, HangingSession())
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "ok")
        self.assertLess(startup_seconds, 2.0, "startup blocked on the vision probe")
        self.assertLess(health_seconds, 0.5, "/health made a network call")
        self.assertFalse(body["vision"]["worker_reachable"])
        self.assertEqual(body["vision"]["state"], "degraded")

    def test_health_still_ok_when_vision_is_broken(self):
        # A broken vision provider must never turn Lucy's healthcheck red.
        body, first, _, _ = _probe(RAILWAY_ENV, HangingSession())
        self.assertEqual(first.json()["status"], "ok")

    def test_reachable_worker_is_not_reported_ready_while_gateway_pending(self):
        # THE rule: worker connectivity is not evidence that live video works.
        body, _, _, _ = _probe(RAILWAY_ENV, FakeSession(FakeResponse(200, "ok")))
        vision = body["vision"]
        self.assertTrue(vision["worker_reachable"])
        self.assertEqual(vision["gateway"], "pending_deployment")
        self.assertEqual(vision["gateway_reason"], "realtime_gateway_not_configured")
        self.assertEqual(vision["state"], "unavailable",
                         "a reachable worker must NOT read as ready")

    def test_railway_variable_set_parses_as_documented(self):
        # Guards the deployed configuration itself: these are the exact values
        # set on the Railway service.
        body, _, _, _ = _probe(RAILWAY_ENV, FakeSession(FakeResponse(200, "ok")))
        vision = body["vision"]
        self.assertTrue(vision["enabled"])
        self.assertTrue(vision["worker_configured"])
        self.assertEqual(vision["target_fps"], 2.0)
        self.assertEqual(vision["max_queue_size"], 4)

    def test_health_leaks_no_infrastructure_urls(self):
        # /health is unauthenticated; Modal endpoints are server-side config.
        _, first, _, _ = _probe(RAILWAY_ENV, FakeSession(FakeResponse(200, "ok")))
        raw = first.text
        self.assertNotIn("modal.run", raw)
        self.assertNotIn("MINICPMO", raw)
        self.assertNotIn(WORKER_URL, raw)

    def test_disabled_provider_reports_disabled_and_probes_nothing(self):
        def explode(*a, **k):
            raise AssertionError("must not touch the network when disabled")

        env = dict(RAILWAY_ENV, MINICPMO_ENABLED="false")
        body, first, _, _ = _probe(env, types.SimpleNamespace(get=explode, close=None))
        self.assertEqual(first.json()["status"], "ok")
        self.assertEqual(body["vision"]["state"], "disabled")
        self.assertIsNone(body["vision"]["worker_reachable"])


if __name__ == "__main__":
    unittest.main()

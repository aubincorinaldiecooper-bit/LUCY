"""Shared fixtures for the MiniCPM-o vision test files.

One home for the constants, config/env factories, and aiohttp-boundary fakes
that the vision tests share — previously copied per file, where the copies had
already begun to drift (get vs post fakes, closed-tracking vs not).

Importable by bare name because pytest, with no __init__.py in tests/, inserts
this directory onto sys.path before importing a test module — the same
mechanism that lets the test files import the top-level product modules.
Not named test_*.py, so pytest never collects it.
"""

import asyncio
import types
from unittest import mock

import minicpmo_provider as mp

# The deployed Modal worker (health/connectivity only — never a realtime
# endpoint). Kept in ONE place: it encodes an account/app name, so when it
# rotates, this is the only line that changes.
WORKER_URL = "https://aubincorinaldiecooper--minicpmo45-realtime-minicpmo-worker.modal.run"
# What MINICPMO_REALTIME_URL will conceptually hold once the Gateway deploys.
GATEWAY_URL = "wss://minicpmo-gateway.example.com/v1/realtime?mode=video"
FRAME = "data:image/jpeg;base64,AAAA"


def minicpmo_env(**overrides):
    """Baseline MiniCPM-o environment dict; None-valued overrides are dropped."""
    env = {"MINICPMO_ENABLED": "true", "MINICPMO_WORKER_URL": WORKER_URL}
    env.update({k: v for k, v in overrides.items() if v is not None})
    return env


def make_minicpmo_config(**overrides):
    """A MiniCPMOConfig with every field defaulted; override what the test pins.

    The single place that must grow when the frozen dataclass gains a field.
    """
    kwargs = dict(
        enabled=True,
        worker_url=WORKER_URL,
        realtime_url="",
        gateway_token="",
        connect_timeout_seconds=120.0,
        session_timeout_seconds=900.0,
        health_timeout_seconds=10.0,
        target_fps=2.0,
        max_queue_size=4,
        max_connect_attempts=2,
        max_total_connects=6,
        reconnect_backoff_seconds=5.0,
        min_state_update_interval_seconds=4.0,
    )
    kwargs.update(overrides)
    return mp.MiniCPMOConfig(**kwargs)


class FakeResponse:
    """aiohttp response fake: status + optional text/json bodies."""

    def __init__(self, status=200, text_body="", json_body=None):
        self.status = status
        self._text_body = text_body
        self._json_body = json_body

    async def text(self):
        return self._text_body

    async def json(self):
        return self._json_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """aiohttp session fake for GET-based probes, with close tracking."""

    def __init__(self, response=None, raise_on_get=None):
        self._response = response if response is not None else FakeResponse()
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


class HangingSession(FakeSession):
    """Modal unreachable: every probe runs out its timeout."""

    def __init__(self):
        super().__init__(raise_on_get=asyncio.TimeoutError())


def patch_aiohttp(module, fake_session):
    """Replace ``module.aiohttp`` with a fake serving ``fake_session``.

    The target modules import aiohttp at module level, so the fake must replace
    the name bound in that module's namespace — patching sys.modules would be a
    no-op against the already-bound reference.
    """
    fake = types.SimpleNamespace(
        ClientSession=lambda timeout=None: fake_session,
        ClientTimeout=lambda **k: None,
    )
    return mock.patch.object(module, "aiohttp", fake)

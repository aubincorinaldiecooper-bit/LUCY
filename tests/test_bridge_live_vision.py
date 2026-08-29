"""ArcheRealtimeSession <-> MiniCPM-o live vision wiring.

Pins the integration rules at the seam:
  - live vision is off unless MINICPMO_ENABLED=true, and when off the bridge
    behaves EXACTLY as before (no extra objects, no extra calls),
  - camera frames fan out to the live session without disturbing the existing
    OpenRouter vision path,
  - visual context reaches Arche only through the session-instructions path
    (never a conversation item), so it can't interrupt or duplicate a turn,
  - when live vision is unavailable/degraded the bridge falls back to the
    OpenRouter summary, and failing that to no visual note at all — the voice
    conversation is unaffected either way,
  - closing the session releases the vision session,
  - the vision session carries the Lucy session id.
"""

import asyncio
import unittest
from unittest import mock

import inworld_realtime_bridge as bridge
import minicpmo_provider as mp
import vision_session as vsn

WORKER_URL = "https://aubincorinaldiecooper--minicpmo45-realtime-minicpmo-worker.modal.run"
FRAME = "data:image/jpeg;base64,AAAA"


class _NullTransport(bridge.ArcheTransport):
    def __init__(self):
        self.closed = False
        self.session = None

    def bind(self, session):
        self.session = session

    async def start(self):
        return

    async def write_output_pcm(self, pcm):
        return 0

    async def emit_event(self, payload):
        return

    async def aclose(self):
        self.closed = True


def _settings(**overrides):
    env = {
        "INWORLD_API_KEY": "k",
        "INWORLD_REALTIME_SESSION_ID": "lucy-session-xyz",
    }
    env.update(overrides)
    with mock.patch.dict("os.environ", env, clear=True):
        return bridge.load_inworld_realtime_settings(instructions="BASE PROMPT")


def _minicpmo_env(**overrides):
    env = {"MINICPMO_ENABLED": "true", "MINICPMO_WORKER_URL": WORKER_URL}
    env.update(overrides)
    return env


def _make_session(env=None):
    settings = _settings()
    with mock.patch.dict("os.environ", env or {}, clear=True):
        return bridge.ArcheRealtimeSession(settings, _NullTransport())


class DisabledByDefaultTests(unittest.TestCase):
    def test_no_vision_session_when_minicpmo_disabled(self):
        session = _make_session({})
        self.assertIsNone(session._live_vision)
        self.assertFalse(session.live_vision_enabled)
        self.assertIsNone(session.live_vision_status)

    def test_start_and_stop_are_no_ops_when_disabled(self):
        session = _make_session({})
        self.assertFalse(session.start_live_vision())
        asyncio.run(session.stop_live_vision())  # must not raise

    def test_frames_still_reach_the_openrouter_path_when_disabled(self):
        # The pre-existing behaviour must be untouched with MiniCPM-o off.
        session = _make_session({})
        session.set_video_frame(FRAME)
        self.assertEqual(session._latest_video_frame_data_url, FRAME)

    def test_frame_sample_interval_unchanged_when_live_vision_off(self):
        session = _make_session({})
        self.assertEqual(
            session.frame_sample_interval_seconds(),
            session.vision_config.frame_sample_interval_seconds,
        )


class SessionAssociationTests(unittest.TestCase):
    def test_vision_session_carries_the_lucy_session_id(self):
        session = _make_session(_minicpmo_env())
        self.assertIsNotNone(session._live_vision)
        self.assertEqual(session._live_vision.session_id, session.settings.session_id)
        self.assertEqual(session._live_vision.session_id, "lucy-session-xyz")

    def test_two_sessions_get_independent_vision_sessions(self):
        # No registry, no lookup by id: a vision session is owned by exactly one
        # conversation, so video cannot be attached to another user's session.
        a = _make_session(_minicpmo_env())
        b = _make_session(_minicpmo_env())
        self.assertIsNot(a._live_vision, b._live_vision)
        self.assertIsNot(a._live_vision.queue, b._live_vision.queue)

    def test_pending_gateway_leaves_the_session_unavailable(self):
        session = _make_session(_minicpmo_env())
        self.assertFalse(session.start_live_vision())
        status = session.live_vision_status
        self.assertEqual(status["state"], "unavailable")
        self.assertEqual(status["gateway"], "pending_deployment")

    def test_worker_url_is_not_used_as_the_realtime_url(self):
        session = _make_session(_minicpmo_env())
        cfg = session._live_vision.config
        self.assertEqual(cfg.worker_url, WORKER_URL)
        self.assertEqual(cfg.realtime_url, "")
        self.assertNotEqual(cfg.realtime_url, cfg.worker_url)


class FrameFanoutTests(unittest.TestCase):
    def test_set_video_frame_offers_to_the_live_session(self):
        session = _make_session(_minicpmo_env())
        offered = []
        session._live_vision.offer_frame = lambda frame: offered.append(frame) or True
        session.set_video_frame(FRAME)
        self.assertEqual(offered, [FRAME])
        # And the OpenRouter path still holds the frame.
        self.assertEqual(session._latest_video_frame_data_url, FRAME)

    def test_non_image_payloads_are_rejected_before_fanout(self):
        session = _make_session(_minicpmo_env())
        offered = []
        session._live_vision.offer_frame = lambda frame: offered.append(frame) or True
        session.set_video_frame("http://example.com/x.jpg")
        session.set_video_frame(None)
        self.assertEqual(offered, [])

    def test_sample_interval_speeds_up_for_the_live_provider(self):
        # A live provider that wants frames pulls the sampler up to its target
        # FPS (0.25s at 4 FPS), rather than the lazy 2s OpenRouter default.
        session = _make_session(_minicpmo_env(VISION_TARGET_FPS="4"))
        self.assertTrue(session._live_vision.wants_frames)
        self.assertAlmostEqual(session.frame_sample_interval_seconds(), 0.25)

    def test_sample_interval_reverts_when_the_provider_stops_wanting_frames(self):
        # Gateway pending -> the provider settles to unavailable, and the
        # sampler must fall back to the pre-existing OpenRouter cadence rather
        # than burning CPU encoding frames nothing will consume.
        session = _make_session(_minicpmo_env(VISION_TARGET_FPS="4"))
        session.start_live_vision()
        self.assertFalse(session._live_vision.wants_frames)
        self.assertEqual(
            session.frame_sample_interval_seconds(),
            session.vision_config.frame_sample_interval_seconds,
        )

    def test_camera_is_not_sampled_at_all_when_both_providers_are_off(self):
        session = _make_session({})
        self.assertFalse(session.wants_camera_frames)

    def test_camera_is_sampled_before_connect_completes(self):
        # The bug this guards: gating the sampler on a *completed* connect would
        # skip a user who joined with their camera already on, because the
        # transport subscribes existing tracks during the connect window.
        session = _make_session(_minicpmo_env(MINICPMO_REALTIME_URL="wss://gw.example.com/v1/realtime"))
        self.assertTrue(session.wants_camera_frames)
        self.assertFalse(session.live_vision_enabled, "not connected yet")

    def test_pending_gateway_stops_the_camera_being_touched(self):
        session = _make_session(_minicpmo_env())
        session.start_live_vision()
        self.assertFalse(session.wants_camera_frames)


class VisionSummarySelectionTests(unittest.TestCase):
    """Which provider's visual note reaches Arche, and the fallback order."""

    def test_openrouter_summary_used_when_no_live_state(self):
        session = _make_session(_minicpmo_env())
        session.latest_vision_summary = "a desk with a lamp"
        self.assertEqual(session._effective_vision_summary(), "a desk with a lamp")

    def test_live_state_wins_over_openrouter_summary(self):
        session = _make_session(_minicpmo_env())
        session.latest_vision_summary = "a desk with a lamp"
        session._live_vision.rolling.ingest(
            {"scene_summary": "User picked up a red book.", "objects": ["red book"]},
            now=1.0,
        )
        summary = session._effective_vision_summary()
        self.assertIn("red book", summary)

    def test_degraded_live_vision_falls_back_to_openrouter(self):
        # The mandatory degradation path: MiniCPM-o unavailable must not remove
        # the visual context Lucy already had.
        session = _make_session(_minicpmo_env())
        session.latest_vision_summary = "a desk with a lamp"
        session._live_vision._set_state(mp.VisionProviderState.DEGRADED, "gateway_disconnected")
        session._live_vision.rolling.clear()
        self.assertEqual(session._effective_vision_summary(), "a desk with a lamp")

    def test_no_visual_note_at_all_is_valid(self):
        session = _make_session(_minicpmo_env())
        self.assertIsNone(session._effective_vision_summary())

    def test_visual_context_reaches_instructions_never_a_conversation_item(self):
        session = _make_session(_minicpmo_env())
        session._live_vision.rolling.ingest(
            {"scene_summary": "User picked up a red book.", "objects": ["red book"]},
            now=1.0,
        )
        update = session._compose_instructions_update()
        # session.update carrying instructions — not conversation.item.create.
        self.assertEqual(update["type"], "session.update")
        self.assertIn("red book", update["session"]["instructions"])
        self.assertIn("BASE PROMPT", update["session"]["instructions"])

    def test_visual_note_never_tells_arche_she_has_a_camera(self):
        session = _make_session(_minicpmo_env())
        session._live_vision.rolling.ingest({"scene_summary": "A red book."}, now=1.0)
        instructions = session._compose_instructions_update()["session"]["instructions"]
        # Reuses the existing privacy-preserving perception note.
        self.assertIn("never mention", instructions.lower())


class LiveUpdateFlowTests(unittest.TestCase):
    def test_visual_update_triggers_an_instructions_refresh(self):
        session = _make_session(_minicpmo_env())
        reasons = []

        async def fake_send(reason):
            reasons.append(reason)
            return True

        session._send_instructions_update = fake_send
        state = vsn.RollingVisualState(min_update_interval_seconds=0.0)
        state.ingest({"scene_summary": "A red book."}, now=1.0)
        asyncio.run(session._on_live_visual_update(state.current))
        self.assertEqual(reasons, ["minicpmo_live_vision"])
        self.assertIsNotNone(session.latest_live_visual_state)

    def test_stop_live_vision_clears_state_and_refreshes_instructions(self):
        session = _make_session(_minicpmo_env())
        reasons = []

        async def fake_send(reason):
            reasons.append(reason)
            return True

        session._send_instructions_update = fake_send
        session._live_vision.rolling.ingest({"scene_summary": "A red book."}, now=1.0)

        asyncio.run(session.stop_live_vision(reason="camera_stopped"))

        self.assertIsNone(session.latest_live_visual_state)
        self.assertIsNone(session._effective_vision_summary())
        self.assertIn("live_vision_stopped", reasons)

    def test_start_live_vision_survives_a_provider_fault(self):
        session = _make_session(_minicpmo_env())

        def explode():
            raise RuntimeError("provider blew up")

        session._live_vision.start = explode
        self.assertFalse(session.start_live_vision())  # must not raise

    def test_stop_live_vision_survives_a_provider_fault(self):
        session = _make_session(_minicpmo_env())

        async def explode(reason="x"):
            raise RuntimeError("provider blew up")

        session._live_vision.stop = explode
        session._send_instructions_update = lambda reason: asyncio.sleep(0)
        asyncio.run(session.stop_live_vision())  # must not raise


class TeardownTests(unittest.TestCase):
    def test_aclose_releases_the_vision_session(self):
        session = _make_session(_minicpmo_env())
        closed = []

        async def fake_aclose():
            closed.append(True)

        session._live_vision.aclose = fake_aclose
        asyncio.run(session.aclose())
        self.assertEqual(closed, [True])

    def test_aclose_survives_a_vision_teardown_fault(self):
        session = _make_session(_minicpmo_env())

        async def explode():
            raise RuntimeError("teardown blew up")

        session._live_vision.aclose = explode
        asyncio.run(session.aclose())  # voice teardown must still complete
        self.assertTrue(session.transport.closed)

    def test_aclose_with_no_vision_session_is_unchanged(self):
        session = _make_session({})
        asyncio.run(session.aclose())
        self.assertTrue(session.transport.closed)


if __name__ == "__main__":
    unittest.main()

import asyncio
import os
import unittest
from unittest.mock import patch

import audiointeraction_shadow
from audiointeraction_shadow import (
    DECISION_KEEP_SILENCE,
    DECISION_TEXT_BEGIN,
    AudioInteractionShadow,
    build_shadow_from_env,
    classify_disagreement,
    parse_decision_message,
)


def make_shadow(**kwargs) -> AudioInteractionShadow:
    kwargs.setdefault("endpoint", "ws://sidecar.test/ws")
    kwargs.setdefault("timeout_ms", 200)
    kwargs.setdefault("debug_text", False)
    kwargs.setdefault("ws_factory", lambda: (_ for _ in ()).throw(RuntimeError("not used")))
    return AudioInteractionShadow(**kwargs)


class OffModeTests(unittest.TestCase):
    def test_mode_off_builds_nothing(self):
        with patch.dict(os.environ, {"AUDIOINTERACTION_MODE": "off", "AUDIOINTERACTION_ENDPOINT": "ws://x"}):
            self.assertIsNone(build_shadow_from_env())

    def test_mode_unset_defaults_to_off(self):
        env = dict(os.environ)
        env.pop("AUDIOINTERACTION_MODE", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(audiointeraction_shadow.audiointeraction_mode(), "off")
            self.assertIsNone(build_shadow_from_env())

    def test_unknown_mode_treated_as_off(self):
        with patch.dict(os.environ, {"AUDIOINTERACTION_MODE": "production"}):
            self.assertEqual(audiointeraction_shadow.audiointeraction_mode(), "off")

    def test_shadow_mode_without_endpoint_disabled(self):
        with patch.dict(os.environ, {"AUDIOINTERACTION_MODE": "shadow", "AUDIOINTERACTION_ENDPOINT": ""}):
            self.assertIsNone(build_shadow_from_env())

    def test_shadow_mode_with_endpoint_builds(self):
        with patch.dict(os.environ, {"AUDIOINTERACTION_MODE": "shadow", "AUDIOINTERACTION_ENDPOINT": "ws://sidecar:5002/ws"}):
            shadow = build_shadow_from_env(ws_factory=lambda: None)
            self.assertIsInstance(shadow, AudioInteractionShadow)


class DecisionParsingTests(unittest.TestCase):
    def test_parses_canonical_decisions(self):
        self.assertEqual(parse_decision_message('{"decision": "KEEP_SILENCE"}'), (DECISION_KEEP_SILENCE, "", None))
        self.assertEqual(parse_decision_message('{"decision": "TEXT_BEGIN", "text": "hey"}'), (DECISION_TEXT_BEGIN, "hey", None))

    def test_parses_model_state_variants(self):
        self.assertEqual(parse_decision_message('{"state": "speak"}')[0], DECISION_TEXT_BEGIN)
        self.assertEqual(parse_decision_message('{"state": "silent"}')[0], DECISION_KEEP_SILENCE)
        self.assertEqual(parse_decision_message('{"action": "respond"}')[0], DECISION_TEXT_BEGIN)
        self.assertEqual(parse_decision_message('{"decision": "<no need to response>"}')[0], DECISION_KEEP_SILENCE)

    def test_parses_infer_ms(self):
        decision, _, infer_ms = parse_decision_message('{"decision": "KEEP_SILENCE", "infer_ms": 41.5}')
        self.assertEqual(decision, DECISION_KEEP_SILENCE)
        self.assertEqual(infer_ms, 41.5)

    def test_parses_bytes_payload(self):
        self.assertEqual(parse_decision_message(b'{"decision": "TEXT_BEGIN"}')[0], DECISION_TEXT_BEGIN)

    def test_rejects_garbage(self):
        self.assertIsNone(parse_decision_message("not json"))
        self.assertIsNone(parse_decision_message('{"decision": "DANCE"}'))
        self.assertIsNone(parse_decision_message('["list"]'))
        self.assertIsNone(parse_decision_message('{"other": 1}'))

    def test_parse_errors_counted_not_raised(self):
        shadow = make_shadow()
        shadow.handle_sidecar_message("garbage")
        self.assertEqual(shadow.counters["parse_errors"], 1)
        self.assertIsNone(shadow.latest_decision)


class ComparisonTests(unittest.TestCase):
    def test_agreement_matrix(self):
        self.assertEqual(classify_disagreement(True, DECISION_TEXT_BEGIN), "agree")
        self.assertEqual(classify_disagreement(False, DECISION_KEEP_SILENCE), "agree")
        self.assertEqual(classify_disagreement(True, DECISION_KEEP_SILENCE), "lucy_committed_shadow_kept_silence")
        self.assertEqual(classify_disagreement(False, DECISION_TEXT_BEGIN), "lucy_held_shadow_would_speak")

    def test_compare_with_fresh_decision_agreeing(self):
        shadow = make_shadow()
        shadow.handle_sidecar_message('{"decision": "TEXT_BEGIN"}')
        result = shadow.compare_at_turn_commit(1, "COMMIT_NOW", True)
        self.assertEqual(result["agreement"], "agree")
        self.assertTrue(result["decision_before_commit"])
        self.assertEqual(shadow.counters["agreements"], 1)
        self.assertEqual(shadow.counters["disagreements"], 0)

    def test_compare_disagreement_lucy_committed_shadow_silent(self):
        shadow = make_shadow()
        shadow.handle_sidecar_message('{"decision": "KEEP_SILENCE"}')
        result = shadow.compare_at_turn_commit(2, "COMMIT_NOW", True)
        self.assertEqual(result["agreement"], "lucy_committed_shadow_kept_silence")
        self.assertEqual(shadow.counters["disagreement_lucy_committed_shadow_kept_silence"], 1)

    def test_compare_disagreement_lucy_held_shadow_speaks(self):
        shadow = make_shadow()
        shadow.handle_sidecar_message('{"decision": "TEXT_BEGIN"}')
        result = shadow.compare_at_turn_commit(3, "HOLD_FOR_CONTINUATION", False)
        self.assertEqual(result["agreement"], "lucy_held_shadow_would_speak")
        self.assertEqual(shadow.counters["disagreement_lucy_held_shadow_would_speak"], 1)

    def test_compare_without_any_decision(self):
        shadow = make_shadow()
        result = shadow.compare_at_turn_commit(4, "COMMIT_NOW", True)
        self.assertEqual(result["shadow_decision"], "none")
        self.assertEqual(result["agreement"], "no_decision")
        self.assertEqual(shadow.counters["no_decision_turns"], 1)

    def test_stale_decision_not_reused_for_next_turn(self):
        shadow = make_shadow()
        shadow.handle_sidecar_message('{"decision": "TEXT_BEGIN"}')
        first = shadow.compare_at_turn_commit(5, "COMMIT_NOW", True)
        self.assertEqual(first["agreement"], "agree")
        second = shadow.compare_at_turn_commit(6, "COMMIT_NOW", True)
        self.assertEqual(second["agreement"], "no_decision")

    def test_late_decision_logged_after_no_decision_turn(self):
        shadow = make_shadow()
        shadow.compare_at_turn_commit(7, "COMMIT_NOW", True)
        self.assertEqual(shadow.counters["late_decisions"], 0)
        shadow.handle_sidecar_message('{"decision": "KEEP_SILENCE"}')
        self.assertEqual(shadow.counters["late_decisions"], 1)
        shadow.handle_sidecar_message('{"decision": "KEEP_SILENCE"}')
        self.assertEqual(shadow.counters["late_decisions"], 1)


class FailureSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_with_unavailable_sidecar_never_raises(self):
        attempts = []

        async def failing_factory():
            attempts.append(1)
            raise ConnectionRefusedError("sidecar down")

        shadow = make_shadow(ws_factory=failing_factory)
        task = asyncio.get_running_loop().create_task(shadow.run())
        await asyncio.sleep(0.7)
        await shadow.aclose()
        task.cancel()
        self.assertGreaterEqual(shadow.counters["connect_errors"], 1)
        self.assertGreaterEqual(len(attempts), 1)

    async def test_feed_frame_never_raises(self):
        shadow = make_shadow(max_queue_frames=2)

        class Frame:
            data = b"\x00\x01"

        for _ in range(10):
            shadow.feed_frame(Frame())
        self.assertEqual(shadow.counters["frames_dropped"], 8)

        class BadFrame:
            @property
            def data(self):
                raise RuntimeError("no data")

        shadow.feed_frame(BadFrame())

    async def test_aclose_is_idempotent_and_logs_summary(self):
        shadow = make_shadow()
        await shadow.aclose()
        await shadow.aclose()


class SummaryTests(unittest.TestCase):
    def test_summary_counts_and_latency_fields(self):
        shadow = make_shadow()
        shadow.handle_sidecar_message('{"decision": "TEXT_BEGIN", "infer_ms": 100}')
        shadow.compare_at_turn_commit(1, "COMMIT_NOW", True)
        shadow.compare_at_turn_commit(2, "COMMIT_NOW", True)
        shadow.handle_sidecar_message('{"decision": "KEEP_SILENCE", "infer_ms": 300}')
        summary = shadow.summary()
        self.assertEqual(summary["comparisons"], 2)
        self.assertEqual(summary["decisions_total"], 2)
        self.assertEqual(summary["decisions_text_begin"], 1)
        self.assertEqual(summary["decisions_keep_silence"], 1)
        self.assertEqual(summary["no_decision_turns"], 1)
        self.assertEqual(summary["late_decisions"], 1)
        self.assertEqual(summary["decision_before_commit_count"], 1)
        self.assertIsNotNone(summary["avg_decision_age_seconds"])
        self.assertEqual(summary["avg_sidecar_infer_ms"], 200.0)
        self.assertEqual(summary["max_sidecar_infer_ms"], 300.0)
        shadow.log_summary()

    def test_debug_text_off_by_default(self):
        env = dict(os.environ)
        env.pop("AUDIOINTERACTION_DEBUG_TEXT", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(audiointeraction_shadow.audiointeraction_debug_text())

    def test_decision_text_redacted_unless_debug_enabled(self):
        shadow = make_shadow(debug_text=False)
        with self.assertLogs("audiointeraction_shadow", level="INFO") as captured:
            shadow.handle_sidecar_message('{"decision": "TEXT_BEGIN", "text": "my private words"}')
        joined = "\n".join(captured.output)
        self.assertNotIn("my private words", joined)
        self.assertIn("decision_text_length=16", joined)

        debug_shadow = make_shadow(debug_text=True)
        with self.assertLogs("audiointeraction_shadow", level="INFO") as captured:
            debug_shadow.handle_sidecar_message('{"decision": "TEXT_BEGIN", "text": "my private words"}')
        self.assertIn("my private words", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()

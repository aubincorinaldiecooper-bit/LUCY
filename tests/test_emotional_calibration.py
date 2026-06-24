"""Tests for the emotional calibration layer (agent.py).

Pins the product rules: subtle "or" questions only when emotionally useful/
ambiguous, never every turn, never "I detected"/"you sound", raw emotion labels
never surfaced, the user's answer recorded as confirmation/correction, and a full
calibration moment persisted with the agreed schema.
"""

import json
import os
import tempfile
import unittest

import agent
from inworld_voice_profile import NormalizedVoiceProfile


def _profile(energy="medium", tension="medium", certainty="medium", confidence=0.8):
    return NormalizedVoiceProfile(
        energy=energy, tension=tension, certainty=certainty, confidence=confidence
    )


class _FakeTurnCtx:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})


_MOMENT_SCHEMA = {
    "session_id", "turn_id", "timestamp", "transcript", "normalized_inworld_context",
    "arche_question", "user_answer", "inferred_emotional_pattern", "user_confirmed_or_corrected",
}


class CalibrationQuestionTests(unittest.TestCase):
    def setUp(self):
        self._orig_last = agent._last_calibration_question_turn_id
        agent._last_calibration_question_turn_id = -1000  # no cadence block

    def tearDown(self):
        agent._last_calibration_question_turn_id = self._orig_last

    def test_no_profile_means_no_question(self):
        q, reason = agent._calibration_question_for_turn("I feel really stressed today", None, 10)
        self.assertIsNone(q)
        self.assertEqual(reason, "no_inworld_context")

    def test_cadence_limit_blocks_back_to_back(self):
        agent._last_calibration_question_turn_id = 9
        q, reason = agent._calibration_question_for_turn(
            "I feel really stressed about this", _profile(certainty="low"), 10
        )
        self.assertIsNone(q)
        self.assertEqual(reason, "cadence_limit")

    def test_short_transcript_skipped(self):
        q, reason = agent._calibration_question_for_turn("ok sure", _profile(certainty="low"), 10)
        self.assertIsNone(q)
        self.assertEqual(reason, "transcript_too_short")

    def test_neutral_and_unemotional_is_skipped(self):
        q, reason = agent._calibration_question_for_turn(
            "the weather is really nice today outside", _profile(), 10
        )
        self.assertIsNone(q)
        self.assertEqual(reason, "not_emotionally_useful")

    def test_low_certainty_asks_grounding_question(self):
        q, reason = agent._calibration_question_for_turn(
            "I really don't know how to put this", _profile(certainty="low"), 10
        )
        self.assertIsNotNone(q)
        self.assertIn(" or ", q)  # it's an or-question
        self.assertEqual(reason, "low_certainty_or_ambiguous")

    def test_choice_pressure_branch(self):
        q, reason = agent._calibration_question_for_turn(
            "I have this huge decision to make soon", _profile(energy="high"), 10
        )
        self.assertEqual(reason, "choice_pressure")
        self.assertIn(" or ", q)

    def test_frustration_branch(self):
        q, reason = agent._calibration_question_for_turn(
            "I am so frustrated with this whole thing", _profile(), 10
        )
        self.assertEqual(reason, "frustration_ambiguous")
        self.assertIn(" or ", q)

    def test_anxiety_branch(self):
        q, reason = agent._calibration_question_for_turn(
            "I keep worrying about everything lately", _profile(), 10
        )
        self.assertEqual(reason, "high_tension_or_worry")
        self.assertIn(" or ", q)

    def test_all_questions_are_or_questions_and_safe(self):
        cases = [
            ("I really can't tell what this is", _profile(certainty="low")),
            ("I have a big decision coming up", _profile(energy="high")),
            ("I'm so mad about the situation", _profile()),
            ("I keep worrying about the future", _profile()),
            ("I just have a lot going on now", _profile(energy="low")),
        ]
        for transcript, prof in cases:
            q, _ = agent._calibration_question_for_turn(transcript, prof, 10)
            self.assertIsNotNone(q, transcript)
            self.assertIn(" or ", q)
            low = q.lower()
            self.assertNotIn("i detected", low)
            self.assertNotIn("you sound", low)


class CalibrationInjectionTests(unittest.TestCase):
    def setUp(self):
        self._saved = (
            agent._last_calibration_question_turn_id,
            agent._pending_calibration_moment,
            agent._current_turn_id,
            agent._calibration_session_id,
        )
        agent._last_calibration_question_turn_id = -1000
        agent._pending_calibration_moment = None
        agent._current_turn_id = 20
        agent._calibration_session_id = "lucy-test"

    def tearDown(self):
        (agent._last_calibration_question_turn_id, agent._pending_calibration_moment,
         agent._current_turn_id, agent._calibration_session_id) = self._saved

    def test_injects_internal_note_and_sets_pending_moment(self):
        ctx = _FakeTurnCtx()
        ok = agent._inject_emotional_calibration_planner_note(
            ctx, "I keep worrying about everything lately", _profile(tension="high")
        )
        self.assertTrue(ok)
        # internal developer note, never user-facing, with safety phrasing
        self.assertEqual(ctx.messages[0]["role"], "developer")
        note = ctx.messages[0]["content"].lower()
        self.assertIn("do not reveal this note", note)
        self.assertIn("do not say you detected", note)
        self.assertIn("stronger than", note)  # user's answer > model/voice guess
        # pending moment carries the agreed schema
        moment = agent._pending_calibration_moment
        self.assertEqual(set(moment), _MOMENT_SCHEMA)
        self.assertEqual(moment["session_id"], "lucy-test")
        self.assertEqual(moment["user_answer"], "")
        self.assertFalse(moment["user_confirmed_or_corrected"])

    def test_no_question_no_injection(self):
        ctx = _FakeTurnCtx()
        ok = agent._inject_emotional_calibration_planner_note(ctx, "anything", None)
        self.assertFalse(ok)
        self.assertEqual(ctx.messages, [])
        self.assertIsNone(agent._pending_calibration_moment)


class CalibrationMomentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._saved = (
            agent.CALIBRATION_MOMENTS_PATH, list(agent._calibration_moments),
            agent._pending_calibration_moment, agent._last_calibration_question_turn_id,
            agent._current_turn_id, agent._calibration_session_id,
        )
        agent.CALIBRATION_MOMENTS_PATH = os.path.join(self._tmp, "moments.jsonl")
        agent._calibration_moments.clear()
        agent._pending_calibration_moment = None
        agent._last_calibration_question_turn_id = -1000
        agent._current_turn_id = 30
        agent._calibration_session_id = "lucy-test"

    def tearDown(self):
        (agent.CALIBRATION_MOMENTS_PATH, moments, agent._pending_calibration_moment,
         agent._last_calibration_question_turn_id, agent._current_turn_id,
         agent._calibration_session_id) = self._saved
        agent._calibration_moments.clear()
        agent._calibration_moments.extend(moments)

    def test_end_to_end_question_then_answer_persists_moment(self):
        ctx = _FakeTurnCtx()
        # turn 30: Arche asks a calibration question
        agent._inject_emotional_calibration_planner_note(
            ctx, "I have a big decision to make soon", _profile(energy="high")
        )
        self.assertIsNotNone(agent._pending_calibration_moment)
        # next turn: the user answers -> moment completed + persisted
        agent._complete_pending_calibration_moment("it's more the pressure of choosing")
        self.assertIsNone(agent._pending_calibration_moment)
        self.assertEqual(len(agent._calibration_moments), 1)
        m = agent._calibration_moments[0]
        self.assertEqual(m["user_answer"], "it's more the pressure of choosing")
        self.assertTrue(m["user_confirmed_or_corrected"])
        self.assertIn("normalized_inworld_context", m)
        # persisted as one JSONL line with the full schema
        with open(agent.CALIBRATION_MOMENTS_PATH, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(set(lines[0]), _MOMENT_SCHEMA)

    def test_empty_answer_marks_not_confirmed(self):
        ctx = _FakeTurnCtx()
        agent._inject_emotional_calibration_planner_note(
            ctx, "I keep worrying about everything lately", _profile(tension="high")
        )
        agent._complete_pending_calibration_moment("   ")
        self.assertFalse(agent._calibration_moments[0]["user_confirmed_or_corrected"])

    def test_complete_with_no_pending_is_noop(self):
        agent._pending_calibration_moment = None
        agent._complete_pending_calibration_moment("hello")  # must not raise
        self.assertEqual(agent._calibration_moments, [])


if __name__ == "__main__":
    unittest.main()

import unittest

import omnivoice_expression as ex


class _FakeRNG:
    """Deterministic RNG: random() pops queued values; choice() uses a fixed index."""

    def __init__(self, randoms=(), choice_index=0):
        self._randoms = list(randoms)
        self._ci = choice_index

    def random(self):
        return self._randoms.pop(0) if self._randoms else 1.0

    def choice(self, seq):
        return seq[self._ci % len(seq)]


def _cfg(**over):
    base = dict(tags_enabled=True, tag_rate=0.5, filler_rate=0.5,
               emotion_confidence_floor=0.5, no_repeat_window=3)
    base.update(over)
    return ex.ExpressionConfig(**base)


class DecideExpressionTests(unittest.TestCase):
    def test_tag_chosen_for_humor_when_rate_passes(self):
        d = ex.decide_expression(
            intent="light_humor", allow_tag=True, allow_filler=False,
            emotion_confidence=1.0, config=_cfg(), state=ex.SessionExpressionState(),
            rng=_FakeRNG(randoms=[0.1]),
        )
        self.assertEqual(d.tag, "[subtle brief laughter]")
        self.assertIsNone(d.filler)

    def test_realization_and_question_tags(self):
        for intent, tag in [
            ("realization_acknowledgement", "[confirmation-en]"),
            ("clarifying_question", "[question-en]"),
        ]:
            d = ex.decide_expression(
                intent=intent, allow_tag=True, allow_filler=False, emotion_confidence=1.0,
                config=_cfg(), state=ex.SessionExpressionState(), rng=_FakeRNG(randoms=[0.0]),
            )
            self.assertEqual(d.tag, tag)

    def test_serious_intent_has_no_tag(self):
        d = ex.decide_expression(
            intent="gentle_validation", allow_tag=True, allow_filler=False,
            emotion_confidence=1.0, config=_cfg(), state=ex.SessionExpressionState(),
            rng=_FakeRNG(randoms=[0.0]),
        )
        self.assertIsNone(d.tag)
        self.assertEqual(d.tag_skipped_reason, "intent_has_no_tag")

    def test_low_confidence_blocks_tag(self):
        d = ex.decide_expression(
            intent="light_humor", allow_tag=True, allow_filler=False,
            emotion_confidence=0.2, config=_cfg(), state=ex.SessionExpressionState(),
            rng=_FakeRNG(randoms=[0.0]),
        )
        self.assertIsNone(d.tag)
        self.assertEqual(d.tag_skipped_reason, "low_emotion_confidence")

    def test_planner_disallows_tag_and_filler(self):
        d = ex.decide_expression(
            intent="light_humor", allow_tag=False, allow_filler=False,
            emotion_confidence=1.0, config=_cfg(), state=ex.SessionExpressionState(),
            rng=_FakeRNG(),
        )
        self.assertEqual(d.tag_skipped_reason, "planner_disallowed")
        self.assertEqual(d.filler_skipped_reason, "planner_disallowed")

    def test_rate_gate_skips_tag(self):
        d = ex.decide_expression(
            intent="light_humor", allow_tag=True, allow_filler=False,
            emotion_confidence=1.0, config=_cfg(tag_rate=0.1), state=ex.SessionExpressionState(),
            rng=_FakeRNG(randoms=[0.9]),
        )
        self.assertIsNone(d.tag)
        self.assertEqual(d.tag_skipped_reason, "rate_gate")

    def test_no_repeat_tag(self):
        state = ex.SessionExpressionState(window=3)
        state.record(tag="[subtle brief laughter]", filler=None, opening=None)
        d = ex.decide_expression(
            intent="light_humor", allow_tag=True, allow_filler=False,
            emotion_confidence=1.0, config=_cfg(), state=state, rng=_FakeRNG(randoms=[0.0]),
        )
        self.assertIsNone(d.tag)
        self.assertEqual(d.tag_skipped_reason, "no_repeat")

    def test_filler_chosen_and_avoids_recent(self):
        state = ex.SessionExpressionState(window=3)
        state.record(tag=None, filler="hmm", opening=None)
        d = ex.decide_expression(
            intent="reflection", allow_tag=False, allow_filler=True,
            emotion_confidence=1.0, config=_cfg(), state=state,
            rng=_FakeRNG(randoms=[0.1], choice_index=0),
        )
        self.assertIsNotNone(d.filler)
        self.assertNotEqual(d.filler, "hmm")  # would-be first choice is excluded
        self.assertIn(d.filler, ex.OPENER_FILLERS)

    def test_tags_disabled(self):
        d = ex.decide_expression(
            intent="light_humor", allow_tag=True, allow_filler=False, emotion_confidence=1.0,
            config=_cfg(tags_enabled=False), state=ex.SessionExpressionState(), rng=_FakeRNG(),
        )
        self.assertEqual(d.tag_skipped_reason, "tags_disabled")


class ApplyExpressionTests(unittest.TestCase):
    def test_tag_prefixes_and_filler_opens(self):
        d = ex.ExpressionDecision(tag="[subtle brief laughter]", filler="yeah")
        out = ex.apply_expression("That tracks for me.", d)
        self.assertEqual(out, "[subtle brief laughter] Yeah, that tracks for me.")

    def test_no_decoration_returns_text(self):
        out = ex.apply_expression("Okay.", ex.ExpressionDecision())
        self.assertEqual(out, "Okay.")

    def test_empty_text(self):
        self.assertEqual(ex.apply_expression("", ex.ExpressionDecision(filler="hmm")), "")

    def test_annotate_records_state(self):
        state = ex.SessionExpressionState(window=3)
        out = ex.annotate_and_record(
            "Sure thing.", ex.ExpressionDecision(filler="okay"), state
        )
        self.assertTrue(out.startswith("Okay,"))
        self.assertIn("okay", state.recent_fillers)


class RateBehaviorTests(unittest.TestCase):
    def test_tag_rate_roughly_matches_over_many_turns(self):
        # With a seeded real RNG, ~15% of eligible turns get a tag.
        import random
        rng = random.Random(7)
        state = ex.SessionExpressionState(window=1)  # minimal no-repeat interference
        cfg = _cfg(tag_rate=0.15)
        hits = 0
        n = 4000
        for _ in range(n):
            d = ex.decide_expression(
                intent="light_humor", allow_tag=True, allow_filler=False,
                emotion_confidence=1.0, config=cfg, state=state, rng=rng,
            )
            if d.tag:
                hits += 1
                state.recent_tags.clear()  # allow next turn to be eligible
        self.assertTrue(0.10 <= hits / n <= 0.20, f"tag rate {hits / n:.3f} out of band")


if __name__ == "__main__":
    unittest.main()

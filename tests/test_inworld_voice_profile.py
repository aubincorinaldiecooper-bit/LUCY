import unittest

import inworld_voice_profile as ivp


def _profile(emotion=None, pitch=None, vocal_style=None, accent=None):
    p = {}
    if emotion:
        p["emotion"] = [{"label": e, "confidence": c} for e, c in emotion]
    if pitch:
        p["pitch"] = [{"label": l, "confidence": c} for l, c in pitch]
    if vocal_style:
        p["vocalStyle"] = [{"label": l, "confidence": c} for l, c in vocal_style]
    if accent:
        p["accent"] = [{"label": l, "confidence": c} for l, c in accent]
    return p


class NormalizeTests(unittest.TestCase):
    def test_empty_is_neutral(self):
        n = ivp.normalize_voice_profile(None)
        self.assertEqual((n.energy, n.tension, n.certainty), ("medium", "medium", "medium"))
        self.assertEqual(n.emotion_confidence, 0.0)
        self.assertIn("confidence", n.to_dict())
        self.assertNotIn("emotion_confidence", n.to_dict())

    def test_high_confidence_emotion_drives_dims(self):
        n = ivp.normalize_voice_profile(
            _profile(emotion=[("angry", 0.9), ("sad", 0.2)], pitch=[("high", 0.8)]),
        )
        self.assertEqual(n.energy, "high")
        self.assertEqual(n.tension, "high")
        self.assertEqual(n.emotion_confidence, 0.9)
        self.assertEqual(n.pitch, "high")

    def test_low_confidence_emotion_collapses_to_neutral(self):
        n = ivp.normalize_voice_profile(
            _profile(emotion=[("angry", 0.2)]), emotion_confidence_floor=0.5
        )
        self.assertEqual((n.energy, n.tension, n.certainty), ("medium", "medium", "medium"))
        # but the raw confidence is still reported
        self.assertEqual(n.emotion_confidence, 0.2)

    def test_top_label_is_highest_confidence(self):
        n = ivp.normalize_voice_profile(
            _profile(emotion=[("sad", 0.4), ("calm", 0.85)])
        )
        # calm wins -> low energy/tension
        self.assertEqual(n.energy, "low")
        self.assertEqual(n.tension, "low")

    def test_vocal_style_adjusts_certainty(self):
        whisper = ivp.normalize_voice_profile(
            _profile(emotion=[("calm", 0.9)], vocal_style=[("whispering", 0.9)])
        )
        self.assertEqual(whisper.certainty, "low")
        self.assertEqual(whisper.vocal_style, "whispering")

    def test_snake_case_and_nested_result(self):
        msg = {"result": {"voice_profile": _profile(emotion=[("happy", 0.9)])}}
        n = ivp.normalize_from_message(msg)
        self.assertEqual(n.energy, "high")

    def test_planner_summary_never_contains_raw_emotion(self):
        n = ivp.normalize_voice_profile(
            _profile(emotion=[("sad", 0.95)], pitch=[("low", 0.9)], vocal_style=[("crying", 0.8)])
        )
        summary = n.planner_summary()
        self.assertNotIn("sad", summary)
        self.assertIn("energy", summary)
        self.assertIn("pitch low", summary)


if __name__ == "__main__":
    unittest.main()

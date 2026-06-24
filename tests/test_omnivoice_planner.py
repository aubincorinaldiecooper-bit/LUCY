import json
import unittest

import omnivoice_planner as pl


class ParsePlannerOutputTests(unittest.TestCase):
    def test_parses_full_json(self):
        raw = json.dumps(
            {
                "response_text": "That makes sense.",
                "response_intent": "realization_acknowledgement",
                "desired_delivery": {
                    "warmth": "high", "pace": "slow", "energy": "low",
                    "allow_micro_filler": False, "allow_expressive_tag": True,
                },
            }
        )
        out = pl.parse_planner_output(raw)
        self.assertTrue(out.parsed_as_json)
        self.assertEqual(out.response_text, "That makes sense.")
        self.assertEqual(out.response_intent, "realization_acknowledgement")
        self.assertEqual(out.desired_delivery.warmth, "high")
        self.assertFalse(out.desired_delivery.allow_micro_filler)

    def test_parses_json_in_code_fence(self):
        raw = "```json\n{\"response_text\": \"Hi there.\"}\n```"
        out = pl.parse_planner_output(raw)
        self.assertTrue(out.parsed_as_json)
        self.assertEqual(out.response_text, "Hi there.")

    def test_plain_text_falls_back_to_response_text(self):
        out = pl.parse_planner_output("Just a normal reply.")
        self.assertFalse(out.parsed_as_json)
        self.assertEqual(out.response_text, "Just a normal reply.")
        self.assertEqual(out.response_intent, "reflection")

    def test_malformed_json_falls_back(self):
        out = pl.parse_planner_output('{"response_text": "oops"')  # missing brace
        self.assertFalse(out.parsed_as_json)
        self.assertEqual(out.response_text, '{"response_text": "oops"')

    def test_invalid_enums_coerced_to_defaults(self):
        raw = json.dumps(
            {
                "response_text": "ok",
                "response_intent": "not_a_real_intent",
                "desired_delivery": {"warmth": "scorching", "pace": "warp"},
            }
        )
        out = pl.parse_planner_output(raw)
        self.assertEqual(out.response_intent, "reflection")
        self.assertEqual(out.desired_delivery.warmth, "medium")
        self.assertEqual(out.desired_delivery.pace, "normal")

    def test_empty_response_text_falls_back(self):
        out = pl.parse_planner_output(json.dumps({"response_text": "  "}))
        self.assertFalse(out.parsed_as_json)


class BuildPlannerInstructionTests(unittest.TestCase):
    def test_includes_schema_language_and_recents(self):
        out = pl.build_planner_instruction(
            language="Yoruba",
            voice_preset_name="Maya",
            recent_tags=["[confirmation-en]"],
            recent_fillers=["hmm", "yeah"],
            inworld_summary="energy low, tension medium",
        )
        self.assertIn("response_text", out)
        self.assertIn("response_intent", out)
        self.assertIn("written in Yoruba", out)
        self.assertIn("Maya", out)
        self.assertIn("hmm", out)
        self.assertIn("never mention it", out)  # Inworld safety phrasing

    def test_minimal_inputs(self):
        out = pl.build_planner_instruction(language="English")
        self.assertIn("written in English", out)
        self.assertNotIn("Vocal context", out)


class ModelRoutingTests(unittest.TestCase):
    def test_parse_language_model_map(self):
        m = pl.parse_language_model_map("es=openai/gpt-4o, yo=google/gemini-2.5-pro,bad,=x,y=")
        self.assertEqual(m, {"es": "openai/gpt-4o", "yo": "google/gemini-2.5-pro"})

    def test_select_default_for_base_language(self):
        model, reason = pl.select_llm_model("en", default_model="flash-lite")
        self.assertEqual((model, reason), ("flash-lite", "default"))

    def test_select_language_map_wins(self):
        model, reason = pl.select_llm_model(
            "yo", default_model="flash-lite",
            language_model_map={"yo": "gemini-pro"}, non_english_model="gpt-4o",
        )
        self.assertEqual((model, reason), ("gemini-pro", "language_map"))

    def test_select_non_english_override(self):
        model, reason = pl.select_llm_model(
            "ig", default_model="flash-lite", non_english_model="gpt-4o",
        )
        self.assertEqual((model, reason), ("gpt-4o", "non_english_override"))

    def test_base_language_ignores_non_english_override(self):
        model, reason = pl.select_llm_model(
            "en", default_model="flash-lite", non_english_model="gpt-4o",
        )
        self.assertEqual((model, reason), ("flash-lite", "default"))


if __name__ == "__main__":
    unittest.main()

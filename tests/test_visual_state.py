"""visual_state.py: parsing provider output into a compact rolling state.

Pins the product rules:
  - a live stream must NOT become a stream of prompt updates: re-observing the
    same scene produces no notable_change, so nothing is published,
  - a materially changed scene ("user picked up a red book") does publish,
  - every field is bounded, so a chatty model can't balloon the prompt,
  - malformed provider messages are ignored, never raised — they arrive on the
    vision receive loop, where an exception would kill the session,
  - clearing the state genuinely removes the visual note (camera off).
"""

import unittest

import visual_state as vs


class ParsingTests(unittest.TestCase):
    def test_parses_the_documented_shape(self):
        payload = {
            "scene_summary": "User picked up a red book.",
            "people": ["one adult"],
            "objects": ["red book", "desk"],
            "actions": ["picked up"],
            "visible_text": [],
            "notable_change": "User picked up a red book.",
            "confidence": 0.82,
        }
        state = vs.visual_state_from_payload(payload, previous=None, timestamp=10.0)
        self.assertEqual(state.scene_summary, "User picked up a red book.")
        self.assertEqual(state.objects, ("red book", "desk"))
        self.assertEqual(state.actions, ("picked up",))
        self.assertEqual(state.people, ("one adult",))
        self.assertEqual(state.confidence, 0.82)
        self.assertEqual(state.notable_change, "User picked up a red book.")
        self.assertEqual(state.timestamp, 10.0)

    def test_as_dict_matches_the_agreed_schema(self):
        state = vs.visual_state_from_payload(
            {"scene_summary": "A desk.", "objects": ["mug"]}, previous=None, timestamp=1.0
        )
        self.assertEqual(
            set(state.as_dict()),
            {"timestamp", "scene_summary", "people", "objects", "actions",
             "visible_text", "notable_change", "confidence"},
        )

    def test_parses_plain_prose(self):
        state = vs.visual_state_from_payload(
            "A person sitting at a desk.", previous=None, timestamp=1.0
        )
        self.assertEqual(state.scene_summary, "A person sitting at a desk.")

    def test_parses_json_string_and_fenced_json(self):
        for raw in (
            '{"scene_summary": "A red book on a desk."}',
            '```json\n{"scene_summary": "A red book on a desk."}\n```',
            '```\n{"scene_summary": "A red book on a desk."}\n```',
        ):
            state = vs.visual_state_from_payload(raw, previous=None, timestamp=1.0)
            self.assertEqual(state.scene_summary, "A red book on a desk.", raw)

    def test_parses_bytes_and_envelopes(self):
        state = vs.visual_state_from_payload(
            b'{"visual_state": {"scene_summary": "A lamp."}}', previous=None, timestamp=1.0
        )
        self.assertEqual(state.scene_summary, "A lamp.")

    def test_alternate_field_names(self):
        for key in ("summary", "description", "text", "content"):
            state = vs.visual_state_from_payload({key: "A chair."}, previous=None, timestamp=1.0)
            self.assertEqual(state.scene_summary, "A chair.", key)

    def test_object_lists_of_dicts_are_flattened(self):
        state = vs.visual_state_from_payload(
            {"scene_summary": "x", "objects": [{"label": "red book"}, {"name": "mug"}]},
            previous=None, timestamp=1.0,
        )
        self.assertEqual(state.objects, ("red book", "mug"))

    def test_comma_string_lists_are_accepted(self):
        state = vs.visual_state_from_payload(
            {"scene_summary": "x", "objects": "red book, mug, mug"},
            previous=None, timestamp=1.0,
        )
        self.assertEqual(state.objects, ("red book", "mug"))  # deduped

    def test_fields_are_bounded(self):
        state = vs.visual_state_from_payload(
            {
                "scene_summary": "z" * 5000,
                "objects": [f"object-{i}" for i in range(100)],
            },
            previous=None, timestamp=1.0,
        )
        self.assertLessEqual(len(state.scene_summary), vs.MAX_SUMMARY_CHARS)
        self.assertLessEqual(len(state.objects), vs.MAX_LIST_ITEMS)

    def test_confidence_is_clamped_and_bad_values_dropped(self):
        def conf(value):
            return vs.visual_state_from_payload(
                {"scene_summary": "x", "confidence": value}, previous=None, timestamp=1.0
            ).confidence

        self.assertEqual(conf(1.7), 1.0)
        self.assertEqual(conf(-2), 0.0)
        self.assertIsNone(conf("high"))
        self.assertIsNone(conf(None))
        self.assertIsNone(conf(float("nan")))


class MalformedMessageTests(unittest.TestCase):
    """A bad message must yield None, never an exception."""

    def test_unusable_payloads_return_none(self):
        for payload in (
            None, "", "   ", {}, [], 42, 3.5, True,
            {"type": "pong"},                      # heartbeat
            {"type": "ack", "ok": True},           # lifecycle ack
            '{"broken": ',                          # truncated JSON -> prose, but empty of fields
            b"", [1, 2, 3],
            {"scene_summary": ""},                 # present but empty
            {"objects": []},                       # present but empty
        ):
            self.assertIsNone(
                vs.visual_state_from_payload(payload, previous=None, timestamp=1.0),
                repr(payload),
            )

    def test_malformed_payloads_never_raise(self):
        class Hostile:
            def __str__(self):
                raise RuntimeError("nope")

        # Exotic shapes are ignored rather than propagated to the receive loop.
        for payload in ({"objects": {"a": 1}}, {"confidence": object()}, set()):
            try:
                vs.visual_state_from_payload(payload, previous=None, timestamp=1.0)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"raised on {payload!r}: {exc}")

    def test_rolling_ingest_of_garbage_leaves_state_untouched(self):
        rolling = vs.RollingVisualState(min_update_interval_seconds=0.0)
        self.assertTrue(rolling.ingest({"scene_summary": "A red book."}, now=1.0))
        good = rolling.current
        self.assertFalse(rolling.ingest({"type": "pong"}, now=2.0))
        # A heartbeat must not clobber a good read with an empty one.
        self.assertIs(rolling.current, good)


class NotableChangeTests(unittest.TestCase):
    def test_first_observation_publishes(self):
        rolling = vs.RollingVisualState(min_update_interval_seconds=4.0)
        self.assertTrue(rolling.ingest({"scene_summary": "A person at a desk."}, now=1.0))
        self.assertEqual(rolling.updates_published, 1)

    def test_static_scene_does_not_publish_again(self):
        # The core anti-narration rule: 2 FPS on an unchanging room must not
        # push an instructions update per frame.
        rolling = vs.RollingVisualState(min_update_interval_seconds=0.0)
        payload = {"scene_summary": "A person at a desk.", "objects": ["desk"]}
        self.assertTrue(rolling.ingest(payload, now=1.0))
        for i in range(20):
            self.assertFalse(rolling.ingest(dict(payload), now=2.0 + i * 0.5))
        self.assertEqual(rolling.updates_published, 1)
        self.assertEqual(rolling.updates_ingested, 21)

    def test_trivial_rewording_is_not_a_change(self):
        rolling = vs.RollingVisualState(min_update_interval_seconds=0.0)
        rolling.ingest({"scene_summary": "A person at a desk."}, now=1.0)
        self.assertFalse(rolling.ingest({"scene_summary": "a person at a desk"}, now=2.0))

    def test_new_object_publishes(self):
        rolling = vs.RollingVisualState(min_update_interval_seconds=0.0)
        rolling.ingest({"scene_summary": "A person at a desk.", "objects": ["desk"]}, now=1.0)
        published = rolling.ingest(
            {"scene_summary": "A person holding a book.",
             "objects": ["desk", "red book"], "actions": ["picked up"]},
            now=2.0,
        )
        self.assertTrue(published)
        self.assertIsNotNone(rolling.current.notable_change)
        self.assertIn("picked up", rolling.current.notable_change)

    def test_publishing_is_rate_limited(self):
        rolling = vs.RollingVisualState(min_update_interval_seconds=4.0)
        rolling.ingest({"scene_summary": "s0", "objects": ["a"]}, now=0.0)
        # Changed, but too soon — state still updates, publication does not.
        self.assertFalse(rolling.ingest({"scene_summary": "s1", "objects": ["b"]}, now=1.0))
        self.assertEqual(rolling.current.objects, ("b",))
        self.assertTrue(rolling.ingest({"scene_summary": "s2", "objects": ["c"]}, now=10.0))

    def test_provider_supplied_notable_change_wins(self):
        rolling = vs.RollingVisualState(min_update_interval_seconds=0.0)
        rolling.ingest({"scene_summary": "A desk."}, now=1.0)
        rolling.ingest(
            {"scene_summary": "A desk.", "notable_change": "User picked up a red book."},
            now=2.0,
        )
        self.assertEqual(rolling.current.notable_change, "User picked up a red book.")


class ContextLineTests(unittest.TestCase):
    def test_context_line_is_compact_prose_not_json(self):
        rolling = vs.RollingVisualState(min_update_interval_seconds=0.0)
        rolling.ingest(
            {"scene_summary": "User picked up a red book.",
             "objects": ["red book"], "actions": ["picked up"]},
            now=1.0,
        )
        line = rolling.context_line()
        self.assertIn("red book", line)
        # It is folded into a natural-language perception note, so it must not
        # look like a serialized payload.
        for token in ("{", "}", '"scene_summary"', "[", "]"):
            self.assertNotIn(token, line)

    def test_context_line_is_none_before_any_observation(self):
        self.assertIsNone(vs.RollingVisualState().context_line())

    def test_clear_removes_the_visual_note(self):
        # Camera off must genuinely drop the note, not leave Arche describing a
        # room she can no longer see.
        rolling = vs.RollingVisualState(min_update_interval_seconds=0.0)
        rolling.ingest({"scene_summary": "A red book."}, now=1.0)
        self.assertIsNotNone(rolling.context_line())
        rolling.clear()
        self.assertIsNone(rolling.context_line())
        self.assertIsNone(rolling.current)

    def test_state_is_immutable(self):
        state = vs.visual_state_from_payload({"scene_summary": "x"}, previous=None, timestamp=1.0)
        with self.assertRaises(Exception):
            state.scene_summary = "y"


if __name__ == "__main__":
    unittest.main()

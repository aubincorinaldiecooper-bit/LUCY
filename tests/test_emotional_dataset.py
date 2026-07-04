"""Dataset collection + calibration ground truth for the emotional analyzer.

Pins the product rules:
  - profile captures and calibration moments are persisted through a bounded,
    best-effort background writer (never blocks, never raises into the bridge),
  - calibration asks at most one subtle either/or question at a time, cadence
    limited, only when emotionally useful, and the user's next utterance
    completes the moment,
  - the bridge injects the question as a one-shot instructions note and clears
    it after the answer,
  - raw *detected* labels still never appear in anything sent to Inworld.
"""

import asyncio
import types
import unittest

import emotional_dataset as ed
import inworld_realtime_bridge as irb
from inworld_voice_profile import NormalizedVoiceProfile


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class _RecordingDb:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail

    def __call__(self, sql, params):
        if self.fail:
            raise RuntimeError("db down")
        self.rows.append((sql, params))


def _writer(db=None, **kwargs):
    config = ed.EmotionalDatasetConfig(enabled=True, db_url="postgres://test")
    return ed.EmotionalDatasetWriter(config, db_writer=db or _RecordingDb(), **kwargs)


class ConfigTests(unittest.TestCase):
    def test_disabled_without_flag(self):
        config = ed.EmotionalDatasetConfig(enabled=False, db_url="postgres://x")
        self.assertEqual(config.is_usable(), (False, "dataset_disabled"))

    def test_disabled_without_db_url(self):
        config = ed.EmotionalDatasetConfig(enabled=True, db_url="")
        self.assertEqual(config.is_usable(), (False, "database_url_missing"))

    def test_usable(self):
        config = ed.EmotionalDatasetConfig(enabled=True, db_url="postgres://x")
        self.assertEqual(config.is_usable(), (True, "ok"))


class WriterTests(unittest.TestCase):
    def test_profile_event_row_written(self):
        db = _RecordingDb()

        async def scenario():
            writer = _writer(db)
            writer.record_profile_event(
                session_id="s1",
                source_event="conversation.item.input_audio_transcription.completed",
                raw_profile={"emotion": [{"label": "happy", "confidence": 0.9}]},
                normalized_profile={"energy": "high"},
                transcript="hello there friend",
                emotion_confidence=0.9,
            )
            await writer._flush_pending()
            return writer

        writer = asyncio.run(scenario())
        self.assertEqual(len(db.rows), 1)
        sql, params = db.rows[0]
        self.assertIn("voice_profile_events", sql)
        self.assertEqual(params[0], "s1")
        self.assertIn("happy", params[3])  # raw labels DO land in the dataset
        self.assertEqual(writer.counters["rows_written"], 1)

    def test_calibration_moment_row_written(self):
        db = _RecordingDb()

        async def scenario():
            writer = _writer(db)
            writer.record_calibration_moment(
                {
                    "session_id": "s1",
                    "turn_id": "4",
                    "transcript": "i feel stuck about this decision",
                    "arche_question": "Is this more about fear, guilt, or the pressure of choosing?",
                    "user_answer": "honestly, pressure",
                    "user_confirmed_or_corrected": True,
                    "inferred_emotional_pattern": "energy=low; tension=high; certainty=medium",
                    "normalized_profile": {"energy": "low"},
                }
            )
            await writer._flush_pending()

        asyncio.run(scenario())
        self.assertEqual(len(db.rows), 1)
        sql, params = db.rows[0]
        self.assertIn("emotional_calibration_moments", sql)
        self.assertEqual(params[4], "honestly, pressure")
        self.assertTrue(params[5])

    def test_write_error_counted_not_raised(self):
        async def scenario():
            writer = _writer(_RecordingDb(fail=True))
            writer.record_calibration_moment({"session_id": "s"})
            await writer._flush_pending()
            return writer

        writer = asyncio.run(scenario())
        self.assertEqual(writer.counters["write_errors"], 1)
        self.assertEqual(writer.counters["rows_written"], 0)

    def test_bounded_queue_drops_oldest(self):
        async def scenario():
            writer = _writer(max_queue_rows=2)
            for i in range(3):
                writer.record_calibration_moment({"session_id": f"s{i}"})
            return writer

        writer = asyncio.run(scenario())
        self.assertEqual(writer.counters["rows_dropped"], 1)
        self.assertEqual(len(writer._queue), 2)

    def test_aclose_drains_queue(self):
        db = _RecordingDb()

        async def scenario():
            writer = _writer(db)
            writer.start()
            writer.record_calibration_moment({"session_id": "s"})
            await writer.aclose()

        asyncio.run(scenario())
        self.assertEqual(len(db.rows), 1)


# ---------------------------------------------------------------------------
# Calibration heuristics + tracker
# ---------------------------------------------------------------------------


def _profile(**overrides):
    kwargs = dict(energy="low", tension="high", certainty="medium", confidence=0.8)
    kwargs.update(overrides)
    return NormalizedVoiceProfile(**kwargs)


class CalibrationQuestionTests(unittest.TestCase):
    def test_no_profile_no_question(self):
        question, reason = ed.calibration_question_for("i feel really worried about it", None)
        self.assertIsNone(question)
        self.assertEqual(reason, "no_voice_profile_context")

    def test_short_transcript_skipped(self):
        question, reason = ed.calibration_question_for("i am fine", _profile())
        self.assertIsNone(question)
        self.assertEqual(reason, "transcript_too_short")

    def test_neutral_and_irrelevant_skipped(self):
        question, reason = ed.calibration_question_for(
            "the weather is quite nice today", _profile(energy="medium", tension="medium")
        )
        self.assertIsNone(question)
        self.assertEqual(reason, "not_emotionally_useful")

    def test_low_certainty_question(self):
        question, reason = ed.calibration_question_for(
            "i do not really know how to say this", _profile(certainty="low")
        )
        self.assertEqual(reason, "low_certainty_or_ambiguous")
        self.assertIn("heavy, tense, or just unclear", question)

    def test_high_tension_question(self):
        question, reason = ed.calibration_question_for(
            "there is just so much going on right now", _profile(tension="high")
        )
        self.assertEqual(reason, "high_tension_or_worry")
        self.assertIn("anxiety", question)


class CalibrationTrackerTests(unittest.TestCase):
    def test_arms_then_completes_with_next_turn(self):
        tracker = ed.CalibrationTracker(session_id="room-1")
        completed, question = tracker.on_user_transcript(
            "i feel really worried about all of this", _profile(tension="high")
        )
        self.assertIsNone(completed)
        self.assertIsNotNone(question)
        completed, question2 = tracker.on_user_transcript("more like pressure honestly", _profile())
        self.assertIsNone(question2)  # cadence blocks an immediate second question
        self.assertEqual(completed["user_answer"], "more like pressure honestly")
        self.assertTrue(completed["user_confirmed_or_corrected"])
        self.assertEqual(completed["session_id"], "room-1")
        self.assertIn("tension=high", completed["inferred_emotional_pattern"])

    def test_cadence_allows_question_after_min_turns(self):
        tracker = ed.CalibrationTracker(session_id="r")
        emotional = "i feel really worried about all of this"
        _, q1 = tracker.on_user_transcript(emotional, _profile(tension="high"))
        self.assertIsNotNone(q1)
        _, q2 = tracker.on_user_transcript(emotional, _profile(tension="high"))
        _, q3 = tracker.on_user_transcript(emotional, _profile(tension="high"))
        self.assertIsNone(q2)
        self.assertIsNone(q3)
        _, q4 = tracker.on_user_transcript(emotional, _profile(tension="high"))
        self.assertIsNotNone(q4)


# ---------------------------------------------------------------------------
# Bridge integration
# ---------------------------------------------------------------------------


def _settings():
    return irb.InworldRealtimeSettings(
        api_key="k",
        session_id="room-1",
        websocket_url="wss://example.test/session",
        model="openai/gpt-4o-mini",
        stt_model="assemblyai/u3-rt-pro",
        tts_model="inworld-tts-2",
        voice="Luna",
        speed=1.0,
        turn_detection_type="semantic_vad",
        turn_detection_eagerness="medium",
        turn_detection_create_response=True,
        turn_detection_interrupt_response=True,
        instructions="Be concise.",
        timeout_seconds=60.0,
        voice_profile_enabled=True,
        input_format="pcm16",
        output_format="pcm16",
        auth_scheme="basic",
        tts_delivery_mode="CREATIVE",
        tts_segmenter_strategy="full_turn",
        tts_steering_handling="emit_once",
        emotion_confidence_floor=0.5,
    )


class _FakeWs:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _FakeWriter:
    def __init__(self):
        self.profile_events = []
        self.calibration_moments = []

    def record_profile_event(self, **kwargs):
        self.profile_events.append(kwargs)

    def record_calibration_moment(self, moment):
        self.calibration_moments.append(moment)


class _FakeMemoryLayer:
    def __init__(self):
        self.remembered = []
        self.closed = False

    def schedule_remember(self, role, content, turn_id=None, **kw):
        self.remembered.append({"role": role, "content": content, "turn_id": turn_id})

    async def aclose(self):
        self.closed = True


def _profile_event(emotion="frustrated", confidence=0.9, transcript="i am so frustrated with all of this"):
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": transcript,
        "voiceProfile": {"emotion": [{"label": emotion, "confidence": confidence}]},
    }


def _run_bridge_scenario(*events, calibration_enabled=True, memory_layer=None):
    async def scenario():
        room = types.SimpleNamespace(
            local_participant=types.SimpleNamespace(publish_track=lambda *a, **k: asyncio.sleep(0)),
            remote_participants={},
            on=lambda *a, **k: None,
            off=lambda *a, **k: None,
        )
        bridge = irb.InworldRealtimeLiveKitBridge(
            room,
            _settings(),
            dataset_writer=_FakeWriter(),
            calibration_enabled=calibration_enabled,
        )
        bridge._ws = _FakeWs()
        bridge._session_ready.set()
        # Memory now resolves lazily via a background task (see
        # BridgeMemoryResolutionTests below for that behavior); these
        # scenarios drive events directly without going through
        # session.created, so simulate "already resolved" here.
        if memory_layer is not None:
            bridge._memory_layer = memory_layer
            bridge._memory_resolved.set()
        for event in events:
            await bridge._handle_inworld_message(event)
        return bridge

    return asyncio.run(scenario())


class BridgeDatasetIntegrationTests(unittest.TestCase):
    def test_profile_capture_persists_raw_and_normalized(self):
        bridge = _run_bridge_scenario(_profile_event())
        events = bridge._dataset_writer.profile_events
        self.assertEqual(len(events), 1)
        row = events[0]
        self.assertEqual(row["session_id"], "room-1")
        self.assertEqual(row["raw_profile"]["emotion"][0]["label"], "frustrated")
        self.assertEqual(row["normalized_profile"]["tension"], "high")
        self.assertEqual(row["transcript"], "i am so frustrated with all of this")

    def test_calibration_question_injected_and_answer_persisted(self):
        first = _profile_event()
        answer = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "more like disappointment i think",
        }
        bridge = _run_bridge_scenario(first, answer)
        # The question went out as a one-shot instructions note...
        instruction_sends = [
            m["session"]["instructions"] for m in bridge._ws.sent if m["type"] == "session.update"
        ]
        self.assertTrue(any("calibration note" in text.lower() for text in instruction_sends))
        # ...and the user's next utterance became a persisted ground-truth moment.
        moments = bridge._dataset_writer.calibration_moments
        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0]["user_answer"], "more like disappointment i think")
        self.assertTrue(moments[0]["user_confirmed_or_corrected"])
        # The note is cleared after the answer: the final instructions sent no
        # longer carry the calibration note.
        self.assertNotIn("calibration note", instruction_sends[-1].lower())

    def test_calibration_disabled_no_questions_no_moments(self):
        bridge = _run_bridge_scenario(_profile_event(), calibration_enabled=False)
        self.assertEqual(bridge._dataset_writer.calibration_moments, [])
        instruction_sends = [
            m["session"]["instructions"] for m in bridge._ws.sent if m["type"] == "session.update"
        ]
        for text in instruction_sends:
            self.assertNotIn("calibration", text.lower())

    def test_detected_labels_never_sent_to_inworld(self):
        bridge = _run_bridge_scenario(_profile_event(emotion="angry"))
        sent_text = str(bridge._ws.sent).lower()
        for label in ("angry", "happy", "sad", "fearful", "surprised", "tender"):
            self.assertNotIn(label, sent_text)

    def test_no_writer_is_safe(self):
        async def scenario():
            room = types.SimpleNamespace(
                local_participant=types.SimpleNamespace(publish_track=lambda *a, **k: asyncio.sleep(0)),
                remote_participants={},
                on=lambda *a, **k: None,
                off=lambda *a, **k: None,
            )
            bridge = irb.InworldRealtimeLiveKitBridge(room, _settings(), calibration_enabled=True)
            bridge._ws = _FakeWs()
            bridge._session_ready.set()
            await bridge._handle_inworld_message(_profile_event())
            await bridge._handle_inworld_message(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "more like disappointment i think",
                }
            )
            return bridge

        bridge = asyncio.run(scenario())
        self.assertIsNotNone(bridge.latest_voice_profile)


# ---------------------------------------------------------------------------
# Long-term memory wiring (preload note is exercised at the run_inworld_
# realtime_bridge level via agent.py; these cover the per-turn writes and the
# confirmed-pattern-to-memory bridge that make up the rest of the loop).
# ---------------------------------------------------------------------------


class FormatConfirmedPatternContentTests(unittest.TestCase):
    def test_starts_with_prefix_and_includes_answer(self):
        content = ed.format_confirmed_pattern_content("energy=low; tension=high", "more like pressure")
        self.assertTrue(content.startswith(ed.EMOTIONAL_PATTERN_PREFIX))
        self.assertIn("more like pressure", content)

    def test_no_answer_still_valid(self):
        content = ed.format_confirmed_pattern_content("energy=low", "")
        self.assertEqual(content, ed.EMOTIONAL_PATTERN_PREFIX + "energy=low")


class RememberConfirmedPatternTests(unittest.TestCase):
    def _moment(self, confirmed=True, pattern="energy=high; tension=medium"):
        return {
            "session_id": "s",
            "turn_id": "4",
            "transcript": "i have a big decision to make",
            "arche_question": "fear or pressure?",
            "user_answer": "more like pressure" if confirmed else "",
            "inferred_emotional_pattern": pattern,
            "user_confirmed_or_corrected": confirmed,
        }

    def test_confirmed_pattern_written(self):
        mem = _FakeMemoryLayer()
        ed.remember_confirmed_pattern(mem, self._moment(confirmed=True))
        self.assertEqual(len(mem.remembered), 1)
        entry = mem.remembered[0]
        self.assertEqual(entry["role"], "emotional_calibration")
        self.assertTrue(entry["content"].startswith(ed.EMOTIONAL_PATTERN_PREFIX))
        self.assertIn("more like pressure", entry["content"])
        self.assertEqual(entry["turn_id"], 4)

    def test_unconfirmed_not_written(self):
        mem = _FakeMemoryLayer()
        ed.remember_confirmed_pattern(mem, self._moment(confirmed=False))
        self.assertEqual(mem.remembered, [])

    def test_none_memory_layer_is_noop(self):
        ed.remember_confirmed_pattern(None, self._moment(confirmed=True))  # must not raise

    def test_empty_pattern_not_written(self):
        mem = _FakeMemoryLayer()
        ed.remember_confirmed_pattern(mem, self._moment(confirmed=True, pattern=""))
        self.assertEqual(mem.remembered, [])

    def test_non_numeric_turn_id_degrades_to_none(self):
        mem = _FakeMemoryLayer()
        moment = self._moment(confirmed=True)
        moment["turn_id"] = "not-a-number"
        ed.remember_confirmed_pattern(mem, moment)
        self.assertIsNone(mem.remembered[0]["turn_id"])


class BridgeMemoryIntegrationTests(unittest.TestCase):
    def test_user_transcript_remembered(self):
        mem = _FakeMemoryLayer()
        answer = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "i am training for a marathon",
        }
        bridge = _run_bridge_scenario(answer, calibration_enabled=False, memory_layer=mem)
        self.assertEqual(len(mem.remembered), 1)
        entry = mem.remembered[0]
        self.assertEqual(entry["role"], "user")
        self.assertEqual(entry["content"], "i am training for a marathon")
        self.assertEqual(entry["turn_id"], 1)
        self.assertIs(bridge._memory_layer, mem)

    def test_assistant_transcript_remembered(self):
        mem = _FakeMemoryLayer()
        user_turn = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "how do i train for a marathon",
        }
        assistant_turn = {
            "type": "response.output_audio_transcript.done",
            "transcript": "start with a slow weekly long run.",
        }
        _run_bridge_scenario(user_turn, assistant_turn, calibration_enabled=False, memory_layer=mem)
        self.assertEqual(len(mem.remembered), 2)
        self.assertEqual(mem.remembered[0]["role"], "user")
        self.assertEqual(mem.remembered[1]["role"], "assistant")
        self.assertEqual(mem.remembered[1]["content"], "start with a slow weekly long run.")
        # Assistant turn shares the same turn_id as the user turn that triggered it.
        self.assertEqual(mem.remembered[1]["turn_id"], mem.remembered[0]["turn_id"])

    def test_turn_id_increments_across_multiple_user_turns(self):
        mem = _FakeMemoryLayer()
        first = {"type": "conversation.item.input_audio_transcription.completed", "transcript": "first turn here"}
        second = {"type": "conversation.item.input_audio_transcription.completed", "transcript": "second turn here"}
        _run_bridge_scenario(first, second, calibration_enabled=False, memory_layer=mem)
        self.assertEqual([e["turn_id"] for e in mem.remembered], [1, 2])

    def test_confirmed_calibration_pattern_reaches_memory(self):
        mem = _FakeMemoryLayer()
        first = _profile_event()
        answer = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "more like disappointment i think",
        }
        _run_bridge_scenario(first, answer, calibration_enabled=True, memory_layer=mem)
        pattern_entries = [e for e in mem.remembered if e["role"] == "emotional_calibration"]
        self.assertEqual(len(pattern_entries), 1)
        self.assertTrue(pattern_entries[0]["content"].startswith(ed.EMOTIONAL_PATTERN_PREFIX))
        self.assertIn("more like disappointment i think", pattern_entries[0]["content"])

    def test_no_memory_layer_is_safe(self):
        answer = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "i am training for a marathon",
        }
        bridge = _run_bridge_scenario(answer, calibration_enabled=False, memory_layer=None)  # must not raise
        self.assertIsNone(bridge._memory_layer)


# ---------------------------------------------------------------------------
# Lazy memory resolution: constructing/starting the bridge must never block
# on Postgres preload, or a fast talker's opening words get dropped while
# nothing is listening to their mic track yet. Memory resolves exactly once,
# the first time session.created triggers the first session.update.
# ---------------------------------------------------------------------------


def _fake_room():
    return types.SimpleNamespace(
        local_participant=types.SimpleNamespace(publish_track=lambda *a, **k: asyncio.sleep(0)),
        remote_participants={},
        on=lambda *a, **k: None,
        off=lambda *a, **k: None,
    )


class BridgeMemoryResolutionTests(unittest.TestCase):
    def test_construction_does_not_resolve_memory(self):
        async def scenario():
            async def resolve():
                return _FakeMemoryLayer(), "what we remember"

            task = asyncio.ensure_future(resolve())
            bridge = irb.InworldRealtimeLiveKitBridge(_fake_room(), _settings(), memory_task=task)
            bridge._ws = _FakeWs()
            # Nothing has awaited the task yet — bridge construction (and
            # everything track-subscription-critical inside it) is instant.
            self.assertIsNone(bridge._memory_layer)
            self.assertFalse(bridge._memory_resolved.is_set())
            await bridge.aclose()

        asyncio.run(scenario())

    def test_session_created_resolves_memory_and_bakes_note_into_instructions(self):
        async def scenario():
            mem = _FakeMemoryLayer()

            async def resolve():
                return mem, "what we remember about them"

            task = asyncio.ensure_future(resolve())
            bridge = irb.InworldRealtimeLiveKitBridge(_fake_room(), _settings(), memory_task=task)
            bridge._ws = _FakeWs()
            await bridge._handle_inworld_message({"type": "session.created"})
            self.assertIs(bridge._memory_layer, mem)
            self.assertIn("what we remember about them", bridge.settings.instructions)
            sent = bridge._ws.sent[0]
            self.assertIn("what we remember about them", sent["session"]["instructions"])
            await bridge.aclose()

        asyncio.run(scenario())

    def test_no_preload_note_leaves_instructions_unchanged(self):
        async def scenario():
            async def resolve():
                return None, None

            task = asyncio.ensure_future(resolve())
            bridge = irb.InworldRealtimeLiveKitBridge(_fake_room(), _settings(), memory_task=task)
            bridge._ws = _FakeWs()
            base_instructions = bridge.settings.instructions
            await bridge._handle_inworld_message({"type": "session.created"})
            self.assertEqual(bridge.settings.instructions, base_instructions)
            await bridge.aclose()

        asyncio.run(scenario())

    def test_ensure_memory_resolved_only_awaits_task_once(self):
        async def scenario():
            calls = {"n": 0}

            async def resolve():
                calls["n"] += 1
                return None, None

            task = asyncio.ensure_future(resolve())
            bridge = irb.InworldRealtimeLiveKitBridge(_fake_room(), _settings(), memory_task=task)
            bridge._ws = _FakeWs()
            await bridge._ensure_memory_resolved()
            await bridge._ensure_memory_resolved()
            await bridge._ensure_memory_resolved()
            self.assertEqual(calls["n"], 1)
            await bridge.aclose()

        asyncio.run(scenario())

    def test_aclose_before_resolution_cancels_task_cleanly(self):
        async def scenario():
            async def hanging():
                await asyncio.sleep(30)
                return None, None

            task = asyncio.ensure_future(hanging())
            bridge = irb.InworldRealtimeLiveKitBridge(_fake_room(), _settings(), memory_task=task)
            bridge._ws = _FakeWs()
            await bridge.aclose()  # must not hang or raise
            self.assertTrue(task.cancelled())
            self.assertIsNone(bridge._memory_layer)

        asyncio.run(scenario())

    def test_memory_task_exception_degrades_gracefully(self):
        async def scenario():
            async def boom():
                raise RuntimeError("db exploded")

            task = asyncio.ensure_future(boom())
            bridge = irb.InworldRealtimeLiveKitBridge(_fake_room(), _settings(), memory_task=task)
            bridge._ws = _FakeWs()
            await bridge._handle_inworld_message({"type": "session.created"})  # must not raise
            self.assertIsNone(bridge._memory_layer)
            self.assertEqual(len(bridge._ws.sent), 1)  # session.update still sent
            await bridge.aclose()

        asyncio.run(scenario())

    def test_no_memory_task_is_safe(self):
        async def scenario():
            bridge = irb.InworldRealtimeLiveKitBridge(_fake_room(), _settings())
            bridge._ws = _FakeWs()
            await bridge._handle_inworld_message({"type": "session.created"})  # must not raise
            self.assertEqual(len(bridge._ws.sent), 1)
            await bridge.aclose()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

"""openrouter_metering.py: per-user OpenRouter spend metering.

Pins the product rules:
  - off unless OPENROUTER_METERING_ENABLED=true AND a DATABASE_URL exists
    (a meter with nowhere to write is inert),
  - one persisted row per call, carrying token counts + charged cost,
    attributed to user_ref (billing key) + session_id,
  - record_from_response never raises and no-ops cleanly when there's no usage
    block or the response is malformed,
  - fetch_spend_summary aggregates rows into totals + per-feature/per-model
    breakdowns for forwarding as a meter.
"""

import asyncio
import unittest
from unittest import mock

import openrouter_metering as meter


class MeteringEnabledTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(meter.metering_enabled())

    def test_enabled_truthy_values(self):
        for value in ("true", "1", "yes", "on", "TRUE", "On"):
            with mock.patch.dict("os.environ", {"OPENROUTER_METERING_ENABLED": value}, clear=True):
                self.assertTrue(meter.metering_enabled(), value)

    def test_usage_body_requests_cost_accounting(self):
        self.assertEqual(meter.openrouter_usage_body(), {"usage": {"include": True}})


class ParseUsageTests(unittest.TestCase):
    def test_extracts_tokens_and_cost(self):
        data = {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 8,
                "total_tokens": 128,
                "cost": 0.000042,
            }
        }
        parsed = meter.parse_openrouter_usage(data)
        self.assertEqual(parsed["prompt_tokens"], 120)
        self.assertEqual(parsed["completion_tokens"], 8)
        self.assertEqual(parsed["total_tokens"], 128)
        self.assertEqual(parsed["cost_usd"], 0.000042)

    def test_missing_cost_defaults_to_zero(self):
        parsed = meter.parse_openrouter_usage({"usage": {"total_tokens": 10}})
        self.assertEqual(parsed["cost_usd"], 0.0)
        self.assertEqual(parsed["total_tokens"], 10)

    def test_no_usage_block_returns_none(self):
        self.assertIsNone(meter.parse_openrouter_usage({"choices": []}))

    def test_non_dict_returns_none(self):
        self.assertIsNone(meter.parse_openrouter_usage(None))
        self.assertIsNone(meter.parse_openrouter_usage("nope"))

    def test_garbage_token_values_coerce_to_zero(self):
        parsed = meter.parse_openrouter_usage(
            {"usage": {"prompt_tokens": "x", "cost": "nope"}}
        )
        self.assertEqual(parsed["prompt_tokens"], 0)
        self.assertEqual(parsed["cost_usd"], 0.0)


class SpendMeterEnabledTests(unittest.TestCase):
    def test_inert_when_flag_off(self):
        m = meter.SpendMeter(session_id="s1", enabled=False, db_url="postgres://x")
        self.assertFalse(m.enabled)

    def test_inert_without_db_url(self):
        m = meter.SpendMeter(session_id="s1", enabled=True, db_url=None)
        self.assertFalse(m.enabled)

    def test_enabled_when_flag_and_db_present(self):
        m = meter.SpendMeter(session_id="s1", enabled=True, db_url="postgres://x")
        self.assertTrue(m.enabled)


class SpendMeterRecordTests(unittest.TestCase):
    def _meter(self, **kw):
        writes = []
        m = meter.SpendMeter(
            session_id=kw.pop("session_id", "sess-1"),
            user_ref=kw.pop("user_ref", "user-9"),
            enabled=True,
            db_url="postgres://x",
            db_writer=lambda sql, params: writes.append((sql, params)),
            **kw,
        )
        return m, writes

    def test_disabled_meter_writes_nothing(self):
        writes = []
        m = meter.SpendMeter(
            session_id="s",
            enabled=False,
            db_url="postgres://x",
            db_writer=lambda sql, params: writes.append(params),
        )
        m.record_from_response("vision", "google/gemini", {"usage": {"cost": 1.0}})
        self.assertEqual(writes, [])

    def test_no_usage_block_writes_nothing(self):
        m, writes = self._meter()
        m.record_from_response("vision", "google/gemini", {"choices": []})
        self.assertEqual(writes, [])

    def test_records_row_with_attribution_and_cost(self):
        m, writes = self._meter(user_ref="clerk_123", session_id="room-7")
        data = {
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 12,
                "total_tokens": 212,
                "cost": 0.00013,
            }
        }
        # No running loop here -> inline write (still persisted, not dropped).
        m.record_from_response("mood", "google/gemini-2.5-flash-lite", data)
        self.assertEqual(len(writes), 1)
        _sql, params = writes[0]
        # (user_ref, session_id, feature, model, prompt, completion, total, cost)
        self.assertEqual(params[0], "clerk_123")
        self.assertEqual(params[1], "room-7")
        self.assertEqual(params[2], "mood")
        self.assertEqual(params[3], "google/gemini-2.5-flash-lite")
        self.assertEqual(params[4], 200)
        self.assertEqual(params[5], 12)
        self.assertEqual(params[6], 212)
        self.assertEqual(params[7], 0.00013)

    def test_user_ref_none_still_records_by_session(self):
        m, writes = self._meter(user_ref=None)
        m.record_from_response("vision", "m", {"usage": {"total_tokens": 5}})
        self.assertEqual(len(writes), 1)
        self.assertIsNone(writes[0][1][0])  # user_ref
        self.assertEqual(writes[0][1][1], "sess-1")  # session_id

    def test_write_failure_is_swallowed(self):
        def boom(sql, params):
            raise RuntimeError("db down")

        m = meter.SpendMeter(
            session_id="s", enabled=True, db_url="postgres://x", db_writer=boom
        )
        # Must not raise even though the writer explodes.
        m.record_from_response("vision", "m", {"usage": {"total_tokens": 5}})

    def test_background_write_under_running_loop_drains_on_close(self):
        captured = []

        async def scenario():
            m = meter.SpendMeter(
                session_id="s",
                user_ref="u",
                enabled=True,
                db_url="postgres://x",
                db_writer=lambda sql, params: captured.append(params),
            )
            m.record_from_response("vision", "m", {"usage": {"cost": 0.5, "total_tokens": 3}})
            # The write is scheduled as a background task; aclose drains it.
            await m.aclose()

        asyncio.run(scenario())
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][7], 0.5)  # cost_usd


class FetchSpendSummaryTests(unittest.TestCase):
    def test_aggregates_totals_and_breakdowns(self):
        rows = [
            # feature, model, calls, prompt, completion, total, cost
            ("vision", "google/gemini", 3, 600, 30, 630, 0.0009),
            ("mood", "google/gemini", 2, 400, 20, 420, 0.0004),
            ("mood", "anthropic/haiku", 1, 100, 10, 110, 0.0002),
        ]
        summary = meter.fetch_spend_summary(
            user_ref="u1", db_url="postgres://x", db_reader=lambda sql, params: rows
        )
        self.assertEqual(summary["user_ref"], "u1")
        self.assertEqual(summary["total_calls"], 6)
        self.assertEqual(summary["total_tokens"], 630 + 420 + 110)
        self.assertAlmostEqual(summary["total_cost_usd"], 0.0015)
        # by_feature
        self.assertEqual(summary["by_feature"]["vision"]["calls"], 3)
        self.assertEqual(summary["by_feature"]["mood"]["calls"], 3)
        self.assertAlmostEqual(summary["by_feature"]["mood"]["cost_usd"], 0.0006)
        # by_model
        self.assertEqual(summary["by_model"]["google/gemini"]["calls"], 5)
        self.assertAlmostEqual(summary["by_model"]["google/gemini"]["cost_usd"], 0.0013)

    def test_filters_build_where_clause(self):
        seen = {}

        def reader(sql, params):
            seen["sql"] = sql
            seen["params"] = params
            return []

        meter.fetch_spend_summary(
            user_ref="u1",
            session_id="s2",
            since="2026-01-01",
            db_url="postgres://x",
            db_reader=reader,
        )
        self.assertIn("user_ref = %s", seen["sql"])
        self.assertIn("session_id = %s", seen["sql"])
        self.assertIn("created_at >= %s", seen["sql"])
        self.assertEqual(seen["params"], ("u1", "s2", "2026-01-01"))

    def test_no_filters_has_no_where_clause(self):
        seen = {}

        def reader(sql, params):
            seen["sql"] = sql
            return []

        meter.fetch_spend_summary(db_url="postgres://x", db_reader=reader)
        self.assertNotIn("WHERE", seen["sql"])


if __name__ == "__main__":
    unittest.main()

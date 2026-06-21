import logging
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import runtime_context


def _toronto(hour: int, minute: int) -> datetime:
    return datetime(2026, 6, 21, hour, minute, tzinfo=ZoneInfo("America/Toronto"))


class DatetimeGuardFreshnessTests(unittest.TestCase):
    def setUp(self):
        # Build a context "at" 3:15 PM Toronto — its stored values freeze here.
        self.ctx = runtime_context.build_runtime_context(
            client_timezone="America/Toronto", now=_toronto(15, 15)
        )

    def test_stored_context_reflects_init_time(self):
        self.assertIn("3:15", self.ctx.current_time)
        self.assertEqual(self.ctx.session_timezone, "America/Toronto")

    def test_answer_recomputes_and_does_not_reuse_cached_time(self):
        at_315 = runtime_context.answer_datetime_intent(self.ctx, "time", now=_toronto(15, 15))
        self.assertIn("3:15 PM", at_315)

        # Same context, clock advanced to 3:18 → must update, not reuse 3:15.
        at_318 = runtime_context.answer_datetime_intent(self.ctx, "time", now=_toronto(15, 18))
        self.assertIn("3:18 PM", at_318)
        self.assertNotIn("3:15", at_318)
        self.assertIn("America/Toronto", at_318)

    def test_snapshot_recomputes_per_call(self):
        date_318, time_318 = runtime_context.current_datetime_snapshot(self.ctx, now=_toronto(15, 18))
        self.assertEqual(time_318, "3:18 PM")
        self.assertEqual(date_318, "2026-06-21")

    def test_part_of_day_uses_current_hour(self):
        self.assertIn("evening", runtime_context.answer_datetime_intent(self.ctx, "part_of_day", now=_toronto(19, 0)))
        self.assertIn("afternoon", runtime_context.answer_datetime_intent(self.ctx, "part_of_day", now=_toronto(15, 0)))


class DatetimeGuardLogFormattingTests(unittest.TestCase):
    def test_guard_log_line_has_matching_placeholders(self):
        # Reproduces the exact log call the date/time guard makes and asserts it
        # formats without a "not all arguments converted" error (the #4 bug:
        # placeholder/arg count mismatch).
        record = logging.LogRecord(
            name="agent",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=(
                "Date/time guard triggered: turn_id=%s datetime_guard_triggered=%s "
                "datetime_intent=%s datetime_answer_source=%s search_called=%s "
                "session_timezone=%s runtime_current_date=%s runtime_current_time=%s "
                "text_length=%s"
            ),
            args=(3, True, "time", "runtime_context", False, "America/Toronto", "2026-06-21", "3:18 PM", 24),
            exc_info=None,
        )
        # getMessage() raises if the placeholder/arg counts don't match.
        formatted = record.getMessage()
        self.assertIn("datetime_answer_source=runtime_context", formatted)
        self.assertIn("runtime_current_time=3:18 PM", formatted)


if __name__ == "__main__":
    unittest.main()

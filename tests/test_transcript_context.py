import asyncio
import os
import unittest
from unittest.mock import patch

from transcript_context import (
    TranscriptContext,
    _context_from_llm_payload,
    detect_transcript_context,
    interpret_transcript_context,
)


class TranscriptContextDeterministicTests(unittest.TestCase):
    def assert_intent(self, text: str, intent: str) -> TranscriptContext:
        context = detect_transcript_context(text)
        self.assertEqual(context.detected_intent, intent, text)
        self.assertEqual(context.source, "deterministic")
        return context

    def test_numeric_fragment(self):
        context = self.assert_intent("1968 and 746", "numeric_fragment")
        self.assertTrue(context.ambiguity_detected)
        self.assertTrue(context.clarification_suggested)
        self.assertIn("numeric fragment", context.llm_context_note or "")

    def test_sri_lankan_language_request(self):
        context = self.assert_intent("Can you speak Sri Lankan?", "language_request")
        self.assertTrue(context.ambiguity_detected)
        self.assertIn("Sinhala or Tamil", context.llm_context_note or "")

    def test_fragmented_sri_lankan_language_request(self):
        context = self.assert_intent("What language do I want you to... Can you speak Sri Lankan?", "language_request")
        self.assertTrue(context.ambiguity_detected)

    def test_choice_delegation(self):
        context = self.assert_intent("Anyone pick anyone that works for you.", "choice_delegation")
        self.assertTrue(context.ambiguity_detected)

    def test_voice_change_request(self):
        self.assert_intent("Speak to you in another voice. I don't want to speak to this voice anymore.", "voice_change_request")

    def test_profanity_reaction(self):
        context = self.assert_intent("Oh, fucking Uber.", "profanity_reaction")
        self.assertTrue(context.ambiguity_detected)

    def test_date_time_question(self):
        self.assert_intent("What time is it right now?", "date_time_question")

    def test_email_request(self):
        context = self.assert_intent("Can you send me an email?", "tool_request_email")
        self.assertTrue(context.clarification_suggested)

    def test_search_request(self):
        context = self.assert_intent("Look that up.", "tool_request_search")
        self.assertTrue(context.clarification_suggested)

    def test_document_request(self):
        context = self.assert_intent("Make a Word doc with that.", "tool_request_document")
        self.assertTrue(context.ambiguity_detected)

    def test_backchannel(self):
        self.assert_intent("Yeah.", "greeting_or_backchannel")


class TranscriptContextLLMTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict(os.environ, {"TRANSCRIPT_CONTEXT_LLM_ENABLED": "true", "TRANSCRIPT_CONTEXT_LLM_TIMEOUT_MS": "25"})
        self.env.start()

    async def asyncTearDown(self):
        self.env.stop()

    async def test_llm_valid_json_context_wins(self):
        async def caller(deterministic: TranscriptContext) -> TranscriptContext:
            return TranscriptContext(
                original_text=deterministic.original_text,
                cleaned_text=deterministic.cleaned_text,
                should_replace_user_text=False,
                llm_context_note="Use recent context.",
                ambiguity_detected=False,
                clarification_suggested=False,
                detected_intent="reference_to_prior_context",
                confidence=0.82,
                source="llm",
            )

        context = await interpret_transcript_context("send that to me", llm_caller=caller)
        self.assertEqual(context.source, "llm")
        self.assertEqual(context.detected_intent, "reference_to_prior_context")

    async def test_llm_timeout_falls_back(self):
        async def caller(deterministic: TranscriptContext) -> TranscriptContext:
            await asyncio.sleep(0.1)
            return deterministic

        context = await interpret_transcript_context("send that to me", llm_caller=caller)
        self.assertEqual(context.source, "deterministic_timeout_fallback")

    async def test_invalid_json_falls_back(self):
        async def caller(deterministic: TranscriptContext) -> TranscriptContext:
            raise ValueError("invalid json")

        context = await interpret_transcript_context("send that to me", llm_caller=caller)
        self.assertEqual(context.source, "deterministic_llm_error_fallback")

    async def test_llm_error_falls_back(self):
        async def caller(deterministic: TranscriptContext) -> TranscriptContext:
            raise RuntimeError("provider error")

        context = await interpret_transcript_context("send that to me", llm_caller=caller)
        self.assertEqual(context.source, "deterministic_llm_error_fallback")

    async def test_invented_unsupported_meaning_is_rejected(self):
        deterministic = detect_transcript_context("Oh, fucking Uber.")
        with self.assertRaises(ValueError):
            _context_from_llm_payload(
                {
                    "cleaned_text": "The user wants to book an Uber ride",
                    "should_replace_user_text": True,
                    "detected_intent": "tool_request_search",
                    "ambiguity_detected": False,
                    "clarification_suggested": False,
                    "llm_context_note": "Book a ride.",
                    "confidence": 0.91,
                },
                deterministic,
            )


if __name__ == "__main__":
    unittest.main()

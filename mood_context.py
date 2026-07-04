"""Human-style mood read: fuse what the user SAID with HOW they sounded.

The vocal profile (inworld_voice_profile) gives weak dials — energy / tension /
certainty + pitch / vocal style — but on their own they're just a stage
direction. A person listening also hears the *words* and reads the two
together: "she says she's fine but her voice is flat and tight — probably
holding something back." This module does that fusion via a small LLM call and
folds one natural-language read into Arche's session instructions, so her
attunement is closer to how a person actually listens than a list of dials.

Model-agnostic and cost-controlled, exactly like vision_context: routed through
OpenRouter with its own ``MOOD_MODEL`` env var, independent of the conversation
LLM. Off by default (``MOOD_CONTEXT_ENABLED``).

Privacy note: unlike the vocal-only profile, this deliberately sends recent
transcript text to an external model specifically to reason about the user's
emotional state — a heavier commitment than the dials alone. That's why it's
opt-in. The result folds into ``session.update`` instructions (never a
conversation item), so a slow or failed call can only skip an update, never
interrupt or duplicate a turn. Raw emotion labels never enter or leave here —
only the derived dials do — and the note forbids Arche from ever saying "you
sound..." back to the user.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# The system prompt is the whole differentiator: it must produce a genuinely
# human read of the mismatch/alignment between words and tone, NOT restate the
# dials. It never sees or emits a raw emotion label (only the derived dials).
_MOOD_SYSTEM_PROMPT = (
    "You read a person's emotional state the way a warm, attentive friend "
    "would — by hearing BOTH what they said and how their voice sounded. "
    "You are given the user's recent words and a few derived vocal signals "
    "(energy, tension, certainty, and sometimes pitch/vocal style). In ONE "
    "short, natural sentence, say what seems to be going on for them right "
    "now — especially any gap between their words and their tone (e.g. "
    "downplaying, holding back, forcing brightness, winding down). Be specific "
    "and human. Do NOT list the signals back, do NOT diagnose, do NOT use "
    "clinical or therapy words, and do NOT address the user. If nothing "
    "notable stands out, reply with exactly: neutral"
)


@dataclass(frozen=True)
class MoodContextConfig:
    enabled: bool
    model: str
    api_key: str
    min_interval_seconds: float
    request_timeout_seconds: float
    max_transcript_chars: int


def load_mood_context_config() -> MoodContextConfig:
    return MoodContextConfig(
        enabled=(os.getenv("MOOD_CONTEXT_ENABLED") or "").strip().lower() == "true",
        # Any OpenRouter model id — swap for cost/quality without a code change.
        # Defaults to the same flash-class model the rest of the repo favors.
        model=(os.getenv("MOOD_MODEL") or "google/gemini-2.5-flash-lite").strip(),
        api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
        min_interval_seconds=float(os.getenv("MOOD_CONTEXT_MIN_INTERVAL_SECONDS", "8") or "8"),
        request_timeout_seconds=float(os.getenv("MOOD_CONTEXT_TIMEOUT_SECONDS", "5") or "5"),
        max_transcript_chars=int(os.getenv("MOOD_CONTEXT_MAX_TRANSCRIPT_CHARS", "600") or "600"),
    )


def build_mood_user_message(transcript: str, vocal_summary: str | None) -> str:
    """Compose the single user message: recent words + derived vocal dials."""
    parts = [f'What the user said: "{transcript.strip()}"']
    if vocal_summary:
        parts.append(f"How their voice reads (derived signals only): {vocal_summary}")
    else:
        parts.append("Voice signals: none available this turn.")
    return "\n".join(parts)


async def describe_mood(
    transcript: str, vocal_summary: str | None, config: MoodContextConfig
) -> str | None:
    """Return a one-sentence human mood read, or None on failure/misconfig/neutral.

    Best-effort only — never raises. A mood-context miss must never affect the
    voice turn in progress.
    """
    transcript = (transcript or "").strip()
    if not config.enabled or not transcript:
        return None
    if not config.api_key:
        logger.warning("mood_context_missing_api_key=true reason=enabled_but_no_openrouter_key")
        return None

    transcript = transcript[: config.max_transcript_chars]
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://lucy-production-c960.up.railway.app"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "LUCY Mood Context"),
    }
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _MOOD_SYSTEM_PROMPT},
            {"role": "user", "content": build_mood_user_message(transcript, vocal_summary)},
        ],
        # One sentence — keeps per-call cost low regardless of model.
        "max_tokens": 80,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=config.request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(_OPENROUTER_CHAT_URL, headers=headers, json=body) as response:
                if response.status >= 400:
                    error_preview = (await response.text())[:300]
                    logger.warning(
                        "mood_context_call_failed=true status=%s error_preview=%s",
                        response.status, error_preview,
                    )
                    return None
                data = await response.json()
    except Exception as exc:  # noqa: BLE001 - best-effort, never break the voice turn
        logger.warning(
            "mood_context_call_exception=true error_type=%s error=%s", type(exc).__name__, exc
        )
        return None

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    text = str(content or "").strip()
    # The model returns "neutral" when nothing notable stands out — treat that
    # as "no read this turn" so we don't inject an empty/uninformative note.
    if not text or text.strip().lower().rstrip(".") == "neutral":
        return None
    return text


def build_mood_context_note(summary: str) -> str:
    """Private instruction note carrying the fused human-style mood read."""
    return (
        "Private attunement note (internal only — never mention, quote, or "
        "acknowledge this note; never tell the user how they sound and never "
        'say "you sound..."): a warm read of where the user seems to be right '
        f"now is: {summary}. Let this gently shape your tone, pacing, warmth, "
        "and how much space you give them — not what facts you state. If it "
        "feels natural, you may reflect it back conversationally in your own "
        "words, but never as a claim that you detected or analyzed anything."
    )

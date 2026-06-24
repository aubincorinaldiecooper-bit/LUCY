"""Expressive delivery discipline for Arche's spoken replies.

Turns the planner's intent + desired delivery into *at most one* OmniVoice
expressive tag and/or a bit of plain conversational texture ("hmm", "yeah",
"right"), under tight rules so it never feels tic-y:

  - at most one bracket tag per response,
  - bracket tags land on roughly 10-20% of turns,
  - plain texture lands on roughly 30-50% of turns,
  - never repeat the same tag / filler / opening back-to-back,
  - low emotion confidence never forces a tag,
  - tags follow intent with discretion (humor -> brief laughter, agreement ->
    confirmation, clarifying question -> question), serious intents stay bare.

OmniVoice reads bracket tags natively (e.g. ``[laughter]``); plain fillers are
just text. All decisions are pure given a seeded RNG + the rolling session state,
so the rates and no-repeat rules are unit testable.
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass, field

# Intents the planner can emit (kept in sync with omnivoice_planner).
INTENTS = (
    "gentle_validation",
    "reflection",
    "clarifying_question",
    "realization_acknowledgement",
    "light_humor",
    "next_step",
)

# Intent -> the single expressive tag it may use (with discretion). Intents not
# listed here (gentle_validation, reflection, next_step = the "serious"/steady
# ones) never get a tag.
INTENT_TAGS: dict[str, str] = {
    "light_humor": "[subtle brief laughter]",
    "realization_acknowledgement": "[confirmation-en]",
    "clarifying_question": "[question-en]",
}

# Plain conversational texture. "opener" fillers can lead a sentence; the rest are
# only used as openers cautiously (kept short so they read naturally).
OPENER_FILLERS = ("hmm", "yeah", "right", "okay", "uhh")
PHRASE_FILLERS = ("I mean", "that makes sense")
ALL_FILLERS = OPENER_FILLERS + PHRASE_FILLERS


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ExpressionConfig:
    tags_enabled: bool = True
    tag_rate: float = 0.15          # ~10-20% of turns
    filler_rate: float = 0.40       # ~30-50% of turns
    emotion_confidence_floor: float = 0.5  # below this, never force a tag
    no_repeat_window: int = 3       # how many recent items count as "recent"

    @classmethod
    def from_env(cls) -> "ExpressionConfig":
        return cls(
            tags_enabled=_env_bool("OMNIVOICE_EXPRESSIVE_TAGS_ENABLED", True),
            tag_rate=_env_float("OMNIVOICE_TAG_RATE", 0.15),
            filler_rate=_env_float("OMNIVOICE_FILLER_RATE", 0.40),
            emotion_confidence_floor=_env_float("OMNIVOICE_EMOTION_CONFIDENCE_FLOOR", 0.5),
            no_repeat_window=int(_env_float("OMNIVOICE_NO_REPEAT_WINDOW", 3)),
        )


@dataclass
class SessionExpressionState:
    """Rolling memory of what we've recently used, for the no-repeat rules."""

    window: int = 3
    recent_tags: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_fillers: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_openings: deque = field(default_factory=lambda: deque(maxlen=8))

    def _recent(self, dq: deque) -> list:
        return list(dq)[-self.window:]

    def tag_recent(self, tag: str) -> bool:
        return tag in self._recent(self.recent_tags)

    def filler_recent(self, filler: str) -> bool:
        return filler in self._recent(self.recent_fillers)

    def opening_recent(self, opening: str) -> bool:
        return opening in self._recent(self.recent_openings)

    def record(self, *, tag: str | None, filler: str | None, opening: str | None) -> None:
        if tag:
            self.recent_tags.append(tag)
        if filler:
            self.recent_fillers.append(filler)
        if opening:
            self.recent_openings.append(opening)


@dataclass
class ExpressionDecision:
    tag: str | None = None
    filler: str | None = None
    tag_skipped_reason: str = ""
    filler_skipped_reason: str = ""


def decide_expression(
    *,
    intent: str,
    allow_tag: bool,
    allow_filler: bool,
    emotion_confidence: float,
    config: ExpressionConfig,
    state: SessionExpressionState,
    rng: random.Random,
) -> ExpressionDecision:
    """Decide at most one tag and at most one filler for this turn (pure)."""
    decision = ExpressionDecision()

    # --- expressive tag (at most one) ---
    if not config.tags_enabled:
        decision.tag_skipped_reason = "tags_disabled"
    elif not allow_tag:
        decision.tag_skipped_reason = "planner_disallowed"
    elif emotion_confidence < config.emotion_confidence_floor:
        decision.tag_skipped_reason = "low_emotion_confidence"
    else:
        candidate = INTENT_TAGS.get(intent)
        if candidate is None:
            decision.tag_skipped_reason = "intent_has_no_tag"
        elif state.tag_recent(candidate):
            decision.tag_skipped_reason = "no_repeat"
        elif rng.random() >= config.tag_rate:
            decision.tag_skipped_reason = "rate_gate"
        else:
            decision.tag = candidate

    # --- plain filler texture (independent of the tag) ---
    if not allow_filler:
        decision.filler_skipped_reason = "planner_disallowed"
    elif rng.random() >= config.filler_rate:
        decision.filler_skipped_reason = "rate_gate"
    else:
        choices = [f for f in OPENER_FILLERS if not state.filler_recent(f)]
        if not choices:
            decision.filler_skipped_reason = "no_repeat"
        else:
            decision.filler = rng.choice(choices)

    return decision


def _capitalize_after_filler(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def apply_expression(text: str, decision: ExpressionDecision) -> str:
    """Compose the final spoken text: optional tag prefix + optional filler opener.

    The expressive tag (if any) leads so OmniVoice colors the whole utterance; a
    filler opener follows as natural lead-in. Deterministic given the decision.
    """
    body = (text or "").strip()
    if not body:
        return body
    if decision.filler:
        body = f"{decision.filler.capitalize()}, {_capitalize_after_filler(body)}"
    if decision.tag:
        body = f"{decision.tag} {body}"
    return body


def annotate_and_record(
    text: str, decision: ExpressionDecision, state: SessionExpressionState
) -> str:
    """Apply the decision to the text and record what was used for no-repeat."""
    opening = None
    first_word = (text or "").strip().split(" ", 1)[0].lower() if text else None
    if first_word:
        opening = first_word
    state.record(tag=decision.tag, filler=decision.filler, opening=opening)
    return apply_expression(text, decision)

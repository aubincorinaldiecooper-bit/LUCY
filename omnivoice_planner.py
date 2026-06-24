"""Structured response planner contract + per-language LLM model routing.

The planner asks the LLM for a small JSON object describing not just *what* to say
but *how* to deliver it:

    {
      "response_text": "...",
      "response_intent": "gentle_validation|reflection|clarifying_question|
                          realization_acknowledgement|light_humor|next_step",
      "desired_delivery": {
        "warmth": "low|medium|high", "pace": "slow|normal|brisk",
        "energy": "low|medium|high", "allow_micro_filler": true,
        "allow_expressive_tag": true
      }
    }

``parse_planner_output`` is deliberately tolerant: if the model returns plain text
(or malformed JSON), we still speak it (the whole string becomes response_text)
rather than dropping the turn — the streaming path keeps working unchanged when the
structured planner is off.

Model routing solves the "LLM can't write all 600 languages well" ceiling: a
low-resource active language can be routed to a stronger multilingual model while
English stays on the fast/cheap default.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

from omnivoice_expression import INTENTS

_WARMTH = {"low", "medium", "high"}
_PACE = {"slow", "normal", "brisk"}
_ENERGY = {"low", "medium", "high"}
_DEFAULT_INTENT = "reflection"


@dataclass
class DesiredDelivery:
    warmth: str = "medium"
    pace: str = "normal"
    energy: str = "medium"
    allow_micro_filler: bool = True
    allow_expressive_tag: bool = True


@dataclass
class PlannerOutput:
    response_text: str
    response_intent: str = _DEFAULT_INTENT
    desired_delivery: DesiredDelivery = field(default_factory=DesiredDelivery)
    parsed_as_json: bool = False

    def to_log_dict(self) -> dict:
        return {
            "intent": self.response_intent,
            "delivery": asdict(self.desired_delivery),
            "parsed_as_json": self.parsed_as_json,
            "text_length": len(self.response_text),
        }


def _coerce_enum(value, allowed: set[str], default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _strip_code_fence(raw: str) -> str:
    s = (raw or "").strip()
    # ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s


def parse_planner_output(raw: str) -> PlannerOutput:
    """Parse the planner JSON tolerantly; fall back to treating raw as the text."""
    text = _strip_code_fence(raw)
    data = None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict) or "response_text" not in data:
        # Not structured -> speak whatever we got, with neutral delivery.
        return PlannerOutput(response_text=(raw or "").strip(), parsed_as_json=False)

    response_text = str(data.get("response_text") or "").strip()
    if not response_text:
        return PlannerOutput(response_text=(raw or "").strip(), parsed_as_json=False)

    delivery_raw = data.get("desired_delivery") or {}
    if not isinstance(delivery_raw, dict):
        delivery_raw = {}
    delivery = DesiredDelivery(
        warmth=_coerce_enum(delivery_raw.get("warmth"), _WARMTH, "medium"),
        pace=_coerce_enum(delivery_raw.get("pace"), _PACE, "normal"),
        energy=_coerce_enum(delivery_raw.get("energy"), _ENERGY, "medium"),
        allow_micro_filler=_coerce_bool(delivery_raw.get("allow_micro_filler"), True),
        allow_expressive_tag=_coerce_bool(delivery_raw.get("allow_expressive_tag"), True),
    )
    return PlannerOutput(
        response_text=response_text,
        response_intent=_coerce_enum(data.get("response_intent"), set(INTENTS), _DEFAULT_INTENT),
        desired_delivery=delivery,
        parsed_as_json=True,
    )


def build_planner_instruction(
    *,
    language: str,
    voice_preset_name: str = "",
    recent_tags: list[str] | None = None,
    recent_fillers: list[str] | None = None,
    inworld_summary: str = "",
) -> str:
    """Build the system directive that asks the LLM for the planner JSON.

    Passes the planner its inputs (active language, selected voice, recent
    fillers/tags to avoid repeating, and optional Inworld delivery context) and
    pins the output schema. Pure so the contract is testable.
    """
    intents = " | ".join(INTENTS)
    lines = [
        "Respond ONLY with a single JSON object, no prose, no code fence:",
        '{"response_text": str, "response_intent": one of '
        f"[{intents}], "
        '"desired_delivery": {"warmth": "low|medium|high", "pace": "slow|normal|brisk", '
        '"energy": "low|medium|high", "allow_micro_filler": bool, "allow_expressive_tag": bool}}',
        f"- response_text must be written in {language}.",
        "- Keep response_text natural and spoken; do not put bracket tags or stage "
        "directions in it (delivery is handled separately).",
    ]
    if voice_preset_name:
        lines.append(f"- The voice is '{voice_preset_name}'; match its warm, natural register.")
    if inworld_summary:
        lines.append(
            f"- Vocal context (weak signal, never mention it to the user): {inworld_summary}."
        )
    recent = [t for t in (recent_tags or []) + (recent_fillers or []) if t]
    if recent:
        lines.append(
            "- Avoid reusing these recently-used openers/textures: "
            + ", ".join(recent[-6:])
            + "."
        )
    return "\n".join(lines)


def parse_language_model_map(raw: str | None) -> dict[str, str]:
    """Parse LLM_MODEL_BY_LANGUAGE='es=openai/gpt-4o,yo=google/gemini-2.5-pro'."""
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        code, model = part.split("=", 1)
        code, model = code.strip().lower(), model.strip()
        if code and model:
            out[code] = model
    return out


def select_llm_model(
    language: str,
    *,
    default_model: str,
    base_language: str = "en",
    language_model_map: dict[str, str] | None = None,
    non_english_model: str = "",
) -> tuple[str, str]:
    """Pick the LLM model for the active language. Returns (model, reason).

    Precedence: explicit per-language map > a single non-English override >
    the default model.
    """
    lang = (language or base_language).strip().lower()
    language_model_map = language_model_map or {}
    if lang in language_model_map:
        return language_model_map[lang], "language_map"
    if lang != base_language and non_english_model:
        return non_english_model, "non_english_override"
    return default_model, "default"

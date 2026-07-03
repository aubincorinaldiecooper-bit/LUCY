"""Best-effort camera-frame description, fed into Arche's session instructions.

Model-agnostic and cost-controlled by design: routed through OpenRouter (already
used elsewhere in this codebase for the Hume CLM path), so the vision model is a
single env var (``VISION_MODEL``) fully independent of ``INWORLD_REALTIME_MODEL``
(the conversation LLM) — swap providers/models for cost without touching code.

Off by default (``VISION_CONTEXT_ENABLED`` unset/false): the bridge never
subscribes to a camera track at all in that case, so this module makes no calls
and costs nothing unless deliberately enabled.

The result is a short text summary folded into ``session.update`` instructions
(see build_vision_context_note + inworld_realtime_bridge._compose_instructions_update)
— never a conversation item — so a slow or failed vision call can only skip an
update, never interrupt or duplicate a voice turn.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

_VISION_PROMPT = (
    "In one short, plain sentence, describe what is visibly in view. Neutral, "
    "factual content only — objects, setting, general scene. Do not guess at "
    "the person's mood, identity, health, or intentions; do not comment on "
    "appearance."
)


@dataclass(frozen=True)
class VisionContextConfig:
    enabled: bool
    model: str
    api_key: str
    min_interval_seconds: float
    frame_sample_interval_seconds: float
    max_frame_dimension: int
    request_timeout_seconds: float


def load_vision_context_config() -> VisionContextConfig:
    return VisionContextConfig(
        enabled=(os.getenv("VISION_CONTEXT_ENABLED") or "").strip().lower() == "true",
        # Any OpenRouter model id works — swap freely for cost/quality without a
        # code change. Defaults to a Gemini flash-class model: image inputs
        # count as only a few hundred tokens there, versus gpt-4o-mini whose
        # per-image token multiplier makes frequent snapshots cost roughly
        # gpt-4o prices. Same model family the repo already uses for LLM_MODEL.
        model=(os.getenv("VISION_MODEL") or "google/gemini-2.5-flash-lite").strip(),
        api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
        min_interval_seconds=float(os.getenv("VISION_CONTEXT_MIN_INTERVAL_SECONDS", "12") or "12"),
        frame_sample_interval_seconds=float(os.getenv("VISION_FRAME_SAMPLE_INTERVAL_SECONDS", "2") or "2"),
        max_frame_dimension=int(os.getenv("VISION_FRAME_MAX_DIMENSION", "512") or "512"),
        request_timeout_seconds=float(os.getenv("VISION_CONTEXT_TIMEOUT_SECONDS", "6") or "6"),
    )


async def describe_frame(image_data_url: str, config: VisionContextConfig) -> str | None:
    """Return a short neutral description, or None on any failure/misconfig.

    Best-effort only — never raises. A vision-context miss must never affect
    the voice turn in progress.
    """
    if not config.enabled or not image_data_url:
        return None
    if not config.api_key:
        logger.warning("vision_context_missing_api_key=true reason=enabled_but_no_openrouter_key")
        return None

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://lucy-production-c960.up.railway.app"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "LUCY Vision Context"),
    }
    body = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        # A one-sentence answer needs very few tokens; keeps per-call cost low
        # regardless of which model is configured.
        "max_tokens": 60,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=config.request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(_OPENROUTER_CHAT_URL, headers=headers, json=body) as response:
                if response.status >= 400:
                    error_preview = (await response.text())[:300]
                    logger.warning(
                        "vision_context_call_failed=true status=%s error_preview=%s",
                        response.status, error_preview,
                    )
                    return None
                data = await response.json()
    except Exception as exc:  # noqa: BLE001 - best-effort, never break the voice turn
        logger.warning(
            "vision_context_call_exception=true error_type=%s error=%s", type(exc).__name__, exc
        )
        return None

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    text = str(content or "").strip()
    return text or None


def build_vision_context_note(summary: str) -> str:
    """Private instruction note carrying the latest camera-frame description."""
    return (
        "Private perception context (internal only — never mention, quote, or "
        "acknowledge a camera, image, or that you are \"seeing\" anything; speak "
        f"as though you simply notice things): currently visible: {summary}. "
        "Mention this only if it is naturally relevant to the conversation."
    )

"""Compact rolling visual state distilled from a live MiniCPM-o video session.

Live vision produces a *stream*: at VISION_TARGET_FPS the provider can emit an
observation every few hundred milliseconds. Feeding that stream into Arche's
prompt would drown the conversation in narration ("a person is sitting", "a
person is still sitting", ...) and blow up token cost for no gain.

So this module keeps exactly one compact state per session — the latest read of
the room — and tracks what *changed* since the previous read. Arche's reasoning
layer consumes that state, never the frame stream:

    {"timestamp": ..., "scene_summary": ..., "people": [...], "objects": [...],
     "actions": [...], "visible_text": [...], "notable_change": ...,
     "confidence": ...}

``notable_change`` is the important field. It is set only when the scene
materially moved (new/removed objects or actions, or a genuinely different
summary) — "User picked up a red book." — which is what makes an update worth
pushing into the session instructions at all. A static scene re-observed at
2 FPS produces state updates with ``notable_change=None``, and the caller
(vision_session) skips the instruction refresh entirely.

Nothing here does I/O, holds a socket, or touches a frame buffer: it is pure
data + comparison, so it is cheap to call on every provider message and trivial
to test. Frames themselves never enter this module — only the provider's text
description of them — so no user media can leak into a log line built from a
VisualState.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# A live model can return a long paragraph; the whole point of this layer is to
# stay compact, so every free-text field is clamped. These bound what can reach
# Arche's instructions, independent of what the provider decides to say.
MAX_SUMMARY_CHARS = 240
MAX_CHANGE_CHARS = 160
MAX_LIST_ITEMS = 8
MAX_LIST_ITEM_CHARS = 48

# Fenced ```json blocks are common in vision-model output; strip the fence
# before parsing rather than failing the whole message.
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

# A provider is supposed to send descriptions, but a misbehaving or echoing
# Gateway could send the frame back. Free text that looks like encoded media is
# dropped rather than described: everything that becomes a scene_summary is
# forwarded to Inworld, so this is the boundary that keeps user media on-box.
_MEDIA_PREFIXES = ("data:", "/9j/", "iVBOR", "R0lGOD", "UklGR")
# Longest plausible one-sentence description. Anything longer that arrives as
# bare text is not a description.
_MAX_PROSE_CHARS = 2_000


def _looks_like_media(text: str) -> bool:
    stripped = text.lstrip()
    if stripped[:64].lower().startswith("data:"):
        return True
    if stripped.startswith(_MEDIA_PREFIXES):
        return True
    if len(stripped) > _MAX_PROSE_CHARS:
        # Long, space-free, base64-alphabet text is encoded data, not prose.
        sample = stripped[:512]
        if " " not in sample:
            return True
    return False


def _clamp(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    # Collapse whitespace so a multi-line model answer becomes one clean line.
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _clamp_list(value: Any) -> tuple[str, ...]:
    """Normalize a provider list field to a bounded tuple of short strings.

    Accepts a list, a comma-separated string, or a single scalar — live models
    are inconsistent about this and a shape mismatch must never lose the whole
    observation.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        items: list[Any] = [part for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            # e.g. {"label": "red book", "confidence": 0.9} -> "red book"
            item = item.get("label") or item.get("name") or item.get("text") or ""
        elif isinstance(item, (list, tuple, set)):
            # A nested container would otherwise be str()'d into Python repr
            # ("['a', 'b']") and shown to Arche as an object name.
            item = ", ".join(str(x) for x in item if isinstance(x, (str, int, float)))
        elif not isinstance(item, (str, int, float)):
            continue
        if isinstance(item, str) and _looks_like_media(item):
            continue
        text = _clamp(item, MAX_LIST_ITEM_CHARS)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= MAX_LIST_ITEMS:
            break
    return tuple(out)


def _clamp_confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return max(0.0, min(1.0, number))


@dataclass(frozen=True)
class VisualState:
    """One compact read of what the camera currently shows.

    Immutable by design: ``update`` returns a new state, so a state handed to
    the instruction builder can never be mutated underneath it mid-turn.
    """

    timestamp: float
    scene_summary: str
    people: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    visible_text: tuple[str, ...] = ()
    notable_change: str | None = None
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view — the shape Arche's reasoning layer consumes."""
        return {
            "timestamp": self.timestamp,
            "scene_summary": self.scene_summary,
            "people": list(self.people),
            "objects": list(self.objects),
            "actions": list(self.actions),
            "visible_text": list(self.visible_text),
            "notable_change": self.notable_change,
            "confidence": self.confidence,
        }

    def to_context_line(self) -> str:
        """One compact line for the session instructions.

        Deliberately *not* the full JSON: the conversation LLM does better with
        a natural sentence than a serialized struct, and this is folded into the
        same private perception note the OpenRouter vision path already uses
        (inworld_realtime_bridge.build_vision_context_note), so the phrasing has
        to read like an observation, not a payload.
        """
        parts: list[str] = []
        if self.scene_summary:
            parts.append(self.scene_summary)
        change = (self.notable_change or "").rstrip("…")
        if change and not _normalize_summary(self.scene_summary).startswith(
            _normalize_summary(change)
        ):
            parts.append(f"Just changed: {self.notable_change}")
        if self.people:
            parts.append("People: " + ", ".join(self.people))
        if self.objects:
            parts.append("Visible: " + ", ".join(self.objects))
        if self.actions:
            parts.append("Happening: " + ", ".join(self.actions))
        if self.visible_text:
            parts.append("Text in view: " + "; ".join(self.visible_text))
        return " ".join(p.rstrip(".") + "." for p in parts if p).strip()


def _material_difference(previous: VisualState | None, current: VisualState) -> str | None:
    """Describe what meaningfully changed, or None if the scene is unchanged.

    "Meaningful" is deliberately coarse: appearing/disappearing objects and
    actions, or a different summary sentence. Re-observing the same room at
    2 FPS must produce None, or every frame would trigger an instructions
    refresh and we would be right back to narrating the stream.
    """
    if previous is None:
        # The first observation of a session is itself the notable change.
        return current.scene_summary or None

    new_objects = [o for o in current.objects if o.lower() not in {p.lower() for p in previous.objects}]
    gone_objects = [o for o in previous.objects if o.lower() not in {p.lower() for p in current.objects}]
    new_actions = [a for a in current.actions if a.lower() not in {p.lower() for p in previous.actions}]
    people_changed = {p.lower() for p in current.people} != {p.lower() for p in previous.people}

    fragments: list[str] = []
    if new_actions:
        fragments.append(", ".join(new_actions))
    if people_changed and current.people:
        fragments.append("people: " + ", ".join(current.people))
    elif people_changed:
        fragments.append("no one in view")
    if new_objects:
        fragments.append("now visible: " + ", ".join(new_objects))
    if gone_objects and not new_objects:
        # Only worth mentioning when nothing new replaced it; otherwise the
        # "now visible" fragment already carries the change.
        fragments.append("no longer visible: " + ", ".join(gone_objects))

    if fragments:
        return _clamp("; ".join(fragments), MAX_CHANGE_CHARS)

    # No object/action delta — fall back to the summary sentence itself, but
    # only if it actually reads differently. Normalizing case/punctuation stops
    # trivial rewordings of the same scene from counting as a change.
    if _normalize_summary(current.scene_summary) != _normalize_summary(previous.scene_summary):
        return _clamp(current.scene_summary, MAX_CHANGE_CHARS) or None
    return None


# Values providers use to mean "nothing changed", normalized by _normalize_summary.
_NO_CHANGE_SENTINELS = {
    "", "none", "no change", "nochange", "null", "nil", "n a", "na",
    "no notable change", "nothing", "no significant change", "unchanged",
}


def _normalize_summary(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def parse_provider_payload(payload: Any) -> dict[str, Any] | None:
    """Best-effort extraction of a visual-observation dict from a provider message.

    Live providers are messy: the same worker may send a JSON object, a JSON
    string, a ```json fenced block, or plain prose. Every one of those must
    either yield a usable dict or be ignored — a malformed message can never
    raise, because it arrives on the vision receive loop and an exception there
    would tear down the session (and, before this returns None, the voice turn's
    sibling task).
    """
    if payload is None:
        return None

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - defensive; decode with replace shouldn't raise
            return None

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        fenced = _JSON_FENCE.match(text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            # Text that *tried* to be JSON and failed is a truncated or corrupt
            # message, not an observation — treating it as prose would put a
            # fragment like '{"scene_summary":' straight into Arche's prompt.
            if text[:1] in ("{", "["):
                return None
            if _looks_like_media(text):
                return None
            # Genuine plain prose from the model is still a usable observation.
            return {"scene_summary": text}

    if isinstance(payload, list):
        # Some workers wrap the observation in a single-element list.
        payload = next((item for item in payload if isinstance(item, dict)), None)

    if not isinstance(payload, dict):
        return None

    # Unwrap the common realtime envelopes rather than treating them as the
    # observation itself.
    if any(payload.get(k) for k in ("scene_summary", "summary", "description",
                                    "objects", "people", "actions", "visible_text")):
        # The observation is right here — do not descend into an envelope key
        # (e.g. a "state": "ok" status string) and lose it.
        return payload
    for key in ("visual_state", "state", "observation", "result", "data", "response"):
        inner = payload.get(key)
        if isinstance(inner, dict):
            payload = inner
            break
        if isinstance(inner, str) and inner.strip():
            if _looks_like_media(inner):
                return None
            return {"scene_summary": inner.strip()}

    if not any(
        payload.get(k)
        for k in ("scene_summary", "summary", "description", "text", "content",
                  "objects", "people", "actions", "visible_text")
    ):
        return None
    return payload


def visual_state_from_payload(
    payload: Any,
    *,
    previous: VisualState | None,
    timestamp: float,
) -> VisualState | None:
    """Build the next rolling state from one provider message, or None.

    Returns None when the message carries no usable observation (a heartbeat,
    an ack, a malformed blob) — the caller then leaves the current state alone
    rather than clobbering a good read with an empty one.
    """
    data = parse_provider_payload(payload)
    if data is None:
        return None

    raw_summary = str(
        data.get("scene_summary")
        or data.get("summary")
        or data.get("description")
        or data.get("text")
        or data.get("content")
        or ""
    )
    # Checked BEFORE clamping: clamping a data URL to 240 chars would produce a
    # short string that no longer looks like media but is still user media.
    # Everything that becomes scene_summary is forwarded to Inworld.
    summary = "" if _looks_like_media(raw_summary) else _clamp(raw_summary, MAX_SUMMARY_CHARS)
    objects = _clamp_list(data.get("objects"))
    people = _clamp_list(data.get("people"))
    actions = _clamp_list(data.get("actions"))
    visible_text = _clamp_list(data.get("visible_text") or data.get("ocr"))

    if not summary and not objects and not people and not actions and not visible_text:
        return None

    candidate = VisualState(
        timestamp=timestamp,
        scene_summary=summary,
        people=people,
        objects=objects,
        actions=actions,
        visible_text=visible_text,
        confidence=_clamp_confidence(data.get("confidence")),
    )

    # Prefer the provider's own change sentence when it offers one; otherwise
    # derive it from the delta against the previous state.
    provider_change = _clamp(data.get("notable_change") or "", MAX_CHANGE_CHARS)
    if _normalize_summary(provider_change) in _NO_CHANGE_SENTINELS:
        # Providers commonly fill this field with "none" / "no change" rather
        # than omitting it; publishing that verbatim would announce a change
        # that did not happen.
        provider_change = ""
    notable_change = provider_change or _material_difference(previous, candidate)

    from dataclasses import replace as _replace

    return _replace(candidate, notable_change=notable_change or None)


@dataclass
class RollingVisualState:
    """Holds the current state for one vision session and gates instruction pushes.

    ``ingest`` returns True only when the caller should refresh Arche's session
    instructions: i.e. the scene materially changed *and* enough time has passed
    since the last push. Everything else updates the in-memory state silently,
    so the reasoning layer always has a current read without the conversation
    being interrupted by it.
    """

    min_update_interval_seconds: float = 4.0
    current: VisualState | None = None
    updates_ingested: int = 0
    updates_published: int = 0
    _last_published_at: float = field(default=0.0, repr=False)
    # A change detected but held back by the rate limit. Without this the
    # change would be lost for good: `current` has already moved on, so the
    # next comparison sees no delta and the prompt never learns about it.
    _change_pending: bool = field(default=False, repr=False)

    def ingest(self, payload: Any, *, now: float) -> bool:
        state = visual_state_from_payload(payload, previous=self.current, timestamp=now)
        if state is None:
            return False
        previous = self.current
        self.current = state
        self.updates_ingested += 1

        if state.notable_change is not None:
            self._change_pending = True
        if not self._change_pending:
            return False
        # First observation always publishes; after that, rate-limit so a busy
        # scene can't push an instructions update every frame. A change held
        # back here stays pending and publishes on the first ingest after the
        # window, rather than being dropped.
        if previous is not None and (now - self._last_published_at) < self.min_update_interval_seconds:
            return False
        self._last_published_at = now
        self.updates_published += 1
        self._change_pending = False
        return True

    def context_line(self) -> str | None:
        if self.current is None:
            return None
        return self.current.to_context_line() or None

    def clear(self) -> None:
        """Drop the visual read (camera off / session ended).

        The next instructions refresh then rebuilds without a vision note —
        build_instructions_session_update always recomposes from the base
        prompt, so a cleared state genuinely removes the note rather than
        leaving a stale one in place.
        """
        self.current = None
        self._last_published_at = 0.0
        self._change_pending = False

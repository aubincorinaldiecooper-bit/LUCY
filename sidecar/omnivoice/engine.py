"""OmniVoice synthesis engine for the sidecar.

This is the one place you plug in the real OmniVoice (k2-fsa/OmniVoice) model.
The LUCY worker only speaks HTTP to this service (see app.py), so everything
model-specific lives here behind ``synthesize_pcm`` which returns mono 16-bit PCM.

Two modes:
  - real  (default): lazily loads OmniVoice from OMNIVOICE_MODEL_PATH on
    OMNIVOICE_DEVICE and synthesizes. The actual call is marked TODO below —
    fill it in with the repo's API; everything around it (HTTP, auth, WAV/PCM
    framing, resampling contract) is done.
  - dummy (OMNIVOICE_DUMMY=true): returns a short low tone with no model, so you
    can deploy + wire the worker end-to-end (provider selection, Hume fallback,
    voice pool) before the GPU/model is ready.
"""

from __future__ import annotations

import logging
import math
import os
import struct

logger = logging.getLogger("omnivoice.engine")

DUMMY = (os.getenv("OMNIVOICE_DUMMY", "false").strip().lower() in {"1", "true", "yes", "on"})
DEFAULT_DEVICE = (os.getenv("OMNIVOICE_DEVICE") or "cuda").strip().lower()
DEFAULT_MODEL_PATH = (os.getenv("OMNIVOICE_MODEL_PATH") or "").strip()

_model = None  # lazily-loaded real model handle


class EngineUnavailable(RuntimeError):
    """Raised when the real model can't be loaded; the worker falls back to Hume."""


def _dummy_pcm(text: str, sample_rate: int) -> bytes:
    """A short, quiet 220 Hz tone sized loosely to the text (plumbing only)."""
    seconds = max(0.4, min(3.0, 0.04 * max(1, len(text))))
    n = int(sample_rate * seconds)
    amp = 1500  # low amplitude, well within int16
    frames = bytearray()
    for i in range(n):
        sample = int(amp * math.sin(2 * math.pi * 220.0 * (i / sample_rate)))
        frames += struct.pack("<h", sample)
    return bytes(frames)


def _load_model(device: str, model_path: str):
    """Load the OmniVoice model once. Fill in with the repo's loader."""
    global _model
    if _model is not None:
        return _model
    try:
        # TODO(omnivoice): import + load the real model, e.g.:
        #   from omnivoice import OmniVoice
        #   _model = OmniVoice.from_pretrained(model_path or "k2-fsa/OmniVoice",
        #                                      device=device)
        raise ImportError("OmniVoice model loader not wired yet")
    except Exception as e:  # noqa: BLE001
        raise EngineUnavailable(f"omnivoice model load failed: {e}") from e
    return _model


def synthesize_pcm(
    text: str,
    *,
    voice: str | None,
    language: str,
    expressive_tags: bool,
    sample_rate: int,
    device: str | None = None,
    model_path: str | None = None,
) -> bytes:
    """Return mono 16-bit little-endian PCM at ``sample_rate`` for ``text``.

    Raises EngineUnavailable if the real model isn't loadable (the worker then
    falls back to Hume, so the session never goes silent).
    """
    if DUMMY:
        return _dummy_pcm(text, sample_rate)

    model = _load_model(device or DEFAULT_DEVICE, model_path or DEFAULT_MODEL_PATH)
    # TODO(omnivoice): call the model and return int16 PCM mono at `sample_rate`.
    #   OmniVoice supports voice cloning / voice design / auto voice and inline
    #   non-verbal tags like [laughter]; pass `voice`, `language`, and (when
    #   `expressive_tags`) keep the bracket tags in `text`. Resample to
    #   `sample_rate` and convert to int16 mono before returning.
    #   waveform = model.generate(text=text, voice=voice, language=language, ...)
    #   return to_int16_mono(waveform, sample_rate)
    raise EngineUnavailable("omnivoice synthesize not wired yet")  # remove once implemented

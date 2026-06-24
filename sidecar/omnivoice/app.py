"""OmniVoice TTS sidecar — the HTTP service the LUCY worker calls.

Implements exactly the contract the worker's omnivoice_tts.py was built to:

    POST /synthesize           (Authorization: Bearer <OMNIVOICE_API_KEY> if set)
      body: {text, voice, language, sample_rate, audio_format,
             expressive_tags, device, model_path}
      200 -> audio bytes: WAV container, or raw s16le PCM mono when
             audio_format=pcm_s16le. 503 -> worker falls back to Hume.
    GET /health, GET /        -> liveness (the worker prewarms by hitting "/").

Model specifics live in engine.py; this file is just framing, auth, and errors.
Run: uvicorn app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import io
import os
import wave

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from engine import EngineUnavailable, synthesize_pcm

API_KEY = (os.getenv("OMNIVOICE_API_KEY") or "").strip()

app = FastAPI(title="OmniVoice TTS sidecar")


class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = None
    language: str = "en"
    sample_rate: int = 24000
    audio_format: str = "wav"  # "wav" | "pcm_s16le"
    expressive_tags: bool = True
    device: str | None = None
    model_path: str | None = None


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _check_auth(authorization: str | None) -> None:
    if not API_KEY:
        return  # auth not configured on this sidecar
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@app.get("/")
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest, authorization: str | None = Header(default=None)) -> Response:
    _check_auth(authorization)
    if not (req.text or "").strip():
        raise HTTPException(status_code=400, detail="text is required")
    try:
        pcm = synthesize_pcm(
            req.text,
            voice=req.voice,
            language=req.language,
            expressive_tags=req.expressive_tags,
            sample_rate=req.sample_rate,
            device=req.device,
            model_path=req.model_path,
        )
    except EngineUnavailable as e:
        # 503 -> the worker's FallbackAdapter moves on to Hume.
        raise HTTPException(status_code=503, detail=str(e)) from e

    fmt = (req.audio_format or "wav").strip().lower()
    if fmt in {"pcm", "pcm_s16le", "raw"}:
        return Response(content=pcm, media_type="audio/pcm")
    return Response(content=_pcm_to_wav(pcm, req.sample_rate), media_type="audio/wav")

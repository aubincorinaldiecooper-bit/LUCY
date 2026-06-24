# OmniVoice TTS sidecar (reference)

A small HTTP service that wraps [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
so the LUCY worker can use OmniVoice as its TTS provider over HTTP, with automatic
fallback to Hume if this service is unavailable.

The worker only speaks HTTP to this service — all model/GPU specifics stay here.

## Contract (what the worker calls)

```
POST /synthesize        Authorization: Bearer <OMNIVOICE_API_KEY>   (only if set)
  body: {
    "text": "...", "voice": "<preset id>"|null, "language": "en",
    "sample_rate": 24000, "audio_format": "wav" | "pcm_s16le",
    "expressive_tags": true, "device": "cuda"|"cpu", "model_path": "..."
  }
  200 -> audio bytes: WAV container, or raw little-endian s16 PCM mono when
         audio_format=pcm_s16le.
  503 -> worker falls back to Hume (use for "model not ready / failed").

GET /health, GET /  ->  {"status":"ok"}   (the worker prewarms by hitting "/")
```

## Run it

Dummy mode (no model — verify the worker wiring end-to-end first):
```
OMNIVOICE_DUMMY=true uvicorn app:app --host 0.0.0.0 --port 8080
```
Docker:
```
docker build -t omnivoice-sidecar .
docker run -p 8080:8080 -e OMNIVOICE_DUMMY=true omnivoice-sidecar
```

## Wire the real model

Edit `engine.py` — two TODOs:
1. `_load_model()` — load OmniVoice from `OMNIVOICE_MODEL_PATH` on `OMNIVOICE_DEVICE`.
2. `synthesize_pcm()` — call the model and return **mono int16 PCM at `sample_rate`**.
   OmniVoice supports voice cloning / voice design / auto voice and inline
   non-verbal tags (`[laughter]`, …); keep bracket tags in `text` when
   `expressive_tags` is true. Resample to `sample_rate` and convert to int16 mono.

Raise `EngineUnavailable` on any load/synthesis failure so the worker degrades to
Hume instead of the session going silent.

## Env vars

| var | meaning |
|-----|---------|
| `OMNIVOICE_DUMMY` | `true` → return a test tone, no model |
| `OMNIVOICE_DEVICE` | `cuda` / `mps` / `cpu` |
| `OMNIVOICE_MODEL_PATH` | local path or HF id for the weights |
| `OMNIVOICE_API_KEY` | if set, require `Authorization: Bearer <key>` (optional; protects *this* endpoint) |

## Point the worker at it

On the LUCY worker (Railway): `OMNIVOICE_ENABLED=true`, `TTS_PROVIDER=omnivoice`,
`TTS_FALLBACK_PROVIDER=hume`, `OMNIVOICE_URL=https://<this-service>` (and
`OMNIVOICE_API_KEY` matching, if you set one here).

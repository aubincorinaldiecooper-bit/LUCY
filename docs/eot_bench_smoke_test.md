# End-of-turn (EOT) evaluation — smoke-test plan

Goal: compare our current endpointing against the LiveKit audio `TurnDetector`
(`livekit-agents >= 1.6.1`, `inference.TurnDetector`) on **long-turn / pause-heavy**
speech, using LiveKit's [`eot-bench`](https://github.com/livekit/eot-bench)
methodology. This is a **smoke test**, not a gating benchmark — keep the sample
small and run it manually.

## What we're measuring
- **False cutoffs on mid-turn pauses** — does the detector commit while the user
  is mid-thought (the outdoor/long-turn failure mode)?
- **Response latency after a true end-of-turn** — how long after the user truly
  finishes before we commit.
- **Long-turn behavior** across pause-heavy samples (20–40s thoughts).

## Setup
```bash
git clone https://github.com/livekit/eot-bench
cd eot-bench
uv sync            # or: pip install -e .
export LIVEKIT_API_KEY=...   # only needed for v1 via LiveKit Inference
export LIVEKIT_API_SECRET=...
```

## 1. Smoke test (small sample first)
Run streaming predictions for the audio TurnDetector on a handful of long/pause-heavy
clips before any full run:
```bash
eot-harness predict-streaming \
  --model livekit/turn-detector \
  --dataset ./samples/long_turns_smoke.jsonl \
  --limit 10 \
  --out ./out/turn_detector_smoke.jsonl
```

## 2. Compare current vs audio TurnDetector
```bash
eot-harness compare-models \
  --a ./out/baseline_endpointing.jsonl \
  --b ./out/turn_detector_smoke.jsonl \
  --report ./out/compare_report.md
```
Read `compare_report.md` for false-cutoff rate and post-EOT latency deltas.

## 3. Tie back to our runtime logs
After a Railway session with `LIVEKIT_TURN_DETECTOR_ENABLED=true`, correlate
bench findings with these log lines (added in agent.py):
- `livekit_agents_version`, `livekit_turn_detector_available`,
  `livekit_turn_detector_enabled`, `turn_detector_class`, `turn_detector_version`,
  `turn_detector_default_selection`
- `turn_detection_resolved`, `livekit_turn_detector_active`,
  `turn_detector_version_active`, `turn_detector_fallback_used`,
  `endpointing_dynamic_enabled`, `text_turn_detector_used=false`
- `audio_enhancement_enabled/provider/model/applied_stage/error`

## Tuning loop (only if logs show a problem)
- Premature commits on long pauses → raise `LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD`
  (start unset = SDK default; nudge up in small steps).
- Excessive dead air after true EOT → lower it.
- Re-run the smoke test after each change; do not tune from a single live anecdote.

## Notes
- Use the **audio** TurnDetector only; the deprecated text turn detector is not used
  (`text_turn_detector_used=false`).
- `version=auto` lets the SDK pick `v1` via LiveKit Inference when reachable, else the
  local `v1-mini`. On Railway (outside LiveKit Cloud), confirm which was selected from
  `turn_detector_default_selection` / `turn_detector_version_active` in the logs.

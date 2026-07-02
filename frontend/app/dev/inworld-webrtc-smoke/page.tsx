"use client";

import { useMemo, useState } from "react";

type SmokeResult = {
  connected?: boolean;
  session_updated?: boolean;
  response_created?: boolean;
  event_counts?: Record<string, number>;
  audio_delta_count?: number;
  audio_delta_total_chars?: number;
  saw_audio_done?: boolean;
  saw_audio_transcript?: boolean;
  saw_text_done?: boolean;
  saw_response_done?: boolean;
  errors?: unknown[];
  first_events?: unknown[];
  last_events?: unknown[];
  [key: string]: unknown;
};

const HIGHLIGHT_FIELDS = [
  "audio_delta_count",
  "audio_delta_total_chars",
  "saw_audio_done",
  "saw_audio_transcript",
  "saw_text_done",
  "saw_response_done",
  "errors",
  "event_counts",
] as const;

function formatValue(value: unknown) {
  if (value === undefined) return "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number" || typeof value === "string") return String(value);
  return JSON.stringify(value, null, 2);
}

export default function InworldWebSocketSmokePage() {
  const enabled = process.env.NEXT_PUBLIC_ENABLE_INWORLD_WEBRTC_SMOKE === "true";
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SmokeResult | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const prettyJson = useMemo(() => (result ? JSON.stringify(result, null, 2) : ""), [result]);
  const audioDeltaCount = typeof result?.audio_delta_count === "number" ? result.audio_delta_count : 0;
  const sawResponseDone = result?.saw_response_done === true;

  const runSmokeTest = async () => {
    if (!enabled || loading) return;

    setLoading(true);
    setStatus(null);
    setResult(null);

    try {
      const response = await fetch("/api/inworld/ws-smoke-test", {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      const text = await response.text();
      let parsed: SmokeResult;
      try {
        parsed = text ? (JSON.parse(text) as SmokeResult) : {};
      } catch {
        parsed = { error: "invalid_json_response", status: response.status, body: text };
      }
      setResult(parsed);
      setStatus(`Backend returned ${response.status}`);
    } catch (error) {
      setResult({ error: "request_failed", detail: error instanceof Error ? error.message : String(error) });
      setStatus("Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#0B1020] px-6 py-10 text-slate-100">
      <div className="mx-auto flex max-w-5xl flex-col gap-8">
        <header className="space-y-3">
          <p className="text-sm font-medium uppercase tracking-[0.3em] text-cyan-300">Developer smoke test</p>
          <h1 className="text-4xl font-semibold tracking-tight">Inworld WebSocket audio smoke test</h1>
          <p className="max-w-2xl text-lg text-slate-300">
            Tests whether Inworld WebSocket emits response.output_audio.delta for Luna.
          </p>
        </header>

        {!enabled ? (
          <section className="rounded-2xl border border-amber-400/30 bg-amber-400/10 p-5 text-amber-100">
            Set <code>NEXT_PUBLIC_ENABLE_INWORLD_WEBRTC_SMOKE=true</code> to enable this smoke-test page.
          </section>
        ) : null}

        <section className="rounded-3xl border border-white/10 bg-white/[0.06] p-6 shadow-2xl shadow-black/20">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-xl font-semibold">Run direct backend WebSocket check</h2>
              <p className="mt-1 text-sm text-slate-300">
                Calls the server-side Next proxy, which forwards to the backend using the server-only smoke token.
              </p>
            </div>
            <button
              type="button"
              onClick={() => void runSmokeTest()}
              disabled={!enabled || loading}
              className="rounded-full bg-cyan-300 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Running…" : "Run WebSocket smoke test"}
            </button>
          </div>

          {loading ? <p className="mt-6 text-cyan-200">Loading WebSocket smoke-test result…</p> : null}
          {status ? <p className="mt-6 text-sm text-slate-300">{status}</p> : null}

          {result ? (
            <div className="mt-6 space-y-6">
              {audioDeltaCount > 0 ? (
                <div className="rounded-2xl border border-emerald-400/40 bg-emerald-400/10 p-4 text-emerald-100">
                  ✅ Inworld WebSocket audio deltas received
                </div>
              ) : null}
              {audioDeltaCount === 0 && sawResponseDone ? (
                <div className="rounded-2xl border border-rose-400/40 bg-rose-400/10 p-4 text-rose-100">
                  ❌ No Inworld WebSocket audio deltas received
                </div>
              ) : null}

              <div className="grid gap-3 md:grid-cols-2">
                {HIGHLIGHT_FIELDS.map((field) => (
                  <article key={field} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                    <h3 className="text-sm font-semibold uppercase tracking-wide text-cyan-200">{field}</h3>
                    <pre className="mt-2 whitespace-pre-wrap break-words text-sm text-slate-100">
                      {formatValue(result[field])}
                    </pre>
                  </article>
                ))}
              </div>

              <section>
                <h2 className="mb-3 text-xl font-semibold">Raw JSON result</h2>
                <pre className="max-h-[520px] overflow-auto rounded-2xl border border-white/10 bg-slate-950 p-4 text-sm leading-relaxed text-slate-100">
                  {prettyJson}
                </pre>
              </section>
            </div>
          ) : null}
        </section>
      </div>
    </main>
  );
}

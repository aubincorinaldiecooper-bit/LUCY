"use client";

import { useState } from "react";

/**
 * Dev-only smoke test: connect directly to Inworld's Realtime WebSocket
 * endpoint (via the Python backend, bypassing LiveKit) and answer one
 * question: "Does Inworld emit response.output_audio.delta events for
 * Luna?" Lives at /dev/inworld-webrtc-smoke.
 */

const ENABLED = process.env.NEXT_PUBLIC_ENABLE_INWORLD_WEBRTC_SMOKE === "true";

interface WsSmokeResult {
  mode?: string;
  model?: string | null;
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
  response_transcript?: string;
  errors?: unknown[];
  first_events?: unknown[];
  last_events?: unknown[];
  error?: string;
  [key: string]: unknown;
}

const Highlight = ({ label, val }: { label: string; val: React.ReactNode }) => (
  <tr>
    <td style={{ padding: "4px 8px", color: "#888" }}>{label}</td>
    <td style={{ padding: "4px 8px", fontFamily: "monospace" }}>{val}</td>
  </tr>
);

export default function InworldWebSocketSmokePage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<WsSmokeResult | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const run = async (mode: "text" | "image" = "text") => {
    setLoading(true);
    setFetchError(null);
    setResult(null);
    try {
      const resp = await fetch("/api/inworld/ws-smoke-test", {
        method: "POST",
        ...(mode === "image"
          ? {
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ mode: "image" }),
            }
          : {}),
      });
      const text = await resp.text();
      let parsed: WsSmokeResult;
      try {
        parsed = JSON.parse(text) as WsSmokeResult;
      } catch {
        setFetchError(`non_json_response status=${resp.status} body=${text.slice(0, 500)}`);
        return;
      }
      setResult(parsed);
    } catch (err) {
      setFetchError(`request_failed: ${String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const audioDeltaCount = result?.audio_delta_count ?? 0;
  const sawResponseDone = result?.saw_response_done === true;
  const isImageRun = result?.mode === "image";
  const transcript = (result?.response_transcript ?? "").toLowerCase();

  let resultMessage: string | null = null;
  if (result) {
    if (isImageRun) {
      // The test image is a red circle on white; a grounded reply must name it.
      const sawImage = /red/.test(transcript) && /circ/.test(transcript);
      if (sawImage) {
        resultMessage = "✅ Vision confirmed — the reply describes the red circle";
      } else if (transcript) {
        resultMessage =
          "⚠️ Got a reply but it does not clearly describe the image — read the transcript below";
      } else if (sawResponseDone || (result.errors?.length ?? 0) > 0) {
        resultMessage = "❌ No transcript — image content part likely rejected (see errors)";
      }
    } else if (audioDeltaCount > 0) {
      resultMessage = "✅ Inworld WebSocket audio deltas received";
    } else if (audioDeltaCount === 0 && sawResponseDone) {
      resultMessage = "❌ No Inworld WebSocket audio deltas received";
    }
  }

  if (!ENABLED) {
    return (
      <div style={{ padding: 24, maxWidth: 800, margin: "0 auto", fontFamily: "system-ui" }}>
        <h1 style={{ fontSize: 22, fontWeight: 700 }}>Inworld WebSocket audio smoke test</h1>
        <p style={{ color: "#666" }}>
          Disabled. Set <code>NEXT_PUBLIC_ENABLE_INWORLD_WEBRTC_SMOKE=true</code> to enable this
          page.
        </p>
      </div>
    );
  }

  return (
    <div style={{ padding: 24, maxWidth: 1100, margin: "0 auto", fontFamily: "system-ui" }}>
      <h1 style={{ fontSize: 22, fontWeight: 700 }}>Inworld WebSocket audio smoke test</h1>
      <p style={{ color: "#666", marginBottom: 12 }}>
        Tests whether Inworld WebSocket emits response.output_audio.delta for Luna.
      </p>

      <div style={{ display: "flex", gap: 8, marginBottom: 16, alignItems: "center" }}>
        <button onClick={() => run("text")} disabled={loading} style={{ padding: "8px 14px" }}>
          {loading ? "Running…" : "Run WebSocket smoke test"}
        </button>
        <button onClick={() => run("image")} disabled={loading} style={{ padding: "8px 14px" }}>
          {loading ? "Running…" : "Run image vision test"}
        </button>
        {loading && <span style={{ color: "#444" }}>Waiting on Inworld…</span>}
      </div>

      {resultMessage && (
        <p style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>{resultMessage}</p>
      )}

      {fetchError && (
        <pre
          style={{
            background: "#fee",
            color: "#900",
            padding: 12,
            fontFamily: "monospace",
            fontSize: 12,
            overflowX: "auto",
            borderRadius: 4,
            marginBottom: 16,
          }}
        >
          {fetchError}
        </pre>
      )}

      {result && (
        <>
          {result.response_transcript ? (
            <>
              <h2 style={{ fontSize: 16, marginTop: 8 }}>Reply transcript</h2>
              <p
                style={{
                  background: "#f4f4f4",
                  padding: 12,
                  borderRadius: 4,
                  fontFamily: "monospace",
                  fontSize: 13,
                  marginBottom: 16,
                }}
              >
                {result.response_transcript}
              </p>
            </>
          ) : null}

          <h2 style={{ fontSize: 16, marginTop: 8 }}>Key fields</h2>
          <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: 16 }}>
            <tbody>
              <Highlight label="mode" val={result.mode ?? "—"} />
              <Highlight label="model" val={result.model ?? "—"} />
              <Highlight label="audio_delta_count" val={result.audio_delta_count ?? "—"} />
              <Highlight
                label="audio_delta_total_chars"
                val={result.audio_delta_total_chars ?? "—"}
              />
              <Highlight label="saw_audio_done" val={result.saw_audio_done ? "✅" : "❌"} />
              <Highlight
                label="saw_audio_transcript"
                val={result.saw_audio_transcript ? "✅" : "❌"}
              />
              <Highlight label="saw_text_done" val={result.saw_text_done ? "✅" : "❌"} />
              <Highlight
                label="saw_response_done"
                val={result.saw_response_done ? "✅" : "❌"}
              />
              <Highlight
                label="errors"
                val={
                  Array.isArray(result.errors) && result.errors.length > 0
                    ? JSON.stringify(result.errors)
                    : "none"
                }
              />
              <Highlight
                label="event_counts"
                val={result.event_counts ? JSON.stringify(result.event_counts) : "—"}
              />
            </tbody>
          </table>

          <h2 style={{ fontSize: 16, marginTop: 16 }}>Full result</h2>
          <pre
            style={{
              background: "#111",
              color: "#0f0",
              padding: 12,
              fontFamily: "monospace",
              fontSize: 12,
              overflowX: "auto",
              borderRadius: 4,
              whiteSpace: "pre-wrap",
            }}
          >
            {JSON.stringify(result, null, 2)}
          </pre>
        </>
      )}
    </div>
  );
}

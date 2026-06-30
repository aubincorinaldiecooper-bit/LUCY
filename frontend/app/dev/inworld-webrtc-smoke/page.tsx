"use client";


import { useEffect, useRef, useState } from "react";


/**

 * Dev-only smoke test: bypass LiveKit entirely and connect to Inworld's

 * Realtime WebRTC endpoint directly. Answers one question:

 *   "Can Inworld produce audible Luna audio through its own WebRTC path?"

 *

 * Lives at /dev/inworld-webrtc-smoke. Audio is delivered as a remote RTP

 * audio track on the RTCPeerConnection — NOT as data-channel audio deltas.

 * We do not read audio bytes off the data channel; we attach the remote

 * track to an <audio> element and let the browser play it.

 */


type ConnState = "idle" | "starting" | "ice_fetched" | "mic_granted" | "pc_created" | "dc_open" | "session_updated" | "ready" | "error" | "stopped";


interface Status {

  iceFetched: boolean;

  iceRaw: unknown;

  iceError: string | null;

  micGranted: boolean;

  pcState: RTCPeerConnectionState | "n/a";

  iceConnState: RTCIceConnectionState | "n/a";

  signalingState: RTCSignalingState | "n/a";

  dcState: string;

  sessionUpdated: boolean;

  responseCreated: boolean;

  remoteAudioTrackId: string | null;

  remoteStreamAttached: boolean;

  inboundPackets: number;

  inboundBytes: number;

  inboundJitterMs: number | null;

  inboundAudioLevel: number | null;

  lastEventType: string;

  lastErrorJson: string | null;

  callerLog: string[];

  audible: "untested" | "yes" | "no";

}


const INITIAL: Status = {

  iceFetched: false,

  iceRaw: null,

  iceError: null,

  micGranted: false,

  pcState: "n/a",

  iceConnState: "n/a",

  signalingState: "n/a",

  dcState: "closed",

  sessionUpdated: false,

  responseCreated: false,

  remoteAudioTrackId: null,

  remoteStreamAttached: false,

  inboundPackets: 0,

  inboundBytes: 0,

  inboundJitterMs: null,

  inboundAudioLevel: null,

  lastEventType: "none",

  lastErrorJson: null,

  callerLog: [],

  audible: "untested",

};


export default function InworldWebRTCSmokePage() {

  const [connState, setConnState] = useState<ConnState>("idle");

  const [status, setStatus] = useState<Status>(INITIAL);

  const statusRef = useRef(status);

  statusRef.current = status;

  const pcRef = useRef<RTCPeerConnection | null>(null);

  const dcRef = useRef<RTCDataChannel | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const localStreamRef = useRef<MediaStream | null>(null);

  const statsIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);


  const log = (msg: string) => {

    setStatus((s) => ({

      ...s,

      callerLog: [...s.callerLog.slice(-200), `${new Date().toISOString().slice(11, 23)}  ${msg}`],

    }));

  };


  const setStatusPatch = (patch: Partial<Status>) =>

    setStatus((s) => ({ ...s, ...patch }));


  const start = async () => {

    if (connState !== "idle" && connState !== "stopped" && connState !== "error") return;

    setStatus(INITIAL);

    setConnState("starting");

    log("[inworld-webrtc-smoke] starting");


    // -- 1. Fetch ICE servers

    let iceServers: RTCIceServer[] = [];

    try {

      log("[ice] GET /api/inworld/webrtc/ice");

      const iceResp = await fetch("/api/inworld/webrtc/ice", { method: "GET" });

      const iceText = await iceResp.text();

      log(`[ice] status=${iceResp.status} bytes=${iceText.length}`);

      if (!iceResp.ok) {

        setStatusPatch({ iceError: `ice_fetch_failed status=${iceResp.status} body=${iceText.slice(0, 200)}` });

        log(`[ice] ${statusRef.current.iceError}`);

        setConnState("error");

        return;

      }

      const iceBody = JSON.parse(iceText) as unknown;

      // Inworld ICE endpoint response shapes we've observed in docs / logs:

      //   1. bare array:          [{ urls: ... }, ...]

      //   2. camelCase wrapper:    { iceServers: [...] }

      //   3. snake_case wrapper:   { ice_servers: [...] }

      // We accept any of the three so a server-side rename doesn't silently

      // leave us with a peer connection that has zero ICE servers.

      const ics = (obj: unknown, key: string): RTCIceServer[] | null => {

        if (obj && typeof obj === "object") {

          const v = (obj as Record<string, unknown>)[key];

          if (Array.isArray(v)) return v as RTCIceServer[];

        }

        return null;

      };

      if (Array.isArray(iceBody)) {

        iceServers = iceBody as RTCIceServer[];

      } else if (iceBody && typeof iceBody === "object") {

        const fromCamel = ics(iceBody, "iceServers");

        const fromSnake = ics(iceBody, "ice_servers");

        iceServers = fromCamel ?? fromSnake ?? [];

      } else {

        iceServers = [];

      }

      setStatusPatch({ iceFetched: true, iceRaw: iceBody });

      setConnState("ice_fetched");

      log(`[ice] fetched ${iceServers.length} server(s)`);

    } catch (err) {

      const msg = `ice_fetch_exception: ${String(err)}`;

      setStatusPatch({ iceError: msg });

      log(`[ice] ${msg}`);

      setConnState("error");

      return;

    }


    // -- 2. Get mic

    try {

      log("[mic] getUserMedia {audio:true}");

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      localStreamRef.current = stream;

      setStatusPatch({ micGranted: true });

      setConnState("mic_granted");

      log("[mic] granted");

    } catch (err) {

      const msg = `mic_getUserMedia_failed: ${String(err)}`;

      log(`[mic] ${msg}`);

      setConnState("error");

      return;

    }


    // -- 3. Create peer connection

    const pc = new RTCPeerConnection({ iceServers });

    pcRef.current = pc;

    setConnState("pc_created");

    // Initial snapshot so the UI table shows real state immediately, not "n/a".

    setStatusPatch({

      pcState: pc.connectionState,

      iceConnState: pc.iceConnectionState,

      signalingState: pc.signalingState,

    });

    log(`[pc] created with ${iceServers.length} ICE server(s) signalingState=${pc.signalingState}`);


    // -- 4. Add mic track

    try {

      const audioTracks = localStreamRef.current!.getAudioTracks();

      for (const track of audioTracks) {

        pc.addTrack(track, localStreamRef.current!);

      }

      log(`[pc] added ${audioTracks.length} mic track(s)`);

    } catch (err) {

      log(`[pc] addTrack_failed: ${String(err)}`);

    }


    // -- 5. Create data channel "oai-events"

    const dc = pc.createDataChannel("oai-events");

    dcRef.current = dc;

    setStatusPatch({ dcState: dc.readyState });

    log(`[dc] created name=oai-events initial_state=${dc.readyState}`);


    dc.onopen = () => {

      log("[dc] open");

      setStatusPatch({ dcState: dc.readyState });

      setConnState("dc_open");

      // Send session.update

      const sessionUpdate = {

        type: "session.update",

        session: {

          type: "realtime",

          model: "groq/openai/gpt-oss-120b",

          instructions:

            "You are Arche, a concise warm voice companion. Reply naturally and briefly.",

          output_modalities: ["audio"],

          audio: {

            input: {

              transcription: { model: "assemblyai/u3-rt-pro" },

              turn_detection: {

                type: "semantic_vad",

                eagerness: "low",

                create_response: true,

                interrupt_response: true,

              },

            },

            output: {

              model: "inworld-tts-2",

              voice: "Luna",

              speed: 1.0,

            },

          },

          providerData: {

            stt: { voice_profile: false },

            tts: {

              delivery_mode: "CREATIVE",

              segmenter_strategy: "full_turn",

              steering_handling: "emit_once",

            },

          },

        },

      };

      dc.send(JSON.stringify(sessionUpdate));

      log("[session.update] sent");

    };

    dc.onmessage = (ev) => {

      const data = typeof ev.data === "string" ? ev.data : "(non-string)";

      let parsed: { type?: string } | null = null;

      try {

        parsed = JSON.parse(data) as { type?: string };

      } catch {

        // not JSON; that's fine, just log the snippet.

      }

      const type = parsed?.type ?? "non_json";

      setStatusPatch({

        lastEventType: type,

        sessionUpdated: statusRef.current.sessionUpdated || type === "session.updated",

        responseCreated: statusRef.current.responseCreated || type === "response.created",

        lastErrorJson: type === "error" ? data : statusRef.current.lastErrorJson,

      });

      log(`[dc] message type=${type} bytes=${data.length}`);


      if (type === "session.updated") {

        log("[handshake] session.updated -> conversation.item.create + response.create");

        const itemCreate = {

          type: "conversation.item.create",

          item: {

            type: "message",

            role: "user",

            content: [

              { type: "input_text", text: "Say hello in one short sentence." },

            ],

          },

        };

        const responseCreate = {

          type: "response.create",

          response: {

            output_modalities: ["audio"],

            instructions: "Reply out loud in one short sentence.",

          },

        };

        try {

          dc.send(JSON.stringify(itemCreate));

          log("[handshake] conversation.item.create sent");

        } catch (e) {

          log(`[handshake] itemCreate_send_failed: ${String(e)}`);

        }

        try {

          dc.send(JSON.stringify(responseCreate));

          log("[handshake] response.create sent");

          setConnState("ready");

        } catch (e) {

          log(`[handshake] responseCreate_send_failed: ${String(e)}`);

        }

      }

    };

    dc.onerror = (ev) => {

      // RTCErrorEvent has no .error property on some browsers; coerce safely.

      const errMsg = (ev as unknown as { error?: unknown }).error ?? "(no detail)";

      log(`[dc] error: ${String(errMsg)}`);

    };

    dc.onclose = () => {

      log(`[dc] close readyState=${dc.readyState}`);

      setStatusPatch({ dcState: dc.readyState });

    };


    // -- 6. Remote track

    pc.ontrack = (ev) => {

      const t = ev.track;

      log(

        `[inworld_webrtc_remote_track=true] kind=${t.kind} id=${t.id} readyState=${t.readyState}`,

      );

      setStatusPatch({

        remoteAudioTrackId: t.id,

      });

      if (!audioRef.current) {

        log("[audio] no <audio> ref on the page");

        return;

      }

      // Explicit MediaStream construction from the remote track — clearer and

      // gives us a single track handle we can introspect.

      const remoteStream = new MediaStream([t]);

      audioRef.current.srcObject = remoteStream;

      setStatusPatch({ remoteStreamAttached: true });

      log(`[audio] attached remote MediaStream id=${remoteStream.id} tracks=${remoteStream.getTracks().length}`);

      audioRef.current.play().then(

        () => log("[audio] play() resolved"),

        (e) => log(`[audio] play() rejected: ${String(e)}`),

      );

    };


    pc.onicecandidate = (ev) => {

      if (!ev.candidate) {

        log("[pc] ICE gathering complete (null candidate)");

      }

    };

    pc.oniceconnectionstatechange = () => {

      log(`[pc] iceConnectionState=${pc.iceConnectionState}`);

      setStatusPatch({ iceConnState: pc.iceConnectionState });

    };

    pc.onconnectionstatechange = () => {

      log(`[pc] connectionState=${pc.connectionState}`);

      setStatusPatch({ pcState: pc.connectionState });

    };

    pc.onsignalingstatechange = () => {

      log(`[pc] signalingState=${pc.signalingState}`);

      setStatusPatch({ signalingState: pc.signalingState });

    };


    // -- 7. SDP offer/answer

    try {

      const offer = await pc.createOffer();

      await pc.setLocalDescription(offer);

      log(`[sdp] createOffer + setLocalDescription done (${offer.sdp.length} bytes)`);


      // Wait for ICE gathering to complete so the offer has all candidates.

      const offerWithCandidates = await new Promise<RTCSessionDescriptionInit>(

        (resolve, reject) => {

          if (pc.iceGatheringState === "complete") {

            resolve(pc.localDescription!);

            return;

          }

          const check = () => {

            if (pc.iceGatheringState === "complete") {

              pc.removeEventListener("icegatheringstatechange", check);

              resolve(pc.localDescription!);

            }

          };

          pc.addEventListener("icegatheringstatechange", check);

          // Safety timeout after 4s; Inworld probably accepts trickle ICE too,

          // but the docs example in the user prompt asks for full gathering.

          setTimeout(() => {

            pc.removeEventListener("icegatheringstatechange", check);

            if (pc.localDescription) resolve(pc.localDescription);

            else reject(new Error("ice gathering timed out with no localDescription"));

          }, 4000);

        },

      );


      log(`[sdp] posting offer to /api/inworld/webrtc/call (${offerWithCandidates.sdp!.length} bytes)`);

      const callResp = await fetch("/api/inworld/webrtc/call", {

        method: "POST",

        headers: { "Content-Type": "application/sdp" },

        body: offerWithCandidates.sdp!,

      });

      const answerText = await callResp.text();

      log(`[sdp] answer received status=${callResp.status} bytes=${answerText.length}`);

      if (!callResp.ok) {

        log(`[sdp] call_failed body=${answerText.slice(0, 200)}`);

        setConnState("error");

        return;

      }

      await pc.setRemoteDescription({ type: "answer", sdp: answerText });

      log(`[sdp] setRemoteDescription done`);

    } catch (err) {

      log(`[sdp] exception: ${String(err)}`);

      setConnState("error");

      return;

    }


    // -- 8. getStats polling

    if (statsIntervalRef.current) clearInterval(statsIntervalRef.current);

    statsIntervalRef.current = setInterval(async () => {

      const pc = pcRef.current;

      if (!pc) return;

      try {

        const statsReport = await pc.getStats();

        let packets = 0;

        let bytes = 0;

        let jitter: number | null = null;

        let audioLevel: number | null = null;

        let rtt: number | null = null;

        statsReport.forEach((report) => {

          if (report.type === "inbound-rtp" && (report as { kind?: string }).kind === "audio") {

            const r = report as { packetsReceived?: number; bytesReceived?: number; jitter?: number; audioLevel?: number };

            packets += r.packetsReceived ?? 0;

            bytes += r.bytesReceived ?? 0;

            if (typeof r.jitter === "number") jitter = r.jitter * 1000; // seconds -> ms

            if (typeof r.audioLevel === "number") audioLevel = r.audioLevel;

          }

          if (report.type === "candidate-pair" && (report as { state?: string }).state === "succeeded") {

            const r = report as { currentRoundTripTime?: number };

            if (typeof r.currentRoundTripTime === "number") rtt = r.currentRoundTripTime * 1000;

          }

        });

        setStatusPatch({

          inboundPackets: packets,

          inboundBytes: bytes,

          inboundJitterMs: jitter,

          inboundAudioLevel: audioLevel,

        });

      } catch {

        // ignore getStats failures

      }

    }, 1000);

  };


  const stop = () => {

    if (statsIntervalRef.current) {

      clearInterval(statsIntervalRef.current);

      statsIntervalRef.current = null;

    }

    if (dcRef.current) {

      try { dcRef.current.close(); } catch { /* ignore */ }

      dcRef.current = null;

    }

    if (pcRef.current) {

      try { pcRef.current.close(); } catch { /* ignore */ }

      pcRef.current = null;

    }

    if (localStreamRef.current) {

      for (const t of localStreamRef.current.getTracks()) t.stop();

      localStreamRef.current = null;

    }

    if (audioRef.current) {

      audioRef.current.srcObject = null;

    }

    setConnState("stopped");

    log("[stop] all tracks/PC/DC closed");

  };


  useEffect(() => {

    return () => {

      stop();

    };

    // eslint-disable-next-line react-hooks/exhaustive-deps

  }, []);


  const Row = ({ label, val }: { label: string; val: React.ReactNode }) => (

    <tr>

      <td style={{ padding: "4px 8px", color: "#888" }}>{label}</td>

      <td style={{ padding: "4px 8px", fontFamily: "monospace" }}>{val}</td>

    </tr>

  );


  return (

    <div style={{ padding: 24, maxWidth: 1100, margin: "0 auto", fontFamily: "system-ui" }}>

      <h1 style={{ fontSize: 22, fontWeight: 700 }}>Inworld Realtime WebRTC smoke test</h1>

      <p style={{ color: "#666", marginBottom: 12 }}>

        Direct to <code>wss://api.inworld.ai/api/v1/realtime</code> — bypassing LiveKit.

        Goal: prove Inworld can produce audible Luna audio over its own WebRTC path.

      </p>


      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>

        <button

          onClick={start}

          disabled={["starting", "ice_fetched", "mic_granted", "pc_created", "dc_open", "session_updated", "ready"].includes(connState)}

          style={{ padding: "8px 14px" }}

        >

          Start smoke test

        </button>

        <button onClick={stop} style={{ padding: "8px 14px" }}>

          Stop

        </button>

        <span style={{ alignSelf: "center", color: "#444" }}>state: <b>{connState}</b></span>

      </div>


      <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: 16 }}>

        <tbody>

          <Row label="ICE fetched" val={status.iceFetched ? "✅" : "❌"} />

          <Row label="ICE server count" val={

            Array.isArray(status.iceRaw)

              ? status.iceRaw.length

              : Array.isArray((status.iceRaw as { iceServers?: unknown[] } | null)?.iceServers)

                ? ((status.iceRaw as { iceServers: unknown[] }).iceServers.length)

                : Array.isArray((status.iceRaw as { ice_servers?: unknown[] } | null)?.ice_servers)

                  ? ((status.iceRaw as { ice_servers: unknown[] }).ice_servers.length)

                  : "?"

          } />

          <Row label="mic granted" val={status.micGranted ? "✅" : "❌"} />

          <Row label="peer connection state" val={String(status.pcState)} />

          <Row label="ICE connection state" val={String(status.iceConnState)} />

          <Row label="signaling state" val={String(status.signalingState)} />

          <Row label="data channel state" val={status.dcState} />

          <Row label="session.updated received" val={status.sessionUpdated ? "✅" : "—"} />

          <Row label="response.created received" val={status.responseCreated ? "✅" : "—"} />

          <Row label="remote audio track id" val={status.remoteAudioTrackId ?? "—"} />

          <Row label="remote MediaStream attached" val={status.remoteStreamAttached ? "✅" : "—"} />

          <Row label="inbound RTP packets" val={status.inboundPackets} />

          <Row label="inbound RTP bytes" val={status.inboundBytes} />

          <Row label="inbound jitter (ms)" val={status.inboundJitterMs ?? "—"} />

          <Row label="inbound audioLevel (0..1)" val={status.inboundAudioLevel ?? "—"} />

          <Row label="last event type" val={status.lastEventType} />

          <Row label="audible?" val={

            status.audible === "yes"

              ? "✅ YES"

              : status.audible === "no"

                ? "❌ NO"

                : "— (operator)"

          } />

        </tbody>

      </table>


      <div style={{ marginBottom: 16 }}>

        <label style={{ display: "block", marginBottom: 4, fontWeight: 600 }}>

          After playback: was Luna audible?

        </label>

        <button

          onClick={() => setStatusPatch({ audible: "yes" })}

          style={{ padding: "6px 12px", marginRight: 8 }}

        >

          YES — Luna is audible

        </button>

        <button

          onClick={() => setStatusPatch({ audible: "no" })}

          style={{ padding: "6px 12px" }}

        >

          NO — silent / only transcript

        </button>

      </div>


      <h2 style={{ fontSize: 16, marginTop: 8 }}>Console log</h2>

      <div

        style={{

          background: "#111",

          color: "#0f0",

          padding: 12,

          fontFamily: "monospace",

          fontSize: 12,

          height: 280,

          overflowY: "auto",

          whiteSpace: "pre-wrap",

          borderRadius: 4,

        }}

      >

        {status.callerLog.map((l, i) => <div key={i}>{l}</div>)}

      </div>


      {status.lastErrorJson && (

        <>

          <h2 style={{ fontSize: 16, marginTop: 16, color: "#b00" }}>Last Inworld error event</h2>

          <pre

            style={{

              background: "#fee",

              color: "#900",

              padding: 12,

              fontFamily: "monospace",

              fontSize: 12,

              overflowX: "auto",

              borderRadius: 4,

            }}

          >

            {status.lastErrorJson}

          </pre>

        </>

      )}


      {status.iceError && (

        <pre

          style={{

            background: "#fee",

            color: "#900",

            padding: 12,

            fontFamily: "monospace",

            fontSize: 12,

            overflowX: "auto",

            borderRadius: 4,

            marginTop: 12,

          }}

        >

          {status.iceError}

        </pre>

      )}


      <audio ref={audioRef} autoPlay playsInline style={{ display: "none" }} />

    </div>

  );

}


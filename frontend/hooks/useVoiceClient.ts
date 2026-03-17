"use client";

import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { useCallback, useRef, useState } from "react";

export type VoiceState = "idle" | "initializing" | "connecting" | "connected" | "muted";

export function useVoiceClient() {
  const [state, setState] = useState<VoiceState>("idle");
  const [isMuted, setIsMuted] = useState(false);
  const clientRef = useRef<PipecatClient | null>(null);

  const connect = useCallback(async (micDeviceId?: string) => {
    if (clientRef.current) {
      return;
    }

    setState("initializing");

    const transport = new SmallWebRTCTransport({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      enableMic: true,
      enableCam: false,
      ...(micDeviceId ? { micDeviceId } : {}),
    } as any);

    const client = new PipecatClient({
      transport,
      callbacks: {
        onTransportStateChanged: (s) => {
          if (s === "authenticating" || s === "connecting") {
            setState("connecting");
          }
        },
        onConnected: () => {
          setState("connected");
        },
        onDisconnected: () => {
          setState("idle");
          setIsMuted(false);
          clientRef.current = null;
        },
        onTrackStarted: (track, participant) => {
          if (participant?.local) {
            return;
          }

          let audioEl = document.getElementById("lucy-remote-audio") as HTMLAudioElement | null;
          if (!audioEl) {
            audioEl = document.createElement("audio");
            audioEl.id = "lucy-remote-audio";
            audioEl.autoplay = true;
            audioEl.playsInline = true;
            audioEl.style.display = "none";
            document.body.appendChild(audioEl);
          }

          audioEl.srcObject = new MediaStream([track]);
        },
        onError: () => {
          setState("idle");
          setIsMuted(false);
          clientRef.current = null;
        },
      },
    });

    clientRef.current = client;

    try {
      await client.connect({ webrtcUrl: "/api/offer" });
    } catch {
      setState("idle");
      setIsMuted(false);
      clientRef.current = null;
    }
  }, []);

  const disconnect = useCallback(async () => {
    await clientRef.current?.disconnect();
  }, []);

  const toggleMute = useCallback(() => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    clientRef.current?.enableMic(!nextMuted);
    setState(nextMuted ? "muted" : "connected");
  }, [isMuted]);

  return { state, connect, disconnect, toggleMute, isMuted };
}

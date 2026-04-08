"use client";

import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceState = "idle" | "initializing" | "connecting" | "connected" | "muted";

type VoiceTransportOptions = {
  iceServers: RTCIceServer[];
  enableMic: boolean;
  enableCam: boolean;
  micDeviceId?: string;
};

function resolveOfferUrl() {
  const rawApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/^['"]|['"]$/g, "");

  if (!rawApiUrl) {
    throw new Error("NEXT_PUBLIC_API_URL is not configured");
  }

  const baseUrl = /^https?:\/\//i.test(rawApiUrl) ? rawApiUrl : `https://${rawApiUrl}`;
  return new URL("/api/offer", baseUrl).toString();
}

export function useVoiceClient() {
  const [state, setState] = useState<VoiceState>("idle");
  const [isMuted, setIsMuted] = useState(false);
  const clientRef = useRef<PipecatClient | null>(null);
  const activeMicDeviceIdRef = useRef<string | undefined>(undefined);

  const cleanupAudioElement = useCallback(() => {
    const audioEl = document.getElementById("lucy-remote-audio") as HTMLAudioElement | null;
    if (!audioEl) {
      return;
    }

    audioEl.srcObject = null;
    audioEl.remove();
  }, []);

  const disconnect = useCallback(async () => {
    try {
      await clientRef.current?.disconnect();
    } finally {
      clientRef.current = null;
      activeMicDeviceIdRef.current = undefined;
      cleanupAudioElement();
    }
  }, [cleanupAudioElement]);

  const connect = useCallback(async (micDeviceId?: string) => {
    if (clientRef.current) {
      if (activeMicDeviceIdRef.current === micDeviceId) {
        return;
      }

      await disconnect();
    }

    setState("initializing");

    const transportOptions: VoiceTransportOptions = {
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      enableMic: true,
      enableCam: false,
      ...(micDeviceId ? { micDeviceId } : {}),
    };

    const transport = new SmallWebRTCTransport(transportOptions);

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
          activeMicDeviceIdRef.current = undefined;
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
            audioEl.setAttribute("playsinline", "true");
            audioEl.style.display = "none";
            document.body.appendChild(audioEl);
          }

          audioEl.srcObject = new MediaStream([track]);
        },
        onError: () => {
          setState("idle");
          setIsMuted(false);
          clientRef.current = null;
          activeMicDeviceIdRef.current = undefined;
        },
      },
    });

    clientRef.current = client;

    try {
      await client.connect({
        webrtcRequestParams: {
          endpoint: resolveOfferUrl(),
        },
      });
      activeMicDeviceIdRef.current = micDeviceId;
    } catch {
      setState("idle");
      setIsMuted(false);
      clientRef.current = null;
      activeMicDeviceIdRef.current = undefined;
    }
  }, [disconnect]);

  useEffect(() => {
    return () => {
      void disconnect();
      cleanupAudioElement();
    };
  }, [cleanupAudioElement, disconnect]);

  const toggleMute = useCallback(() => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    clientRef.current?.enableMic(!nextMuted);
    setState(nextMuted ? "muted" : "connected");
  }, [isMuted]);

  return { state, connect, disconnect, toggleMute, isMuted };
}

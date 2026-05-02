"use client";

import DailyIframe, { type DailyCall } from "@daily-co/daily-js";
import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceState = "idle" | "initializing" | "connecting" | "connected" | "muted";

type DailySessionResponse = {
  room_url: string;
  token: string;
};

function resolveSessionUrl() {
  const rawApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/^['"]|['"]$/g, "");

  if (!rawApiUrl) {
    throw new Error("NEXT_PUBLIC_API_URL is not configured");
  }

  const baseUrl = /^https?:\/\//i.test(rawApiUrl) ? rawApiUrl : `https://${rawApiUrl}`;
  return new URL("/api/daily/session", baseUrl).toString();
}

async function createDailySession(modelId?: string): Promise<DailySessionResponse> {
  const response = await fetch(resolveSessionUrl(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(modelId ? { model_id: modelId } : {}),
  });

  if (!response.ok) {
    throw new Error(`Failed to create Daily session (${response.status})`);
  }

  return response.json() as Promise<DailySessionResponse>;
}

export function useVoiceClient() {
  const [state, setState] = useState<VoiceState>("idle");
  const [isMuted, setIsMuted] = useState(false);
  const callRef = useRef<DailyCall | null>(null);
  const joinStartedAtRef = useRef<number | null>(null);
  const remoteAudioElsRef = useRef<Map<string, HTMLAudioElement>>(new Map());

  const cleanupRemoteAudioEls = useCallback(() => {
    remoteAudioElsRef.current.forEach((audioEl) => {
      audioEl.pause();
      audioEl.srcObject = null;
    });
    remoteAudioElsRef.current.clear();
  }, []);

  const disconnect = useCallback(async () => {
    const call = callRef.current;

    if (!call) {
      return;
    }

    try {
      const meetingState = call.meetingState();
      if (meetingState === "joining-meeting" || meetingState === "joined-meeting") {
        await call.leave();
      }
    } finally {
      if (!call.isDestroyed()) {
        try {
          await call.destroy();
        } catch (error) {
          if (!(error instanceof DOMException && error.name === "InvalidStateError")) {
            throw error;
          }
        }
      }

      callRef.current = null;
      joinStartedAtRef.current = null;
      cleanupRemoteAudioEls();
      setState("idle");
      setIsMuted(false);
    }
  }, [cleanupRemoteAudioEls]);

  const connect = useCallback(async (modelId?: string) => {
    if (callRef.current) {
      return;
    }

    setState("initializing");

    try {
      const session = await createDailySession(modelId);
      const call = DailyIframe.createCallObject({
        audioSource: true,
        videoSource: false,
        startAudioOff: false,
        subscribeToTracksAutomatically: true,
      });

      (call as any).on("joined-meeting", () => {
        console.debug("[daily] joined meeting");
        setState("connecting");
      });

      (call as any).on("left-meeting", () => {
        console.debug("[daily] left meeting");
        setState("idle");
        setIsMuted(false);
        callRef.current = null;
        joinStartedAtRef.current = null;
        cleanupRemoteAudioEls();
      });

      (call as any).on("error", () => {
        console.debug("[daily] meeting error");
        setState("idle");
        setIsMuted(false);
        callRef.current = null;
        joinStartedAtRef.current = null;
        cleanupRemoteAudioEls();
      });

      (call as any).on("track-started", (event: { participant: { local: boolean } | null; track: MediaStreamTrack; type: string }) => {
        if (event.participant?.local || event.type !== "audio") {
          return;
        }

        const stream = new MediaStream([event.track]);
        const audio = new Audio();
        audio.autoplay = true;
        audio.srcObject = stream;
        void audio.play()
          .then(() => console.debug(`[daily] remote audio playing track=${event.track.id}`))
          .catch((error: unknown) => {
            if (error instanceof Error) {
              console.debug(`[daily] remote audio play() failed track=${event.track.id} name=${error.name} message=${error.message}`);
            } else {
              console.debug(`[daily] remote audio play() failed track=${event.track.id}`, error);
            }
          });
        remoteAudioElsRef.current.set(event.track.id, audio);

        // ✅ Latency tracking preserved from codex branch
        const joinStartedAt = joinStartedAtRef.current;
        if (joinStartedAt) {
          const firstResponseLatencyMs = performance.now() - joinStartedAt;
          if (firstResponseLatencyMs > 2000) {
            console.warn(`[daily] first response latency ${Math.round(firstResponseLatencyMs)}ms (>2000ms target)`);
          } else {
            console.debug(`[daily] first response latency ${Math.round(firstResponseLatencyMs)}ms`);
          }
          joinStartedAtRef.current = null;
        }

        setState(isMuted ? "muted" : "connected");
      });

      (call as any).on("track-stopped", (event: { track: MediaStreamTrack; type: string }) => {
        if (event.type !== "audio") return;
        const existingAudioEl = remoteAudioElsRef.current.get(event.track.id);
        if (!existingAudioEl) return;
        existingAudioEl.pause();
        existingAudioEl.srcObject = null;
        remoteAudioElsRef.current.delete(event.track.id);
      });

      callRef.current = call;
      setState("connecting");
      joinStartedAtRef.current = performance.now();

      await call.join({
        url: session.room_url,
        token: session.token,
        startAudioOff: false,
      });

      call.setLocalAudio(!isMuted);

    } catch {
      if (callRef.current && !callRef.current.isDestroyed()) {
        try {
          await callRef.current.destroy();
        } catch (error) {
          if (!(error instanceof DOMException && error.name === "InvalidStateError")) {
            throw error;
          }
        }
      }

      callRef.current = null;
      joinStartedAtRef.current = null;
      cleanupRemoteAudioEls();
      setIsMuted(false);
      setState("idle");
    }
  }, [cleanupRemoteAudioEls, isMuted]);

  useEffect(() => {
    return () => {
      void disconnect();
      cleanupRemoteAudioEls();
    };
  }, [cleanupRemoteAudioEls, disconnect]);

  const toggleMute = useCallback(() => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    callRef.current?.setLocalAudio(!nextMuted);
    setState(nextMuted ? "muted" : "connected");
  }, [isMuted]);

  return { state, connect, disconnect, toggleMute, isMuted };
}
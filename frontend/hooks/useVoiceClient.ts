"use client";

import DailyIframe, { type DailyCall } from "@daily-co/daily-js";
import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceState = "idle" | "initializing" | "connecting" | "connected" | "muted";

type DailySessionResponse = {
  room_url: string;
  token: string;
};

const REMOTE_AUDIO_ELEMENT_ID = "lucy-remote-audio";

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

  const ensureRemoteAudioElement = useCallback(() => {
    let audioEl = document.getElementById(REMOTE_AUDIO_ELEMENT_ID) as HTMLAudioElement | null;

    if (!audioEl) {
      audioEl = document.createElement("audio");
      audioEl.id = REMOTE_AUDIO_ELEMENT_ID;
      audioEl.autoplay = true;
      audioEl.setAttribute("playsinline", "true");
      audioEl.style.display = "none";
      document.body.appendChild(audioEl);
    }

    return audioEl;
  }, []);

  const clearRemoteAudioElement = useCallback(() => {
    const audioEl = document.getElementById(REMOTE_AUDIO_ELEMENT_ID) as HTMLAudioElement | null;
    if (!audioEl) {
      return;
    }

    audioEl.pause();
    audioEl.srcObject = null;
    audioEl.remove();
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
      clearRemoteAudioElement();
      setState("idle");
      setIsMuted(false);
    }
  }, [clearRemoteAudioElement]);

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
        setState(isMuted ? "muted" : "connected");
      });

      (call as any).on("left-meeting", () => {
        console.debug("[daily] left meeting");
        setState("idle");
        setIsMuted(false);
        callRef.current = null;
        clearRemoteAudioElement();
      });

      (call as any).on("error", () => {
        console.debug("[daily] meeting error");
        clearRemoteAudioElement();
        setState("idle");
        setIsMuted(false);
        callRef.current = null;
      });

      (call as any).on("track-started", (event: { participant: { local: boolean } | null; track: MediaStreamTrack; type: string }) => {
        if (event.participant?.local || event.type !== "audio") {
          return;
        }

        const audioEl = ensureRemoteAudioElement();
        audioEl.srcObject = new MediaStream([event.track]);
        void audioEl.play().catch((error) => {
          console.debug("[daily] remote audio playback blocked", error);
        });
      });

      (call as any).on("track-stopped", (event: { type: string }) => {
        if (event.type === "audio") {
          clearRemoteAudioElement();
        }
      });

      callRef.current = call;
      setState("connecting");

      await call.join({
        url: session.room_url,
        token: session.token,
        startAudioOff: false,
      });

      call.setLocalAudio(!isMuted);
      const audioEl = ensureRemoteAudioElement();
      void audioEl.play().catch((error) => {
        console.debug("[daily] remote audio element not ready to play yet", error);
      });

      setState(isMuted ? "muted" : "connected");
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
      clearRemoteAudioElement();
      setIsMuted(false);
      setState("idle");
    }
  }, [clearRemoteAudioElement, ensureRemoteAudioElement, isMuted]);

  useEffect(() => {
    return () => {
      void disconnect();
      clearRemoteAudioElement();
    };
  }, [clearRemoteAudioElement, disconnect]);

  const toggleMute = useCallback(() => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    callRef.current?.setLocalAudio(!nextMuted);
    setState(nextMuted ? "muted" : "connected");
  }, [isMuted]);

  return { state, connect, disconnect, toggleMute, isMuted };
}

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

async function createDailySession(): Promise<DailySessionResponse> {
  const response = await fetch(resolveSessionUrl(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
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
  const activeMicDeviceIdRef = useRef<string | undefined>(undefined);

  const disconnect = useCallback(async () => {
    const call = callRef.current;

    if (!call) {
      activeMicDeviceIdRef.current = undefined;
      return;
    }

    try {
      if (call.meetingState() === "joined-meeting") {
        await call.leave();
      }
    } finally {
      await call.destroy();
      callRef.current = null;
      activeMicDeviceIdRef.current = undefined;
      setState("idle");
      setIsMuted(false);
    }
  }, []);

  const connect = useCallback(async (micDeviceId?: string) => {
    if (callRef.current) {
      if (activeMicDeviceIdRef.current === micDeviceId) {
        return;
      }

      await disconnect();
    }

    setState("initializing");

    try {
      const session = await createDailySession();
      const call = DailyIframe.createCallObject({
        audioSource: true,
        videoSource: false,
      });

      (call as any).on("joined-meeting", () => {
        setState(isMuted ? "muted" : "connected");
      });

      (call as any).on("left-meeting", () => {
        setState("idle");
        setIsMuted(false);
        callRef.current = null;
        activeMicDeviceIdRef.current = undefined;
      });

      (call as any).on("error", () => {
        setState("idle");
        setIsMuted(false);
        callRef.current = null;
        activeMicDeviceIdRef.current = undefined;
      });

      callRef.current = call;
      setState("connecting");

      if (micDeviceId) {
        await call.setInputDevicesAsync({ audioDeviceId: micDeviceId });
      }

      await call.join({
        url: session.room_url,
        token: session.token,
      });

      activeMicDeviceIdRef.current = micDeviceId;
      call.setLocalAudio(!isMuted);
      setState(isMuted ? "muted" : "connected");
    } catch {
      if (callRef.current) {
        await callRef.current.destroy();
      }

      callRef.current = null;
      activeMicDeviceIdRef.current = undefined;
      setIsMuted(false);
      setState("idle");
    }
  }, [disconnect, isMuted]);

  useEffect(() => {
    return () => {
      void disconnect();
    };
  }, [disconnect]);

  const toggleMute = useCallback(() => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    callRef.current?.setLocalAudio(!nextMuted);
    setState(nextMuted ? "muted" : "connected");
  }, [isMuted]);

  return { state, connect, disconnect, toggleMute, isMuted };
}

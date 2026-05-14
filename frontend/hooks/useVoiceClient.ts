"use client";

import { RemoteTrack, Room, RoomEvent, Track } from "livekit-client";
import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceState = "idle" | "initializing" | "connecting" | "connected" | "muted";

type SessionResponse = { room_url: string; token: string };

function resolveSessionUrl() {
  const rawApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/^['"]|['"]$/g, "");
  if (!rawApiUrl) throw new Error("NEXT_PUBLIC_API_URL is not configured");
  const baseUrl = /^https?:\/\//i.test(rawApiUrl) ? rawApiUrl : `https://${rawApiUrl}`;
  return new URL("/api/livekit/session", baseUrl).toString();
}

async function createSession(model?: string): Promise<SessionResponse> {
  const response = await fetch(resolveSessionUrl(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(model ? { model } : {}),
  });
  if (!response.ok) throw new Error(`Failed to create session (${response.status})`);
  return response.json() as Promise<SessionResponse>;
}

export function useVoiceClient() {
  const [state, setState] = useState<VoiceState>("idle");
  const [isMuted, setIsMuted] = useState(false);
  const roomRef = useRef<Room | null>(null);
  const remoteAudioElsRef = useRef<Set<HTMLMediaElement>>(new Set());

  const clearRemoteAudioElements = useCallback(() => {
    for (const el of remoteAudioElsRef.current) {
      el.remove();
    }
    remoteAudioElsRef.current.clear();
  }, []);

  const disconnect = useCallback(async () => {
    const room = roomRef.current;
    if (!room) return;
    await room.disconnect();
    clearRemoteAudioElements();
    roomRef.current = null;
    setState("idle");
    setIsMuted(false);
  }, [clearRemoteAudioElements]);

  const connect = useCallback(async (model?: string) => {
    if (roomRef.current) return;
    setState("initializing");
    try {
      const session = await createSession(model);
      const room = new Room();
      roomRef.current = room;
      room.on(RoomEvent.Connected, () => setState(isMuted ? "muted" : "connected"));
      room.on(RoomEvent.Disconnected, () => {
        clearRemoteAudioElements();
        roomRef.current = null;
        setState("idle");
      });
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return;
        const audioElement = (track as RemoteTrack).attach();
        audioElement.autoplay = true;
        audioElement.setAttribute("playsinline", "true");
        audioElement.style.display = "none";
        document.body.appendChild(audioElement);
        remoteAudioElsRef.current.add(audioElement);
        audioElement.play().catch((err) => {
          console.warn("Remote audio autoplay was blocked by the browser", err);
        });
      });
      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return;
        const detachedEls = (track as RemoteTrack).detach();
        detachedEls.forEach((el) => {
          remoteAudioElsRef.current.delete(el);
          el.remove();
        });
      });
      setState("connecting");
      await room.connect(session.room_url, session.token);
      if (typeof room.startAudio === "function") {
        await room.startAudio().catch((err) => {
          console.warn("Failed to start room audio playback", err);
        });
      }
      await room.localParticipant.setMicrophoneEnabled(!isMuted);
    } catch {
      clearRemoteAudioElements();
      setState("idle");
      roomRef.current = null;
    }
  }, [clearRemoteAudioElements, isMuted]);

  const toggleMute = useCallback(async () => {
    const next = !isMuted;
    setIsMuted(next);
    await roomRef.current?.localParticipant.setMicrophoneEnabled(!next);
    if (roomRef.current) setState(next ? "muted" : "connected");
  }, [isMuted]);

  useEffect(() => () => {
    clearRemoteAudioElements();
    void disconnect();
  }, [clearRemoteAudioElements, disconnect]);

  return { state, connect, disconnect, toggleMute, isMuted };
}

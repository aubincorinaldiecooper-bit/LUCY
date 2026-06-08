"use client";

import { RemoteTrack, Room, RoomEvent, Track } from "livekit-client";
import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceState = "idle" | "initializing" | "connecting" | "connected" | "muted";

type SessionResponse = { room_url: string; token: string };

function getClientTimezone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
}

function resolveSessionUrl() {
  const rawApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/^['"]|['"]$/g, "");
  if (!rawApiUrl) throw new Error("NEXT_PUBLIC_API_URL is not configured");
  const baseUrl = /^https?:\/\//i.test(rawApiUrl) ? rawApiUrl : `https://${rawApiUrl}`;
  return new URL("/api/livekit/session", baseUrl).toString();
}

async function createSession(model?: string): Promise<SessionResponse> {
  const client_timezone = getClientTimezone();
  const payload = { ...(model ? { model } : {}), client_timezone };
  if (process.env.NODE_ENV === "development") {
    console.debug("LiveKit session timezone payload", {
      client_timezone,
      session_payload_keys: Object.keys(payload),
    });
  }
  const response = await fetch(resolveSessionUrl(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`Failed to create session (${response.status})`);
  return response.json() as Promise<SessionResponse>;
}

export function useVoiceClient() {
  const [state, setState] = useState<VoiceState>("idle");
  const [isMuted, setIsMuted] = useState(false);
  const roomRef = useRef<Room | null>(null);
  const connectAttemptRef = useRef(0);
  const remoteAudioElsRef = useRef<Set<HTMLMediaElement>>(new Set());
  const audioContextRef = useRef<AudioContext | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const mediaSourceNodesRef = useRef<Map<HTMLMediaElement, MediaElementAudioSourceNode>>(new Map());

  const getRemoteAudioGain = useCallback(() => {
    const raw = Number.parseFloat(process.env.NEXT_PUBLIC_REMOTE_AUDIO_GAIN ?? "1.35");
    if (!Number.isFinite(raw)) return 1.35;
    return Math.max(1.0, Math.min(2.0, raw));
  }, []);

  const setupAudioGainForElement = useCallback((audioElement: HTMLMediaElement) => {
    if (typeof window === "undefined") return;
    const AudioContextCtor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) return;

    try {
      const audioContext = audioContextRef.current ?? new AudioContextCtor();
      audioContextRef.current = audioContext;

      let gainNode = gainNodeRef.current;
      if (!gainNode) {
        gainNode = audioContext.createGain();
        gainNode.connect(audioContext.destination);
        gainNodeRef.current = gainNode;
      }
      gainNode.gain.value = getRemoteAudioGain();
      if (!mediaSourceNodesRef.current.has(audioElement)) {
        const sourceNode = audioContext.createMediaElementSource(audioElement);
        sourceNode.connect(gainNode);
        mediaSourceNodesRef.current.set(audioElement, sourceNode);
      }

      if (audioContext.state === "suspended") {
        void audioContext.resume().catch(() => {
          // Browser autoplay policies may delay resume until user gesture.
        });
      }
    } catch (err) {
      console.warn("Failed to setup remote audio gain", err);
    }
  }, [getRemoteAudioGain]);

  const cleanupAudioNodeForElement = useCallback((audioElement: HTMLMediaElement) => {
    const sourceNode = mediaSourceNodesRef.current.get(audioElement);
    if (!sourceNode) return;
    try {
      sourceNode.disconnect();
    } catch {
      // best-effort disconnect
    }
    mediaSourceNodesRef.current.delete(audioElement);
  }, []);

  const teardownAudioGainResources = useCallback(() => {
    mediaSourceNodesRef.current.forEach((sourceNode) => {
      try {
        sourceNode.disconnect();
      } catch {
        // best-effort disconnect
      }
    });
    mediaSourceNodesRef.current.clear();

    if (gainNodeRef.current) {
      try {
        gainNodeRef.current.disconnect();
      } catch {
        // best-effort disconnect
      }
      gainNodeRef.current = null;
    }

    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => {
        // best-effort close
      });
      audioContextRef.current = null;
    }
  }, []);

  const clearRemoteAudioElements = useCallback(() => {
    for (const el of remoteAudioElsRef.current) {
      cleanupAudioNodeForElement(el);
      el.remove();
    }
    remoteAudioElsRef.current.clear();
    teardownAudioGainResources();
  }, [cleanupAudioNodeForElement, teardownAudioGainResources]);

  const disconnect = useCallback(async () => {
    connectAttemptRef.current += 1;
    const room = roomRef.current;
    if (!room) {
      clearRemoteAudioElements();
      setState("idle");
      setIsMuted(false);
      return;
    }
    await room.disconnect();
    clearRemoteAudioElements();
    roomRef.current = null;
    setState("idle");
    setIsMuted(false);
  }, [clearRemoteAudioElements]);

  const connect = useCallback(async (model?: string) => {
    if (roomRef.current) return;
    const attemptId = connectAttemptRef.current + 1;
    connectAttemptRef.current = attemptId;
    setState("initializing");
    try {
      const session = await createSession(model);
      if (connectAttemptRef.current !== attemptId) return;
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
        audioElement.volume = 1.0;
        audioElement.setAttribute("playsinline", "true");
        audioElement.style.display = "none";
        document.body.appendChild(audioElement);
        remoteAudioElsRef.current.add(audioElement);
        setupAudioGainForElement(audioElement);
        audioElement.play().catch((err) => {
          console.warn("Remote audio autoplay was blocked by the browser", err);
        });
      });
      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return;
        const detachedEls = (track as RemoteTrack).detach();
        detachedEls.forEach((el) => {
          cleanupAudioNodeForElement(el);
          remoteAudioElsRef.current.delete(el);
          el.remove();
        });
      });
      setState("connecting");
      await room.connect(session.room_url, session.token);
      if (connectAttemptRef.current !== attemptId) {
        await room.disconnect();
        return;
      }
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
  }, [clearRemoteAudioElements, cleanupAudioNodeForElement, isMuted, setupAudioGainForElement]);

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

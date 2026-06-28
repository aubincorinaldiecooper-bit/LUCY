"use client";

import { RemoteTrack, Room, RoomEvent, Track } from "livekit-client";
import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceState = "idle" | "initializing" | "connecting" | "connected" | "muted";

type SessionResponse = { room_url: string; token: string };

function getClientTimezone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
}

function resolveSessionUrl() {
  // Call our same-origin BFF route (app/api/livekit/session). It validates the
  // Better Auth session here — where the cookie is readable — then forwards to
  // the Python backend with the verified user_id. Same-origin also means the
  // session cookie is sent automatically and no CORS is involved.
  return "/api/livekit/session";
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

export function useVoiceClient(options?: { onServerDisconnect?: () => void }) {
  const [state, setState] = useState<VoiceState>("idle");
  const [isMuted, setIsMuted] = useState(false);
  const roomRef = useRef<Room | null>(null);
  const connectAttemptRef = useRef(0);
  // Distinguish a disconnect WE triggered (End button / leaving) from one the
  // server initiated (e.g. the agent hit the session time limit and deleted the
  // room). Only the latter should fire onServerDisconnect.
  const userInitiatedDisconnectRef = useRef(false);
  const onServerDisconnectRef = useRef<(() => void) | undefined>(options?.onServerDisconnect);
  onServerDisconnectRef.current = options?.onServerDisconnect;
  // Single audible path for the agent's voice: each remote audio track is
  // attached to ONE <audio> element and played natively. We deliberately do not
  // also route the track through a parallel Web Audio graph. Running both a media
  // element and a MediaStream source for the same WebRTC track played two
  // overlapping, slightly-offset copies (the "duplicated voice" bug) whose tails
  // desynced and sounded clipped (the "tail cutoff" bug). One element == one copy
  // == a clean tail. Loudness is left at the source level; if a boost is needed it
  // must be added without introducing a second audible stream.
  const remoteAudioElsRef = useRef<Set<HTMLMediaElement>>(new Set());

  const clearRemoteAudioElements = useCallback(() => {
    for (const el of remoteAudioElsRef.current) {
      try {
        el.pause();
      } catch {
        // best-effort pause before teardown
      }
      el.srcObject = null;
      el.remove();
    }
    remoteAudioElsRef.current.clear();
  }, []);

  const disconnect = useCallback(async () => {
    connectAttemptRef.current += 1;
    userInitiatedDisconnectRef.current = true;
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
    userInitiatedDisconnectRef.current = false;
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
        // Server/agent ended the room (e.g. session time limit) rather than the
        // user pressing End — surface it so the UI can show the end screen.
        if (!userInitiatedDisconnectRef.current) {
          onServerDisconnectRef.current?.();
        }
        userInitiatedDisconnectRef.current = false;
      });
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return;
        // Tear down any previously attached remote audio element first so a
        // re-subscribe (reconnect, track republish) can never stack a second
        // audible copy. This app only ever has one remote audio track (the agent).
        clearRemoteAudioElements();
        const audioElement = (track as RemoteTrack).attach();
        audioElement.autoplay = true;
        audioElement.volume = 1.0;
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

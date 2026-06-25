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
  const remoteAudioElsRef = useRef<Set<HTMLMediaElement>>(new Set());
  const audioContextRef = useRef<AudioContext | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const compressorNodeRef = useRef<DynamicsCompressorNode | null>(null);
  const mediaSourceNodesRef = useRef<Map<HTMLMediaElement, AudioNode>>(new Map());

  const getRemoteAudioGain = useCallback(() => {
    // Makeup gain applied AFTER the compressor. Because the compressor tames
    // peaks, this can safely sit above 1.0 without clipping; default 2.0
    // (~+6 dB) and a 4.0 ceiling give real perceived loudness headroom.
    const raw = Number.parseFloat(process.env.NEXT_PUBLIC_REMOTE_AUDIO_GAIN ?? "2.0");
    if (!Number.isFinite(raw)) return 2.0;
    return Math.max(1.0, Math.min(4.0, raw));
  }, []);

  const setupAudioGainForTrack = useCallback((track: RemoteTrack, audioElement: HTMLMediaElement) => {
    if (typeof window === "undefined") return;
    const AudioContextCtor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) return;

    const mediaStreamTrack = track.mediaStreamTrack;
    if (!mediaStreamTrack) return;

    try {
      const audioContext = audioContextRef.current ?? new AudioContextCtor();
      audioContextRef.current = audioContext;

      // Signal chain: MediaStream source -> compressor -> makeup gain -> output.
      // The compressor raises perceived loudness (evens out the moderate Hume
      // output level) and protects against clipping when the makeup gain is
      // pushed above 1.0.
      let compressorNode = compressorNodeRef.current;
      if (!compressorNode) {
        compressorNode = audioContext.createDynamicsCompressor();
        compressorNode.threshold.value = -24; // dB: start compressing here
        compressorNode.knee.value = 30;        // soft knee for natural feel
        compressorNode.ratio.value = 4;         // gentle 4:1 compression
        compressorNode.attack.value = 0.003;
        compressorNode.release.value = 0.25;
        compressorNodeRef.current = compressorNode;
      }

      let gainNode = gainNodeRef.current;
      if (!gainNode) {
        gainNode = audioContext.createGain();
        compressorNode.connect(gainNode);
        gainNode.connect(audioContext.destination);
        gainNodeRef.current = gainNode;
      }
      gainNode.gain.value = getRemoteAudioGain();
      if (!mediaSourceNodesRef.current.has(audioElement)) {
        // Tap the track's MediaStream directly instead of the <audio> element.
        // createMediaElementSource inserts a media-element buffer whose
        // end-of-stream handling clips the tail of each response (confirmed:
        // server delivers full audio + trailing silence, cut is downstream).
        // A MediaStreamAudioSourceNode plays the raw track and avoids that.
        // The element stays attached but muted only to keep the WebRTC audio
        // pipeline pulling frames.
        const sourceNode = audioContext.createMediaStreamSource(new MediaStream([mediaStreamTrack]));
        sourceNode.connect(compressorNode);
        mediaSourceNodesRef.current.set(audioElement, sourceNode);
        audioElement.muted = true;
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

    if (compressorNodeRef.current) {
      try {
        compressorNodeRef.current.disconnect();
      } catch {
        // best-effort disconnect
      }
      compressorNodeRef.current = null;
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
        const audioElement = (track as RemoteTrack).attach();
        audioElement.autoplay = true;
        audioElement.volume = 1.0;
        audioElement.setAttribute("playsinline", "true");
        audioElement.style.display = "none";
        document.body.appendChild(audioElement);
        remoteAudioElsRef.current.add(audioElement);
        setupAudioGainForTrack(track as RemoteTrack, audioElement);
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
  }, [clearRemoteAudioElements, cleanupAudioNodeForElement, isMuted, setupAudioGainForTrack]);

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

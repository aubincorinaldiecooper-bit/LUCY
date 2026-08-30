"use client";

import { RemoteTrack, Room, RoomEvent, Track } from "livekit-client";
import { useCallback, useEffect, useRef, useState } from "react";

type VoiceState = "idle" | "initializing" | "connecting" | "connected" | "muted";

type SessionResponse = { room_url: string; token: string };

// Build marker. Bump `id` whenever client audio/playout behavior changes so that a
// glance at the browser console on connect confirms which bundle is actually live —
// frontend deploys and a stale browser cache are the usual reason a fix "isn't
// working." `audioPath: "single"` is the one-<audio>-element playout that removed the
// duplicated-voice / tail-cutoff artifact; if you don't see this line (or it shows an
// older id) when a call connects, the browser is still running a cached old bundle.
const FRONTEND_BUILD = { id: "2026-07-09-inworld-realtime-state-machine", audioPath: "single" } as const;

// The track name published by the Inworld Realtime bridge in Python:
//   `rtc.LocalAudioTrack.create_audio_track("arche-inworld-realtime", source)`
// We listen for this specific track so bridge → browser audio playback can be
// proven via console logs without polluting logs from any other remote audio.
const INWORLD_REALTIME_TRACK_NAME = "arche-inworld-realtime";

// One owner for the vision flag: gates the Camera button in ListeningPage and
// auto-publish here — with vision off, the hook must never publish a camera
// track and the UI must not offer the button. (The env access stays a literal
// so Next.js can inline it at build time.)
export const VISION_ENABLED = process.env.NEXT_PUBLIC_VISION_ENABLED === "true";

// Turning the camera on is sticky: it stays on (including across sessions on
// this device) until the user explicitly turns it off. Stored client-side only.
const CAMERA_PREFERENCE_STORAGE_KEY = "arche.cameraPreferred";

function readCameraPreference(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(CAMERA_PREFERENCE_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

function writeCameraPreference(on: boolean) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CAMERA_PREFERENCE_STORAGE_KEY, on ? "true" : "false");
  } catch {
    // Storage unavailable (private mode etc.): the choice just won't persist.
  }
}

function logInworldRealtime(
  event: string,
  extra?: Record<string, unknown>,
  level: "info" | "warn" = "info",
) {
  const fn = level === "warn" ? console.warn : console.info;
  const payload = `[LUCY inworld-realtime] ${event}` + (extra ? ` ${JSON.stringify(extra)}` : "");
  fn(payload);
}

function getClientTimezone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
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
  // Same-origin BFF route (app/api/livekit/session): it validates the Better
  // Auth session where the cookie is readable, then forwards to the Python
  // backend with the verified user_id. Same-origin also means the session
  // cookie is sent automatically and no CORS is involved.
  const response = await fetch("/api/livekit/session", {
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
  // Camera requires one explicit opt-in, then stays on — including across
  // sessions on this device — until the user explicitly turns it off. The
  // remembered choice is re-applied (visibly) on each connect; it is never
  // enabled without the user having chosen it at some point.
  const [isCameraOn, setIsCameraOn] = useState(false);
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
  // == a clean tail. Loudness IS boosted, but without a second audible stream:
  // setupAudioGainForElement re-routes each element through a single GainNode
  // (createMediaElementSource replaces the element's direct output rather than
  // duplicating it), to NEXT_PUBLIC_REMOTE_AUDIO_GAIN.
  const remoteAudioElsRef = useRef<Set<HTMLMediaElement>>(new Set());
  const remoteTrackAudioElsRef = useRef<Map<RemoteTrack, Set<HTMLMediaElement>>>(new Map());
  const audioContextRef = useRef<AudioContext | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const mediaSourceNodesRef = useRef<Map<HTMLMediaElement, MediaElementAudioSourceNode>>(new Map());
  const audioCleanupTimersRef = useRef<Map<HTMLMediaElement, ReturnType<typeof setTimeout>>>(new Map());
  const remoteAudioTailCleanupDelayMs = 500;

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
    remoteTrackAudioElsRef.current.clear();

    if (gainNodeRef.current) {
      try {
        gainNodeRef.current.disconnect();
      } catch {
        // best-effort disconnect before teardown
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

  const cleanupRemoteAudioElement = useCallback((el: HTMLMediaElement, delayMs = 0) => {
    const existingTimer = audioCleanupTimersRef.current.get(el);
    if (existingTimer) {
      clearTimeout(existingTimer);
      audioCleanupTimersRef.current.delete(el);
    }

    const runCleanup = () => {
      audioCleanupTimersRef.current.delete(el);
      cleanupAudioNodeForElement(el);
      remoteAudioElsRef.current.delete(el);
      remoteTrackAudioElsRef.current.forEach((trackEls, remoteTrack) => {
        trackEls.delete(el);
        if (trackEls.size === 0) {
          remoteTrackAudioElsRef.current.delete(remoteTrack);
        }
      });
      el.remove();
    };

    if (delayMs > 0) {
      const timer = setTimeout(runCleanup, delayMs);
      audioCleanupTimersRef.current.set(el, timer);
      return;
    }

    runCleanup();
  }, [cleanupAudioNodeForElement]);

  const clearRemoteAudioElements = useCallback(() => {
    audioCleanupTimersRef.current.forEach((timer) => clearTimeout(timer));
    audioCleanupTimersRef.current.clear();
    for (const el of remoteAudioElsRef.current) {
      cleanupRemoteAudioElement(el);
    }
    remoteAudioElsRef.current.clear();
    teardownAudioGainResources();
  }, [cleanupRemoteAudioElement, teardownAudioGainResources]);

  const disconnect = useCallback(async () => {
    connectAttemptRef.current += 1;
    userInitiatedDisconnectRef.current = true;
    const room = roomRef.current;
    if (room) {
      await room.disconnect();
      roomRef.current = null;
    }
    clearRemoteAudioElements();
    setState("idle");
    setIsMuted(false);
    setIsCameraOn(false);
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
      room.on(RoomEvent.Connected, () => {
        // Confirms the running bundle at a glance — `audioPath=single` means the
        // duplicated-voice / tail-cutoff fix is live; absence means a cached old bundle.
        console.info(
          `[LUCY build] ${FRONTEND_BUILD.id} · audioPath=${FRONTEND_BUILD.audioPath} ` +
            "(single native <audio> element — duplicated-voice/tail-cutoff fix live)",
        );
        setState(isMuted ? "muted" : "connected");
      });
      room.on(RoomEvent.Disconnected, () => {
        clearRemoteAudioElements();
        roomRef.current = null;
        setState("idle");
        setIsCameraOn(false);
        // Server/agent ended the room (e.g. session time limit) rather than the
        // user pressing End — surface it so the UI can show the end screen.
        if (!userInitiatedDisconnectRef.current) {
          onServerDisconnectRef.current?.();
        }
        userInitiatedDisconnectRef.current = false;
      });
      room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
        if (track.kind !== Track.Kind.Audio) return;
        const remoteTrack = track as RemoteTrack;
        // ``remoteTrack.name`` does not exist on the livekit-client TS types
        // (``RemoteTrack<Kind>`` does not declare it), even though the runtime
        // instance often has a backfilled name. Read from the publication
        // instead — ``RemoteTrackPublication.trackName`` is the documented
        // public API and is always present.
        const trackName = (publication as unknown as { trackName?: string } | undefined)?.trackName ?? "";
        const isInworldRealtime = trackName === INWORLD_REALTIME_TRACK_NAME;

        if (isInworldRealtime) {
          const trackSid =
            (publication as unknown as { trackSid?: string } | undefined)?.trackSid ?? "unknown";
          const participantIdentity =
            (participant as unknown as { identity?: string } | undefined)?.identity ?? "unknown";
          logInworldRealtime("track_subscribed", {
            track_name: trackName,
            track_kind: remoteTrack.kind,
            track_id: (remoteTrack as unknown as { id?: string }).id ?? "unknown",
            track_sid: trackSid,
            participant_identity: participantIdentity,
          });
          // Standalone alias for easy log-grep — matches the convention used by
          // the backend's ``inworld_arche_*`` flag names.
          logInworldRealtime("frontend_arche_audio_track_detected=true", {
            track_sid: trackSid,
            participant_identity: participantIdentity,
          });
        }

        // Tear down any previously attached remote audio element first so a
        // re-subscribe (reconnect, track republish) can never stack a second
        // audible copy. This app only ever has one remote audio track (the agent).
        clearRemoteAudioElements();
        const audioElement = remoteTrack.attach();
        audioElement.autoplay = true;
        audioElement.muted = false;
        audioElement.volume = 1.0;
        audioElement.setAttribute("playsinline", "true");
        audioElement.style.display = "none";
        document.body.appendChild(audioElement);
        remoteAudioElsRef.current.add(audioElement);
        const trackEls = remoteTrackAudioElsRef.current.get(remoteTrack) ?? new Set<HTMLMediaElement>();
        trackEls.add(audioElement);
        remoteTrackAudioElsRef.current.set(remoteTrack, trackEls);
        setupAudioGainForElement(audioElement);

        if (isInworldRealtime) {
          logInworldRealtime("audio_element_attached", {
            track_name: trackName,
            audio_element_count: remoteAudioElsRef.current.size,
            audio_element_muted: audioElement.muted,
            audio_element_volume: audioElement.volume,
          });
        }

        audioElement.addEventListener(
          "play",
          () => {
            if (isInworldRealtime) {
              logInworldRealtime("audio_playback_started", {
                track_name: trackName,
                current_time: audioElement.currentTime,
                duration: audioElement.duration,
                audio_context_state: audioContextRef.current?.state ?? "unknown",
              });
            }
          },
          { once: true },
        );
        audioElement.addEventListener(
          "ended",
          () => {
            if (isInworldRealtime) {
              logInworldRealtime("audio_playback_completed", {
                track_name: trackName,
                duration: audioElement.duration,
                audio_context_state: audioContextRef.current?.state ?? "unknown",
              });
            }
            cleanupRemoteAudioElement(audioElement, remoteAudioTailCleanupDelayMs);
          },
          { once: true },
        );
        audioElement.play().then(
          () => {
            if (isInworldRealtime) {
              logInworldRealtime("audio_play_success", {
                track_name: trackName,
                audio_context_state: audioContextRef.current?.state ?? "unknown",
              });
              // Standalone alias for easy log-grep.
              logInworldRealtime("frontend_arche_audio_play_success=true", {
                audio_context_state: audioContextRef.current?.state ?? "unknown",
              });
            }
          },
          (err) => {
            if (isInworldRealtime) {
              logInworldRealtime("audio_playback_blocked", { error: String(err) }, "warn");
            } else {
              console.warn("Remote audio autoplay was blocked by the browser", err);
            }
          },
        );
      });
      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return;
        const remoteTrack = track as RemoteTrack;
        const attachedEls = remoteTrackAudioElsRef.current.get(remoteTrack);
        const cleanupAttachedEl = (el: HTMLMediaElement) => {
          const timer = setTimeout(() => {
            try {
              remoteTrack.detach(el);
            } catch {
              // best-effort detach after tail hold
            }
            cleanupRemoteAudioElement(el);
          }, remoteAudioTailCleanupDelayMs);
          audioCleanupTimersRef.current.set(el, timer);
        };

        if (attachedEls?.size) {
          attachedEls.forEach(cleanupAttachedEl);
          remoteTrackAudioElsRef.current.delete(remoteTrack);
          return;
        }

        // Fallback for any element LiveKit knows about that was not tracked above.
        setTimeout(() => {
          remoteTrack.detach().forEach((el) => cleanupRemoteAudioElement(el));
        }, remoteAudioTailCleanupDelayMs);
      });
      setState("connecting");
      await room.connect(session.room_url, session.token);
      if (connectAttemptRef.current !== attemptId) {
        await room.disconnect();
        return;
      }
      if (typeof room.startAudio === "function") {
        logInworldRealtime("room_start_audio_invoked", {
          audio_context_state_before: audioContextRef.current?.state ?? "unknown",
        });
        await room.startAudio().catch((err) => {
          logInworldRealtime("room_start_audio_failed", { error: String(err) }, "warn");
          console.warn("Failed to start room audio playback", err);
        });
        logInworldRealtime("room_start_audio_resolved", {
          audio_context_state_after: audioContextRef.current?.state ?? "unknown",
        });
      }
      await room.localParticipant.setMicrophoneEnabled(!isMuted);
      // Re-apply the remembered camera choice: a user who turned the camera on
      // keeps it on for future sessions until they turn it off. The visible
      // "Camera on" state + status line make this never covert.
      if (VISION_ENABLED && readCameraPreference()) {
        try {
          await room.localParticipant.setCameraEnabled(true);
          setIsCameraOn(true);
        } catch (err) {
          // Permission revoked / camera gone since last time: stay off.
          console.warn("Failed to re-enable camera from remembered preference", err);
          setIsCameraOn(false);
        }
      }
    } catch {
      clearRemoteAudioElements();
      setState("idle");
      roomRef.current = null;
    }
  }, [clearRemoteAudioElements, cleanupRemoteAudioElement, isMuted, setupAudioGainForElement]);

  const toggleMute = useCallback(async () => {
    const next = !isMuted;
    setIsMuted(next);
    await roomRef.current?.localParticipant.setMicrophoneEnabled(!next);
    if (roomRef.current) setState(next ? "muted" : "connected");
  }, [isMuted]);

  // Publishes/unpublishes the camera track over the existing LiveKit room. The
  // agent-side bridge samples it for vision context only when the backend's
  // VISION_CONTEXT_ENABLED flag is on; with it off a published track is ignored.
  // The chosen state is remembered (localStorage) and re-applied on future
  // connects until the user toggles it off.
  const toggleCamera = useCallback(async () => {
    const room = roomRef.current;
    if (!room) return;
    const next = !isCameraOn;
    try {
      await room.localParticipant.setCameraEnabled(next);
      setIsCameraOn(next);
      writeCameraPreference(next);
    } catch (err) {
      // Permission denied / no camera: stay off rather than showing a lying
      // "camera on" state, and don't persist an ON choice that never took
      // effect.
      console.warn("Failed to toggle camera", err);
      setIsCameraOn(false);
      if (!next) writeCameraPreference(false);
    }
  }, [isCameraOn]);

  useEffect(() => () => {
    clearRemoteAudioElements();
    void disconnect();
  }, [clearRemoteAudioElements, disconnect]);

  return { state, connect, disconnect, toggleMute, isMuted, toggleCamera, isCameraOn };
}


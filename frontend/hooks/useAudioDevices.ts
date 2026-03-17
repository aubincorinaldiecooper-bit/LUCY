"use client";

import { useEffect, useState } from "react";

export function useAudioDevices() {
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [speakers, setSpeakers] = useState<MediaDeviceInfo[]>([]);

  useEffect(() => {
    let mounted = true;
    let stream: MediaStream | null = null;

    const refreshDevices = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (!mounted) {
          return;
        }
        setMics(devices.filter((device) => device.kind === "audioinput"));
        setSpeakers(devices.filter((device) => device.kind === "audiooutput"));
      } catch {
        if (!mounted) {
          return;
        }
        setMics([]);
        setSpeakers([]);
      }
    };

    const init = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch {
        setMics([]);
        setSpeakers([]);
        return;
      }
      await refreshDevices();
    };

    init();
    navigator.mediaDevices.addEventListener("devicechange", refreshDevices);

    return () => {
      mounted = false;
      navigator.mediaDevices.removeEventListener("devicechange", refreshDevices);
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return { mics, speakers };
}

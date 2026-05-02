"use client";

import { useCallback, useMemo, useRef, useState } from "react";

export function useWaveform(barCount = 15, _active = false) {
  const [barHeights, setBarHeights] = useState<number[]>(() => Array(barCount).fill(4));
  const [micAmplitude, setMicAmplitude] = useState(0);
  const rafIdRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const smoothedAmplitudeRef = useRef(0);

  const resetBars = useMemo(() => Array(barCount).fill(4), [barCount]);

  const stopWaveform = useCallback(async () => {
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    analyserRef.current = null;

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }

    smoothedAmplitudeRef.current = 0;
    setMicAmplitude(0);
    setBarHeights(resetBars);
  }, [resetBars]);

  const startWaveform = useCallback(
    async (stream: MediaStream) => {
      await stopWaveform();

      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 64;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const tick = () => {
        if (!analyserRef.current) {
          return;
        }

        analyserRef.current.getByteFrequencyData(dataArray);

        const nextHeights = Array.from({ length: barCount }, (_, index) => {
          const val = dataArray[index % dataArray.length] ?? 0;
          return Math.max(4, (val / 255) * 44);
        });
        setBarHeights(nextHeights);

        const sum = dataArray.reduce((acc, value) => acc + value, 0);
        const rawAmplitude = dataArray.length ? sum / dataArray.length / 255 : 0;
        const smoothed = smoothedAmplitudeRef.current * 0.8 + rawAmplitude * 0.2;
        smoothedAmplitudeRef.current = smoothed;
        setMicAmplitude(Math.min(1, smoothed));

        rafIdRef.current = requestAnimationFrame(tick);
      };

      tick();
    },
    [barCount, stopWaveform]
  );

  return { barHeights, micAmplitude, startWaveform, stopWaveform };
}

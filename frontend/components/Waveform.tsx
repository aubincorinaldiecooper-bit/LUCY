"use client";

import { motion } from "framer-motion";

type WaveformProps = {
  barHeights: number[];
  active: boolean;
};

export default function Waveform({ barHeights, active }: WaveformProps) {
  return (
    <div className="w-full min-h-[52px] px-4 bg-ctrl-bg border border-ctrl-border rounded-[10px] flex items-center justify-center gap-[3px] overflow-hidden">
      {barHeights.map((height, index) => (
        <motion.span
          key={index}
          className="rounded-full bg-[var(--text-primary)] w-[2px] sm:w-[3px]"
          animate={active ? { height, opacity: 1 } : { height: [4, 10, 4], opacity: 0.12 }}
          transition={
            active
              ? { duration: 0.05, ease: "easeOut" }
              : { duration: 3, repeat: Infinity, ease: "easeInOut", delay: index * 0.2 }
          }
        />
      ))}
    </div>
  );
}

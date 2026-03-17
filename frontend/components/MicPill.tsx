"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Mic, MicOff } from "lucide-react";
import type { VoiceState } from "@/hooks/useVoiceClient";

type MicPillProps = {
  state: VoiceState;
  onToggleMute: () => void;
};

export default function MicPill({ state, onToggleMute }: MicPillProps) {
  const muted = state === "muted";
  const active = state === "connected";
  const disabled = state === "idle" || state === "initializing" || state === "connecting";

  return (
    <div
      className={`h-[44px] px-3 rounded-[10px] border flex items-center gap-2.5 transition-all duration-200 ${
        muted ? "bg-red-500/10 border-destructive" : "bg-ctrl-bg border-ctrl-border"
      } ${disabled ? "opacity-35 pointer-events-none" : ""}`}
    >
      <motion.button
        type="button"
        className="p-0 bg-transparent border-none cursor-pointer text-text-secondary"
        onClick={onToggleMute}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={muted ? "muted" : "unmuted"}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={muted ? "text-destructive" : "text-text-secondary"}
          >
            {muted ? <MicOff size={18} strokeWidth={1.75} /> : <Mic size={18} strokeWidth={1.75} />}
          </motion.span>
        </AnimatePresence>
      </motion.button>

      <div className="flex items-center gap-1.5">
        {Array.from({ length: 4 }).map((_, index) => (
          <motion.span
            key={index}
            className={`w-2 h-2 rounded-full ${muted ? "bg-destructive" : "bg-accent"}`}
            animate={
              muted
                ? { opacity: 0.3, scale: 0.85 }
                : active
                  ? { opacity: [0.2, 1, 0.2], scale: [0.85, 1.2, 0.85] }
                  : { opacity: 0.2, scale: 0.85 }
            }
            transition={
              active && !muted
                ? {
                    duration: 1.2,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: index * 0.15,
                  }
                : { duration: 0.2 }
            }
          />
        ))}
      </div>
    </div>
  );
}

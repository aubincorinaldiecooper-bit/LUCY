"use client";

import { AnimatePresence, motion } from "framer-motion";
import type { VoiceState } from "@/hooks/useVoiceClient";

type ConnectButtonProps = {
  state: VoiceState;
  onConnect: () => void;
  onDisconnect: () => void;
};

export default function ConnectButton({ state, onConnect, onDisconnect }: ConnectButtonProps) {
  const isLoading = state === "initializing" || state === "connecting";
  const isConnected = state === "connected" || state === "muted";

  const className = isLoading
    ? "bg-ctrl-bg text-text-secondary border border-ctrl-border cursor-not-allowed"
    : isConnected
      ? "bg-destructive text-white"
      : "bg-accent text-white";

  return (
    <motion.button
      type="button"
      className={`flex-1 h-[44px] rounded-[10px] font-semibold text-[0.92rem] transition-all duration-200 ${className}`}
      onClick={isConnected ? onDisconnect : onConnect}
      disabled={isLoading}
      whileHover={!isLoading ? { scale: 1.01 } : undefined}
      whileTap={!isLoading ? { scale: 0.97 } : undefined}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={state}
          className="inline-flex items-center justify-center gap-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16 }}
        >
          {isLoading && <span className="w-4 h-4 rounded-full border-2 border-current border-t-transparent animate-spin" />}
          {state === "initializing"
            ? "Initializing…"
            : state === "connecting"
              ? "Connecting…"
              : isConnected
                ? "Disconnect"
                : "Connect"}
        </motion.span>
      </AnimatePresence>
    </motion.button>
  );
}

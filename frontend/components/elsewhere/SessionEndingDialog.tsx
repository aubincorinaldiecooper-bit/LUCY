"use client";

import { AnimatePresence, motion } from "framer-motion";

function formatTime(seconds: number) {
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  const remainingSeconds = safe % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

// Shown when the agent signals the session is about to end (~1 minute out). It's
// informational — there's no dismiss; the countdown runs to 0:00 and then the
// session ends on its own. Kept subtle so it doesn't feel like an error.
export function SessionEndingDialog({
  open,
  secondsLeft,
}: {
  open: boolean;
  secondsLeft: number;
}) {
  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center px-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25 }}
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" aria-hidden="true" />
          <motion.div
            role="status"
            aria-live="polite"
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="relative z-10 w-full max-w-sm rounded-2xl border border-white/10 bg-[#15151c] px-7 py-8 text-center text-white shadow-2xl"
          >
            <p className="text-sm font-light tracking-tight text-white/70">A minute remains</p>
            <p className="mt-3 font-mono text-4xl font-bold tracking-wider text-white tabular-nums">
              {formatTime(secondsLeft)}
            </p>
            <p className="mt-3 text-xs font-light leading-relaxed text-white/55">
              The session will wrap up when the timer reaches zero.
            </p>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

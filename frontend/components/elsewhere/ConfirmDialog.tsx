"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect } from "react";

// A small modal confirmation. Used during a live session so that clicking the
// "Elsewhere" brand (which leaves the conversation) can't happen by accident.
// Backdrop click and the Escape key both cancel; the confirm button is focused
// on open for keyboard users.
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  cancelLabel = "Cancel",
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  description?: string;
  confirmLabel: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onCancel]);

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center px-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          <button
            type="button"
            aria-label={cancelLabel}
            onClick={onCancel}
            className="absolute inset-0 cursor-default bg-black/60 backdrop-blur-sm"
          />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-label={title}
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="relative z-10 w-full max-w-sm rounded-2xl border border-white/10 bg-[#15151c] p-6 text-white shadow-2xl"
          >
            <h2 className="text-base font-light tracking-tight text-white">{title}</h2>
            {description ? (
              <p className="mt-2 text-sm font-light leading-relaxed text-white/65">{description}</p>
            ) : null}
            <div className="mt-6 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={onCancel}
                className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-light text-white/80 transition-colors hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/30"
              >
                {cancelLabel}
              </button>
              <button
                type="button"
                autoFocus
                onClick={onConfirm}
                className="rounded-xl border border-[#FF6B6B]/40 bg-[#FF6B6B]/15 px-4 py-2 text-sm font-light text-[#FF8C8C] transition-colors hover:border-[#FF6B6B] hover:bg-[#FF6B6B]/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#FF6B6B]/40"
              >
                {confirmLabel}
              </button>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

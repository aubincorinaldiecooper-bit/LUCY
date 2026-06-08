"use client";

import { AnimatePresence, motion, MotionConfig, type HTMLMotionProps, type Transition } from "framer-motion";
import { ArrowRight, Check, Loader2, Mail, X } from "lucide-react";
import { FormEvent, ReactNode, forwardRef, useEffect, useRef, useState } from "react";

type MorphButtonProps = Omit<HTMLMotionProps<"button">, "children"> & {
  text: string;
  isLoading?: boolean;
  icon?: ReactNode;
};

const MorphButton = forwardRef<HTMLButtonElement, MorphButtonProps>(
  ({ text, isLoading = false, icon, className = "", disabled, onClick, ...props }, ref) => {
    const transition: Transition = {
      type: "spring",
      stiffness: 150,
      damping: 25,
      mass: 1,
    };

    return (
      <MotionConfig transition={transition}>
        <motion.button
          ref={ref}
          layout
          className={`relative flex h-12 items-center justify-center overflow-hidden rounded-full border border-[#B86B4D] bg-[#B86B4D] font-medium text-white shadow-sm shadow-[#B86B4D]/20 transition-colors hover:bg-[#A55D42] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#B86B4D]/25 disabled:cursor-not-allowed disabled:opacity-60 ${isLoading ? "px-0" : "px-7"} ${className}`}
          disabled={disabled || isLoading}
          onClick={(event) => {
            if (!isLoading) onClick?.(event);
          }}
          whileTap={!isLoading ? { scale: 0.98 } : undefined}
          {...props}
        >
          <AnimatePresence mode="popLayout" initial={false}>
            {isLoading ? (
              <motion.span
                key="loader"
                className="flex w-12 items-center justify-center"
                initial={{ opacity: 0, scale: 0.85, filter: "blur(8px)" }}
                animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, scale: 0.85, filter: "blur(8px)" }}
              >
                <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
                <span className="sr-only">Submitting</span>
              </motion.span>
            ) : (
              <motion.span
                key="content"
                className="flex items-center gap-2 whitespace-nowrap"
                initial={{ opacity: 0, y: 8, filter: "blur(8px)" }}
                animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                exit={{ opacity: 0, y: -8, filter: "blur(8px)" }}
              >
                <motion.span layout>{text}</motion.span>
                {icon ? <motion.span layout>{icon}</motion.span> : null}
              </motion.span>
            )}
          </AnimatePresence>
        </motion.button>
      </MotionConfig>
    );
  },
);

MorphButton.displayName = "MorphButton";

export function EarlyAccessModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const submitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (submitTimerRef.current) clearTimeout(submitTimerRef.current);
    };
  }, []);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (isSubmitting) return;
    // TODO: Wire to the production early-access/email-capture endpoint when one exists.
    setIsSubmitting(true);
    submitTimerRef.current = setTimeout(() => {
      setIsSubmitting(false);
      setSubmitted(true);
    }, 450);
  };

  const closeAndReset = () => {
    if (submitTimerRef.current) {
      clearTimeout(submitTimerRef.current);
      submitTimerRef.current = null;
    }
    onClose();
    setSubmitted(false);
    setIsSubmitting(false);
    setEmail("");
  };

  return (
    <AnimatePresence>
      {isOpen ? (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
          <button aria-label="Close early access modal" className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={closeAndReset} />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="early-access-title"
            initial={{ y: 18, opacity: 0, scale: 0.96 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 18, opacity: 0, scale: 0.96 }}
            transition={{ type: "spring", damping: 26, stiffness: 280 }}
            className="relative w-[calc(100vw-32px)] max-w-[540px] rounded-[28px] border border-black/[0.08] bg-[#FAFAFA] p-7 text-[#1C1C1E] shadow-2xl shadow-black/20 sm:p-9"
          >
            <button
              onClick={closeAndReset}
              aria-label="Close"
              className="absolute right-4 top-4 flex h-9 w-9 items-center justify-center rounded-full text-[#86868B] transition-colors hover:bg-black/[0.04] hover:text-[#1C1C1E]"
            >
              <X size={18} />
            </button>

            <AnimatePresence mode="wait" initial={false}>
              {submitted ? (
                <motion.div
                  key="success"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="flex flex-col items-center py-5 text-center"
                >
                  <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-full bg-[#34C759]/10 text-[#34C759]">
                    <Check size={24} />
                  </div>
                  <h3 id="early-access-title" className="mb-2 text-2xl font-semibold tracking-tight text-[#1C1C1E]">
                    Got it — we&apos;ll reach out soon.
                  </h3>
                  <p className="mb-6 max-w-sm text-sm leading-6 text-[#86868B]">Thanks for helping shape Elsewhere at the beginning.</p>
                  <button onClick={closeAndReset} className="text-sm font-medium text-[#B86B4D] transition-colors hover:text-[#A55D42]">
                    Close
                  </button>
                </motion.div>
              ) : (
                <motion.div key="form" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
                  <div className="mb-7 pr-8">
                    <h3 id="early-access-title" className="mb-3 text-[28px] font-semibold leading-tight tracking-tight text-[#1C1C1E]">
                      Help build Elsewhere
                    </h3>
                    <p className="text-base leading-7 text-[#86868B]">Leave your email if you want early access or want to help shape what this becomes.</p>
                  </div>

                  <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                    <label htmlFor="early-access-email" className="text-sm font-medium text-[#1C1C1E]">
                      Email
                    </label>
                    <div className="relative">
                      <Mail className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-[#86868B]" aria-hidden="true" />
                      <input
                        id="early-access-email"
                        type="email"
                        placeholder="you@example.com"
                        value={email}
                        onChange={(event) => setEmail(event.target.value)}
                        required
                        disabled={isSubmitting}
                        className="h-14 w-full rounded-2xl border border-black/[0.12] bg-white px-12 text-base text-[#1C1C1E] outline-none transition-all placeholder:text-[#A1A1A6] focus:border-[#B86B4D] focus:ring-4 focus:ring-[#B86B4D]/10 disabled:cursor-not-allowed disabled:opacity-70"
                      />
                    </div>
                    <div className="flex justify-end pt-1">
                      <MorphButton
                        type="submit"
                        text="Email Arche"
                        isLoading={isSubmitting}
                        icon={<ArrowRight size={18} strokeWidth={2.5} aria-hidden="true" />}
                        className="w-full sm:w-auto"
                      />
                    </div>
                  </form>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

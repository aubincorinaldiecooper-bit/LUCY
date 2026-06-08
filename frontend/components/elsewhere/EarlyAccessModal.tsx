"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Check, X } from "lucide-react";
import { FormEvent, useState } from "react";
import { Input } from "./Input";

const EARLY_ACCESS_IMAGE = "https://res.cloudinary.com/dvsfba1ww/image/upload/v1780867666/ChatGPT_Image_Jun_7_2026_05_27_30_PM_kkun6b.png";

export function EarlyAccessModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    // TODO: Wire to the production early-access/email-capture endpoint when one exists.
    setSubmitted(true);
  };

  const closeAndReset = () => {
    onClose();
    setSubmitted(false);
    setEmail("");
  };

  return (
    <AnimatePresence>
      {isOpen ? (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
          <button aria-label="Close early access modal" className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={closeAndReset} />
          <motion.div
            initial={{ y: 20, opacity: 0, scale: 0.95 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 20, opacity: 0, scale: 0.95 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="relative flex w-full max-w-[680px] flex-col overflow-hidden rounded-3xl bg-[#FAFAFA] shadow-2xl"
          >
            <div className="relative h-56 w-full shrink-0 overflow-hidden bg-gray-100">
              <img src={EARLY_ACCESS_IMAGE} alt="Early access visual" className="absolute inset-0 h-full w-full object-cover" />
              <button onClick={closeAndReset} aria-label="Close" className="absolute right-4 top-4 z-10 flex h-8 w-8 items-center justify-center rounded-full bg-black/30 text-white backdrop-blur-md transition-colors hover:bg-black/50">
                <X size={14} />
              </button>
            </div>
            <div className="flex flex-1 flex-col px-6 py-7 sm:px-10 sm:py-8">
              {submitted ? (
                <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="py-8 text-center">
                  <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[#34C759]/10 text-[#34C759]">
                    <Check size={24} />
                  </div>
                  <h3 className="mb-1 text-lg font-semibold text-[#1C1C1E]">You&apos;re on the list.</h3>
                  <p className="mb-6 text-sm text-[#86868B]">We&apos;ll be in touch soon.</p>
                  <button onClick={closeAndReset} className="text-sm font-medium text-[#B86B4D] hover:text-[#A55D42]">Close</button>
                </motion.div>
              ) : (
                <>
                  <h3 className="mb-3 text-[30px] font-bold leading-tight text-[#1C1C1E]">Help build Elsewhere</h3>
                  <p className="mb-6 text-[18px] leading-relaxed text-[#86868B]">Leave your email if you want early access or want to help shape what this becomes.</p>
                  <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:items-end">
                    <Input type="email" placeholder="Email address" value={email} onChange={(event) => setEmail(event.target.value)} required />
                    <button type="submit" className="flex h-[56px] w-full items-center justify-center gap-2.5 rounded-xl bg-[#B86B4D] px-7 py-4 text-base font-medium text-white shadow-lg shadow-[#B86B4D]/20 transition-all hover:bg-[#A55D42] active:scale-95 sm:w-auto">
                      Email Arche
                      <ArrowRight size={18} strokeWidth={2.5} />
                    </button>
                  </form>
                </>
              )}
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

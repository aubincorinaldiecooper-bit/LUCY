"use client";

import { motion } from "framer-motion";
import { FormEvent, useState } from "react";
import { Button } from "./Button";
import { Input } from "./Input";
import { PageTransition } from "./PageTransition";

const END_IMAGE = "https://res.cloudinary.com/dvsfba1ww/image/upload/q_auto/f_auto/v1780860202/ChatGPT_Image_Jun_7_2026_02_34_47_PM_uwszje.png";

export function EndSessionPage({ onReturnHome, onOpenEarlyAccess }: { onReturnHome: () => void; onOpenEarlyAccess: () => void }) {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    // TODO: Wire to the production feedback/email-capture endpoint when one exists.
    setSubmitted(true);
    setEmail("");
  };

  return (
    <PageTransition>
      <div className="relative flex h-screen flex-1 flex-col overflow-hidden bg-[#FAFAFA]">
        <nav className="fixed top-0 z-40 flex w-full items-center justify-between border-b border-[#E5E5E5]/50 bg-[#FAFAFA]/90 px-5 py-4 backdrop-blur-xl md:px-8">
          <div className="text-sm font-semibold tracking-tight text-[#1C1C1E]">Elsewhere</div>
          <Button variant="outline" onClick={onOpenEarlyAccess} className="!rounded-full !px-4 !py-1.5 !text-xs">Early access</Button>
        </nav>
        <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-5 pb-6 pt-20 md:px-8">
          <div className="flex w-full max-w-lg flex-col items-center">
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.6 }} className="mb-6 w-full max-w-[400px]">
              <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl bg-white shadow-sm ring-1 ring-black/5">
                <img src={END_IMAGE} alt="Reflection and clarity" className="h-full w-full object-cover" />
              </div>
            </motion.div>
            <motion.h2 initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.6 }} className="mb-4 text-center text-2xl font-semibold leading-[1.2] tracking-tight text-[#1C1C1E] md:text-3xl">What do you want Elsewhere to be?</motion.h2>
            <motion.p initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.6 }} className="mb-8 max-w-md text-center text-sm leading-relaxed text-[#86868B] md:text-base">We&apos;re building Elsewhere with the people who use it first. Tell us what felt useful, what was missing, and what you&apos;d want this to become.</motion.p>
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.6 }} className="mb-6 w-full max-w-sm space-y-2.5">
              {submitted ? (
                <div className="rounded-xl bg-[#34C759]/10 px-4 py-3 text-center">
                  <p className="text-xs font-medium text-[#34C759]">Thanks for helping shape Elsewhere.</p>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-2.5">
                  <Input type="email" placeholder="Enter your email" value={email} onChange={(event) => setEmail(event.target.value)} required />
                  <Button type="submit" variant="primary" className="w-full !rounded-xl !py-2.5">Help build Elsewhere</Button>
                </form>
              )}
            </motion.div>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5, duration: 0.6 }}>
              <button onClick={onReturnHome} className="text-xs text-[#86868B] underline decoration-[#86868B]/30 underline-offset-4 transition-colors hover:text-[#1C1C1E]">Return home</button>
            </motion.div>
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6, duration: 0.6 }} className="mt-8 text-center text-[10px] text-[#A1A1A6]">
              By continuing, you agree to the <span className="underline decoration-[#A1A1A6]/30 underline-offset-2">Terms</span> and <span className="underline decoration-[#A1A1A6]/30 underline-offset-2">Privacy Policy</span>.
            </motion.p>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}

"use client";

import { motion } from "framer-motion";
import AccountLink from "@/components/auth/AccountLink";
import { BrandHome } from "./BrandHome";
import { FeedbackForm } from "./FeedbackForm";
import { PageTransition } from "./PageTransition";
import { ShaderBackground } from "./ShaderBackground";

export function EndSessionPage({ onReturnHome }: { onReturnHome: () => void }) {
  return (
    <PageTransition>
      <div className="relative flex h-screen flex-1 flex-col overflow-hidden bg-black text-white">
        <ShaderBackground />
        <nav className="fixed left-0 right-0 top-0 z-40 flex w-full items-center justify-between px-6 py-5 md:px-10 lg:px-16">
          <BrandHome onClick={onReturnHome} />
          <AccountLink variant="dark" />
        </nav>
        <div className="relative z-10 flex flex-1 flex-col items-center justify-center overflow-y-auto px-5 pb-6 pt-24 md:px-8">
          <div className="flex w-full max-w-lg flex-col items-center">
            <motion.h2 initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.6 }} className="mb-4 text-center text-3xl font-extralight leading-[1.1] tracking-tight text-white md:text-4xl">What do you want Arche to be?</motion.h2>
            <motion.p initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.6 }} className="mb-8 max-w-md text-center text-sm font-light leading-relaxed text-white/70 md:text-base">We&apos;re building Arche with the people who use it first. Tell us what you want Arche to do, what was missing, and how we can make it better.</motion.p>
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.6 }} className="mb-6 w-full max-w-sm">
              <FeedbackForm />
            </motion.div>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5, duration: 0.6 }}>
              <button onClick={onReturnHome} className="text-xs font-light text-white/60 underline decoration-white/20 underline-offset-4 transition-colors hover:text-white">Return home</button>
            </motion.div>
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6, duration: 0.6 }} className="mt-8 text-center text-[10px] font-extralight text-white/40">
              By continuing, you agree to the <span className="underline decoration-white/20 underline-offset-2">Terms</span> and <span className="underline decoration-white/20 underline-offset-2">Privacy Policy</span>.
            </motion.p>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}

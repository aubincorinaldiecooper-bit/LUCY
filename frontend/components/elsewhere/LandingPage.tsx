"use client";

import { motion } from "framer-motion";
import AccountLink from "@/components/auth/AccountLink";
import { BrandHome } from "./BrandHome";
import { PageTransition } from "./PageTransition";
import { ShaderBackground } from "./ShaderBackground";

export function LandingPage({
  onStartSession,
  onHome,
}: {
  onStartSession: () => void;
  onHome?: () => void;
}) {
  return (
    <PageTransition>
      <section className="relative isolate flex h-screen w-screen overflow-hidden bg-black text-white">
        <ShaderBackground />

        <nav className="fixed left-0 right-0 top-0 z-40 flex w-full items-center justify-between px-6 py-5 md:px-10 lg:px-16">
          <BrandHome onClick={onHome} />
          <AccountLink variant="dark" />
        </nav>

        <div className="relative z-10 mx-auto flex w-full max-w-3xl flex-col items-center justify-center gap-6 px-6 pb-20 pt-32 text-center sm:gap-8 sm:pt-40 md:px-10 lg:px-16">
          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.75, ease: "easeOut" }}
            className="text-5xl font-extralight leading-[1.05] tracking-tight text-white sm:text-6xl md:text-7xl"
          >
            Turn scattered thoughts into direction.
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.75, ease: "easeOut" }}
            className="max-w-xl text-base font-light leading-relaxed tracking-tight text-white/75 sm:text-lg"
          >
            Elsewhere helps you work through decisions, organize what is competing for your attention, and leave with a clearer next step.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.75, ease: "easeOut" }}
            className="flex flex-wrap items-center justify-center gap-3 pt-2"
          >
            <button
              type="button"
              onClick={onStartSession}
              className="rounded-2xl border border-white/10 bg-white/10 px-5 py-3 text-sm font-light tracking-tight text-white backdrop-blur-sm transition-colors duration-300 hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-white/30"
            >
              Meet Arche
            </button>
          </motion.div>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.7 }}
            className="pt-4 text-[10px] font-extralight tracking-tight text-white/55 md:text-xs"
          >
            By continuing, you agree to the <span className="underline decoration-white/20 underline-offset-2">Terms</span> and <span className="underline decoration-white/20 underline-offset-2">Privacy Policy</span>.
          </motion.p>
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/40 to-transparent" />
      </section>
    </PageTransition>
  );
}

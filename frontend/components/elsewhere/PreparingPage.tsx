"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { PageTransition } from "./PageTransition";
import { ShaderBackground } from "./ShaderBackground";

export function PreparingPage({ onCancel }: { onCancel: () => void }) {
  const [showCancel, setShowCancel] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShowCancel(true), 1500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <PageTransition>
      <div className="relative flex h-screen flex-1 flex-col items-center justify-center overflow-hidden bg-black text-white">
        <ShaderBackground />
        <div className="relative z-10 flex flex-col items-center px-8 text-center">
          <motion.h2 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-3 text-2xl font-extralight tracking-tight text-white md:text-3xl">Preparing your space.</motion.h2>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="mb-16 text-sm font-light text-white/70 md:text-base">Take a breath. We&apos;ll begin in a moment.</motion.p>
          <AnimatePresence>
            {showCancel ? (
              <motion.button initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} onClick={onCancel} className="rounded-full border border-white/10 bg-white/10 px-5 py-2 text-sm font-light text-white/80 backdrop-blur-sm transition-colors hover:bg-white/20">
                Cancel
              </motion.button>
            ) : null}
          </AnimatePresence>
        </div>
      </div>
    </PageTransition>
  );
}

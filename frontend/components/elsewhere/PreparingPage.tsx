"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { PageTransition } from "./PageTransition";

export function PreparingPage({ onCancel }: { onCancel: () => void }) {
  const [showCancel, setShowCancel] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShowCancel(true), 1500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <PageTransition>
      <div className="relative flex h-screen flex-1 flex-col items-center justify-center overflow-hidden bg-[#F9F9F7]">
        <div className="absolute inset-0 flex items-center justify-center" aria-hidden="true">
          <motion.div animate={{ scale: [1, 1.2, 1], opacity: [0.4, 0.7, 0.4] }} transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }} className="h-[240px] w-[240px] rounded-full bg-gradient-to-br from-[#F0E6D8] via-white to-[#E8E0D0] blur-3xl md:h-[300px] md:w-[300px]" />
        </div>
        <div className="relative z-10 flex flex-col items-center px-8 text-center">
          <motion.h2 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-3 text-xl font-semibold tracking-tight text-[#1C1C1E] md:text-2xl">Preparing your space.</motion.h2>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="mb-16 text-sm font-light text-[#86868B] md:text-base">Take a breath. We&apos;ll begin in a moment.</motion.p>
          <AnimatePresence>
            {showCancel ? (
              <motion.button initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} onClick={onCancel} className="rounded-full border border-white/50 bg-white/50 px-5 py-2 text-sm font-medium text-[#86868B] shadow-sm backdrop-blur-sm transition-colors hover:text-[#1C1C1E]">
                Cancel
              </motion.button>
            ) : null}
          </AnimatePresence>
        </div>
      </div>
    </PageTransition>
  );
}

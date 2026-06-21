"use client";

import { motion } from "framer-motion";
import { Mic, MicOff, PhoneOff } from "lucide-react";
import { PageTransition } from "./PageTransition";

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

export function ListeningPage({
  isMuted,
  timer,
  onToggleMute,
  onEnd,
}: {
  isMuted: boolean;
  timer: number;
  onToggleMute: () => void;
  onEnd: () => void;
}) {
  return (
    <PageTransition>
      <div className="relative flex h-screen flex-1 flex-col overflow-hidden bg-black text-white">
        <nav className="fixed left-0 right-0 top-0 z-40 flex w-full items-center justify-between px-6 py-5 md:px-10 lg:px-16">
          <div className="flex items-center gap-2.5">
            <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-[#E0C9A8] to-[#C4A882]" />
            <span className="text-sm font-light tracking-tight text-white/85">Elsewhere</span>
          </div>
        </nav>
        <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-6 pt-20">
          {/* Warm soft gradient blob — kept, tuned to glow on the dark backdrop. */}
          <div className="relative mb-8 flex h-[100px] w-[100px] items-center justify-center md:mb-10 md:h-[130px] md:w-[130px]" aria-hidden="true">
            <motion.div animate={{ scale: [1, 1.1, 1], rotate: [0, 10, -10, 0] }} transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }} className="absolute inset-0 rounded-full bg-gradient-to-tr from-[#FF9A9E] via-[#FECFEF] to-[#FFD194] opacity-70 blur-[30px]" />
            <motion.div animate={{ scale: [1.1, 0.9, 1.1], rotate: [0, -20, 20, 0] }} transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }} className="absolute inset-2 rounded-full bg-gradient-to-bl from-[#FF6B6B] via-[#FF8C42] to-[#F9D423] opacity-60 blur-[25px]" />
            <motion.div animate={{ scale: [1, 1.05, 1], rotate: [0, 5, -5, 0] }} transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }} className="absolute inset-0 rounded-full bg-gradient-to-r from-[#FFD194] via-[#FF9A9E] to-[#FF6B6B] opacity-50 blur-[15px] mix-blend-screen" />
          </div>
          <div className="relative z-10 mb-8 text-center md:mb-10">
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="font-mono text-2xl font-bold tracking-wider text-white md:text-3xl">{formatTime(timer)}</motion.p>
            {isMuted ? <p className="mt-2 text-xs text-white/50">Microphone muted</p> : null}
          </div>
          <div className="relative z-10 mb-10 flex items-center gap-4 md:mb-14 md:gap-6">
            <motion.button whileTap={{ scale: 0.96 }} onClick={onToggleMute} className={`flex w-28 items-center justify-center gap-2 rounded-xl py-3 text-sm font-light backdrop-blur-sm transition-all ${isMuted ? "border border-white/20 bg-white/15 text-white hover:bg-white/25" : "border border-white/10 bg-white/5 text-white/70 hover:bg-white/10"}`}>
              {isMuted ? <MicOff size={16} /> : <Mic size={16} />}
              {isMuted ? "Unmute" : "Mute"}
            </motion.button>
            <motion.button whileTap={{ scale: 0.96 }} onClick={onEnd} className="flex w-28 items-center justify-center gap-2 rounded-xl border border-[#FF6B6B]/40 bg-[#FF6B6B]/10 py-3 text-sm font-light text-[#FF8C8C] backdrop-blur-sm transition-all hover:border-[#FF6B6B] hover:bg-[#FF6B6B]/20">
              <PhoneOff size={16} />
              End
            </motion.button>
          </div>
          <p className="relative z-10 max-w-xs text-center text-[10px] font-extralight leading-relaxed text-white/45 md:text-xs">This companion can make mistakes. For urgent support, contact emergency services or someone you trust.</p>
        </div>
      </div>
    </PageTransition>
  );
}

"use client";

import { motion } from "framer-motion";
import { Mic, MicOff, PhoneOff } from "lucide-react";
import { Button } from "./Button";
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
      <div className="relative flex h-screen flex-1 flex-col overflow-hidden bg-[#FAFAFA]">
        <nav className="fixed top-0 z-40 flex w-full items-center justify-between border-b border-gray-100/50 bg-[#FAFAFA]/90 px-5 py-4 backdrop-blur-xl md:px-8">
          <div className="flex items-center gap-2.5">
            <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-[#E0C9A8] to-[#C4A882]" />
            <span className="text-sm font-semibold tracking-tight text-[#1C1C1E]">Elsewhere</span>
          </div>
        </nav>
        <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-6 pt-20">
          <div className="relative mb-8 flex h-[100px] w-[100px] items-center justify-center md:mb-10 md:h-[130px] md:w-[130px]" aria-hidden="true">
            <motion.div animate={{ scale: [1, 1.1, 1], rotate: [0, 10, -10, 0] }} transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }} className="absolute inset-0 rounded-full bg-gradient-to-tr from-[#FF9A9E] via-[#FECFEF] to-[#FFD194] opacity-60 blur-[30px]" />
            <motion.div animate={{ scale: [1.1, 0.9, 1.1], rotate: [0, -20, 20, 0] }} transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }} className="absolute inset-2 rounded-full bg-gradient-to-bl from-[#FF6B6B] via-[#FF8C42] to-[#F9D423] opacity-50 blur-[25px]" />
            <motion.div animate={{ scale: [1, 1.05, 1], rotate: [0, 5, -5, 0] }} transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }} className="absolute inset-0 rounded-full bg-gradient-to-r from-[#FFD194] via-[#FF9A9E] to-[#FF6B6B] opacity-40 blur-[15px] mix-blend-multiply" />
          </div>
          <div className="relative z-10 mb-8 text-center md:mb-10">
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="font-mono text-2xl font-bold tracking-wider text-[#1C1C1E] md:text-3xl">{formatTime(timer)}</motion.p>
            <p className="mt-2 text-xs text-[#86868B]">{isMuted ? "Microphone muted" : "Arche is listening"}</p>
          </div>
          <div className="relative z-10 mb-10 flex items-center gap-4 md:mb-14 md:gap-6">
            <motion.button whileTap={{ scale: 0.96 }} onClick={onToggleMute} className={`flex w-28 items-center justify-center gap-2 rounded-xl py-3 text-sm font-medium transition-all ${isMuted ? "bg-[#E5E5E5] text-[#1C1C1E] hover:bg-[#D4D4D4]" : "bg-transparent text-[#86868B] hover:bg-[#F5F5F7]"}`}>
              {isMuted ? <MicOff size={16} /> : <Mic size={16} />}
              {isMuted ? "Unmute" : "Mute"}
            </motion.button>
            <Button variant="danger" onClick={onEnd} className="!w-28 !rounded-xl !py-3 !text-sm">
              <PhoneOff size={16} />
              End
            </Button>
          </div>
          <p className="relative z-10 max-w-xs text-center text-[10px] font-light leading-relaxed text-[#86868B] opacity-60 md:text-xs">This companion can make mistakes. For urgent support, contact emergency services or someone you trust.</p>
        </div>
      </div>
    </PageTransition>
  );
}

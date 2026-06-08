"use client";

import { motion } from "framer-motion";
import { Button } from "./Button";
import { PageTransition } from "./PageTransition";

const LANDING_IMAGE = "https://res.cloudinary.com/dvsfba1ww/image/upload/q_auto/f_auto/v1780860203/ChatGPT_Image_Jun_7_2026_02_34_37_PM_tyymkc.png";

export function LandingPage({ onStartSession, onOpenEarlyAccess }: { onStartSession: () => void; onOpenEarlyAccess: () => void }) {
  return (
    <PageTransition>
      <div className="relative flex h-screen flex-1 flex-col overflow-hidden bg-[#FAFAFA]">
        <nav className="fixed top-0 z-40 flex w-full items-center justify-between border-b border-[#E5E5E5]/50 bg-[#FAFAFA]/90 px-5 py-4 backdrop-blur-xl md:px-8">
          <div className="text-sm font-semibold tracking-tight text-[#1C1C1E]">Elsewhere</div>
          <Button variant="outline" onClick={onOpenEarlyAccess} className="!rounded-full !px-4 !py-1.5 !text-xs">
            Early access
          </Button>
        </nav>
        <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-5 pb-6 pt-20 md:px-8">
          <div className="flex w-full max-w-4xl flex-col items-center">
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.6 }} className="mb-4 text-center md:mb-5">
              <h1 className="text-2xl font-semibold leading-[1.15] tracking-tight text-[#1C1C1E] sm:text-3xl md:text-4xl lg:text-5xl">Turn scattered thoughts into direction.</h1>
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.6 }} className="mb-4 w-full max-w-[720px] md:mb-5">
              <div className="relative aspect-[16/7] w-full overflow-hidden rounded-xl bg-white shadow-sm ring-1 ring-black/5 md:aspect-[2/1]">
                <img src={LANDING_IMAGE} alt="Organized thoughts" className="h-full w-full object-cover" />
              </div>
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.6 }} className="mb-4 w-full max-w-[520px] md:mb-5">
              <p className="text-center text-sm leading-relaxed text-[#86868B] md:text-base">Elsewhere helps you work through decisions, organize what is competing for your attention, and leave with a clearer next step.</p>
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.6 }} className="mb-4">
              <Button onClick={onStartSession} variant="primary" className="!px-8 !py-2.5 !text-sm shadow-lg shadow-[#D4A373]/25">
                Meet Arche
              </Button>
            </motion.div>
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5, duration: 0.6 }} className="text-center text-[10px] text-[#A1A1A6] md:text-xs">
              By continuing, you agree to the <span className="underline decoration-[#A1A1A6]/30 underline-offset-2">Terms</span> and <span className="underline decoration-[#A1A1A6]/30 underline-offset-2">Privacy Policy</span>.
            </motion.p>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}

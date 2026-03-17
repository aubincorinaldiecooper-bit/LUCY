"use client";

import dynamic from "next/dynamic";
import { AnimatePresence, motion } from "framer-motion";
import { Mic, Moon, Sun } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useAudioDevices } from "@/hooks/useAudioDevices";
import { useVoiceClient } from "@/hooks/useVoiceClient";
import { useWaveform } from "@/hooks/useWaveform";

const ConnectButton = dynamic(() => import("@/components/ConnectButton"), { ssr: false });
const MicPill = dynamic(() => import("@/components/MicPill"), { ssr: false });
const SettingsPanel = dynamic(() => import("@/components/SettingsPanel"), { ssr: false });
const Waveform = dynamic(() => import("@/components/Waveform"), { ssr: false });

const item = (delay: number) => ({
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.35, delay },
});

export default function HomePage() {
  const { state, connect, disconnect, toggleMute } = useVoiceClient();
  const { mics, speakers } = useAudioDevices();
  const [selectedMic, setSelectedMic] = useState("");
  const [selectedSpeaker, setSelectedSpeaker] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [canHover, setCanHover] = useState(false);

  const active = state === "connected" || state === "muted";
  const { barHeights, startWaveform, stopWaveform } = useWaveform(15, active);

  useEffect(() => {
    const query = window.matchMedia("(hover: hover)");
    const update = () => setCanHover(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    let stream: MediaStream | null = null;
    const run = async () => {
      if (active && state === "connected") {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: selectedMic ? { deviceId: { exact: selectedMic } } : true,
          });
          await startWaveform(stream);
        } catch {
          await stopWaveform();
        }
      } else {
        await stopWaveform();
      }
    };

    run();

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
      void stopWaveform();
    };
  }, [active, selectedMic, startWaveform, state, stopWaveform]);

  const radial = useMemo(
    () =>
      darkMode
        ? "radial-gradient(circle at top right, rgba(74,222,128,0.04), transparent 45%)"
        : "radial-gradient(circle at top right, rgba(74,222,128,0.06), transparent 45%)",
    [darkMode]
  );

  const handleConnect = async () => {
    await connect(selectedMic || undefined);
  };

  const toggleTheme = () => {
    const html = document.documentElement;
    const nextDark = html.getAttribute("data-theme") !== "dark";
    html.setAttribute("data-theme", nextDark ? "dark" : "light");
    setDarkMode(nextDark);
  };

  return (
    <main className="min-h-screen w-full px-4 flex flex-col items-center justify-center gap-3 bg-bg" style={{ backgroundImage: radial }}>
      <motion.div
        {...item(0)}
        className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-border bg-surface backdrop-blur shadow-sm"
      >
        <Mic size={16} strokeWidth={1.75} className="text-text-primary" />
        <span className="font-bold tracking-tight text-text-primary">LUCY</span>
      </motion.div>

      <motion.p {...item(0.08)} className="text-sm text-text-secondary text-center">
        Your voice. Your conversation.
      </motion.p>

      <motion.h1 {...item(0.14)} className="text-text-primary text-2xl sm:text-3xl font-bold tracking-tight text-center">
        Try it
      </motion.h1>

      <motion.div
        {...item(0.2)}
        className="w-full max-w-[400px] rounded-2xl p-4 sm:p-5 bg-surface backdrop-blur-md border border-border shadow-md"
        whileHover={canHover ? { y: -2 } : undefined}
      >
        <div className="flex flex-col gap-2.5">
          <div className="flex gap-2 items-stretch">
            <ConnectButton state={state} onConnect={handleConnect} onDisconnect={disconnect} />
            <MicPill state={state} onToggleMute={toggleMute} />
            <SettingsPanel
              mics={mics}
              speakers={speakers}
              selectedMic={selectedMic}
              selectedSpeaker={selectedSpeaker}
              onMicChange={setSelectedMic}
              onSpeakerChange={setSelectedSpeaker}
            />
          </div>
          <Waveform barHeights={barHeights} active={state === "connected"} />
        </div>
      </motion.div>

      <motion.div {...item(0.26)} className="flex items-center gap-3 text-sm text-text-secondary text-center">
        <a href="#" className="hover:text-text-primary transition-colors">
          Privacy
        </a>
        <span className="opacity-40">•</span>
        <a href="#" className="hover:text-text-primary transition-colors">
          Support
        </a>
      </motion.div>

      <motion.button
        type="button"
        onClick={toggleTheme}
        className="fixed bottom-5 right-5 z-50 w-[38px] h-[38px] rounded-full border border-border bg-surface-solid shadow-sm flex items-center justify-center"
        whileHover={canHover ? { scale: 1.08 } : undefined}
        whileTap={{ scale: 0.95 }}
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={darkMode ? "moon" : "sun"}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="text-text-primary"
          >
            {darkMode ? <Moon size={16} strokeWidth={1.75} /> : <Sun size={16} strokeWidth={1.75} />}
          </motion.span>
        </AnimatePresence>
      </motion.button>
    </main>
  );
}

"use client";

import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ConversationBar } from "@/components/ui/conversation-bar";
import { useAudioDevices } from "@/hooks/useAudioDevices";
import { useVoiceClient } from "@/hooks/useVoiceClient";
import { useWaveform } from "@/hooks/useWaveform";

const SettingsPanel = dynamic(() => import("@/components/SettingsPanel"), { ssr: false });

export default function HomePage() {
  const { state, connect, disconnect, toggleMute } = useVoiceClient();
  const { mics, speakers } = useAudioDevices();
  const [selectedModelId, setSelectedModelId] = useState("gpt-4o");
  const [selectedMic, setSelectedMic] = useState("");
  const [selectedSpeaker, setSelectedSpeaker] = useState("");

  const active = state === "connected" || state === "muted";
  const { barHeights, micAmplitude, startWaveform, stopWaveform } = useWaveform(15, active);

  useEffect(() => {
    let stream: MediaStream | null = null;
    const run = async () => {
      if (active && state === "connected") {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
  }, [active, startWaveform, state, stopWaveform]);

  const glowOpacity = useMemo(() => 0.12 + micAmplitude * 0.28, [micAmplitude]);

  const handleConnect = useCallback(() => {
    void connect(selectedModelId);
  }, [connect, selectedModelId]);

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#FAFAF8]">
      <div
        className="absolute inset-0 -z-10"
        style={{
          background:
            "radial-gradient(circle at center, #FDFDFB 0%, #FAFAF8 45%, #F1EFE8 100%)",
        }}
      />

      <motion.div
        className="absolute inset-0 -z-10 pointer-events-none"
        style={{
          background:
            "radial-gradient(circle at center, rgba(10,147,150,0.38) 0%, rgba(10,147,150,0.18) 28%, rgba(10,147,150,0) 58%)",
        }}
        animate={{ opacity: glowOpacity }}
        transition={{ duration: 0.12, ease: "easeOut" }}
      />

      <motion.div
        initial={{ opacity: 0, x: -12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className="absolute top-8 left-6 md:left-10 z-20"
      >
        <h1 className="text-[#1E293B] font-semibold text-lg tracking-[0.12em] uppercase">Paw</h1>
      </motion.div>

      {/* Shared flex column → cat and bar share exact same center */}
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-start pt-[16vh]">
        <motion.div
          initial={{ opacity: 0, scale: 0.92, y: 16 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
          className="w-72 h-72 md:w-80 md:h-80"
          style={{ filter: "drop-shadow(0 4px 12px rgba(0,0,0,0.08))" }}
        >
          <DotLottieReact
            src="/assets/cat-ocean.json"
            loop
            autoplay
            speed={0.3}
            style={{ width: "100%", height: "100%" }}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.45 }}
          className="mt-10 w-full max-w-[560px] px-4"
        >
          <ConversationBar
            className="w-full"
            state={state}
            barHeights={barHeights}
            onConnect={handleConnect}
            onDisconnect={disconnect}
            onToggleMute={toggleMute}
            selectedModelId={selectedModelId}
            onModelChange={setSelectedModelId}
            rightSlot={
              <SettingsPanel
                mics={mics}
                speakers={speakers}
                selectedMic={selectedMic}
                selectedSpeaker={selectedSpeaker}
                onMicChange={setSelectedMic}
                onSpeakerChange={setSelectedSpeaker}
              />
            }
          />
        </motion.div>
      </div>
    </main>
  );
}
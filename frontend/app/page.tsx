"use client";

import { AnimatePresence } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";
import { EarlyAccessModal } from "@/components/elsewhere/EarlyAccessModal";
import { EndSessionPage } from "@/components/elsewhere/EndSessionPage";
import { LandingPage } from "@/components/elsewhere/LandingPage";
import { ListeningPage } from "@/components/elsewhere/ListeningPage";
import { PreparingPage } from "@/components/elsewhere/PreparingPage";
import { useVoiceClient } from "@/hooks/useVoiceClient";

function HomePage() {
  const { state, connect, disconnect, toggleMute, isMuted } = useVoiceClient();
  const [selectedModelId] = useState("openai/gpt-4o");
  const [hadCall, setHadCall] = useState(false);
  const [timer, setTimer] = useState(0);
  const [showEarlyAccess, setShowEarlyAccess] = useState(false);

  const isActiveCall = state === "connected" || state === "muted";

  useEffect(() => {
    if (!isActiveCall) return;
    const interval = setInterval(() => setTimer((current) => current + 1), 1000);
    return () => clearInterval(interval);
  }, [isActiveCall]);

  const handleStart = useCallback(() => {
    setHadCall(false);
    setTimer(0);
    void connect(selectedModelId);
  }, [connect, selectedModelId]);

  const handleCancelPreparing = useCallback(() => {
    void disconnect();
    setHadCall(false);
    setTimer(0);
  }, [disconnect]);

  const handleEndCall = useCallback(async () => {
    await disconnect();
    setHadCall(true);
  }, [disconnect]);

  const handleReturnHome = useCallback(() => {
    setHadCall(false);
    setTimer(0);
  }, []);

  const view = useMemo(() => {
    if (state === "initializing" || state === "connecting") return "preparing";
    if (isActiveCall) return "listening";
    if (state === "idle" && hadCall) return "ended";
    return "landing";
  }, [hadCall, isActiveCall, state]);

  return (
    <div className="min-h-screen bg-[#FAFAFA] font-sans text-[#1C1C1E] antialiased selection:bg-[#D4A373]/20">
      <AnimatePresence mode="wait">
        {view === "landing" ? <LandingPage key="landing" onStartSession={handleStart} onOpenEarlyAccess={() => setShowEarlyAccess(true)} /> : null}
        {view === "preparing" ? <PreparingPage key="preparing" onCancel={handleCancelPreparing} /> : null}
        {view === "listening" ? (
          <ListeningPage
            key="listening"
            isMuted={isMuted}
            timer={timer}
            onToggleMute={() => void toggleMute()}
            onEnd={() => void handleEndCall()}
            onOpenEarlyAccess={() => setShowEarlyAccess(true)}
          />
        ) : null}
        {view === "ended" ? <EndSessionPage key="ended" onReturnHome={handleReturnHome} onOpenEarlyAccess={() => setShowEarlyAccess(true)} /> : null}
      </AnimatePresence>
      <EarlyAccessModal isOpen={showEarlyAccess} onClose={() => setShowEarlyAccess(false)} />
    </div>
  );
}

export default HomePage;

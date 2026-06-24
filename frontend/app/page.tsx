"use client";

import { AnimatePresence } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";
import { EndSessionPage } from "@/components/elsewhere/EndSessionPage";
import { LandingPage } from "@/components/elsewhere/LandingPage";
import { ListeningPage } from "@/components/elsewhere/ListeningPage";
import { PreparingPage } from "@/components/elsewhere/PreparingPage";
import { SessionEndingDialog } from "@/components/elsewhere/SessionEndingDialog";
import { useVoiceClient } from "@/hooks/useVoiceClient";

function HomePage() {
  const [selectedModelId] = useState("openai/gpt-4o");
  const [hadCall, setHadCall] = useState(false);
  const [timer, setTimer] = useState(0);
  // When the agent signals the session is ending, we anchor an absolute end time
  // and tick toward it to drive the countdown popup.
  const [endingAtMs, setEndingAtMs] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState(0);

  // The agent ended the session on its own (e.g. the 7-minute time limit): show
  // the end screen, same as if the user had pressed End.
  const handleServerDisconnect = useCallback(() => {
    setHadCall(true);
    setEndingAtMs(null);
  }, []);

  const handleSessionEndingSoon = useCallback((secondsRemaining: number) => {
    setEndingAtMs(Date.now() + secondsRemaining * 1000);
    setNowMs(Date.now());
  }, []);

  const { state, connect, disconnect, toggleMute, isMuted } = useVoiceClient({
    onServerDisconnect: handleServerDisconnect,
    onSessionEndingSoon: handleSessionEndingSoon,
  });

  const isActiveCall = state === "connected" || state === "muted";

  useEffect(() => {
    if (endingAtMs === null) return;
    const id = setInterval(() => setNowMs(Date.now()), 500);
    return () => clearInterval(id);
  }, [endingAtMs]);

  const endingSecondsLeft =
    endingAtMs === null ? null : Math.max(0, Math.ceil((endingAtMs - nowMs) / 1000));

  useEffect(() => {
    if (!isActiveCall) return;
    const interval = setInterval(() => setTimer((current) => current + 1), 1000);
    return () => clearInterval(interval);
  }, [isActiveCall]);

  const handleStart = useCallback(() => {
    setHadCall(false);
    setTimer(0);
    setEndingAtMs(null);
    void connect(selectedModelId);
  }, [connect, selectedModelId]);

  const handleCancelPreparing = useCallback(() => {
    void disconnect();
    setHadCall(false);
    setTimer(0);
    setEndingAtMs(null);
  }, [disconnect]);

  const handleEndCall = useCallback(async () => {
    await disconnect();
    setHadCall(true);
    setEndingAtMs(null);
  }, [disconnect]);

  const handleReturnHome = useCallback(() => {
    setHadCall(false);
    setTimer(0);
  }, []);

  // Leaving mid-session via the "Elsewhere" brand: end the call and go straight
  // home (the landing view), skipping the feedback screen since the user is
  // explicitly navigating away rather than wrapping up.
  const handleLeaveSession = useCallback(async () => {
    await disconnect();
    setHadCall(false);
    setTimer(0);
  }, [disconnect]);

  const view = useMemo(() => {
    if (state === "initializing" || state === "connecting") return "preparing";
    if (isActiveCall) return "listening";
    if (state === "idle" && hadCall) return "ended";
    return "landing";
  }, [hadCall, isActiveCall, state]);

  return (
    <div className="min-h-screen bg-[#FAFAFA] font-sans text-[#1C1C1E] antialiased selection:bg-[#D4A373]/20">
      <AnimatePresence mode="wait">
        {view === "landing" ? (
          <LandingPage key="landing" onStartSession={handleStart} onHome={handleReturnHome} />
        ) : null}
        {view === "preparing" ? <PreparingPage key="preparing" onCancel={handleCancelPreparing} /> : null}
        {view === "listening" ? (
          <ListeningPage
            key="listening"
            isMuted={isMuted}
            timer={timer}
            onToggleMute={() => void toggleMute()}
            onEnd={() => void handleEndCall()}
            onLeaveHome={() => void handleLeaveSession()}
          />
        ) : null}
        {view === "ended" ? <EndSessionPage key="ended" onReturnHome={handleReturnHome} /> : null}
      </AnimatePresence>
      <SessionEndingDialog
        open={view === "listening" && endingSecondsLeft !== null}
        secondsLeft={endingSecondsLeft ?? 0}
      />
    </div>
  );
}

export default HomePage;

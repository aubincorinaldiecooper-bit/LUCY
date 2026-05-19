"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, MessageCircle, Mic, MicOff, PhoneOff } from "lucide-react";
import { useVoiceClient } from "@/hooks/useVoiceClient";

type MousePos = { x: number; y: number };

type CharacterPos = { faceX: number; faceY: number; bodySkew: number };

const COLORS = {
  bg: "#F9FAF7",
  text: "#4A5D23",
  muted: "#8C9675",
  surface: "#E8E9EB",
  red: "#E84545",
  purple: "#6C3FF5",
  black: "#2D2D2D",
  orange: "#FF9B6B",
  yellow: "#E8D754",
};

function useBlinking(enabled: boolean) {
  const [isBlinking, setIsBlinking] = useState(false);

  useEffect(() => {
    if (!enabled) {
      setIsBlinking(false);
      return;
    }

    let outerTimer: ReturnType<typeof setTimeout> | undefined;
    let innerTimer: ReturnType<typeof setTimeout> | undefined;

    const schedule = () => {
      outerTimer = setTimeout(() => {
        setIsBlinking(true);
        innerTimer = setTimeout(() => {
          setIsBlinking(false);
          schedule();
        }, 150);
      }, Math.random() * 4000 + 3000);
    };

    schedule();

    return () => {
      if (outerTimer) clearTimeout(outerTimer);
      if (innerTimer) clearTimeout(innerTimer);
    };
  }, [enabled]);

  return isBlinking;
}

function useIsTouchDevice() {
  const [isTouch, setIsTouch] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setIsTouch(window.matchMedia("(pointer: coarse)").matches);
  }, []);

  return isTouch;
}

function SimpleEye({ isBlinking, size = 14, pupilSize = 5, pupilOffset = { x: 0, y: 0 } }: { isBlinking: boolean; size?: number; pupilSize?: number; pupilOffset?: { x: number; y: number } }) {
  return (
    <div
      className="rounded-full bg-white flex items-center justify-center transition-all duration-150"
      style={{ width: size, height: isBlinking ? 2 : size, overflow: "hidden" }}
    >
      {!isBlinking ? (
        <div
          className="rounded-full"
          style={{ width: pupilSize, height: pupilSize, backgroundColor: COLORS.black, transform: `translate(${pupilOffset.x}px, ${pupilOffset.y}px)` }}
        />
      ) : null}
    </div>
  );
}

function Pupil({ size, offset }: { size: number; offset: { x: number; y: number } }) {
  return (
    <div
      className="rounded-full transition-transform duration-100 ease-out"
      style={{ width: size, height: size, backgroundColor: COLORS.black, transform: `translate(${offset.x}px, ${offset.y}px)` }}
    />
  );
}

function EyeBall({ size, pupilSize, isBlinking, pupilOffset }: { size: number; pupilSize: number; isBlinking: boolean; pupilOffset: { x: number; y: number } }) {
  return (
    <div
      className="rounded-full flex items-center justify-center transition-all duration-150"
      style={{ width: size, height: isBlinking ? 2 : size, backgroundColor: "white", overflow: "hidden" }}
    >
      {!isBlinking ? <Pupil size={pupilSize} offset={pupilOffset} /> : null}
    </div>
  );
}

function PurpleCharacter({ isBlinking, scale = 1 }: { isBlinking: boolean; scale?: number }) {
  const width = 140 * scale;
  const height = 200 * scale;
  const eyeSize = 12 * scale;
  const pupilSize = 4 * scale;
  const eyeTop = 55 * scale;

  return (
    <div className="relative" style={{ width, height, backgroundColor: COLORS.purple, borderRadius: "20px 20px 0 0" }}>
      <div className="absolute flex gap-3" style={{ left: "50%", transform: "translateX(-50%)", top: eyeTop }}>
        <SimpleEye isBlinking={isBlinking} size={eyeSize} pupilSize={pupilSize} pupilOffset={{ x: 1, y: 0 }} />
        <SimpleEye isBlinking={isBlinking} size={eyeSize} pupilSize={pupilSize} pupilOffset={{ x: 1, y: 0 }} />
      </div>
    </div>
  );
}

function formatTime(seconds: number) {
  const mins = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const secs = (seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function HomePage() {
  const { state, connect, disconnect, toggleMute } = useVoiceClient();
  const [selectedModelId] = useState("openai/gpt-4o");
  const [hadCall, setHadCall] = useState(false);
  const [timer, setTimer] = useState(0);

  const isTouch = useIsTouchDevice();
  const trackingEnabled = !isTouch;

  const [mousePos, setMousePos] = useState<MousePos>({ x: 0, y: 0 });
  useEffect(() => {
    if (!trackingEnabled) return;
    const onMove = (event: MouseEvent) => setMousePos({ x: event.clientX, y: event.clientY });
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [trackingEnabled]);

  const purpleBlink = useBlinking(true);
  const blackBlink = useBlinking(true);
  const callBlink = useBlinking(state === "connected" || state === "muted");

  const purpleRef = useRef<HTMLDivElement | null>(null);
  const blackRef = useRef<HTMLDivElement | null>(null);
  const yellowRef = useRef<HTMLDivElement | null>(null);
  const orangeRef = useRef<HTMLDivElement | null>(null);

  const calculatePosition = useCallback(
    (ref: React.RefObject<HTMLDivElement | null>): CharacterPos => {
      if (!trackingEnabled || !ref.current) return { faceX: 0, faceY: 0, bodySkew: 0 };
      const rect = ref.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 3;
      const deltaX = mousePos.x - centerX;
      const deltaY = mousePos.y - centerY;
      return {
        faceX: Math.max(-15, Math.min(15, deltaX / 20)),
        faceY: Math.max(-10, Math.min(10, deltaY / 30)),
        bodySkew: Math.max(-6, Math.min(6, -deltaX / 120)),
      };
    },
    [mousePos, trackingEnabled],
  );

  const purplePos = calculatePosition(purpleRef);
  const blackPos = calculatePosition(blackRef);
  const yellowPos = calculatePosition(yellowRef);
  const orangePos = calculatePosition(orangeRef);

  useEffect(() => {
    const isActiveCall = state === "connected" || state === "muted";
    if (!isActiveCall) return;
    const interval = setInterval(() => setTimer((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, [state]);

  const handleStart = useCallback(() => {
    setHadCall(false);
    setTimer(0);
    void connect(selectedModelId);
  }, [connect, selectedModelId]);

  const handleEndCall = useCallback(async () => {
    await disconnect();
    setHadCall(true);
  }, [disconnect]);

  const handleReset = useCallback(() => {
    setHadCall(false);
    setTimer(0);
  }, []);

  const view = useMemo(() => {
    if (state === "connecting" || state === "initializing") return "connecting";
    if (state === "connected" || state === "muted") return "call";
    if (state === "idle" && hadCall) return "ended";
    return "landing";
  }, [hadCall, state]);

  if (view === "connecting") {
    return <ConnectingView />;
  }

  if (view === "call") {
    return (
      <CallView
        isMuted={state === "muted"}
        timer={timer}
        onToggleMute={() => void toggleMute()}
        onEndCall={() => void handleEndCall()}
        isBlinking={callBlink}
      />
    );
  }

  if (view === "ended") {
    return <EndedView onReset={handleReset} />;
  }

  return (
    <LandingView
      onStart={handleStart}
      refs={{ purpleRef, blackRef, yellowRef, orangeRef }}
      positions={{ purplePos, blackPos, yellowPos, orangePos }}
      isPurpleBlinking={purpleBlink}
      isBlackBlinking={blackBlink}
    />
  );
}

function AppFrame({ children }: { children: React.ReactNode }) {
  return (
    <main className="min-h-screen w-full relative overflow-hidden" style={{ backgroundColor: COLORS.bg, color: COLORS.text }}>
      {children}
      <div
        className="absolute inset-0 opacity-[0.03] pointer-events-none"
        style={{ backgroundImage: `radial-gradient(${COLORS.text} 1px, transparent 1px)`, backgroundSize: "24px 24px" }}
      />
    </main>
  );
}

function FooterText() {
  return <div className="text-xs text-[#8C9675] text-center px-4">By continuing, you agree to Sine Studio&apos;s Terms of Use and Privacy Policy.</div>;
}

function ConnectingView() {
  return (
    <AppFrame>
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-4">
        <div className="w-full max-w-xs sm:max-w-sm bg-[#E8E9EB] rounded-3xl py-10 px-6 text-center shadow-sm animate-[fadeIn_0.45s_ease-out]">
          <div className="flex justify-center gap-1 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-[#4A5D23] animate-bounce" />
            <div className="w-1.5 h-1.5 rounded-full bg-[#4A5D23] animate-bounce [animation-delay:120ms]" />
            <div className="w-1.5 h-1.5 rounded-full bg-[#4A5D23] animate-bounce [animation-delay:240ms]" />
          </div>
          <p className="text-lg font-medium">Connecting...</p>
        </div>
        <div className="absolute bottom-7 left-0 right-0"><FooterText /></div>
      </div>
    </AppFrame>
  );
}

function CallView({ isMuted, timer, onToggleMute, onEndCall, isBlinking }: { isMuted: boolean; timer: number; onToggleMute: () => void; onEndCall: () => void; isBlinking: boolean }) {
  return (
    <AppFrame>
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-between py-8 px-4">
        <div />
        <div className="w-full flex flex-col items-center gap-8 animate-[fadeIn_0.45s_ease-out]">
          <div className="h-80 flex items-end justify-center w-full">
            <PurpleCharacter isBlinking={isBlinking} scale={1.2} />
          </div>
          <div className="font-mono text-lg text-[#8C9675]">{formatTime(timer)}</div>
          <div className="flex items-center gap-2 bg-white p-2 rounded-full border border-slate-100 shadow-md">
            <button onClick={onToggleMute} className="px-5 py-3 rounded-full bg-[#F9FAF7] font-medium flex items-center gap-2">
              {isMuted ? <MicOff size={18} /> : <Mic size={18} />}
              {isMuted ? "Unmute" : "Mute"}
            </button>
            <button onClick={onEndCall} className="px-5 py-3 rounded-full text-white font-medium flex items-center gap-2" style={{ backgroundColor: COLORS.red }}>
              <PhoneOff size={18} />
              End call
            </button>
          </div>
        </div>
        <div className="pb-2 text-xs text-[#8C9675]">AI responses may be inaccurate. Verify important information.</div>
      </div>
    </AppFrame>
  );
}

function EndedView({ onReset }: { onReset: () => void }) {
  return (
    <AppFrame>
      <div className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4">
        <div className="w-full max-w-md bg-[#E8E9EB] rounded-3xl shadow-sm p-8 text-center space-y-6 animate-[fadeIn_0.45s_ease-out]">
          <div>
            <h2 className="text-xl font-bold mb-2">Subscription coming soon</h2>
            <p className="text-sm text-[#8C9675]">Includes long-term memory, access to new features, and 30 minute sessions.</p>
          </div>
          <button disabled className="w-full py-3.5 rounded-full bg-slate-400 text-white opacity-70 cursor-not-allowed">Subscribe</button>
          <button onClick={onReset} className="w-full py-3.5 rounded-full bg-white font-medium flex items-center justify-center gap-2">
            Have another conversation
            <ArrowRight size={16} />
          </button>
        </div>
        <div className="absolute bottom-7 left-0 right-0"><FooterText /></div>
      </div>
    </AppFrame>
  );
}

function LandingView({
  onStart,
  refs,
  positions,
  isPurpleBlinking,
  isBlackBlinking,
}: {
  onStart: () => void;
  refs: { purpleRef: React.RefObject<HTMLDivElement | null>; blackRef: React.RefObject<HTMLDivElement | null>; yellowRef: React.RefObject<HTMLDivElement | null>; orangeRef: React.RefObject<HTMLDivElement | null> };
  positions: { purplePos: CharacterPos; blackPos: CharacterPos; yellowPos: CharacterPos; orangePos: CharacterPos };
  isPurpleBlinking: boolean;
  isBlackBlinking: boolean;
}) {
  return (
    <AppFrame>
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-4 py-10">
        <div className="w-full max-w-[550px] h-[320px] sm:h-[380px] md:h-[400px] relative scale-[0.74] sm:scale-[0.86] md:scale-100 origin-bottom animate-[fadeIn_0.45s_ease-out]">
          <div
            ref={refs.purpleRef}
            className="absolute bottom-0 shadow-lg transition-transform duration-200"
            style={{ left: 70, width: 180, height: 400, backgroundColor: COLORS.purple, borderRadius: "10px 10px 0 0", transform: `skewX(${positions.purplePos.bodySkew}deg)` }}
          >
            <div className="absolute flex gap-8" style={{ left: 45 + positions.purplePos.faceX, top: 40 + positions.purplePos.faceY }}>
              <EyeBall size={18} pupilSize={7} isBlinking={isPurpleBlinking} pupilOffset={{ x: positions.purplePos.faceX / 3, y: positions.purplePos.faceY / 4 }} />
              <EyeBall size={18} pupilSize={7} isBlinking={isPurpleBlinking} pupilOffset={{ x: positions.purplePos.faceX / 3, y: positions.purplePos.faceY / 4 }} />
            </div>
          </div>

          <div
            ref={refs.blackRef}
            className="absolute bottom-0 shadow-lg transition-transform duration-200"
            style={{ left: 240, width: 120, height: 310, backgroundColor: COLORS.black, borderRadius: "8px 8px 0 0", transform: `skewX(${positions.blackPos.bodySkew}deg)` }}
          >
            <div className="absolute flex gap-6" style={{ left: 26 + positions.blackPos.faceX, top: 32 + positions.blackPos.faceY }}>
              <EyeBall size={16} pupilSize={6} isBlinking={isBlackBlinking} pupilOffset={{ x: positions.blackPos.faceX / 3, y: positions.blackPos.faceY / 4 }} />
              <EyeBall size={16} pupilSize={6} isBlinking={isBlackBlinking} pupilOffset={{ x: positions.blackPos.faceX / 3, y: positions.blackPos.faceY / 4 }} />
            </div>
          </div>

          <div
            ref={refs.orangeRef}
            className="absolute bottom-0 shadow-lg transition-transform duration-200"
            style={{ left: 0, width: 240, height: 200, backgroundColor: COLORS.orange, borderRadius: "120px 120px 0 0", transform: `skewX(${positions.orangePos.bodySkew}deg)` }}
          >
            <div className="absolute flex gap-8" style={{ left: 82 + positions.orangePos.faceX, top: 90 + positions.orangePos.faceY }}>
              <Pupil size={12} offset={{ x: positions.orangePos.faceX / 3, y: positions.orangePos.faceY / 4 }} />
              <Pupil size={12} offset={{ x: positions.orangePos.faceX / 3, y: positions.orangePos.faceY / 4 }} />
            </div>
          </div>

          <div
            ref={refs.yellowRef}
            className="absolute bottom-0 shadow-lg transition-transform duration-200"
            style={{ left: 310, width: 140, height: 230, backgroundColor: COLORS.yellow, borderRadius: "70px 70px 0 0", transform: `skewX(${positions.yellowPos.bodySkew}deg)` }}
          >
            <div className="absolute flex gap-6" style={{ left: 52 + positions.yellowPos.faceX, top: 40 + positions.yellowPos.faceY }}>
              <Pupil size={12} offset={{ x: positions.yellowPos.faceX / 3, y: positions.yellowPos.faceY / 4 }} />
              <Pupil size={12} offset={{ x: positions.yellowPos.faceX / 3, y: positions.yellowPos.faceY / 4 }} />
            </div>
            <div className="absolute w-20 h-1 rounded-full" style={{ left: 40 + positions.yellowPos.faceX, top: 88 + positions.yellowPos.faceY, backgroundColor: COLORS.black }} />
          </div>
        </div>

        <h2 className="mt-8 text-2xl font-semibold tracking-tight animate-[fadeIn_0.45s_ease-out]">imaginary friends</h2>
        <button
          onClick={onStart}
          className="mt-6 inline-flex items-center gap-2 px-6 py-3 text-white font-medium rounded-full shadow-lg transition hover:brightness-95 animate-[fadeIn_0.45s_ease-out]"
          style={{ backgroundColor: COLORS.purple }}
        >
          <MessageCircle className="size-4" />
          start a conversation
        </button>

        <div className="absolute bottom-7 left-0 right-0"><FooterText /></div>
      </div>
    </AppFrame>
  );
}

export default HomePage;

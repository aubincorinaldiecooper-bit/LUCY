"use client";

import { motion } from "framer-motion";
import { ArrowUp, ChevronDown, Keyboard, Mic, MicOff, Phone, X } from "lucide-react";
import { type KeyboardEvent, type ReactNode, useCallback, useMemo, useState } from "react";
import type { VoiceState } from "@/hooks/useVoiceClient";
import Waveform from "@/components/Waveform";

type ConversationMessage = {
  source: "user" | "ai";
  message: string;
};

type ConversationBarProps = {
  state: VoiceState;
  barHeights: number[];
  className?: string;
  waveformClassName?: string;
  rightSlot?: ReactNode;
  onConnect: () => void;
  onDisconnect: () => void;
  onToggleMute: () => void;
  onMessage?: (message: ConversationMessage) => void;
  onSendMessage?: (message: string) => void;
};

function DotPlaceholder() {
  return (
    <div className="w-full h-full flex items-center justify-center gap-[6px] px-3">
      {Array.from({ length: 18 }).map((_, i) => (
        <span key={i} className="w-[3px] h-[3px] rounded-full bg-[#CBD5E1]" />
      ))}
    </div>
  );
}

export function ConversationBar({
  state,
  barHeights,
  className,
  waveformClassName,
  rightSlot,
  onConnect,
  onDisconnect,
  onToggleMute,
  onMessage,
  onSendMessage,
}: ConversationBarProps) {
  const [keyboardOpen, setKeyboardOpen] = useState(false);
  const [textInput, setTextInput] = useState("");

  const isConnected = state === "connected" || state === "muted";
  const isMuted = state === "muted";
  const isLoading = state === "initializing" || state === "connecting";
  const handleStartOrEnd = useCallback(() => {
    if (isConnected || isLoading) {
      onDisconnect();
      return;
    }
    onConnect();
  }, [isConnected, isLoading, onConnect, onDisconnect]);

  const handleSend = useCallback(() => {
    const trimmed = textInput.trim();
    if (!trimmed || !isConnected) {
      return;
    }

    onSendMessage?.(trimmed);
    onMessage?.({ source: "user", message: trimmed });
    setTextInput("");
  }, [isConnected, onMessage, onSendMessage, textInput]);

  const handleTextKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const statusLabel = useMemo(() => {
    if (state === "initializing") return "Initializing…";
    if (state === "connecting") return "Connecting…";
    if (state === "connected") return "Connected";
    if (state === "muted") return "Muted";
    return "Ready";
  }, [state]);

  return (
    <div className={className}>
      <div className="rounded-[2rem] border border-[#CBD5E1] bg-[#F8FAFC]/85 backdrop-blur-md shadow-[0_10px_30px_rgba(15,23,42,0.12)] p-2.5 sm:p-3">
        <div className="flex flex-col-reverse">
          <div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-[50px] rounded-2xl border border-[#E2E8F0] bg-[#EFF3F7] px-3 flex items-center justify-center overflow-hidden">
                <div className={`w-full ${waveformClassName ?? ""}`}>
                  <Waveform barHeights={barHeights} active={isConnected && !isMuted} />
                </div>
              </div>

              <button
                type="button"
                className="h-10 w-10 rounded-full flex items-center justify-center text-[#64748B] hover:bg-[#E2E8F0] disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={onToggleMute}
                disabled={!isConnected}
                aria-pressed={isMuted}
                aria-label={isMuted ? "Unmute microphone" : "Mute microphone"}
              >
                {isMuted ? <MicOff size={19} /> : <Mic size={19} />}
              </button>

              <button
                type="button"
                className="relative h-10 w-10 rounded-full flex items-center justify-center text-[#475569] hover:bg-[#E2E8F0] disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={() => setKeyboardOpen((prev) => !prev)}
                disabled={!isConnected}
                aria-pressed={keyboardOpen}
                aria-label="Toggle keyboard"
              >
                <Keyboard
                  className={`absolute transition-all duration-200 ${
                    keyboardOpen ? "opacity-0 scale-75" : "opacity-100 scale-100"
                  }`}
                  size={19}
                />
                <ChevronDown
                  className={`absolute transition-all duration-200 ${
                    keyboardOpen ? "opacity-100 scale-100" : "opacity-0 scale-75"
                  }`}
                  size={19}
                />
              </button>

              <button
                type="button"
                className="h-10 w-10 rounded-full flex items-center justify-center text-[#475569] hover:bg-[#E2E8F0] disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={handleStartOrEnd}
                aria-label={isConnected || isLoading ? "End conversation" : "Start conversation"}
              >
                {isConnected || isLoading ? <X size={19} /> : <Phone size={19} />}
              </button>

              {rightSlot}
            </div>

            <p className="mt-1 px-1 text-[11px] tracking-wide uppercase text-[#64748B]">{statusLabel}</p>
          </div>

          <motion.div
            initial={false}
            animate={{ maxHeight: keyboardOpen ? 132 : 0, opacity: keyboardOpen ? 1 : 0 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="relative pb-2 px-1">
              <textarea
                value={textInput}
                onChange={(event) => setTextInput(event.target.value)}
                onKeyDown={handleTextKeyDown}
                placeholder="Send a text update..."
                className="w-full min-h-[96px] resize-none rounded-xl border border-[#E2E8F0] bg-white px-3 py-2 pr-12 text-sm text-[#1E293B] outline-none focus:border-[#94A3B8]"
                disabled={!isConnected}
              />
              <button
                type="button"
                className="absolute right-4 bottom-5 h-8 w-8 rounded-full flex items-center justify-center text-[#334155] hover:bg-[#F1F5F9] disabled:opacity-35"
                onClick={handleSend}
                disabled={!isConnected || !textInput.trim()}
                aria-label="Send message"
              >
                <ArrowUp size={16} />
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";
import { ArrowUp, Keyboard, Mic, MicOff, X } from "lucide-react";
import { type KeyboardEvent, useCallback, useState } from "react";
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

  const handleConnectOrEnd = useCallback(() => {
    if (isConnected || isLoading) {
      onDisconnect();
      return;
    }
    onConnect();
  }, [isConnected, isLoading, onConnect, onDisconnect]);

  const handleSend = useCallback(() => {
    const trimmed = textInput.trim();
    if (!trimmed || !isConnected) return;
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

  return (
    <div className={className}>
      <div className="rounded-[2rem] border border-[#CBD5E1] bg-[#F8FAFC]/85 backdrop-blur-md shadow-[0_10px_30px_rgba(15,23,42,0.12)] p-3 sm:p-4">
        <motion.div
          initial={false}
          animate={{ maxHeight: keyboardOpen ? 132 : 0, opacity: keyboardOpen ? 1 : 0 }}
          transition={{ duration: 0.22, ease: "easeOut" }}
          className="overflow-hidden"
        >
          <div className="relative pb-3">
            <textarea
              value={textInput}
              onChange={(event) => setTextInput(event.target.value)}
              onKeyDown={handleTextKeyDown}
              placeholder="Send a text update..."
              className="w-full min-h-[96px] resize-none rounded-2xl border border-[#E2E8F0] bg-white px-3 py-2 pr-12 text-sm text-[#1E293B] outline-none focus:border-[#94A3B8]"
              disabled={!isConnected}
            />
            <button
              type="button"
              className="absolute right-3 bottom-6 h-8 w-8 rounded-full flex items-center justify-center text-[#334155] hover:bg-[#F1F5F9] disabled:opacity-35"
              onClick={handleSend}
              disabled={!isConnected || !textInput.trim()}
              aria-label="Send message"
            >
              <ArrowUp size={16} />
            </button>
          </div>
        </motion.div>

        <div className="flex items-center gap-2">
          <div className="flex-1 h-[52px] rounded-2xl border border-[#E2E8F0] bg-[#EFF3F7] px-3 flex items-center justify-center overflow-hidden">
            {isConnected ? <Waveform barHeights={barHeights} active={!isMuted} /> : <DotPlaceholder />}
          </div>

          <button
            type="button"
            className="h-10 w-10 rounded-full flex items-center justify-center text-[#64748B] hover:bg-[#E2E8F0] disabled:opacity-40 disabled:cursor-not-allowed"
            onClick={onToggleMute}
            disabled={!isConnected}
            aria-label={isMuted ? "Unmute microphone" : "Mute microphone"}
          >
            {isMuted ? <MicOff size={22} /> : <Mic size={22} />}
          </button>

          <button
            type="button"
            className="h-10 w-10 rounded-full flex items-center justify-center text-[#64748B] hover:bg-[#E2E8F0] disabled:opacity-40 disabled:cursor-not-allowed"
            onClick={() => setKeyboardOpen((prev) => !prev)}
            disabled={!isConnected}
            aria-label="Toggle keyboard"
          >
            <Keyboard size={22} />
          </button>

          <button
            type="button"
            className="h-10 w-10 rounded-full flex items-center justify-center text-[#475569] hover:bg-[#E2E8F0]"
            onClick={handleConnectOrEnd}
            aria-label={isConnected || isLoading ? "End conversation" : "Start conversation"}
          >
            <X size={24} />
          </button>
        </div>
      </div>
    </div>
  );
}

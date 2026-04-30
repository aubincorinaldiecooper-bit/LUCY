"use client";

import { motion } from "framer-motion";
import {
  ArrowUp,
  Check,
  ChevronDown,
  Keyboard,
  Mic,
  MicOff,
  Phone,
  Search,
  X,
} from "lucide-react";
import {
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useMemo,
  useState,
} from "react";
import type { VoiceState } from "@/hooks/useVoiceClient";
import Waveform from "@/components/Waveform";

type ConversationMessage = {
  source: "user" | "ai";
  message: string;
};

type ModelOption = {
  id: string;
  name: string;
  provider: "openai" | "anthropic" | "minimax" | "deepseek";
  badge: string;
  tone: string;
  bg: string;
};

const MODEL_OPTIONS: ModelOption[] = [
  { id: "gpt-4.1", name: "GPT-4.1", provider: "openai", badge: "G", tone: "#4A6CF7", bg: "#F0F4FF" },
  { id: "gpt-4o", name: "GPT-4o", provider: "openai", badge: "G", tone: "#4A6CF7", bg: "#F0F4FF" },
  { id: "claude-sonnet-4", name: "Claude Sonnet 4", provider: "anthropic", badge: "C", tone: "#D9783E", bg: "#FFF3E0" },
  { id: "claude-opus-4", name: "Claude Opus 4", provider: "anthropic", badge: "C", tone: "#D9783E", bg: "#FFF3E0" },
  { id: "minimax-m1", name: "MiniMax M1", provider: "minimax", badge: "M", tone: "#8B5CF6", bg: "#F3E8FF" },
  { id: "minimax-text-01", name: "MiniMax Text-01", provider: "minimax", badge: "M", tone: "#8B5CF6", bg: "#F3E8FF" },
  { id: "deepseek-r1", name: "DeepSeek R1", provider: "deepseek", badge: "D", tone: "#0891B2", bg: "#E0F2FE" },
  { id: "deepseek-v3", name: "DeepSeek V3", provider: "deepseek", badge: "D", tone: "#0891B2", bg: "#E0F2FE" },
];

const PROVIDER_LABEL: Record<ModelOption["provider"], string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  minimax: "MiniMax",
  deepseek: "DeepSeek",
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
  selectedModelId?: string;
  onModelChange?: (modelId: string) => void;
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
  selectedModelId = "gpt-4o",
  onModelChange,
}: ConversationBarProps) {
  const [keyboardOpen, setKeyboardOpen] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [search, setSearch] = useState("");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const isConnected = state === "connected" || state === "muted";
  const isMuted = state === "muted";
  const isLoading = state === "initializing" || state === "connecting";

  const selectedModel = useMemo(
    () => MODEL_OPTIONS.find((m) => m.id === selectedModelId) ?? MODEL_OPTIONS[0],
    [selectedModelId]
  );

  const filteredModels = useMemo(() => {
    const query = search.trim().toLowerCase();
    return MODEL_OPTIONS.filter(
      (model) => !query || `${model.name} ${model.provider}`.toLowerCase().includes(query)
    );
  }, [search]);

  const groupedModels = useMemo(() => {
    const groups: Record<ModelOption["provider"], ModelOption[]> = {
      openai: [],
      anthropic: [],
      minimax: [],
      deepseek: [],
    };
    filteredModels.forEach((model) => groups[model.provider].push(model));
    return groups;
  }, [filteredModels]);

  const handleStartOrEnd = useCallback(() => {
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
        {/* Click-away overlay for model dropdown */}
        {dropdownOpen && (
          <button
            type="button"
            className="fixed inset-0 z-40"
            onClick={() => setDropdownOpen(false)}
            aria-label="Close model menu"
          />
        )}

        <div className="flex flex-col-reverse">
          {/* Collapsible text input (above the bar) */}
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

          {/* Main bar + status */}
          <div>
            <div className="flex items-center gap-2">
              {/* Waveform area */}
              <div className="flex-1 h-[50px] rounded-2xl border border-[#E2E8F0] bg-[#EFF3F7] px-3 flex items-center justify-center overflow-hidden">
                <div className={`w-full ${waveformClassName ?? ""}`}>
                  <Waveform barHeights={barHeights} active={isConnected && !isMuted} />
                </div>
              </div>

              {/* Mute button */}
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

              {/* Keyboard toggle */}
              <button
                type="button"
                className="h-10 w-10 rounded-full flex items-center justify-center text-[#475569] hover:bg-[#E2E8F0] disabled:opacity-40 disabled:cursor-not-allowed"
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

              {/* Model selector button (only when not connected) */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => {
                    if (isConnected || isLoading) return;
                    setDropdownOpen((prev) => !prev);
                  }}
                  className="inline-flex items-center gap-1 h-10 px-2 rounded-full text-[#475569] hover:bg-[#E2E8F0] disabled:opacity-40 disabled:cursor-not-allowed"
                  disabled={isConnected || isLoading}
                  aria-expanded={dropdownOpen}
                  aria-haspopup="listbox"
                >
                  <span
                    className="inline-flex h-[18px] w-[18px] items-center justify-center rounded text-[9px] font-semibold"
                    style={{ backgroundColor: selectedModel.bg, color: selectedModel.tone }}
                  >
                    {selectedModel.badge}
                  </span>
                  <span className="text-[11px] hidden sm:inline">{selectedModel.name}</span>
                  <ChevronDown
                    size={14}
                    className={`transition-transform ${dropdownOpen ? "rotate-180" : ""}`}
                  />
                </button>

                {/* Model dropdown (opens above the button) */}
                <div
                  className={`absolute bottom-[calc(100%+8px)] right-0 z-50 w-[210px] rounded-xl border border-[#E2E8F0] bg-white shadow-xl transition-all ${
                    dropdownOpen ? "visible opacity-100 translate-y-0" : "invisible opacity-0 translate-y-1"
                  }`}
                >
                  <div className="p-2 border-b border-[#E2E8F0]">
                    <div className="relative">
                      <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-[#94A3B8]" />
                      <input
                        value={search}
                        onChange={(event) => setSearch(event.target.value)}
                        placeholder="Search model..."
                        className="w-full rounded-md border border-[#E2E8F0] bg-[#F8FAFC] py-1 pl-6 pr-2 text-[10px] text-[#1E293B] outline-none focus:border-[#94A3B8]"
                      />
                    </div>
                  </div>

                  <div className="max-h-[220px] overflow-y-auto p-1">
                    {(Object.keys(groupedModels) as ModelOption["provider"][]).map((provider) => {
                      const models = groupedModels[provider];
                      if (!models.length) return null;

                      return (
                        <div key={provider} className="mb-1">
                          <div className="px-2 py-1 text-[8px] uppercase tracking-widest text-[#94A3B8]">
                            {PROVIDER_LABEL[provider]}
                          </div>
                          {models.map((model) => (
                            <button
                              key={model.id}
                              type="button"
                              className={`w-full flex items-center gap-2 rounded-md px-2 py-1 text-left hover:bg-[#F1F5F9] ${
                                selectedModelId === model.id ? "bg-[#F1F5F9]" : ""
                              }`}
                              onClick={() => {
                                onModelChange?.(model.id);
                                setDropdownOpen(false);
                                setSearch("");
                              }}
                            >
                              <span
                                className="inline-flex h-[18px] w-[18px] items-center justify-center rounded text-[9px] font-semibold"
                                style={{ backgroundColor: model.bg, color: model.tone }}
                              >
                                {model.badge}
                              </span>
                              <span className="flex-1 text-[10px] text-[#334155]">{model.name}</span>
                              {selectedModelId === model.id && <Check size={12} className="text-[#3B82F6]" />}
                            </button>
                          ))}
                        </div>
                      );
                    })}
                    {!filteredModels.length && (
                      <div className="px-3 py-5 text-center text-[10px] text-[#94A3B8]">No models found</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Connect / Disconnect button */}
              <button
                type="button"
                className="relative h-10 w-10 rounded-full flex items-center justify-center text-[#475569] hover:bg-[#E2E8F0] disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={handleStartOrEnd}
                aria-label={isConnected || isLoading ? "End conversation" : "Start conversation"}
              >
                {isConnected || isLoading ? <X size={19} /> : <Phone size={19} />}
              </button>

              {/* External slot (e.g., SettingsPanel) */}
              {rightSlot}
            </div>

            <p className="mt-1 px-1 text-[11px] tracking-wide uppercase text-[#64748B]">{statusLabel}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

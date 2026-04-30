"use client";

import { motion } from "framer-motion";
import { Check, ChevronDown, Mic, MicOff, Phone, Search, X } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import type { VoiceState } from "@/hooks/useVoiceClient";
import Waveform from "@/components/Waveform";

type ModelOption = {
  id: string;
  name: string;
  provider: "openai" | "anthropic" | "minimax" | "deepseek";
  badge: string;
  tone: string;
  bg: string;
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
  selectedModelId?: string;
  onModelChange?: (modelId: string) => void;
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

function DotPlaceholder() {
  const heights = [4,5,6,7,8,9,10,12,14,16,18,20,22,24,22,20,18,16,14,12,10,9,8,7,6,5,4];
  return (
    <div className="w-full h-full flex items-center justify-center gap-[4px] px-3">
      {heights.map((height, i) => (
        <span key={i} className="w-[4px] rounded-full bg-[#D5D0C8]" style={{ height }} />
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
  selectedModelId = "gpt-4o",
  onModelChange,
}: ConversationBarProps) {
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

  const handleConnectOrDisconnect = () => {
    if (isConnected || isLoading) {
      onDisconnect();
      return;
    }
    onConnect();
  };

  const statusText = isLoading
    ? "CONNECTING"
    : isConnected
    ? isMuted
      ? "MUTED"
      : "LISTENING"
    : "READY";
  const statusColor = isConnected ? "#D9934E" : "#6DB87A";

  return (
    <div className={className}>
      <div className="relative rounded-[2.1rem] border border-white/90 bg-white/75 backdrop-blur-[24px] p-5 shadow-[0_1px_2px_rgba(120,110,95,0.04),0_4px_12px_rgba(120,110,95,0.06),0_16px_48px_rgba(120,110,95,0.08)]">
        {dropdownOpen && (
          <button
            type="button"
            className="fixed inset-0 z-40"
            onClick={() => setDropdownOpen(false)}
            aria-label="Close model menu"
          />
        )}

        <div className="relative z-50 flex items-center justify-between">
          <div className="h-12 w-[62%] rounded-2xl border border-[#EAE6DF] bg-[#FAFAF8] px-3 flex items-center justify-center overflow-hidden shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
            {isConnected ? (
              <div className={`w-full ${waveformClassName ?? ""}`}>
                <Waveform barHeights={barHeights} active={!isMuted} />
              </div>
            ) : (
              <DotPlaceholder />
            )}
          </div>

          <div className="ml-4 flex items-center gap-3">
            <button
              type="button"
              onClick={onToggleMute}
              disabled={!isConnected}
              className="h-[34px] w-[34px] rounded-[10px] border border-[#DCD7CD]/80 bg-[#FAF8F5]/90 text-[#A8A296] hover:text-[#D9934E] disabled:opacity-50"
              aria-label={isMuted ? "Unmute mic" : "Mute mic"}
            >
              <span className="flex items-center justify-center">
                {isMuted ? <MicOff size={16} /> : <Mic size={16} />}
              </span>
            </button>

            <button
              type="button"
              onClick={handleConnectOrDisconnect}
              className="h-[34px] w-[34px] rounded-[10px] border border-[#DCD7CD]/80 bg-[#FAF8F5]/90 text-[#A8A296] hover:text-[#D9934E]"
              aria-label={isConnected || isLoading ? "Disconnect" : "Connect"}
            >
              <span className="flex items-center justify-center">
                {isConnected || isLoading ? <X size={16} /> : <Phone size={16} />}
              </span>
            </button>

            {/* External slot (e.g., SettingsPanel) */}
            {rightSlot}
          </div>
        </div>

        <div className="relative z-50 mt-3 border-t border-[#DCD7CD]/60 pt-3 flex items-center gap-2">
          <div
            className="h-3 w-3 rounded-full shadow-[0_0_12px_rgba(109,184,122,0.35)]"
            style={{ backgroundColor: statusColor }}
          />
          <span className="text-[10px] font-medium uppercase tracking-[0.15em] text-[#A8A296]">
            {statusText === "LISTENING" ? "CONNECTED" : statusText}
          </span>
          <div className="mx-1 h-3 w-px bg-[#DCD7CD]" />

          <div className="relative">
            <button
              type="button"
              onClick={() => {
                if (isConnected || isLoading) return;
                setDropdownOpen((prev) => !prev);
              }}
              className="inline-flex items-center gap-1 rounded-md px-1 py-0.5 hover:bg-[#FAF8F5]"
              aria-expanded={dropdownOpen}
              aria-haspopup="listbox"
              disabled={isConnected || isLoading}
            >
              <span
                className="inline-flex h-[18px] w-[18px] items-center justify-center rounded text-[9px] font-semibold"
                style={{ backgroundColor: selectedModel.bg, color: selectedModel.tone }}
              >
                {selectedModel.badge}
              </span>
              <span className="text-[10px] text-[#8A8578]">{selectedModel.name}</span>
              <ChevronDown
                size={12}
                className={`text-[#C8C3BA] transition-transform ${dropdownOpen ? "rotate-180" : ""}`}
              />
            </button>

            <div
              className={`absolute bottom-[calc(100%+8px)] left-1/2 z-50 w-[210px] -translate-x-1/2 rounded-xl border border-white/90 bg-white/95 shadow-xl transition-all ${
                dropdownOpen ? "visible opacity-100 translate-y-0" : "invisible opacity-0 translate-y-1"
              }`}
            >
              <div className="p-2 border-b border-[#DCD7CD]/40">
                <div className="relative">
                  <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-[#C8C3BA]" />
                  <input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Search model..."
                    className="w-full rounded-md border border-[#DC
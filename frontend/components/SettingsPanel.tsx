"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Settings } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type SettingsPanelProps = {
  mics: MediaDeviceInfo[];
  speakers: MediaDeviceInfo[];
  selectedMic: string;
  selectedSpeaker: string;
  onMicChange: (deviceId: string) => void;
  onSpeakerChange: (deviceId: string) => void;
};

export default function SettingsPanel({
  mics,
  speakers,
  selectedMic,
  selectedSpeaker,
  onMicChange,
  onSpeakerChange,
}: SettingsPanelProps) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onMouseDown = (event: MouseEvent) => {
      if (!wrapperRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, []);

  const selectClass =
    "w-full py-2 pl-3 pr-8 rounded-[8px] border border-ctrl-border bg-ctrl-bg text-text-primary text-[0.87rem] appearance-none focus:border-accent focus:outline-none transition-colors";

  const chevronStyle = {
    backgroundImage:
      "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23888888' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")",
    backgroundRepeat: "no-repeat",
    backgroundPosition: "right 0.65rem center",
    backgroundSize: "14px",
  } as const;

  return (
    <div className="relative" ref={wrapperRef}>
      <motion.button
        type="button"
        className={`w-[44px] h-[44px] rounded-[10px] border flex items-center justify-center bg-ctrl-bg ${
          open ? "border-accent" : "border-ctrl-border"
        }`}
        onClick={() => setOpen((prev) => !prev)}
        whileHover={{ scale: 1.02 }}
      >
        <motion.span whileHover={{ rotate: 30 }} transition={{ duration: 0.3 }} className="text-text-secondary">
          <Settings size={18} strokeWidth={1.75} />
        </motion.span>
      </motion.button>

      <AnimatePresence>
        {open && (
          <>
            <motion.button
              type="button"
              className="fixed inset-0 bg-black/20 z-40 sm:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="fixed sm:absolute bottom-0 sm:bottom-auto left-0 sm:left-auto right-0 sm:right-0 sm:top-[calc(100%+8px)] z-50 w-full sm:w-[270px] p-5 sm:p-4 bg-surface-solid border-t sm:border border-border rounded-t-2xl sm:rounded-2xl shadow-lg"
            >
              <div className="w-10 h-1 bg-border rounded-full mx-auto mb-4 sm:hidden" />

              <p className="text-[0.72rem] uppercase tracking-widest text-text-muted font-semibold mb-1.5">Microphone</p>
              <select
                className={selectClass}
                style={chevronStyle}
                value={selectedMic}
                onChange={(event) => onMicChange(event.target.value)}
              >
                <option value="">Default</option>
                {mics.map((mic) => (
                  <option key={mic.deviceId} value={mic.deviceId}>
                    {mic.label || "Microphone"}
                  </option>
                ))}
              </select>

              <div className="h-px bg-border my-3" />

              <p className="text-[0.72rem] uppercase tracking-widest text-text-muted font-semibold mb-1.5">Speaker</p>
              <select
                className={selectClass}
                style={chevronStyle}
                value={selectedSpeaker}
                onChange={(event) => onSpeakerChange(event.target.value)}
              >
                <option value="">Default</option>
                {speakers.map((speaker) => (
                  <option key={speaker.deviceId} value={speaker.deviceId}>
                    {speaker.label || "Speaker"}
                  </option>
                ))}
              </select>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

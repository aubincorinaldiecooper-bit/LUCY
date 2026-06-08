"use client";

import { motion } from "framer-motion";
import type { HTMLMotionProps } from "framer-motion";
import type { ReactNode } from "react";
import { clsx } from "clsx";

type ButtonVariant = "primary" | "ghost" | "outline" | "danger";

type ButtonProps = HTMLMotionProps<"button"> & {
  children: ReactNode;
  variant?: ButtonVariant;
};

const variants: Record<ButtonVariant, string> = {
  primary: "bg-[#D4A373] text-white hover:bg-[#C49262] shadow-lg shadow-[#D4A373]/20",
  ghost: "bg-transparent text-[#86868B] hover:text-[#1C1C1E] hover:bg-[#F5F5F7]",
  outline: "border border-[#E5E5E5] text-[#1C1C1E] hover:bg-[#F5F5F7] bg-white",
  danger: "border border-[#FF3B30]/20 text-[#FF3B30] hover:border-[#FF3B30] hover:bg-[#FF3B30]/5 bg-white",
};

export function Button({ children, variant = "primary", className, disabled, ...props }: ButtonProps) {
  return (
    <motion.button
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      className={clsx(
        "relative inline-flex items-center justify-center gap-2 rounded-full px-6 py-2.5 text-sm font-medium transition-all duration-300 ease-out disabled:cursor-not-allowed disabled:opacity-50",
        variants[variant],
        className,
      )}
      disabled={disabled}
      {...props}
    >
      {children}
    </motion.button>
  );
}

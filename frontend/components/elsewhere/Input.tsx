"use client";

import type { InputHTMLAttributes } from "react";
import { clsx } from "clsx";

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={clsx(
        "h-[60px] w-full rounded-xl border border-[#E5E5E5] bg-[#F9F9F9] px-5 py-4 text-base text-[#1C1C1E] placeholder:text-[#A1A1A6] transition-all focus:border-[#B86B4D] focus:outline-none focus:ring-2 focus:ring-[#B86B4D]/20",
        className,
      )}
      {...props}
    />
  );
}

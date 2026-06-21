"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { useSession } from "@/lib/auth-client";
import { Button } from "./Button";

type Status = "idle" | "sending" | "sent" | "error";

// End-of-session feedback. Requires sign-in: signed-out users see a prompt to
// sign in; signed-in users get a free-text field. Submissions are emailed to the
// team via /api/feedback (AgentMail).
export function FeedbackForm() {
  const { data: session, isPending } = useSession();
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  if (isPending) {
    return <p className="text-center text-sm text-[#86868B]">Loading…</p>;
  }

  if (!session?.user) {
    return (
      <div className="flex flex-col items-center gap-3 text-center">
        <p className="text-sm text-[#86868B]">Sign in to share what you want Arche to do.</p>
        <Link
          href="/sign-in"
          className="rounded-xl bg-[#B86B4D] px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#A55D42]"
        >
          Sign in to leave feedback
        </Link>
      </div>
    );
  }

  if (status === "sent") {
    return (
      <div className="rounded-xl bg-[#34C759]/10 px-4 py-3 text-center">
        <p className="text-xs font-medium text-[#34C759]">Thanks — Arche will write back to you by email shortly.</p>
      </div>
    );
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed || status === "sending") return;
    setStatus("sending");
    setError(null);

    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });
      if (!response.ok) {
        const data = (await response.json().catch(() => null)) as { error?: string } | null;
        setError(data?.error ?? "Couldn't send your feedback. Please try again.");
        setStatus("error");
        return;
      }
      setStatus("sent");
      setMessage("");
    } catch {
      setError("Couldn't send your feedback. Please try again.");
      setStatus("error");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-2.5">
      <textarea
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        required
        maxLength={5000}
        rows={4}
        placeholder="What do you want Arche to do? Or how can we make Arche better?"
        disabled={status === "sending"}
        className="w-full resize-none rounded-xl border border-black/[0.12] bg-white px-4 py-3 text-sm text-[#1C1C1E] outline-none transition-all placeholder:text-[#A1A1A6] focus:border-[#B86B4D] focus:ring-4 focus:ring-[#B86B4D]/10 disabled:cursor-not-allowed disabled:opacity-70"
      />
      <Button
        type="submit"
        variant="primary"
        className="w-full !rounded-xl !py-2.5"
        disabled={status === "sending" || message.trim().length === 0}
      >
        {status === "sending" ? "Sending…" : "Send feedback"}
      </Button>
      {error ? (
        <p role="alert" className="text-center text-xs text-[#FF3B30]">
          {error}
        </p>
      ) : null}
    </form>
  );
}

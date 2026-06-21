"use client";

import { useState } from "react";
import { authClient } from "@/lib/auth-client";

type Status = "idle" | "sending" | "sent" | "error";

/**
 * Minimal email magic-link sign-in. Enter email -> Better Auth emails a one-time
 * sign-in link -> clicking it lands the user on `callbackURL`, signed in.
 */
export default function MagicLinkSignIn({ callbackURL = "/" }: { callbackURL?: string }) {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setStatus("sending");
    setError(null);

    const { error } = await authClient.signIn.magicLink({ email, callbackURL });

    if (error) {
      setStatus("error");
      setError(error.message ?? "Something went wrong. Try again.");
      return;
    }
    setStatus("sent");
  }

  if (status === "sent") {
    return (
      <p className="text-sm text-neutral-300">
        Check your email — we sent a sign-in link to <strong>{email}</strong>.
      </p>
    );
  }

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-3">
      <input
        type="email"
        required
        autoComplete="email"
        value={email}
        onChange={(event) => setEmail(event.target.value)}
        placeholder="you@example.com"
        className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-white outline-none focus:border-neutral-400"
      />
      <button
        type="submit"
        disabled={status === "sending" || email.length === 0}
        className="rounded-md bg-white px-3 py-2 text-sm font-medium text-black disabled:opacity-50"
      >
        {status === "sending" ? "Sending…" : "Send magic link"}
      </button>
      {error ? (
        <p role="alert" className="text-sm text-red-400">
          {error}
        </p>
      ) : null}
    </form>
  );
}

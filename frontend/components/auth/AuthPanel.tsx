"use client";

import { authClient, useSession } from "@/lib/auth-client";
import MagicLinkSignIn from "@/components/auth/MagicLinkSignIn";

/**
 * Session-aware wrapper: shows the magic-link form when signed out, and the
 * signed-in email + a sign-out button when signed in. Lets you QA the full loop
 * (request link -> click -> land here signed in).
 */
export default function AuthPanel({ callbackURL = "/" }: { callbackURL?: string }) {
  const { data: session, isPending } = useSession();

  if (isPending) {
    return <p className="text-sm text-neutral-400">Loading…</p>;
  }

  if (session?.user) {
    return (
      <div className="flex flex-col gap-3">
        <p className="text-sm text-neutral-300">
          Signed in as <strong>{session.user.email}</strong>
        </p>
        <button
          type="button"
          onClick={() => authClient.signOut()}
          className="rounded-md border border-neutral-700 px-3 py-2 text-sm text-white hover:border-neutral-400"
        >
          Sign out
        </button>
      </div>
    );
  }

  return <MagicLinkSignIn callbackURL={callbackURL} />;
}

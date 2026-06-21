"use client";

import Link from "next/link";
import { authClient, useSession } from "@/lib/auth-client";

// Small account affordance for the app's light-themed pages. Renders nothing
// while the session is resolving (avoids a flash), a "Sign in" link when signed
// out, and the email + a "Sign out" button when signed in.
export default function AccountLink() {
  const { data: session, isPending } = useSession();

  if (isPending) return null;

  if (session?.user) {
    return (
      <div className="flex items-center gap-3 text-sm">
        <span className="hidden text-[#6B6B70] sm:inline">{session.user.email}</span>
        <button
          type="button"
          onClick={() => authClient.signOut()}
          className="rounded-full border border-[#1C1C1E]/15 px-3 py-1 text-[#1C1C1E] transition-colors hover:border-[#1C1C1E]/40"
        >
          Sign out
        </button>
      </div>
    );
  }

  return (
    <Link
      href="/sign-in"
      className="rounded-full border border-[#1C1C1E]/15 px-3 py-1 text-sm text-[#1C1C1E] transition-colors hover:border-[#1C1C1E]/40"
    >
      Sign in
    </Link>
  );
}

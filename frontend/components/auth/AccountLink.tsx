"use client";

import Link from "next/link";
import { authClient, useSession } from "@/lib/auth-client";

type Variant = "light" | "dark";

const STYLES: Record<Variant, { pill: string; email: string }> = {
  light: {
    pill: "border border-[#1C1C1E]/15 text-[#1C1C1E] hover:border-[#1C1C1E]/40",
    email: "text-[#6B6B70]",
  },
  dark: {
    pill: "border border-white/10 bg-white/5 text-white/80 backdrop-blur-sm hover:bg-white/10 hover:text-white",
    email: "text-white/60",
  },
};

// Account affordance used in the app navs. Renders nothing while the session
// resolves (avoids a flash), a "Sign in" link when signed out, and the email +
// a "Sign out" button when signed in. `variant` themes it for the dark landing
// hero vs. the light in-app pages.
export default function AccountLink({ variant = "light" }: { variant?: Variant }) {
  const { data: session, isPending } = useSession();
  const styles = STYLES[variant];

  if (isPending) return null;

  if (session?.user) {
    return (
      <div className="flex items-center gap-3 text-xs font-light">
        <span className={`hidden sm:inline ${styles.email}`}>{session.user.email}</span>
        <button
          type="button"
          onClick={() => authClient.signOut()}
          className={`rounded-full px-4 py-1.5 transition-colors ${styles.pill}`}
        >
          Sign out
        </button>
      </div>
    );
  }

  return (
    <Link href="/sign-in" className={`rounded-full px-4 py-1.5 text-xs font-light transition-colors ${styles.pill}`}>
      Sign in
    </Link>
  );
}

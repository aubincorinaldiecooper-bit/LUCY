"use client";

import { createAuthClient } from "better-auth/react";
import { magicLinkClient } from "better-auth/client/plugins";

// Browser-side client. baseURL defaults to the current origin; set
// NEXT_PUBLIC_BETTER_AUTH_URL only if the auth API is served from another host.
export const authClient = createAuthClient({
  baseURL: process.env.NEXT_PUBLIC_BETTER_AUTH_URL,
  plugins: [magicLinkClient()],
});

export const { signIn, signOut, useSession } = authClient;

/**
 * Sign out of every device, not just this browser.
 *
 * `signOut()` alone revokes only the current session, leaving other devices
 * logged in. We first revoke ALL of this user's sessions (server-side, in
 * Postgres) so every other device is signed out on its next request, then clear
 * this browser's cookie. Revoking is best-effort: if it fails we still sign out
 * locally so the user isn't stuck signed in here.
 */
export async function signOutEverywhere(): Promise<void> {
  try {
    await authClient.revokeSessions();
  } catch {
    // best-effort: fall through to clearing the local session below
  }
  await authClient.signOut();
}

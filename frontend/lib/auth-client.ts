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

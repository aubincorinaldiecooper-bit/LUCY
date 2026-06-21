import { auth } from "@/lib/auth";
import { toNextJsHandler } from "better-auth/next-js";

// Mounts every Better Auth endpoint under /api/auth/* (sign-in, verify, session,
// sign-out, etc.). The magic-link flow uses /api/auth/sign-in/magic-link and the
// link's verify callback.
export const { POST, GET } = toNextJsHandler(auth);

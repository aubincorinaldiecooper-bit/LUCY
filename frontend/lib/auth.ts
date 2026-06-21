import { betterAuth } from "better-auth";
import { magicLink } from "better-auth/plugins";
import { Pool } from "pg";
import { createHash } from "crypto";
import { sendAgentMailEmail } from "@/lib/agentmail";
import { buildMagicLinkEmail } from "@/lib/magic-link-email";

// Short, non-reversible email fingerprint for logs (never log the raw address).
function hashEmail(email: string): string {
  return createHash("sha256").update(email.trim().toLowerCase()).digest("hex").slice(0, 12);
}

// Parse a magic-link URL into safe-to-log parts: the origin only (never the
// token), and the callbackURL query param.
function safeUrlParts(url: string): { origin: string; callbackURL: string } {
  try {
    const parsed = new URL(url);
    return { origin: parsed.origin, callbackURL: parsed.searchParams.get("callbackURL") ?? "none" };
  } catch {
    return { origin: "unparseable", callbackURL: "unknown" };
  }
}

// Better Auth lives in the Next.js app and stores its tables (user, session,
// account, verification) in the SAME Postgres your agent uses. It reads
// BETTER_AUTH_SECRET and BETTER_AUTH_URL from the environment automatically.
//
// We don't throw at import time on a missing DATABASE_URL: that would break
// `next build` in setups that build without runtime secrets. The pg Pool is
// lazy (it doesn't connect until the first query), so a missing connection
// string only surfaces at request time. Warn instead so it's visible in logs.
if (!process.env.DATABASE_URL) {
  console.warn(
    "[better-auth] DATABASE_URL is not set. Point it at your Postgres service " +
      "before serving auth requests (on Railway, ${{Postgres.DATABASE_URL}})."
  );
}

export const auth = betterAuth({
  database: new Pool({ connectionString: process.env.DATABASE_URL }),
  plugins: [
    magicLink({
      // Token lifetime in seconds (Better Auth default is 300 = 5 minutes).
      expiresIn: 300,
      // Called when a user requests a sign-in link. `url` already contains the
      // one-time token and the callbackURL; the user clicks it and is signed in.
      sendMagicLink: async ({ email, url }) => {
        const emailHash = hashEmail(email);
        const { origin, callbackURL } = safeUrlParts(url);
        // Safe, structured observability — no raw email, no token.
        console.log(
          `[better-auth] magic_link_requested email_hash=${emailHash} ` +
            `better_auth_url=${process.env.BETTER_AUTH_URL ?? "unset"} ` +
            `url_origin=${origin} callback_url=${callbackURL}`
        );
        if (process.env.NODE_ENV !== "production") {
          // Dev only: full link (contains the token) for local testing. Never in prod.
          console.log(`[better-auth] magic link for ${email}: ${url}`);
        }
        try {
          await sendMagicLinkEmail(email, url);
          console.log(`[better-auth] magic_link_send status=success email_hash=${emailHash}`);
        } catch (error) {
          console.error(
            `[better-auth] magic_link_send status=failure email_hash=${emailHash} ` +
              `error_type=${error instanceof Error ? error.name : "unknown"} ` +
              `error_message=${error instanceof Error ? error.message : String(error)}`
          );
          throw error;
        }
      },
    }),
  ],
});

/**
 * Sends the magic-link email via AgentMail (the same inbox the agent uses).
 * Requires AGENTMAIL_API_KEY / AGENTMAIL_INBOX_ID / AGENTMAIL_FROM_EMAIL on the
 * frontend service. The 5-minute expiry mirrors the plugin's `expiresIn`.
 */
async function sendMagicLinkEmail(email: string, url: string): Promise<void> {
  const { subject, text, html } = buildMagicLinkEmail(url);
  await sendAgentMailEmail({ to: email, subject, text, html });
}

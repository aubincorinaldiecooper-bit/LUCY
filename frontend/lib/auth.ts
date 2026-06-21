import { betterAuth } from "better-auth";
import { magicLink } from "better-auth/plugins";
import { Pool } from "pg";
import { sendAgentMailEmail } from "@/lib/agentmail";

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
        if (process.env.NODE_ENV !== "production") {
          // Dev convenience: surface the link in server logs so you can test
          // without an email provider wired up yet.
          console.log(`[better-auth] magic link for ${email}: ${url}`);
        }
        await sendMagicLinkEmail(email, url);
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
  const subject = "Your Lucy sign-in link";
  const text =
    `Click to sign in to Lucy:\n\n${url}\n\n` +
    "This link expires in 5 minutes. If you didn't request it, you can ignore this email.";
  const html =
    `<p>Click to sign in to Lucy:</p>` +
    `<p><a href="${url}">Sign in to Lucy</a></p>` +
    `<p style="color:#666;font-size:13px">This link expires in 5 minutes. ` +
    `If you didn't request it, you can ignore this email.</p>`;

  await sendAgentMailEmail({ to: email, subject, text, html });
}

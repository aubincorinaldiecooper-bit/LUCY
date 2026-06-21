import { betterAuth } from "better-auth";
import { magicLink } from "better-auth/plugins";
import { Pool } from "pg";

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
 * TODO: wire this to your real email provider. The cheapest options reuse infra
 * you already have:
 *   - Resend:  await resend.emails.send({ from, to: email, subject: "Sign in to Lucy",
 *                html: `<a href="${url}">Click to sign in</a>` })
 *   - AgentMail / your backend: POST an email job to your existing email service.
 *   - SMTP (nodemailer): transporter.sendMail({ to: email, html: ... })
 *
 * In dev the link is logged above, so this can stay a no-op locally. In
 * production it throws until implemented, so a missing provider fails loudly
 * instead of silently "succeeding" without delivering the link.
 */
async function sendMagicLinkEmail(email: string, url: string): Promise<void> {
  if (process.env.NODE_ENV === "production") {
    throw new Error(
      "sendMagicLinkEmail is not implemented. Wire your email provider in " +
        "frontend/lib/auth.ts before deploying magic-link sign-in."
    );
  }
  // No-op in non-production: the link was already logged for manual testing.
  void email;
  void url;
}

// Minimal AgentMail outbound sender, mirroring the Python agentmail_client.py
// send_email() helper so the frontend can send transactional email (e.g. the
// magic-link) from the same configured inbox. Server-only: it reads the secret
// AGENTMAIL_API_KEY, so never import this into a client component.
import "server-only";

const AGENTMAIL_BASE_URL = (process.env.AGENTMAIL_BASE_URL ?? "https://api.agentmail.to/v0").replace(/\/+$/, "");
const TIMEOUT_MS = Number.parseInt(process.env.AGENTMAIL_TIMEOUT_SECONDS ?? "10", 10) * 1000;

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

export async function sendAgentMailEmail(params: {
  to: string | string[];
  subject: string;
  text?: string;
  html?: string;
}): Promise<void> {
  const { to, subject, text, html } = params;
  if (!text && !html) throw new Error("sendAgentMailEmail requires text or html");

  const apiKey = requiredEnv("AGENTMAIL_API_KEY");
  const inboxId = encodeURIComponent(requiredEnv("AGENTMAIL_INBOX_ID"));
  const fromEmail = requiredEnv("AGENTMAIL_FROM_EMAIL");

  const payload: Record<string, unknown> = { to, subject, reply_to: fromEmail };
  if (text) payload.text = text;
  if (html) payload.html = html;

  const response = await fetch(`${AGENTMAIL_BASE_URL}/inboxes/${inboxId}/messages/send`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "User-Agent": "truthful-abundance/agentmail-frontend",
    },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(Number.isFinite(TIMEOUT_MS) && TIMEOUT_MS > 0 ? TIMEOUT_MS : 10000),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`AgentMail send failed (${response.status}): ${body.slice(0, 500)}`);
  }
}

import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { sendAgentMailEmail } from "@/lib/agentmail";

const MAX_FEEDBACK_CHARS = 5000;

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Collect end-of-session feedback. Sign-in is REQUIRED — verified here via the
// Better Auth session (the UI gates too, this is the server-side enforcement).
// Each submission is emailed to the team inbox via AgentMail.
export async function POST(request: Request) {
  const session = await auth.api.getSession({ headers: request.headers });
  if (!session?.user) {
    return NextResponse.json({ error: "Please sign in to leave feedback." }, { status: 401 });
  }

  let body: { message?: string } = {};
  try {
    body = await request.json();
  } catch {
    body = {};
  }

  const message = (body.message ?? "").trim();
  if (!message) {
    return NextResponse.json({ error: "Feedback can't be empty." }, { status: 400 });
  }
  if (message.length > MAX_FEEDBACK_CHARS) {
    return NextResponse.json({ error: "Feedback is too long." }, { status: 400 });
  }

  const to = process.env.FEEDBACK_TO_EMAIL || process.env.AGENTMAIL_FROM_EMAIL;
  if (!to) {
    return NextResponse.json(
      { error: "Feedback destination is not configured." },
      { status: 500 }
    );
  }

  const userEmail = session.user.email ?? "unknown";
  const subject = `Lucy feedback from ${userEmail}`;
  const text = `From: ${userEmail} (user id: ${session.user.id})\n\n${message}`;
  const html =
    `<p><strong>From:</strong> ${escapeHtml(userEmail)} ` +
    `(user id: ${escapeHtml(session.user.id)})</p>` +
    `<p style="white-space:pre-wrap">${escapeHtml(message)}</p>`;

  try {
    await sendAgentMailEmail({ to, subject, text, html });
  } catch {
    return NextResponse.json(
      { error: "Couldn't send your feedback right now. Please try again." },
      { status: 502 }
    );
  }

  return NextResponse.json({ ok: true });
}

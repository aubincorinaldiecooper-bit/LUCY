import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";

const MAX_FEEDBACK_CHARS = 5000;

// Collect end-of-session feedback. Sign-in is REQUIRED — verified here via the
// Better Auth session. We forward the verified user's email + message to the
// Python backend (server-to-server, shared secret), which generates Arche's
// reply and emails it back to the user. That makes the loop autonomous: the
// agent itself responds to feedback.
export async function POST(request: Request) {
  const session = await auth.api.getSession({ headers: request.headers });
  if (!session?.user?.email) {
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

  const backendUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  const sharedSecret = process.env.SESSION_IDENTITY_SHARED_SECRET;
  if (!backendUrl || !sharedSecret) {
    return NextResponse.json({ error: "Feedback is not configured." }, { status: 500 });
  }

  try {
    const upstream = await fetch(new URL("/api/feedback", backendUrl).toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Auth": sharedSecret },
      body: JSON.stringify({ email: session.user.email, message }),
    });
    if (!upstream.ok) {
      return NextResponse.json(
        { error: "Couldn't send your feedback right now. Please try again." },
        { status: 502 }
      );
    }
  } catch {
    return NextResponse.json(
      { error: "Couldn't send your feedback right now. Please try again." },
      { status: 502 }
    );
  }

  return NextResponse.json({ ok: true });
}

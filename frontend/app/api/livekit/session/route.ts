import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";

// BFF route: the browser calls THIS (same-origin, so the Better Auth cookie is
// sent), we validate the session here, then call the Python backend
// server-to-server with the verified user_id plus a shared secret so the backend
// can trust it. Anonymous callers still get a session (guest scope) — identity
// is additive, never required.
export async function POST(request: Request) {
  const backendUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!backendUrl) {
    return NextResponse.json({ error: "Backend URL not configured" }, { status: 500 });
  }

  let body: { model?: string; client_timezone?: string } = {};
  try {
    body = await request.json();
  } catch {
    body = {};
  }

  // Validate the Better Auth session where the cookie actually lives.
  let userId: string | undefined;
  try {
    const session = await auth.api.getSession({ headers: request.headers });
    userId = session?.user?.id;
  } catch {
    userId = undefined; // treat as anonymous on any verification error
  }

  const sharedSecret = process.env.SESSION_IDENTITY_SHARED_SECRET;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  // Only assert identity to the backend when we have both a verified user and
  // the shared secret to prove this call is from our trusted server.
  if (userId && sharedSecret) {
    headers["X-Internal-Auth"] = sharedSecret;
  }

  const upstream = await fetch(new URL("/api/livekit/session", backendUrl).toString(), {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: body.model,
      client_timezone: body.client_timezone,
      ...(userId && sharedSecret ? { user_id: userId } : {}),
    }),
  });

  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

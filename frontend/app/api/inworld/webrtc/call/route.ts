import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const DEFAULT_API_BASE = "https://lucy-production-c960.up.railway.app";

function getApiBase() {
  return process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_BASE;
}

export async function POST(request: Request) {
  const apiBase = getApiBase();
  const upstreamUrl = new URL("/api/inworld/webrtc/call", apiBase).toString();
  const offerSdp = await request.text();

  console.log(`[inworld-webrtc-call-bff] POST → ${upstreamUrl}`);

  const upstream = await fetch(upstreamUrl, {
    method: "POST",
    headers: {
      Accept: "application/sdp",
      "Content-Type": "application/sdp",
    },
    body: offerSdp,
    cache: "no-store",
  });

  const answerSdp = await upstream.text();

  return new NextResponse(answerSdp, {
    status: upstream.status,
    headers: { "Content-Type": "application/sdp" },
  });
}

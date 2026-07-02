import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const DEFAULT_API_BASE = "https://lucy-production-c960.up.railway.app";

function getApiBase() {
  return process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_BASE;
}

export async function GET() {
  const apiBase = getApiBase();
  const upstreamUrl = new URL("/api/inworld/webrtc/ice", apiBase).toString();

  console.log(`[inworld-webrtc-ice-bff] GET → ${upstreamUrl}`);

  const upstream = await fetch(upstreamUrl, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  const text = await upstream.text();

  return new NextResponse(text, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}

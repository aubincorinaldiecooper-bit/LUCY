import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const DEFAULT_API_BASE = "https://lucy-production-c960.up.railway.app";

function getApiBase() {
  return process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_BASE;
}

export async function POST() {
  const apiBase = getApiBase();
  const upstreamUrl = new URL("/api/inworld/ws-smoke-test", apiBase).toString();
  const token = process.env.INWORLD_WS_SMOKE_TOKEN || "";

  console.log(`[inworld-ws-smoke-test-bff] POST → ${upstreamUrl}`);

  try {
    const upstream = await fetch(upstreamUrl, {
      method: "POST",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token}`,
      },
      cache: "no-store",
    });

    const text = await upstream.text();

    return new NextResponse(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
      },
    });
  } catch (err) {
    console.error(`[inworld-ws-smoke-test-bff] exception: ${String(err)}`);
    return NextResponse.json(
      { error: "ws_smoke_proxy_exception", detail: String(err) },
      { status: 502 },
    );
  }
}

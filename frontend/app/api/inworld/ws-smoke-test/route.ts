import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function POST() {
  if (process.env.NEXT_PUBLIC_ENABLE_INWORLD_WEBRTC_SMOKE !== "true") {
    return NextResponse.json({ error: "ws_smoke_disabled" }, { status: 404 });
  }

  const apiBase = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL;
  if (!apiBase) {
    return NextResponse.json({ error: "backend_api_url_missing" }, { status: 500 });
  }

  const smokeToken = process.env.INWORLD_WS_SMOKE_TOKEN;
  if (!smokeToken) {
    return NextResponse.json({ error: "ws_smoke_token_missing" }, { status: 500 });
  }

  try {
    const upstreamUrl = new URL("/api/inworld/ws-smoke-test", apiBase).toString();
    const upstream = await fetch(upstreamUrl, {
      method: "POST",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${smokeToken}`,
      },
      cache: "no-store",
    });

    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "ws_smoke_proxy_exception",
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 502 },
    );
  }
}

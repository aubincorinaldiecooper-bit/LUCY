import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // No blanket /api/* rewrite. Every /api/* route is served by a local
  // Next.js route handler:
  //   - /api/auth/*          -> Better Auth (app/api/auth/[...all]/route.ts)
  //   - /api/livekit/session -> app/api/livekit/session/route.ts
  //   - /api/feedback        -> app/api/feedback/route.ts
  // The livekit/feedback handlers forward to the Python backend themselves via
  // BACKEND_API_URL / NEXT_PUBLIC_API_URL. A previous `'/api/:path*' ->
  // 'http://localhost:8000/api/:path*'` rewrite proxied EVERYTHING (including
  // /api/auth/*) to port 8000 inside the frontend container, where nothing
  // listens — so Better Auth never ran and every auth call failed with
  // ECONNREFUSED (surfaced as HTTP 500 in the browser).
};

export default nextConfig;

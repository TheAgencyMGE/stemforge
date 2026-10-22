import type { NextConfig } from "next";

// The browser talks to /api on the same origin; Next rewrites it to the FastAPI service.
// STEMFORGE_API_URL is read at build/start time (e.g. http://api:8000 inside Docker Compose).
const apiUrl = process.env.STEMFORGE_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/api/:path*` }];
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "no-referrer" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};

export default nextConfig;

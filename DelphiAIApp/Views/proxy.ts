import { auth } from "@/lib/auth";
import { NextResponse } from "next/server";

const PROTECTED_PREFIXES = ["/bets", "/api/bets"];

function buildCsp(nonce: string): string {
  const apiOrigin = process.env.NEXT_PUBLIC_FASTAPI_URL ?? "";
  return [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    `connect-src 'self'${apiOrigin ? " " + apiOrigin : ""}`,
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "object-src 'none'",
  ].join("; ");
}

export default auth((req) => {
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");
  const csp = buildCsp(nonce);

  const url = req.nextUrl;
  const isProtected = PROTECTED_PREFIXES.some((p) => url.pathname.startsWith(p));

  const requestHeaders = new Headers(req.headers);
  requestHeaders.set("x-nonce", nonce);

  if (!isProtected || req.auth?.user) {
    const response = NextResponse.next({ request: { headers: requestHeaders } });
    response.headers.set("Content-Security-Policy", csp);
    return response;
  }

  // Unauthenticated → redirect HTML, 401 JSON.
  if (url.pathname.startsWith("/api/")) {
    const response = NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    response.headers.set("Content-Security-Policy", csp);
    return response;
  }

  const signin = new URL("/auth/signin", url);
  signin.searchParams.set("callbackUrl", url.pathname);
  const response = NextResponse.redirect(signin);
  response.headers.set("Content-Security-Policy", csp);
  return response;
});

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};

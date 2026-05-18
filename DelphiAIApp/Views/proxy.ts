import { auth } from "@/lib/auth";
import { NextResponse } from "next/server";

const PROTECTED_PREFIXES = ["/bets", "/api/bets"];

export default auth((req) => {
  const url = req.nextUrl;
  const isProtected = PROTECTED_PREFIXES.some((p) => url.pathname.startsWith(p));
  if (!isProtected) return NextResponse.next();
  if (req.auth?.user) return NextResponse.next();

  // Unauthenticated → redirect HTML, 401 JSON.
  if (url.pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const signin = new URL("/auth/signin", url);
  signin.searchParams.set("callbackUrl", url.pathname);
  return NextResponse.redirect(signin);
});

export const config = {
  matcher: ["/bets/:path*", "/api/bets/:path*"],
};

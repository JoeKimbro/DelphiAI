import { NextResponse } from "next/server";
import { z } from "zod";
import bcrypt from "bcryptjs";
import { headers } from "next/headers";
import { query } from "@/lib/db";
import { passwordSchema } from "@/lib/common-passwords";
import { tooManySignups, recordAttempt, clientIp } from "@/lib/rate-limit";

const signupSchema = z.object({
  email: z.string().email().max(255),
  password: passwordSchema,
  name: z.string().min(1).max(100).optional(),
});

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const parsed = signupSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid input", issues: parsed.error.issues },
      { status: 400 }
    );
  }
  const { email, password, name } = parsed.data;

  const ip = clientIp(await headers());
  if (await tooManySignups(ip)) {
    return NextResponse.json(
      { error: "Too many signups from this network. Try again later." },
      { status: 429 }
    );
  }

  const existing = await query<{ id: string }>(
    `SELECT id FROM users WHERE email = $1 LIMIT 1`,
    [email]
  );
  if (existing.length > 0) {
    return NextResponse.json({ error: "Email already registered" }, { status: 409 });
  }

  const hash = await bcrypt.hash(password, 12);
  const inserted = await query<{ id: string; email: string; name: string | null }>(
    `INSERT INTO users (email, name, password_hash)
     VALUES ($1, $2, $3)
     RETURNING id, email, name`,
    [email, name ?? null, hash]
  );

  await recordAttempt("__signup__", ip, true);

  return NextResponse.json({ user: inserted[0] }, { status: 201 });
}

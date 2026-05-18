import { NextResponse } from "next/server";
import { z } from "zod";
import bcrypt from "bcryptjs";
import { query } from "@/lib/db";

const signupSchema = z.object({
  email: z.string().email().max(255),
  password: z.string().min(8).max(128),
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

  return NextResponse.json({ user: inserted[0] }, { status: 201 });
}

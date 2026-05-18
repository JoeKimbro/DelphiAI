import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { query } from "@/lib/db";

const createSchema = z.object({
  event_name: z.string().min(1).max(200),
  event_id: z.string().max(200).optional(),
  fighter_picked: z.string().min(1).max(100),
  opponent: z.string().min(1).max(100),
  bet_type: z.enum(["moneyline", "method", "round", "parlay"]),
  units_wagered: z.number().positive().max(10_000_000),
  american_odds: z.number().int().min(-100_000).max(100_000),
  model_prob: z.number().min(0).max(1).optional(),
  notes: z.string().max(500).optional(),
});

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const rows = await query(
    `SELECT id, event_name, event_id, fighter_picked, opponent, bet_type,
            units_wagered, american_odds, model_prob, result, units_returned,
            notes, placed_at, resolved_at
     FROM user_bets
     WHERE user_id = $1
     ORDER BY placed_at DESC`,
    [session.user.id]
  );
  return NextResponse.json({ bets: rows });
}

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const parsed = createSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid input", issues: parsed.error.issues },
      { status: 400 }
    );
  }
  const b = parsed.data;
  const inserted = await query(
    `INSERT INTO user_bets
       (user_id, event_name, event_id, fighter_picked, opponent, bet_type,
        units_wagered, american_odds, model_prob, notes)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
     RETURNING *`,
    [
      session.user.id,
      b.event_name,
      b.event_id ?? null,
      b.fighter_picked,
      b.opponent,
      b.bet_type,
      b.units_wagered,
      b.american_odds,
      b.model_prob ?? null,
      b.notes ?? null,
    ]
  );
  return NextResponse.json({ bet: inserted[0] }, { status: 201 });
}

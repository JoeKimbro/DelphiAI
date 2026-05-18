import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { query } from "@/lib/db";

const resolveSchema = z.object({
  result: z.enum(["won", "lost", "push"]),
});

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const { id } = await params;
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const parsed = resolveSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid input" }, { status: 400 });
  }
  const { result } = parsed.data;

  const rows = await query<{
    units_wagered: number;
    american_odds: number;
  }>(
    `SELECT units_wagered, american_odds FROM user_bets
     WHERE id = $1 AND user_id = $2`,
    [id, session.user.id]
  );
  if (rows.length === 0) {
    return NextResponse.json({ error: "Bet not found" }, { status: 404 });
  }

  let unitsReturned = 0;
  if (result === "won") {
    const odds = rows[0].american_odds;
    const profit =
      odds >= 0
        ? Number(rows[0].units_wagered) * (odds / 100)
        : Number(rows[0].units_wagered) * (100 / Math.abs(odds));
    unitsReturned = Number(rows[0].units_wagered) + profit;
  } else if (result === "push") {
    unitsReturned = Number(rows[0].units_wagered);
  } else {
    unitsReturned = 0;
  }

  const updated = await query(
    `UPDATE user_bets
       SET result = $1, units_returned = $2, resolved_at = NOW()
     WHERE id = $3 AND user_id = $4
     RETURNING *`,
    [result, unitsReturned, id, session.user.id]
  );
  return NextResponse.json({ bet: updated[0] });
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const { id } = await params;
  await query(`DELETE FROM user_bets WHERE id = $1 AND user_id = $2`, [
    id,
    session.user.id,
  ]);
  return NextResponse.json({ ok: true });
}

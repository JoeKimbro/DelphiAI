import { notFound } from "next/navigation";
import { apiFetch, ApiError } from "@/lib/api";
import type { FighterProfile } from "@/lib/types";
import { EloHistoryChart } from "@/components/charts/EloHistory";
import { shortDate } from "@/lib/format";

async function fetchFighter(slug: string): Promise<FighterProfile | null> {
  try {
    return await apiFetch<FighterProfile>(`/api/fighters/${encodeURIComponent(slug)}`);
  } catch (e) {
    if (e instanceof ApiError) {
      if (e.status === 404) return null;
      console.error(`[fighter] ${slug} → ${e.status} ${e.message}`);
    }
    throw e;
  }
}

const STAT_ROWS: Array<{
  key: keyof FighterProfile["career_stats"];
  label: string;
  fmt?: (v: number) => string;
}> = [
  { key: "slpm", label: "SLpM" },
  { key: "str_acc", label: "Str Acc %" },
  { key: "sapm", label: "SApM" },
  { key: "str_def", label: "Str Def %" },
  { key: "td_avg", label: "TD Avg /15m" },
  { key: "td_acc", label: "TD Acc %" },
  { key: "td_def", label: "TD Def %" },
  { key: "sub_avg", label: "Sub Avg /15m" },
  { key: "avg_fight_duration", label: "Avg Duration (min)" },
  { key: "first_round_finish_rate", label: "R1 Finish %" },
  { key: "decision_rate", label: "Decision %" },
];

export default async function FighterPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const fighter = await fetchFighter(slug);
  if (!fighter) notFound();

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      {/* Hero */}
      <section className="rounded-2xl border border-border bg-surface p-8">
        {fighter.nickname && (
          <span className="text-[10px] font-bold uppercase tracking-widest text-gold">
            &quot;{fighter.nickname}&quot;
          </span>
        )}
        <h1 className="mt-2 text-4xl font-black uppercase tracking-tight text-text md:text-5xl">
          {fighter.name}
        </h1>
        <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-muted">
          {fighter.weight_class && (
            <span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-2">
                Division
              </span>{" "}
              <span className="font-medium text-text">{fighter.weight_class}</span>
            </span>
          )}
          {fighter.stance && (
            <span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-2">
                Stance
              </span>{" "}
              <span className="font-medium text-text">{fighter.stance}</span>
            </span>
          )}
          {fighter.height && (
            <span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-2">
                Height
              </span>{" "}
              <span className="font-medium text-text">{fighter.height}</span>
            </span>
          )}
          {fighter.reach && (
            <span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-2">
                Reach
              </span>{" "}
              <span className="font-medium text-text">{fighter.reach}</span>
            </span>
          )}
          {fighter.place_of_birth && (
            <span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-2">
                From
              </span>{" "}
              <span className="font-medium text-text">{fighter.place_of_birth}</span>
            </span>
          )}
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-4">
          <KpiTile
            label="Record"
            value={`${fighter.record.wins ?? 0}-${fighter.record.losses ?? 0}${
              (fighter.record.draws ?? 0) > 0 ? `-${fighter.record.draws}` : ""
            }`}
          />
          <KpiTile
            label="ELO"
            value={
              fighter.career_stats.elo_rating != null
                ? Math.round(fighter.career_stats.elo_rating).toString()
                : "—"
            }
            accent="gold"
          />
          <KpiTile
            label="Peak ELO"
            value={
              fighter.career_stats.peak_elo != null
                ? Math.round(fighter.career_stats.peak_elo).toString()
                : "—"
            }
          />
          <KpiTile
            label="Days Inactive"
            value={fighter.days_since_last_fight?.toString() ?? "—"}
            accent={(fighter.days_since_last_fight ?? 0) > 365 ? "danger" : "default"}
          />
        </div>
      </section>

      {/* ELO chart */}
      {fighter.elo_history.length > 0 && (
        <section className="mt-6 rounded-xl border border-border bg-surface p-5">
          <h3 className="text-sm font-bold uppercase tracking-widest text-muted">
            ELO History
          </h3>
          <EloHistoryChart points={fighter.elo_history} />
        </section>
      )}

      {/* Stats + Recent fights */}
      <section className="mt-6 grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-border bg-surface p-5">
          <h3 className="text-sm font-bold uppercase tracking-widest text-muted">
            Career Stats
          </h3>
          <table className="mt-4 w-full text-sm">
            <tbody>
              {STAT_ROWS.map((row) => {
                const v = fighter.career_stats[row.key];
                return (
                  <tr key={row.key} className="border-t border-border first:border-t-0">
                    <td className="py-2 text-muted">{row.label}</td>
                    <td className="py-2 text-right font-mono tabular-nums text-text">
                      {v != null ? v.toString() : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="overflow-hidden rounded-xl border border-border bg-surface">
          <h3 className="border-b border-border p-5 text-sm font-bold uppercase tracking-widest text-muted">
            Recent Fights
          </h3>
          {fighter.recent_fights.length === 0 ? (
            <p className="p-5 text-sm text-muted">No fight history available.</p>
          ) : (
            <table className="w-full text-sm">
              <tbody>
                {fighter.recent_fights.map((f, i) => {
                  const won = f.result?.toLowerCase() === "win";
                  return (
                    <tr
                      key={`${f.date}-${i}`}
                      className="border-t border-border first:border-t-0"
                    >
                      <td className="px-5 py-3">
                        <div className="text-text">{f.opponentname || "—"}</div>
                        <div className="text-[11px] text-muted">{f.eventname}</div>
                      </td>
                      <td className="px-5 py-3 text-xs text-muted">
                        {shortDate(f.date)}
                      </td>
                      <td className="px-5 py-3 text-right">
                        <span
                          className={
                            won
                              ? "rounded-md border border-success/40 bg-success/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-success"
                              : "rounded-md border border-red/40 bg-red/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-red"
                          }
                        >
                          {f.result}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-right text-xs text-muted">
                        {f.method} R{f.round ?? "?"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </div>
  );
}

function KpiTile({
  label,
  value,
  accent = "default",
}: {
  label: string;
  value: string;
  accent?: "default" | "gold" | "danger";
}) {
  const accentClass =
    accent === "gold"
      ? "text-gold"
      : accent === "danger"
        ? "text-red"
        : "text-text";
  return (
    <div className="rounded-lg border border-border bg-bg/40 p-4">
      <div className="text-[10px] font-bold uppercase tracking-widest text-muted">
        {label}
      </div>
      <div className={`mt-1 font-mono text-2xl font-black tabular-nums ${accentClass}`}>
        {value}
      </div>
    </div>
  );
}

export const dynamic = "force-dynamic";

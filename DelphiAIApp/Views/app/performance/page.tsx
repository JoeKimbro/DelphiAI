import { apiFetch, ApiError } from "@/lib/api";
import type { PerformanceSummary } from "@/lib/types";
import { StatCard } from "@/components/ui/StatCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { CalibrationCurve } from "@/components/charts/CalibrationCurve";
import { AccuracyByTier } from "@/components/charts/AccuracyByTier";
import { ConfidenceBadge } from "@/components/predictions/ConfidenceBadge";
import { ProbabilityBar } from "@/components/predictions/ProbabilityBar";
import { YearFilter } from "@/components/performance/YearFilter";
import { BarChart3, Crosshair, Target, TrendingUp } from "lucide-react";

async function safeFetch<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch (e) {
    if (e instanceof ApiError) {
      console.error(`[performance] ${path} → ${e.status} ${e.message}`);
    }
    return null;
  }
}

export default async function PerformancePage({
  searchParams,
}: {
  searchParams: Promise<{ year?: string }>;
}) {
  const sp = await searchParams;
  const yearParam = Number.parseInt(sp.year ?? "", 10);
  const year = Number.isFinite(yearParam) ? yearParam : null;
  const summary = await safeFetch<PerformanceSummary>(
    year != null
      ? `/api/performance/summary?year=${year}`
      : "/api/performance/summary"
  );
  const stats = summary?.stats;

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <div className="flex items-end justify-between">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-widest text-gold">
            Model Validation
          </span>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-text">
            Performance
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <YearFilter
            years={summary?.available_years ?? []}
            selected={summary?.selected_year ?? year}
          />
          <span className="text-xs font-medium uppercase tracking-widest text-muted">
            {summary?.prediction_type ?? "live"}
          </span>
        </div>
      </div>

      {!stats ? (
        <div className="mt-6">
          <EmptyState
            title="No resolved predictions yet"
            description="Once predictions are resolved via `python -m ml.update_results`, this dashboard will populate with accuracy, calibration, and ROI metrics."
          />
        </div>
      ) : (
        <>
          {/* KPI Strip */}
          <section className="mt-6 grid gap-4 md:grid-cols-4">
            <StatCard
              label="Overall Accuracy"
              value={`${stats.accuracy.toFixed(1)}%`}
              sublabel={`${stats.correct}/${stats.total} predictions`}
              accent="default"
              icon={<Target className="h-4 w-4" />}
            />
            <StatCard
              label="High-Conf Accuracy"
              value={`${stats.high_conf_accuracy.toFixed(0)}%`}
              sublabel={`${stats.high_conf_correct}/${stats.high_conf_total} 65%+ picks`}
              accent="gold"
              icon={<Crosshair className="h-4 w-4" />}
            />
            {(() => {
              const hc = stats.paper_trading?.high_conf;
              const reportable = hc?.reportable ?? false;
              const roi = reportable ? hc!.roi : null;
              return (
                <StatCard
                  label="Paper ROI · High-Conf"
                  value={
                    roi != null
                      ? `${roi >= 0 ? "+" : ""}${roi.toFixed(1)}%`
                      : "—"
                  }
                  sublabel={
                    roi != null
                      ? "Model-implied odds · flat stake"
                      : `Need ${stats.min_roi_bets ?? 100}+ resolved bets (${hc?.bets ?? 0} so far)`
                  }
                  accent={
                    roi == null ? "default" : roi >= 0 ? "success" : "danger"
                  }
                  icon={<TrendingUp className="h-4 w-4" />}
                />
              );
            })()}
            <StatCard
              label="Events"
              value={stats.num_events}
              sublabel="Tracked events"
              icon={<BarChart3 className="h-4 w-4" />}
            />
          </section>

          {/* Charts */}
          <section className="mt-6 grid gap-4 lg:grid-cols-2">
            <ChartCard
              title="Accuracy by Confidence Tier"
              subtitle="HIGH (≥70%), MED (60–70%), LOW (55–60%), TOSS (50–55%)"
            >
              <AccuracyByTier stats={stats} />
            </ChartCard>

            <ChartCard
              title="Calibration Curve"
              subtitle="Predicted probability vs actual outcome. The gold diagonal is ideal."
            >
              <CalibrationCurve stats={stats} />
            </ChartCard>
          </section>

          {/* Tier detail */}
          <section className="mt-6 grid gap-4 md:grid-cols-4">
            {(["HIGH", "MED", "LOW", "TOSS"] as const).map((tier) => {
              const t = stats.tier_stats[tier];
              if (!t) return null;
              return (
                <div
                  key={tier}
                  className="rounded-xl border border-border bg-surface p-5"
                >
                  <div className="flex items-center justify-between">
                    <ConfidenceBadge confidence={tier} />
                    <span className="font-mono text-xs text-muted">
                      {t.correct}/{t.total}
                    </span>
                  </div>
                  <div className="mt-3 font-mono text-2xl font-black text-text">
                    {t.accuracy.toFixed(0)}%
                  </div>
                  <ProbabilityBar
                    leftPct={t.accuracy / 100}
                    leftLabel=""
                    rightLabel=""
                    className="mt-2"
                  />
                </div>
              );
            })}
          </section>

          {/* Source comparison */}
          {stats.source_stats && Object.keys(stats.source_stats).length > 0 && (
            <section className="mt-6 rounded-xl border border-border bg-surface p-6">
              <h3 className="text-sm font-bold uppercase tracking-widest text-muted">
                Source Comparison
              </h3>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                {Object.entries(stats.source_stats).map(([src, s]) => (
                  <div
                    key={src}
                    className="rounded-lg border border-border bg-bg/40 p-4"
                  >
                    <div className="text-[10px] font-bold uppercase tracking-widest text-muted">
                      {src}
                    </div>
                    <div className="mt-1 flex items-baseline gap-3">
                      <span className="font-mono text-2xl font-black text-text">
                        {s.accuracy.toFixed(1)}%
                      </span>
                      <span className="font-mono text-xs text-muted">
                        {s.correct}/{s.total}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Per-event table */}
          {stats.event_stats && Object.keys(stats.event_stats).length > 0 && (
            <section className="mt-6 overflow-hidden rounded-xl border border-border bg-surface">
              <h3 className="border-b border-border px-6 py-4 text-sm font-bold uppercase tracking-widest text-muted">
                Per-Event History
              </h3>
              <table className="w-full text-sm">
                <thead className="bg-bg/40 text-xs uppercase tracking-widest text-muted">
                  <tr>
                    <th className="px-6 py-3 text-left">Event</th>
                    <th className="px-6 py-3 text-left">Date</th>
                    <th className="px-6 py-3 text-right">Record</th>
                    <th className="px-6 py-3 text-right">Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.event_stats).map(([event, s]) => (
                      <tr
                        key={event}
                        className="border-t border-border hover:bg-bg/40"
                      >
                        <td className="px-6 py-3 font-medium text-text">
                          {event}
                        </td>
                        <td className="px-6 py-3 text-muted">{s.date || "—"}</td>
                        <td className="px-6 py-3 text-right font-mono tabular-nums text-muted">
                          {s.correct}/{s.total}
                        </td>
                        <td
                          className={`px-6 py-3 text-right font-mono font-bold tabular-nums ${
                            s.accuracy >= 65
                              ? "text-gold"
                              : s.accuracy >= 50
                                ? "text-text"
                                : "text-red"
                          }`}
                        >
                          {s.accuracy.toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <h3 className="text-sm font-bold uppercase tracking-widest text-muted">
        {title}
      </h3>
      {subtitle && <p className="mt-1 text-xs text-muted-2">{subtitle}</p>}
      <div className="mt-4">{children}</div>
    </div>
  );
}

export const dynamic = "force-dynamic";

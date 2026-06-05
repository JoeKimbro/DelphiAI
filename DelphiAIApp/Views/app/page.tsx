import Link from "next/link";
import { apiFetch, apiFetchRevalidate, ApiError } from "@/lib/api";
import { formatPercent } from "@/lib/format";
import type {
  EventPredictions,
  EventSummary,
  PerformanceSummary,
} from "@/lib/types";
import { StatCard } from "@/components/ui/StatCard";
import { NumberCounter } from "@/components/ui/NumberCounter";
import { EmptyState } from "@/components/ui/EmptyState";
import { ConfidenceBadge } from "@/components/predictions/ConfidenceBadge";
import { ProbabilityBar } from "@/components/predictions/ProbabilityBar";
import { MainEventPreview } from "@/components/predictions/MainEventPreview";
import { Activity, ArrowRight, Calendar, Target, TrendingUp, Trophy } from "lucide-react";

async function safeFetch<T>(path: string, ttlSeconds?: number): Promise<T | null> {
  try {
    return ttlSeconds != null
      ? await apiFetchRevalidate<T>(path, ttlSeconds)
      : await apiFetch<T>(path);
  } catch (e) {
    if (e instanceof ApiError) {
      console.error(`[dashboard] ${path} → ${e.status} ${e.message}`);
    }
    return null;
  }
}

export default async function DashboardPage() {
  // Kick off both list-style fetches in parallel; both are safe to cache for a
  // couple of minutes because backend changes propagate via the prefetch cron.
  const eventsPromise = safeFetch<{ events: EventSummary[] }>(
    "/api/events/upcoming",
    120
  );
  const summaryPromise = safeFetch<PerformanceSummary>(
    "/api/performance/summary",
    120
  );

  // Need the slug to fetch its predictions, so await upcoming first — then
  // parallel-await summary + predictions so the third call doesn't serialize.
  const eventsResp = await eventsPromise;
  const nextEvent = eventsResp?.events?.[0];
  const predictionsPromise = nextEvent
    ? safeFetch<EventPredictions>(
        `/api/events/${nextEvent.id}/predictions?cached_only=true`
      )
    : Promise.resolve(null);

  const [summary, nextEventPredictions] = await Promise.all([
    summaryPromise,
    predictionsPromise,
  ]);
  const mainEventFight = nextEventPredictions?.fights?.[0] ?? null;
  const stats = summary?.stats;
  const highConf = stats?.high_conf_accuracy ?? null;
  const overall = stats?.accuracy ?? null;
  const highConfROI = stats?.paper_trading?.high_conf?.roi ?? null;
  const totalResolved = stats?.total ?? 0;

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      {/* Hero strip */}
      <section className="relative overflow-hidden rounded-2xl border border-border bg-surface p-8">
        <div className="absolute inset-0 bg-grid opacity-50" />
        <div className="relative flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full border border-red/40 bg-red/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-red">
              <Activity className="h-3 w-3" />
              Live Model
            </span>
            <h1 className="mt-3 text-4xl font-black tracking-tight text-text md:text-5xl">
              DELPHI<span className="text-red">AI</span>
            </h1>
            <p className="mt-2 max-w-xl text-sm text-muted md:text-base">
              XGBoost + ELO blended UFC predictions. Calibrated probabilities,
              method-of-victory breakdowns, and live ROI tracking.
            </p>
          </div>

          {nextEvent ? (
            <Link
              href={`/events/${nextEvent.id}`}
              className="group flex flex-col rounded-xl border border-gold/40 bg-bg/60 p-5 transition-colors hover:border-gold"
            >
              <span className="text-[10px] font-bold uppercase tracking-widest text-gold">
                <Calendar className="mr-1 inline h-3 w-3" />
                Next Event
              </span>
              <span className="mt-1 max-w-[260px] truncate text-lg font-black uppercase tracking-tight text-text">
                {nextEvent.name}
              </span>
              <span className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-gold group-hover:translate-x-1 transition-transform">
                View Predictions <ArrowRight className="h-3 w-3" />
              </span>
            </Link>
          ) : (
            <div className="rounded-xl border border-border bg-bg/60 p-5 text-sm text-muted">
              No upcoming event found.
            </div>
          )}
        </div>
      </section>

      {/* Main event preview */}
      {nextEvent && (
        <div className="mt-6">
          <MainEventPreview
            event={nextEvent}
            fight={mainEventFight}
            startsAt={nextEventPredictions?.date ?? null}
          />
        </div>
      )}

      {/* KPIs */}
      <section className="mt-6 grid gap-4 md:grid-cols-4">
        <StatCard
          label="High-Conf Accuracy"
          value={
            highConf != null ? (
              <NumberCounter value={highConf} decimals={0} suffix="%" />
            ) : (
              "—"
            )
          }
          sublabel={`${stats?.high_conf_total ?? 0} resolved bets`}
          accent="gold"
          icon={<Trophy className="h-4 w-4" />}
        />
        <StatCard
          label="Overall Accuracy"
          value={
            overall != null ? (
              <NumberCounter value={overall} decimals={1} suffix="%" />
            ) : (
              "—"
            )
          }
          sublabel={`${totalResolved} predictions`}
          accent="default"
          icon={<Target className="h-4 w-4" />}
        />
        <StatCard
          label="ROI · High-Conf"
          value={
            highConfROI != null ? (
              <NumberCounter
                value={highConfROI}
                decimals={1}
                suffix="%"
                prefix={highConfROI >= 0 ? "+" : ""}
              />
            ) : (
              "—"
            )
          }
          sublabel="Flat-stake paper trading"
          accent={highConfROI != null && highConfROI < 0 ? "danger" : "success"}
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <StatCard
          label="Events Tracked"
          value={<NumberCounter value={stats?.num_events ?? 0} />}
          sublabel="Live + backtest"
          icon={<Calendar className="h-4 w-4" />}
        />
      </section>

      {/* Calibration teaser + upcoming */}
      <section className="mt-6 grid gap-4 lg:grid-cols-3">
        <div className="rounded-xl border border-border bg-surface p-5 lg:col-span-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold uppercase tracking-widest text-muted">
              Calibration
            </h2>
            <Link
              href="/performance"
              className="text-xs font-medium text-gold hover:underline"
            >
              View full report →
            </Link>
          </div>
          <p className="mt-2 text-sm text-muted">
            Predicted probability vs actual outcome across all confidence buckets.
          </p>
          {stats?.bucket_stats ? (
            <div className="mt-4 grid gap-3 sm:grid-cols-5">
              {Object.entries(stats.bucket_stats)
                .sort()
                .map(([bucket, b]) => (
                  <div
                    key={bucket}
                    className="rounded-lg border border-border bg-bg/40 p-3"
                  >
                    <div className="text-[10px] font-bold uppercase tracking-widest text-muted">
                      {bucket}
                    </div>
                    <div className="mt-1 font-mono text-xl font-black text-text">
                      {b.accuracy.toFixed(0)}%
                    </div>
                    <div className="text-[10px] text-muted-2">{b.total} fights</div>
                  </div>
                ))}
            </div>
          ) : (
            <EmptyState
              title="No resolved predictions yet"
              description="Once predictions are resolved via update_results, the calibration curve will appear here."
            />
          )}
        </div>

        <div className="rounded-xl border border-border bg-surface p-5">
          <h2 className="text-sm font-bold uppercase tracking-widest text-muted">
            Upcoming Events
          </h2>
          {eventsResp?.events?.length ? (
            <ul className="mt-3 space-y-2">
              {eventsResp.events.slice(0, 6).map((e) => (
                <li key={e.id}>
                  <Link
                    href={`/events/${e.id}`}
                    className="group flex items-center justify-between rounded-lg border border-border bg-bg/40 p-3 transition-colors hover:border-red/50 hover:bg-red/5"
                  >
                    <span className="truncate pr-3 text-sm font-medium text-text">
                      {e.name}
                    </span>
                    <ArrowRight className="h-3 w-3 shrink-0 text-muted group-hover:text-red" />
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-muted">No upcoming events found.</p>
          )}
        </div>
      </section>

      {/* Tier breakdown */}
      {stats?.tier_stats && (
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
                <div className="mt-3">
                  <div className="font-mono text-2xl font-black text-text">
                    {t.accuracy.toFixed(0)}%
                  </div>
                  <ProbabilityBar
                    leftPct={t.accuracy / 100}
                    leftLabel=""
                    rightLabel=""
                    className="mt-2"
                  />
                </div>
              </div>
            );
          })}
        </section>
      )}
    </div>
  );
}

// Revalidate the dashboard segment every 2 min so the revalidated fetches in
// safeFetch can actually cache — `force-dynamic` would override them to 0.
export const revalidate = 120;

import Link from "next/link";
import { History, ArrowRight, Calendar } from "lucide-react";
import { apiFetchRevalidate, ApiError } from "@/lib/api";
import { EmptyState } from "@/components/ui/EmptyState";
import { shortDate } from "@/lib/format";
import type { PastEventSummary } from "@/lib/types";

async function safeFetch<T>(path: string): Promise<T | null> {
  try {
    // Past events change at most once per fight night; 2 min is conservative.
    return await apiFetchRevalidate<T>(path, 120);
  } catch (e) {
    if (e instanceof ApiError) {
      console.error(`[past-events] ${path} → ${e.status} ${e.message}`);
    }
    return null;
  }
}

function accuracyTone(accuracy: number): string {
  if (accuracy >= 65) return "text-gold";
  if (accuracy >= 50) return "text-text";
  return "text-red";
}

export default async function PastEventsPage() {
  const data = await safeFetch<{ events: PastEventSummary[] }>(
    "/api/events/past?limit=4"
  );
  const events = data?.events ?? [];

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <div className="flex items-end justify-between">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-widest text-red">
            History
          </span>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-text">
            Past Events
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-muted">
            Predictions vs. actual results for the four most recently completed
            cards. As new events resolve, the oldest drops off.
          </p>
        </div>
      </div>

      {events.length === 0 ? (
        <div className="mt-6">
          <EmptyState
            title="No completed events yet"
            description="Once update_results runs against a finished card, it will appear here."
            icon={<History className="h-8 w-8" />}
          />
        </div>
      ) : (
        <ul className="mt-6 grid gap-3 md:grid-cols-2">
          {events.map((e) => (
            <li key={e.id}>
              <Link
                href={`/events/past/${e.id}`}
                className="group flex flex-col gap-3 rounded-xl border border-border bg-surface p-5 transition-colors hover:border-red/50"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-gold">
                      Final
                    </span>
                    <div className="mt-1 truncate text-lg font-black uppercase tracking-tight text-text">
                      {e.name}
                    </div>
                    {e.date && (
                      <div className="mt-1 inline-flex items-center gap-1.5 text-xs text-muted">
                        <Calendar className="h-3.5 w-3.5" />
                        {shortDate(e.date)}
                      </div>
                    )}
                  </div>
                  <ArrowRight className="mt-2 h-4 w-4 shrink-0 text-muted transition-transform group-hover:translate-x-1 group-hover:text-red" />
                </div>

                <div className="flex items-baseline justify-between border-t border-border pt-3 text-sm">
                  <span className="font-bold uppercase tracking-widest text-muted text-[10px]">
                    Record
                  </span>
                  <div className="flex items-baseline gap-3 font-mono tabular-nums">
                    <span className="text-muted">
                      {e.correct}/{e.total}
                    </span>
                    <span
                      className={`text-base font-bold ${accuracyTone(e.accuracy)}`}
                    >
                      {e.accuracy.toFixed(0)}%
                    </span>
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export const revalidate = 120;

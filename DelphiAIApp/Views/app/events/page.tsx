import Link from "next/link";
import { Calendar, ArrowRight } from "lucide-react";
import { apiFetchRevalidate, ApiError } from "@/lib/api";
import { EmptyState } from "@/components/ui/EmptyState";
import type { EventSummary } from "@/lib/types";

async function safeFetch<T>(path: string): Promise<T | null> {
  try {
    // Cache for 2 minutes — matches the backend's _UPCOMING_TTL_SEC so the UI
    // doesn't refetch faster than the upstream source ever updates.
    return await apiFetchRevalidate<T>(path, 120);
  } catch (e) {
    if (e instanceof ApiError) {
      console.error(`[events] ${path} → ${e.status} ${e.message}`);
    }
    return null;
  }
}

export default async function EventsPage() {
  const data = await safeFetch<{ events: EventSummary[] }>("/api/events/upcoming");
  const events = data?.events ?? [];

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <div className="flex items-end justify-between">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-widest text-red">
            Schedule
          </span>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-text">
            Upcoming Events
          </h1>
        </div>
      </div>

      {/* The prefetch job (ml.prefetch_predictions --top 4) precomputes only the
          next 4 events, which UFC.com lists soonest-first. Those open instantly
          from cache; anything further out is a cache miss that runs a live
          scrape + predict pipeline and takes a few seconds. */}
      <p className="mt-3 text-xs text-muted">
        The next 4 events are preloaded and open instantly. Selecting an event
        further out runs a live prediction, which can take a few seconds.
      </p>

      {events.length === 0 ? (
        <div className="mt-6">
          <EmptyState
            title="No upcoming events"
            description="UFC.com had no upcoming events at the moment, or the scraper hit an error. Check the FastAPI logs."
            icon={<Calendar className="h-8 w-8" />}
          />
        </div>
      ) : (
        <ul className="mt-6 grid gap-3 md:grid-cols-2">
          {events.map((e) => (
            <li key={e.id}>
              <Link
                href={`/events/${e.id}`}
                className="group flex items-center justify-between rounded-xl border border-border bg-surface p-5 transition-colors hover:border-red/50"
              >
                <div className="min-w-0">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-gold">
                    UFC
                  </span>
                  <div className="mt-1 truncate text-lg font-black uppercase tracking-tight text-text">
                    {e.name}
                  </div>
                </div>
                <ArrowRight className="h-4 w-4 shrink-0 text-muted transition-transform group-hover:translate-x-1 group-hover:text-red" />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// Note: no `force-dynamic` here — that would override our 120s fetch revalidate
// and turn every navigation into a fresh round-trip. With revalidation the page
// still re-renders on each request, but the backend fetch is reused for 2 min.
export const revalidate = 120;

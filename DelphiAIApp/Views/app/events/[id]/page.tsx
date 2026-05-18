import { notFound } from "next/navigation";
import { Calendar, MapPin, Flame } from "lucide-react";
import { apiFetch, ApiError } from "@/lib/api";
import type { EventPredictions } from "@/lib/types";
import { FightCard } from "@/components/predictions/FightCard";
import { StaggerList } from "@/components/ui/StaggerList";
import { shortDate } from "@/lib/format";

async function fetchPredictions(id: string): Promise<EventPredictions | null> {
  try {
    return await apiFetch<EventPredictions>(`/api/events/${id}/predictions`);
  } catch (e) {
    if (e instanceof ApiError) {
      console.error(`[event] ${id} → ${e.status} ${e.message}`);
      if (e.status === 404) return null;
    }
    throw e;
  }
}

export default async function EventDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const data = await fetchPredictions(id);
  if (!data) notFound();

  const fights = [...data.fights].sort((a, b) => {
    const order = { HIGH: 0, MED: 1, LOW: 2, TOSS: 3 } as const;
    return order[a.confidence] - order[b.confidence];
  });

  const titleCount = fights.filter((f) => f.is_title).length;
  const highConfCount = fights.filter((f) => f.confidence === "HIGH").length;

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <section className="rounded-2xl border border-border bg-surface p-8">
        <span className="text-[10px] font-bold uppercase tracking-widest text-red">
          {data.source === "cache" ? "Cached Predictions" : "Live Predictions"}
        </span>
        <h1 className="mt-2 text-3xl font-black uppercase tracking-tight text-text md:text-4xl">
          {data.name}
        </h1>
        <div className="mt-3 flex flex-wrap items-center gap-4 text-sm text-muted">
          {data.date && (
            <span className="inline-flex items-center gap-1.5">
              <Calendar className="h-4 w-4" />
              {shortDate(data.date)}
            </span>
          )}
          <span className="inline-flex items-center gap-1.5">
            <Flame className="h-4 w-4 text-red" />
            {fights.length} fights
          </span>
          {highConfCount > 0 && (
            <span className="inline-flex items-center gap-1.5 text-gold">
              <MapPin className="h-4 w-4" />
              {highConfCount} high-confidence picks
            </span>
          )}
          {titleCount > 0 && (
            <span className="inline-flex items-center gap-1.5 text-gold">
              Championship × {titleCount}
            </span>
          )}
        </div>
      </section>

      <section className="mt-6">
        <StaggerList className="space-y-3">
          {fights.map((f, i) => (
            <FightCard
              key={`${f.fighter1}-${f.fighter2}-${i}`}
              fight={f}
            />
          ))}
        </StaggerList>
      </section>
    </div>
  );
}

export const dynamic = "force-dynamic";

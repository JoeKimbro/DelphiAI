import { notFound } from "next/navigation";
import { Calendar, Flame, Check, X } from "lucide-react";
import { apiFetch, ApiError } from "@/lib/api";
import type { EventPredictions } from "@/lib/types";
import { FightCard } from "@/components/predictions/FightCard";
import { StaggerList } from "@/components/ui/StaggerList";
import { shortDate } from "@/lib/format";

async function fetchPastEvent(id: string): Promise<EventPredictions | null> {
  try {
    return await apiFetch<EventPredictions>(`/api/events/past/${id}`);
  } catch (e) {
    if (e instanceof ApiError) {
      console.error(`[past-event] ${id} → ${e.status} ${e.message}`);
      if (e.status === 404) return null;
    }
    throw e;
  }
}

export default async function PastEventDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const data = await fetchPastEvent(id);
  if (!data) notFound();

  const fights = [...data.fights].sort((a, b) => {
    const order = { HIGH: 0, MED: 1, LOW: 2, TOSS: 3 } as const;
    return order[a.confidence] - order[b.confidence];
  });

  const resolved = fights.filter((f) => f.was_correct != null);
  const correct = resolved.filter((f) => f.was_correct === true).length;
  const total = resolved.length;
  const accuracy = total ? (correct / total) * 100 : 0;
  const titleCount = fights.filter((f) => f.is_title).length;

  const accuracyTone =
    accuracy >= 65 ? "text-gold" : accuracy >= 50 ? "text-text" : "text-red";

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <section className="rounded-2xl border border-border bg-surface p-8">
        <span className="text-[10px] font-bold uppercase tracking-widest text-gold">
          Results Final
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
          {titleCount > 0 && (
            <span className="inline-flex items-center gap-1.5 text-gold">
              Championship × {titleCount}
            </span>
          )}
          {total > 0 && (
            <span className="inline-flex items-center gap-1.5">
              <Check className="h-4 w-4 text-gold" />
              {correct}
            </span>
          )}
          {total > 0 && total - correct > 0 && (
            <span className="inline-flex items-center gap-1.5">
              <X className="h-4 w-4 text-red" />
              {total - correct}
            </span>
          )}
          {total > 0 && (
            <span className={`font-mono font-bold tabular-nums ${accuracyTone}`}>
              {accuracy.toFixed(0)}%
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
              showActuals
            />
          ))}
        </StaggerList>
      </section>
    </div>
  );
}

export const dynamic = "force-dynamic";

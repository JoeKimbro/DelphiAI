import Link from "next/link";
import { ArrowRight, Calendar, Crown, Flame } from "lucide-react";
import type { EventSummary, FightPrediction } from "@/lib/types";
import { shortDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import { FighterMini } from "@/components/fighters/FighterMini";
import { ProbabilityBar } from "./ProbabilityBar";
import { ConfidenceBadge } from "./ConfidenceBadge";

export function MainEventPreview({
  event,
  fight,
}: {
  event: EventSummary;
  fight: FightPrediction | null;
}) {
  const f1Fav = fight ? fight.f1_prob >= 0.5 : false;
  const isTitle = fight?.is_title ?? false;

  return (
    <section
      className={cn(
        "overflow-hidden rounded-2xl border bg-surface p-6",
        isTitle ? "border-gold/50 glow-gold" : "border-border"
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-red/40 bg-red/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-red">
            <Flame className="h-3 w-3" />
            Main Event
          </span>
          {isTitle && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-gold/40 bg-gold/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-gold">
              <Crown className="h-3 w-3" />
              Title Fight
            </span>
          )}
          <span className="text-sm font-bold uppercase tracking-tight text-text">
            {event.name}
          </span>
          {event.date && (
            <span className="inline-flex items-center gap-1 text-xs text-muted">
              <Calendar className="h-3 w-3" />
              {shortDate(event.date)}
            </span>
          )}
        </div>
        <Link
          href={`/events/${event.id}`}
          className="group inline-flex items-center gap-1.5 rounded-lg border border-gold/40 bg-gold/10 px-4 py-2 text-xs font-bold uppercase tracking-widest text-gold transition-colors hover:border-gold hover:bg-gold/20"
        >
          View Full Card
          <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
        </Link>
      </div>

      {fight ? (
        <div className="mt-6 grid items-center gap-4 md:grid-cols-[1fr_auto_1fr]">
          <FighterMini
            name={fight.fighter1}
            elo={fight.f1_elo}
            adjElo={fight.f1_adj_elo}
            isFavorite={f1Fav}
          />

          <div className="flex min-w-[200px] flex-col items-center gap-2">
            <ProbabilityBar
              leftPct={fight.f1_prob}
              leftLabel={`${Math.round(fight.f1_prob * 100)}%`}
              rightLabel={`${Math.round(fight.f2_prob * 100)}%`}
              className="w-[200px]"
            />
            <div className="flex items-center gap-2">
              <ConfidenceBadge confidence={fight.confidence} />
              {fight.method && (
                <span className="rounded-md border border-border bg-surface-raised px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-muted">
                  {fight.method}
                </span>
              )}
            </div>
          </div>

          <FighterMini
            name={fight.fighter2}
            elo={fight.f2_elo}
            adjElo={fight.f2_adj_elo}
            isFavorite={!f1Fav}
            align="right"
          />
        </div>
      ) : (
        <p className="mt-6 text-sm text-muted">
          Predictions for this card are still being prepared. Open the full card
          to trigger a live run.
        </p>
      )}
    </section>
  );
}

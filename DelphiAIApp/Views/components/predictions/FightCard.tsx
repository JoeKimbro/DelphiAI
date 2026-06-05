"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { ChevronDown, Crown, AlertTriangle, Check, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatAmericanOdds, formatPercent } from "@/lib/format";
import type { FightPrediction } from "@/lib/types";
import { FighterMini } from "@/components/fighters/FighterMini";
import { ProbabilityBar } from "./ProbabilityBar";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { MethodBreakdown } from "./MethodBreakdown";
import { RoundChart } from "./RoundChart";

export function FightCard({
  fight,
  showActuals = false,
}: {
  fight: FightPrediction;
  showActuals?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const f1Fav = fight.f1_prob >= 0.5;
  const f2Fav = !f1Fav;
  const hasActuals =
    showActuals &&
    (fight.actual_winner != null || fight.was_correct != null);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "overflow-hidden rounded-xl border bg-surface",
        fight.is_title
          ? "border-gold/50 glow-gold"
          : "border-border hover:border-border/80"
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="grid w-full grid-cols-[1fr_auto_1fr] items-center gap-4 p-4 text-left"
      >
        <FighterMini
          name={fight.fighter1}
          elo={fight.f1_elo}
          adjElo={fight.f1_adj_elo}
          isFavorite={f1Fav}
        />

        <div className="flex min-w-[180px] flex-col items-center gap-2">
          {fight.is_title && (
            <span className="inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-gold">
              <Crown className="h-3 w-3" />
              Title Fight
            </span>
          )}
          <ProbabilityBar
            leftPct={fight.f1_prob}
            leftLabel={`${Math.round(fight.f1_prob * 100)}%`}
            rightLabel={`${Math.round(fight.f2_prob * 100)}%`}
            className="w-[180px]"
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

        <div className="flex items-center justify-end gap-3">
          <FighterMini
            name={fight.fighter2}
            elo={fight.f2_elo}
            adjElo={fight.f2_adj_elo}
            isFavorite={f2Fav}
            align="right"
          />
          <ChevronDown
            className={cn(
              "h-4 w-4 shrink-0 text-muted transition-transform",
              open && "rotate-180"
            )}
          />
        </div>
      </button>

      {hasActuals && <ActualResultRow fight={fight} />}

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden border-t border-border bg-bg/40"
          >
            <div className="grid gap-6 p-6 md:grid-cols-3">
              <section>
                <div className="mb-3 flex items-center gap-2">
                  <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted">
                    Market Odds
                  </h4>
                  <OddsSourceBadge source={fight.odds_source} />
                </div>
                <div className="space-y-2">
                  <OddsRow
                    name={fight.fighter1}
                    prob={fight.f1_prob}
                    odds={fight.f1_american}
                    edgePct={fight.f1_edge_pct}
                    isFav={f1Fav}
                  />
                  <OddsRow
                    name={fight.fighter2}
                    prob={fight.f2_prob}
                    odds={fight.f2_american}
                    edgePct={fight.f2_edge_pct}
                    isFav={f2Fav}
                  />
                </div>
              </section>

              <section>
                <h4 className="mb-3 text-[10px] font-bold uppercase tracking-widest text-muted">
                  Method of Victory
                </h4>
                <MethodBreakdown
                  ko={fight.ko_pct}
                  sub={fight.sub_pct}
                  dec={fight.dec_pct}
                />
              </section>

              <section>
                <h4 className="mb-3 text-[10px] font-bold uppercase tracking-widest text-muted">
                  Finish Round
                </h4>
                <RoundChart
                  r1={fight.r1_finish}
                  r2={fight.r2_finish}
                  r3={fight.r3_finish}
                  decision={fight.dec_pct}
                />
              </section>
            </div>

            {(fight.notes?.length ||
              (fight.f1_inj ?? 0) > 0 ||
              (fight.f2_inj ?? 0) > 0 ||
              (fight.f1_rust ?? 0) > 0 ||
              (fight.f2_rust ?? 0) > 0) && (
              <div className="flex flex-wrap gap-2 border-t border-border p-4">
                {fight.notes?.map((n) => (
                  <span
                    key={n}
                    className="inline-flex items-center gap-1 rounded-md border border-red/40 bg-red/10 px-2 py-1 text-[11px] font-medium text-red"
                  >
                    <AlertTriangle className="h-3 w-3" />
                    {n}
                  </span>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function ActualResultRow({ fight }: { fight: FightPrediction }) {
  const correct = fight.was_correct === true;
  const incorrect = fight.was_correct === false;
  const Icon = correct ? Check : incorrect ? X : null;
  const tone = correct
    ? "text-gold"
    : incorrect
      ? "text-red"
      : "text-muted";

  const method = fight.actual_method ? fight.actual_method.toUpperCase() : null;
  const winner = fight.actual_winner ?? "—";
  const round = fight.actual_round;

  return (
    <div className="flex items-center justify-between gap-3 border-t border-border bg-bg/40 px-4 py-2.5 text-xs">
      <span className="font-bold uppercase tracking-widest text-muted">
        Actual
      </span>
      <div className={cn("flex items-center gap-2 font-mono tabular-nums", tone)}>
        {Icon && <Icon className="h-3.5 w-3.5" />}
        <span className="font-bold text-text">{winner}</span>
        {method && (
          <span className="text-muted">
            via <span className="text-text">{method}</span>
          </span>
        )}
        {round != null && (
          <span className="text-muted">
            R<span className="text-text">{round}</span>
          </span>
        )}
      </div>
    </div>
  );
}

function OddsRow({
  name,
  prob,
  odds,
  edgePct,
  isFav,
}: {
  name: string;
  prob: number;
  odds: number | null | undefined;
  edgePct?: number;
  isFav: boolean;
}) {
  const edge = edgePct ?? 0;
  const showEdge = odds != null && Math.abs(edge) >= 1;
  const edgeTone =
    edge >= 1
      ? "border-gold/40 bg-gold/10 text-gold"
      : "border-border bg-bg/40 text-muted-2";

  return (
    <div className="flex items-center justify-between text-sm">
      <span
        className={cn(
          "truncate pr-3",
          isFav ? "font-bold text-text" : "text-muted"
        )}
      >
        {name}
      </span>
      <div className="flex items-baseline gap-3 font-mono tabular-nums">
        {showEdge && (
          <span
            className={cn(
              "rounded-md border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-widest",
              edgeTone
            )}
          >
            {edge >= 0 ? "+" : ""}
            {edge.toFixed(1)} EDGE
          </span>
        )}
        <span className="text-xs text-muted">{formatPercent(prob)}</span>
        <span className={cn("text-base", isFav ? "text-gold" : "text-muted-2")}>
          {formatAmericanOdds(odds)}
        </span>
      </div>
    </div>
  );
}

function OddsSourceBadge({
  source,
}: {
  source?: "bestfightodds" | "historical_archive" | "unavailable";
}) {
  if (!source || source === "unavailable") {
    return (
      <span className="rounded-md border border-border bg-bg/40 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-widest text-muted-2">
        Unavailable
      </span>
    );
  }
  const label =
    source === "bestfightodds" ? "BestFightOdds" : "Closing Line";
  return (
    <span className="rounded-md border border-border bg-bg/40 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-widest text-muted">
      {label}
    </span>
  );
}

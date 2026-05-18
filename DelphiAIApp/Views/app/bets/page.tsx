"use client";

import { useMemo } from "react";
import { AnimatePresence } from "framer-motion";
import { useBets, type Bet } from "@/lib/queries";
import { BetForm } from "@/components/bets/BetForm";
import { BetRow } from "@/components/bets/BetRow";
import { StatCard } from "@/components/ui/StatCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Coins, Target, TrendingUp, Wallet } from "lucide-react";

export default function BetsPage() {
  const { data, isLoading, error } = useBets();
  const bets = data?.bets ?? [];

  const summary = useMemo(() => {
    let total = 0;
    let resolved = 0;
    let wins = 0;
    let wagered = 0;
    let returned = 0;
    for (const b of bets) {
      total += 1;
      wagered += Number(b.units_wagered);
      if (b.result) {
        resolved += 1;
        if (b.result === "won") wins += 1;
        if (b.units_returned != null) returned += Number(b.units_returned);
      }
    }
    const profit = returned - bets.filter((b) => b.result).reduce((s, b) => s + Number(b.units_wagered), 0);
    const roi = wagered > 0 && resolved > 0 ? (profit / bets.filter((b) => b.result).reduce((s, b) => s + Number(b.units_wagered), 0)) * 100 : null;
    return {
      total,
      resolved,
      wins,
      wagered,
      profit,
      roi,
      winRate: resolved > 0 ? (wins / resolved) * 100 : null,
    };
  }, [bets]);

  const active = bets.filter((b) => !b.result);
  const history = bets.filter((b) => b.result);

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="flex items-end justify-between">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-widest text-gold">
            Personal Tracker
          </span>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-text">My Bets</h1>
        </div>
        <BetForm />
      </div>

      {/* Summary KPIs */}
      <section className="mt-6 grid gap-4 md:grid-cols-4">
        <StatCard
          label="Bets Placed"
          value={summary.total}
          sublabel={`${summary.resolved} resolved`}
          icon={<Coins className="h-4 w-4" />}
        />
        <StatCard
          label="Win Rate"
          value={summary.winRate != null ? `${summary.winRate.toFixed(0)}%` : "—"}
          sublabel={summary.resolved > 0 ? `${summary.wins}/${summary.resolved}` : "—"}
          accent="gold"
          icon={<Target className="h-4 w-4" />}
        />
        <StatCard
          label="Units Wagered"
          value={summary.wagered.toFixed(2)}
          sublabel="Total stake (resolved + open)"
          icon={<Wallet className="h-4 w-4" />}
        />
        <StatCard
          label="Net Profit"
          value={`${summary.profit >= 0 ? "+" : ""}${summary.profit.toFixed(2)}`}
          sublabel={
            summary.roi != null
              ? `${summary.roi >= 0 ? "+" : ""}${summary.roi.toFixed(1)}% ROI`
              : "—"
          }
          accent={summary.profit >= 0 ? "success" : "danger"}
          icon={<TrendingUp className="h-4 w-4" />}
        />
      </section>

      {/* Active bets */}
      <section className="mt-8">
        <h2 className="mb-3 text-sm font-bold uppercase tracking-widest text-muted">
          Open
        </h2>
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
          </div>
        ) : error ? (
          <div className="rounded-md border border-red/40 bg-red/10 p-4 text-sm text-red">
            {(error as Error).message}
          </div>
        ) : active.length === 0 ? (
          <EmptyState
            title="No open bets"
            description="Log a bet to start tracking your ROI against the model."
          />
        ) : (
          <BetTable bets={active} />
        )}
      </section>

      {/* History */}
      {history.length > 0 && (
        <section className="mt-8">
          <h2 className="mb-3 text-sm font-bold uppercase tracking-widest text-muted">
            History
          </h2>
          <BetTable bets={history} />
        </section>
      )}
    </div>
  );
}

function BetTable({ bets }: { bets: Bet[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border bg-surface">
      <table className="w-full text-sm">
        <thead className="bg-bg/40 text-xs uppercase tracking-widest text-muted">
          <tr>
            <th className="px-4 py-3 text-left">Pick</th>
            <th className="px-4 py-3 text-left">Event</th>
            <th className="px-4 py-3 text-left">Type</th>
            <th className="px-4 py-3 text-right">Units</th>
            <th className="px-4 py-3 text-right">Odds</th>
            <th className="px-4 py-3 text-right">Result</th>
            <th className="px-4 py-3 text-right">P/L</th>
            <th className="px-4 py-3 text-right">Placed</th>
            <th className="px-4 py-3"></th>
          </tr>
        </thead>
        <tbody>
          <AnimatePresence initial={false}>
            {bets.map((b) => (
              <BetRow key={b.id} bet={b} />
            ))}
          </AnimatePresence>
        </tbody>
      </table>
    </div>
  );
}

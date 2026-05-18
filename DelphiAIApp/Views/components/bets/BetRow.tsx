"use client";

import { motion } from "framer-motion";
import { Loader2, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatAmericanOdds, shortDate } from "@/lib/format";
import { useResolveBet, useDeleteBet, type Bet } from "@/lib/queries";

export function BetRow({ bet }: { bet: Bet }) {
  const resolve = useResolveBet();
  const del = useDeleteBet();

  const wagered = Number(bet.units_wagered);
  const returned = bet.units_returned != null ? Number(bet.units_returned) : null;
  const profit = returned != null ? returned - wagered : null;

  return (
    <motion.tr
      layout
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="border-t border-border hover:bg-bg/30"
    >
      <td className="px-4 py-3">
        <div className="font-medium text-text">{bet.fighter_picked}</div>
        <div className="text-[11px] text-muted">vs {bet.opponent}</div>
      </td>
      <td className="px-4 py-3 text-sm text-muted">{bet.event_name}</td>
      <td className="px-4 py-3 text-xs uppercase tracking-widest text-muted">{bet.bet_type}</td>
      <td className="px-4 py-3 text-right font-mono tabular-nums text-text">{wagered.toFixed(2)}</td>
      <td className="px-4 py-3 text-right font-mono tabular-nums text-gold">
        {formatAmericanOdds(bet.american_odds)}
      </td>
      <td className="px-4 py-3 text-right">
        {bet.result ? (
          <span
            className={cn(
              "rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest",
              bet.result === "won" && "border-success/40 bg-success/10 text-success",
              bet.result === "lost" && "border-red/40 bg-red/10 text-red",
              bet.result === "push" && "border-border bg-surface text-muted"
            )}
          >
            {bet.result}
          </span>
        ) : (
          <div className="inline-flex gap-1">
            <ResolveButton id={bet.id} target="won" pending={resolve.isPending} mut={resolve} />
            <ResolveButton id={bet.id} target="lost" pending={resolve.isPending} mut={resolve} />
            <ResolveButton id={bet.id} target="push" pending={resolve.isPending} mut={resolve} />
          </div>
        )}
      </td>
      <td
        className={cn(
          "px-4 py-3 text-right font-mono tabular-nums",
          profit == null ? "text-muted" : profit >= 0 ? "text-success" : "text-red"
        )}
      >
        {profit != null
          ? `${profit >= 0 ? "+" : ""}${profit.toFixed(2)}`
          : "—"}
      </td>
      <td className="px-4 py-3 text-right text-xs text-muted-2">{shortDate(bet.placed_at)}</td>
      <td className="px-4 py-3 text-right">
        <button
          type="button"
          onClick={() => del.mutate(bet.id)}
          className="rounded-md p-1 text-muted-2 hover:bg-red/10 hover:text-red"
          aria-label="Delete bet"
        >
          {del.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
        </button>
      </td>
    </motion.tr>
  );
}

function ResolveButton({
  id,
  target,
  pending,
  mut,
}: {
  id: string;
  target: "won" | "lost" | "push";
  pending: boolean;
  mut: ReturnType<typeof useResolveBet>;
}) {
  return (
    <button
      type="button"
      disabled={pending}
      onClick={() => mut.mutate({ id, result: target })}
      className={cn(
        "rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest transition-colors disabled:opacity-50",
        target === "won" && "border-success/40 text-success hover:bg-success/10",
        target === "lost" && "border-red/40 text-red hover:bg-red/10",
        target === "push" && "border-border text-muted hover:bg-bg/40"
      )}
    >
      {target}
    </button>
  );
}

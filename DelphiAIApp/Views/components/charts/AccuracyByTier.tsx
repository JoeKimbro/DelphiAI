"use client";

import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";
import type { PerformanceStats } from "@/lib/types";

const TIER_COLOR: Record<string, string> = {
  HIGH: "#F59E0B",
  MED: "#DC2626",
  LOW: "#94A3B8",
  TOSS: "#64748B",
};

export function AccuracyByTier({ stats }: { stats: PerformanceStats }) {
  const data = (["HIGH", "MED", "LOW", "TOSS"] as const)
    .filter((t) => stats.tier_stats[t])
    .map((t) => ({
      tier: t,
      accuracy: stats.tier_stats[t].accuracy,
      total: stats.tier_stats[t].total,
      correct: stats.tier_stats[t].correct,
    }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
          <XAxis dataKey="tier" stroke="#94A3B8" tick={{ fontSize: 11, fontWeight: 700 }} />
          <YAxis
            domain={[0, 100]}
            stroke="#94A3B8"
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            contentStyle={{ background: "#1E293B", border: "1px solid #475569", borderRadius: 8, color: "#F8FAFC" }}
            labelStyle={{ color: "#94A3B8" }}
            itemStyle={{ color: "#F8FAFC" }}
            formatter={((_v: unknown, _n: unknown, item: { payload?: { correct: number; total: number; accuracy: number } }) => [
              `${item?.payload?.correct ?? 0}/${item?.payload?.total ?? 0} (${(item?.payload?.accuracy ?? 0).toFixed(1)}%)`,
              "Record",
            ]) as never}
          />
          <Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.tier} fill={TIER_COLOR[d.tier]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EloHistoryPoint } from "@/lib/types";
import { shortDate } from "@/lib/format";

export function EloHistoryChart({ points }: { points: EloHistoryPoint[] }) {
  const data = points.map((p) => ({
    date: p.fightdate,
    elo: Math.round(Number(p.eloafterfight ?? 0)),
    result: p.result,
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="date"
            stroke="#94A3B8"
            tick={{ fontSize: 10 }}
            tickFormatter={shortDate}
          />
          <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: "#1E293B", border: "1px solid #475569", borderRadius: 8, color: "#F8FAFC" }}
            itemStyle={{ color: "#F8FAFC" }}
            labelFormatter={((label: unknown) => shortDate(typeof label === "string" ? label : null)) as never}
          />
          <Line
            type="monotone"
            dataKey="elo"
            stroke="#F59E0B"
            strokeWidth={2}
            dot={{ r: 3, fill: "#F59E0B" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

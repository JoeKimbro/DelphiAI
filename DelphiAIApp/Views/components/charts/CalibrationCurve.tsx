"use client";

import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { PerformanceStats } from "@/lib/types";

const BUCKET_MIDS: Record<string, number> = {
  "50-55%": 52.5,
  "55-60%": 57.5,
  "60-65%": 62.5,
  "65-70%": 67.5,
  "70-100%": 80,
};

export function CalibrationCurve({ stats }: { stats: PerformanceStats }) {
  const data = Object.entries(stats.bucket_stats)
    .map(([k, v]) => ({
      predicted: BUCKET_MIDS[k] ?? 50,
      actual: v.accuracy,
      bucket: k,
      total: v.total,
    }))
    .sort((a, b) => a.predicted - b.predicted);

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="predicted"
            type="number"
            domain={[50, 85]}
            stroke="#94A3B8"
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => `${v}%`}
            label={{
              value: "Predicted probability",
              position: "insideBottom",
              offset: -2,
              fill: "#94A3B8",
              fontSize: 11,
            }}
          />
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
            formatter={((value: unknown, name: unknown) => [
              `${Number(value ?? 0).toFixed(1)}%`,
              name === "actual" ? "Actual" : String(name ?? ""),
            ]) as never}
            labelFormatter={((_l: unknown, payload: Array<{ payload?: { bucket?: string } }> | undefined) =>
              payload?.[0]?.payload?.bucket ?? "") as never}
          />
          <ReferenceLine
            stroke="#F59E0B"
            strokeDasharray="4 4"
            ifOverflow="extendDomain"
            segment={[
              { x: 50, y: 50 },
              { x: 85, y: 85 },
            ]}
          />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#DC2626"
            strokeWidth={3}
            dot={{ r: 5, fill: "#DC2626", stroke: "#F8FAFC", strokeWidth: 1 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

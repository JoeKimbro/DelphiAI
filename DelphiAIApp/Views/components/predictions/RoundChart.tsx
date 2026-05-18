"use client";

import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";

export function RoundChart({
  r1,
  r2,
  r3,
  decision,
}: {
  r1: number | null;
  r2: number | null;
  r3: number | null;
  decision: number | null;
}) {
  const data = [
    { name: "R1", value: (r1 ?? 0) * 100, color: "#DC2626" },
    { name: "R2", value: (r2 ?? 0) * 100, color: "#F59E0B" },
    { name: "R3", value: (r3 ?? 0) * 100, color: "#94A3B8" },
    { name: "Dec", value: (decision ?? 0) * 100, color: "#64748B" },
  ];

  return (
    <div className="h-32 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <XAxis dataKey="name" stroke="#94A3B8" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
          <Tooltip
            cursor={{ fill: "rgba(220, 38, 38, 0.08)" }}
            formatter={((v: unknown) => [`${Number(v ?? 0).toFixed(0)}%`, "Probability"]) as never}
            contentStyle={{ background: "#1E293B", border: "1px solid #475569", borderRadius: 8, color: "#F8FAFC" }}
            labelStyle={{ color: "#94A3B8" }}
            itemStyle={{ color: "#F8FAFC" }}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.name} fill={d.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { formatPercent } from "@/lib/format";

const ROWS: Array<{ key: "ko_pct" | "sub_pct" | "dec_pct"; label: string; tone: string }> = [
  { key: "ko_pct", label: "KO/TKO", tone: "bg-red" },
  { key: "sub_pct", label: "Submission", tone: "bg-gold" },
  { key: "dec_pct", label: "Decision", tone: "bg-muted" },
];

export function MethodBreakdown({
  ko,
  sub,
  dec,
  className,
}: {
  ko: number | null;
  sub: number | null;
  dec: number | null;
  className?: string;
}) {
  const data = { ko_pct: ko ?? 0, sub_pct: sub ?? 0, dec_pct: dec ?? 0 };
  return (
    <div className={cn("space-y-2", className)}>
      {ROWS.map((row) => {
        const value = data[row.key];
        return (
          <div key={row.key} className="flex items-center gap-3">
            <span className="w-20 text-xs font-medium uppercase tracking-wider text-muted">
              {row.label}
            </span>
            <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-raised">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(value ?? 0) * 100}%` }}
                transition={{ type: "spring", stiffness: 110, damping: 20 }}
                className={cn("absolute inset-y-0 left-0", row.tone)}
              />
            </div>
            <span className="w-12 text-right font-mono text-xs tabular-nums text-text">
              {formatPercent(value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

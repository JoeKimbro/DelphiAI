"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function ProbabilityBar({
  leftPct,
  className,
  leftLabel,
  rightLabel,
}: {
  leftPct: number; // 0..1
  leftLabel?: string;
  rightLabel?: string;
  className?: string;
}) {
  const left = Math.max(0, Math.min(1, leftPct));
  const right = 1 - left;
  const leftIsFav = left >= 0.5;

  return (
    <div className={cn("w-full", className)}>
      <div className="relative h-3 w-full overflow-hidden rounded-full bg-surface-raised">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${left * 100}%` }}
          transition={{ type: "spring", stiffness: 110, damping: 20 }}
          className={cn(
            "absolute inset-y-0 left-0",
            leftIsFav ? "bg-red glow-red" : "bg-red/40"
          )}
        />
        <div
          className="absolute inset-y-0 z-10 w-px bg-bg/80"
          style={{ left: "50%" }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-xs tabular-nums">
        <span className={cn(leftIsFav ? "text-red font-bold" : "text-muted")}>
          {leftLabel ?? `${Math.round(left * 100)}%`}
        </span>
        <span className={cn(!leftIsFav ? "text-red font-bold" : "text-muted")}>
          {rightLabel ?? `${Math.round(right * 100)}%`}
        </span>
      </div>
    </div>
  );
}

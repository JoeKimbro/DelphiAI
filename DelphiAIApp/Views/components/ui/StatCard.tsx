import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

export function StatCard({
  label,
  value,
  sublabel,
  accent = "default",
  className,
  icon,
}: {
  label: string;
  value: ReactNode;
  sublabel?: ReactNode;
  accent?: "default" | "red" | "gold" | "success" | "danger";
  className?: string;
  icon?: ReactNode;
}) {
  const accents: Record<string, string> = {
    default: "text-text",
    red: "text-red",
    gold: "text-gold",
    success: "text-success",
    danger: "text-danger",
  };
  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-surface p-5",
        "transition-colors hover:border-border/80",
        className
      )}
    >
      <div className="flex items-start justify-between">
        <span className="text-[10px] font-bold uppercase tracking-widest text-muted">
          {label}
        </span>
        {icon && <span className="text-muted">{icon}</span>}
      </div>
      <div
        className={cn(
          "mt-2 font-mono text-3xl font-black tabular-nums",
          accents[accent]
        )}
      >
        {value}
      </div>
      {sublabel && (
        <div className="mt-1 text-xs text-muted">{sublabel}</div>
      )}
    </div>
  );
}

import Link from "next/link";
import { cn } from "@/lib/utils";
import { formatElo, slugifyFighter } from "@/lib/format";

export function FighterMini({
  name,
  elo,
  adjElo,
  isFavorite,
  align = "left",
}: {
  name: string;
  elo?: number | null;
  adjElo?: number | null;
  isFavorite?: boolean;
  align?: "left" | "right";
}) {
  const delta = elo != null && adjElo != null ? Math.round(adjElo - elo) : 0;
  return (
    <Link
      href={`/fighters/${slugifyFighter(name)}`}
      className={cn(
        "group block",
        align === "right" ? "text-right" : "text-left"
      )}
    >
      <div
        className={cn(
          "text-base font-black uppercase tracking-tight transition-colors group-hover:text-red",
          isFavorite ? "text-text" : "text-muted"
        )}
      >
        {name}
      </div>
      <div
        className={cn(
          "mt-0.5 flex items-baseline gap-2 text-[11px] font-medium uppercase tracking-wider text-muted",
          align === "right" && "justify-end"
        )}
      >
        <span className="font-mono tabular-nums">ELO {formatElo(elo)}</span>
        {delta !== 0 && (
          <span
            className={cn(
              "font-mono tabular-nums",
              delta < 0 ? "text-red" : "text-success"
            )}
          >
            {delta > 0 ? "+" : ""}
            {delta}
          </span>
        )}
      </div>
    </Link>
  );
}

import type { Confidence } from "./types";

export function formatPercent(prob: number | null | undefined, digits = 0): string {
  if (prob == null || Number.isNaN(prob)) return "—";
  return `${(prob * 100).toFixed(digits)}%`;
}

export function formatAmericanOdds(odds: number | null | undefined): string {
  if (odds == null || Number.isNaN(odds)) return "—";
  return odds >= 0 ? `+${odds}` : `${odds}`;
}

export function formatElo(elo: number | null | undefined): string {
  if (elo == null) return "—";
  return Math.round(elo).toString();
}

export function formatEloDelta(delta: number | null | undefined): string {
  if (delta == null) return "";
  const rounded = Math.round(delta);
  if (rounded === 0) return "0";
  return rounded > 0 ? `▲ +${rounded}` : `▼ ${rounded}`;
}

export function confidenceColor(conf: Confidence): {
  bg: string;
  text: string;
  border: string;
} {
  switch (conf) {
    case "HIGH":
      return { bg: "bg-gold/10", text: "text-gold", border: "border-gold/40" };
    case "MED":
      return { bg: "bg-red/10", text: "text-red", border: "border-red/40" };
    case "LOW":
      return { bg: "bg-surface-raised", text: "text-muted", border: "border-border" };
    case "TOSS":
    default:
      return { bg: "bg-surface", text: "text-muted-2", border: "border-border" };
  }
}

export function slugifyFighter(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}

export function shortDate(date: string | null | undefined): string {
  if (!date) return "—";
  try {
    return new Date(date).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return date;
  }
}

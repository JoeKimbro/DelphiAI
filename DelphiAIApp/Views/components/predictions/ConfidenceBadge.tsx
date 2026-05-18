import { cn } from "@/lib/utils";
import { confidenceColor } from "@/lib/format";
import type { Confidence } from "@/lib/types";

export function ConfidenceBadge({
  confidence,
  className,
}: {
  confidence: Confidence;
  className?: string;
}) {
  const { bg, text, border } = confidenceColor(confidence);
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest",
        bg,
        text,
        border,
        className
      )}
    >
      {confidence}
    </span>
  );
}

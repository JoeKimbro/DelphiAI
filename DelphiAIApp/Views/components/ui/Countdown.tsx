"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Clock } from "lucide-react";

type Parts = {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
  expired: boolean;
};

function partsUntil(target: number, now: number): Parts {
  const diff = Math.max(0, target - now);
  const seconds = Math.floor(diff / 1000) % 60;
  const minutes = Math.floor(diff / (1000 * 60)) % 60;
  const hours = Math.floor(diff / (1000 * 60 * 60)) % 24;
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  return { days, hours, minutes, seconds, expired: diff === 0 };
}

const CT_ZONE = "America/Chicago";

const CT_TIME_FMT = new Intl.DateTimeFormat("en-US", {
  timeZone: CT_ZONE,
  weekday: "short",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZoneName: "short",
});

/**
 * Bare countdown wheels — no card chrome. Use inside other components where
 * the surrounding section already provides border/background. The `targetIso`
 * is treated as an absolute instant (it carries an offset from the backend),
 * but the local-time display under the wheels is rendered in CT.
 */
export function CountdownInline({
  targetIso,
  size = "md",
  className,
}: {
  targetIso: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  const target = new Date(targetIso).getTime();
  const [parts, setParts] = useState<Parts | null>(null);

  useEffect(() => {
    if (Number.isNaN(target)) return;
    setParts(partsUntil(target, Date.now()));
    const id = window.setInterval(
      () => setParts(partsUntil(target, Date.now())),
      1000
    );
    return () => window.clearInterval(id);
  }, [target]);

  if (Number.isNaN(target)) return null;

  if (parts?.expired) {
    return (
      <div className={"text-sm font-bold uppercase tracking-widest text-red " + (className ?? "")}>
        Live now
      </div>
    );
  }

  const ctLabel = CT_TIME_FMT.format(new Date(target));

  return (
    <div className={"flex flex-col items-center gap-2 " + (className ?? "")}>
      <div className="flex items-end justify-center gap-1.5">
        {parts && parts.days > 0 && (
          <Segment
            value={parts.days}
            label="DAYS"
            width={parts.days >= 100 ? 3 : 2}
            size={size}
          />
        )}
        <Segment value={parts?.hours ?? 0} label="HRS" size={size} />
        <Colon size={size} />
        <Segment value={parts?.minutes ?? 0} label="MIN" size={size} />
        <Colon size={size} />
        <Segment value={parts?.seconds ?? 0} label="SEC" size={size} />
      </div>
      <span className="font-mono text-[10px] uppercase tracking-widest text-muted-2">
        {ctLabel}
      </span>
    </div>
  );
}

/**
 * Card-style countdown with header chip. Used for hero/full-width contexts.
 */
export function Countdown({
  targetIso,
  label = "Fights begin in",
  className,
}: {
  targetIso: string;
  label?: string;
  className?: string;
}) {
  if (Number.isNaN(new Date(targetIso).getTime())) return null;

  return (
    <div
      className={
        "relative overflow-hidden rounded-2xl border border-border bg-surface p-6 " +
        (className ?? "")
      }
    >
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-30" />
      <div className="relative flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <span className="inline-flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-gold">
          <Clock className="h-3 w-3" />
          {label}
        </span>
        <CountdownInline targetIso={targetIso} />
      </div>
    </div>
  );
}

const sizeMap = {
  sm: { box: "h-10 w-7", text: "text-2xl", colon: "h-10 text-xl pb-4" },
  md: { box: "h-14 w-9", text: "text-3xl", colon: "h-14 text-3xl pb-5" },
  lg: { box: "h-20 w-14", text: "text-5xl", colon: "h-20 text-4xl pb-7" },
} as const;

function Segment({
  value,
  label,
  width = 2,
  size,
}: {
  value: number;
  label: string;
  width?: number;
  size: "sm" | "md" | "lg";
}) {
  const digits = String(value).padStart(width, "0").split("").map(Number);
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="flex gap-1">
        {digits.map((d, i) => (
          <DigitWheel key={`${label}-${i}`} digit={d} size={size} />
        ))}
      </div>
      <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-muted-2">
        {label}
      </span>
    </div>
  );
}

function Colon({ size }: { size: "sm" | "md" | "lg" }) {
  return (
    <div
      className={
        "flex items-center px-0.5 font-mono font-black text-muted-2 " +
        sizeMap[size].colon
      }
    >
      :
    </div>
  );
}

/**
 * Vertical-roll digit. We stack 0-9 in a column and translate the column
 * by `-digit * 10%` so the active value sits in the visible window. A
 * spring transition gives the slot-machine wheel feel.
 */
function DigitWheel({
  digit,
  size,
}: {
  digit: number;
  size: "sm" | "md" | "lg";
}) {
  const { box, text } = sizeMap[size];
  return (
    <div
      className={
        "relative overflow-hidden rounded-md border border-border bg-bg/60 " +
        box
      }
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-1/2 bg-gradient-to-b from-bg/80 to-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-1/2 bg-gradient-to-t from-bg/80 to-transparent" />
      <motion.div
        className="flex flex-col"
        initial={false}
        animate={{ y: `-${digit * 10}%` }}
        transition={{ type: "spring", stiffness: 280, damping: 26 }}
        style={{ height: "1000%" }}
      >
        {Array.from({ length: 10 }, (_, n) => (
          <div
            key={n}
            className={
              "flex items-center justify-center font-mono font-black tabular-nums text-text " +
              box +
              " " +
              text
            }
          >
            {n}
          </div>
        ))}
      </motion.div>
    </div>
  );
}

"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { useTransition } from "react";
import { ChevronDown } from "lucide-react";

export function YearFilter({
  years,
  selected,
}: {
  years: number[];
  selected?: number | null;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [pending, startTransition] = useTransition();

  function onChange(value: string) {
    const params = new URLSearchParams(searchParams.toString());
    if (value === "all") {
      params.delete("year");
    } else {
      params.set("year", value);
    }
    const qs = params.toString();
    startTransition(() => {
      router.push(qs ? `${pathname}?${qs}` : pathname);
    });
  }

  return (
    <label
      className={
        "relative inline-flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-xs font-medium uppercase tracking-widest text-muted " +
        (pending ? "opacity-60" : "")
      }
    >
      <span className="text-muted-2">Year</span>
      <select
        value={selected ?? "all"}
        onChange={(e) => onChange(e.target.value)}
        disabled={pending}
        className="appearance-none bg-transparent pr-5 font-mono font-bold text-text outline-none [&>option]:bg-surface [&>option]:text-text"
      >
        <option value="all">All</option>
        {years.map((y) => (
          <option key={y} value={y}>
            {y}
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2 h-3.5 w-3.5 text-muted" />
    </label>
  );
}

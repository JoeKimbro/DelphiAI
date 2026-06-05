"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

export type Bet = {
  id: string;
  event_name: string;
  event_id: string | null;
  fighter_picked: string;
  opponent: string;
  bet_type: "moneyline" | "method" | "round" | "parlay";
  units_wagered: string | number;
  american_odds: number;
  model_prob: string | number | null;
  result: "won" | "lost" | "push" | null;
  units_returned: string | number | null;
  notes: string | null;
  placed_at: string;
  resolved_at: string | null;
};

const KEYS = {
  bets: ["bets"] as const,
};

/** Error carrying the HTTP status so callers can branch on auth/validation vs transient failures. */
export class HttpError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new HttpError(res.status, body?.error ?? `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

/** Retry transient/5xx/network failures, but never 4xx (auth, validation) — retrying those is pointless. */
function retryNon4xx(failureCount: number, error: unknown): boolean {
  if (error instanceof HttpError && error.status >= 400 && error.status < 500) {
    return false;
  }
  return failureCount < 2;
}

export function useBets(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: KEYS.bets,
    queryFn: () => jsonFetch<{ bets: Bet[] }>("/api/bets"),
    enabled: options?.enabled ?? true,
    retry: retryNon4xx,
  });
}

export function useCreateBet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: Omit<Bet, "id" | "result" | "units_returned" | "placed_at" | "resolved_at">) =>
      jsonFetch<{ bet: Bet }>("/api/bets", {
        method: "POST",
        body: JSON.stringify(input),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEYS.bets }),
  });
}

export function useResolveBet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, result }: { id: string; result: "won" | "lost" | "push" }) =>
      jsonFetch<{ bet: Bet }>(`/api/bets/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ result }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEYS.bets }),
  });
}

export function useDeleteBet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      jsonFetch<{ ok: boolean }>(`/api/bets/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEYS.bets }),
  });
}

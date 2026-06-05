const BASE = process.env.FASTAPI_URL ?? "http://localhost:8000";
// Read on the server only. Never NEXT_PUBLIC_-prefixed so the key can't leak
// to the browser bundle. All apiFetch callers are Server Components / route
// handlers, so `process.env` resolves at request time on the Node runtime.
const API_KEY = process.env.DELPHI_API_KEY ?? "";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function _fetchOrThrow<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
      ...(init.headers ?? {}),
    },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore
    }
    throw new ApiError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  // Default no-store keeps detail pages always-fresh against PredictionTracking.
  return _fetchOrThrow<T>(path, {
    ...init,
    cache: init?.cache ?? "no-store",
  });
}

/**
 * List-view fetch with Next.js revalidation. Use for endpoints whose payloads
 * change at most a few times per hour (upcoming events, past events, performance
 * summary). Repeat navigations within `ttlSeconds` hit the Next.js cache instead
 * of re-querying FastAPI, so back-button + sidebar refreshes stay instant.
 */
export async function apiFetchRevalidate<T>(
  path: string,
  ttlSeconds: number,
  init?: RequestInit
): Promise<T> {
  return _fetchOrThrow<T>(path, {
    ...init,
    next: { revalidate: ttlSeconds },
  });
}

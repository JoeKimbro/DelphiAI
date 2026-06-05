"""
Application security middleware: API-key auth + per-IP sliding-window rate
limiting. Pure stdlib (no extra deps).

Both layers are env-driven so local development is unaffected unless you opt
in:

  - DELPHI_API_KEY  -> if set, every request to /api/* must carry
                       `X-API-Key: <value>` (constant-time compare). Unset =
                       no auth (current behaviour).
  - DELPHI_DISABLE_RATELIMIT  -> any truthy value turns off rate limiting
                       (handy for tests). Default is on.
  - DELPHI_ENABLE_DOCS  -> truthy to expose /docs and /redoc. Default off
                       in production-style deployments.

`/health`, the root, and OPTIONS preflight are always allowed through.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import deque

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

API_KEY = os.environ.get("DELPHI_API_KEY", "").strip() or None
RATE_LIMIT_DISABLED = os.environ.get("DELPHI_DISABLE_RATELIMIT", "").strip().lower() in {
    "1", "true", "yes", "on",
}
DOCS_ENABLED = os.environ.get("DELPHI_ENABLE_DOCS", "").strip().lower() in {
    "1", "true", "yes", "on",
}

# Paths that bypass auth + rate limit (health checks must always succeed).
_PUBLIC_PATHS: frozenset[str] = frozenset({"/", "/health"})

# Per-endpoint rate buckets: (max_requests, window_seconds). Matched by
# (method, path-prefix). First match wins; longest prefix first.
_RATE_BUCKETS: tuple[tuple[str, str, int, int], ...] = (
    # Expensive: trigger scrapers / model runs / bulk DB writes
    ("POST", "/api/results/auto-resolve", 2, 60),
    ("POST", "/api/results/update", 5, 60),
    # Per-event predict can scrape UFC.com + run the model on cache miss
    ("GET",  "/api/events/upcoming", 30, 60),
    # Generous default for read endpoints
    ("*",    "/api/", 120, 60),
)


# ----------------------------------------------------------------------------
# Rate limiter (sliding window, in-memory)
# ----------------------------------------------------------------------------

_buckets_lock = threading.Lock()
# key: (client_ip, bucket_index) -> deque[timestamp]
_buckets: dict[tuple[str, int], deque[float]] = {}


def _match_bucket(method: str, path: str) -> tuple[int, int, int] | None:
    """Return (bucket_index, max_requests, window_sec) for the first matching rule, else None."""
    for i, (m, prefix, max_req, window) in enumerate(_RATE_BUCKETS):
        if (m == "*" or m == method) and path.startswith(prefix):
            return i, max_req, window
    return None


def _client_ip(request: Request) -> str:
    # Cloudflare / Cloudflared injects the real client IP here.
    xff = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request) -> None:
    """Raise 429 if the caller has exceeded the matched bucket. No-op otherwise."""
    if RATE_LIMIT_DISABLED:
        return
    matched = _match_bucket(request.method, request.url.path)
    if matched is None:
        return
    bucket_idx, max_req, window = matched
    ip = _client_ip(request)
    now = time.monotonic()
    key = (ip, bucket_idx)

    with _buckets_lock:
        dq = _buckets.get(key)
        if dq is None:
            dq = deque()
            _buckets[key] = dq
        # Drop timestamps outside the window
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= max_req:
            retry_in = max(1, int(dq[0] + window - now))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({max_req}/{window}s). Retry in ~{retry_in}s.",
                headers={"Retry-After": str(retry_in)},
            )
        dq.append(now)


# ----------------------------------------------------------------------------
# API-key auth
# ----------------------------------------------------------------------------

def _check_api_key(request: Request) -> None:
    if API_KEY is None:
        return  # auth disabled
    presented = request.headers.get("x-api-key", "")
    # Constant-time compare to avoid timing-based key guessing
    if not presented or not hmac.compare_digest(presented, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


# ----------------------------------------------------------------------------
# Combined middleware
# ----------------------------------------------------------------------------

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Order: public-path bypass -> auth -> rate limit -> downstream handler.

    OPTIONS preflight requests are always allowed so browser CORS works.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Always allow CORS preflight + health/root
        if request.method == "OPTIONS" or path in _PUBLIC_PATHS:
            return await call_next(request)

        try:
            _check_api_key(request)
            _check_rate_limit(request)
        except HTTPException as e:
            return JSONResponse(
                content={"detail": e.detail},
                status_code=e.status_code,
                headers=e.headers or {},
            )

        return await call_next(request)


def warn_if_unconfigured() -> None:
    """Print loud warnings on boot when running without the safety nets."""
    if API_KEY is None:
        print(
            "[security] DELPHI_API_KEY not set — API is open. "
            "Set it in .env before exposing publicly.",
            flush=True,
        )
    if RATE_LIMIT_DISABLED:
        print("[security] Rate limiting DISABLED by env.", flush=True)
    if DOCS_ENABLED:
        print("[security] /docs and /redoc are exposed.", flush=True)

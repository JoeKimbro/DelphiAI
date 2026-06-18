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
# Separate, higher-privilege key for expensive/mutating endpoints (model runs,
# scrapes, bulk DB writes). The read key (API_KEY) is deployed broadly — it
# lives in the Next.js server env for every dashboard fetch — so gating the
# costly write paths behind a DIFFERENT secret means a leaked read key still
# can't trigger them. When unset, those paths fall back to the read key (current
# behaviour) so this is non-breaking until an operator opts in.
ADMIN_API_KEY = os.environ.get("DELPHI_ADMIN_API_KEY", "").strip() or None
RATE_LIMIT_DISABLED = os.environ.get("DELPHI_DISABLE_RATELIMIT", "").strip().lower() in {
    "1", "true", "yes", "on",
}
DOCS_ENABLED = os.environ.get("DELPHI_ENABLE_DOCS", "").strip().lower() in {
    "1", "true", "yes", "on",
}
DELPHI_ENV = os.environ.get("DELPHI_ENV", "").strip().lower()

# In production the API must not boot open. Fail fast at import time so a
# misconfigured deploy crashes loudly instead of silently serving unauthed.
if DELPHI_ENV == "production" and API_KEY is None:
    raise RuntimeError(
        "DELPHI_ENV=production but DELPHI_API_KEY is unset — refusing to start "
        "an unauthenticated API."
    )

# Reject request bodies larger than this (DoS guard). 256 KB is generous for
# this API's JSON payloads. Tunable via env for unusual deployments.
MAX_BODY_BYTES = int(os.environ.get("DELPHI_MAX_BODY_BYTES", str(256 * 1024)))

# Paths that bypass auth + rate limit (health checks must always succeed).
_PUBLIC_PATHS: frozenset[str] = frozenset({"/", "/health"})

# Expensive / state-mutating routes that require the admin key (when configured)
# on top of the normal API key. Matched by (method, path-prefix); "*" = any
# method. The frontend never calls these — they're ops/cron only.
_ADMIN_RULES: tuple[tuple[str, str], ...] = (
    ("POST", "/api/results/"),
)

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


def _is_admin_path(method: str, path: str) -> bool:
    return any(
        (m == "*" or m == method) and path.startswith(prefix)
        for m, prefix in _ADMIN_RULES
    )


def _check_admin_key(request: Request) -> None:
    """Require the admin key on expensive/mutating routes when one is set."""
    if ADMIN_API_KEY is None:
        return  # separation not configured — normal key check already applied
    if not _is_admin_path(request.method, request.url.path):
        return
    presented = request.headers.get("x-admin-key", "")
    if not presented or not hmac.compare_digest(presented, ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid admin key for this operation.",
            headers={"WWW-Authenticate": "X-Admin-Key"},
        )


# ----------------------------------------------------------------------------
# Response security headers
# ----------------------------------------------------------------------------

# Static response headers applied to every response. HSTS is safe because the
# edge (Vercel/Railway/Cloudflare) terminates TLS; browsers ignore it on plain
# HTTP localhost, so local dev is unaffected.
_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
}
# Docs (Swagger/ReDoc) need inline scripts + a CDN, so the locked-down CSP is
# skipped for those paths only.
_DOC_PATHS = ("/docs", "/redoc", "/openapi.json")
_API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"


def _apply_security_headers(request: Request, response) -> None:
    for k, v in _SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    if not request.url.path.startswith(_DOC_PATHS):
        response.headers.setdefault("Content-Security-Policy", _API_CSP)


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
            response = await call_next(request)
            _apply_security_headers(request, response)
            return response

        # Reject oversized bodies before auth so a flood is cheap to turn away.
        clen = request.headers.get("content-length")
        if clen is not None and clen.isdigit() and int(clen) > MAX_BODY_BYTES:
            resp = JSONResponse(
                content={"detail": f"Request body too large (> {MAX_BODY_BYTES} bytes)."},
                status_code=413,
            )
            _apply_security_headers(request, resp)
            return resp

        try:
            _check_api_key(request)
            _check_admin_key(request)
            _check_rate_limit(request)
        except HTTPException as e:
            resp = JSONResponse(
                content={"detail": e.detail},
                status_code=e.status_code,
                headers=e.headers or {},
            )
            _apply_security_headers(request, resp)
            return resp

        response = await call_next(request)
        _apply_security_headers(request, response)
        return response


def warn_if_unconfigured() -> None:
    """Print loud warnings on boot when running without the safety nets."""
    if API_KEY is None:
        print(
            "[security] DELPHI_API_KEY not set — API is open. "
            "Set it in .env before exposing publicly.",
            flush=True,
        )
    if ADMIN_API_KEY is None:
        print(
            "[security] DELPHI_ADMIN_API_KEY not set — expensive endpoints "
            "(/api/results/*) fall back to the shared read key. Set a separate "
            "admin key to require it for those operations.",
            flush=True,
        )
    if RATE_LIMIT_DISABLED:
        print("[security] Rate limiting DISABLED by env.", flush=True)
    if DOCS_ENABLED:
        print("[security] /docs and /redoc are exposed.", flush=True)

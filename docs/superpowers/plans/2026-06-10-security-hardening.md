# Security Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden DelphiAI with security headers, brute-force-resistant auth, HTTPS/TLS enforcement, edge-WAF cooperation, automated DB backups, dependency patching, plus injection and DDoS defenses.

**Architecture:** Production is Vercel (Next.js) → Railway (FastAPI) → Neon (Postgres), with Cloudflare available at the API edge. HTTPS/WAF are platform-provisioned, so the app cooperates (HSTS, force-TLS-to-DB, trust proxy headers, body caps) and ships a runbook. Auth throttling is DB-backed because Vercel functions share no memory. All changes are env-gated so local dev (no TLS, no API key) keeps working.

**Tech Stack:** FastAPI/Starlette + psycopg2 (Python 3.12, pytest); Next.js 16 + next-auth + pg + zod (TypeScript, Vitest); PostgreSQL; GitHub Actions; Dependabot.

---

## File Structure

**Created:**
- `DelphiAIApp/Views/vitest.config.ts` — Vitest config (jsdom, `@/*` alias)
- `DelphiAIApp/Views/lib/rate-limit.ts` — DB-backed auth throttle + client-IP helper
- `DelphiAIApp/Views/lib/common-passwords.ts` — blocklist of weak passwords
- `DelphiAIApp/Views/lib/__tests__/rate-limit.test.ts`
- `DelphiAIApp/Views/lib/__tests__/password-policy.test.ts`
- `DelphiAIApp/Views/lib/__tests__/next-config-headers.test.ts`
- `DelphiAIApp/Models/db/schemas_auth.sql` — (modify) add `auth_attempts`
- `DelphiAIApp/Models/db/migrations/006_add_auth_attempts.sql`
- `DelphiAIApp/Models/tests/test_security_headers.py`
- `DelphiAIApp/Models/tests/test_sql_identifier_allowlist.py`
- `DelphiAIApp/Models/db/sql_identifiers.py` — table/column allowlist guard
- `scripts/backup_db.py` — pg_dump backup with testable command builder
- `DelphiAIApp/Models/tests/test_backup_db.py`
- `.github/dependabot.yml`
- `.github/workflows/security-audit.yml`
- `.github/workflows/db-backup.yml`
- `docs/SECURITY.md` — HTTPS + WAF + DDoS + patching runbook

**Modified:**
- `DelphiAIApp/security.py` — response security headers, CSP, body-size cap, prod API-key enforcement
- `DelphiAIApp/Views/next.config.ts` — `async headers()`
- `DelphiAIApp/Views/lib/auth.ts` — lockout + record attempts in `authorize()`
- `DelphiAIApp/Views/app/api/auth/signup/route.ts` — signup throttle + password policy
- `DelphiAIApp/Views/lib/api.ts` — request timeout in `apiFetch`
- `DelphiAIApp/Views/lib/db.ts` — `sslmode=require` for managed `DATABASE_URL`
- `DelphiAIApp/Models/data/load_to_db.py` — allowlist-guard dynamic table SQL
- `DelphiAIApp/Models/data/scrapers/ufc_scraper/pipelines.py` — allowlist-guard dynamic UPDATE
- `DelphiAIApp/Views/package.json` — Vitest devDeps + `test` script

---

## Phase 0 — Test tooling (Vitest)

### Task 0: Stand up Vitest in `Views/`

**Files:**
- Modify: `DelphiAIApp/Views/package.json`
- Create: `DelphiAIApp/Views/vitest.config.ts`

- [ ] **Step 1: Install dev dependencies**

Run (from `DelphiAIApp/Views`):
```bash
npm install -D vitest@^2 @testing-library/react@^16 @testing-library/jest-dom@^6 jsdom@^25 @vitejs/plugin-react@^4
```
Expected: packages added to `devDependencies`, no errors.

- [ ] **Step 2: Add the test script**

In `DelphiAIApp/Views/package.json`, add to `"scripts"`:
```json
    "test": "vitest run",
    "test:watch": "vitest"
```

- [ ] **Step 3: Create the Vitest config**

`DelphiAIApp/Views/vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
```

- [ ] **Step 4: Add a smoke test and run it**

Create `DelphiAIApp/Views/lib/__tests__/smoke.test.ts`:
```ts
import { describe, it, expect } from "vitest";
describe("vitest", () => {
  it("runs", () => {
    expect(1 + 1).toBe(2);
  });
});
```
Run (from `DelphiAIApp/Views`): `npm test`
Expected: 1 passed.

- [ ] **Step 5: Delete the smoke test and commit**

```bash
rm DelphiAIApp/Views/lib/__tests__/smoke.test.ts
git add DelphiAIApp/Views/package.json DelphiAIApp/Views/package-lock.json DelphiAIApp/Views/vitest.config.ts
git commit -m "test: add Vitest setup for frontend unit tests"
```

---

## Phase 1 — Security headers (both tiers)

### Task 1: FastAPI response security headers + CSP

**Files:**
- Modify: `DelphiAIApp/security.py`
- Test: `DelphiAIApp/Models/tests/test_security_headers.py`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Models/tests/test_security_headers.py`:
```python
"""Security header + body-cap + prod-key tests via FastAPI TestClient."""
import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DELPHI_DISABLE_RATELIMIT", "1")
    monkeypatch.delenv("DELPHI_API_KEY", raising=False)
    monkeypatch.delenv("DELPHI_ENV", raising=False)
    import DelphiAIApp.security as sec
    importlib.reload(sec)
    import DelphiAIApp.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "strict-transport-security" in r.headers
    assert "referrer-policy" in r.headers


def test_api_csp_is_locked_down(client):
    r = client.get("/health")
    assert "default-src 'none'" in r.headers["content-security-policy"]
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_security_headers.py -v`
Expected: FAIL — `KeyError: 'x-content-type-options'`.

- [ ] **Step 3: Implement headers in `security.py`**

In `DelphiAIApp/security.py`, add near the config block:
```python
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
```

Then in `SecurityMiddleware.dispatch`, wrap the returns so headers land on **every** response. Replace the body of `dispatch` with:
```python
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or path in _PUBLIC_PATHS:
            response = await call_next(request)
            _apply_security_headers(request, response)
            return response

        try:
            _check_api_key(request)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_security_headers.py -v`
Expected: `test_security_headers_present` and `test_api_csp_is_locked_down` PASS.

- [ ] **Step 5: Commit**

```bash
git add DelphiAIApp/security.py DelphiAIApp/Models/tests/test_security_headers.py
git commit -m "feat(security): add response security headers + locked-down API CSP"
```

### Task 2: Next.js security headers

**Files:**
- Modify: `DelphiAIApp/Views/next.config.ts`
- Test: `DelphiAIApp/Views/lib/__tests__/next-config-headers.test.ts`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Views/lib/__tests__/next-config-headers.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import config from "@/next.config";

describe("next.config security headers", () => {
  it("sets the expected headers on all routes", async () => {
    const rules = await config.headers!();
    const all = rules.find((r) => r.source === "/(.*)");
    expect(all).toBeDefined();
    const names = all!.headers.map((h) => h.key.toLowerCase());
    expect(names).toContain("strict-transport-security");
    expect(names).toContain("x-frame-options");
    expect(names).toContain("x-content-type-options");
    expect(names).toContain("referrer-policy");
    expect(names).toContain("permissions-policy");
    expect(names).toContain("content-security-policy");
  });

  it("uses a pragmatic CSP allowing inline styles only", () => {
    return config.headers!().then((rules) => {
      const csp = rules
        .find((r) => r.source === "/(.*)")!
        .headers.find((h) => h.key.toLowerCase() === "content-security-policy")!
        .value;
      expect(csp).toContain("script-src 'self'");
      expect(csp).toContain("style-src 'self' 'unsafe-inline'");
      expect(csp).toContain("frame-ancestors 'none'");
      expect(csp).not.toContain("'unsafe-eval'");
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Views`): `npm test -- next-config-headers`
Expected: FAIL — `config.headers` is undefined.

- [ ] **Step 3: Implement `headers()` in `next.config.ts`**

Replace `DelphiAIApp/Views/next.config.ts` with:
```ts
import type { NextConfig } from "next";

// Pragmatic CSP: recharts/framer-motion/next-auth inject inline styles, so
// style-src allows 'unsafe-inline'. Scripts stay 'self' (no 'unsafe-inline'/
// 'unsafe-eval'). connect-src includes the API origin when set so client-side
// fetches to FastAPI are allowed; otherwise same-origin only.
const apiOrigin = process.env.NEXT_PUBLIC_FASTAPI_URL ?? "";
const csp = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  `connect-src 'self'${apiOrigin ? " " + apiOrigin : ""}`,
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "object-src 'none'",
].join("; ");

const securityHeaders = [
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains; preload" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), interest-cohort=()" },
  { key: "Content-Security-Policy", value: csp },
];

const nextConfig: NextConfig = {
  async headers() {
    return [{ source: "/(.*)", headers: securityHeaders }];
  },
};

export default nextConfig;
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Views`): `npm test -- next-config-headers`
Expected: both tests PASS.

- [ ] **Step 5: Verify the app still renders, then commit**

Run (from `DelphiAIApp/Views`): `npm run build`
Expected: build completes without CSP-related runtime errors.
```bash
git add DelphiAIApp/Views/next.config.ts DelphiAIApp/Views/lib/__tests__/next-config-headers.test.ts
git commit -m "feat(security): add Next.js security headers + pragmatic CSP"
```

---

## Phase 2 — Injection hardening

### Task 3: SQL identifier allowlist guard

**Files:**
- Create: `DelphiAIApp/Models/db/sql_identifiers.py`
- Test: `DelphiAIApp/Models/tests/test_sql_identifier_allowlist.py`
- Modify: `DelphiAIApp/Models/data/load_to_db.py`, `DelphiAIApp/Models/data/scrapers/ufc_scraper/pipelines.py`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Models/tests/test_sql_identifier_allowlist.py`:
```python
import sys
from pathlib import Path

import pytest

_MODELS = Path(__file__).resolve().parents[1]
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

from db.sql_identifiers import safe_identifier, ALLOWED_TABLES


def test_known_table_passes():
    assert safe_identifier("FighterStats", ALLOWED_TABLES) == "FighterStats"


def test_injection_payload_rejected():
    with pytest.raises(ValueError):
        safe_identifier("users; DROP TABLE users;--", ALLOWED_TABLES)


def test_unknown_identifier_rejected():
    with pytest.raises(ValueError):
        safe_identifier("not_a_table", ALLOWED_TABLES)
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_sql_identifier_allowlist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db.sql_identifiers'`.

- [ ] **Step 3: Implement the guard**

`DelphiAIApp/Models/db/sql_identifiers.py`:
```python
"""Allowlist guard for SQL identifiers interpolated into f-string queries.

These statements interpolate table/column NAMES (psycopg2 cannot parameterize
identifiers). All call sites pass hardcoded internal constants, but routing them
through this guard makes the safety explicit and fails loudly if that ever
changes.
"""
from __future__ import annotations

ALLOWED_TABLES: frozenset[str] = frozenset({
    "Fights", "CareerStats", "FighterStats",
    "PointInTimeStats", "MatchupFeatures", "OpponentQuality",
    "PreUfcCareer", "EloHistory",
})


def safe_identifier(name: str, allowed: frozenset[str]) -> str:
    """Return `name` if it is in `allowed`, else raise ValueError."""
    if name not in allowed:
        raise ValueError(f"Refusing to interpolate unknown SQL identifier: {name!r}")
    return name
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_sql_identifier_allowlist.py -v`
Expected: 3 passed.

- [ ] **Step 5: Apply the guard at the dynamic-SQL call sites**

In `DelphiAIApp/Models/data/load_to_db.py`, add the import near the top:
```python
from db.sql_identifiers import safe_identifier, ALLOWED_TABLES
```
Change the clear loop (`for table in tables_to_clear:`) so the table is guarded:
```python
    for table in tables_to_clear:
        try:
            cursor.execute(f"DELETE FROM {safe_identifier(table, ALLOWED_TABLES)}")
            print(f"   Cleared {table}")
```
And the count loop (`for display_name, table_name in ml_tables:`):
```python
            cursor.execute(f"SELECT COUNT(*) FROM {safe_identifier(table_name, ALLOWED_TABLES)}")
```

In `DelphiAIApp/Models/data/scrapers/ufc_scraper/pipelines.py`, guard the dynamic UPDATE column names. Just before building `sql`, add:
```python
                    _allowed_cols = set(field_mapping.values())
                    for _f in update_fields:
                        _col = _f.split(" =", 1)[0].strip()
                        if _col not in _allowed_cols:
                            raise ValueError(f"Refusing unknown column in UPDATE: {_col!r}")
```
(The `WHERE FighterID = %s` and all values already use `%s` placeholders — leave them.)

- [ ] **Step 6: Re-run the guard test + commit**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_sql_identifier_allowlist.py -v`
Expected: 3 passed.
```bash
git add DelphiAIApp/Models/db/sql_identifiers.py DelphiAIApp/Models/tests/test_sql_identifier_allowlist.py DelphiAIApp/Models/data/load_to_db.py DelphiAIApp/Models/data/scrapers/ufc_scraper/pipelines.py
git commit -m "feat(security): allowlist-guard dynamic SQL identifiers (injection defense-in-depth)"
```

### Task 4: No `dangerouslySetInnerHTML` gate

**Files:**
- Test: `DelphiAIApp/Views/lib/__tests__/no-dangerous-html.test.ts`

- [ ] **Step 1: Write the test (acts as a regression gate)**

`DelphiAIApp/Views/lib/__tests__/no-dangerous-html.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { globSync } from "node:fs";

// Lightweight grep over the source tree (excludes node_modules + tests).
import { execSync } from "node:child_process";

describe("xss surface", () => {
  it("has no dangerouslySetInnerHTML in app/components/lib", () => {
    let hits = "";
    try {
      hits = execSync(
        `git grep -n "dangerouslySetInnerHTML" -- app components lib`,
        { cwd: fileURLToPath(new URL("../../", import.meta.url)) }
      ).toString();
    } catch {
      hits = ""; // git grep exits 1 when no matches — that's the pass case
    }
    expect(hits).toBe("");
  });
});
```

- [ ] **Step 2: Run it**

Run (from `DelphiAIApp/Views`): `npm test -- no-dangerous-html`
Expected: PASS (no current usages). If it FAILS, the offending file is printed — replace that usage with safe rendering before continuing.

- [ ] **Step 3: Commit**

```bash
git add DelphiAIApp/Views/lib/__tests__/no-dangerous-html.test.ts
git commit -m "test(security): gate against dangerouslySetInnerHTML (XSS)"
```

---

## Phase 3 — Strong authentication

### Task 5: `auth_attempts` table + migration

**Files:**
- Modify: `DelphiAIApp/Models/db/schemas_auth.sql`
- Create: `DelphiAIApp/Models/db/migrations/006_add_auth_attempts.sql`

- [ ] **Step 1: Add the table to `schemas_auth.sql`**

Append to `DelphiAIApp/Models/db/schemas_auth.sql` (before the `COMMENT ON` block):
```sql
CREATE TABLE IF NOT EXISTS auth_attempts (
    id              BIGSERIAL PRIMARY KEY,
    email           TEXT NOT NULL,
    ip              TEXT NOT NULL,
    success         BOOLEAN NOT NULL DEFAULT FALSE,
    attempted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_auth_attempts_lookup
    ON auth_attempts (email, ip, attempted_at);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip
    ON auth_attempts (ip, attempted_at);
```
And add a matching comment in the `COMMENT ON` block:
```sql
COMMENT ON TABLE auth_attempts IS 'Login/signup attempt log for brute-force lockout + signup throttling.';
```

- [ ] **Step 2: Create the migration**

`DelphiAIApp/Models/db/migrations/006_add_auth_attempts.sql`:
```sql
-- 006: brute-force lockout + signup-throttle support
CREATE TABLE IF NOT EXISTS auth_attempts (
    id              BIGSERIAL PRIMARY KEY,
    email           TEXT NOT NULL,
    ip              TEXT NOT NULL,
    success         BOOLEAN NOT NULL DEFAULT FALSE,
    attempted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_lookup
    ON auth_attempts (email, ip, attempted_at);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip
    ON auth_attempts (ip, attempted_at);
```

- [ ] **Step 3: Apply locally (if Postgres is up) and commit**

Run (optional, if Docker Postgres running):
```bash
docker exec -i delphi_postgres psql -U $DB_USER -d $DB_NAME < DelphiAIApp/Models/db/migrations/006_add_auth_attempts.sql
```
Expected: `CREATE TABLE` / `CREATE INDEX`.
```bash
git add DelphiAIApp/Models/db/schemas_auth.sql DelphiAIApp/Models/db/migrations/006_add_auth_attempts.sql
git commit -m "feat(security): add auth_attempts table for lockout + signup throttle"
```

### Task 6: DB-backed rate-limit helper

**Files:**
- Create: `DelphiAIApp/Views/lib/rate-limit.ts`
- Test: `DelphiAIApp/Views/lib/__tests__/rate-limit.test.ts`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Views/lib/__tests__/rate-limit.test.ts`:
```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the db module before importing the unit under test.
const queryMock = vi.fn();
vi.mock("@/lib/db", () => ({ query: queryMock }));

import { clientIp, isLockedOut, recordAttempt, tooManySignups } from "@/lib/rate-limit";

beforeEach(() => queryMock.mockReset());

describe("clientIp", () => {
  it("reads the first x-forwarded-for hop", () => {
    const h = new Headers({ "x-forwarded-for": "1.2.3.4, 10.0.0.1" });
    expect(clientIp(h)).toBe("1.2.3.4");
  });
  it("falls back to 'unknown'", () => {
    expect(clientIp(new Headers())).toBe("unknown");
  });
});

describe("isLockedOut", () => {
  it("locks out at the failure threshold", async () => {
    queryMock.mockResolvedValueOnce([{ fails: "5" }]);
    expect(await isLockedOut("a@b.com", "1.2.3.4")).toBe(true);
  });
  it("allows below threshold", async () => {
    queryMock.mockResolvedValueOnce([{ fails: "4" }]);
    expect(await isLockedOut("a@b.com", "1.2.3.4")).toBe(false);
  });
});

describe("tooManySignups", () => {
  it("blocks past the per-IP signup cap", async () => {
    queryMock.mockResolvedValueOnce([{ n: "3" }]);
    expect(await tooManySignups("9.9.9.9")).toBe(true);
  });
});

describe("recordAttempt", () => {
  it("inserts a row", async () => {
    queryMock.mockResolvedValueOnce([]);
    await recordAttempt("a@b.com", "1.2.3.4", false);
    expect(queryMock).toHaveBeenCalledOnce();
    const sql = queryMock.mock.calls[0][0] as string;
    expect(sql).toMatch(/insert into auth_attempts/i);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Views`): `npm test -- rate-limit`
Expected: FAIL — cannot resolve `@/lib/rate-limit`.

- [ ] **Step 3: Implement `rate-limit.ts`**

`DelphiAIApp/Views/lib/rate-limit.ts`:
```ts
import { query } from "./db";

// Tunables (see spec): 5 failed logins / 15 min → lockout; 3 signups / 15 min / IP.
const LOGIN_WINDOW_MIN = 15;
const LOGIN_MAX_FAILS = 5;
const SIGNUP_WINDOW_MIN = 15;
const SIGNUP_MAX = 3;

/** First hop of x-forwarded-for (Vercel sets this), else "unknown". */
export function clientIp(headers: Headers): string {
  const xff = headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  return headers.get("x-real-ip")?.trim() || "unknown";
}

/** True when (email, ip) has ≥ LOGIN_MAX_FAILS failures since the last success in-window. */
export async function isLockedOut(email: string, ip: string): Promise<boolean> {
  const rows = await query<{ fails: string }>(
    `SELECT COUNT(*) AS fails
       FROM auth_attempts
      WHERE email = $1 AND ip = $2
        AND success = FALSE
        AND attempted_at > NOW() - ($3 || ' minutes')::interval
        AND attempted_at > COALESCE(
              (SELECT MAX(attempted_at) FROM auth_attempts
                WHERE email = $1 AND ip = $2 AND success = TRUE),
              '-infinity'::timestamptz)`,
    [email, ip, String(LOGIN_WINDOW_MIN)]
  );
  return Number(rows[0]?.fails ?? 0) >= LOGIN_MAX_FAILS;
}

/** True when this IP has created ≥ SIGNUP_MAX accounts in the window. */
export async function tooManySignups(ip: string): Promise<boolean> {
  const rows = await query<{ n: string }>(
    `SELECT COUNT(*) AS n
       FROM auth_attempts
      WHERE ip = $1 AND email = '__signup__' AND success = TRUE
        AND attempted_at > NOW() - ($2 || ' minutes')::interval`,
    [ip, String(SIGNUP_WINDOW_MIN)]
  );
  return Number(rows[0]?.n ?? 0) >= SIGNUP_MAX;
}

/** Log an attempt. Best-effort: also prunes rows older than 24h. */
export async function recordAttempt(email: string, ip: string, success: boolean): Promise<void> {
  await query(
    `INSERT INTO auth_attempts (email, ip, success) VALUES ($1, $2, $3)`,
    [email, ip, success]
  );
  // Cheap opportunistic cleanup; ignore failures.
  query(`DELETE FROM auth_attempts WHERE attempted_at < NOW() - interval '24 hours'`).catch(
    () => {}
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Views`): `npm test -- rate-limit`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add DelphiAIApp/Views/lib/rate-limit.ts DelphiAIApp/Views/lib/__tests__/rate-limit.test.ts
git commit -m "feat(security): DB-backed auth throttle helper (lockout + signup cap)"
```

### Task 7: Password policy (length + common-password blocklist)

**Files:**
- Create: `DelphiAIApp/Views/lib/common-passwords.ts`
- Test: `DelphiAIApp/Views/lib/__tests__/password-policy.test.ts`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Views/lib/__tests__/password-policy.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { passwordSchema } from "@/lib/common-passwords";

describe("passwordSchema", () => {
  it("rejects < 10 chars", () => {
    expect(passwordSchema.safeParse("short1!a").success).toBe(false);
  });
  it("rejects a common password", () => {
    expect(passwordSchema.safeParse("password123").success).toBe(false);
  });
  it("accepts a strong-enough password", () => {
    expect(passwordSchema.safeParse("Tr0ubad0ur-Sunset").success).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Views`): `npm test -- password-policy`
Expected: FAIL — cannot resolve `@/lib/common-passwords`.

- [ ] **Step 3: Implement the blocklist + schema**

`DelphiAIApp/Views/lib/common-passwords.ts`:
```ts
import { z } from "zod";

// Top weak passwords (lowercased). NIST guidance: length-first + reject known-
// breached/common values, no forced symbol/complexity rules.
export const COMMON_PASSWORDS = new Set<string>([
  "password", "password1", "password123", "12345678", "123456789", "1234567890",
  "qwertyuiop", "qwerty123", "iloveyou", "admin123", "letmein123", "welcome123",
  "monkey123", "dragon123", "football1", "baseball1", "sunshine1", "princess1",
  "trustno1", "abc123456", "passw0rd", "p@ssw0rd", "changeme123", "delphiai",
]);

export const passwordSchema = z
  .string()
  .min(10, "Password must be at least 10 characters")
  .max(128)
  .refine((p) => !COMMON_PASSWORDS.has(p.toLowerCase()), {
    message: "Password is too common",
  });
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Views`): `npm test -- password-policy`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add DelphiAIApp/Views/lib/common-passwords.ts DelphiAIApp/Views/lib/__tests__/password-policy.test.ts
git commit -m "feat(security): stronger password policy (min 10 + common-password blocklist)"
```

### Task 8: Wire throttle + policy into login and signup

**Files:**
- Modify: `DelphiAIApp/Views/lib/auth.ts`, `DelphiAIApp/Views/app/api/auth/signup/route.ts`

- [ ] **Step 1: Harden `authorize()` in `lib/auth.ts`**

In `DelphiAIApp/Views/lib/auth.ts`, add imports:
```ts
import { headers } from "next/headers";
import { isLockedOut, recordAttempt, clientIp } from "./rate-limit";
```
Replace the body of `authorize: async (raw) => { ... }` with:
```ts
      authorize: async (raw) => {
        const parsed = credentialsSchema.safeParse(raw);
        if (!parsed.success) return null;
        const { email, password } = parsed.data;
        const ip = clientIp(await headers());

        // Brute-force lockout (generic failure — no account-existence oracle).
        if (await isLockedOut(email, ip)) return null;

        const rows = await query<UserRow>(
          `SELECT id, email, name, password_hash FROM users WHERE email = $1 LIMIT 1`,
          [email]
        );
        const user = rows[0];
        const ok = user ? await bcrypt.compare(password, user.password_hash) : false;
        await recordAttempt(email, ip, ok);
        if (!ok || !user) return null;
        return { id: user.id, email: user.email, name: user.name ?? user.email };
      },
```

- [ ] **Step 2: Harden the signup route**

In `DelphiAIApp/Views/app/api/auth/signup/route.ts`, update imports + schema + throttle:
```ts
import { headers } from "next/headers";
import { passwordSchema } from "@/lib/common-passwords";
import { tooManySignups, recordAttempt, clientIp } from "@/lib/rate-limit";
```
Change `signupSchema` to use the shared password policy:
```ts
const signupSchema = z.object({
  email: z.string().email().max(255),
  password: passwordSchema,
  name: z.string().min(1).max(100).optional(),
});
```
At the very start of `POST`, after parsing JSON succeeds and before the DB insert, add the per-IP throttle (place right after `const { email, name, password } = parsed.data;`... but compute ip first). Concretely, insert immediately after the `safeParse` success block:
```ts
  const ip = clientIp(await headers());
  if (await tooManySignups(ip)) {
    return NextResponse.json(
      { error: "Too many signups from this network. Try again later." },
      { status: 429 }
    );
  }
```
And after a successful insert (right before the final `return NextResponse.json(... 201)`), log it so the cap counts:
```ts
  await recordAttempt("__signup__", ip, true);
```

- [ ] **Step 3: Type-check + build**

Run (from `DelphiAIApp/Views`): `npx tsc --noEmit`
Expected: no errors.
Run: `npm test`
Expected: existing security tests still pass.

- [ ] **Step 4: Commit**

```bash
git add DelphiAIApp/Views/lib/auth.ts DelphiAIApp/Views/app/api/auth/signup/route.ts
git commit -m "feat(security): enforce login lockout + signup throttle + password policy"
```

### Task 9: Require API key in production (fail-fast)

**Files:**
- Modify: `DelphiAIApp/security.py`
- Test: `DelphiAIApp/Models/tests/test_security_headers.py` (add a case)

- [ ] **Step 1: Add the failing test**

Append to `DelphiAIApp/Models/tests/test_security_headers.py`:
```python
def test_production_requires_api_key(monkeypatch):
    monkeypatch.setenv("DELPHI_ENV", "production")
    monkeypatch.delenv("DELPHI_API_KEY", raising=False)
    import DelphiAIApp.security as sec
    with pytest.raises(RuntimeError):
        importlib.reload(sec)
    # Reset module state for other tests.
    monkeypatch.delenv("DELPHI_ENV", raising=False)
    importlib.reload(sec)
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_security_headers.py::test_production_requires_api_key -v`
Expected: FAIL — no RuntimeError raised.

- [ ] **Step 3: Implement the fail-fast check**

In `DelphiAIApp/security.py`, after the `API_KEY = ...` / `DOCS_ENABLED = ...` config lines, add:
```python
DELPHI_ENV = os.environ.get("DELPHI_ENV", "").strip().lower()

# In production the API must not boot open. Fail fast at import time so a
# misconfigured deploy crashes loudly instead of silently serving unauthed.
if DELPHI_ENV == "production" and API_KEY is None:
    raise RuntimeError(
        "DELPHI_ENV=production but DELPHI_API_KEY is unset — refusing to start "
        "an unauthenticated API."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_security_headers.py -v`
Expected: all PASS (including the new case).

- [ ] **Step 5: Commit**

```bash
git add DelphiAIApp/security.py DelphiAIApp/Models/tests/test_security_headers.py
git commit -m "feat(security): require DELPHI_API_KEY when DELPHI_ENV=production"
```

---

## Phase 4 — DDoS app-layer defenses

### Task 10: Request body-size cap (FastAPI)

**Files:**
- Modify: `DelphiAIApp/security.py`
- Test: `DelphiAIApp/Models/tests/test_security_headers.py` (add a case)

- [ ] **Step 1: Add the failing test**

Append to `DelphiAIApp/Models/tests/test_security_headers.py`:
```python
def test_oversized_body_rejected(client):
    big = "x" * (300 * 1024)  # 300 KB > 256 KB cap
    r = client.post(
        "/api/results/update",
        data=big,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_security_headers.py::test_oversized_body_rejected -v`
Expected: FAIL (likely 401/404/422, not 413).

- [ ] **Step 3: Implement the cap in `SecurityMiddleware.dispatch`**

In `DelphiAIApp/security.py`, add a constant near the other config:
```python
MAX_BODY_BYTES = int(os.environ.get("DELPHI_MAX_BODY_BYTES", str(256 * 1024)))
```
Then in `dispatch`, immediately after the `OPTIONS`/public-path bypass block, add the cap check (runs before auth so a flood is cheap to reject):
```python
        clen = request.headers.get("content-length")
        if clen is not None and clen.isdigit() and int(clen) > MAX_BODY_BYTES:
            resp = JSONResponse(
                content={"detail": f"Request body too large (> {MAX_BODY_BYTES} bytes)."},
                status_code=413,
            )
            _apply_security_headers(request, resp)
            return resp
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_security_headers.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add DelphiAIApp/security.py DelphiAIApp/Models/tests/test_security_headers.py
git commit -m "feat(security): reject oversized request bodies (413) to limit DoS"
```

### Task 11: Outbound fetch timeout (Next.js → FastAPI)

**Files:**
- Modify: `DelphiAIApp/Views/lib/api.ts`
- Test: `DelphiAIApp/Views/lib/__tests__/api-timeout.test.ts`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Views/lib/__tests__/api-timeout.test.ts`:
```ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { apiFetch, ApiError } from "@/lib/api";

afterEach(() => vi.restoreAllMocks());

describe("apiFetch timeout", () => {
  it("passes an AbortSignal to fetch", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 })
    );
    await apiFetch("/health");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it("throws ApiError(504) when the request aborts", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      Object.assign(new Error("aborted"), { name: "AbortError" })
    );
    await expect(apiFetch("/health")).rejects.toBeInstanceOf(ApiError);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Views`): `npm test -- api-timeout`
Expected: FAIL — no signal passed / no ApiError on abort.

- [ ] **Step 3: Add the timeout to `_fetchOrThrow`**

In `DelphiAIApp/Views/lib/api.ts`, add a constant under the existing config:
```ts
const REQUEST_TIMEOUT_MS = Number(process.env.DELPHI_FETCH_TIMEOUT_MS ?? 15000);
```
Wrap the fetch in `_fetchOrThrow` with an AbortController:
```ts
async function _fetchOrThrow<T>(path: string, init: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
        ...(init.headers ?? {}),
      },
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new ApiError(504, `Upstream timed out after ${REQUEST_TIMEOUT_MS}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }

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
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Views`): `npm test -- api-timeout`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add DelphiAIApp/Views/lib/api.ts DelphiAIApp/Views/lib/__tests__/api-timeout.test.ts
git commit -m "feat(security): add request timeout to apiFetch (resource-exhaustion guard)"
```

---

## Phase 5 — HTTPS / TLS to the database

### Task 12: Require `sslmode=require` for managed Postgres (Next.js)

**Files:**
- Modify: `DelphiAIApp/Views/lib/db.ts`
- Test: `DelphiAIApp/Views/lib/__tests__/db-ssl.test.ts`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Views/lib/__tests__/db-ssl.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { resolveSsl } from "@/lib/db";

describe("resolveSsl", () => {
  it("requires SSL for a remote DATABASE_URL", () => {
    expect(resolveSsl("postgres://u:p@ep.neon.tech/db")).toEqual({
      rejectUnauthorized: true,
    });
  });
  it("disables SSL for localhost", () => {
    expect(resolveSsl("postgres://u:p@localhost:5433/db")).toBe(false);
  });
  it("respects an explicit sslmode=disable", () => {
    expect(resolveSsl("postgres://u:p@ep.neon.tech/db?sslmode=disable")).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Views`): `npm test -- db-ssl`
Expected: FAIL — `resolveSsl` not exported.

- [ ] **Step 3: Implement + use `resolveSsl` in `db.ts`**

In `DelphiAIApp/Views/lib/db.ts`, add an exported helper and use it in `makePool`:
```ts
/** TLS policy for pg: require for managed/remote hosts, off for localhost. */
export function resolveSsl(connectionString: string): false | { rejectUnauthorized: boolean } {
  try {
    const u = new URL(connectionString);
    if (u.searchParams.get("sslmode") === "disable") return false;
    const host = u.hostname;
    if (host === "localhost" || host === "127.0.0.1") return false;
    return { rejectUnauthorized: true };
  } catch {
    return false;
  }
}
```
Update the `DATABASE_URL` branch of `makePool`:
```ts
  if (process.env.DATABASE_URL) {
    return new Pool({
      connectionString: process.env.DATABASE_URL,
      ssl: resolveSsl(process.env.DATABASE_URL),
    });
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Views`): `npm test -- db-ssl`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add DelphiAIApp/Views/lib/db.ts DelphiAIApp/Views/lib/__tests__/db-ssl.test.ts
git commit -m "feat(security): require TLS to managed Postgres from the web tier"
```

> Note: `Models/db/postgres.py` already supports TLS via `DATABASE_URL` (carries its own `sslmode`) and the optional `DB_SSLMODE` var — documented in `docs/SECURITY.md`, no code change needed.

---

## Phase 6 — Automated backups

### Task 13: Backup script with a testable command builder

**Files:**
- Create: `scripts/backup_db.py`
- Test: `DelphiAIApp/Models/tests/test_backup_db.py`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Models/tests/test_backup_db.py`:
```python
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.backup_db import build_pg_dump_cmd, backup_filename


def test_build_cmd_uses_database_url():
    cmd = build_pg_dump_cmd("postgres://u:p@host/db", "/tmp/out.sql.gz")
    assert cmd[0] == "pg_dump"
    assert "postgres://u:p@host/db" in cmd
    assert "-Fc" in cmd


def test_backup_filename_is_timestamped():
    name = backup_filename("delphi")
    assert name.startswith("delphi-")
    assert name.endswith(".dump")
```

- [ ] **Step 2: Run test to verify it fails**

Run (from project root): `python -m pytest DelphiAIApp/Models/tests/test_backup_db.py -v`
Expected: FAIL — `No module named 'scripts.backup_db'`.

- [ ] **Step 3: Implement the script**

`scripts/backup_db.py`:
```python
"""Logical Postgres backup via pg_dump (custom format).

Resolves the connection from DATABASE_URL (Neon/Railway) or DB_* env vars
(local docker-compose). Writes a timestamped .dump to BACKUP_DIR (default
./backups) and prunes dumps older than BACKUP_RETENTION_DAYS.

Usage:
    python -m scripts.backup_db
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def backup_filename(db_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{db_name}-{stamp}.dump"


def _dsn_from_env() -> tuple[str, str]:
    """Return (dsn, db_name) from DATABASE_URL or DB_* vars."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        from urllib.parse import urlparse
        name = urlparse(url).path.lstrip("/") or "database"
        return url, name
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    name = os.getenv("DB_NAME", "delphi_db")
    user = os.getenv("DB_USER", "")
    pw = os.getenv("DB_PASSWORD", "")
    dsn = f"postgresql://{user}:{pw}@{host}:{port}/{name}"
    return dsn, name


def build_pg_dump_cmd(dsn: str, out_path: str) -> list[str]:
    # -Fc = custom (compressed, restorable with pg_restore); -f = output file.
    return ["pg_dump", "-Fc", "-f", out_path, dsn]


def prune_old(backup_dir: Path, retention_days: int) -> int:
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for f in backup_dir.glob("*.dump"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    return removed


def main() -> int:
    backup_dir = Path(os.getenv("BACKUP_DIR", "backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    retention = int(os.getenv("BACKUP_RETENTION_DAYS", "14"))

    dsn, db_name = _dsn_from_env()
    out_path = backup_dir / backup_filename(db_name)
    cmd = build_pg_dump_cmd(dsn, str(out_path))

    print(f"[backup] writing {out_path} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[backup] FAILED: {result.stderr}", file=sys.stderr)
        return result.returncode
    print(f"[backup] OK ({out_path.stat().st_size} bytes)", flush=True)

    pruned = prune_old(backup_dir, retention)
    if pruned:
        print(f"[backup] pruned {pruned} dump(s) older than {retention}d", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run (from project root): `python -m pytest DelphiAIApp/Models/tests/test_backup_db.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/backup_db.py DelphiAIApp/Models/tests/test_backup_db.py
git commit -m "feat(security): add pg_dump backup script with retention pruning"
```

### Task 14: Scheduled backup workflow

**Files:**
- Create: `.github/workflows/db-backup.yml`

- [ ] **Step 1: Create the workflow**

`.github/workflows/db-backup.yml`:
```yaml
name: DB Backup
on:
  schedule:
    - cron: "0 7 * * *" # daily 07:00 UTC
  workflow_dispatch: {}

jobs:
  backup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install PostgreSQL client (pg_dump)
        run: |
          sudo apt-get update
          sudo apt-get install -y postgresql-client
      - name: Run backup
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          BACKUP_DIR: backups
          BACKUP_RETENTION_DAYS: "14"
        run: python -m scripts.backup_db
      - name: Upload backup artifact
        uses: actions/upload-artifact@v4
        with:
          name: db-backup-${{ github.run_id }}
          path: backups/*.dump
          retention-days: 14
```

- [ ] **Step 2: Validate the YAML parses**

Run (from project root): `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/db-backup.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/db-backup.yml
git commit -m "ci(security): daily Neon backup workflow with artifact upload"
```

---

## Phase 7 — Updates & patching

### Task 15: Dependabot config

**Files:**
- Create: `.github/dependabot.yml`

- [ ] **Step 1: Create the config**

`.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/DelphiAIApp/Views"
    schedule:
      interval: "weekly"
    groups:
      minor-and-patch:
        update-types: ["minor", "patch"]
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      minor-and-patch:
        update-types: ["minor", "patch"]
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

- [ ] **Step 2: Validate the YAML parses**

Run (from project root): `python -c "import yaml; yaml.safe_load(open('.github/dependabot.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/dependabot.yml
git commit -m "ci(security): enable Dependabot for npm, pip, and actions"
```

### Task 16: Dependency-audit workflow

**Files:**
- Create: `.github/workflows/security-audit.yml`

- [ ] **Step 1: Create the workflow**

`.github/workflows/security-audit.yml`:
```yaml
name: Security Audit
on:
  pull_request: {}
  schedule:
    - cron: "0 6 * * 1" # weekly Monday 06:00 UTC
  workflow_dispatch: {}

jobs:
  npm-audit:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: DelphiAIApp/Views
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
      - run: npm ci
      - name: npm audit (high+)
        run: npm audit --audit-level=high

  pip-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pip-audit
      - name: pip-audit
        run: pip-audit -r requirements.txt
```

- [ ] **Step 2: Validate the YAML parses**

Run (from project root): `python -c "import yaml; yaml.safe_load(open('.github/workflows/security-audit.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/security-audit.yml
git commit -m "ci(security): npm audit + pip-audit workflow"
```

---

## Phase 8 — Runbook

### Task 17: `docs/SECURITY.md` (HTTPS + WAF + DDoS + patching)

**Files:**
- Create: `docs/SECURITY.md`

- [ ] **Step 1: Write the runbook**

`docs/SECURITY.md`:
```markdown
# DelphiAI Security Runbook

Production topology: **Vercel (Next.js) → Railway (FastAPI) → Neon (Postgres)**,
with **Cloudflare** available in front of the Railway API.

## 1. HTTPS / SSL
- **Vercel** terminates TLS automatically for the app + any custom domain. No action
  beyond adding the domain in the Vercel dashboard.
- **Railway** serves the API over HTTPS on `*.up.railway.app`; add a custom domain
  for a managed cert if desired.
- **Neon**: connections require TLS. Ensure `DATABASE_URL` carries `sslmode=require`
  (the web tier enforces this via `resolveSsl` in `lib/db.ts`; the Python tier honors
  the URL's own `sslmode` / the `DB_SSLMODE` var).
- **HSTS** is sent by both tiers (2-year, `includeSubDomains; preload`). Submit the
  apex domain at https://hstspreload.org once stable.

## 2. WAF
- **Frontend (Vercel Firewall):** enable in Project → Firewall. Turn on the managed
  ruleset + Attack Challenge Mode. Add a rate-limit rule on `/api/auth/*`.
- **API (Cloudflare):** proxy the Railway custom domain (orange cloud). Enable the
  Cloudflare Managed Ruleset (OWASP core), Bot Fight Mode, and a rate-limiting rule
  on `/api/*`. The app already reads `cf-connecting-ip`, so per-IP limits stay correct.

## 3. DDoS — layered defense
1. **Edge (volumetric):** Cloudflare + Vercel rate limiting/challenge (above) — primary.
2. **App-layer (abuse):** FastAPI per-IP sliding-window limiter (`security.py`);
   Next.js login lockout + signup throttle (`auth_attempts`).
3. **Resource exhaustion:** 256 KB request-body cap (FastAPI → 413), 15 s `apiFetch`
   timeout, and Neon's **pooled** connection endpoint to avoid connection exhaustion.

## 4. Authentication
- Login: 5 failed attempts per (email, IP) in 15 min → lockout (generic failure).
- Signup: 3 accounts per IP / 15 min → 429.
- Passwords: min 10 chars + common-password blocklist (NIST length-first).
- `DELPHI_ENV=production` requires `DELPHI_API_KEY`, else the API refuses to boot.
- Required env: `AUTH_SECRET` (next-auth), `DELPHI_API_KEY`, `DATABASE_URL`.

## 5. Backups
- Primary: Neon point-in-time restore / branching.
- Secondary: `scripts/backup_db.py` via `.github/workflows/db-backup.yml` (daily,
  `pg_dump -Fc`, 14-day retention, uploaded as an artifact). Restore with
  `pg_restore -d <DATABASE_URL> <file>.dump`.

## 6. Updates & patching
- Dependabot (`.github/dependabot.yml`): weekly npm + pip + actions PRs.
- `.github/workflows/security-audit.yml`: `npm audit --audit-level=high` + `pip-audit`
  on every PR and weekly.
- **Cadence:** triage critical/high CVEs within 72h; merge grouped minor/patch PRs weekly.

## Future hardening (not yet implemented)
MFA/TOTP, OAuth providers, secret-manager vault, nonce-based strict CSP, IaC.
```

- [ ] **Step 2: Commit**

```bash
git add docs/SECURITY.md
git commit -m "docs(security): add HTTPS/WAF/DDoS/backup/patching runbook"
```

---

## Final verification

- [ ] **Backend suite:** from `DelphiAIApp/Models`, run `python -m pytest tests/test_security_headers.py tests/test_sql_identifier_allowlist.py tests/test_backup_db.py -v` → all pass.
- [ ] **Frontend suite:** from `DelphiAIApp/Views`, run `npm test` → all pass.
- [ ] **Type check:** from `DelphiAIApp/Views`, run `npx tsc --noEmit` → clean.
- [ ] **Manual header check (optional, with API running):** `curl -I http://localhost:8000/health` shows the security headers + CSP.
- [ ] **YAML:** all three `.github` YAML files parse.
- [ ] **Local dev still works:** FastAPI boots without `DELPHI_API_KEY` (warning only, `DELPHI_ENV` unset); web app builds.
```

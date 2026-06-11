# Design: Security Hardening — Headers, Auth, HTTPS, WAF, Backups, Patching

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan

## Goal

Implement the security features requested for DelphiAI across six workstreams,
plus explicit **injection** and **DDoS** protection. Production topology
(from `docs/deployment-plan.md`):

```
Vercel (Next.js) ──HTTP + X-API-Key──► Railway (FastAPI) ──► Neon (Postgres)
       │  (Vercel Firewall / WAF)         │ (Cloudflare in front)
       └─ auto TLS                         └─ auto TLS,  cf-connecting-ip trusted
```

Because hosting is platform-managed, HTTPS and WAF are mostly *provisioned at
the edge*. The app's job is to (a) cooperate correctly (HSTS, force-HTTPS,
trust proxy headers, require TLS to DB) and (b) ship a precise runbook.

## Confirmed decisions
- **CSP:** pragmatic, enforced (not nonce-based strict). `style-src` allows
  `'unsafe-inline'` for recharts/framer-motion/next-auth.
- **Auth throttle storage:** DB-backed `auth_attempts` table (serverless-safe;
  Vercel functions share no memory).
- Add **injection** hardening/verification and **DDoS** layered defense.

## Out of scope (YAGNI)
MFA/TOTP, OAuth providers, secret-vault, nonce strict CSP, IaC/Terraform. Noted
as future options in `docs/SECURITY.md`.

---

## Workstream 1 — Security Headers *(code, both tiers)*

**Next.js** — `Views/next.config.ts` `async headers()` applied to all routes:
- `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=(), interest-cohort=()`
- `Content-Security-Policy` (pragmatic):
  `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';
   img-src 'self' data:; font-src 'self'; connect-src 'self' <FASTAPI_URL>;
   frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'`
  - `connect-src` includes the FastAPI origin (env-driven) so the browser can
    reach the API. Built from `process.env.NEXT_PUBLIC_*`/`FASTAPI_URL` at build.

**FastAPI** — `DelphiAIApp/security.py`, headers added to every response in
`SecurityMiddleware.dispatch` (after `call_next`):
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: no-referrer`
- `Strict-Transport-Security` (same as above) — safe because edge is HTTPS.
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'` for
  API/JSON responses; **skipped** for `/docs`,`/redoc`,`/openapi.json` (Swagger
  needs inline + CDN) when `DOCS_ENABLED`.
- `X-Frame-Options: DENY`.

**Test:** `Views/lib/__tests__` not needed; assert headers via a FastAPI
`TestClient` test (`Models/tests/test_security_headers.py`) and a Next config
unit check. Manual: `curl -I`.

## Workstream 2 — Strong Authentication *(code — biggest gap)*

**2a. `auth_attempts` table** (new, added to `Models/db/schemas.sql` + a
migration-safe `CREATE TABLE IF NOT EXISTS`):
```sql
CREATE TABLE IF NOT EXISTS auth_attempts (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT NOT NULL,
    ip          TEXT NOT NULL,
    success     BOOLEAN NOT NULL DEFAULT FALSE,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_lookup
    ON auth_attempts (email, ip, attempted_at);
```
- New helper `Views/lib/rate-limit.ts`:
  `recordAttempt(email, ip, success)` and
  `isLockedOut(email, ip)` → true when ≥ **5 failed** attempts for that
  (email, ip) within the last **15 minutes** and no later success.
- Wire into `lib/auth.ts` `authorize()`: check `isLockedOut` first → return
  `null` (generic failure, no oracle); record failure on bad password, clear/
  record success on good login. Read client IP from `x-forwarded-for` (Vercel).
- Wire into `app/api/auth/signup/route.ts`: throttle account-creation per IP
  (≥ **3** signups / 15 min / IP → 429) to stop account-spam DoS.
- Periodic cleanup: signup/login best-effort `DELETE` of rows older than 24h
  (cheap, no cron needed).

**2b. Password policy** (`signupSchema` in signup route + `credentialsSchema`):
- Raise min length 8 → **10**, keep max 128.
- Reject a small embedded **common-password blocklist** (top ~50) — length-first,
  NIST-aligned, no forced symbol/complexity rules.

**2c. Production API-key enforcement** (`security.py`):
- New `DELPHI_ENV` read. When `DELPHI_ENV=production` and `DELPHI_API_KEY`
  unset → raise at import/boot (fail fast) instead of only warning.

**2d. Session hardening (verify + document):**
- Confirm `AUTH_SECRET` required (next-auth throws if unset in prod) — document.
- next-auth defaults to `__Secure-`/`httpOnly`/`sameSite=lax` cookies in prod —
  document, no code change unless missing.

**Tests:** `Views/lib/__tests__/rate-limit.test.ts` (mock `query`): lockout
after threshold, window expiry, success resets. `signupSchema` validation tests.

## Workstream 3 — HTTPS / SSL *(code + runbook)*

- **DB TLS:** `Views/lib/db.ts` and `Models/db/postgres.py` — when a managed
  `DATABASE_URL` is present, ensure `sslmode=require` (append if absent). Keeps
  local Docker (no SSL) working via the `DB_*` fallback path.
- **HSTS / force-HTTPS:** HSTS header (WS1). Edge platforms already redirect
  HTTP→HTTPS; documented, not coded.
- **Runbook** (`docs/SECURITY.md` → HTTPS section): Vercel custom-domain TLS,
  Railway TLS + custom domain, Neon `sslmode=require`, HSTS preload submission,
  cert auto-renewal note (managed).

## Workstream 4 — WAF *(config + runbook)*

- **Frontend (Vercel):** enable **Vercel Firewall** — managed ruleset + Attack
  Challenge Mode; optional `Views/vercel.json` for any committed config. Add
  Vercel rate-limit rules on `/api/auth/*`.
- **API (Railway):** put behind **Cloudflare** (orange-cloud proxy). The app
  already trusts `cf-connecting-ip`, so the rate limiter stays correct. Enable
  Cloudflare Managed WAF ruleset + a rate-limiting rule on `/api/*`.
- **Runbook** (`docs/SECURITY.md` → WAF section): exact Cloudflare/Vercel steps
  and the specific recommended rules (OWASP managed ruleset, bot fight mode,
  per-path rate rules).

## Workstream 5 — Automated Backups *(script + scheduler)*

- `scripts/backup_db.py` — `pg_dump` (via `DATABASE_URL` or `DB_*`), gzip,
  timestamped filename, optional upload to Cloudflare R2 (reuses the R2 creds
  the deployment plan already provisions); local dir fallback. Prune backups
  older than N days.
- `.github/workflows/db-backup.yml` — scheduled cron (daily), runs the script
  against Neon using repo secrets, uploads artifact + R2.
- Document Neon's built-in **PITR/branching** as primary; this script as
  portable belt-and-suspenders (also covers local Docker dev).

## Workstream 6 — Updates & Patching *(tooling/config)*

- `.github/dependabot.yml` — weekly: `npm` (`Views/`), `pip` (root
  `requirements.txt`), `github-actions`. Grouped minor/patch PRs.
- `.github/workflows/security-audit.yml` — on PR + weekly cron: `npm audit
  --audit-level=high` (Views) and `pip-audit` (root). Non-blocking warn first,
  tightened later.
- `docs/SECURITY.md` → patching-cadence section (triage SLA for critical CVEs).

## Cross-cutting — Injection protection *(verify + harden)*

Audit finding: **all request-handling DB access is parameterized** (FastAPI
psycopg2 `%s`, Next.js pg `$1`). f-string SQL exists only in batch/scraper/
maintenance scripts and interpolates **hardcoded table/column identifiers**, not
user input.

- Add an `_ALLOWED_TABLES` allowlist assertion to the two dynamic-identifier
  spots that touch the widest surface: `Models/data/load_to_db.py` (`DELETE
  FROM {table}`, `COUNT(*) FROM {table_name}`) and `pipelines.py` dynamic
  `UPDATE ... SET {fields}` — assert each identifier ∈ known set before
  formatting. Defense-in-depth; no behavior change for valid inputs.
- Verify every request input is schema-validated: zod (all Next routes — already
  present), and add explicit validation on any FastAPI path params/query that
  reach SQL.
- XSS: confirm no `dangerouslySetInnerHTML` in `Views/` (grep gate); React
  auto-escaping + the CSP (WS1) are the layered defense.
- **Test:** `Models/tests/test_sql_injection.py` — feed a classic payload
  (`' OR 1=1; DROP TABLE…`) through a representative Service/Controller path and
  assert it is treated as a literal (no error, no row leak).

## Cross-cutting — DDoS protection *(layered)*

1. **Edge (volumetric):** Cloudflare (API) + Vercel Firewall (frontend) rate
   limiting and challenge — primary defense (WS4 runbook).
2. **App-layer (abuse):** FastAPI per-IP sliding-window limiter already exists —
   **verify buckets cover the new/auth paths**; add an `auth_attempts` throttle
   on the Next.js auth routes (WS2a).
3. **Resource-exhaustion:**
   - **Body-size cap:** reject `Content-Length` over a limit (e.g. 256 KB) in
     `SecurityMiddleware` and in Next.js auth/bets routes → 413.
   - **Request timeout** on outbound `apiFetch` (`lib/api.ts`) so a slow API
     can't pin Vercel functions.
   - **DB pool / connection** note: use Neon **pooled** endpoint (deployment
     plan A.4) to avoid connection-exhaustion DoS — documented.
- **Test:** rate-limit + 413 body-cap covered by `test_security_headers.py` /
  middleware tests.

---

## Build order (each workstream independently shippable)
1. WS1 Security Headers (fast, high value, low risk)
2. Injection hardening + test (small, verify-heavy)
3. WS2 Strong Auth (`auth_attempts` table + throttle + policy + prod key)
4. DDoS app-layer (body cap, timeout, bucket coverage)
5. WS3 HTTPS code bits (`sslmode=require`)
6. WS5 Backups script + workflow
7. WS6 Dependabot + audit workflow
8. WS3/WS4 runbook → `docs/SECURITY.md` (consolidates HTTPS + WAF + DDoS/patch docs)

## Success criteria
- `curl -I` on both tiers shows the headers; CSP doesn't break the rendered app.
- Lockout triggers after 5 failed logins; signup throttles per IP.
- `DELPHI_ENV=production` without `DELPHI_API_KEY` refuses to boot.
- Injection payload test passes (treated as literal).
- Body-size cap returns 413; oversized auth payload rejected.
- Backup script produces a restorable `pg_dump`; workflow runs green.
- Dependabot + audit workflow active.
- `docs/SECURITY.md` runbook covers Vercel/Railway/Cloudflare/Neon setup.
- No production behavior regressions; local dev (no TLS, no API key) still works.

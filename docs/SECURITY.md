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

> **Caveat:** the FastAPI sliding-window limiter is **in-memory and per-process** —
> it is not shared across instances and resets on cold start/redeploy. Treat it as
> a cheap second layer; the **edge WAF rate limits are the real volumetric defense**.

## 4. Authentication
- Login: 5 failed attempts per (email, IP) in 15 min → lockout (generic failure).
- Signup: 3 accounts per IP / 15 min → 429.
- Passwords: min 10 chars + common-password blocklist (NIST length-first).
- `DELPHI_ENV=production` requires `DELPHI_API_KEY`, else the API refuses to boot.
- Required env: `AUTH_SECRET` (next-auth), `DELPHI_API_KEY`, `DATABASE_URL`.
- Client IP for throttling prefers the edge-set `x-real-ip` (Vercel) / `cf-connecting-ip`
  (Cloudflare); the leftmost `x-forwarded-for` hop is only a fallback for local dev,
  so a client cannot forge an IP to evade lockout in production.
- **CSRF:** the next-auth session is a JWT cookie with the default **`SameSite=Lax`,
  `HttpOnly`, `Secure`** flags. All state-changing bet endpoints are `POST`/`PATCH`/
  `DELETE`, which `SameSite=Lax` does **not** send cross-site — so they are not
  CSRF-triggerable. (Bet data is a non-sensitive personal tracker; no anti-tamper
  beyond injection-safe input validation is warranted.)
- **Privilege separation:** expensive/mutating routes (`POST /api/results/*` — model
  runs, scrapes, bulk writes) honor an optional **`DELPHI_ADMIN_API_KEY`**. When set,
  they require it via `X-Admin-Key` on top of the read `DELPHI_API_KEY`, so a leaked
  read key (deployed broadly in the web tier) can't trigger them. Unset = falls back
  to the read key (non-breaking). The frontend never calls these routes.

## 4a. Model artifact integrity (anti-RCE)
- The XGBoost+calibrator artifact is loaded with `pickle`, which executes arbitrary
  code — so a tampered model from the R2 bucket would be RCE in the API container.
- `model_loader` verifies a **SHA-256 before unpickling**, against a trusted digest
  from `DELPHI_MODEL_SHA256` (pinned at deploy; survives a fully compromised bucket)
  or a `<model>.sha256` sidecar (corruption/partial-tamper guard). Mismatch ⇒ refuse
  to load. `publish_model` prints the digest + uploads the sidecar; `fetch_model`
  verifies the download against the pin (and disables redirect-following).

## 4b. SSRF / server-side fetches
- The on-demand fighter scraper follows a detail URL taken as an `<a href>` off a
  scraped page, so the target isn't always one we constructed. `ml/net_guard.safe_get`
  restricts scheme to http(s), host to a **data-source allowlist**, and resolved IPs
  to **public space** (blocks `169.254.169.254`/loopback/RFC-1918, incl. DNS rebind),
  and re-validates every redirect hop. Reusable for the other request-path fetches.

## 5. Backups
- Primary: Neon point-in-time restore / branching.
- Secondary: `scripts/backup_db.py` via `.github/workflows/db-backup.yml` (daily,
  `pg_dump -Fc`, 14-day retention). The dump is **gpg-AES256-encrypted** with the
  `BACKUP_PASSPHRASE` secret before it is uploaded as an artifact — the workflow
  refuses to run if that secret is unset, so a plaintext DB dump is never egressed.
  The dump credentials are passed to `pg_dump` via `PGPASSWORD` (never in argv) and
  any libpq errors are redacted before logging. Restore with:
  `gpg --decrypt <file>.dump.gpg > restore.dump && pg_restore -d <DATABASE_URL> restore.dump`.

## 6. Updates & patching
- Dependabot (`.github/dependabot.yml`): weekly npm + pip + actions PRs.
- `.github/workflows/security-audit.yml`: `npm audit --audit-level=high` + `pip-audit`
  on every PR and weekly.
- **Cadence:** triage critical/high CVEs within 72h; merge grouped minor/patch PRs weekly.

## Future hardening (not yet implemented)
MFA/TOTP, OAuth providers, secret-manager vault + key rotation, nonce-based strict
CSP, IaC. Extend `net_guard.safe_get` to the remaining request-path fetches
(event-page scrapes in `predict_card` / `update_results` / `upcoming_predictor`).

> Deliberately out of scope: account lifecycle (email verify/MFA/reset) and bet
> anti-tampering — bets are a non-sensitive personal tracker, so confidentiality
> and result-integrity controls there add no security value.

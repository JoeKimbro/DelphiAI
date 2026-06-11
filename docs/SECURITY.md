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
- Client IP for throttling prefers the edge-set `x-real-ip` (Vercel) / `cf-connecting-ip`
  (Cloudflare); the leftmost `x-forwarded-for` hop is only a fallback for local dev,
  so a client cannot forge an IP to evade lockout in production.

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

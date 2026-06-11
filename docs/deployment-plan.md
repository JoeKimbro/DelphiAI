# Deployment Foundation Plan — Dockerfile, Model Delivery, Migrations

Status legend: ⬜ not started · 🟦 in progress · ✅ done

## Target topology

```
Vercel (Next.js)  ──HTTP+X-API-Key──►  Railway (FastAPI container)
      │                                       │
      └────────────pg──────────┐              │ psycopg2
                               ▼              ▼
                        Neon (managed Postgres, pooled)
                               ▲
                               │ push data weekly
                     GitHub Actions cron (weekly_update)
                               ▲
                          Cloudflare R2 ──model_latest.pkl──► Railway (boot pull)
```

### Hosting decisions
- **Frontend → Vercel** (free Hobby). Native Next.js build, no Dockerfile.
- **Backend (FastAPI) → Railway** (~$5/mo, stays warm). Render free tier rejected: 15-min
  idle spin-down → ~50s cold start, which breaks the slow live-scrape predict path.
- **Database → Neon** (free Postgres, pooled endpoint for Vercel serverless). Supabase is an
  equivalent alternative; the only delta is the model would live in Supabase Storage instead of R2.
- **Model storage → Cloudflare R2** (free 10GB, S3-compatible), pulled at container boot.
- **Scrapers → GitHub Actions cron** running `weekly_update`, pushing data to Neon.

### Confirmed sub-decisions
- Backend image is **light**: the predict-time injury scrape uses only `requests` + `bs4`
  (no headless browser). Heavy Scrapy/Playwright scraping lives in the Actions cron, not the container.
- `DELPHI_API_KEY` becomes **required in production** (gated by a `DELPHI_ENV=production` flag);
  today the API boots open with only a warning.

---

## Phase 0 — Prerequisite fixes (blocks everything) — ✅ COMPLETE
- ✅ **0.1** Generate `requirements.txt` (was empty). Pinned from validated local env (curated, not raw freeze).
- ✅ **0.2** Fix `Models/db/postgres.py`: no longer raises when `.env` is missing; `load_dotenv` only if file exists.
- ✅ **0.3** Support a single `DATABASE_URL` connection string (Neon/Railway/Vercel hand you one);
  `DB_*` vars remain the local-dev fallback, with optional `DB_SSLMODE` for managed Postgres.
- ✅ **0.4** Added `DELPHI_MODEL_PATH` override in `model_loader.py` (`MLPredictor`) so a boot-downloaded
  model can live anywhere on the container's ephemeral disk; falls back to bundled artifacts for local dev.

### Verification (all passed)
- Local `DB_*` path resolves + real `SELECT 1` connection succeeds.
- `DATABASE_URL` takes priority and is passed verbatim to the pool.
- Missing config raises a clean `EnvironmentError` (no more `FileNotFoundError`).
- Default model path still loads bundled model; `DELPHI_MODEL_PATH` override is honored.

## Phase A — Containerize backend (+ wire Vercel)
- ⬜ **A.1** `Dockerfile` (root): `python:3.12-slim`, install `requirements.txt`, copy `DelphiAIApp/`,
  run `uvicorn DelphiAIApp.main:app --host 0.0.0.0 --port $PORT`.
- ⬜ **A.2** `.dockerignore`: exclude `node_modules`, `Views/`, `__pycache__`, `*.pkl`, scraper logs, `.git`.
- ⬜ **A.3** `railway.toml` (or Render blueprint): Docker build, `$PORT`, env vars, release command = migrations.
- ⬜ **A.4** Vercel: project root = `DelphiAIApp/Views`; env `FASTAPI_URL`, `DELPHI_API_KEY`, `AUTH_SECRET`,
  Neon **pooled** `DATABASE_URL`.
- ⬜ **A.5** Keep `docker-compose.yml` as local-dev only (Postgres + pgAdmin).

## Phase B — Model delivery via R2
- ⬜ **B.1** Boot-time fetch: pull `model_latest.pkl` from R2 if `MODEL_URL` set and local copy missing/stale.
- ⬜ **B.2** `scripts/publish_model.py`: upload `model_latest.pkl` to R2 after local retraining.
- ⬜ **B.3** Add `MODEL_URL`/R2 creds to Railway env. Model never enters the repo.

## Phase C — Migrations (yoyo-migrations)
- ⬜ **C.1** Add `yoyo-migrations`; create `migrations/`.
- ⬜ **C.2** Baseline `0001_initial.sql` = current `db/schemas.sql` (replaces lost `docker-entrypoint-initdb`).
- ⬜ **C.3** Runner: Railway release command `yoyo apply --database $DATABASE_URL`; manual for local; guard step in cron.
- ⬜ **C.4** Going forward: every schema change is a new numbered file; stop editing `schemas.sql` in place.

## Phase D — Scraper cron
- ⬜ **D.1** `.github/workflows/weekly_update.yml`: scheduled `setup-python`, install deps,
  `DATABASE_URL` secret, run migrations then `python -m ml.weekly_update`.

---

## Env var matrix

| Var | Railway (API) | Vercel (web) | GH Actions |
|---|---|---|---|
| `DATABASE_URL` | ✅ pooled | ✅ pooled | ✅ direct |
| `DELPHI_API_KEY` | ✅ enforce | ✅ sends it | — |
| `DELPHI_CORS_ORIGINS` | ✅ Vercel domain | — | — |
| `MODEL_URL` / R2 creds | ✅ | — | — |
| `AUTH_SECRET` / `AUTH_URL` | — | ✅ | — |
| `FASTAPI_URL` | — | ✅ | — |

## Suggested sequence
Phase 0 → C → B → A → D. Get config/migrations correct against a real Neon DB first, then model
delivery, then containerize, then automate scraping. Each phase is independently testable.

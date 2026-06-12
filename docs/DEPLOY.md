# DelphiAI Deployment Checklist

Topology: **Vercel (Next.js)** → **Railway (FastAPI container)** → **Neon (Postgres)**,
with the model in **Cloudflare R2**. See `docs/deployment-plan.md` for rationale.

> **Boot requirement:** the API opens its Postgres connection pool at startup
> (FastAPI lifespan), so it needs a **reachable** `DATABASE_URL` to boot — a
> missing/unreachable DB crashes the container. Railway → Neon satisfies this.

## 1. Neon (database)
1. Create a project; copy the **pooled** connection string.
2. It already carries `sslmode=require` — keep it.

## 2. Railway (API)
1. New project → Deploy from repo. Railway reads `railway.toml` (Docker build).
2. Set service variables:
   | Var | Value |
   |---|---|
   | `DATABASE_URL` | Neon pooled string (required at boot) |
   | `DELPHI_API_KEY` | a long random secret |
   | `DELPHI_ENV` | `production` (forces API-key enforcement) |
   | `DELPHI_CORS_ORIGINS` | the Vercel app URL |
   | `MODEL_URL` | public/presigned R2 URL to `model_latest.pkl` |
   | `DELPHI_MODEL_PATH` | `/tmp/model_latest.pkl` |
3. The release command runs `yoyo apply` against `DATABASE_URL` automatically
   (applies `0001_initial` on a fresh Neon DB, then any later migrations).

## 3. Cloudflare R2 (model storage)
1. Create a bucket (e.g. `delphi-models`).
2. Create an API token with object read/write; locally set
   `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`.
3. After each retrain: `pip install -r requirements-dev.txt && python scripts/publish_model.py`.
4. Make the object reachable from Railway via a public bucket URL or a presigned
   URL, and set that as `MODEL_URL`.

## 4. Vercel (web)
1. New project → import repo → **Root Directory = `DelphiAIApp/Views`** (Next.js auto-detected).
2. Set env vars:
   | Var | Value |
   |---|---|
   | `FASTAPI_URL` | Railway API URL |
   | `NEXT_PUBLIC_FASTAPI_URL` | Railway API URL (client fetches) |
   | `DELPHI_API_KEY` | same secret as Railway |
   | `AUTH_SECRET` | a long random secret (next-auth) |
   | `DATABASE_URL` | Neon **pooled** string |

## 5. First deploy — verify
- Railway deploy logs: release ran migrations; `[model] OK (… bytes)`; uvicorn up.
- `curl https://<railway-app>/health` → `{"status":"ok","database":true}`.
- Vercel app loads and calls the API without CORS errors.

## Local parity (verified)
- `docker build -t delphi-api .` then run with a reachable DB:
  ```bash
  docker run --rm -p 8000:8000 \
    -e DATABASE_URL="postgresql://USER:PW@host.docker.internal:5433/delphi_db" \
    -e DELPHI_MODEL_PATH=/tmp/model_latest.pkl delphi-api
  curl localhost:8000/health   # -> {"status":"ok","database":true}
  ```
  (`host.docker.internal` reaches the local docker-compose Postgres on 5433.)
- Apply migrations locally:
  `yoyo apply --batch --database "<local DATABASE_URL>" DelphiAIApp/Models/migrations`.

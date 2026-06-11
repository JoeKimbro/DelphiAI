# Deployment Infrastructure Implementation Plan (Phases C → B → A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DelphiAI deployable to Railway (FastAPI container) + Neon (Postgres) + Cloudflare R2 (model storage) by adding versioned migrations, boot-time model delivery, and a light production container — all locally verifiable before any cloud account exists.

**Architecture:** A `yoyo-migrations` runner applies a single baseline (the current `schemas.sql` + `schemas_auth.sql`) plus future numbered migrations against `$DATABASE_URL`; the pre-yoyo `db/migrations/00X` deltas are retired to history. The model never enters the repo or image — `scripts/fetch_model.py` pulls `model_latest.pkl` from an HTTPS `MODEL_URL` at container boot (plain `requests`, no S3 SDK), while `scripts/publish_model.py` uploads it to R2 after local retraining (lazy `boto3`). A `python:3.12-slim` Dockerfile installs a **light** `requirements-api.txt` (no Scrapy/Playwright), runs migrations as Railway's release command, and starts uvicorn via `scripts/entrypoint.sh`.

**Tech Stack:** Python 3.12, FastAPI/uvicorn, psycopg2-binary, yoyo-migrations, requests, boto3 (dev-only), Docker, Railway, Cloudflare R2, Vercel, PostgreSQL (Neon).

**Follows the deployment plan's recommended sequence (`docs/deployment-plan.md`): Phase 0 ✅ → C → B → A → D.** (Phase D, the GitHub Actions scraper cron, is out of scope for this plan.)

---

## File Structure

**Created:**
- `yoyo.ini` — yoyo config (sources dir, batch mode; DB supplied on CLI)
- `DelphiAIApp/Models/migrations/0001_initial.py` — baseline migration (full current schema)
- `DelphiAIApp/Models/db/migrations/README.md` — marks `001`–`006` as retired pre-yoyo history
- `scripts/fetch_model.py` — boot-time model download from `MODEL_URL`
- `scripts/publish_model.py` — upload model to R2 after retraining
- `DelphiAIApp/Models/tests/test_fetch_model.py`
- `DelphiAIApp/Models/tests/test_publish_model.py`
- `requirements-api.txt` — light container deps (no Scrapy/Playwright)
- `requirements-dev.txt` — dev-only deps (boto3 for publishing)
- `scripts/entrypoint.sh` — fetch model → exec uvicorn
- `Dockerfile` — `python:3.12-slim` API image
- `.dockerignore`
- `railway.toml` — Docker build, release = migrations, start = entrypoint
- `docs/DEPLOY.md` — account-provisioning + env-var checklist (Neon/Railway/Vercel/R2)

**Modified:**
- `requirements.txt` — add `yoyo-migrations`
- `docs/deployment-plan.md` — flip Phase A/B/C statuses to ✅ as completed

---

## Phase C — Migrations (yoyo-migrations)

### Task 1: Add yoyo-migrations + config + migrations directory

**Files:**
- Modify: `requirements.txt`
- Create: `yoyo.ini`
- Create: `DelphiAIApp/Models/migrations/` (directory)

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt` under the `# --- Database ---` section (after `python-dotenv`):
```
yoyo-migrations==8.2.0
```

- [ ] **Step 2: Install it and confirm the CLI exists**

Run (from repo root):
```bash
pip install "yoyo-migrations==8.2.0"
yoyo --version
```
Expected: a version string prints (e.g. `8.2.0`). If pip cannot find `8.2.0`, use the newest `8.x` it offers and update the pin in `requirements.txt` to match.

- [ ] **Step 3: Create `yoyo.ini`**

`yoyo.ini` (repo root):
```ini
[DEFAULT]
# Migration sources. The database is supplied on the CLI
# (--database "$DATABASE_URL") so no secret lives in the repo.
sources = DelphiAIApp/Models/migrations
# Non-interactive: apply all pending migrations without prompting.
batch_mode = on
# Record applied migrations in a dedicated table.
migration_table = _yoyo_migration
```

- [ ] **Step 4: Create the migrations directory with a marker**

```bash
mkdir -p DelphiAIApp/Models/migrations
```
Create `DelphiAIApp/Models/migrations/__init__.py` is **not** needed (yoyo loads files directly), but create a placeholder so the empty dir is tracked until Task 2 adds the baseline:
```bash
touch DelphiAIApp/Models/migrations/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt yoyo.ini DelphiAIApp/Models/migrations/.gitkeep
git commit -m "feat(deploy): add yoyo-migrations runner + config (Phase C.1)"
```

### Task 2: Baseline migration = full current schema

**Files:**
- Create: `DelphiAIApp/Models/migrations/0001_initial.py`
- Delete: `DelphiAIApp/Models/migrations/.gitkeep`

**Why Python, not raw `.sql`:** `schemas.sql` has no dollar-quoted functions, but routing both schema files through one `cursor.execute` per file (psycopg2 runs multi-statement strings) avoids any fragile statement-splitting and keeps the baseline a single source of truth.

- [ ] **Step 1: Write the baseline migration**

`DelphiAIApp/Models/migrations/0001_initial.py`:
```python
"""yoyo baseline: the full current schema (schemas.sql + schemas_auth.sql).

The pre-yoyo deltas in db/migrations/001-006 are already folded into these two
files, so this single baseline reproduces a complete, current database on a
fresh server (Neon/Railway). Executed as a Python step (not raw .sql) so
psycopg2 runs each file's statements in one shot — no statement-splitting.

Forward rule: never edit this file or schemas.sql for new changes. Add a new
numbered migration (0002_*.py / 0002_*.sql) instead.
"""
from pathlib import Path

from yoyo import step

_DB = Path(__file__).resolve().parents[1] / "db"


def apply_step(conn):
    cur = conn.cursor()
    for fname in ("schemas.sql", "schemas_auth.sql"):
        cur.execute((_DB / fname).read_text(encoding="utf-8"))


steps = [step(apply_step)]
```

- [ ] **Step 2: Remove the placeholder**

```bash
rm DelphiAIApp/Models/migrations/.gitkeep
```

- [ ] **Step 3: Sanity-check the file imports**

Run (from repo root):
```bash
python -c "import importlib.util, pathlib; \
p='DelphiAIApp/Models/migrations/0001_initial.py'; \
s=importlib.util.spec_from_file_location('m', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); \
print('steps:', len(m.steps))"
```
Expected: `steps: 1` (and no ImportError — confirms `yoyo` is importable and the module is well-formed).

- [ ] **Step 4: Commit**

```bash
git add DelphiAIApp/Models/migrations/0001_initial.py
git rm --cached DelphiAIApp/Models/migrations/.gitkeep 2>/dev/null || true
git commit -m "feat(deploy): yoyo baseline migration = current full schema (Phase C.2)"
```

### Task 3: Verify against a fresh database + retire the old deltas

**Files:**
- Create: `DelphiAIApp/Models/db/migrations/README.md`

This task proves the baseline applies cleanly to an **empty** database using the local docker Postgres (`delphi_postgres`, port 5433) as a scratch host — no cloud account required. Supply your local `DB_USER` / `DB_PASSWORD` from `.env`.

- [ ] **Step 1: Create a throwaway database**

Run (from repo root; substitute your local superuser into `$DB_USER`):
```bash
docker exec delphi_postgres psql -U "$DB_USER" -c "DROP DATABASE IF EXISTS delphi_migrate_test;"
docker exec delphi_postgres psql -U "$DB_USER" -c "CREATE DATABASE delphi_migrate_test;"
```
Expected: `CREATE DATABASE`.

- [ ] **Step 2: Apply migrations to the empty DB**

```bash
yoyo apply --batch --database "postgresql://$DB_USER:$DB_PASSWORD@localhost:5433/delphi_migrate_test" DelphiAIApp/Models/migrations
```
Expected: applies `0001_initial` with no error.

- [ ] **Step 3: Verify the tables exist**

```bash
docker exec delphi_postgres psql -U "$DB_USER" -d delphi_migrate_test -c "\dt" | grep -i "fights\|users\|auth_attempts"
```
Expected: core tables (`Fights`, `users`, `auth_attempts`, …) are listed. Also confirm idempotency — a second apply is a no-op:
```bash
yoyo apply --batch --database "postgresql://$DB_USER:$DB_PASSWORD@localhost:5433/delphi_migrate_test" DelphiAIApp/Models/migrations
```
Expected: `0 migrations to apply` (yoyo records `0001_initial` as done).

- [ ] **Step 4: Drop the scratch DB**

```bash
docker exec delphi_postgres psql -U "$DB_USER" -c "DROP DATABASE delphi_migrate_test;"
```

- [ ] **Step 5: Retire the pre-yoyo deltas with a README**

`DelphiAIApp/Models/db/migrations/README.md`:
```markdown
# Retired pre-yoyo migrations

The numbered SQL files in this directory (`001`–`006`) are **historical**
deltas that were hand-applied before yoyo-migrations existed. They are **already
folded into `../schemas.sql` / `../schemas_auth.sql`**, which the yoyo baseline
(`../../migrations/0001_initial.py`) reproduces in full.

**Do not run these against a fresh database** — the baseline already creates
everything they added. They are kept only as a change log.

Going forward, every schema change is a **new** numbered migration in
`DelphiAIApp/Models/migrations/` (e.g. `0002_*.py`). Stop editing `schemas.sql`
in place.
```

- [ ] **Step 6: Commit**

```bash
git add DelphiAIApp/Models/db/migrations/README.md
git commit -m "docs(deploy): retire pre-yoyo deltas; verify baseline on fresh DB (Phase C.3/C.4)"
```

---

## Phase B — Model delivery via R2

### Task 4: Boot-time model fetch (`scripts/fetch_model.py`)

**Files:**
- Create: `scripts/fetch_model.py`
- Test: `DelphiAIApp/Models/tests/test_fetch_model.py`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Models/tests/test_fetch_model.py`:
```python
"""Unit tests for the boot-time model fetcher's pure helpers."""
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fetch_model import resolve_target, should_fetch


def test_should_fetch_when_missing(tmp_path):
    assert should_fetch(tmp_path / "nope.pkl", 168) is True


def test_no_fetch_when_fresh(tmp_path):
    f = tmp_path / "m.pkl"
    f.write_bytes(b"x")
    assert should_fetch(f, 168) is False


def test_fetch_when_stale(tmp_path):
    f = tmp_path / "m.pkl"
    f.write_bytes(b"x")
    old = time.time() - 200 * 3600  # ~8.3 days old
    os.utime(f, (old, old))
    assert should_fetch(f, 168) is True


def test_resolve_target_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DELPHI_MODEL_PATH", str(tmp_path / "x.pkl"))
    assert resolve_target() == tmp_path / "x.pkl"
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_fetch_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_model'`.

- [ ] **Step 3: Implement `scripts/fetch_model.py`**

`scripts/fetch_model.py`:
```python
"""Boot-time model fetch from object storage (Cloudflare R2 / any HTTPS URL).

If MODEL_URL is set and the local model file is missing or stale, download it to
the path MLPredictor loads (DELPHI_MODEL_PATH, else the bundled artifacts
location). Uses a plain HTTPS GET (requests) so the API container needs no S3
SDK; publishing (which needs credentials) lives in publish_model.py.

Usage:
    python scripts/fetch_model.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_TARGET = (
    Path(__file__).resolve().parents[1]
    / "DelphiAIApp" / "Models" / "ml" / "artifacts" / "model_latest.pkl"
)
MAX_AGE_HOURS = float(os.getenv("MODEL_MAX_AGE_HOURS", "168"))  # 7 days


def resolve_target() -> Path:
    """Where the model should land — DELPHI_MODEL_PATH or the bundled location."""
    env_path = os.getenv("DELPHI_MODEL_PATH", "").strip()
    return Path(env_path) if env_path else DEFAULT_TARGET


def should_fetch(target: Path, max_age_hours: float) -> bool:
    """True if the target is missing or older than max_age_hours."""
    if not target.exists():
        return True
    age_hours = (time.time() - target.stat().st_mtime) / 3600
    return age_hours > max_age_hours


def download(url: str, target: Path) -> None:
    """Stream the model to a temp file, then atomically rename into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(target)


def main() -> int:
    url = os.getenv("MODEL_URL", "").strip()
    if not url:
        print("[model] MODEL_URL unset — using bundled/local model.", flush=True)
        return 0
    target = resolve_target()
    if not should_fetch(target, MAX_AGE_HOURS):
        print(f"[model] {target} is fresh — skipping download.", flush=True)
        return 0
    print(f"[model] downloading {url} -> {target} ...", flush=True)
    try:
        download(url, target)
    except Exception as e:  # noqa: BLE001 — boot must not crash on a transient pull
        print(f"[model] download failed: {e}", file=sys.stderr)
        return 1
    print(f"[model] OK ({target.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_fetch_model.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_model.py DelphiAIApp/Models/tests/test_fetch_model.py
git commit -m "feat(deploy): boot-time model fetch from MODEL_URL (Phase B.1)"
```

### Task 5: Model publisher (`scripts/publish_model.py`)

**Files:**
- Create: `scripts/publish_model.py`
- Create: `requirements-dev.txt`
- Test: `DelphiAIApp/Models/tests/test_publish_model.py`

- [ ] **Step 1: Write the failing test**

`DelphiAIApp/Models/tests/test_publish_model.py`:
```python
"""Unit tests for the R2 publisher's credential/kwarg builder (no network)."""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.publish_model import build_client_kwargs


def test_build_client_kwargs_from_env(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "models")
    kw = build_client_kwargs()
    assert kw["service_name"] == "s3"
    assert kw["endpoint_url"].endswith("cloudflarestorage.com")
    assert kw["region_name"] == "auto"


def test_build_client_kwargs_missing_raises(monkeypatch):
    for k in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(EnvironmentError):
        build_client_kwargs()
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_publish_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.publish_model'`.

- [ ] **Step 3: Implement `scripts/publish_model.py`**

`scripts/publish_model.py`:
```python
"""Upload the freshly retrained model to Cloudflare R2 (S3-compatible).

Run locally AFTER retraining. Needs R2 credentials (which NEVER enter the
container): R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.
boto3 is imported lazily so the API image stays free of the S3 SDK.

Usage:
    pip install -r requirements-dev.txt   # one-time, gets boto3
    python scripts/publish_model.py [path/to/model.pkl]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "DelphiAIApp" / "Models" / "ml" / "artifacts" / "model_latest.pkl"
)
OBJECT_KEY = os.getenv("R2_MODEL_KEY", "model_latest.pkl")
_REQUIRED = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


def build_client_kwargs() -> dict:
    """boto3 S3-client kwargs for R2 from env; raises if any var is missing."""
    missing = [k for k in _REQUIRED if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing R2 env vars: {', '.join(missing)}")
    return {
        "service_name": "s3",
        "endpoint_url": os.environ["R2_ENDPOINT_URL"],
        "aws_access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "region_name": "auto",
    }


def main() -> int:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    if not model_path.exists():
        print(f"[publish] no model at {model_path}", file=sys.stderr)
        return 1
    import boto3  # lazy: only the publisher needs the SDK

    client = boto3.client(**build_client_kwargs())
    bucket = os.environ["R2_BUCKET"]
    print(f"[publish] uploading {model_path} -> r2://{bucket}/{OBJECT_KEY} ...", flush=True)
    client.upload_file(str(model_path), bucket, OBJECT_KEY)
    print("[publish] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Create `requirements-dev.txt`**

`requirements-dev.txt` (repo root):
```
# Dev-only tooling, never installed into the API container.
# boto3 powers scripts/publish_model.py (uploads the model to R2).
-r requirements.txt
boto3==1.35.99
```
> If pip cannot resolve `boto3==1.35.99`, drop the pin to `boto3` (latest) — this file is dev-only and never built into the image.

- [ ] **Step 5: Run test to verify it passes**

Run (from `DelphiAIApp/Models`): `python -m pytest tests/test_publish_model.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/publish_model.py requirements-dev.txt DelphiAIApp/Models/tests/test_publish_model.py
git commit -m "feat(deploy): R2 model publisher + dev requirements (Phase B.2)"
```

---

## Phase A — Containerize backend + wire Vercel

### Task 6: Light container requirements

**Files:**
- Create: `requirements-api.txt`

The API image must **not** carry Scrapy/scrapy-playwright (heavy, browser-dependent — those run in the Actions cron). `requirements-api.txt` is the runtime subset plus `yoyo-migrations` (needed for the Railway release command, which runs inside the image).

- [ ] **Step 1: Create `requirements-api.txt`**

`requirements-api.txt` (repo root):
```
# Light production API image deps (Linux container).
# Excludes Scrapy/scrapy-playwright — bulk scraping runs in the Actions cron,
# not in the API container. Keep in sync with requirements.txt for shared pins.

# --- Web API ---
fastapi==0.136.1
uvicorn[standard]==0.46.0
starlette==1.0.0
pydantic==2.13.4

# --- Database + migrations ---
psycopg2-binary==2.9.11
python-dotenv==1.2.1
yoyo-migrations==8.2.0

# --- ML / prediction pipeline ---
xgboost==3.1.3
scikit-learn==1.8.0
numpy==2.4.2
pandas==3.0.0
joblib==1.5.3

# --- Live predict-time injury scrape (lightweight; no browser) ---
requests==2.32.5
beautifulsoup4==4.14.3
lxml==6.0.2
```
> Keep the `yoyo-migrations` pin identical to whatever Task 1 settled on.

- [ ] **Step 2: Verify it installs cleanly in isolation**

Run (from repo root):
```bash
python -m venv /tmp/apienv && /tmp/apienv/bin/pip install -r requirements-api.txt && echo "OK"
```
Expected: ends with `OK`, no resolver errors. (On Windows use `python -m venv $env:TEMP\apienv; & $env:TEMP\apienv\Scripts\pip install -r requirements-api.txt`.) Remove the venv afterward.

- [ ] **Step 3: Commit**

```bash
git add requirements-api.txt
git commit -m "feat(deploy): light API container requirements (no Scrapy/Playwright)"
```

### Task 7: Dockerfile + entrypoint + .dockerignore

**Files:**
- Create: `scripts/entrypoint.sh`
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create the entrypoint**

`scripts/entrypoint.sh`:
```sh
#!/usr/bin/env sh
set -e
# Pull the model from R2 if MODEL_URL is set; a failure here must not stop boot
# (e.g. local runs with no MODEL_URL — fetch_model exits 0 and is a no-op).
python scripts/fetch_model.py || echo "[entrypoint] model fetch skipped/failed; continuing"
exec uvicorn DelphiAIApp.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

- [ ] **Step 2: Create the Dockerfile**

`Dockerfile` (repo root):
```dockerfile
# Light FastAPI image. The model arrives from R2 at boot (not baked in); bulk
# scraping (Scrapy/Playwright) runs in CI, not here.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install deps first for layer caching (psycopg2-binary needs no build toolchain).
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# App code, migrations, and boot scripts (.dockerignore strips Views/, *.pkl, etc.).
COPY DelphiAIApp ./DelphiAIApp
COPY scripts ./scripts
COPY yoyo.ini ./

EXPOSE 8000
CMD ["sh", "scripts/entrypoint.sh"]
```

- [ ] **Step 3: Create `.dockerignore`**

`.dockerignore` (repo root):
```
.git
.gitignore
.env
.venv/
**/__pycache__/
**/*.pyc
# Frontend is deployed separately on Vercel.
DelphiAIApp/Views/
**/node_modules/
# Model is pulled from R2 at boot — never baked into the image.
*.pkl
*.pkl.disabled
DelphiAIApp/Models/ml/artifacts/
# Bulk data / logs / local backups have no place in the image.
DelphiAIApp/Models/data/output/
**/logs/
backups/
*.csv
docs/
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore scripts/entrypoint.sh
git commit -m "feat(deploy): light Dockerfile + entrypoint + dockerignore (Phase A.1/A.2)"
```

### Task 8: Build + smoke-test the image locally, add `railway.toml`

**Files:**
- Create: `railway.toml`

`/health` swallows DB errors and returns HTTP 200 with `{"database": false}` when no DB is reachable, so a boot smoke test needs **no** database — a 200 proves the image starts and serves.

- [ ] **Step 1: Build the image**

Run (from repo root):
```bash
docker build -t delphi-api .
```
Expected: build completes; final line `naming to docker.io/library/delphi-api`.

- [ ] **Step 2: Run it and smoke-test `/health`**

```bash
docker run -d --rm -e DELPHI_MODEL_PATH=/tmp/model_latest.pkl -e PORT=8000 -p 8000:8000 --name delphi-api-test delphi-api
# give uvicorn a moment to bind, then:
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
docker logs delphi-api-test | tail -20
docker stop delphi-api-test
```
Expected: `curl` prints `200`; logs show `[model] MODEL_URL unset — using bundled/local model.` and `Uvicorn running on http://0.0.0.0:8000`.

- [ ] **Step 3: Create `railway.toml`**

`railway.toml` (repo root):
```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
# Apply pending migrations once per deploy, before the new version takes traffic.
releaseCommand = "yoyo apply --batch --database \"$DATABASE_URL\" DelphiAIApp/Models/migrations"
startCommand = "sh scripts/entrypoint.sh"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

- [ ] **Step 4: Commit**

```bash
git add railway.toml
git commit -m "feat(deploy): Railway config (Docker build, release=migrations, start=uvicorn) (Phase A.3)"
```

### Task 9: Deploy checklist doc + flip plan statuses

**Files:**
- Create: `docs/DEPLOY.md`
- Modify: `docs/deployment-plan.md`

Cloud provisioning (creating Neon/Railway/Vercel/R2 and setting secrets) is inherently manual; this task captures it as a checklist instead of code so the wiring is reproducible.

- [ ] **Step 1: Write `docs/DEPLOY.md`**

`docs/DEPLOY.md`:
```markdown
# DelphiAI Deployment Checklist

Topology: **Vercel (Next.js)** → **Railway (FastAPI container)** → **Neon (Postgres)**,
with the model in **Cloudflare R2**. See `docs/deployment-plan.md` for rationale.

## 1. Neon (database)
1. Create a project; copy the **pooled** connection string.
2. It already carries `sslmode=require` — keep it.

## 2. Railway (API)
1. New project → Deploy from repo. Railway reads `railway.toml` (Docker build).
2. Set service variables:
   | Var | Value |
   |---|---|
   | `DATABASE_URL` | Neon pooled string |
   | `DELPHI_API_KEY` | a long random secret |
   | `DELPHI_ENV` | `production` (forces API-key enforcement) |
   | `DELPHI_CORS_ORIGINS` | the Vercel app URL |
   | `MODEL_URL` | public/presigned R2 URL to `model_latest.pkl` |
   | `DELPHI_MODEL_PATH` | `/tmp/model_latest.pkl` |
3. The release command runs `yoyo apply` against `DATABASE_URL` automatically.

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

## Local parity
- `docker build -t delphi-api . && docker run --rm -p 8000:8000 delphi-api` then
  `curl localhost:8000/health` → 200 (`degraded` without a DB is expected).
- Apply migrations locally: `yoyo apply --batch --database "<local DATABASE_URL>" DelphiAIApp/Models/migrations`.
```

- [ ] **Step 2: Flip statuses in `docs/deployment-plan.md`**

In `docs/deployment-plan.md`, change the Phase A/B/C headers and their sub-items from `⬜` to `✅` where this plan implemented them. Specifically:
- Phase A: A.1, A.2, A.3 → ✅ (A.4 Vercel = documented in `docs/DEPLOY.md`; A.5 docker-compose unchanged → mark ✅).
- Phase B: B.1, B.2 → ✅ (B.3 = env wiring, documented in `docs/DEPLOY.md`).
- Phase C: C.1, C.2, C.3, C.4 → ✅.
Leave **Phase D** (scraper cron) as ⬜ — out of scope for this plan.

- [ ] **Step 3: Commit**

```bash
git add docs/DEPLOY.md docs/deployment-plan.md
git commit -m "docs(deploy): provisioning checklist + mark Phases A/B/C done"
```

---

## Final verification

- [ ] **Migrations apply to an empty DB:** Task 3 — `yoyo apply` against a fresh scratch DB creates all tables; a second apply is a no-op.
- [ ] **Python suites pass:** from `DelphiAIApp/Models`, run `python -m pytest tests/test_fetch_model.py tests/test_publish_model.py -v` → all pass.
- [ ] **Image builds + boots:** `docker build -t delphi-api .` succeeds; `curl localhost:8000/health` → 200.
- [ ] **Light image confirmed:** `docker run --rm delphi-api pip freeze | grep -i "scrapy\|playwright"` prints **nothing**.
- [ ] **Config parses:** `python -c "import tomllib; tomllib.load(open('railway.toml','rb')); print('ok')"` → `ok`.
- [ ] **No secrets/model in repo or image:** `git status` clean; `.dockerignore` excludes `*.pkl` and `.env`.
- [ ] **Local dev unaffected:** existing `python -m pytest ml/tests/` still passes; `docker-compose up -d` still serves local Postgres on 5433.
```

# Running DelphiAI From a Clone

This guide gets DelphiAI running on your own machine: a local database, the prediction model, the API, and the dashboard.

**What's in the repo:**
- **Fighter and fight data** as CSV snapshots in `DelphiAIApp/Models/data/output/` (fighters, fights, ELO history, point-in-time stats). You load these into your own database in step 4. The newest date in `fights.csv` is how current the snapshot is.
- **Database schema and migrations.**

**What's not in the repo:**
- **The trained model** (`model_latest.pkl`). Download it from the latest GitHub Release, or train your own (step 5).
- **The maintainer's prediction history and user accounts.** Your database starts with no tracked predictions.
- **Secrets.** You make your own `.env` from `.env.example`.

> Never point your setup at the production database. Everything below runs against a Postgres instance on your machine.

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.12 |
| Node.js | 22+ |
| Docker Desktop | any recent version (runs PostgreSQL) |
| Git | any |

Commands below are for bash. On Windows PowerShell, use `copy` instead of `cp`.

## 1. Clone and configure

```bash
git clone https://github.com/JoeKimbro/DelphiAI.git
cd DelphiAI
cp .env.example .env
```

The defaults in `.env` already match the Docker database. You don't need to change anything to run locally.

## 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Start the database and finish the schema

```bash
docker-compose up -d
```

On its first start, the container creates `delphi_db` and runs `DelphiAIApp/Models/db/schemas.sql`. Two more pieces are needed: the auth tables (used for dashboard sign-in) and the numbered migrations.

```bash
# Auth tables
docker exec -i delphi_postgres psql -U delphi_user -d delphi_db < DelphiAIApp/Models/db/schemas_auth.sql

# The baseline migration is what Docker just did, so mark it applied, then apply the rest
yoyo mark  --database postgresql://delphi_user:delphi_password@localhost:5433/delphi_db -r 0001_initial
yoyo apply --database postgresql://delphi_user:delphi_password@localhost:5433/delphi_db
```

pgAdmin is available at http://localhost:5050 (`admin@delphi.local` / `admin123`). Inside pgAdmin, the database host is `postgres` and the port is `5432`.

## 4. Load the data

```bash
cd DelphiAIApp/Models/data
python validate_data.py          # optional quality check
python load_to_db.py

cd ..                             # now in DelphiAIApp/Models
python -m ml.populate_styles      # classify fighter styles
python -m ml.update_adjusted_elos # ring-rust / injury ELO adjustments
```

## 5. Get the model

Pick one option. Either way, the model ends up at `DelphiAIApp/Models/ml/artifacts/model_latest.pkl`.

**Option A: download it (fastest).**
1. From the repo's latest **GitHub Release**, download `model_latest.pkl` and `model_latest.pkl.sha256`.
2. Put both files in `DelphiAIApp/Models/ml/artifacts/`.

The loader checks the file against the `.sha256` file before loading it. The model is a pickle, and loading a pickle runs code, so only use a model file from the official Release.

**Option B: train your own.** This trains on the data you loaded in step 4 and writes `model_latest.pkl`:

```bash
# from DelphiAIApp/Models
python -m ml.train_model_v3
```

Check that it works:

```bash
python -m ml.predict_fight "Islam Makhachev" "Charles Oliveira"
```

## 6. Run the API

From the repo root:

```bash
uvicorn DelphiAIApp.main:app --reload --port 8000
```

Check it at http://localhost:8000/health, which should return `{"status":"ok","database":true}`. With no `DELPHI_API_KEY` set, the API accepts requests without a key, which is fine on localhost.

## 7. Run the dashboard

```bash
cd DelphiAIApp/Views
cp .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000 and create an account at `/auth/signup`. Accounts are stored in your local database.

## 8. Run the tests

```bash
# Python, from DelphiAIApp/Models
python -m pytest ml/tests/

# Frontend, from DelphiAIApp/Views
npm test
```

## Optional: refresh the data yourself

The CSV snapshot is enough to run everything. To scrape fresh data you need a headless browser:

```bash
python -m playwright install chromium

cd DelphiAIApp/Models/data/scrapers
python scrape_all.py --test      # small trial run
python scrape_all.py             # full scrape: several hours
```

Then reload the data (step 4). See `CLI_COMMANDS.md` and `WEEKLY_PIPELINE.md` for the full pipeline.

## Troubleshooting

| Problem | Fix |
|---|---|
| Can't connect on port 5432 | The database is published on **5433**, not 5432. |
| Schema errors after changing `.env` | Docker only runs the schema on the very first start. Reset with `docker-compose down -v` (this deletes local DB data), then repeat step 3. |
| `No model found at ...model_latest.pkl` | Do step 5. |
| `Model integrity check failed` | The `.sha256` file doesn't match the model. Download both files again from the same Release. |
| Dashboard sign-in fails | Check that `DelphiAIApp/Views/.env.local` exists and that step 3's auth tables were created. |

## Further reading

- `README.md`: project overview and betting strategy
- `CLI_COMMANDS.md`: every command
- `WEEKLY_PIPELINE.md`: the weekly scrape → load → predict pipeline
- `docs/DEPLOY.md`: production deployment (Railway, Vercel, Neon, R2)

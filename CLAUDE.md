# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DelphiAI is a UFC fight prediction system that blends XGBoost machine learning with an ELO rating system. It scrapes data from UFC.com and UFCStats, stores it in PostgreSQL, and exposes predictions via a FastAPI backend.

## Environment Setup

```bash
# Start PostgreSQL + pgAdmin
docker-compose up -d

# Install dependencies
pip install -r requirements.txt
```

Database connection is configured via `.env` (port 5433, not the default 5432).

## Common Commands

All ML commands are run from `DelphiAIApp/Models/`:

```bash
# Predict a full event
python -m ml.predict_card "UFC 326"

# Predict a single fight
python -m ml.predict_fight "Fighter A" "Fighter B"

# Compare predictions vs actual results post-event
python -m ml.update_results "UFC 326"

# Aggregate performance metrics
python -m ml.performance_summary

# Historical backtesting
python -m ml.backtest --year 2025

# Retrain model
python -m ml.train_model_v3
```

Data pipeline commands (from `DelphiAIApp/Models/data/scrapers/`):

```bash
python scrape_all.py           # Scrape UFC.com + UFCStats
python scrape_all.py --test    # Limited scrape for testing
```

Data loading (from `DelphiAIApp/Models/data/`):

```bash
python validate_data.py        # Check data quality
python load_to_db.py           # Load CSV output to PostgreSQL
python load_to_db.py --clear   # Reset tables before loading
```

ELO/feature updates (from `DelphiAIApp/Models/`):

```bash
python -m ml.update_adjusted_elos   # Recalculate ring rust + injury ELO
python -m ml.populate_styles        # Classify fighter styles
python -m ml.weekly_update          # Full end-to-end pipeline
```

## Architecture

### Layer Structure

```
Scrapy (UFC.com + UFCStats)
    → CSV output files (Models/data/output/)
        → PostgreSQL (via load_to_db.py)
            → Prediction Pipeline
                → FastAPI endpoints (Controllers/)
                    → Dashboard UI (Views/)
```

### Core Components

**`Models/data/`** — Data ingestion layer
- `scrapers/ufc_scraper/` — Scrapy project with spiders for UFC.com and UFCStats
- `load_to_db.py` — Loads CSV exports into PostgreSQL
- `features.py` — ELO calculation logic
- `db/schemas.sql` — Full PostgreSQL schema (60+ tables)
- `db/postgres.py` — Connection pooling and cursor helpers

**`Models/ml/`** — Prediction engine
- `train_model_v3.py` — XGBoost training pipeline; trains on 4,138 fights (2019+), augmented to 8,276 by storing each fight in both orientations (A vs B, B vs A) to prevent positional bias
- `model_loader.py` — Feature engineering and model serving wrapper; central to the prediction pipeline
- `predict_fight.py` — Single fight CLI prediction
- `predict_card.py` — Full event prediction; also scrapes live injury data
- `backtest.py` — Validates on historical fights using point-in-time stats to prevent data leakage
- `update_results.py` / `performance_summary.py` — Post-event tracking and ROI reporting
- `artifacts/` — Saved model files and config

**`Controllers/`** — FastAPI route handlers (largely stubbed)
**`Services/`** — Business logic layer (largely stubbed)
**`Views/`** — Jinja2 templates and dashboard assets

### Prediction Pipeline (9 stages)

1. Fetch fighter career stats + ELO history from PostgreSQL
2. Apply ring rust decay (proportional + flat penalties for 6–24+ months inactive)
3. Apply injury ELO penalties (−17 to −55 ELO based on severity)
4. Engineer 20 differential features (physical, striking, grappling, career, stylistic, interaction terms)
5. Run calibrated XGBoost (350 trees, isotonic calibration)
6. Symmetry check — validate A→B and B→A are complementary
7. Blend 50% ML probability + 50% ELO probability
8. Apply final injury/ring-rust shifts
9. Cap output to [0.10, 0.90]; predict method of victory + round probabilities

### Key Design Decisions

- **Symmetry via augmentation**: Training data stores every fight twice (both fighter orderings) so the model has no positional bias
- **Isotonic calibration**: Custom `IsotonicCalibrator` maps XGBoost raw probabilities to real-world frequencies
- **Point-in-time stats**: Separate `PointInTimeStats` table snapshots stats at fight time, enabling leak-free backtesting
- **Style classification**: Wrestlers (TD avg > 2.5), Grapplers (Sub avg > 1.0), Strikers (SLpM > 4.0 + TD < 2.0), Balanced
- **Injury caching**: `predict_card.py` caches scraped injury data for 7 days

### Database

PostgreSQL runs on port **5433** (not the standard 5432). Connection details are in `.env`. Key tables include fighter stats, ELO history, matchup predictions, and point-in-time snapshots.

### Tests

Tests live in `Models/ml/tests/`. Run them from `DelphiAIApp/Models/`:

```bash
python -m pytest ml/tests/
```

## Reference Docs

- `InformationThatHelps/HowTheModelWorks.md` — Deep-dive on the full prediction pipeline, ELO formulas, feature rationale, and blending logic
- `CLI_COMMANDS.md` — User-facing command reference
- `README.md` — Project overview and betting strategy guidelines

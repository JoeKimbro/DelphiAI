# DelphiAI Command Line Reference

---

## Predict a Full Fight Card

Automatically finds all fights on a UFC event and runs predictions for each.
Predictions are **auto-saved to the database** for tracking.

```bash
cd DelphiAIApp/Models
python -m ml.predict_card "Event Name"
```

**Examples:**
```bash
cd DelphiAIApp/Models
python -m ml.predict_card "Strickland vs Hernandez"
python -m ml.predict_card "UFC 312"
python -m ml.predict_card "UFC 326"

# Scrape from a specific UFC.com event URL
python -m ml.predict_card --url "https://www.ufc.com/event/ufc-fight-night-february-21-2026"

# List upcoming UFC events
python -m ml.predict_card --upcoming

# Manual fight list (when event isn't in DB or on UFC.com)
python -m ml.predict_card --manual "Sean Strickland,Anthony Hernandez" "Geoff Neal,Uros Medic"
```

**Options:**
| Flag | Description |
|------|-------------|
| `--detail` or `-d` | Show full per-fight analysis |
| `--url` or `-u` | Scrape a specific UFC.com event page URL |
| `--upcoming` | List upcoming UFC events from UFC.com |
| `--manual` or `-m` | Manually specify fights as "Fighter1,Fighter2" pairs |
| `--refresh` or `-r` | Force fresh injury check from UFC.com (slower) |

**Output includes:**
- Card summary with picks, confidence, and predicted method
- Per-fight details (ELO, win %, method breakdown, round predictions)
- Betting insights (strongest picks, upset watch, finish vs decision likelihood)
- Auto-saves predictions to database for live tracking

---

## Update Results After an Event

After an event completes, fetch actual results and compare against stored predictions.

```bash
cd DelphiAIApp/Models
python -m ml.update_results "Event Name"
```

**Examples:**
```bash
cd DelphiAIApp/Models
python -m ml.update_results "Strickland vs Hernandez"

# Provide a specific UFC.com URL for result scraping
python -m ml.update_results "Strickland vs Hernandez" --url "https://www.ufc.com/event/ufc-fight-night-february-21-2026"

# List all tracked events and their status
python -m ml.update_results --list

# Re-fetch results even if already resolved
python -m ml.update_results "Strickland vs Hernandez" --force
```

**Options:**
| Flag | Description |
|------|-------------|
| `--url` or `-u` | UFC.com event URL for scraping results |
| `--list` or `-l` | List all tracked events with status |
| `--force` or `-f` | Re-fetch results even if already resolved |

**Output includes:**
- Per-fight comparison (predicted vs actual winner)
- Overall accuracy and confidence tier breakdown
- High confidence (65%+) accuracy
- Method accuracy and biggest misses
- Paper trading ROI

---

## Performance Summary

View aggregate performance with **separate backtest and live sections**.

```bash
cd DelphiAIApp/Models
python -m ml.performance_summary
```

**Examples:**
```bash
cd DelphiAIApp/Models

# Full report (backtest + live shown separately)
python -m ml.performance_summary

# Only backtest results
python -m ml.performance_summary --type backtest

# Only live (ongoing 2026) results
python -m ml.performance_summary --type live

# Last 5 events only
python -m ml.performance_summary --last 5

# Filter by event name
python -m ml.performance_summary --event "Strickland"
```

**Options:**
| Flag | Description |
|------|-------------|
| `--type` or `-t` | Show only `backtest` or `live` results |
| `--last` or `-n` | Only include the last N events |
| `--event` or `-e` | Filter by event name |

**Output includes:**
- **Backtest section**: Historical validation results (2023-2025)
- **Live section**: Ongoing 2026 predictions
- Backtest vs Live comparison
- Per-section: accuracy, confidence tiers, calibration, paper trading ROI
- Model status assessment and betting readiness

---

## Backtest on Historical Fights

Run predictions against past fights with known outcomes to validate the model.
Uses historical ELO and point-in-time stats to prevent data leakage.

```bash
cd DelphiAIApp/Models
python -m ml.backtest
```

**Examples:**
```bash
cd DelphiAIApp/Models

# Backtest all 2025 fights (default - true out-of-sample)
python -m ml.backtest

# Specific year
python -m ml.backtest --year 2024

# Multiple years with combined summary
python -m ml.backtest --years 2023 2024 2025

# Date range
python -m ml.backtest --start 2025-01-01 --end 2025-07-01

# Single event
python -m ml.backtest --event "UFC 311"

# Clear previous backtest data and re-run fresh
python -m ml.backtest --clear
```

**Options:**
| Flag | Description |
|------|-------------|
| `--year` or `-y` | Year to backtest (default: 2025) |
| `--years` | Multiple years (e.g. `--years 2023 2024 2025`) |
| `--start` | Start date (YYYY-MM-DD) |
| `--end` | End date (YYYY-MM-DD) |
| `--event` or `-e` | Filter to a specific event name |
| `--clear` or `-c` | Clear previous backtest data before running |

**Output includes:**
- Pre-flight data quality check (ELO/PIT stats coverage)
- Per-event accuracy as it runs
- Overall accuracy and confidence tier breakdown
- Calibration check and paper trading ROI
- Data leakage detection warnings
- Multi-year combined summary (when using `--years`)

---

## Weekly Workflow

```bash
cd DelphiAIApp/Models

# BEFORE the event (Wednesday/Thursday):
python -m ml.predict_card "UFC 326"
# Predictions auto-saved to database

# AFTER the event (Sunday):
python -m ml.update_results "UFC 326"
# Compares predictions vs actual results

# CHECK cumulative performance (anytime):
python -m ml.performance_summary
# Shows backtest vs live side-by-side
```

---

## Data Pipeline

### Weekly Update (After Saturday Fight Cards)
Runs everything in order: scrape, validate, load, ELO, styles, adjustments.

```bash
cd DelphiAIApp/Models
python -m ml.weekly_update
python -m ml.weekly_update --skip-scrape    # Already scraped, just recalculate
python -m ml.weekly_update --dry-run        # Preview steps
```

### Scrape UFC Data
```bash
cd DelphiAIApp/Models/data/scrapers
python scrape_all.py
python scrape_all.py --fresh       # Clear old data first
python scrape_all.py --test        # Limited pages for testing
```

### Validate & Load Data
```bash
cd DelphiAIApp/Models/data
python validate_data.py            # Check data quality
python validate_data.py --fix      # Auto-fix and save
python load_to_db.py               # Load everything to PostgreSQL
python load_to_db.py --clear       # Clear tables and reload
python load_to_db.py --core-only   # Just fighters/fights/career
```

### Update ELOs
```bash
cd DelphiAIApp/Models
python -m ml.update_adjusted_elos
python -m ml.update_adjusted_elos --check-injuries   # Also check injuries
python -m ml.update_adjusted_elos --dry-run           # Preview changes
```

### Populate Fighter Styles
```bash
cd DelphiAIApp/Models
python -m ml.populate_styles --run-migration   # First time
python -m ml.populate_styles                   # After new data
```

---

## Train the ML Model

```bash
cd DelphiAIApp/Models
python -m ml.train_model_v3
python -m ml.train_model_v3 --min-year 2020    # Train on 2020+ data only
```

**What it does:**
1. Loads fights with point-in-time ELO and career stats
2. Engineers 22 features (differentials, interactions, polynomials)
3. Splits chronologically: 70% train / 15% validation / 15% holdout
4. Trains Logistic Regression, Random Forest, XGBoost, and Stacking Ensemble
5. Calibrates probabilities with isotonic regression
6. Saves versioned model to `ml/artifacts/`

---

## First-Time Setup

```bash
cd DelphiAIApp/Models

# 1. Load data
cd data && python load_to_db.py && cd ..

# 2. Update ELOs and styles
python -m ml.update_adjusted_elos
python -m ml.populate_styles --run-migration

# 3. Train ML model
python -m ml.train_model_v3

# 4. Run backtest to validate
python -m ml.backtest --years 2023 2024 2025 --clear

# 5. Predict your first card!
python -m ml.predict_card "UFC 326"
```

---

## Understanding the Output

### Fight Prediction
- **ELO**: Final adjusted rating (includes ring rust + injury penalties)
- **WIN %**: Probability of winning (ML+ELO blended)
- **Source**: Shows "ML+ELO (50% ML)" or "ELO" (fallback)
- **Confidence**: HIGH (70%+), MED (60-70%), LOW (55-60%), TOSS (<55%)
- **Method**: KO/TKO, Sub, or Dec with percentages
- **Notes**: `[inj]` = recent injury, `[rust]` = inactive fighter

### Backtest Results
- **Data Quality**: % of fighters using historical vs current ELO/stats
- **Leakage Warning**: Flags if accuracy exceeds 75% (unrealistic)
- **Calibration**: Predicted vs actual win rates by probability bucket
- **Paper Trading**: Simulated ROI on 65%+ confidence picks

---

## Troubleshooting

### Database Connection Failed
- Ensure Docker containers are running: `docker-compose up -d`
- Check `.env` file has correct DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

### Fighter Not Found
- Check spelling against UFC.com
- Try partial name: `"Strickland"` instead of full name

### Module Not Found Error
- Make sure you're in the `DelphiAIApp/Models` directory
- All `python -m ml.*` commands must run from there

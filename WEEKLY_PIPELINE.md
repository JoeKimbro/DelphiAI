# DelphiAI — Weekly Pipeline

All commands run from `DelphiAIApp/Models/` unless stated otherwise.

---

## Every Fight Week — Step by Step

### WEDNESDAY / THURSDAY — Before the Event

**1. Predict the card**
```bash
python -m ml.predict_card "Event Name"
# e.g. python -m ml.predict_card "UFC 327"
```
Predictions are auto-saved to the database. Run this a few days out so injury/rust data is fresh.

---

### SUNDAY — After the Event

**2. Update results**
```bash
python -m ml.update_results "Event Name"
```
Pulls actual results from UFC.com and scores our predictions. Wait ~24 hours after the event for UFC.com to publish full results.

**3. Check performance**
```bash
python -m ml.performance_summary
```
Shows live accuracy, confidence tier breakdown, and ROI vs backtest. Watch the MED (60-65%) tier and ROI on 65%+ picks — those are the early warning signals.

---

### MONDAY / TUESDAY — Data & Model Refresh

**4. Run the full weekly update**
```bash
python -m ml.weekly_update
```
This runs the entire pipeline in order:
- Scrapes new fight results from UFC.com + UFCStats
- Validates the scraped CSVs
- Loads updated data into PostgreSQL
- Rebuilds all ELO ratings from scratch (experience-based K-factor applies automatically)
- Repopulates fighter styles
- Applies ring rust + injury ELO adjustments

If scraping already completed separately, skip it:
```bash
python -m ml.weekly_update --skip-scrape
```

**5. Retrain the model — quarterly only**

The model trains on 4,000+ fights. A single event adds ~12 fights (0.3% of training data) — not enough to matter. Retraining too often risks overfitting to recent variance.

**Retrain when:**
- ~100 new fights have been added (roughly every quarter)
- A structural feature change was made (e.g. new ELO K-factor, new features)
- The monthly calibration check shows the isotonic calibrator has drifted badly

```bash
python -m ml.train_model_v3
python -m ml.backtest --year 2025 --clear
```

Expected ranges after retrain:
| Metric | Target |
|--------|--------|
| Overall accuracy | 60–66% |
| High conf accuracy (65%+) | 70–78% |
| ROI on 65%+ picks | +5% to +15% |

> ELO ratings already update every week via `weekly_update` and feed directly into predictions. The model weights don't need to change for new fight results to be reflected.

---

## Full Weekly Checklist

```
[ ] Wednesday/Thursday  — python -m ml.predict_card "Event Name"
[ ] Sunday              — python -m ml.update_results "Event Name"
[ ] Sunday              — python -m ml.performance_summary
[ ] Monday/Tuesday      — python -m ml.weekly_update            (every week)

Every quarter (~100 new fights) or after a feature change:
[ ]                       python -m ml.train_model_v3
[ ]                       python -m ml.backtest --year 2025 --clear
```

---

## Warning Signs — When to Retrain Early

Don't retrain after one bad card — that's variance. Retrain early only if you see a sustained pattern across multiple events:

- MED confidence picks (60-66%) below 50% across **3+ consecutive events**
- Live ROI on 65%+ picks below -10% over **20+ bets**
- High conf accuracy (65%+) below 60% over **15+ live picks**
- A structural feature was changed (always retrain after this)

A single bad card is noise. The 2025 backtest baseline of 70% over 243 fights is the real signal — one event at 38% doesn't move that needle.

---

## Monthly — Full Calibration Check

Once a month, run a multi-year backtest to make sure the model hasn't drifted:
```bash
python -m ml.backtest --years 2024 2025 --clear
python -m ml.performance_summary
```

If the 65-70% calibration bucket actual win rate drops more than 15% below predicted, the isotonic calibrator needs retraining — retrain the full model with `train_model_v3`.

---

## UFCStats Lag Note

UFCStats (detailed strike/grappling stats) typically updates **24-48 hours** after an event. UFC.com (basic results) updates same night. If `weekly_update` runs before UFCStats updates, re-run it the next day:
```bash
python -m ml.weekly_update --skip-scrape   # if UFC.com data already loaded
# or
python -m ml.weekly_update                 # full re-scrape
```

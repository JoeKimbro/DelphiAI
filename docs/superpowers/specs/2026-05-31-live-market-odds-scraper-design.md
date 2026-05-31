# Live Market Odds Scraper — Design Spec

**Date:** 2026-05-31
**Status:** Approved design, pending spec review
**Author:** DelphiAI

## 1. Motivation

The live prediction path has no real market odds. Today, `event_service._enrich_with_odds()`
(`DelphiAIApp/Services/event_service.py:282`) converts the *model's own* probability to an
American line via `probability_to_american(f1_prob)` and hardcodes `edge_pct = 0.0`. Every
"edge" shown in the UI is therefore a placeholder, and the system effectively compares the
model against itself (or against the synthetic `realistic_odds_estimator`).

This feature adds a scheduled scraper that captures **real BestFightOdds (BFO) lines for
upcoming fights** at two points in the event week, stores them, and uses them to compute a
**real `edge_pct = model_prob − market_implied_prob`** in the live path.

## 2. Goals / Non-Goals

### Goals
- Capture real best-available market odds for upcoming fights, twice per event:
  on the **Wednesday** preceding the event and on the **day before the actual event date**.
- Store captures durably, keyed so they join cleanly to `PredictionTracking`.
- Replace the placeholder `edge_pct` with a real model-vs-market edge when odds exist.
- Be idempotent (safe to re-run a day) and non-destructive (a scrape failure never wipes
  existing odds or breaks the predictions UI).

### Non-Goals (explicitly out of scope)
- **ROI recompute.** `performance_summary` paper-trading keeps using `1/prob`. Using real
  decimal odds in ROI is deferred to a later phase.
- **Backfill.** No historical/archive odds for already-resolved events. Forward-only.
- **Frontend line-movement UI.** No new Wednesday→close visualization. Existing odds/edge
  fields simply start carrying real values.
- **Consensus/average line or per-book storage.** We store only the single best line per
  fighter per capture.
- **Continuous polling / full time series.** Exactly two snapshots per event.

## 3. Decisions (locked)

| Decision | Choice |
|---|---|
| Data source | Extend BestFightOdds scraping (reuse existing parse logic) |
| Capture cadence | 2 snapshots: `wednesday` + `day_before` (event_date − 1) |
| Line selection | Best available (max American) per fighter |
| Backfill | Forward-only |
| Trigger | One daily job that self-selects which events are due today |
| Scraper shape | Standalone module `ml/scrape_odds.py` (not a Scrapy spider) |
| Phase 2 scope | Real `edge_pct` only; ROI recompute deferred |

## 4. Architecture

### 4.1 Schema — migration `006_add_fight_odds.sql`

Follows the existing `DelphiAIApp/Models/db/migrations/001`–`005` pattern.

```sql
CREATE TABLE IF NOT EXISTS fight_odds (
    id              BIGSERIAL PRIMARY KEY,
    event_url       TEXT NOT NULL,        -- UFC.com slug; join key to PredictionTracking
    event_name      TEXT,
    event_date      DATE,
    fighter1_name   TEXT NOT NULL,        -- canonical (our) names, our orientation
    fighter2_name   TEXT NOT NULL,
    f1_best_american  INTEGER,
    f2_best_american  INTEGER,
    f1_best_decimal   NUMERIC(7,3),
    f2_best_decimal   NUMERIC(7,3),
    f1_best_book      TEXT,
    f2_best_book      TEXT,
    capture_label   TEXT NOT NULL,        -- 'wednesday' | 'day_before'
    captured_at     TIMESTAMPTZ DEFAULT NOW(),
    source          TEXT DEFAULT 'bestfightodds',
    UNIQUE (event_url, fighter1_name, fighter2_name, capture_label)
);

CREATE INDEX IF NOT EXISTS idx_fight_odds_lookup
    ON fight_odds (event_url, fighter1_name, fighter2_name);
```

The `UNIQUE(event_url, fighters, capture_label)` constraint gives idempotency (re-running a
day upserts via `ON CONFLICT`) and caps storage at two rows per fight. Odds are stored under
**our** canonical `event_url` and fighter names (mapped to our fighter1/fighter2 orientation)
so the downstream join is exact.

### 4.2 `ml/scrape_odds.py` — daily self-deciding job

Responsibilities, each as an independently testable unit:

- `upcoming_events_with_dates(conn) -> list[Event]`
  Source our upcoming events (slug + event_date + fighter pairs). Reuses
  `predict_card.scrape_upcoming_events()` / `event_service.list_upcoming_events()`.

- `due_captures(events, today) -> list[(Event, label)]`
  For each event with date `D`:
  - `day_before = D − 1 day`
  - `wednesday = the Wednesday strictly before D, within the same 7-day window`
    (largest `W` with `W.weekday()==2`, `W < D`, `(D−W).days <= 6`).
  - If `today == wednesday` → `(event, "wednesday")`.
  - If `today == day_before` → `(event, "day_before")`.
  - **Collision** (Thursday card → `wednesday == day_before`): emit a single
    `(event, "day_before")` (closing line takes precedence).

- `fetch_bfo_odds(event) -> list[BfoFight]`
  Locate the event on BFO and parse per-fighter odds across books. Reuses
  `parse_american_odds` logic from the existing archive spider
  (`.../spiders/bestfightodds.py`) and the BFO upcoming-events discovery already present in
  `ml/upcoming_predictor.py`. `requests`-based, polite delay, realistic UA.

- `match_fights(our_event, bfo_fights) -> list[MatchedFight]`
  Map BFO fighters to our fighter1/fighter2 by normalized surname (see §4.4). Best line per
  fighter = `max(american across books)` (max American = max decimal = most bettor-favorable
  for both favorites and underdogs). Unmatched fights are logged and skipped — never guessed.

- `store_odds(conn, our_event, matched, label)`
  `INSERT ... ON CONFLICT (event_url, fighter1_name, fighter2_name, capture_label) DO UPDATE`.

- `main()` — CLI:
  - default: run for all events due today.
  - `--event <slug>` + `--label {wednesday,day_before}`: force a capture now (testing / missed day).
  - `--dry-run`: scrape + match + print, no DB write.
  Structured logging mirroring `ml/prefetch_predictions.py`.

### 4.3 Edge integration — `event_service._enrich_with_odds`

Add a lookup and rewrite the edge fields:

- `lookup_best_odds(conn, event_url, f1_name, f2_name) -> row | None`
  Prefer `capture_label = 'day_before'`, fall back to `'wednesday'`.
- If a row exists:
  - `f1_american` / `f2_american` ← the **real** best lines.
  - `f1_implied = american_to_implied(f1_best_american)` (and f2).
  - `edge_pct = (model_prob − implied) * 100` per fighter (percentage points).
  - `odds_source = "market"`, plus `odds_book` and `odds_captured_at`.
- If no row (old event, or before Wednesday): unchanged behavior — model-derived
  `probability_to_american(prob)`, `edge_pct = 0.0`, `odds_source = "model"`.

`american_to_implied`: negative `a` → `(-a)/(-a+100)`; positive `a` → `100/(a+100)`.
`american_to_decimal` already exists in `realistic_odds_estimator`.

### 4.4 Name matching (the primary risk)

BFO and UFC.com spell fighters differently (accents, nicknames, "Jr."). Isolated in two units:

- `_normalize_name(name) -> str`: Unicode NFKD accent-strip, lowercase, strip punctuation and
  suffixes (Jr/Sr/III), collapse whitespace.
- `_match_pair(our_f1, our_f2, bfo_a, bfo_b) -> orientation | None`: match on normalized
  surname (last token), require both fighters of a bout to match, return the orientation that
  maps BFO→our (f1/f2). No match → `None` (logged, skipped).

A miss is **visible** (warning log, fight absent from `fight_odds`) rather than silently wrong.

## 5. Data Flow

```
Daily job (Windows Task Scheduler → python -m ml.scrape_odds)
  → upcoming_events_with_dates()        (our slugs + dates + fighters)
  → due_captures(events, today)         (which events, which label)
  → for each due event:
       fetch_bfo_odds()                 (BFO scrape + parse, best line/book per fighter)
       match_fights()                   (BFO ↔ our orientation, surname match)
       store_odds()                     (upsert into fight_odds)
  ─────────────────────────────────────
Live read path (unchanged trigger):
  GET event predictions
  → _enrich_with_odds()
       lookup_best_odds()               (day_before ▸ wednesday)
       hit  → real line + real edge_pct (odds_source="market")
       miss → model-derived line, edge 0 (odds_source="model")
```

## 6. Error Handling

- **BFO unreachable / timeout / parse failure:** log and exit non-fatally. Existing
  `fight_odds` rows are untouched; the predictions UI keeps working off the model fallback.
- **Event unmatched (BFO has no matching card):** log, skip event.
- **Fight unmatched within a matched event:** log, skip that fight; store the rest.
- **Re-run same day:** idempotent upsert (no duplicate rows).
- **No events due today:** no-op with a log line.

## 7. Testing

Unit (no network — fixture HTML for BFO parsing):
- `_normalize_name` — accents, "Jr.", nicknames, casing.
- `_match_pair` — correct orientation, swapped order, no-match.
- `due_captures` — Saturday card (Wed 3 days prior + Fri), Sunday card, Thursday-card
  collision (single `day_before`), event not due today.
- `american_to_implied` and best-line `max()` selection (favorite vs underdog).
- `fetch_bfo_odds` parsing against saved fixture HTML.

Integration (test DB):
- `store_odds` upsert idempotency (two runs → still two rows max per fight).
- `_enrich_with_odds`: real edge when a row exists; clean model fallback when absent.

## 8. Phasing

- **Phase 1 (core):** migration `006`, `ml/scrape_odds.py` (fetch + match + upsert + date
  logic + CLI), Windows Task Scheduler entry calling the module daily, tests.
- **Phase 2 (payoff, in this spec):** real `edge_pct` in `_enrich_with_odds` with model
  fallback + `odds_source`.
- **Deferred (separate spec later):** ROI recompute in `performance_summary` using captured
  decimal odds; frontend Wednesday→close line-movement display.

## 9. Affected / New Files

- **New:** `DelphiAIApp/Models/db/migrations/006_add_fight_odds.sql`
- **New:** `DelphiAIApp/Models/ml/scrape_odds.py`
- **New:** `DelphiAIApp/Models/ml/tests/test_scrape_odds.py` (+ BFO fixture HTML)
- **Modified:** `DelphiAIApp/Services/event_service.py` (`_enrich_with_odds` + `lookup_best_odds`)
- **Reused (no change):** archive spider parse logic, `realistic_odds_estimator`
  (`american_to_decimal`, `probability_to_american`), `db/postgres.py`,
  `predict_card.scrape_upcoming_events`.

## 10. Open Questions

None — all design forks resolved in §3.

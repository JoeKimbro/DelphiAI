"""Add fight_odds table for scheduled BestFightOdds snapshot captures.

Two snapshots per upcoming fight: one on the Wednesday before the event
and one on the day before. Keyed by (event_url, fighter1_name, fighter2_name,
capture_label) so re-running a day is idempotent via ON CONFLICT upsert.
"""
from yoyo import step

SQL = """
CREATE TABLE IF NOT EXISTS fight_odds (
    id                BIGSERIAL PRIMARY KEY,
    event_url         TEXT NOT NULL,
    event_name        TEXT,
    event_date        DATE,
    fighter1_name     TEXT NOT NULL,
    fighter2_name     TEXT NOT NULL,
    f1_best_american  INTEGER,
    f2_best_american  INTEGER,
    f1_best_decimal   NUMERIC(7,3),
    f2_best_decimal   NUMERIC(7,3),
    capture_label     TEXT NOT NULL,
    captured_at       TIMESTAMPTZ DEFAULT NOW(),
    source            TEXT DEFAULT 'bestfightodds',
    UNIQUE (event_url, fighter1_name, fighter2_name, capture_label)
);

CREATE INDEX IF NOT EXISTS idx_fight_odds_lookup
    ON fight_odds (event_url, fighter1_name, fighter2_name);
"""

steps = [step(SQL)]

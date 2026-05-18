-- DelphiAI Auth + Bet Tracking Schema
-- Run once against the same Postgres instance (port 5433) used by schemas.sql:
--   psql -h localhost -p 5433 -U <user> -d <db> -f schemas_auth.sql
--
-- These tables are independent of the ML/fight tables — keep separate so a
-- destructive `load_to_db.py --clear` cannot wipe user data.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT UNIQUE NOT NULL,
    name            TEXT,
    password_hash   TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token   TEXT UNIQUE NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);

CREATE TABLE IF NOT EXISTS user_bets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_name      TEXT NOT NULL,
    event_id        TEXT,                       -- UFC.com slug for joining to event view
    fighter_picked  TEXT NOT NULL,
    opponent        TEXT NOT NULL,
    bet_type        TEXT NOT NULL,              -- 'moneyline' | 'method' | 'round' | 'parlay'
    units_wagered   NUMERIC(10,2) NOT NULL,
    american_odds   INTEGER NOT NULL,
    model_prob      NUMERIC(5,4),               -- snapshot of model probability at bet time
    result          TEXT,                       -- NULL | 'won' | 'lost' | 'push'
    units_returned  NUMERIC(10,2),
    notes           TEXT,
    placed_at       TIMESTAMPTZ DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_user_bets_user_id ON user_bets(user_id);
CREATE INDEX IF NOT EXISTS idx_user_bets_event ON user_bets(event_name);
CREATE INDEX IF NOT EXISTS idx_user_bets_unresolved
    ON user_bets(user_id, placed_at)
    WHERE result IS NULL;

COMMENT ON TABLE users IS 'DelphiAI registered users (email + bcrypt password).';
COMMENT ON TABLE sessions IS 'Server-side session records for Auth.js or fallback cookie auth.';
COMMENT ON TABLE user_bets IS 'Personal bet log — independent of model predictions, tracks user-placed wagers and ROI.';
COMMENT ON COLUMN user_bets.model_prob IS 'Model probability snapshot at bet placement time, so later model retraining does not retroactively change this.';
COMMENT ON COLUMN user_bets.event_id IS 'UFC.com slug (e.g. "ufc-326-pereira-vs-ankalaev-2") for joining back to the event view.';

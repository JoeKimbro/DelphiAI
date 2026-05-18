-- Migration 005: event_url index on PredictionTracking
-- The service layer now uses event_url as the cache key (event_service.py:
-- _fetch_cached_predictions_by_url). Without this index the cache-hit SELECT
-- falls back to a sequential scan as the table grows.
-- Run with: psql -d delphi_db -f 005_add_event_url_index.sql

CREATE INDEX IF NOT EXISTS idx_pt_event_url
    ON PredictionTracking(event_url);

-- Partial index speeds _resolved_event_slugs (list_upcoming_events filter).
CREATE INDEX IF NOT EXISTS idx_pt_event_url_resolved
    ON PredictionTracking(event_url)
    WHERE was_correct IS NOT NULL;

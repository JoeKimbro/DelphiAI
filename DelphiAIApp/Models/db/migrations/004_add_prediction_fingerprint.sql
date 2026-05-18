-- Migration 004: input_fingerprint column on PredictionTracking
-- Used by ml/prefetch_predictions.py to detect whether the fighter-state inputs
-- to a prediction have changed since the last cached run. Same fingerprint =>
-- skip the prediction pipeline entirely (no UFC.com injury scrape, no model call,
-- no DB write).
-- Run with: psql -d delphi_db -f 004_add_prediction_fingerprint.sql

ALTER TABLE PredictionTracking
ADD COLUMN IF NOT EXISTS input_fingerprint VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_pt_fingerprint
    ON PredictionTracking(event_name, fighter1_name, fighter2_name)
    WHERE input_fingerprint IS NOT NULL;

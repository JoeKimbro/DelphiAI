-- Migration 003: PredictionLog table
-- Tracks every prediction made by the ML model for evaluation and auditing.
-- Run with: psql -d delphi_db -f 003_prediction_log.sql

-- Create PredictionLog table
CREATE TABLE IF NOT EXISTS PredictionLog (
    PredictionID SERIAL PRIMARY KEY,
    
    -- Fighters
    FighterAName VARCHAR(200) NOT NULL,
    FighterBName VARCHAR(200) NOT NULL,
    FighterAID INTEGER REFERENCES FighterStats(FighterID),
    FighterBID INTEGER REFERENCES FighterStats(FighterID),
    
    -- Prediction
    PredictedWinnerName VARCHAR(200),
    ProbabilityA DECIMAL(5,4) NOT NULL,        -- ML model probability for fighter A
    ProbabilityB DECIMAL(5,4) NOT NULL,        -- ML model probability for fighter B
    EloProbabilityA DECIMAL(5,4),              -- ELO-only probability for comparison
    EloProbabilityB DECIMAL(5,4),
    EloA DECIMAL(8,2),                         -- Adjusted ELO at prediction time
    EloB DECIMAL(8,2),
    
    -- Model info
    ModelVersion VARCHAR(100),                  -- e.g. "v3_20260203_143000"
    ProbabilitySource VARCHAR(50) DEFAULT 'ml', -- 'ml' or 'elo' (fallback)
    
    -- Feature snapshot (for reproducibility)
    Features JSONB,                             -- Full feature vector used
    
    -- Outcome (filled in after fight)
    FightDate DATE,                             -- Scheduled fight date
    ActualWinner VARCHAR(200),                  -- Filled after fight
    ActualMethod VARCHAR(100),                  -- KO/TKO, SUB, DEC
    WasCorrect BOOLEAN,                         -- Was prediction correct?
    
    -- Metadata
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ResolvedAt TIMESTAMP,                       -- When outcome was recorded
    
    -- Confidence level
    ConfidenceLevel VARCHAR(20)                  -- HIGH, MEDIUM, LOW, TOSS-UP
);

-- Index for querying unresolved predictions
CREATE INDEX IF NOT EXISTS idx_predlog_unresolved 
    ON PredictionLog(WasCorrect) WHERE WasCorrect IS NULL;

-- Index for querying by model version
CREATE INDEX IF NOT EXISTS idx_predlog_version 
    ON PredictionLog(ModelVersion);

-- Index for querying by date
CREATE INDEX IF NOT EXISTS idx_predlog_created 
    ON PredictionLog(CreatedAt DESC);

-- Index for querying by fighter
CREATE INDEX IF NOT EXISTS idx_predlog_fighters 
    ON PredictionLog(FighterAName, FighterBName);

"""Add all columns and tables missing from the baseline schema.

Covers:
  - CareerStats: InactivityPenalty, InjuryPenalty, AdjustedEloRating, AdjustmentsCalculatedAt
  - FighterStats: Nationality, LastInjuryCheckDate, InjuryDetails, UFCRanking
  - New table: InjuryCheckLog
  - New table: PredictionLog
"""
from yoyo import step

steps = [
    step(
        """
        -- CareerStats adjustments columns
        ALTER TABLE CareerStats
            ADD COLUMN IF NOT EXISTS InactivityPenalty INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS InjuryPenalty INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS AdjustedEloRating DECIMAL(8,2),
            ADD COLUMN IF NOT EXISTS AdjustmentsCalculatedAt TIMESTAMP;

        -- FighterStats extra columns
        ALTER TABLE FighterStats
            ADD COLUMN IF NOT EXISTS Nationality VARCHAR(50),
            ADD COLUMN IF NOT EXISTS LastInjuryCheckDate TIMESTAMP,
            ADD COLUMN IF NOT EXISTS InjuryDetails JSONB,
            ADD COLUMN IF NOT EXISTS UFCRanking INTEGER;

        -- Injury check audit log
        CREATE TABLE IF NOT EXISTS InjuryCheckLog (
            id SERIAL PRIMARY KEY,
            FighterID INTEGER,
            FighterName VARCHAR(100),
            InjuryFound BOOLEAN,
            InjuryKeyword VARCHAR(100),
            InjurySeverity VARCHAR(50),
            EstimatedInjuryDate VARCHAR(50),
            DaysSinceInjury INTEGER,
            ElopenaltyApplied INTEGER,
            NewsArticlesChecked INTEGER DEFAULT 0,
            SourceURL VARCHAR(500),
            RawDetails JSONB,
            Error TEXT,
            CheckedAt TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_injurychecklog_fighterid ON InjuryCheckLog(FighterID);
        CREATE INDEX IF NOT EXISTS idx_injurychecklog_checkedat ON InjuryCheckLog(CheckedAt);

        -- Prediction audit log
        CREATE TABLE IF NOT EXISTS PredictionLog (
            id SERIAL PRIMARY KEY,
            FighterAName VARCHAR(100),
            FighterBName VARCHAR(100),
            FighterAID INTEGER,
            FighterBID INTEGER,
            PredictedWinnerName VARCHAR(100),
            ProbabilityA DECIMAL(5,4),
            ProbabilityB DECIMAL(5,4),
            EloProbabilityA DECIMAL(5,4),
            EloProbabilityB DECIMAL(5,4),
            EloA DECIMAL(8,2),
            EloB DECIMAL(8,2),
            ModelVersion VARCHAR(50),
            ProbabilitySource VARCHAR(50),
            Features JSONB,
            ConfidenceLevel VARCHAR(10),
            PredictedAt TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        ALTER TABLE CareerStats
            DROP COLUMN IF EXISTS InactivityPenalty,
            DROP COLUMN IF EXISTS InjuryPenalty,
            DROP COLUMN IF EXISTS AdjustedEloRating,
            DROP COLUMN IF EXISTS AdjustmentsCalculatedAt;

        ALTER TABLE FighterStats
            DROP COLUMN IF EXISTS Nationality,
            DROP COLUMN IF EXISTS LastInjuryCheckDate,
            DROP COLUMN IF EXISTS InjuryDetails,
            DROP COLUMN IF EXISTS UFCRanking;

        DROP TABLE IF EXISTS InjuryCheckLog;
        DROP TABLE IF EXISTS PredictionLog;
        """
    )
]

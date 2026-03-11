-- Migration 001: Add Adjusted ELO Columns
-- 
-- This migration adds columns to track ELO adjustments for:
-- 1. Inactivity decay (ring rust)
-- 2. Injury penalties
--
-- These adjustments are applied on top of the base ELO rating
-- and should be recalculated periodically.

-- ============================================================================
-- ADD COLUMNS TO CAREERSTATS TABLE
-- ============================================================================

-- Adjusted ELO rating (base ELO - inactivity - injury)
ALTER TABLE CareerStats 
ADD COLUMN IF NOT EXISTS AdjustedEloRating DECIMAL(8,2);

-- Inactivity penalty (ring rust decay)
ALTER TABLE CareerStats 
ADD COLUMN IF NOT EXISTS InactivityPenalty DECIMAL(8,2) DEFAULT 0;

-- Injury penalty (from UFC.com news scraping)
ALTER TABLE CareerStats 
ADD COLUMN IF NOT EXISTS InjuryPenalty DECIMAL(8,2) DEFAULT 0;

-- When the ELO adjustments were last calculated
ALTER TABLE CareerStats 
ADD COLUMN IF NOT EXISTS AdjustmentsCalculatedAt TIMESTAMP;

-- ============================================================================
-- ADD COLUMNS TO FIGHTERSTATS TABLE
-- ============================================================================

-- When we last checked UFC.com for injuries
ALTER TABLE FighterStats 
ADD COLUMN IF NOT EXISTS LastInjuryCheckDate TIMESTAMP;

-- JSON blob with injury details (keyword, severity, date, etc.)
ALTER TABLE FighterStats 
ADD COLUMN IF NOT EXISTS InjuryDetails JSONB;

-- Fighter ranking (for prioritizing injury checks)
ALTER TABLE FighterStats 
ADD COLUMN IF NOT EXISTS UFCRanking INTEGER;

-- ============================================================================
-- CREATE INDEX FOR EFFICIENT QUERIES
-- ============================================================================

-- Index for finding fighters needing ELO recalculation
CREATE INDEX IF NOT EXISTS idx_careerstats_adj_calc 
ON CareerStats(AdjustmentsCalculatedAt);

-- Index for finding fighters needing injury check
CREATE INDEX IF NOT EXISTS idx_fighterstats_injury_check 
ON FighterStats(LastInjuryCheckDate);

-- Index for finding ranked fighters (priority for injury checks)
CREATE INDEX IF NOT EXISTS idx_fighterstats_ranking 
ON FighterStats(UFCRanking) WHERE UFCRanking IS NOT NULL;

-- Index for finding inactive fighters
CREATE INDEX IF NOT EXISTS idx_fighterstats_inactive 
ON FighterStats(DaysSinceLastFight) WHERE DaysSinceLastFight > 180;

-- ============================================================================
-- CREATE VIEW FOR EASY ACCESS TO ADJUSTED ELOS
-- ============================================================================

DROP VIEW IF EXISTS v_fighter_adjusted_elos;

CREATE VIEW v_fighter_adjusted_elos AS
SELECT 
    fs.FighterID,
    fs.Name,
    fs.WeightClass,
    fs.IsActive,
    fs.DaysSinceLastFight,
    fs.LastFightDate,
    fs.UFCRanking,
    cs.EloRating AS RawElo,
    cs.InactivityPenalty,
    cs.InjuryPenalty,
    cs.AdjustedEloRating,
    cs.AdjustmentsCalculatedAt,
    fs.LastInjuryCheckDate,
    fs.InjuryDetails
FROM FighterStats fs
LEFT JOIN CareerStats cs ON fs.FighterID = cs.FighterID
WHERE cs.EloRating IS NOT NULL;

-- ============================================================================
-- CREATE TABLE FOR INJURY CHECK LOG
-- ============================================================================

DROP TABLE IF EXISTS InjuryCheckLog CASCADE;

CREATE TABLE InjuryCheckLog (
    LogID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL,
    FighterName VARCHAR(100),
    CheckedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    InjuryFound BOOLEAN DEFAULT FALSE,
    InjuryKeyword VARCHAR(100),
    InjurySeverity VARCHAR(20),  -- 'major' or 'minor'
    EstimatedInjuryDate DATE,
    DaysSinceInjury INTEGER,
    ElopenaltyApplied DECIMAL(8,2),
    NewsArticlesChecked INTEGER DEFAULT 0,
    SourceURL VARCHAR(500),
    RawDetails JSONB,
    Error TEXT,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE
);

-- Index for injury check history
CREATE INDEX IF NOT EXISTS idx_injury_log_fighter 
ON InjuryCheckLog(FighterID, CheckedAt DESC);

CREATE INDEX IF NOT EXISTS idx_injury_log_date 
ON InjuryCheckLog(CheckedAt DESC);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN CareerStats.AdjustedEloRating IS 
'ELO rating after applying inactivity decay and injury penalties';

COMMENT ON COLUMN CareerStats.InactivityPenalty IS 
'ELO points deducted for ring rust (based on days since last fight)';

COMMENT ON COLUMN CareerStats.InjuryPenalty IS 
'ELO points deducted for recent injuries (from UFC.com news)';

COMMENT ON COLUMN FighterStats.InjuryDetails IS 
'JSON with injury info: {keyword, severity, date, penalty, context}';

COMMENT ON TABLE InjuryCheckLog IS 
'Log of all injury checks for auditing and debugging';

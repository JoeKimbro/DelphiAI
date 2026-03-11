-- Migration 002: Add fighting style classification and style matchup history
-- Run with: psql -d delphi_db -f 002_add_fighting_style.sql

-- ============================================================================
-- 1. Add FightingStyle to FighterStats
-- ============================================================================
ALTER TABLE FighterStats 
ADD COLUMN IF NOT EXISTS FightingStyle VARCHAR(20);

COMMENT ON COLUMN FighterStats.FightingStyle IS 'Classified style: striker, wrestler, grappler, or balanced - derived from career stats';

-- ============================================================================
-- 2. Create StyleMatchupRecord table
-- Tracks how each fighter performs against each opponent style
-- ============================================================================
CREATE TABLE IF NOT EXISTS StyleMatchupRecord (
    StyleRecordID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL,
    FighterURL VARCHAR(255),
    OpponentStyle VARCHAR(20) NOT NULL,  -- striker, wrestler, grappler, balanced
    
    -- Record vs this style
    Wins INTEGER DEFAULT 0,
    Losses INTEGER DEFAULT 0,
    Draws INTEGER DEFAULT 0,
    TotalFights INTEGER DEFAULT 0,
    WinRate DECIMAL(5,2),               -- Win % vs this style (0-100)
    
    -- Finish breakdown vs this style
    KOWins INTEGER DEFAULT 0,
    SubWins INTEGER DEFAULT 0,
    DecWins INTEGER DEFAULT 0,
    KOLosses INTEGER DEFAULT 0,
    SubLosses INTEGER DEFAULT 0,
    DecLosses INTEGER DEFAULT 0,
    
    -- Performance vs this style (averages)
    AvgSLpM DECIMAL(5,2),              -- Avg sig strikes landed per min vs this style
    AvgTDLanded DECIMAL(5,2),          -- Avg takedowns per 15 min vs this style
    AvgSubAttempts DECIMAL(5,2),       -- Avg sub attempts vs this style
    AvgFightDuration DECIMAL(5,2),     -- Avg fight length vs this style (minutes)
    
    -- Meta
    CalculatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    CONSTRAINT unique_fighter_style UNIQUE (FighterID, OpponentStyle)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_stylematchup_fighterid ON StyleMatchupRecord(FighterID);
CREATE INDEX IF NOT EXISTS idx_stylematchup_oppstyle ON StyleMatchupRecord(OpponentStyle);
CREATE INDEX IF NOT EXISTS idx_stylematchup_fighter_style ON StyleMatchupRecord(FighterID, OpponentStyle);
CREATE INDEX IF NOT EXISTS idx_fighterstats_style ON FighterStats(FightingStyle);

-- Comments
COMMENT ON TABLE StyleMatchupRecord IS 'Per-fighter record vs each opponent style (striker/wrestler/grappler/balanced)';
COMMENT ON COLUMN StyleMatchupRecord.OpponentStyle IS 'Style of opponents in this record group';
COMMENT ON COLUMN StyleMatchupRecord.WinRate IS 'Win percentage vs this style (0-100)';

-- ============================================================================
-- 3. Helpful view: Fighter style summary
-- ============================================================================
CREATE OR REPLACE VIEW v_fighter_style_matchups AS
SELECT 
    fs.FighterID,
    fs.Name,
    fs.FightingStyle,
    smr.OpponentStyle,
    smr.Wins,
    smr.Losses,
    smr.TotalFights,
    smr.WinRate,
    smr.KOWins,
    smr.SubWins,
    smr.DecWins
FROM FighterStats fs
JOIN StyleMatchupRecord smr ON fs.FighterID = smr.FighterID
ORDER BY fs.Name, smr.OpponentStyle;

SELECT 'Migration 002 complete: Added FightingStyle column and StyleMatchupRecord table' AS status;

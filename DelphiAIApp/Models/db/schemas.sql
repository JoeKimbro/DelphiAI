-- PostgreSQL Schema for Fighter Stats Database
-- Updated with recommendations for UFC predictive modeling
-- V2: Added ELO history, pre-UFC data, opponent quality, and ML feature caching

-- Drop tables if they exist (in correct order due to foreign keys)
DROP TABLE IF EXISTS PointInTimeStats CASCADE;
DROP TABLE IF EXISTS FighterHistoricalFeatures CASCADE;
DROP TABLE IF EXISTS MatchupFeatures CASCADE;
DROP TABLE IF EXISTS Matchups CASCADE;
DROP TABLE IF EXISTS EloHistory CASCADE;
DROP TABLE IF EXISTS PreUfcCareer CASCADE;
DROP TABLE IF EXISTS OpponentQuality CASCADE;
DROP TABLE IF EXISTS Fights CASCADE;
DROP TABLE IF EXISTS CareerStats CASCADE;
DROP TABLE IF EXISTS FighterStats CASCADE;

-- Create FighterStats table (main fighter information)
CREATE TABLE FighterStats (
    FighterID SERIAL PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    FighterURL VARCHAR(255) UNIQUE, -- UFCStats URL (unique identifier for deduplication)
    Height VARCHAR(20),
    Weight VARCHAR(20),
    Reach VARCHAR(20),
    Stance VARCHAR(50),
    DOB DATE,
    Age INTEGER, -- Calculated from DOB or stored directly
    WeightClass VARCHAR(50),
    Nickname VARCHAR(100), -- Fighter nickname/alias (from UFC.com)
    PlaceOfBirth VARCHAR(100), -- Hometown/birthplace (from UFC.com)
    LegReach VARCHAR(20), -- Leg reach (from UFC.com)
    UFCUrl VARCHAR(255), -- UFC.com profile URL
    TotalFights INTEGER DEFAULT 0,
    Wins INTEGER DEFAULT 0,
    Losses INTEGER DEFAULT 0,
    Draws INTEGER DEFAULT 0,
    LastFightDate DATE, -- Date of most recent fight
    DaysSinceLastFight INTEGER,
    IsActive BOOLEAN DEFAULT TRUE, -- Fought within last 2 years
    Source VARCHAR(50), -- 'ufcstats', 'tapology', etc.
    ScrapedAt TIMESTAMP,
    FightUpdatedAt TIMESTAMP
);

-- Create CareerStats table (one-to-one relationship with FighterStats)
CREATE TABLE CareerStats (
    CSID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL UNIQUE,
    FighterURL VARCHAR(255), -- For joining before FighterID is assigned
    SLpM DECIMAL(5,2), -- Significant Strikes Landed per Minute
    StrAcc DECIMAL(5,2), -- Strike Accuracy (0-100)
    SApM DECIMAL(5,2), -- Significant Strikes Absorbed per Minute
    StrDef DECIMAL(5,2), -- Strike Defense (0-100)
    TDAvg DECIMAL(5,2), -- Takedown Average per 15 minutes
    TDAcc DECIMAL(5,2), -- Takedown Accuracy (0-100) - NEW
    TDDef DECIMAL(5,2), -- Takedown Defense (0-100)
    SubAvg DECIMAL(5,2), -- Submission Average per 15 minutes
    WinStreak_Last3 INTEGER,
    WinsByKO_Last5 INTEGER,
    WinsBySub_Last5 INTEGER,
    AvgFightDuration DECIMAL(5,2), -- Average fight duration in minutes
    FirstRoundFinishRate DECIMAL(5,2), -- Percentage of fights finished in round 1
    DecisionRate DECIMAL(5,2), -- Percentage of fights going to decision
    CareerUpdatedAt TIMESTAMP,
    KO_Round1_Pct DECIMAL(5,2),
    KO_Round2_Pct DECIMAL(5,2),
    KO_Round3_Pct DECIMAL(5,2),
    Sub_Round1_Pct DECIMAL(5,2),
    Sub_Round2_Pct DECIMAL(5,2),
    Sub_Round3_Pct DECIMAL(5,2),
    EloRating DECIMAL(8,2),
    PeakEloRating DECIMAL(8,2),
    Source VARCHAR(50),
    ScrapedAt TIMESTAMP,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE
);

-- Create Fights table (detailed fight information)
CREATE TABLE Fights (
    FightID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL,
    FighterURL VARCHAR(255), -- For joining before FighterID is assigned
    FighterName VARCHAR(100), -- Backup reference
    OpponentID INTEGER,
    OpponentURL VARCHAR(255), -- For joining before OpponentID is assigned
    OpponentName VARCHAR(100), -- Backup if opponent not in FighterStats
    WinnerID INTEGER,
    WinnerName VARCHAR(100), -- Backup reference
    Result VARCHAR(20), -- 'win', 'loss', 'draw', 'nc' (from fighter's perspective)
    Date DATE,
    EventName VARCHAR(200),
    EventURL VARCHAR(255),
    FightURL VARCHAR(255) UNIQUE, -- UFCStats fight detail URL
    Method VARCHAR(50), -- e.g., "KO/TKO", "SUB", "U-DEC", "S-DEC"
    MethodDetail VARCHAR(100), -- e.g., "Punch", "Rear Naked Choke"
    Round INTEGER, -- Which round the fight ended
    Time VARCHAR(10), -- Time in the round (e.g., "2:34")
    Knockdowns VARCHAR(10), -- Knockdowns scored (can be '--' for old fights)
    SigStrikes VARCHAR(20), -- Significant strikes landed
    Takedowns VARCHAR(20), -- Takedowns landed
    SubAttempts VARCHAR(20), -- Submission attempts
    IsMainEvent BOOLEAN DEFAULT FALSE,
    IsTitleFight BOOLEAN DEFAULT FALSE,
    FlukeFlag BOOLEAN DEFAULT FALSE,
    SampleWeight DECIMAL(5,2),
    Source VARCHAR(50),
    ScrapedAt TIMESTAMP,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    FOREIGN KEY (OpponentID) REFERENCES FighterStats(FighterID) ON DELETE SET NULL,
    FOREIGN KEY (WinnerID) REFERENCES FighterStats(FighterID) ON DELETE SET NULL
);

-- ============================================================================
-- ELO HISTORY TABLE - Track ELO at time of each fight
-- ============================================================================
-- Critical for: backtesting, historical analysis, ELO trajectory
CREATE TABLE EloHistory (
    EloHistoryID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL,
    FighterURL VARCHAR(255),
    FightID INTEGER,                      -- Reference to the fight
    FightDate DATE NOT NULL,
    OpponentID INTEGER,
    OpponentURL VARCHAR(255),
    -- ELO at time of fight (BEFORE the fight)
    EloBeforeFight DECIMAL(8,2) NOT NULL,
    OpponentEloBeforeFight DECIMAL(8,2),
    -- ELO after fight
    EloAfterFight DECIMAL(8,2) NOT NULL,
    EloChange DECIMAL(8,2),               -- How much ELO changed
    -- Fight context
    Result VARCHAR(20),                   -- win/loss/draw
    Method VARCHAR(50),
    ExpectedWinProb DECIMAL(5,4),         -- Pre-fight win probability based on ELO
    -- Metadata
    EloSource VARCHAR(50) DEFAULT 'ufc_fights', -- 'ufc_fights' or 'pre_ufc_estimate'
    CalculatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    FOREIGN KEY (FightID) REFERENCES Fights(FightID) ON DELETE SET NULL,
    FOREIGN KEY (OpponentID) REFERENCES FighterStats(FighterID) ON DELETE SET NULL
);

-- ============================================================================
-- POINT-IN-TIME STATS TABLE - Career stats at time of each fight
-- ============================================================================
-- Critical for: ML training without data leakage
-- Stores what a fighter's career stats were BEFORE each fight
CREATE TABLE PointInTimeStats (
    PITID SERIAL PRIMARY KEY,
    FighterID INTEGER,
    FighterURL VARCHAR(255),
    FightDate DATE NOT NULL,
    -- Fight count before this fight
    FightsBefore INTEGER DEFAULT 0,
    WinsBefore INTEGER DEFAULT 0,
    LossesBefore INTEGER DEFAULT 0,
    WinRateBefore DECIMAL(5,3),
    -- Point-in-time performance stats (calculated from prior fights only)
    PIT_SLpM DECIMAL(5,2),              -- Strikes landed per minute (prior fights)
    PIT_StrAcc DECIMAL(5,2),            -- Strike accuracy (prior fights)
    PIT_TDAvg DECIMAL(5,2),             -- Takedown avg per 15 min (prior fights)
    PIT_SubAvg DECIMAL(5,2),            -- Submission avg per 15 min (prior fights)
    PIT_KDRate DECIMAL(5,3),            -- Knockdown rate per minute (prior fights)
    -- Form indicators
    RecentWinRate DECIMAL(5,2),         -- Win rate in last 3 fights
    AvgFightTime DECIMAL(5,2),          -- Average fight duration (minutes)
    FinishRate DECIMAL(5,2),            -- % of wins by finish
    -- Metadata
    HasPriorData BOOLEAN DEFAULT FALSE, -- False for debut fights
    CalculatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE
);

-- Index for fast lookup by fighter and date
CREATE INDEX idx_pit_fighter_date ON PointInTimeStats(FighterID, FightDate);
CREATE INDEX idx_pit_url_date ON PointInTimeStats(FighterURL, FightDate);

-- ============================================================================
-- FIGHTER HISTORICAL FEATURES TABLE - Rolling point-in-time trend features
-- ============================================================================
-- Stores historical context that must be computed only from prior fights.
CREATE TABLE FighterHistoricalFeatures (
    HistoricalFeatureID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL,
    FightID INTEGER NOT NULL,
    FightDate DATE NOT NULL,
    -- Opponent quality rolling windows
    AvgOpponentEloLast3 DECIMAL(8,2),
    AvgOpponentEloLast5 DECIMAL(8,2),
    MaxOpponentEloLast5 DECIMAL(8,2),
    -- Performance velocity
    EloChangeLast3 DECIMAL(8,2),
    EloVelocity DECIMAL(8,2),
    -- Recent form vs career
    FinishRateLast3 DECIMAL(6,3),
    FinishRateLast5 DECIMAL(6,3),
    FinishRateCareer DECIMAL(6,3),
    FinishRateTrending DECIMAL(6,3),
    -- Streak momentum
    CurrentWinStreak INTEGER DEFAULT 0,
    CurrentLossStreak INTEGER DEFAULT 0,
    -- Competition trend
    OpponentQualityTrending DECIMAL(8,2),
    -- Metadata
    HasPriorData BOOLEAN DEFAULT FALSE,
    CalculatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    FOREIGN KEY (FightID) REFERENCES Fights(FightID) ON DELETE CASCADE,
    CONSTRAINT unique_fighter_fight_historical UNIQUE (FighterID, FightID)
);

CREATE INDEX idx_histfeat_fighter_date ON FighterHistoricalFeatures(FighterID, FightDate);
CREATE INDEX idx_histfeat_fightid ON FighterHistoricalFeatures(FightID);

-- ============================================================================
-- PRE-UFC CAREER TABLE - Store pre-UFC fight history and org quality
-- ============================================================================
-- Critical for: initial ELO estimation, prospect evaluation
CREATE TABLE PreUfcCareer (
    PreUfcID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL UNIQUE,
    FighterURL VARCHAR(255),
    -- Record before UFC
    PreUfcWins INTEGER DEFAULT 0,
    PreUfcLosses INTEGER DEFAULT 0,
    PreUfcDraws INTEGER DEFAULT 0,
    PreUfcTotalFights INTEGER DEFAULT 0,
    -- Finish rates (if available)
    PreUfcKoWins INTEGER DEFAULT 0,
    PreUfcSubWins INTEGER DEFAULT 0,
    PreUfcDecisionWins INTEGER DEFAULT 0,
    PreUfcFinishRate DECIMAL(5,2),        -- (KO + Sub) / Wins * 100
    -- Organization quality (1-5 scale, 5 = top regional like Cage Warriors, LFA)
    PrimaryOrg VARCHAR(100),              -- Main org fought in
    OrgQualityTier INTEGER CHECK (OrgQualityTier BETWEEN 1 AND 5),
    -- Org quality guide:
    -- 5: Elite regional (Cage Warriors, LFA, Invicta, ONE, Bellator feeder)
    -- 4: Strong regional (CFFC, Fury FC, DWCS graduates)
    -- 3: Mid-tier regional (most US regionals)
    -- 2: Low-tier regional
    -- 1: Unknown/amateur heavy
    -- Career timing
    YearsAsProBeforeUfc DECIMAL(4,1),
    AgeAtUfcDebut INTEGER,
    DateOfUfcDebut DATE,
    -- Activity/recency
    DaysSinceLastPreUfcFight INTEGER,
    LastPreUfcFightDate DATE,
    -- Estimated initial ELO
    EstimatedInitialElo DECIMAL(8,2),
    EloEstimationMethod VARCHAR(50),      -- 'enhanced', 'simple', 'manual'
    EloEstimationBreakdown JSONB,         -- Store the breakdown factors
    -- Metadata
    DataSource VARCHAR(100),              -- Where we got this data
    DataConfidence VARCHAR(20),           -- 'high', 'medium', 'low'
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE
);

-- ============================================================================
-- OPPONENT QUALITY TABLE - Track strength of schedule metrics
-- ============================================================================
-- Critical for: evaluating fighter quality, ML feature
CREATE TABLE OpponentQuality (
    OQID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL UNIQUE,
    FighterURL VARCHAR(255),
    -- Strength of Schedule (SOS) metrics
    AvgOpponentElo DECIMAL(8,2),          -- Average ELO of opponents faced
    AvgOpponentEloAtFightTime DECIMAL(8,2), -- More accurate: ELO at time of fight
    AvgOpponentWinRate DECIMAL(5,2),      -- Average win% of opponents
    AvgOpponentRank INTEGER,              -- If ranked data available
    -- Quality breakdown
    EliteOpponentWins INTEGER DEFAULT 0,  -- Wins vs 1650+ ELO
    EliteOpponentLosses INTEGER DEFAULT 0,
    GoodOpponentWins INTEGER DEFAULT 0,   -- Wins vs 1550-1650 ELO
    GoodOpponentLosses INTEGER DEFAULT 0,
    AverageOpponentWins INTEGER DEFAULT 0, -- Wins vs 1450-1550 ELO
    AverageOpponentLosses INTEGER DEFAULT 0,
    BelowAverageWins INTEGER DEFAULT 0,   -- Wins vs <1450 ELO
    BelowAverageLosses INTEGER DEFAULT 0,
    -- Quality ratios (for ML)
    EliteWinRate DECIMAL(5,2),            -- Win% vs elite opponents
    QualityWinIndex DECIMAL(5,2),         -- Weighted quality score
    -- Recent opponent quality (last 5 fights)
    RecentAvgOpponentElo DECIMAL(8,2),
    RecentEliteWins INTEGER DEFAULT 0,
    -- Calculated metrics
    ScheduleStrengthRank INTEGER,         -- Rank among all fighters
    ScheduleStrengthPercentile DECIMAL(5,2),
    -- Metadata
    LastCalculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FightsAnalyzed INTEGER DEFAULT 0,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE
);

-- ============================================================================
-- MATCHUP FEATURES TABLE - Cached differentials for ML performance
-- ============================================================================
-- Pre-calculated features for each potential matchup
-- Avoids recalculating on every prediction
CREATE TABLE MatchupFeatures (
    MatchupFeatureID SERIAL PRIMARY KEY,
    Fighter1ID INTEGER NOT NULL,
    Fighter2ID INTEGER NOT NULL,
    -- Physical differentials (Fighter1 - Fighter2)
    HeightDiff_cm DECIMAL(5,1),           -- Positive = Fighter1 taller
    ReachDiff_cm DECIMAL(5,1),            -- Positive = Fighter1 longer reach
    LegReachDiff_cm DECIMAL(5,1),
    AgeDiff DECIMAL(4,1),                 -- Positive = Fighter1 older
    -- ELO differential
    EloDiff DECIMAL(8,2),                 -- Positive = Fighter1 higher ELO
    PeakEloDiff DECIMAL(8,2),
    -- Striking differentials
    SLpMDiff DECIMAL(5,2),                -- Strikes landed per min diff
    SApMDiff DECIMAL(5,2),                -- Strikes absorbed per min diff
    StrAccDiff DECIMAL(5,2),              -- Strike accuracy diff
    StrDefDiff DECIMAL(5,2),              -- Strike defense diff
    -- Grappling differentials
    TDAvgDiff DECIMAL(5,2),               -- Takedown avg diff
    TDAccDiff DECIMAL(5,2),               -- Takedown accuracy diff
    TDDefDiff DECIMAL(5,2),               -- Takedown defense diff
    SubAvgDiff DECIMAL(5,2),              -- Submission avg diff
    -- Quality differentials
    OpponentQualityDiff DECIMAL(5,2),     -- SOS difference
    WinStreakDiff INTEGER,                -- Current win streak diff
    -- Activity differentials
    DaysSinceLastFightDiff INTEGER,       -- Ring rust comparison
    TotalFightsDiff INTEGER,              -- Experience diff
    -- Style matchup info
    Fighter1Style VARCHAR(20),            -- striker/wrestler/grappler/balanced
    Fighter2Style VARCHAR(20),
    StyleMatchupAdvantage INTEGER,        -- -1, 0, or 1 for F2, neutral, F1
    -- Metadata
    CalculatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    IsStale BOOLEAN DEFAULT FALSE,        -- Mark if underlying data changed
    FOREIGN KEY (Fighter1ID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    FOREIGN KEY (Fighter2ID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    CONSTRAINT different_fighters_features CHECK (Fighter1ID != Fighter2ID),
    CONSTRAINT unique_matchup_pair UNIQUE (Fighter1ID, Fighter2ID)
);

-- Create Matchups table (for tracking predictions and model performance)
CREATE TABLE Matchups (
    MatchupID SERIAL PRIMARY KEY,
    Fighter1ID INTEGER NOT NULL,
    Fighter2ID INTEGER NOT NULL,
    EventDate DATE,
    EventName VARCHAR(200),
    WeightClass VARCHAR(50),
    IsTitleFight BOOLEAN DEFAULT FALSE,
    -- Predictions
    PredictedWinnerID INTEGER,
    PredictedMethod VARCHAR(50),
    PredictedWinProbability DECIMAL(5,2), -- 0-100
    PredictedKOProbability DECIMAL(5,2),
    PredictedSubProbability DECIMAL(5,2),
    PredictedDecisionProbability DECIMAL(5,2),
    -- Actuals (filled in after fight)
    ActualWinnerID INTEGER,
    ActualMethod VARCHAR(50),
    ActualRound INTEGER,
    -- Model metadata
    ModelVersion VARCHAR(50),
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Foreign key constraints
    FOREIGN KEY (Fighter1ID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    FOREIGN KEY (Fighter2ID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    FOREIGN KEY (PredictedWinnerID) REFERENCES FighterStats(FighterID) ON DELETE SET NULL,
    FOREIGN KEY (ActualWinnerID) REFERENCES FighterStats(FighterID) ON DELETE SET NULL,
    -- Ensure Fighter1ID and Fighter2ID are different
    CONSTRAINT different_fighters CHECK (Fighter1ID != Fighter2ID)
);

-- ============================================================================
-- INDEXES for better query performance
-- ============================================================================

-- FighterStats indexes
CREATE INDEX idx_fighterstats_weightclass ON FighterStats(WeightClass);
CREATE INDEX idx_fighterstats_name ON FighterStats(Name);
CREATE INDEX idx_fighterstats_fighterurl ON FighterStats(FighterURL);
CREATE INDEX idx_fighterstats_isactive ON FighterStats(IsActive);

-- CareerStats indexes
CREATE INDEX idx_careerstats_fighterid ON CareerStats(FighterID);
CREATE INDEX idx_careerstats_fighterurl ON CareerStats(FighterURL);
CREATE INDEX idx_careerstats_elo ON CareerStats(EloRating);

-- Fights indexes
CREATE INDEX idx_fights_fighterid ON Fights(FighterID);
CREATE INDEX idx_fights_fighterurl ON Fights(FighterURL);
CREATE INDEX idx_fights_opponentid ON Fights(OpponentID);
CREATE INDEX idx_fights_opponenturl ON Fights(OpponentURL);
CREATE INDEX idx_fights_fighturl ON Fights(FightURL);
CREATE INDEX idx_fights_date ON Fights(Date);
CREATE INDEX idx_fights_eventname ON Fights(EventName);

-- EloHistory indexes (critical for historical queries)
CREATE INDEX idx_elohistory_fighterid ON EloHistory(FighterID);
CREATE INDEX idx_elohistory_fightdate ON EloHistory(FightDate);
CREATE INDEX idx_elohistory_fighterid_date ON EloHistory(FighterID, FightDate);
CREATE INDEX idx_elohistory_opponentid ON EloHistory(OpponentID);

-- PreUfcCareer indexes
CREATE INDEX idx_preufc_fighterid ON PreUfcCareer(FighterID);
CREATE INDEX idx_preufc_fighterurl ON PreUfcCareer(FighterURL);
CREATE INDEX idx_preufc_orgquality ON PreUfcCareer(OrgQualityTier);

-- OpponentQuality indexes
CREATE INDEX idx_oppquality_fighterid ON OpponentQuality(FighterID);
CREATE INDEX idx_oppquality_avgelo ON OpponentQuality(AvgOpponentElo);
CREATE INDEX idx_oppquality_sosrank ON OpponentQuality(ScheduleStrengthRank);

-- MatchupFeatures indexes (for fast lookup)
CREATE INDEX idx_matchupfeatures_fighter1 ON MatchupFeatures(Fighter1ID);
CREATE INDEX idx_matchupfeatures_fighter2 ON MatchupFeatures(Fighter2ID);
CREATE INDEX idx_matchupfeatures_pair ON MatchupFeatures(Fighter1ID, Fighter2ID);
CREATE INDEX idx_matchupfeatures_stale ON MatchupFeatures(IsStale) WHERE IsStale = TRUE;

-- Matchups indexes
CREATE INDEX idx_matchups_fighter1 ON Matchups(Fighter1ID);
CREATE INDEX idx_matchups_fighter2 ON Matchups(Fighter2ID);
CREATE INDEX idx_matchups_eventdate ON Matchups(EventDate);
CREATE INDEX idx_matchups_modelversion ON Matchups(ModelVersion);

-- ============================================================================
-- TABLE COMMENTS
-- ============================================================================
COMMENT ON TABLE FighterStats IS 'Main table containing fighter physical attributes and basic information';
COMMENT ON TABLE CareerStats IS 'Career statistics and performance metrics for each fighter';
COMMENT ON TABLE Fights IS 'Detailed fight records including opponents, method, and outcome';
COMMENT ON TABLE EloHistory IS 'ELO rating history - tracks ELO at time of each fight for backtesting';
COMMENT ON TABLE PreUfcCareer IS 'Pre-UFC career data - org quality, finish rates, for initial ELO estimation';
COMMENT ON TABLE OpponentQuality IS 'Strength of schedule metrics - opponent quality breakdown for ML features';
COMMENT ON TABLE MatchupFeatures IS 'Cached feature differentials for matchups - pre-calculated for ML performance';
COMMENT ON TABLE Matchups IS 'Predicted and actual results for fighter matchups, used for model evaluation';

-- Add comments to important columns
COMMENT ON COLUMN FighterStats.Name IS 'Fighter full name';
COMMENT ON COLUMN FighterStats.FighterURL IS 'UFCStats unique URL - used for deduplication and joins';
COMMENT ON COLUMN FighterStats.Age IS 'Current age or age at last fight';
COMMENT ON COLUMN FighterStats.TotalFights IS 'Total number of professional fights';
COMMENT ON COLUMN FighterStats.LastFightDate IS 'Date of most recent fight';
COMMENT ON COLUMN FighterStats.DaysSinceLastFight IS 'Days since last fight (for ring rust analysis)';
COMMENT ON COLUMN FighterStats.IsActive IS 'Fighter has competed within last 2 years';
COMMENT ON COLUMN FighterStats.Source IS 'Data source: ufcstats, ufc_official, etc.';
COMMENT ON COLUMN FighterStats.ScrapedAt IS 'When the data was scraped';
COMMENT ON COLUMN FighterStats.Nickname IS 'Fighter nickname/alias from UFC.com';
COMMENT ON COLUMN FighterStats.PlaceOfBirth IS 'Fighter hometown/birthplace from UFC.com';
COMMENT ON COLUMN FighterStats.LegReach IS 'Leg reach measurement from UFC.com';
COMMENT ON COLUMN FighterStats.UFCUrl IS 'UFC.com athlete profile URL';

COMMENT ON COLUMN CareerStats.FighterURL IS 'For joining to FighterStats before FighterID lookup';
COMMENT ON COLUMN CareerStats.SLpM IS 'Significant Strikes Landed per Minute';
COMMENT ON COLUMN CareerStats.StrAcc IS 'Strike Accuracy (0-100)';
COMMENT ON COLUMN CareerStats.SApM IS 'Significant Strikes Absorbed per Minute';
COMMENT ON COLUMN CareerStats.StrDef IS 'Strike Defense percentage (0-100)';
COMMENT ON COLUMN CareerStats.TDAvg IS 'Takedown Average per 15 minutes';
COMMENT ON COLUMN CareerStats.TDAcc IS 'Takedown Accuracy percentage (0-100)';
COMMENT ON COLUMN CareerStats.TDDef IS 'Takedown Defense percentage (0-100)';
COMMENT ON COLUMN CareerStats.SubAvg IS 'Submission Average per 15 minutes';
COMMENT ON COLUMN CareerStats.AvgFightDuration IS 'Average fight duration in minutes';
COMMENT ON COLUMN CareerStats.FirstRoundFinishRate IS 'Percentage of fights finished in first round';
COMMENT ON COLUMN CareerStats.DecisionRate IS 'Percentage of fights going to decision';
COMMENT ON COLUMN CareerStats.EloRating IS 'Current ELO rating - FiveThirtyEight style with enhancements (style/age/reach/form)';
COMMENT ON COLUMN CareerStats.PeakEloRating IS 'Highest ELO rating ever achieved - useful for detecting declining fighters';

COMMENT ON COLUMN Fights.FighterURL IS 'For joining to FighterStats before FighterID lookup';
COMMENT ON COLUMN Fights.OpponentURL IS 'For joining opponent to FighterStats';
COMMENT ON COLUMN Fights.FightURL IS 'UFCStats fight detail URL - unique identifier';
COMMENT ON COLUMN Fights.Result IS 'Fight result from fighters perspective: win, loss, draw, nc';
COMMENT ON COLUMN Fights.Method IS 'Finish method: KO/TKO, SUB, U-DEC, S-DEC, M-DEC';
COMMENT ON COLUMN Fights.MethodDetail IS 'Specific technique: Punch, Rear Naked Choke, etc.';
COMMENT ON COLUMN Fights.Round IS 'Round in which fight ended (1-5 typically)';
COMMENT ON COLUMN Fights.Time IS 'Time in round when fight ended (MM:SS format)';
COMMENT ON COLUMN Fights.IsMainEvent IS 'Whether this was a main event fight';
COMMENT ON COLUMN Fights.IsTitleFight IS 'Whether this was a championship fight';
COMMENT ON COLUMN Fights.FlukeFlag IS 'Indicates if the fight result was considered a fluke';
COMMENT ON COLUMN Fights.SampleWeight IS 'Statistical weight for analysis purposes (e.g., recency weighting)';

COMMENT ON COLUMN Matchups.PredictedWinProbability IS 'Model-predicted probability of predicted winner winning (0-100)';
COMMENT ON COLUMN Matchups.ModelVersion IS 'Version identifier for the model that made the prediction';
COMMENT ON COLUMN Matchups.CreatedAt IS 'When the prediction was made';
COMMENT ON COLUMN Matchups.UpdatedAt IS 'When the record was last updated (e.g., with actual results)';

-- EloHistory column comments
COMMENT ON COLUMN EloHistory.EloBeforeFight IS 'Fighter ELO rating BEFORE this fight (use for predictions)';
COMMENT ON COLUMN EloHistory.OpponentEloBeforeFight IS 'Opponent ELO at fight time (for quality analysis)';
COMMENT ON COLUMN EloHistory.EloAfterFight IS 'Fighter ELO rating AFTER this fight';
COMMENT ON COLUMN EloHistory.EloChange IS 'Change in ELO from this fight (positive=gained, negative=lost)';
COMMENT ON COLUMN EloHistory.ExpectedWinProb IS 'Pre-fight win probability based on ELO differential (0-1)';

-- PreUfcCareer column comments
COMMENT ON COLUMN PreUfcCareer.OrgQualityTier IS 'Organization quality 1-5: 5=Elite(CW,LFA), 4=Strong, 3=Mid, 2=Low, 1=Unknown';
COMMENT ON COLUMN PreUfcCareer.PreUfcFinishRate IS 'Percentage of pre-UFC wins by finish (KO+Sub)/Wins';
COMMENT ON COLUMN PreUfcCareer.EstimatedInitialElo IS 'ELO estimate for UFC debut based on pre-UFC career';
COMMENT ON COLUMN PreUfcCareer.EloEstimationBreakdown IS 'JSON breakdown of factors used in ELO estimation';
COMMENT ON COLUMN PreUfcCareer.DataConfidence IS 'Confidence in pre-UFC data: high, medium, low';

-- OpponentQuality column comments  
COMMENT ON COLUMN OpponentQuality.AvgOpponentEloAtFightTime IS 'More accurate than AvgOpponentElo - uses historical ELO';
COMMENT ON COLUMN OpponentQuality.EliteOpponentWins IS 'Wins against elite opponents (1650+ ELO)';
COMMENT ON COLUMN OpponentQuality.QualityWinIndex IS 'Weighted score: Elite wins worth more than average wins';
COMMENT ON COLUMN OpponentQuality.ScheduleStrengthPercentile IS 'Percentile rank of schedule strength (0-100)';

-- MatchupFeatures column comments
COMMENT ON COLUMN MatchupFeatures.HeightDiff_cm IS 'Fighter1 height - Fighter2 height (positive = F1 taller)';
COMMENT ON COLUMN MatchupFeatures.ReachDiff_cm IS 'Fighter1 reach - Fighter2 reach (positive = F1 longer)';
COMMENT ON COLUMN MatchupFeatures.EloDiff IS 'Fighter1 ELO - Fighter2 ELO (positive = F1 higher rated)';
COMMENT ON COLUMN MatchupFeatures.StyleMatchupAdvantage IS '-1=F2 advantage, 0=neutral, 1=F1 advantage based on styles';
COMMENT ON COLUMN MatchupFeatures.IsStale IS 'True if underlying fighter data changed since calculation';

-- ============================================================================
-- PREDICTION TRACKING TABLE - Track predictions vs actual results
-- ============================================================================
-- Stores card predictions and their outcomes for model validation.
-- Auto-created by predict_card.py, updated by update_results.py.
CREATE TABLE IF NOT EXISTS PredictionTracking (
    id SERIAL PRIMARY KEY,
    event_name VARCHAR(200) NOT NULL,
    event_date VARCHAR(50),
    event_url VARCHAR(255),
    fighter1_name VARCHAR(100) NOT NULL,
    fighter2_name VARCHAR(100) NOT NULL,
    fighter1_id INTEGER,
    fighter2_id INTEGER,
    is_title_fight BOOLEAN DEFAULT FALSE,
    -- Prediction details
    pick_name VARCHAR(100) NOT NULL,
    pick_fighter_id INTEGER,
    pick_probability DECIMAL(5,4) NOT NULL,     -- 0.0 to 1.0
    fighter1_probability DECIMAL(5,4) NOT NULL,  -- prob for fighter1 specifically
    confidence VARCHAR(10) NOT NULL,             -- HIGH, MED, LOW, TOSS
    prob_source VARCHAR(50),                     -- ML+ELO (50% ML), ELO
    -- Method & round predictions
    predicted_method VARCHAR(50),                -- KO/TKO, Sub, Dec
    predicted_ko DECIMAL(5,4),
    predicted_sub DECIMAL(5,4),
    predicted_dec DECIMAL(5,4),
    predicted_r1 DECIMAL(5,4),
    predicted_r2 DECIMAL(5,4),
    predicted_r3 DECIMAL(5,4),
    -- ELO context
    fighter1_elo DECIMAL(8,2),
    fighter2_elo DECIMAL(8,2),
    -- Actual results (filled after event by update_results.py)
    actual_winner_name VARCHAR(100),
    actual_winner_id INTEGER,
    actual_method VARCHAR(50),
    actual_round INTEGER,
    was_correct BOOLEAN,
    -- Metadata
    prediction_type VARCHAR(10) DEFAULT 'live', -- 'backtest' or 'live'
    model_version VARCHAR(50) DEFAULT 'v3',
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pt_event_name ON PredictionTracking(event_name);
CREATE INDEX IF NOT EXISTS idx_pt_was_correct ON PredictionTracking(was_correct) WHERE was_correct IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pt_prediction_type ON PredictionTracking(prediction_type);

COMMENT ON TABLE PredictionTracking IS 'Tracks card predictions vs actual results for model validation and performance tracking';
COMMENT ON COLUMN PredictionTracking.pick_probability IS 'Probability of the picked winner (0.0 to 1.0)';
COMMENT ON COLUMN PredictionTracking.fighter1_probability IS 'Probability assigned to fighter1 (for directional reference)';
COMMENT ON COLUMN PredictionTracking.confidence IS 'Confidence tier: HIGH (70%+), MED (60-70%), LOW (55-60%), TOSS (50-55%)';
COMMENT ON COLUMN PredictionTracking.was_correct IS 'TRUE if pick matched actual winner, FALSE otherwise, NULL if unresolved';

-- ============================================================================
-- USEFUL VIEWS FOR ML FEATURE ENGINEERING
-- ============================================================================

-- View: Get fighter ELO at any point in time
CREATE OR REPLACE VIEW v_fighter_elo_timeline AS
SELECT 
    eh.FighterID,
    fs.Name as FighterName,
    eh.FightDate,
    eh.EloBeforeFight,
    eh.EloAfterFight,
    eh.EloChange,
    eh.Result,
    eh.OpponentID,
    fs2.Name as OpponentName,
    eh.OpponentEloBeforeFight,
    eh.ExpectedWinProb
FROM EloHistory eh
JOIN FighterStats fs ON eh.FighterID = fs.FighterID
LEFT JOIN FighterStats fs2 ON eh.OpponentID = fs2.FighterID
ORDER BY eh.FighterID, eh.FightDate;

-- View: Fighter summary with quality metrics
CREATE OR REPLACE VIEW v_fighter_quality_summary AS
SELECT 
    fs.FighterID,
    fs.Name,
    fs.WeightClass,
    cs.EloRating,
    cs.PeakEloRating,
    oq.AvgOpponentElo,
    oq.EliteOpponentWins,
    oq.EliteOpponentLosses,
    oq.QualityWinIndex,
    oq.ScheduleStrengthPercentile,
    puc.OrgQualityTier as PreUfcOrgQuality,
    puc.EstimatedInitialElo
FROM FighterStats fs
LEFT JOIN CareerStats cs ON fs.FighterID = cs.FighterID
LEFT JOIN OpponentQuality oq ON fs.FighterID = oq.FighterID
LEFT JOIN PreUfcCareer puc ON fs.FighterID = puc.FighterID;
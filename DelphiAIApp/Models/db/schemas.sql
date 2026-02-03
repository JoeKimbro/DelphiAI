-- PostgreSQL Schema for Fighter Stats Database
-- Updated with recommendations for UFC predictive modeling

-- Drop tables if they exist (in correct order due to foreign keys)
DROP TABLE IF EXISTS Matchups CASCADE;
DROP TABLE IF EXISTS Fights CASCADE;
DROP TABLE IF EXISTS CareerStats CASCADE;
DROP TABLE IF EXISTS FighterStats CASCADE;

-- Create FighterStats table (main fighter information)
CREATE TABLE FighterStats (
    FighterID SERIAL PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Height VARCHAR(20),
    Weight VARCHAR(20),
    Reach VARCHAR(20),
    Stance VARCHAR(50),
    DOB DATE,
    Age INTEGER, -- Calculated from DOB or stored directly
    WeightClass VARCHAR(50),
    Style VARCHAR(100), -- e.g., "striker, wrestler, BJJ"
    TotalFights INTEGER DEFAULT 0,
    Wins INTEGER DEFAULT 0,
    Losses INTEGER DEFAULT 0,
    Draws INTEGER DEFAULT 0,
    FightUpdatedAt TIMESTAMP,
    DaysSinceLastFight INTEGER
);

-- Create CareerStats table (one-to-one relationship with FighterStats)
CREATE TABLE CareerStats (
    CSID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL UNIQUE,
    SLpM DECIMAL(5,2), -- Significant Strikes Landed per Minute
    StrAcc DECIMAL(5,2), -- Strike Accuracy (changed from VARCHAR to DECIMAL)
    SApM DECIMAL(5,2), -- Significant Strikes Absorbed per Minute
    StrDef DECIMAL(5,2), -- Strike Defense (changed from VARCHAR to DECIMAL)
    TDAvg DECIMAL(5,2), -- Takedown Average
    TDDef DECIMAL(5,2), -- Takedown Defense (changed from VARCHAR to DECIMAL)
    SubAvg DECIMAL(5,2), -- Submission Average
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
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE
);

-- Create Fights table (detailed fight information)
CREATE TABLE Fights (
    FightID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL,
    OpponentID INTEGER,
    WinnerID INTEGER,
    Date DATE,
    EventName VARCHAR(200),
    Method VARCHAR(50), -- e.g., "KO/TKO", "Submission", "Decision"
    Round INTEGER, -- Which round the fight ended
    Time VARCHAR(10), -- Time in the round (e.g., "2:34")
    IsMainEvent BOOLEAN DEFAULT FALSE,
    IsTitleFight BOOLEAN DEFAULT FALSE,
    FlukeFlag BOOLEAN DEFAULT FALSE,
    SampleWeight DECIMAL(5,2),
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    FOREIGN KEY (OpponentID) REFERENCES FighterStats(FighterID) ON DELETE SET NULL,
    FOREIGN KEY (WinnerID) REFERENCES FighterStats(FighterID) ON DELETE SET NULL
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

-- Create indexes for better query performance
CREATE INDEX idx_careerstats_fighterid ON CareerStats(FighterID);
CREATE INDEX idx_fights_fighterid ON Fights(FighterID);
CREATE INDEX idx_fights_opponentid ON Fights(OpponentID);
CREATE INDEX idx_fights_date ON Fights(Date);
CREATE INDEX idx_fights_eventname ON Fights(EventName);
CREATE INDEX idx_fighterstats_weightclass ON FighterStats(WeightClass);
CREATE INDEX idx_fighterstats_name ON FighterStats(Name);
CREATE INDEX idx_matchups_fighter1 ON Matchups(Fighter1ID);
CREATE INDEX idx_matchups_fighter2 ON Matchups(Fighter2ID);
CREATE INDEX idx_matchups_eventdate ON Matchups(EventDate);
CREATE INDEX idx_matchups_modelversion ON Matchups(ModelVersion);

-- Add comments to tables for documentation
COMMENT ON TABLE FighterStats IS 'Main table containing fighter physical attributes and basic information';
COMMENT ON TABLE CareerStats IS 'Career statistics and performance metrics for each fighter';
COMMENT ON TABLE Fights IS 'Detailed fight records including opponents, method, and outcome';
COMMENT ON TABLE Matchups IS 'Predicted and actual results for fighter matchups, used for model evaluation';

-- Add comments to important columns
COMMENT ON COLUMN FighterStats.Name IS 'Fighter full name';
COMMENT ON COLUMN FighterStats.Age IS 'Current age or age at last fight';
COMMENT ON COLUMN FighterStats.TotalFights IS 'Total number of professional fights';
COMMENT ON COLUMN FighterStats.DaysSinceLastFight IS 'Days since last fight (for ring rust analysis)';

COMMENT ON COLUMN CareerStats.SLpM IS 'Significant Strikes Landed per Minute';
COMMENT ON COLUMN CareerStats.StrAcc IS 'Strike Accuracy (0-100)';
COMMENT ON COLUMN CareerStats.SApM IS 'Significant Strikes Absorbed per Minute';
COMMENT ON COLUMN CareerStats.StrDef IS 'Strike Defense percentage (0-100)';
COMMENT ON COLUMN CareerStats.TDAvg IS 'Takedown Average per 15 minutes';
COMMENT ON COLUMN CareerStats.TDDef IS 'Takedown Defense percentage (0-100)';
COMMENT ON COLUMN CareerStats.SubAvg IS 'Submission Average per 15 minutes';
COMMENT ON COLUMN CareerStats.AvgFightDuration IS 'Average fight duration in minutes';
COMMENT ON COLUMN CareerStats.FirstRoundFinishRate IS 'Percentage of fights finished in first round';
COMMENT ON COLUMN CareerStats.DecisionRate IS 'Percentage of fights going to decision';
COMMENT ON COLUMN CareerStats.EloRating IS 'Elo rating for fighter strength (opponent-adjusted)';

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
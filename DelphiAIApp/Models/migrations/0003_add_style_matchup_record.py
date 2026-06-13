"""Create StyleMatchupRecord table (missed in baseline schema)."""
from yoyo import step

steps = [
    step(
        """
        CREATE TABLE IF NOT EXISTS StyleMatchupRecord (
            id SERIAL PRIMARY KEY,
            FighterID INTEGER NOT NULL,
            FighterURL VARCHAR(255),
            OpponentStyle VARCHAR(50) NOT NULL,
            Wins INTEGER DEFAULT 0,
            Losses INTEGER DEFAULT 0,
            Draws INTEGER DEFAULT 0,
            TotalFights INTEGER DEFAULT 0,
            WinRate DECIMAL(5,2),
            KOWins INTEGER DEFAULT 0,
            SubWins INTEGER DEFAULT 0,
            DecWins INTEGER DEFAULT 0,
            KOLosses INTEGER DEFAULT 0,
            SubLosses INTEGER DEFAULT 0,
            DecLosses INTEGER DEFAULT 0,
            AvgFightDuration DECIMAL(6,2),
            CalculatedAt TIMESTAMP,
            CONSTRAINT unique_fighter_opp_style UNIQUE (FighterID, OpponentStyle),
            FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_stylematchup_fighterid ON StyleMatchupRecord(FighterID);
        """,
        "DROP TABLE IF EXISTS StyleMatchupRecord",
    )
]

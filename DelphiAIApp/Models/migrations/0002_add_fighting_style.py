"""Add FightingStyle column to FighterStats (missed in baseline schema)."""
from yoyo import step

steps = [
    step(
        "ALTER TABLE FighterStats ADD COLUMN IF NOT EXISTS FightingStyle VARCHAR(50)",
        "ALTER TABLE FighterStats DROP COLUMN IF EXISTS FightingStyle",
    )
]

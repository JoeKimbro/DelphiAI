"""Debug the ELO direction issue"""
import pandas as pd
import psycopg2
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5433'),
    dbname=os.getenv('DB_NAME', 'delphi_db'),
    user=os.getenv('DB_USER', ''),
    password=os.getenv('DB_PASSWORD', ''),
)

print("="*60)
print("DEBUGGING ELO DIRECTION")
print("="*60)

# Get a few sample fights with their ELO
sample = pd.read_sql('''
    SELECT 
        f.FightID, 
        f.Date,
        f.FighterName,
        f.FighterID,
        f.OpponentName,
        f.OpponentID,
        f.Result,
        f.WinnerID,
        e1.EloBeforeFight as f1_elo,
        e2.EloBeforeFight as f2_elo
    FROM Fights f
    JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    WHERE f.WinnerID IS NOT NULL
    ORDER BY f.Date DESC
    LIMIT 20
''', conn)

print("\n=== SAMPLE FIGHTS WITH ELO ===\n")
for _, row in sample.iterrows():
    f1_name = row['fightername']
    f2_name = row['opponentname']
    f1_elo = row['f1_elo']
    f2_elo = row['f2_elo']
    result = row['result']
    winner_id = row['winnerid']
    fighter_id = row['fighterid']
    opponent_id = row['opponentid']
    
    elo_diff = f1_elo - f2_elo
    f1_higher_elo = elo_diff > 0
    f1_won = winner_id == fighter_id
    
    # Who should win based on ELO?
    predicted_winner = f1_name if f1_higher_elo else f2_name
    actual_winner = f1_name if f1_won else f2_name
    correct = (f1_higher_elo and f1_won) or (not f1_higher_elo and not f1_won)
    
    print(f"{row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}")
    print(f"  {f1_name} (ELO {f1_elo:.0f}) vs {f2_name} (ELO {f2_elo:.0f})")
    print(f"  ELO diff: {elo_diff:.0f} | Predicted: {predicted_winner} | Actual: {actual_winner}")
    print(f"  Result column: {result} | Correct: {'YES' if correct else 'NO'}")
    print()

# Check the Result column values and their meaning
print("\n=== RESULT COLUMN ANALYSIS ===")
results = pd.read_sql('''
    SELECT Result, COUNT(*) as count
    FROM Fights
    GROUP BY Result
''', conn)
print(results)

# Check: when Result='win', does FighterID == WinnerID?
print("\n=== CHECKING RESULT CONSISTENCY ===")
consistency = pd.read_sql('''
    SELECT 
        Result,
        CASE 
            WHEN Result = 'win' AND FighterID = WinnerID THEN 'CORRECT'
            WHEN Result = 'loss' AND OpponentID = WinnerID THEN 'CORRECT'
            WHEN Result = 'win' AND FighterID != WinnerID THEN 'MISMATCH'
            WHEN Result = 'loss' AND OpponentID != WinnerID THEN 'MISMATCH'
            ELSE 'OTHER'
        END as consistency,
        COUNT(*) as count
    FROM Fights
    WHERE WinnerID IS NOT NULL
    GROUP BY Result, consistency
''', conn)
print(consistency)

conn.close()

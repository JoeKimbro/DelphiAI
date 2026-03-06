"""Check if there's a position bias in the data"""
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
print("CHECKING POSITION BIAS")
print("="*60)

# Check overall win rate by position
print("\n=== FIGHTER1 WIN RATE (by Result column) ===")
result_counts = pd.read_sql('''
    SELECT Result, COUNT(*) as count
    FROM Fights
    WHERE Result IN ('win', 'loss')
    GROUP BY Result
''', conn)
print(result_counts)

total_fights = result_counts['count'].sum()
wins = result_counts[result_counts['result'] == 'win']['count'].values[0]
print(f"\nFighter1 (listed first) win rate: {wins/total_fights*100:.1f}%")

# Check if Fighter1 tends to have higher/lower ELO
print("\n=== ELO COMPARISON BY POSITION ===")
elo_comparison = pd.read_sql('''
    SELECT 
        AVG(e1.EloBeforeFight) as avg_f1_elo,
        AVG(e2.EloBeforeFight) as avg_f2_elo,
        COUNT(*) as fights
    FROM Fights f
    JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
''', conn)
print(f"Average Fighter1 ELO: {elo_comparison['avg_f1_elo'].values[0]:.1f}")
print(f"Average Fighter2 ELO: {elo_comparison['avg_f2_elo'].values[0]:.1f}")

# Check ELO prediction accuracy considering position bias
print("\n=== ELO PREDICTION ACCOUNTING FOR POSITION ===")
fights = pd.read_sql('''
    SELECT 
        f.FightID,
        f.Result,
        e1.EloBeforeFight as f1_elo,
        e2.EloBeforeFight as f2_elo
    FROM Fights f
    JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    WHERE f.Result IN ('win', 'loss')
''', conn)

fights['elo_diff'] = fights['f1_elo'] - fights['f2_elo']
fights['f1_wins'] = fights['result'] == 'win'
fights['f1_higher_elo'] = fights['elo_diff'] > 0

# Accuracy when Fighter1 has higher ELO
f1_higher = fights[fights['f1_higher_elo']]
print(f"\nWhen Fighter1 has higher ELO ({len(f1_higher)} fights):")
print(f"  Fighter1 actually wins: {f1_higher['f1_wins'].mean()*100:.1f}%")

# Accuracy when Fighter2 has higher ELO  
f2_higher = fights[~fights['f1_higher_elo']]
print(f"\nWhen Fighter2 has higher ELO ({len(f2_higher)} fights):")
print(f"  Fighter2 actually wins: {(~f2_higher['f1_wins']).mean()*100:.1f}%")

# Combined: higher ELO wins
fights['higher_elo_wins'] = (fights['f1_higher_elo'] & fights['f1_wins']) | (~fights['f1_higher_elo'] & ~fights['f1_wins'])
print(f"\nOverall 'Higher ELO wins' accuracy: {fights['higher_elo_wins'].mean()*100:.1f}%")

# But what if we check when ELO diff is meaningful (> 30)?
meaningful = fights[fights['elo_diff'].abs() > 30]
print(f"\nWhen |ELO diff| > 30 ({len(meaningful)} fights):")
meaningful_higher_wins = (meaningful['f1_higher_elo'] & meaningful['f1_wins']) | (~meaningful['f1_higher_elo'] & ~meaningful['f1_wins'])
print(f"  Higher ELO wins: {meaningful_higher_wins.mean()*100:.1f}%")

# When ELO diff > 50
meaningful50 = fights[fights['elo_diff'].abs() > 50]
print(f"\nWhen |ELO diff| > 50 ({len(meaningful50)} fights):")
m50_higher_wins = (meaningful50['f1_higher_elo'] & meaningful50['f1_wins']) | (~meaningful50['f1_higher_elo'] & ~meaningful50['f1_wins'])
print(f"  Higher ELO wins: {m50_higher_wins.mean()*100:.1f}%")

# When ELO diff > 100
meaningful100 = fights[fights['elo_diff'].abs() > 100]
print(f"\nWhen |ELO diff| > 100 ({len(meaningful100)} fights):")
m100_higher_wins = (meaningful100['f1_higher_elo'] & meaningful100['f1_wins']) | (~meaningful100['f1_higher_elo'] & ~meaningful100['f1_wins'])
print(f"  Higher ELO wins: {m100_higher_wins.mean()*100:.1f}%")

conn.close()

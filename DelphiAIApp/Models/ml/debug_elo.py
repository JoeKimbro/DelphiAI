"""Debug ELO lookup issue."""
import os
import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment - same path as train_model.py
env_path = Path(__file__).parent.parent.parent.parent / '.env'
print(f"Looking for .env at: {env_path}")
print(f"Exists: {env_path.exists()}")

if env_path.exists():
    load_dotenv(env_path)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'dbname': os.getenv('DB_NAME', 'delphi_db'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
}

print(f"Connecting to: {DB_CONFIG}")

conn = psycopg2.connect(**DB_CONFIG)

# Check EloHistory data
print("\n=== EloHistory Sample ===")
elo = pd.read_sql('SELECT * FROM EloHistory ORDER BY fightdate LIMIT 10', conn)
print(elo[['fighterid', 'fightdate', 'elobeforefight', 'eloafterfight']].to_string())

print("\n=== EloHistory Stats ===")
stats = pd.read_sql('''
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT fighterid) as unique_fighters,
        MIN(fightdate) as min_date,
        MAX(fightdate) as max_date,
        MIN(elobeforefight) as min_elo,
        MAX(elobeforefight) as max_elo,
        AVG(elobeforefight) as avg_elo
    FROM EloHistory
''', conn)
print(stats.to_string())

print("\n=== Sample Fight from Fights table ===")
fight = pd.read_sql('''
    SELECT fighterid, opponentid, date, fightername, opponentname
    FROM Fights 
    WHERE date IS NOT NULL 
    ORDER BY date DESC 
    LIMIT 3
''', conn)
print(fight.to_string())

# Try to match a fight to EloHistory
print("\n=== Attempting ELO Lookup ===")
if len(fight) > 0:
    fighter_id = fight.iloc[0]['fighterid']
    fight_date = fight.iloc[0]['date']
    print(f"Looking up fighter {fighter_id} on date {fight_date}")
    
    elo_lookup = pd.read_sql(f'''
        SELECT * FROM EloHistory 
        WHERE fighterid = {fighter_id} 
        AND fightdate <= '{fight_date}'
        ORDER BY fightdate DESC
        LIMIT 3
    ''', conn)
    print(elo_lookup.to_string() if len(elo_lookup) > 0 else "No ELO history found!")

conn.close()

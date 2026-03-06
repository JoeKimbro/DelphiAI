"""Debug why training model ELO lookup differs"""
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
print("DEBUG: Training vs Direct SQL ELO Lookup")
print("="*60)

# Load data like training script does
fights_df = pd.read_sql('''
    SELECT 
        FightID, Date as fight_date, FighterID as fighter1_id, 
        OpponentID as fighter2_id, WinnerID, Result
    FROM Fights
    WHERE Date IS NOT NULL
''', conn)

elo_df = pd.read_sql('''
    SELECT FighterID, FightDate, EloBeforeFight, OpponentEloBeforeFight
    FROM EloHistory
''', conn)

print(f"Fights: {len(fights_df)}")
print(f"ELO History: {len(elo_df)}")

# Convert dates
fights_df['fight_date'] = pd.to_datetime(fights_df['fight_date'])
elo_df['fightdate'] = pd.to_datetime(elo_df['fightdate'])

# Simulate training script's get_elo_at_fight_time
def get_elo_at_fight_time(fighter_id, fight_date, elo_df):
    if pd.isna(fighter_id) or pd.isna(fight_date):
        return 1500.0
    
    fighter_elo = elo_df[
        (elo_df['fighterid'] == fighter_id) & 
        (elo_df['fightdate'] <= fight_date)
    ].sort_values('fightdate', ascending=False)
    
    if len(fighter_elo) > 0:
        return fighter_elo.iloc[0]['elobeforefight']
    
    return 1500.0

# Sample 100 fights
sample = fights_df.sample(n=min(100, len(fights_df)), random_state=42)

at_default = 0
matched = 0

for _, fight in sample.iterrows():
    f1_elo = get_elo_at_fight_time(fight['fighter1_id'], fight['fight_date'], elo_df)
    f2_elo = get_elo_at_fight_time(fight['fighter2_id'], fight['fight_date'], elo_df)
    
    if f1_elo == 1500.0:
        at_default += 1
    else:
        matched += 1

print(f"\nSample of 100 fights:")
print(f"  Fighter1 matched ELO: {matched}/100")
print(f"  Fighter1 at default 1500: {at_default}/100")

# Check specifically: does the training script's method return DIFFERENT values
# than a direct SQL join?
print("\n=== COMPARISON: Training Lookup vs Direct SQL ===")

# Pick 10 specific fights
test_fights = fights_df.head(10)

for _, fight in test_fights.iterrows():
    fight_id = fight['fightid']
    f1_id = fight['fighter1_id']
    fight_date = fight['fight_date']
    
    # Training script method
    training_elo = get_elo_at_fight_time(f1_id, fight_date, elo_df)
    
    # Direct SQL
    direct = pd.read_sql(f'''
        SELECT EloBeforeFight 
        FROM EloHistory 
        WHERE FighterID = {f1_id} AND FightDate = '{fight_date.strftime("%Y-%m-%d")}'
    ''', conn)
    
    direct_elo = direct['elobeforefight'].values[0] if len(direct) > 0 else None
    
    match = "MATCH" if training_elo == direct_elo else "DIFFERENT"
    print(f"  Fight {fight_id}: Training={training_elo:.0f}, Direct={direct_elo}, {match}")

# KEY CHECK: How many fights have 1500 ELO in training features?
print("\n=== CHECKING 1500 ELO FREQUENCY IN TRAINING ===")

f1_at_1500 = 0
f2_at_1500 = 0
both_at_1500 = 0
neither_at_1500 = 0

for _, fight in fights_df.iterrows():
    f1_elo = get_elo_at_fight_time(fight['fighter1_id'], fight['fight_date'], elo_df)
    f2_elo = get_elo_at_fight_time(fight['fighter2_id'], fight['fight_date'], elo_df)
    
    if f1_elo == 1500.0:
        f1_at_1500 += 1
    if f2_elo == 1500.0:
        f2_at_1500 += 1
    if f1_elo == 1500.0 and f2_elo == 1500.0:
        both_at_1500 += 1
    if f1_elo != 1500.0 and f2_elo != 1500.0:
        neither_at_1500 += 1

total = len(fights_df)
print(f"Total fights: {total}")
print(f"Fighter1 at default 1500: {f1_at_1500} ({f1_at_1500/total*100:.1f}%)")
print(f"Fighter2 at default 1500: {f2_at_1500} ({f2_at_1500/total*100:.1f}%)")
print(f"BOTH at default 1500 (ELO diff = 0): {both_at_1500} ({both_at_1500/total*100:.1f}%)")
print(f"Neither at 1500 (actual ELO data): {neither_at_1500} ({neither_at_1500/total*100:.1f}%)")

conn.close()

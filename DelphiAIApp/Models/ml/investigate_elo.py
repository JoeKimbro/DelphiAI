"""Investigate why ELO baseline is only 53%"""
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
print("INVESTIGATING ELO BASELINE PERFORMANCE")
print("="*60)

# Check 1: ELO distribution in history
print('\n=== ELO HISTORY ANALYSIS ===')
elo = pd.read_sql('SELECT * FROM EloHistory', conn)
print(f'Total ELO history records: {len(elo)}')
print(f'Unique fighters with ELO history: {elo["fighterid"].nunique()}')
print(f'ELO before fight range: {elo["elobeforefight"].min():.0f} to {elo["elobeforefight"].max():.0f}')
print(f'Mean ELO before fight: {elo["elobeforefight"].mean():.0f}')
print(f'Std ELO: {elo["elobeforefight"].std():.0f}')

# Check records at 1500 (first fights)
at_1500 = (elo['elobeforefight'] == 1500).sum()
print(f'\nRecords at exactly 1500 (first fights): {at_1500} ({at_1500/len(elo)*100:.1f}%)')

# Check 2: How many fights have BOTH fighters with prior ELO history
print('\n=== POINT-IN-TIME ELO MATCHING ===')

# Get fights with dates
fights = pd.read_sql('''
    SELECT FightID, FighterID, OpponentID, Date, Result, WinnerID
    FROM Fights 
    WHERE Date IS NOT NULL
    ORDER BY Date
''', conn)

print(f'Total fights with dates: {len(fights)}')

# Convert dates
fights['Date'] = pd.to_datetime(fights['date'])
elo['fightdate'] = pd.to_datetime(elo['fightdate'])

# For each fight, check if we can find PRIOR ELO history
def get_prior_elo(fighter_id, fight_date, elo_df):
    """Get ELO BEFORE this fight (not at this fight)"""
    if pd.isna(fighter_id):
        return None
    prior = elo_df[(elo_df['fighterid'] == fighter_id) & (elo_df['fightdate'] < fight_date)]
    if len(prior) > 0:
        return prior.iloc[-1]['eloafterfight']  # Use ELO after their last fight
    return None

# Sample check on recent fights
print('\nChecking ELO lookup for recent test set fights...')
test_fights = fights[fights['Date'] >= '2022-07-16'].head(100)

has_f1_elo = 0
has_f2_elo = 0
has_both = 0

for _, fight in test_fights.iterrows():
    f1_elo = get_prior_elo(fight['fighterid'], fight['Date'], elo)
    f2_elo = get_prior_elo(fight['opponentid'], fight['Date'], elo)
    
    if f1_elo is not None:
        has_f1_elo += 1
    if f2_elo is not None:
        has_f2_elo += 1
    if f1_elo is not None and f2_elo is not None:
        has_both += 1

print(f'Sample of 100 test fights:')
print(f'  Fighter1 has prior ELO: {has_f1_elo}/100')
print(f'  Fighter2 has prior ELO: {has_f2_elo}/100')
print(f'  BOTH have prior ELO: {has_both}/100')

# Check 3: Theoretical baseline accuracy
print('\n=== THEORETICAL BASELINE CHECK ===')

# Use ELO at fight time from history
fights_with_elo = pd.read_sql('''
    SELECT f.FightID, f.Date, f.WinnerID, f.FighterID, f.OpponentID,
           e.EloBeforeFight as f1_elo_at_fight
    FROM Fights f
    JOIN EloHistory e ON f.FighterID = e.FighterID AND f.Date = e.FightDate
    WHERE f.Date IS NOT NULL AND f.WinnerID IS NOT NULL
''', conn)

print(f'Fights where Fighter1 has ELO record at fight time: {len(fights_with_elo)}')

# Now get Fighter2's ELO at fight time
f2_elo = pd.read_sql('''
    SELECT f.FightID, e.EloBeforeFight as f2_elo_at_fight
    FROM Fights f
    JOIN EloHistory e ON f.OpponentID = e.FighterID AND f.Date = e.FightDate
''', conn)

merged = fights_with_elo.merge(f2_elo, on='fightid', how='inner')
print(f'Fights where BOTH have ELO at fight time: {len(merged)}')

if len(merged) > 0:
    merged['elo_diff'] = merged['f1_elo_at_fight'] - merged['f2_elo_at_fight']
    merged['f1_wins'] = merged['winnerid'] == merged['fighterid']
    merged['higher_elo_wins'] = ((merged['elo_diff'] > 0) & merged['f1_wins']) | ((merged['elo_diff'] < 0) & ~merged['f1_wins'])
    
    # Accuracy
    accuracy = merged['higher_elo_wins'].mean()
    print(f'\nHigher ELO wins accuracy: {accuracy:.1%}')
    
    # ELO diff distribution
    print(f'\nELO diff statistics:')
    print(f'  Mean: {merged["elo_diff"].mean():.1f}')
    print(f'  Std: {merged["elo_diff"].std():.1f}')
    print(f'  |diff| < 30: {(merged["elo_diff"].abs() < 30).sum()} ({(merged["elo_diff"].abs() < 30).mean()*100:.1f}%)')
    print(f'  |diff| < 50: {(merged["elo_diff"].abs() < 50).sum()} ({(merged["elo_diff"].abs() < 50).mean()*100:.1f}%)')
    
    # Breakdown by ELO diff magnitude
    print('\n=== ACCURACY BY ELO DIFF MAGNITUDE ===')
    bins = [0, 25, 50, 100, 200, 500]
    for i in range(len(bins)-1):
        mask = (merged['elo_diff'].abs() >= bins[i]) & (merged['elo_diff'].abs() < bins[i+1])
        subset = merged[mask]
        if len(subset) > 0:
            acc = subset['higher_elo_wins'].mean()
            print(f'  |diff| {bins[i]}-{bins[i+1]}: {len(subset)} fights, {acc:.1%} accuracy')

conn.close()
print('\n[DONE]')

"""Quick analysis of fighters missing ELO ratings"""
import pandas as pd

fighters = pd.read_csv('output/fighters.csv')
career = pd.read_csv('output/career_stats.csv')
fights = pd.read_csv('output/fights.csv')

# Find fighters with bad ELO (either NaN or 'merged')
missing_elo = career[(career['elo_rating'].isna()) | (career['elo_rating'] == 'merged')]
print('=== Fighters Missing/Bad ELO ===')
print(f'Total: {len(missing_elo)}')

# Merge with fighters to get more info
merged = missing_elo.merge(fighters[['fighter_url', 'name', 'wins', 'losses']], on='fighter_url', how='left')
print()
print('Sample of missing ELO fighters:')
for idx, row in merged[['name', 'wins', 'losses', 'fighter_url']].head(20).iterrows():
    print(f"  {row['name']}: {row['wins']}-{row['losses']}")

# Check if they appear in fights.csv at all
fighter_urls_missing = set(missing_elo['fighter_url'].dropna())

all_fight_urls = set(fights['fighter_url'].dropna()) | set(fights['opponent_url'].dropna())
in_fights = fighter_urls_missing & all_fight_urls
not_in_fights = fighter_urls_missing - all_fight_urls

print()
print(f'Missing ELO fighters WITH UFC fight data: {len(in_fights)}')
print(f'Missing ELO fighters WITHOUT any fight data: {len(not_in_fights)}')

# For those in fights, check how many fights they have
if len(in_fights) > 0:
    print()
    print('=== Fighters with fight data but no ELO (odd cases) ===')
    for url in list(in_fights)[:10]:
        as_fighter = len(fights[fights['fighter_url'] == url])
        as_opponent = len(fights[fights['opponent_url'] == url])
        name = merged[merged['fighter_url'] == url]['name'].values[0] if len(merged[merged['fighter_url'] == url]) > 0 else 'Unknown'
        print(f"  {name}: {as_fighter + as_opponent} fights in DB")

# Sample UFCStats URLs to check
print()
print('=== Sample UFCStats URLs to check for pre-UFC data ===')
sample_urls = list(not_in_fights)[:5]
for url in sample_urls:
    name = merged[merged['fighter_url'] == url]['name'].values[0] if len(merged[merged['fighter_url'] == url]) > 0 else 'Unknown'
    print(f"  {name}: {url}")

# Check pre-UFC records from UFC.com data
print()
print('=== Pre-UFC Records (from UFC.com) ===')
full_merge = missing_elo.merge(fighters[['fighter_url', 'name', 'wins', 'losses', 'draws']], on='fighter_url', how='left')
for idx, row in full_merge[['name', 'wins', 'losses', 'draws']].head(20).iterrows():
    w = int(row['wins']) if pd.notna(row['wins']) else 0
    l = int(row['losses']) if pd.notna(row['losses']) else 0
    d = int(row['draws']) if pd.notna(row['draws']) else 0
    name = row['name'] if pd.notna(row['name']) else 'Unknown'
    print(f"  {name}: {w}-{l}-{d}")

"""Check what data is available for pre-UFC ELO fighters"""
import pandas as pd

fighters = pd.read_csv('output/fighters.csv')
career = pd.read_csv('output/career_stats.csv')
elo = pd.read_csv('output/elo_ratings.csv')

# Get pre-UFC fighters
pre_ufc = elo[elo['elo_source'] == 'pre_ufc_estimate']
pre_ufc_urls = set(pre_ufc['fighter_url'])

# Filter to pre-UFC fighters
pre_ufc_fighters = fighters[fighters['fighter_url'].isin(pre_ufc_urls)]
pre_ufc_career = career[career['fighter_url'].isin(pre_ufc_urls)]

print("=== Data Available for Pre-UFC Fighters ===")
print(f"Total pre-UFC fighters: {len(pre_ufc_urls)}")
print()

# Check fighters.csv data availability
print("From fighters.csv:")
for col in ['age', 'days_since_last_fight', 'last_fight_date', 'total_fights']:
    non_null = pre_ufc_fighters[col].notna().sum()
    print(f"  {col}: {non_null}/{len(pre_ufc_fighters)} ({100*non_null/len(pre_ufc_fighters):.0f}%)")

print()
print("From career_stats.csv:")
for col in ['first_round_finish_rate', 'decision_rate', 'wins_by_ko_last5', 'wins_by_sub_last5', 'slpm', 'sub_avg']:
    non_null = pre_ufc_career[col].notna().sum()
    pct = 100*non_null/len(pre_ufc_career) if len(pre_ufc_career) > 0 else 0
    print(f"  {col}: {non_null}/{len(pre_ufc_career)} ({pct:.0f}%)")

# Sample the data
print()
print("=== Sample Pre-UFC Fighter Data ===")
sample = pre_ufc_fighters[['name', 'age', 'wins', 'losses', 'days_since_last_fight', 'total_fights']].head(10)
print(sample.to_string())

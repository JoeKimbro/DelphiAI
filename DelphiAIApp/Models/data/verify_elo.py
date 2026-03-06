"""Verify ELO calculation results"""
import pandas as pd

career = pd.read_csv('output/career_stats.csv')
elo = pd.read_csv('output/elo_ratings.csv')
fighters = pd.read_csv('output/fighters.csv')

# Check for bad ELO values
bad_elo = career[career['elo_rating'] == 'merged']
print('Fighters with "merged" as ELO:', len(bad_elo))

# Check ELO stats
print()
print('ELO column type:', career['elo_rating'].dtype)
print('Null ELO count:', career['elo_rating'].isna().sum())

# Check pre-UFC estimates
pre_ufc = elo[elo['elo_source'] == 'pre_ufc_estimate']
print()
print('=== Pre-UFC ELO Estimates ===')
print('Total:', len(pre_ufc))
print('ELO range:', pre_ufc['elo_rating'].min(), '-', pre_ufc['elo_rating'].max())
print('Mean ELO:', round(pre_ufc['elo_rating'].mean(), 2))

# Sample some pre-UFC estimates
merged = pre_ufc.merge(fighters[['fighter_url', 'name', 'wins', 'losses']], on='fighter_url', how='left')
print()
print('Sample pre-UFC ELO estimates:')
for _, row in merged.head(15).iterrows():
    w = int(row['wins']) if pd.notna(row['wins']) else 0
    l = int(row['losses']) if pd.notna(row['losses']) else 0
    name = row['name'] if pd.notna(row['name']) else 'Unknown'
    print(f"  {name}: {w}-{l} -> ELO {row['elo_rating']:.0f}")

# Check total ELO coverage
print()
print('=== ELO Coverage ===')
print(f"Fighters in career_stats: {len(career)}")
print(f"Fighters with ELO: {career['elo_rating'].notna().sum()}")
print(f"Coverage: {100 * career['elo_rating'].notna().sum() / len(career):.1f}%")

"""Compare enhanced vs simple pre-UFC ELO estimates"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from features import PreUfcEloEstimator

fighters = pd.read_csv('output/fighters.csv')
elo = pd.read_csv('output/elo_ratings.csv')

# Get pre-UFC fighters
pre_ufc = elo[elo['elo_source'] == 'pre_ufc_estimate']
pre_ufc_urls = set(pre_ufc['fighter_url'])

# Filter to pre-UFC fighters
pre_ufc_fighters = fighters[fighters['fighter_url'].isin(pre_ufc_urls)].copy()

# Create estimator
estimator = PreUfcEloEstimator()

print("=" * 80)
print("ENHANCED PRE-UFC ELO ESTIMATION - COMPARISON")
print("=" * 80)

comparisons = []

for _, row in pre_ufc_fighters.iterrows():
    wins = int(row['wins']) if pd.notna(row['wins']) else 0
    losses = int(row['losses']) if pd.notna(row['losses']) else 0
    draws = int(row['draws']) if pd.notna(row['draws']) else 0
    age = float(row['age']) if pd.notna(row['age']) else None
    days_since = float(row['days_since_last_fight']) if pd.notna(row['days_since_last_fight']) else None
    
    # Simple estimate (record only)
    simple_result = estimator.estimate_elo(wins, losses, draws)
    simple_elo = simple_result['elo']
    
    # Enhanced estimate (with age, activity)
    enhanced_result = estimator.estimate_elo(
        wins, losses, draws,
        age=age,
        days_since_last_fight=days_since
    )
    enhanced_elo = enhanced_result['elo']
    
    comparisons.append({
        'name': row['name'],
        'record': f"{wins}-{losses}-{draws}",
        'age': age,
        'days_since': days_since,
        'simple_elo': simple_elo,
        'enhanced_elo': enhanced_elo,
        'difference': enhanced_elo - simple_elo,
        'breakdown': enhanced_result['breakdown'],
    })

# Sort by difference to show impact
df = pd.DataFrame(comparisons)
df = df.sort_values('difference')

print("\n[MOST PENALIZED by enhanced factors]")
print("-" * 80)
for _, row in df.head(10).iterrows():
    age_str = f"age {row['age']:.0f}" if row['age'] else "age N/A"
    days_str = f"{row['days_since']:.0f}d layoff" if row['days_since'] else "layoff N/A"
    print(f"{row['name']}: {row['record']} ({age_str}, {days_str})")
    print(f"  Simple: {row['simple_elo']:.0f} -> Enhanced: {row['enhanced_elo']:.0f} ({row['difference']:+.0f})")
    breakdown = row['breakdown']
    factors = []
    if 'age_factor' in breakdown and breakdown['age_factor'] != 0:
        factors.append(f"age: {breakdown['age_factor']:+d}")
    if 'career_efficiency' in breakdown and breakdown['career_efficiency'] != 0:
        factors.append(f"career: {breakdown['career_efficiency']:+d}")
    if 'recency' in breakdown and breakdown['recency'] != 0:
        factors.append(f"recency: {breakdown['recency']:+d}")
    if factors:
        print(f"  Factors: {', '.join(factors)}")
    print()

print("\n[MOST BOOSTED by enhanced factors]")
print("-" * 80)
for _, row in df.tail(10).iterrows():
    age_str = f"age {row['age']:.0f}" if row['age'] else "age N/A"
    days_str = f"{row['days_since']:.0f}d layoff" if row['days_since'] else "layoff N/A"
    print(f"{row['name']}: {row['record']} ({age_str}, {days_str})")
    print(f"  Simple: {row['simple_elo']:.0f} -> Enhanced: {row['enhanced_elo']:.0f} ({row['difference']:+.0f})")
    breakdown = row['breakdown']
    factors = []
    if 'age_factor' in breakdown and breakdown['age_factor'] != 0:
        factors.append(f"age: {breakdown['age_factor']:+d}")
    if 'career_efficiency' in breakdown and breakdown['career_efficiency'] != 0:
        factors.append(f"career: {breakdown['career_efficiency']:+d}")
    if 'recency' in breakdown and breakdown['recency'] != 0:
        factors.append(f"recency: {breakdown['recency']:+d}")
    if factors:
        print(f"  Factors: {', '.join(factors)}")
    print()

print("\n[SUMMARY STATISTICS]")
print("-" * 80)
print(f"Total pre-UFC fighters: {len(df)}")
print(f"Average difference: {df['difference'].mean():+.1f}")
print(f"Std deviation: {df['difference'].std():.1f}")
print(f"Max penalty: {df['difference'].min():.0f}")
print(f"Max boost: {df['difference'].max():.0f}")
print(f"Enhanced ELO range: {df['enhanced_elo'].min():.0f} - {df['enhanced_elo'].max():.0f}")

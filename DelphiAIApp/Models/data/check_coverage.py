"""Check ELO coverage breakdown"""
import pandas as pd

elo = pd.read_csv('output/elo_ratings.csv')
career = pd.read_csv('output/career_stats.csv')
fighters = pd.read_csv('output/fighters.csv')

print('=== ELO SOURCE BREAKDOWN ===')
print(elo['elo_source'].value_counts())
print()
print('Total fighters with ELO:', len(elo))

print()
print('=== COVERAGE ===')
print('Fighters in career_stats:', len(career))
print('With ELO rating:', career['elo_rating'].notna().sum())
coverage = 100 * career['elo_rating'].notna().sum() / len(career)
print(f'Coverage: {coverage:.1f}%')

print()
print('=== HOW ELO IS CALCULATED ===')
ufc_fights = elo[elo['elo_source'] == 'ufc_fights']
pre_ufc = elo[elo['elo_source'] == 'pre_ufc_estimate']

print(f'From UFC fight history: {len(ufc_fights)} fighters')
print(f'  - ELO range: {ufc_fights["elo_rating"].min():.0f} - {ufc_fights["elo_rating"].max():.0f}')
print(f'  - These fighters have actual UFC fights to calculate ELO')

print(f'\nFrom pre-UFC record estimate: {len(pre_ufc)} fighters')
print(f'  - ELO range: {pre_ufc["elo_rating"].min():.0f} - {pre_ufc["elo_rating"].max():.0f}')
print(f'  - These fighters have NO UFC fights (signed but not debuted, or legacy)')
print(f'  - Enhanced estimation uses: record + age + career length + recency')

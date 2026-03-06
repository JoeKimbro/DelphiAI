"""
Nationality/Country Analysis

Analyzes if a fighter's country of origin provides any predictive edge.

Hypotheses to test:
1. Do fighters from certain countries outperform ELO expectations?
2. Do certain country matchups have unexpected outcomes?
3. Is there a "home country" advantage when fighting in certain locations?
4. Do fighters from wrestling-heavy countries (Russia, Dagestan) have edge?
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats
import psycopg2
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'dbname': os.getenv('DB_NAME', 'delphi_db'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
}


# Country extraction patterns
COUNTRY_PATTERNS = {
    # Major MMA countries
    'USA': ['United States', 'USA', 'U.S.A', 'America', 'American'],
    'Brazil': ['Brazil', 'Brasil'],
    'Russia': ['Russia', 'Russian Federation'],
    'Dagestan': ['Dagestan'],  # Separate due to unique fighting style
    'UK': ['England', 'United Kingdom', 'UK', 'Great Britain', 'Scotland', 'Wales', 'Ireland', 'Northern Ireland'],
    'Canada': ['Canada'],
    'Mexico': ['Mexico'],
    'Australia': ['Australia'],
    'Japan': ['Japan'],
    'South Korea': ['South Korea', 'Korea'],
    'China': ['China'],
    'Poland': ['Poland'],
    'Netherlands': ['Netherlands', 'Holland'],
    'Sweden': ['Sweden'],
    'Germany': ['Germany'],
    'France': ['France'],
    'Italy': ['Italy'],
    'Spain': ['Spain'],
    'Argentina': ['Argentina'],
    'Peru': ['Peru'],
    'Chile': ['Chile'],
    'Colombia': ['Colombia'],
    'Ecuador': ['Ecuador'],
    'Venezuela': ['Venezuela'],
    'Cuba': ['Cuba'],
    'Jamaica': ['Jamaica'],
    'New Zealand': ['New Zealand'],
    'Thailand': ['Thailand'],
    'Philippines': ['Philippines'],
    'Indonesia': ['Indonesia'],
    'Kazakhstan': ['Kazakhstan'],
    'Uzbekistan': ['Uzbekistan'],
    'Azerbaijan': ['Azerbaijan'],
    'Georgia': ['Georgia'],  # Country, not US state
    'Ukraine': ['Ukraine'],
    'Belarus': ['Belarus'],
    'Czech Republic': ['Czech Republic', 'Czechia'],
    'Croatia': ['Croatia'],
    'Serbia': ['Serbia'],
    'Nigeria': ['Nigeria'],
    'Cameroon': ['Cameroon'],
    'South Africa': ['South Africa'],
    'Egypt': ['Egypt'],
    'Morocco': ['Morocco'],
    'Afghanistan': ['Afghanistan'],
    'Iran': ['Iran'],
    'Iraq': ['Iraq'],
    'Kyrgyzstan': ['Kyrgyzstan'],
    'Tajikistan': ['Tajikistan'],
    'Armenia': ['Armenia'],
    'Moldova': ['Moldova'],
}

# US States (to distinguish from countries)
US_STATES = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
    'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia', 'D.C.'
]

# Regional groupings for analysis
REGIONS = {
    'North America': ['USA', 'Canada', 'Mexico'],
    'South America': ['Brazil', 'Argentina', 'Peru', 'Chile', 'Colombia', 'Ecuador', 'Venezuela'],
    'Western Europe': ['UK', 'France', 'Germany', 'Netherlands', 'Sweden', 'Italy', 'Spain', 'Poland'],
    'Eastern Europe': ['Russia', 'Ukraine', 'Belarus', 'Czech Republic', 'Croatia', 'Serbia', 'Moldova'],
    'Caucasus': ['Dagestan', 'Georgia', 'Azerbaijan', 'Armenia', 'Chechnya'],
    'Central Asia': ['Kazakhstan', 'Uzbekistan', 'Kyrgyzstan', 'Tajikistan'],
    'East Asia': ['Japan', 'South Korea', 'China'],
    'Southeast Asia': ['Thailand', 'Philippines', 'Indonesia'],
    'Oceania': ['Australia', 'New Zealand'],
    'Africa': ['Nigeria', 'Cameroon', 'South Africa', 'Egypt', 'Morocco'],
    'Middle East': ['Iran', 'Iraq', 'Afghanistan'],
    'Caribbean': ['Cuba', 'Jamaica'],
}


def extract_country(place_of_birth):
    """Extract country from place of birth string."""
    if not place_of_birth or pd.isna(place_of_birth):
        return 'Unknown'
    
    pob = str(place_of_birth).strip()
    
    # PRIORITY 1: Check city mappings FIRST (most specific)
    city_mappings = {
        'Makhachkala': 'Dagestan', 'Khasavyurt': 'Dagestan',
        'Grozny': 'Russia',  # Chechnya
        'Tbilisi': 'Georgia',
        'Baku': 'Azerbaijan',
        'Yerevan': 'Armenia',
        'Moscow': 'Russia', 'St. Petersburg': 'Russia', 'Sochi': 'Russia',
        'Sao Paulo': 'Brazil', 'Rio de Janeiro': 'Brazil',
        'London': 'UK', 'Manchester': 'UK', 'Liverpool': 'UK', 'Birmingham': 'UK',
        'Paris': 'France', 'Lyon': 'France', 'Marseille': 'France',
        'Berlin': 'Germany', 'Munich': 'Germany', 'Hamburg': 'Germany',
        'Tokyo': 'Japan', 'Osaka': 'Japan',
        'Seoul': 'South Korea', 'Busan': 'South Korea',
        'Beijing': 'China', 'Shanghai': 'China',
        'Sydney': 'Australia', 'Melbourne': 'Australia',
        'Toronto': 'Canada', 'Montreal': 'Canada', 'Vancouver': 'Canada',
        'Mexico City': 'Mexico', 'Guadalajara': 'Mexico', 'Tijuana': 'Mexico',
        'Warsaw': 'Poland', 'Krakow': 'Poland',
        'Amsterdam': 'Netherlands', 'Rotterdam': 'Netherlands',
        'Stockholm': 'Sweden', 'Gothenburg': 'Sweden',
        'Lagos': 'Nigeria', 'Abuja': 'Nigeria',
        'Almaty': 'Kazakhstan', 'Nur-Sultan': 'Kazakhstan',
        'Tashkent': 'Uzbekistan',
        'Kiev': 'Ukraine', 'Kyiv': 'Ukraine',
        'Minsk': 'Belarus',
        'Prague': 'Czech Republic',
        'Zagreb': 'Croatia',
        'Belgrade': 'Serbia',
        'Buenos Aires': 'Argentina',
        'Lima': 'Peru',
        'Santiago': 'Chile',
        'Bogota': 'Colombia',
        'Havana': 'Cuba',
        'Bangkok': 'Thailand',
        'Manila': 'Philippines',
        'Jakarta': 'Indonesia',
        'Tehran': 'Iran',
        'Kabul': 'Afghanistan',
    }
    
    for city, country in city_mappings.items():
        if city.lower() in pob.lower():
            return country
    
    # PRIORITY 2: Check for US states (to distinguish from country Georgia)
    # But SKIP if Tbilisi is present (already handled above)
    for state in US_STATES:
        if state.lower() in pob.lower():
            # Skip Georgia state check if it's clearly the country
            if state == 'Georgia' and 'tbilisi' in pob.lower():
                continue
            return 'USA'
    
    # PRIORITY 3: Check Dagestan specifically before Russia
    if 'dagestan' in pob.lower():
        return 'Dagestan'
    
    # PRIORITY 4: Check for country patterns
    for country, patterns in COUNTRY_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in pob.lower():
                # Special case: Georgia the country vs Georgia the state
                if country == 'Georgia' and any(s.lower() in pob.lower() for s in ['Atlanta', 'Savannah', 'Augusta']):
                    return 'USA'
                return country
    
    return 'Unknown'


def get_region(country):
    """Get region for a country."""
    for region, countries in REGIONS.items():
        if country in countries:
            return region
    return 'Other'


def load_fight_data_with_nationality():
    """Load fight data with fighter nationalities."""
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        f.FightID, f.Date as fight_date, f.EventName,
        f.FighterName as fighter1_name, f.OpponentName as fighter2_name,
        f.FighterID as fighter1_id, f.OpponentID as fighter2_id,
        f.WinnerID,
        
        -- ELO at fight time
        e1.EloBeforeFight as f1_elo,
        e2.EloBeforeFight as f2_elo,
        
        -- Place of birth
        fs1.PlaceOfBirth as f1_pob,
        fs2.PlaceOfBirth as f2_pob
        
    FROM Fights f
    LEFT JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    LEFT JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    LEFT JOIN FighterStats fs1 ON f.FighterID = fs1.FighterID
    LEFT JOIN FighterStats fs2 ON f.OpponentID = fs2.FighterID
    WHERE f.Date IS NOT NULL AND f.WinnerID IS NOT NULL
      AND f.Date >= '2018-01-01'
    ORDER BY f.Date
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Parse dates
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    
    # Fill defaults
    df['f1_elo'] = df['f1_elo'].fillna(1500)
    df['f2_elo'] = df['f2_elo'].fillna(1500)
    
    # Extract countries
    df['f1_country'] = df['f1_pob'].apply(extract_country)
    df['f2_country'] = df['f2_pob'].apply(extract_country)
    
    # Get regions
    df['f1_region'] = df['f1_country'].apply(get_region)
    df['f2_region'] = df['f2_country'].apply(get_region)
    
    # Calculate derived columns
    df['elo_diff'] = df['f1_elo'] - df['f2_elo']
    df['fighter1_won'] = (df['winnerid'] == df['fighter1_id']).astype(int)
    df['elo_prob_f1'] = df['elo_diff'].apply(lambda x: 1 / (1 + 10**(-x/400)))
    
    return df


def analyze_country_performance(df):
    """Analyze win rates by country."""
    print("\n" + "="*80)
    print("COUNTRY PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Combine F1 and F2 perspectives into one dataset
    # Create a unified view where each row is a fighter in a fight
    
    f1_data = df[['fightid', 'fight_date', 'f1_country', 'f1_elo', 'f2_elo', 'fighter1_won', 'elo_prob_f1']].copy()
    f1_data.columns = ['fightid', 'fight_date', 'country', 'fighter_elo', 'opponent_elo', 'won', 'elo_prob']
    
    f2_data = df[['fightid', 'fight_date', 'f2_country', 'f2_elo', 'f1_elo', 'fighter1_won', 'elo_prob_f1']].copy()
    f2_data.columns = ['fightid', 'fight_date', 'country', 'fighter_elo', 'opponent_elo', 'won', 'elo_prob']
    f2_data['won'] = 1 - f2_data['won']  # Flip for F2 perspective
    f2_data['elo_prob'] = 1 - f2_data['elo_prob']
    
    combined = pd.concat([f1_data, f2_data], ignore_index=True)
    
    # Analyze by country
    country_stats = combined.groupby('country').agg({
        'won': ['sum', 'count', 'mean'],
        'elo_prob': 'mean',
    }).round(3)
    
    country_stats.columns = ['wins', 'fights', 'win_rate', 'expected_win_rate']
    country_stats['edge'] = country_stats['win_rate'] - country_stats['expected_win_rate']
    country_stats = country_stats.sort_values('fights', ascending=False)
    
    print("\nWin Rate by Country (min 20 fights):")
    print("-"*80)
    print(f"{'Country':<20} {'Fights':>8} {'Wins':>8} {'Win%':>8} {'ELO Exp':>8} {'Edge':>8}")
    print("-"*80)
    
    significant_edges = []
    
    for country, row in country_stats.iterrows():
        if row['fights'] >= 20 and country != 'Unknown':
            edge_pct = row['edge'] * 100
            edge_str = f"{edge_pct:+.1f}%"
            
            print(f"{country:<20} {int(row['fights']):>8} {int(row['wins']):>8} "
                  f"{row['win_rate']:.1%}    {row['expected_win_rate']:.1%}    {edge_str:>8}")
            
            if abs(row['edge']) > 0.05:  # More than 5% edge
                significant_edges.append((country, row['fights'], row['win_rate'], row['edge']))
    
    # Statistical significance tests
    print("\n" + "-"*80)
    print("STATISTICALLY SIGNIFICANT EDGES (p < 0.10):")
    print("-"*80)
    
    for country, row in country_stats.iterrows():
        if row['fights'] >= 30 and country != 'Unknown':
            # Binomial test
            n = int(row['fights'])
            k = int(row['wins'])
            p = row['expected_win_rate']
            
            # Two-tailed test
            from scipy.stats import binomtest
            result = binomtest(k, n, p, alternative='two-sided')
            p_value = result.pvalue
            
            if p_value < 0.10:
                edge = row['win_rate'] - row['expected_win_rate']
                direction = "OVER" if edge > 0 else "UNDER"
                print(f"   {country}: {direction}performs by {abs(edge)*100:.1f}% (p={p_value:.3f}, n={n})")
    
    return country_stats


def analyze_matchup_edges(df):
    """Analyze country vs country matchups."""
    print("\n" + "="*80)
    print("COUNTRY MATCHUP ANALYSIS")
    print("="*80)
    
    # Create matchup column
    df['matchup'] = df.apply(lambda x: f"{x['f1_country']} vs {x['f2_country']}", axis=1)
    
    # Analyze matchups
    matchup_stats = df.groupby(['f1_country', 'f2_country']).agg({
        'fighter1_won': ['sum', 'count', 'mean'],
        'elo_prob_f1': 'mean',
    }).round(3)
    
    matchup_stats.columns = ['f1_wins', 'fights', 'f1_win_rate', 'elo_expected']
    matchup_stats['edge'] = matchup_stats['f1_win_rate'] - matchup_stats['elo_expected']
    matchup_stats = matchup_stats.sort_values('fights', ascending=False)
    
    print("\nTop Matchups by Sample Size (min 10 fights):")
    print("-"*80)
    
    interesting_matchups = []
    
    for (f1_country, f2_country), row in matchup_stats.iterrows():
        if row['fights'] >= 10 and f1_country != 'Unknown' and f2_country != 'Unknown':
            edge_pct = row['edge'] * 100
            
            if abs(edge_pct) > 8:  # More than 8% edge
                interesting_matchups.append({
                    'f1_country': f1_country,
                    'f2_country': f2_country,
                    'fights': row['fights'],
                    'f1_win_rate': row['f1_win_rate'],
                    'expected': row['elo_expected'],
                    'edge': row['edge'],
                })
    
    # Sort by edge magnitude
    interesting_matchups.sort(key=lambda x: abs(x['edge']), reverse=True)
    
    for m in interesting_matchups[:15]:
        print(f"   {m['f1_country']} vs {m['f2_country']}: "
              f"Win {m['f1_win_rate']:.1%} (exp {m['expected']:.1%}) | "
              f"Edge {m['edge']*100:+.1f}% | n={int(m['fights'])}")
    
    return matchup_stats


def analyze_region_performance(df):
    """Analyze performance by region."""
    print("\n" + "="*80)
    print("REGIONAL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Combine F1 and F2 perspectives
    f1_data = df[['f1_region', 'fighter1_won', 'elo_prob_f1']].copy()
    f1_data.columns = ['region', 'won', 'elo_prob']
    
    f2_data = df[['f2_region', 'fighter1_won', 'elo_prob_f1']].copy()
    f2_data.columns = ['region', 'won', 'elo_prob']
    f2_data['won'] = 1 - f2_data['won']
    f2_data['elo_prob'] = 1 - f2_data['elo_prob']
    
    combined = pd.concat([f1_data, f2_data], ignore_index=True)
    
    region_stats = combined.groupby('region').agg({
        'won': ['sum', 'count', 'mean'],
        'elo_prob': 'mean',
    }).round(3)
    
    region_stats.columns = ['wins', 'fights', 'win_rate', 'expected']
    region_stats['edge'] = region_stats['win_rate'] - region_stats['expected']
    region_stats = region_stats.sort_values('edge', ascending=False)
    
    print("\nWin Rate by Region:")
    print("-"*80)
    print(f"{'Region':<20} {'Fights':>8} {'Win%':>10} {'Expected':>10} {'Edge':>10}")
    print("-"*80)
    
    for region, row in region_stats.iterrows():
        if row['fights'] >= 20:
            print(f"{region:<20} {int(row['fights']):>8} {row['win_rate']:>10.1%} "
                  f"{row['expected']:>10.1%} {row['edge']*100:>+10.1f}%")
    
    return region_stats


def analyze_caucasus_edge(df):
    """Special analysis for Dagestan/Caucasus fighters."""
    print("\n" + "="*80)
    print("CAUCASUS REGION DEEP DIVE (Dagestan, Chechnya, etc.)")
    print("="*80)
    
    # Define Caucasus countries
    caucasus = ['Dagestan', 'Russia', 'Georgia', 'Azerbaijan', 'Armenia']
    
    # Fighters from Caucasus region
    df['f1_caucasus'] = df['f1_country'].isin(caucasus) | df['f1_pob'].str.contains('Dagestan|Chechnya|Grozny|Makhachkala', case=False, na=False)
    df['f2_caucasus'] = df['f2_country'].isin(caucasus) | df['f2_pob'].str.contains('Dagestan|Chechnya|Grozny|Makhachkala', case=False, na=False)
    
    # Caucasus vs Non-Caucasus
    caucasus_vs_other = df[df['f1_caucasus'] & ~df['f2_caucasus']]
    other_vs_caucasus = df[~df['f1_caucasus'] & df['f2_caucasus']]
    
    print(f"\nCaucasus Fighter as F1 vs Non-Caucasus:")
    if len(caucasus_vs_other) > 0:
        win_rate = caucasus_vs_other['fighter1_won'].mean()
        expected = caucasus_vs_other['elo_prob_f1'].mean()
        print(f"   Fights: {len(caucasus_vs_other)}")
        print(f"   Win rate: {win_rate:.1%}")
        print(f"   ELO expected: {expected:.1%}")
        print(f"   Edge: {(win_rate - expected)*100:+.1f}%")
    
    print(f"\nNon-Caucasus as F1 vs Caucasus Fighter:")
    if len(other_vs_caucasus) > 0:
        win_rate = other_vs_caucasus['fighter1_won'].mean()
        expected = other_vs_caucasus['elo_prob_f1'].mean()
        caucasus_win = 1 - win_rate
        caucasus_exp = 1 - expected
        print(f"   Fights: {len(other_vs_caucasus)}")
        print(f"   Caucasus win rate: {caucasus_win:.1%}")
        print(f"   ELO expected: {caucasus_exp:.1%}")
        print(f"   Edge: {(caucasus_win - caucasus_exp)*100:+.1f}%")
    
    # Combined Caucasus performance
    total_fights = len(caucasus_vs_other) + len(other_vs_caucasus)
    if total_fights > 0:
        # Caucasus wins
        caucasus_wins = caucasus_vs_other['fighter1_won'].sum() + (len(other_vs_caucasus) - other_vs_caucasus['fighter1_won'].sum())
        overall_win_rate = caucasus_wins / total_fights
        
        # Expected based on ELO
        expected_wins = caucasus_vs_other['elo_prob_f1'].sum() + (1 - other_vs_caucasus['elo_prob_f1']).sum()
        expected_rate = expected_wins / total_fights
        
        print(f"\n   OVERALL CAUCASUS PERFORMANCE:")
        print(f"   Total matchups vs non-Caucasus: {total_fights}")
        print(f"   Win rate: {overall_win_rate:.1%}")
        print(f"   ELO expected: {expected_rate:.1%}")
        print(f"   Edge: {(overall_win_rate - expected_rate)*100:+.1f}%")
        
        # Statistical test
        from scipy.stats import binomtest
        result = binomtest(int(caucasus_wins), total_fights, expected_rate, alternative='greater')
        print(f"   P-value (one-tailed): {result.pvalue:.3f}")
        
        if result.pvalue < 0.05:
            print("   >>> STATISTICALLY SIGNIFICANT at p<0.05")


def analyze_brazil_wrestling(df):
    """Analyze Brazilian fighters vs wrestling-heavy countries."""
    print("\n" + "="*80)
    print("BRAZIL VS WRESTLING-HEAVY COUNTRIES")
    print("="*80)
    
    wrestling_countries = ['Russia', 'Dagestan', 'USA', 'Iran', 'Kazakhstan', 'Uzbekistan']
    
    # Brazil vs Wrestling countries
    brazil_vs_wrestling = df[
        (df['f1_country'] == 'Brazil') & 
        (df['f2_country'].isin(wrestling_countries))
    ]
    
    wrestling_vs_brazil = df[
        (df['f2_country'] == 'Brazil') & 
        (df['f1_country'].isin(wrestling_countries))
    ]
    
    print(f"\nBrazil vs Wrestling-Heavy Countries:")
    
    total = len(brazil_vs_wrestling) + len(wrestling_vs_brazil)
    if total > 0:
        brazil_wins = brazil_vs_wrestling['fighter1_won'].sum() + (len(wrestling_vs_brazil) - wrestling_vs_brazil['fighter1_won'].sum())
        brazil_rate = brazil_wins / total
        
        expected = brazil_vs_wrestling['elo_prob_f1'].sum() + (1 - wrestling_vs_brazil['elo_prob_f1']).sum()
        expected_rate = expected / total
        
        print(f"   Total fights: {total}")
        print(f"   Brazil win rate: {brazil_rate:.1%}")
        print(f"   ELO expected: {expected_rate:.1%}")
        print(f"   Edge: {(brazil_rate - expected_rate)*100:+.1f}%")


def identify_actionable_edges(df):
    """Identify actionable betting edges based on nationality."""
    print("\n" + "="*80)
    print("ACTIONABLE NATIONALITY-BASED EDGES")
    print("="*80)
    
    # Import safeguards
    try:
        from ml.safeguards import (
            validate_nationality_edge, 
            validate_sample_size,
            warn_multiple_testing,
            MIN_SAMPLES
        )
        safeguards_available = True
    except ImportError:
        safeguards_available = False
        print("WARNING: Safeguards module not available")
    
    edges = []
    n_tests = 0  # Track for multiple testing correction
    
    # Test 1: Caucasus fighters overperform
    df['f1_caucasus'] = df['f1_pob'].str.contains('Dagestan|Chechnya|Grozny|Makhachkala', case=False, na=False)
    df['f2_caucasus'] = df['f2_pob'].str.contains('Dagestan|Chechnya|Grozny|Makhachkala', case=False, na=False)
    
    subset = df[df['f1_caucasus'] & ~df['f2_caucasus']]
    n_tests += 1
    if len(subset) >= 20:
        win_rate = subset['fighter1_won'].mean()
        expected = subset['elo_prob_f1'].mean()
        edge = win_rate - expected
        
        # Validate with safeguards
        if safeguards_available and len(subset) < MIN_SAMPLES['country_edge']:
            print(f"\n⚠️  Caucasus edge: INSUFFICIENT SAMPLE ({len(subset)} < {MIN_SAMPLES['country_edge']})")
        elif edge > 0.05:
            edges.append(('Caucasus fighter (F1)', len(subset), win_rate, edge))
    
    # Test 2: Brazil vs USA
    subset = df[(df['f1_country'] == 'Brazil') & (df['f2_country'] == 'USA')]
    n_tests += 1
    if len(subset) >= 15:
        win_rate = subset['fighter1_won'].mean()
        expected = subset['elo_prob_f1'].mean()
        edge = win_rate - expected
        if abs(edge) > 0.05:
            edges.append(('Brazil vs USA', len(subset), win_rate, edge))
    
    # Test 3: UK fighters
    subset = df[df['f1_country'] == 'UK']
    n_tests += 1
    if len(subset) >= 30:
        win_rate = subset['fighter1_won'].mean()
        expected = subset['elo_prob_f1'].mean()
        edge = win_rate - expected
        if abs(edge) > 0.05:
            edges.append(('UK fighters', len(subset), win_rate, edge))
    
    # Warn about multiple testing
    if safeguards_available:
        warn_multiple_testing(n_tests)
    
    # Print results with validation
    if edges:
        print("\nPotential Edges Found:")
        print("-"*60)
        for name, n, win_rate, edge in edges:
            direction = "OVER" if edge > 0 else "UNDER"
            print(f"   {name}: {direction}performs ELO by {abs(edge)*100:.1f}% (n={n}, win={win_rate:.1%})")
            
            # Validate each edge
            if safeguards_available:
                from scipy.stats import binomtest
                result = binomtest(int(win_rate * n), n, 0.5)
                validation = validate_nationality_edge(
                    country=name,
                    n_fights=n,
                    win_rate=win_rate,
                    expected_rate=0.5,  # Baseline
                    p_value=result.pvalue
                )
                if validation['warnings']:
                    for w in validation['warnings']:
                        print(f"       ⚠️  {w}")
                print(f"       Recommendation: {validation['recommendation']}")
    else:
        print("\nNo significant nationality-based edges found with current criteria.")
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    ⚠️  CRITICAL WARNING ⚠️                               ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  NATIONALITY EDGES ARE UNRELIABLE FOR BETTING:                           ║
    ║                                                                          ║
    ║  1. CONFOUNDING VARIABLES                                                ║
    ║     - Dagestan edge is likely WRESTLING SKILL, not nationality           ║
    ║     - Control for: fighting style, weight class, era, training camp      ║
    ║                                                                          ║
    ║  2. SURVIVORSHIP BIAS                                                    ║
    ║     - Only successful fighters from each country stay in UFC             ║
    ║     - You're seeing a biased sample of winners                           ║
    ║                                                                          ║
    ║  3. SMALL SAMPLE SIZES                                                   ║
    ║     - Most countries have <50 fighters total                             ║
    ║     - Require 100+ fights per country for reliable inference             ║
    ║                                                                          ║
    ║  4. MULTIPLE TESTING PROBLEM                                             ║
    ║     - Testing 20+ countries = ~1 false positive expected                 ║
    ║     - Apply Bonferroni correction before claiming significance           ║
    ║                                                                          ║
    ║  RECOMMENDATION:                                                         ║
    ║  - Use nationality as ONE feature with LOW weight (5-10%)                ║
    ║  - DON'T bet solely on nationality                                       ║
    ║  - Combine with ELO, style, recent form, and other factors               ║
    ║  - Validate on out-of-sample data before using                           ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    print("="*80)
    print("NATIONALITY & COUNTRY EDGE ANALYSIS")
    print("="*80)
    
    print("\nLoading fight data with nationality...")
    df = load_fight_data_with_nationality()
    print(f"Loaded {len(df)} fights")
    
    # Show country distribution
    print("\nCountry Distribution (Top 15):")
    all_countries = pd.concat([df['f1_country'], df['f2_country']])
    country_counts = all_countries.value_counts()
    for country, count in country_counts.head(15).items():
        print(f"   {country}: {count}")
    
    # Run analyses
    analyze_country_performance(df)
    analyze_matchup_edges(df)
    analyze_region_performance(df)
    analyze_caucasus_edge(df)
    analyze_brazil_wrestling(df)
    identify_actionable_edges(df)


if __name__ == '__main__':
    main()

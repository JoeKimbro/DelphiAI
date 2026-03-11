"""
Validate Edge - Deep analysis of potential edges.

1. Check if the 6.3% ROI on favorites is real or artifact
2. Validate recency bias edge with statistical significance
3. Identify actionable edges
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
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


def load_data():
    """Load fight data."""
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        f.FightID, f.Date as fight_date,
        f.FighterName as fighter1_name, f.OpponentName as fighter2_name,
        f.FighterID as fighter1_id, f.OpponentID as fighter2_id,
        f.WinnerID,
        e1.EloBeforeFight as f1_elo,
        e2.EloBeforeFight as f2_elo,
        pit1.FightsBefore as f1_fights,
        pit1.RecentWinRate as f1_recent_form,
        pit2.FightsBefore as f2_fights,
        pit2.RecentWinRate as f2_recent_form
    FROM Fights f
    LEFT JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    LEFT JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    LEFT JOIN PointInTimeStats pit1 ON f.FighterURL = pit1.FighterURL AND f.Date = pit1.FightDate
    LEFT JOIN PointInTimeStats pit2 ON f.OpponentURL = pit2.FighterURL AND f.Date = pit2.FightDate
    WHERE f.Date IS NOT NULL AND f.WinnerID IS NOT NULL
      AND f.Date >= '2022-01-01'
    ORDER BY f.Date
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    df['f1_elo'] = df['f1_elo'].fillna(1500)
    df['f2_elo'] = df['f2_elo'].fillna(1500)
    df['f1_recent_form'] = df['f1_recent_form'].fillna(0.5)
    df['f2_recent_form'] = df['f2_recent_form'].fillna(0.5)
    df['elo_diff'] = df['f1_elo'] - df['f2_elo']
    df['fighter1_won'] = (df['winnerid'] == df['fighter1_id']).astype(int)
    
    return df


def analyze_positional_bias(df):
    """Check for positional bias (Fighter1 vs Fighter2)."""
    print("="*70)
    print("POSITIONAL BIAS ANALYSIS")
    print("="*70)
    
    # Overall F1 win rate
    f1_wins = df['fighter1_won'].sum()
    total = len(df)
    f1_rate = f1_wins / total
    
    print(f"\nFighter1 wins: {f1_wins}/{total} = {f1_rate:.1%}")
    
    # Expected by ELO
    df['elo_prob_f1'] = df['elo_diff'].apply(lambda x: 1 / (1 + 10**(-x/400)))
    expected_f1_rate = df['elo_prob_f1'].mean()
    
    print(f"Expected by ELO: {expected_f1_rate:.1%}")
    print(f"Difference: {(f1_rate - expected_f1_rate)*100:.1f}%")
    
    # Chi-square test for independence
    observed = f1_wins
    expected = expected_f1_rate * total
    chi2 = (observed - expected)**2 / expected + (total - observed - (total - expected))**2 / (total - expected)
    
    print(f"Chi-square stat: {chi2:.2f}")
    
    if f1_rate > 0.55:
        print("\n>>> WARNING: Significant positional bias detected!")
        print(">>> Fighter1 wins more than expected")
        print(">>> This inflates ROI calculations")


def analyze_elo_accuracy(df):
    """How accurate is ELO at predicting?"""
    print("\n" + "="*70)
    print("ELO ACCURACY ANALYSIS")
    print("="*70)
    
    # Filter to non-tied fights
    df_notied = df[abs(df['elo_diff']) > 25].copy()
    
    # Higher ELO wins?
    df_notied['higher_elo_won'] = (
        ((df_notied['elo_diff'] > 0) & (df_notied['fighter1_won'] == 1)) |
        ((df_notied['elo_diff'] < 0) & (df_notied['fighter1_won'] == 0))
    )
    
    higher_elo_wins = df_notied['higher_elo_won'].sum()
    total = len(df_notied)
    win_rate = higher_elo_wins / total
    
    print(f"\nHigher ELO fighter wins: {higher_elo_wins}/{total} = {win_rate:.1%}")
    print(f"This is the TRUE predictive power of ELO")
    
    # Breakdown by ELO difference magnitude
    print("\nBy ELO difference magnitude:")
    bins = [(25, 75), (75, 150), (150, 250), (250, 500)]
    
    for low, high in bins:
        subset = df_notied[(abs(df_notied['elo_diff']) >= low) & (abs(df_notied['elo_diff']) < high)]
        if len(subset) > 0:
            rate = subset['higher_elo_won'].mean()
            expected = subset['elo_prob_f1'].apply(lambda x: max(x, 1-x)).mean()
            print(f"   ELO diff {low:3d}-{high:3d}: Win rate {rate:.1%} (expected {expected:.1%}) n={len(subset)}")


def analyze_recency_bias_statistically(df):
    """Statistical validation of recency bias edge."""
    print("\n" + "="*70)
    print("RECENCY BIAS EDGE - STATISTICAL VALIDATION")
    print("="*70)
    
    # Fighters with bad recent form but positive ELO
    # (Market might undervalue them)
    
    bad_form_good_elo = df[
        (df['f1_recent_form'] < 0.35) & 
        (df['elo_diff'] > 50)
    ].copy()
    
    print(f"\nFighter1: Bad form (<35%) but ELO favored (+50)")
    print(f"   Fights: {len(bad_form_good_elo)}")
    
    if len(bad_form_good_elo) > 0:
        wins = bad_form_good_elo['fighter1_won'].sum()
        win_rate = wins / len(bad_form_good_elo)
        expected = bad_form_good_elo['elo_prob_f1'].mean()
        
        print(f"   Actual win rate: {win_rate:.1%}")
        print(f"   ELO expected: {expected:.1%}")
        print(f"   Outperformance: {(win_rate - expected)*100:.1f}%")
        
        # Statistical significance (binomial test)
        from scipy.stats import binom
        
        n = len(bad_form_good_elo)
        k = wins
        p = expected
        
        # Probability of getting k or more wins by chance
        p_value = 1 - binom.cdf(k - 1, n, p)
        print(f"   P-value (one-tailed): {p_value:.3f}")
        
        if p_value < 0.05:
            print("   >>> STATISTICALLY SIGNIFICANT at p<0.05")
        else:
            print("   >>> NOT statistically significant (could be luck)")
    
    # Also check the opposite - good form but bad ELO
    good_form_bad_elo = df[
        (df['f1_recent_form'] > 0.70) & 
        (df['elo_diff'] < -50)
    ].copy()
    
    print(f"\nFighter1: Good form (>70%) but ELO underdog (-50)")
    print(f"   Fights: {len(good_form_bad_elo)}")
    
    if len(good_form_bad_elo) > 0:
        wins = good_form_bad_elo['fighter1_won'].sum()
        win_rate = wins / len(good_form_bad_elo)
        expected = good_form_bad_elo['elo_prob_f1'].mean()
        
        print(f"   Actual win rate: {win_rate:.1%}")
        print(f"   ELO expected: {expected:.1%}")
        print(f"   Underperformance: {(expected - win_rate)*100:.1f}%")


def simulate_realistic_betting(df):
    """
    Simulate betting with PROPERLY calculated market odds.
    
    Key: Market odds already account for ELO.
    We need to find where ACTUAL outcomes differ from MARKET.
    """
    print("\n" + "="*70)
    print("REALISTIC BETTING SIMULATION")
    print("="*70)
    
    # Market efficiency assumption:
    # Implied probability = ELO probability + vig
    # If we bet on favorite, we need to win MORE than implied prob to profit
    
    VIG = 0.045  # 4.5% overround
    BREAK_EVEN = 0.524  # Need 52.4% to break even on -110 odds
    
    results = []
    
    for idx, fight in df.iterrows():
        elo_diff = fight['elo_diff']
        f1_won = fight['fighter1_won']
        
        # Skip pick'ems
        if abs(elo_diff) < 25:
            continue
        
        # Market implied probability (ELO + vig)
        true_prob = 1 / (1 + 10**(-abs(elo_diff)/400))  # Favorite's true prob
        implied_prob = min(0.95, true_prob * (1 + VIG/2))
        
        # Convert to decimal odds
        decimal_odds = 1 / implied_prob
        
        # Did favorite win?
        fav_is_f1 = elo_diff > 0
        fav_won = (fav_is_f1 and f1_won) or (not fav_is_f1 and not f1_won)
        
        results.append({
            'elo_diff': abs(elo_diff),
            'true_prob': true_prob,
            'implied_prob': implied_prob,
            'decimal_odds': decimal_odds,
            'fav_won': fav_won,
            'profit': (decimal_odds - 1) if fav_won else -1,
        })
    
    results_df = pd.DataFrame(results)
    
    # Overall stats
    total_bets = len(results_df)
    fav_wins = results_df['fav_won'].sum()
    win_rate = fav_wins / total_bets
    total_profit = results_df['profit'].sum()
    roi = total_profit / total_bets
    
    print(f"\nAlways bet on FAVORITE (ELO-based):")
    print(f"   Bets: {total_bets}")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Break-even needed: {BREAK_EVEN:.1%}")
    print(f"   ROI: {roi*100:.1f}%")
    
    # Is the edge real?
    # Calculate expected ROI if win rate = true probability
    expected_roi = ((win_rate * results_df['decimal_odds'].mean()) - 1) / 1
    print(f"\n   Implied ROI (if markets efficient): ~0%")
    print(f"   Actual ROI: {roi*100:.1f}%")
    
    if roi > 0.03:
        print("\n   >>> Positive ROI detected!")
        print("   >>> Possible explanations:")
        print("       1. Positional bias inflating results")
        print("       2. ELO system genuinely undervalued by market")
        print("       3. Sample size variance")
    
    # Confidence interval for ROI
    profits = results_df['profit'].values
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    n = len(profits)
    se = std_profit / np.sqrt(n)
    ci_low = mean_profit - 1.96 * se
    ci_high = mean_profit + 1.96 * se
    
    print(f"\n   95% CI for ROI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")


def identify_actionable_edges(df):
    """Find specific, actionable betting edges."""
    print("\n" + "="*70)
    print("ACTIONABLE EDGE IDENTIFICATION")
    print("="*70)
    
    edges = []
    
    # === Edge 1: Bad recent form but good ELO ===
    print("\n1. RECENCY BIAS (Bet on bad-form ELO favorites)")
    
    subset = df[(df['f1_recent_form'] < 0.35) & (df['elo_diff'] > 50)]
    if len(subset) >= 20:
        win_rate = subset['fighter1_won'].mean()
        expected = subset['elo_prob_f1'].mean()
        edge = win_rate - expected
        
        if edge > 0.05:  # More than 5% edge
            print(f"   Bets: {len(subset)} | Win: {win_rate:.1%} | Edge: {edge*100:.1f}%")
            edges.append(('Recency bias - F1', len(subset), win_rate, edge))
        else:
            print(f"   No significant edge found (edge={edge*100:.1f}%)")
    else:
        print(f"   Insufficient data ({len(subset)} fights)")
    
    # Also check F2 side
    subset = df[(df['f2_recent_form'] < 0.35) & (df['elo_diff'] < -50)]
    if len(subset) >= 20:
        win_rate = 1 - subset['fighter1_won'].mean()  # F2 wins
        expected = 1 - subset['elo_prob_f1'].mean()
        edge = win_rate - expected
        
        if edge > 0.05:
            print(f"   (F2 side): Bets: {len(subset)} | Win: {win_rate:.1%} | Edge: {edge*100:.1f}%")
            edges.append(('Recency bias - F2', len(subset), win_rate, edge))
    
    # === Edge 2: Experience vs Debut ===
    print("\n2. EXPERIENCE EDGE (Bet on veterans vs debutants)")
    
    df['f1_fights'] = df['f1_fights'].fillna(0)
    df['f2_fights'] = df['f2_fights'].fillna(0)
    
    # Veteran F1 vs Debut F2
    subset = df[(df['f1_fights'] > 10) & (df['f2_fights'] <= 2) & (df['elo_diff'] > 0)]
    if len(subset) >= 20:
        win_rate = subset['fighter1_won'].mean()
        expected = subset['elo_prob_f1'].mean()
        edge = win_rate - expected
        
        print(f"   Veteran vs Debut: Bets: {len(subset)} | Win: {win_rate:.1%} | Edge: {edge*100:.1f}%")
        if edge > 0.03:
            edges.append(('Experience edge', len(subset), win_rate, edge))
    else:
        print(f"   Insufficient data ({len(subset)} fights)")
    
    # === Edge 3: Big ELO favorites ===
    print("\n3. BIG FAVORITES (Bet on ELO > +150)")
    
    subset = df[df['elo_diff'] > 150]
    if len(subset) >= 20:
        win_rate = subset['fighter1_won'].mean()
        expected = subset['elo_prob_f1'].mean()
        edge = win_rate - expected
        
        print(f"   Bets: {len(subset)} | Win: {win_rate:.1%} vs ELO {expected:.1%} | Edge: {edge*100:.1f}%")
    else:
        print(f"   Insufficient data ({len(subset)} fights)")
    
    # === Summary ===
    print("\n" + "-"*70)
    print("SUMMARY OF FOUND EDGES")
    print("-"*70)
    
    if edges:
        for name, n, win_rate, edge in edges:
            print(f"   {name}: {n} bets, {win_rate:.1%} win rate, {edge*100:.1f}% edge")
        
        print("\n   >>> To exploit these edges:")
        print("       1. Get REAL odds data for upcoming fights")
        print("       2. Compare model prob to implied prob")
        print("       3. Bet when edge > 5% over market")
    else:
        print("   No statistically significant edges found")
        print("   Market may be efficient for these patterns")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} fights (2022+)\n")
    
    analyze_positional_bias(df)
    analyze_elo_accuracy(df)
    analyze_recency_bias_statistically(df)
    simulate_realistic_betting(df)
    identify_actionable_edges(df)


if __name__ == '__main__':
    main()

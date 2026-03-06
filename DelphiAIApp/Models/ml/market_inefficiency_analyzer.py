"""
Market Inefficiency Analyzer

Finds where the model might have edge over the betting market.

KEY INSIGHT: 
Our model is NOT going to beat the market on "fair fights" 
(where public info is priced in). We need to find SPECIFIC 
situations where the market is WRONG.

Market Inefficiencies to Test:
1. PUBLIC BIAS - Popular fighters get overbet
2. RECENCY BIAS - Overreaction to last fight result
3. STYLE MATCHUPS - Market misses striker vs wrestler dynamics
4. WEIGHT CLASS DEBUT - Uncertainty on first fight at new weight
5. LAYOFF RETURNS - Long layoffs create pricing uncertainty
6. DEBUT FIGHTERS - New fighters are hard to price
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
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


def load_fights_with_context():
    """Load fights with full context for inefficiency analysis."""
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        f.FightID, f.Date as fight_date, f.EventName,
        f.FighterName as fighter1_name, f.OpponentName as fighter2_name,
        f.FighterID as fighter1_id, f.OpponentID as fighter2_id,
        f.WinnerID, f.Method,
        fs1.WeightClass as weight_class,
        
        -- ELO at fight time
        e1.EloBeforeFight as f1_elo,
        e2.EloBeforeFight as f2_elo,
        
        -- Fighter stats
        fs1.Height as f1_height, fs1.Reach as f1_reach, fs1.Stance as f1_stance,
        fs1.Wins as f1_wins, fs1.Losses as f1_losses,
        fs2.Height as f2_height, fs2.Reach as f2_reach, fs2.Stance as f2_stance,
        fs2.Wins as f2_wins, fs2.Losses as f2_losses,
        
        -- Career stats for style classification
        cs1.SLpM as f1_slpm, cs1.TDAvg as f1_tdavg, cs1.SubAvg as f1_subavg,
        cs2.SLpM as f2_slpm, cs2.TDAvg as f2_tdavg, cs2.SubAvg as f2_subavg,
        
        -- Point in time stats
        pit1.FightsBefore as f1_fights_before,
        pit1.WinsBefore as f1_wins_before,
        pit1.RecentWinRate as f1_recent_form,
        pit2.FightsBefore as f2_fights_before,
        pit2.WinsBefore as f2_wins_before,
        pit2.RecentWinRate as f2_recent_form
        
    FROM Fights f
    LEFT JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    LEFT JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    LEFT JOIN FighterStats fs1 ON f.FighterID = fs1.FighterID
    LEFT JOIN FighterStats fs2 ON f.OpponentID = fs2.FighterID
    LEFT JOIN CareerStats cs1 ON f.FighterID = cs1.FighterID
    LEFT JOIN CareerStats cs2 ON f.OpponentID = cs2.FighterID
    LEFT JOIN PointInTimeStats pit1 ON f.FighterURL = pit1.FighterURL AND f.Date = pit1.FightDate
    LEFT JOIN PointInTimeStats pit2 ON f.OpponentURL = pit2.FighterURL AND f.Date = pit2.FightDate
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
    
    # Calculate derived features
    df['elo_diff'] = df['f1_elo'] - df['f2_elo']
    df['fighter1_won'] = (df['winnerid'] == df['fighter1_id']).astype(int)
    
    # ELO-based probability
    df['elo_prob_f1'] = df['elo_diff'].apply(lambda x: 1 / (1 + 10**(-x/400)))
    
    # Experience
    df['f1_experience'] = df['f1_fights_before'].fillna(0)
    df['f2_experience'] = df['f2_fights_before'].fillna(0)
    df['experience_diff'] = df['f1_experience'] - df['f2_experience']
    
    return df


def simulate_market_odds(elo_diff, popularity_bias=0, recency_bias=0, vig=0.048):
    """
    Simulate market odds based on ELO + market biases.
    
    Args:
        elo_diff: True ELO difference
        popularity_bias: Extra edge for popular fighter (0.02 = 2% overbet)
        recency_bias: Extra edge based on recent result
        vig: Sportsbook margin (4.8% is standard)
    
    Returns:
        (decimal_odds_f1, decimal_odds_f2, implied_prob_f1, implied_prob_f2)
    """
    # True probability
    true_prob = 1 / (1 + 10**(-elo_diff/400))
    
    # Market moves line based on biases
    market_prob = true_prob + popularity_bias + recency_bias
    market_prob = max(0.05, min(0.95, market_prob))
    
    # Add vig
    implied_f1 = market_prob * (1 + vig/2)
    implied_f2 = (1 - market_prob) * (1 + vig/2)
    
    # Convert to decimal odds
    odds_f1 = 1 / implied_f1
    odds_f2 = 1 / implied_f2
    
    return odds_f1, odds_f2, implied_f1, implied_f2


def calculate_edge(model_prob, implied_prob):
    """Calculate edge over market."""
    return model_prob - implied_prob


def simulate_bet_result(bet_on_f1, fighter1_won, decimal_odds):
    """Calculate profit/loss for a bet."""
    if bet_on_f1:
        if fighter1_won:
            return decimal_odds - 1  # Win: profit
        else:
            return -1  # Loss
    else:
        if not fighter1_won:
            return decimal_odds - 1  # Win: profit
        else:
            return -1  # Loss


def classify_fighting_style(slpm, tdavg, subavg):
    """Classify fighter style - delegates to canonical classifier."""
    try:
        from ml.style_classifier import classify_style
    except ModuleNotFoundError:
        from style_classifier import classify_style
    return classify_style(slpm=slpm, td_avg=tdavg, sub_avg=subavg)


def analyze_public_bias(df):
    """
    Test: Do popular fighters get overbet?
    
    Hypothesis: Fighters with more wins, higher profile, get overbet.
    This creates value on lesser-known fighters.
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: PUBLIC BIAS (Popular Fighter Overbetting)")
    print("="*70)
    
    # Proxy for popularity: more wins, more UFC experience
    df = df.copy()
    df['f1_popular'] = (df['f1_experience'] > 10) | (df['f1_wins_before'].fillna(0) > 7)
    df['f2_popular'] = (df['f2_experience'] > 10) | (df['f2_wins_before'].fillna(0) > 7)
    
    # Cases where popular vs unknown
    popular_vs_unknown = df[df['f1_popular'] & ~df['f2_popular']]
    unknown_vs_popular = df[~df['f1_popular'] & df['f2_popular']]
    
    print(f"\nPopular F1 vs Unknown F2: {len(popular_vs_unknown)} fights")
    if len(popular_vs_unknown) > 0:
        pop_win_rate = popular_vs_unknown['fighter1_won'].mean()
        avg_elo_diff = popular_vs_unknown['elo_diff'].mean()
        expected_win_rate = 1 / (1 + 10**(-avg_elo_diff/400))
        
        print(f"   Popular fighter win rate: {pop_win_rate:.1%}")
        print(f"   ELO-implied win rate: {expected_win_rate:.1%}")
        print(f"   Difference: {(pop_win_rate - expected_win_rate)*100:.1f}%")
        
        if pop_win_rate < expected_win_rate:
            print("   >>> FINDING: Popular fighters MAY be overbet (underperform ELO)")
            print("   >>> Potential edge: Bet AGAINST popular fighters when edge > 3%")
    
    print(f"\nUnknown F1 vs Popular F2: {len(unknown_vs_popular)} fights")
    if len(unknown_vs_popular) > 0:
        unk_win_rate = unknown_vs_popular['fighter1_won'].mean()
        avg_elo_diff = unknown_vs_popular['elo_diff'].mean()
        expected_win_rate = 1 / (1 + 10**(-avg_elo_diff/400))
        
        print(f"   Unknown fighter win rate: {unk_win_rate:.1%}")
        print(f"   ELO-implied win rate: {expected_win_rate:.1%}")
        print(f"   Difference: {(unk_win_rate - expected_win_rate)*100:.1f}%")
    
    # Simulate ROI if we bet on unknowns when model shows edge
    print("\n   SIMULATED ROI (Betting on Unknowns with 5% Model Edge):")
    
    total_bets = 0
    total_profit = 0
    
    for idx, fight in popular_vs_unknown.iterrows():
        # Simulate market overpricing popular fighter by 3%
        _, odds_f2, _, implied_f2 = simulate_market_odds(
            fight['elo_diff'], 
            popularity_bias=0.03  # Market favors F1 extra
        )
        
        model_prob_f2 = 1 - fight['elo_prob_f1']
        edge = model_prob_f2 - implied_f2
        
        if edge > 0.05:  # Bet on unknown if 5% edge
            total_bets += 1
            profit = simulate_bet_result(False, fight['fighter1_won'], odds_f2)
            total_profit += profit
    
    if total_bets > 0:
        roi = total_profit / total_bets
        print(f"   Bets placed: {total_bets}")
        print(f"   Total ROI: {roi*100:.1f}%")


def analyze_recency_bias(df):
    """
    Test: Does market overreact to recent fight results?
    
    Hypothesis: Fighter coming off KO loss gets undervalued.
    Fighter on winning streak gets overvalued.
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: RECENCY BIAS (Overreaction to Last Fight)")
    print("="*70)
    
    df = df.copy()
    
    # Approximate recent form
    df['f1_recent_form'] = df['f1_recent_form'].fillna(0.5)
    df['f2_recent_form'] = df['f2_recent_form'].fillna(0.5)
    
    # Fighter coming off bad streak (low recent form)
    bad_form_f1 = df[df['f1_recent_form'] < 0.3]
    good_form_f1 = df[df['f1_recent_form'] > 0.7]
    
    print(f"\nF1 with BAD recent form (<30%): {len(bad_form_f1)} fights")
    if len(bad_form_f1) > 0:
        win_rate = bad_form_f1['fighter1_won'].mean()
        avg_elo_diff = bad_form_f1['elo_diff'].mean()
        expected = 1 / (1 + 10**(-avg_elo_diff/400))
        
        print(f"   Actual win rate: {win_rate:.1%}")
        print(f"   ELO-implied: {expected:.1%}")
        print(f"   Outperformance: {(win_rate - expected)*100:.1f}%")
        
        if win_rate > expected:
            print("   >>> FINDING: Fighters on bad streak MAY be UNDERVALUED")
            print("   >>> Potential edge: Bet on fighters coming off losses when ELO supports")
    
    print(f"\nF1 with GOOD recent form (>70%): {len(good_form_f1)} fights")
    if len(good_form_f1) > 0:
        win_rate = good_form_f1['fighter1_won'].mean()
        avg_elo_diff = good_form_f1['elo_diff'].mean()
        expected = 1 / (1 + 10**(-avg_elo_diff/400))
        
        print(f"   Actual win rate: {win_rate:.1%}")
        print(f"   ELO-implied: {expected:.1%}")
        print(f"   Underperformance: {(expected - win_rate)*100:.1f}%")
        
        if win_rate < expected:
            print("   >>> FINDING: Fighters on win streaks MAY be OVERVALUED")


def analyze_style_matchups(df):
    """
    Test: Does market properly price style matchups?
    
    Hypothesis: Striker vs Wrestler dynamics are mispriced.
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: STYLE MATCHUP INEFFICIENCY")
    print("="*70)
    
    df = df.copy()
    
    # Classify styles
    df['f1_style'] = df.apply(
        lambda x: classify_fighting_style(x['f1_slpm'], x['f1_tdavg'], x['f1_subavg']),
        axis=1
    )
    df['f2_style'] = df.apply(
        lambda x: classify_fighting_style(x['f2_slpm'], x['f2_tdavg'], x['f2_subavg']),
        axis=1
    )
    
    # Analyze each matchup type
    matchups = df.groupby(['f1_style', 'f2_style']).agg({
        'fighter1_won': ['mean', 'count'],
        'elo_diff': 'mean',
        'elo_prob_f1': 'mean'
    }).round(3)
    
    print("\nStyle Matchup Win Rates (F1 Style vs F2 Style):")
    print("-" * 60)
    
    for (f1_style, f2_style), row in matchups.iterrows():
        win_rate = row[('fighter1_won', 'mean')]
        count = row[('fighter1_won', 'count')]
        elo_expected = row[('elo_prob_f1', 'mean')]
        
        if count >= 10:  # Minimum sample
            diff = (win_rate - elo_expected) * 100
            edge_type = "OVER" if diff < 0 else "UNDER"
            
            print(f"   {f1_style:10s} vs {f2_style:10s}: "
                  f"Win: {win_rate:.1%} | ELO exp: {elo_expected:.1%} | "
                  f"n={int(count):3d} | {edge_type}valued by {abs(diff):.1f}%")
    
    # Specific: Strikers vs Wrestlers
    striker_vs_wrestler = df[(df['f1_style'] == 'striker') & (df['f2_style'] == 'wrestler')]
    wrestler_vs_striker = df[(df['f1_style'] == 'wrestler') & (df['f2_style'] == 'striker')]
    
    print(f"\n   STRIKERS vs WRESTLERS:")
    if len(striker_vs_wrestler) >= 10:
        svw_win = striker_vs_wrestler['fighter1_won'].mean()
        svw_exp = striker_vs_wrestler['elo_prob_f1'].mean()
        print(f"   Striker wins: {svw_win:.1%} (ELO expected: {svw_exp:.1%}) n={len(striker_vs_wrestler)}")
    
    if len(wrestler_vs_striker) >= 10:
        wvs_win = wrestler_vs_striker['fighter1_won'].mean()
        wvs_exp = wrestler_vs_striker['elo_prob_f1'].mean()
        print(f"   Wrestler wins: {wvs_win:.1%} (ELO expected: {wvs_exp:.1%}) n={len(wrestler_vs_striker)}")


def analyze_experience_edge(df):
    """
    Test: Is experience properly priced?
    
    Hypothesis: Debut fighters are undervalued or overvalued.
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: EXPERIENCE / DEBUT FIGHTERS")
    print("="*70)
    
    df = df.copy()
    
    # Debut fighters (0-2 UFC fights)
    df['f1_debut'] = df['f1_experience'] <= 2
    df['f2_debut'] = df['f2_experience'] <= 2
    
    # Veteran vs Debut
    vet_vs_debut = df[~df['f1_debut'] & df['f2_debut']]
    debut_vs_vet = df[df['f1_debut'] & ~df['f2_debut']]
    
    print(f"\nVETERAN vs DEBUT: {len(vet_vs_debut)} fights")
    if len(vet_vs_debut) > 0:
        win_rate = vet_vs_debut['fighter1_won'].mean()
        elo_exp = vet_vs_debut['elo_prob_f1'].mean()
        print(f"   Veteran win rate: {win_rate:.1%} (ELO expected: {elo_exp:.1%})")
        
        if win_rate > elo_exp:
            print("   >>> FINDING: Veterans OUTPERFORM vs debuts (experience matters)")
        else:
            print("   >>> FINDING: Debuts hold their own (talent scouts good)")
    
    print(f"\nDEBUT vs VETERAN: {len(debut_vs_vet)} fights")
    if len(debut_vs_vet) > 0:
        win_rate = debut_vs_vet['fighter1_won'].mean()
        elo_exp = debut_vs_vet['elo_prob_f1'].mean()
        print(f"   Debut win rate: {win_rate:.1%} (ELO expected: {elo_exp:.1%})")


def analyze_elo_ranges(df):
    """
    Test: Does model edge vary by ELO difference (favorite size)?
    
    Hypothesis: Big favorites (-300+) might be overbet, underdogs have value.
    """
    print("\n" + "="*70)
    print("ANALYSIS 5: FAVORITE VS UNDERDOG (By ELO Spread)")
    print("="*70)
    
    df = df.copy()
    
    # Categorize by ELO difference
    bins = [
        (-999, -150, "Big Underdog (F1)"),
        (-150, -50, "Small Underdog (F1)"),
        (-50, 50, "Pick'em"),
        (50, 150, "Small Favorite (F1)"),
        (150, 999, "Big Favorite (F1)"),
    ]
    
    print("\nWin rate by ELO spread:")
    print("-" * 60)
    
    for low, high, label in bins:
        subset = df[(df['elo_diff'] > low) & (df['elo_diff'] <= high)]
        if len(subset) >= 20:
            win_rate = subset['fighter1_won'].mean()
            elo_exp = subset['elo_prob_f1'].mean()
            diff = (win_rate - elo_exp) * 100
            
            print(f"   {label:20s}: "
                  f"Win: {win_rate:.1%} | ELO exp: {elo_exp:.1%} | "
                  f"n={len(subset):4d} | Diff: {diff:+.1f}%")
    
    # Big underdogs (F2 as underdog = F1 is big favorite)
    big_favs = df[df['elo_diff'] > 150]  # F1 heavily favored
    big_dogs = df[df['elo_diff'] < -150]  # F1 heavily unfavored
    
    print("\nLarge spread analysis:")
    
    if len(big_favs) > 0:
        # When F1 is big favorite
        fav_win_rate = big_favs['fighter1_won'].mean()
        fav_expected = big_favs['elo_prob_f1'].mean()
        print(f"   Big Favorites (F1): Win {fav_win_rate:.1%} (exp {fav_expected:.1%}) n={len(big_favs)}")
        
        if fav_win_rate < fav_expected - 0.02:
            print("   >>> FINDING: Big favorites UNDERPERFORM → Value on dogs!")
    
    if len(big_dogs) > 0:
        # When F1 is big underdog
        dog_win_rate = big_dogs['fighter1_won'].mean()
        dog_expected = big_dogs['elo_prob_f1'].mean()
        print(f"   Big Underdogs (F1): Win {dog_win_rate:.1%} (exp {dog_expected:.1%}) n={len(big_dogs)}")


def calculate_backtest_roi(df, strategy_fn):
    """
    Calculate ROI for a betting strategy.
    
    Args:
        df: Fight data
        strategy_fn: Function(row) -> (bet_on_f1: bool, edge: float) or None to skip
    
    Returns:
        dict with ROI metrics
    """
    total_bets = 0
    total_wagered = 0
    total_profit = 0
    wins = 0
    
    for idx, fight in df.iterrows():
        decision = strategy_fn(fight)
        
        if decision is None:
            continue
        
        bet_on_f1, edge = decision
        
        # Get odds
        if bet_on_f1:
            odds, _, implied, _ = simulate_market_odds(fight['elo_diff'])
        else:
            _, odds, _, implied = simulate_market_odds(fight['elo_diff'])
        
        # Place bet
        total_bets += 1
        total_wagered += 1
        
        profit = simulate_bet_result(bet_on_f1, fight['fighter1_won'], odds)
        total_profit += profit
        
        if profit > 0:
            wins += 1
    
    if total_bets == 0:
        return {'bets': 0, 'roi': 0, 'win_rate': 0}
    
    return {
        'bets': total_bets,
        'roi': total_profit / total_wagered,
        'win_rate': wins / total_bets,
        'profit': total_profit,
    }


def main():
    """Run full market inefficiency analysis."""
    print("="*70)
    print("MARKET INEFFICIENCY ANALYZER")
    print("Finding where the model might beat the market")
    print("="*70)
    
    # Load data
    print("\nLoading fights with context...")
    df = load_fights_with_context()
    print(f"Loaded {len(df)} fights")
    
    # Test period
    test_df = df[df['fight_date'] >= '2022-01-01'].copy()
    print(f"Test period (2022+): {len(test_df)} fights")
    
    # Run analyses
    analyze_public_bias(test_df)
    analyze_recency_bias(test_df)
    analyze_style_matchups(test_df)
    analyze_experience_edge(test_df)
    analyze_elo_ranges(test_df)
    
    # === FINAL RECOMMENDATIONS ===
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS")
    print("="*70)
    print("""
    Based on the analysis, here are potential edges to explore:
    
    1. BET AGAINST POPULAR FIGHTERS
       - When model shows 5%+ edge on lesser-known opponent
       - Public money inflates popular fighter's line
    
    2. BET ON FIGHTERS COMING OFF LOSSES  
       - Market overreacts to recent results
       - If ELO still supports them, they're undervalued
    
    3. STYLE MATCHUP EXPLOITATION
       - Identify mispriced style matchups
       - e.g., If wrestlers consistently beat strikers more than ELO predicts
    
    4. FADE BIG FAVORITES
       - Big favorites often fail to cover the implied probability
       - Small edge exists on underdogs at good odds
    
    CRITICAL: These need REAL odds data to validate!
    - Run BestFightOdds scraper
    - Compare model probability to actual closing lines
    - Track ROI with real odds, not simulated
    """)


if __name__ == '__main__':
    main()

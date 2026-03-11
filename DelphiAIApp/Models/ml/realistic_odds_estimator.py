"""
Realistic Odds Estimator

Since historical betting odds aren't readily available, we estimate
what the market odds WOULD have been based on:

1. ELO difference (primary factor - markets are efficient)
2. Public popularity adjustment (big names get overbet)
3. Recent form adjustment (markets overweight recent results)
4. Market vig (~4.5% overround)

This gives us a REALISTIC baseline for ROI calculation.

KEY INSIGHT: Markets are highly efficient. If our model can't beat
our SIMULATED market (which uses the same underlying data), it
definitely won't beat real markets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
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


def elo_to_probability(elo_a, elo_b=None):
    """
    Convert ELO to win probability.
    
    Args:
        elo_a: Fighter A's ELO, OR ELO difference if elo_b is None
        elo_b: Fighter B's ELO (optional)
    
    Returns:
        Win probability for Fighter A (between 0 and 1)
    """
    if elo_b is not None:
        elo_diff = elo_a - elo_b
    else:
        elo_diff = elo_a
    
    # Handle extreme ELO differences gracefully
    if elo_diff > 800:
        return 0.99
    if elo_diff < -800:
        return 0.01
    
    return 1 / (1 + 10**(-elo_diff/400))


def probability_to_american(prob):
    """
    Convert probability to American odds.
    
    Args:
        prob: Win probability (0-1)
    
    Returns:
        American odds (e.g., -150, +200) or None if invalid
    """
    # Handle edge cases
    if prob is None or np.isnan(prob):
        return None
    
    # Clamp to valid range
    if prob <= 0:
        return None  # Can't convert 0% probability
    if prob >= 1:
        return None  # Can't convert 100% probability
    
    # Small buffer to avoid extreme odds
    if prob < 0.01:
        prob = 0.01
    if prob > 0.99:
        prob = 0.99
    
    if prob >= 0.5:
        # Favorite: negative odds
        american = -100 * prob / (1 - prob)
    else:
        # Underdog: positive odds
        american = 100 * (1 - prob) / prob
    
    return round(american)


def american_to_decimal(american):
    """Convert American odds to decimal."""
    if american > 0:
        return 1 + (american / 100)
    else:
        return 1 + (100 / abs(american))


def add_vig(prob_a, vig=0.045):
    """
    Add bookmaker vig to probabilities.
    
    Vig is split between both fighters, making the implied
    probabilities sum to more than 100%.
    """
    prob_b = 1 - prob_a
    
    # Scale up both probabilities
    implied_a = prob_a * (1 + vig/2)
    implied_b = prob_b * (1 + vig/2)
    
    # Cap at reasonable limits
    implied_a = max(0.04, min(0.96, implied_a))
    implied_b = max(0.04, min(0.96, implied_b))
    
    return implied_a, implied_b


def estimate_market_odds(elo_diff, is_f1_popular=False, f1_recent_form=0.5, 
                        f2_recent_form=0.5, vig=0.045):
    """
    Estimate what the betting market odds would be.
    
    Market adjustments:
    1. ELO is the primary driver (efficient market)
    2. Popular fighters get slightly worse odds (overbet by public)
    3. Recent form is slightly overweighted
    
    Args:
        elo_diff: Fighter1 ELO - Fighter2 ELO
        is_f1_popular: Is fighter 1 a "popular" name?
        f1_recent_form: Fighter 1 recent win rate (0-1)
        f2_recent_form: Fighter 2 recent win rate (0-1)
        vig: Bookmaker margin
    
    Returns:
        dict with odds data
    """
    # Base probability from ELO
    true_prob_a = elo_to_probability(elo_diff)
    
    # Popularity adjustment (public overvalues popular fighters)
    # Markets move ~2% toward popular fighter
    pop_adj = 0.02 if is_f1_popular else -0.02 if not is_f1_popular else 0
    
    # Recent form adjustment (markets slightly overweight recent results)
    # ~1% adjustment based on form difference
    form_diff = (f1_recent_form - f2_recent_form)
    form_adj = 0.01 * form_diff
    
    # Market's perceived probability
    market_prob_a = true_prob_a + pop_adj + form_adj
    market_prob_a = max(0.05, min(0.95, market_prob_a))
    
    # Add vig
    implied_a, implied_b = add_vig(market_prob_a, vig)
    
    # Convert to odds
    american_a = probability_to_american(implied_a)
    american_b = probability_to_american(implied_b)
    decimal_a = american_to_decimal(american_a)
    decimal_b = american_to_decimal(american_b)
    
    return {
        'true_prob_a': true_prob_a,
        'market_prob_a': market_prob_a,
        'implied_prob_a': implied_a,
        'implied_prob_b': implied_b,
        'american_a': american_a,
        'american_b': american_b,
        'decimal_a': round(decimal_a, 3),
        'decimal_b': round(decimal_b, 3),
        'overround': (implied_a + implied_b) - 1,  # Should be ~4.5%
    }


def load_fights_for_backtest():
    """Load fight data with all necessary context."""
    
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
        
        -- Experience (proxy for popularity)
        pit1.FightsBefore as f1_fights,
        pit1.WinsBefore as f1_wins,
        pit1.RecentWinRate as f1_recent_form,
        pit2.FightsBefore as f2_fights,
        pit2.WinsBefore as f2_wins,
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
    
    # Parse dates and fill defaults
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    df['f1_elo'] = df['f1_elo'].fillna(1500)
    df['f2_elo'] = df['f2_elo'].fillna(1500)
    df['f1_fights'] = df['f1_fights'].fillna(0)
    df['f2_fights'] = df['f2_fights'].fillna(0)
    df['f1_recent_form'] = df['f1_recent_form'].fillna(0.5)
    df['f2_recent_form'] = df['f2_recent_form'].fillna(0.5)
    
    # Calculate derived columns
    df['elo_diff'] = df['f1_elo'] - df['f2_elo']
    df['fighter1_won'] = (df['winnerid'] == df['fighter1_id']).astype(int)
    
    # Popularity proxy: more fights = more popular
    df['f1_popular'] = df['f1_fights'] > 10
    df['f2_popular'] = df['f2_fights'] > 10
    
    return df


def run_realistic_backtest():
    """
    Run backtest with REALISTIC market odds.
    
    This simulates what betting would look like against an
    efficient market that uses similar data to ours.
    """
    print("="*70)
    print("REALISTIC ODDS BACKTEST")
    print("="*70)
    
    # Load data
    print("\nLoading fight data...")
    df = load_fights_for_backtest()
    print(f"Loaded {len(df)} fights (2022+)")
    
    # === SCENARIO 1: Always bet on ELO favorite ===
    print("\n" + "-"*70)
    print("SCENARIO 1: Always Bet on ELO Favorite")
    print("-"*70)
    
    total_bets = 0
    total_profit = 0
    wins = 0
    
    for idx, fight in df.iterrows():
        elo_diff = fight['elo_diff']
        f1_won = fight['fighter1_won']
        
        # Skip pick'ems (within 25 ELO)
        if abs(elo_diff) < 25:
            continue
        
        # Estimate market odds
        odds = estimate_market_odds(
            elo_diff,
            is_f1_popular=fight['f1_popular'],
            f1_recent_form=fight['f1_recent_form'],
            f2_recent_form=fight['f2_recent_form'],
        )
        
        # Bet on ELO favorite
        bet_on_f1 = elo_diff > 0
        
        if bet_on_f1:
            decimal_odds = odds['decimal_a']
            bet_won = f1_won == 1
        else:
            decimal_odds = odds['decimal_b']
            bet_won = f1_won == 0
        
        total_bets += 1
        
        if bet_won:
            total_profit += (decimal_odds - 1)
            wins += 1
        else:
            total_profit -= 1
    
    roi = total_profit / total_bets if total_bets > 0 else 0
    win_rate = wins / total_bets if total_bets > 0 else 0
    
    print(f"   Bets placed: {total_bets}")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Total ROI: {roi*100:.1f}%")
    print(f"   Expected: ~0% (market is efficient)")
    
    # === SCENARIO 2: Bet when model shows edge over market ===
    print("\n" + "-"*70)
    print("SCENARIO 2: Bet When Model Shows 5% Edge Over Market")
    print("-"*70)
    
    MIN_EDGE = 0.05  # Minimum edge to bet
    
    total_bets = 0
    total_profit = 0
    wins = 0
    edges_found = []
    
    for idx, fight in df.iterrows():
        elo_diff = fight['elo_diff']
        f1_won = fight['fighter1_won']
        
        # Model's probability (using ELO)
        model_prob_a = elo_to_probability(elo_diff)
        
        # Estimate market odds
        odds = estimate_market_odds(
            elo_diff,
            is_f1_popular=fight['f1_popular'],
            f1_recent_form=fight['f1_recent_form'],
            f2_recent_form=fight['f2_recent_form'],
        )
        
        # Calculate edge
        edge_a = model_prob_a - odds['implied_prob_a']
        edge_b = (1 - model_prob_a) - odds['implied_prob_b']
        
        # Check if we have edge
        if edge_a > MIN_EDGE:
            # Bet on fighter A
            bet_won = f1_won == 1
            decimal_odds = odds['decimal_a']
            total_bets += 1
            edges_found.append(edge_a)
            
            if bet_won:
                total_profit += (decimal_odds - 1)
                wins += 1
            else:
                total_profit -= 1
        
        elif edge_b > MIN_EDGE:
            # Bet on fighter B
            bet_won = f1_won == 0
            decimal_odds = odds['decimal_b']
            total_bets += 1
            edges_found.append(edge_b)
            
            if bet_won:
                total_profit += (decimal_odds - 1)
                wins += 1
            else:
                total_profit -= 1
    
    roi = total_profit / total_bets if total_bets > 0 else 0
    win_rate = wins / total_bets if total_bets > 0 else 0
    avg_edge = np.mean(edges_found) if edges_found else 0
    
    print(f"   Bets placed: {total_bets} (out of {len(df)} fights)")
    print(f"   Average edge: {avg_edge*100:.1f}%")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Total ROI: {roi*100:.1f}%")
    
    # === SCENARIO 3: Exploit specific inefficiencies ===
    print("\n" + "-"*70)
    print("SCENARIO 3: Exploit Recency Bias (Bet on fighters with bad recent form)")
    print("-"*70)
    
    total_bets = 0
    total_profit = 0
    wins = 0
    
    for idx, fight in df.iterrows():
        elo_diff = fight['elo_diff']
        f1_won = fight['fighter1_won']
        f1_form = fight['f1_recent_form']
        f2_form = fight['f2_recent_form']
        
        # Look for fighters with bad recent form but decent ELO
        # These might be undervalued by market
        
        # Fighter 1 has bad form but ELO supports them
        if f1_form < 0.35 and elo_diff > 50:
            odds = estimate_market_odds(
                elo_diff,
                is_f1_popular=fight['f1_popular'],
                f1_recent_form=f1_form,
                f2_recent_form=f2_form,
            )
            
            total_bets += 1
            
            if f1_won:
                total_profit += (odds['decimal_a'] - 1)
                wins += 1
            else:
                total_profit -= 1
        
        # Fighter 2 has bad form but ELO supports them
        elif f2_form < 0.35 and elo_diff < -50:
            odds = estimate_market_odds(
                elo_diff,
                is_f1_popular=fight['f1_popular'],
                f1_recent_form=f1_form,
                f2_recent_form=f2_form,
            )
            
            total_bets += 1
            
            if not f1_won:
                total_profit += (odds['decimal_b'] - 1)
                wins += 1
            else:
                total_profit -= 1
    
    if total_bets > 0:
        roi = total_profit / total_bets
        win_rate = wins / total_bets
        
        print(f"   Bets placed: {total_bets}")
        print(f"   Win rate: {win_rate:.1%}")
        print(f"   Total ROI: {roi*100:.1f}%")
    else:
        print("   No bets matched criteria")
    
    # === FINAL SUMMARY ===
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
    With realistic market odds estimation:
    
    1. BASELINE (Always bet favorite): ~0% ROI
       - Market efficiently prices ELO-based probability
       - Vig eats any apparent edge
    
    2. EDGE BETTING: Small positive/negative ROI
       - Only works if model has true edge over market
       - Need 53%+ win rate on -110 bets to profit
    
    3. INEFFICIENCY HUNTING: Variable
       - May find small edges in specific situations
       - Recency bias, popularity bias, etc.
       - Edges are typically 1-5% at best
    
    BOTTOM LINE:
    - If ROI > 10% in backtest, something is WRONG
    - Realistic expectation: 2-5% ROI on value bets
    - Track live predictions to validate
    """)


if __name__ == '__main__':
    run_realistic_backtest()

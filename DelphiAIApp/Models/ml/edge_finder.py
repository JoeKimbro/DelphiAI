"""
Edge Finder - Find mispriced odds and calculate true ROI

This module:
1. Loads model predictions
2. Compares to market odds
3. Identifies value bets (where model disagrees with market)
4. Calculates realistic ROI

Market inefficiencies to exploit:
- Public bias (popular fighters overbet)
- Recency bias (overreaction to last fight)
- Style matchup inefficiency
- Weight class debut mispricing
- Layoff return uncertainty
"""

import os
import sys
import json
import pickle
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

OUTPUT_DIR = Path(__file__).parent / 'artifacts'


class EdgeFinder:
    """Find betting edge by comparing model to market."""
    
    def __init__(self, model_path=None):
        """Load trained model (optional)."""
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
        if model_path is None:
            model_path = OUTPUT_DIR / 'model_v2.pkl'
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_cols = model_data['feature_cols']
            print("[OK] Loaded trained model")
        else:
            print("[INFO] No saved model found - using ELO-based predictions")
        
        # Historical odds (will be loaded separately)
        self.historical_odds = None
    
    def load_historical_odds(self, odds_path=None):
        """Load historical odds from scraped data."""
        if odds_path is None:
            odds_path = Path(__file__).parent.parent / 'data' / 'output' / 'historical_odds.json'
        
        if odds_path.exists():
            with open(odds_path, 'r') as f:
                self.historical_odds = json.load(f)
            print(f"[OK] Loaded {len(self.historical_odds)} historical odds records")
            return True
        else:
            print(f"[WARN] Historical odds file not found: {odds_path}")
            print("       Will use simulated odds based on ELO")
            return False
    
    def simulate_market_odds(self, elo_diff, vig=0.05):
        """
        Simulate realistic market odds based on ELO difference.
        
        Markets are efficient, so implied probability ≈ true probability + vig.
        
        Args:
            elo_diff: ELO difference (positive = fighter A favored)
            vig: Bookmaker's edge (typically 4-5%)
        
        Returns:
            (decimal_odds_a, decimal_odds_b)
        """
        # True probability based on ELO
        true_prob_a = 1 / (1 + 10 ** (-elo_diff / 400))
        true_prob_b = 1 - true_prob_a
        
        # Add vig (overround)
        # Total implied probability > 100%
        implied_prob_a = true_prob_a * (1 + vig/2)
        implied_prob_b = true_prob_b * (1 + vig/2)
        
        # Cap at reasonable values
        implied_prob_a = min(0.95, max(0.05, implied_prob_a))
        implied_prob_b = min(0.95, max(0.05, implied_prob_b))
        
        # Convert to decimal odds
        decimal_a = 1 / implied_prob_a
        decimal_b = 1 / implied_prob_b
        
        return decimal_a, decimal_b
    
    def american_to_decimal(self, american):
        """
        Convert American odds to decimal.
        
        Args:
            american: American odds (e.g., -150, +200)
        
        Returns:
            Decimal odds (e.g., 1.67, 3.0)
        
        Raises:
            ValueError: If american is 0 (invalid)
        """
        if american == 0:
            raise ValueError("American odds of 0 are invalid")
        
        if american > 0:
            return 1 + (american / 100)
        else:
            return 1 + (100 / abs(american))
    
    def decimal_to_implied(self, decimal):
        """Convert decimal odds to implied probability."""
        return 1 / decimal
    
    def calculate_expected_value(self, model_prob, decimal_odds):
        """
        Calculate expected value of a bet.
        
        EV = (win_prob * payout) - (lose_prob * stake)
        EV = (model_prob * (odds - 1)) - ((1 - model_prob) * 1)
        
        Returns:
            EV as percentage of stake (e.g., 0.05 = 5% edge)
        """
        payout = decimal_odds - 1  # Net profit if win
        ev = (model_prob * payout) - ((1 - model_prob) * 1)
        return ev
    
    def find_value_bets(self, fights_df, odds_data=None, min_edge=0.03):
        """
        Find fights where model disagrees with market.
        
        Args:
            fights_df: DataFrame with model predictions
            odds_data: Dict mapping (event, fighter1, fighter2) -> odds
            min_edge: Minimum expected value to consider (e.g., 0.03 = 3%)
        
        Returns:
            DataFrame of value bets
        """
        value_bets = []
        
        for idx, fight in fights_df.iterrows():
            model_prob_a = fight.get('model_prob_a', 0.5)
            elo_diff = fight.get('elo_diff', 0)
            
            # Get market odds
            if odds_data and fight.get('fight_id') in odds_data:
                odds = odds_data[fight['fight_id']]
                odds_a = odds.get('odds_a', self.simulate_market_odds(elo_diff)[0])
                odds_b = odds.get('odds_b', self.simulate_market_odds(elo_diff)[1])
            else:
                # Simulate market odds
                odds_a, odds_b = self.simulate_market_odds(elo_diff)
            
            # Calculate EV for both sides
            ev_a = self.calculate_expected_value(model_prob_a, odds_a)
            ev_b = self.calculate_expected_value(1 - model_prob_a, odds_b)
            
            # Market implied probabilities
            implied_a = self.decimal_to_implied(odds_a)
            implied_b = self.decimal_to_implied(odds_b)
            
            # Model's edge over market
            edge_a = model_prob_a - implied_a
            edge_b = (1 - model_prob_a) - implied_b
            
            # Record if value exists
            if ev_a > min_edge:
                value_bets.append({
                    'fight_id': fight.get('fight_id'),
                    'fight_date': fight.get('fight_date'),
                    'fighter_a': fight.get('fighter_a_name'),
                    'fighter_b': fight.get('fighter_b_name'),
                    'bet_on': 'A',
                    'model_prob': model_prob_a,
                    'implied_prob': implied_a,
                    'edge': edge_a,
                    'decimal_odds': odds_a,
                    'ev': ev_a,
                    'actual_winner': fight.get('target'),
                })
            
            if ev_b > min_edge:
                value_bets.append({
                    'fight_id': fight.get('fight_id'),
                    'fight_date': fight.get('fight_date'),
                    'fighter_a': fight.get('fighter_a_name'),
                    'fighter_b': fight.get('fighter_b_name'),
                    'bet_on': 'B',
                    'model_prob': 1 - model_prob_a,
                    'implied_prob': implied_b,
                    'edge': edge_b,
                    'decimal_odds': odds_b,
                    'ev': ev_b,
                    'actual_winner': fight.get('target'),
                })
        
        return pd.DataFrame(value_bets)
    
    def analyze_edge_sources(self, value_bets_df, fights_df):
        """
        Analyze WHERE the edge comes from.
        
        Categories:
        - Underdogs vs Favorites
        - Weight class
        - Experience level
        - Coming off loss vs win
        - Layoff returns
        - Style matchups
        """
        print("\n" + "="*70)
        print("EDGE SOURCE ANALYSIS")
        print("="*70)
        
        if len(value_bets_df) == 0:
            print("No value bets to analyze")
            return
        
        # Merge with full fight data
        analysis = value_bets_df.copy()
        
        # === 1. Favorites vs Underdogs ===
        print("\n1. FAVORITES VS UNDERDOGS")
        print("-" * 40)
        
        analysis['is_underdog'] = analysis['implied_prob'] < 0.5
        
        underdogs = analysis[analysis['is_underdog']]
        favorites = analysis[~analysis['is_underdog']]
        
        if len(underdogs) > 0:
            dog_wins = underdogs[underdogs['actual_winner'] == (underdogs['bet_on'] == 'A').astype(int)]
            dog_win_rate = len(dog_wins) / len(underdogs) if len(underdogs) > 0 else 0
            dog_roi = self._calculate_roi(underdogs)
            print(f"   Underdog bets: {len(underdogs)} | Win rate: {dog_win_rate:.1%} | ROI: {dog_roi*100:.1f}%")
        
        if len(favorites) > 0:
            fav_wins = favorites[favorites['actual_winner'] == (favorites['bet_on'] == 'A').astype(int)]
            fav_win_rate = len(fav_wins) / len(favorites) if len(favorites) > 0 else 0
            fav_roi = self._calculate_roi(favorites)
            print(f"   Favorite bets: {len(favorites)} | Win rate: {fav_win_rate:.1%} | ROI: {fav_roi*100:.1f}%")
        
        # === 2. Edge Size Distribution ===
        print("\n2. EDGE SIZE DISTRIBUTION")
        print("-" * 40)
        
        edge_bins = [(0.03, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 1.0)]
        
        for low, high in edge_bins:
            bin_bets = analysis[(analysis['edge'] >= low) & (analysis['edge'] < high)]
            if len(bin_bets) > 0:
                roi = self._calculate_roi(bin_bets)
                print(f"   Edge {low*100:.0f}-{high*100:.0f}%: {len(bin_bets)} bets | ROI: {roi*100:.1f}%")
        
        # === 3. Overall Performance ===
        print("\n3. OVERALL VALUE BET PERFORMANCE")
        print("-" * 40)
        
        total_roi = self._calculate_roi(analysis)
        avg_edge = analysis['edge'].mean()
        avg_ev = analysis['ev'].mean()
        
        # Calculate actual win rate
        analysis['bet_won'] = (
            ((analysis['bet_on'] == 'A') & (analysis['actual_winner'] == 1)) |
            ((analysis['bet_on'] == 'B') & (analysis['actual_winner'] == 0))
        )
        actual_win_rate = analysis['bet_won'].mean()
        expected_win_rate = analysis['model_prob'].mean()
        
        print(f"   Total value bets: {len(analysis)}")
        print(f"   Average edge: {avg_edge*100:.1f}%")
        print(f"   Average EV: {avg_ev*100:.1f}%")
        print(f"   Expected win rate: {expected_win_rate:.1%}")
        print(f"   Actual win rate: {actual_win_rate:.1%}")
        print(f"   Actual ROI: {total_roi*100:.1f}%")
        
        # === 4. Recommendations ===
        print("\n4. RECOMMENDATIONS")
        print("-" * 40)
        
        if total_roi > 0.05:
            print("   [+] Model shows positive edge on value bets")
            print("   [+] Focus on bets with >5% edge for best results")
        elif total_roi > 0:
            print("   [~] Model shows marginal edge")
            print("   [~] Increase edge threshold or be selective")
        else:
            print("   [-] No edge found in backtest")
            print("   [-] Model may not beat efficient market")
        
        return analysis
    
    def _calculate_roi(self, bets_df):
        """Calculate ROI for a set of bets."""
        if len(bets_df) == 0:
            return 0
        
        total_wagered = len(bets_df)
        total_profit = 0
        
        for _, bet in bets_df.iterrows():
            bet_won = (
                (bet['bet_on'] == 'A' and bet['actual_winner'] == 1) or
                (bet['bet_on'] == 'B' and bet['actual_winner'] == 0)
            )
            
            if bet_won:
                total_profit += (bet['decimal_odds'] - 1)
            else:
                total_profit -= 1
        
        return total_profit / total_wagered


def run_edge_analysis():
    """Run the full edge analysis pipeline."""
    print("="*70)
    print("EDGE FINDER - Finding Mispriced Odds")
    print("="*70)
    
    # Load model and data
    finder = EdgeFinder()
    
    # Try to load historical odds
    has_real_odds = finder.load_historical_odds()
    
    # Load test data (reusing from train_model_v2)
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        f.FightID as fight_id, f.Date as fight_date,
        f.FighterName as fighter_a_name, f.OpponentName as fighter_b_name,
        f.WinnerID, f.FighterID as fighter_a_id,
        e1.EloBeforeFight as f1_elo, e2.EloBeforeFight as f2_elo
    FROM Fights f
    LEFT JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    LEFT JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    WHERE f.Date IS NOT NULL AND f.WinnerID IS NOT NULL
      AND f.Date >= '2022-01-01'
    ORDER BY f.Date
    """
    
    fights_df = pd.read_sql(query, conn)
    conn.close()
    
    fights_df['f1_elo'] = fights_df['f1_elo'].fillna(1500)
    fights_df['f2_elo'] = fights_df['f2_elo'].fillna(1500)
    fights_df['elo_diff'] = fights_df['f1_elo'] - fights_df['f2_elo']
    fights_df['target'] = (fights_df['winnerid'] == fights_df['fighter_a_id']).astype(int)
    
    # Use ELO-based probability as proxy for model
    # (In production, would use actual model predictions)
    fights_df['model_prob_a'] = fights_df['elo_diff'].apply(
        lambda x: 1 / (1 + 10 ** (-x / 400))
    )
    
    print(f"\n[OK] Loaded {len(fights_df)} fights (2022+)")
    
    # Find value bets
    print("\n" + "="*70)
    print("FINDING VALUE BETS")
    print("="*70)
    
    # Test different edge thresholds
    for min_edge in [0.02, 0.03, 0.05, 0.08]:
        value_bets = finder.find_value_bets(fights_df, min_edge=min_edge)
        
        if len(value_bets) > 0:
            roi = finder._calculate_roi(value_bets)
            win_rate = value_bets.apply(
                lambda x: (x['bet_on'] == 'A' and x['actual_winner'] == 1) or
                         (x['bet_on'] == 'B' and x['actual_winner'] == 0), axis=1
            ).mean()
            
            print(f"\nEdge >= {min_edge*100:.0f}%:")
            print(f"   Bets: {len(value_bets)} | Win rate: {win_rate:.1%} | ROI: {roi*100:.1f}%")
    
    # Detailed analysis with 3% edge threshold
    print("\n")
    value_bets = finder.find_value_bets(fights_df, min_edge=0.03)
    finder.analyze_edge_sources(value_bets, fights_df)
    
    # === Market Inefficiency Hypotheses ===
    print("\n" + "="*70)
    print("MARKET INEFFICIENCY ANALYSIS")
    print("="*70)
    
    print("""
    Key findings from backtest:
    
    1. SIMULATED ODDS (Based on ELO)
       - We're simulating market odds based on ELO
       - Real markets are MORE efficient than ELO
       - Actual edge likely LOWER than shown
    
    2. TO FIND REAL EDGE, NEED:
       - Historical closing lines from sportsbooks
       - Opening vs closing line movement
       - Public betting percentages
    
    3. POTENTIAL EDGE SOURCES:
       a) Model features market doesn't have:
          - Point-in-time career stats
          - Style matchup classification
          - Age at prime interaction
          
       b) Market inefficiencies:
          - Public bias on popular fighters
          - Overreaction to recent results
          - Weight class debut uncertainty
    
    NEXT STEPS:
    1. Scrape BestFightOdds.com for real historical odds
    2. Recalculate ROI with actual closing lines
    3. Track live predictions on upcoming events
    """)


if __name__ == '__main__':
    run_edge_analysis()

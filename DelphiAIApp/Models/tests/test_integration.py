"""
Integration Tests

Tests the full pipeline from data to predictions.
These tests ensure components work together correctly.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))


class TestPredictionPipeline:
    """Test the complete prediction pipeline."""
    
    def test_elo_to_probability_to_odds_roundtrip(self):
        """Test ELO → Probability → Odds → Implied Prob roundtrip."""
        from ml.realistic_odds_estimator import elo_to_probability, probability_to_american
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder()
        
        # Start with ELO difference (100 points = ~64% win prob)
        elo_a, elo_b = 1550, 1450  # 100 point difference
        
        # Step 1: ELO → Probability
        prob_a = elo_to_probability(elo_a, elo_b)
        assert 0.55 < prob_a < 0.75  # Should favor higher ELO
        
        # Step 2: Probability → American Odds
        american = probability_to_american(prob_a)
        assert american is not None
        assert american < 0  # Should be favorite (negative American)
        
        # Step 3: American → Decimal
        decimal = finder.american_to_decimal(american)
        assert decimal > 1
        
        # Step 4: Decimal → Implied Prob
        implied = 1 / decimal
        
        # Roundtrip should be close (no vig in this path)
        assert abs(implied - prob_a) < 0.02
    
    def test_market_odds_with_vig_pipeline(self):
        """Test market odds estimation with vig."""
        from ml.realistic_odds_estimator import estimate_market_odds
        
        # Fair 60% probability (ELO diff)
        elo_diff = 200  # F1 is favored
        
        odds = estimate_market_odds(elo_diff)
        
        # Convert to implied probabilities
        implied_f1 = odds['implied_prob_a']
        implied_f2 = odds['implied_prob_b']
        
        # Should sum to >1 due to vig
        total = implied_f1 + implied_f2
        assert total > 1.0, "Vig should make implied probs sum to >1"
    
    def test_edge_calculation_pipeline(self):
        """Test edge calculation from ELO to bet decision."""
        from ml.realistic_odds_estimator import elo_to_probability, estimate_market_odds
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder()
        
        # Fighter A: ELO 1600, Fighter B: ELO 1400
        f1_elo, f2_elo = 1600, 1400
        elo_diff = f1_elo - f2_elo
        
        # Model probability
        model_prob = elo_to_probability(elo_diff)
        
        # Market odds (with some public bias toward F2)
        odds = estimate_market_odds(elo_diff, f1_recent_form=0.5, f2_recent_form=0.6)  # Slight F2 form
        
        # Implied probability
        implied_f1 = odds['implied_prob_a']
        
        # Edge = Model - Market
        edge = model_prob - implied_f1
        
        # Edge should be reasonable
        assert edge > -0.10  # Not too negative
        assert edge < 0.20   # Not unrealistically positive


class TestDataFlowIntegration:
    """Test data flows correctly through system."""
    
    def test_fighter_stats_to_features(self):
        """Test fighter stats become proper features."""
        # Simulate fighter stats
        fighter_data = {
            'name': 'Test Fighter',
            'height': 72,  # inches
            'reach': 75,   # inches
            'stance': 'Orthodox',
            'wins': 15,
            'losses': 3,
            'slpm': 4.5,   # Strikes per minute
            'tdavg': 2.0,  # Takedowns per 15 min
        }
        
        # Height advantage calculation
        opponent_height = 70
        height_diff = fighter_data['height'] - opponent_height
        assert height_diff == 2
        
        # Reach advantage
        opponent_reach = 72
        reach_diff = fighter_data['reach'] - opponent_reach
        assert reach_diff == 3
        
        # Win rate
        total_fights = fighter_data['wins'] + fighter_data['losses']
        win_rate = fighter_data['wins'] / total_fights
        assert 0.80 < win_rate < 0.85
    
    def test_nationality_feature_creation(self):
        """Test nationality becomes proper feature."""
        from ml.analyze_nationality import extract_country, get_region
        
        # Fighter place of birth
        place = "Makhachkala, Dagestan, Russia"
        
        # Extract country
        country = extract_country(place)
        assert country in ['Dagestan', 'Russia']
        
        # Get region
        region = get_region(country)
        assert region in ['Caucasus', 'Europe', 'Eastern Europe']
    
    def test_odds_to_ev_calculation(self):
        """Test odds become proper EV calculation."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder()
        
        # Market odds
        american_odds = -150
        decimal_odds = finder.american_to_decimal(american_odds)
        assert decimal_odds == 1.6666666666666667 or abs(decimal_odds - 1.67) < 0.01
        
        # Implied probability
        implied_prob = 1 / decimal_odds
        assert abs(implied_prob - 0.60) < 0.01
        
        # Model probability
        model_prob = 0.65  # We think 65%
        
        # Edge
        edge = model_prob - implied_prob
        assert abs(edge - 0.05) < 0.01
        
        # Expected Value
        # EV = (win_prob * profit) - (loss_prob * loss)
        # If we bet $100 at -150:
        bet = 100
        profit_if_win = bet / 1.5  # ~$66.67 profit
        
        ev = model_prob * profit_if_win - (1 - model_prob) * bet
        # EV = 0.65 * 66.67 - 0.35 * 100 = 43.34 - 35 = 8.34
        assert ev > 0  # Positive EV bet


class TestBacktestPipeline:
    """Test backtesting pipeline."""
    
    def test_simulated_betting_roi(self):
        """Test simulated betting produces valid ROI."""
        # Simulate 100 bets with edge
        np.random.seed(42)
        
        bets = []
        for _ in range(100):
            # Our edge: 5% above market
            model_prob = 0.55
            market_prob = 0.50
            decimal_odds = 1 / market_prob  # 2.0
            
            # Simulate outcome (based on model prob)
            won = np.random.random() < model_prob
            
            # Calculate P&L
            bet_amount = 100
            if won:
                profit = bet_amount * (decimal_odds - 1)
            else:
                profit = -bet_amount
            
            bets.append(profit)
        
        total_profit = sum(bets)
        total_wagered = 100 * len(bets)
        roi = total_profit / total_wagered
        
        # With 5% edge, expect positive ROI (but with variance)
        # Over 100 bets, could be -20% to +30%
        assert -0.30 < roi < 0.40
    
    def test_kelly_bet_sizing_over_time(self):
        """Test Kelly sizing maintains bankroll."""
        from ml.safeguards import safe_kelly
        
        np.random.seed(42)
        
        initial_bankroll = 10000
        bankroll = initial_bankroll
        
        # Simulate 50 bets
        for _ in range(50):
            if bankroll < 100:  # Stop if busted
                break
            
            # Get Kelly bet
            result = safe_kelly(
                estimated_edge=0.08,
                decimal_odds=2.0,
                bankroll=bankroll
            )
            
            if result['should_bet']:
                bet = result['bet_amount']
                
                # Simulate outcome (55% win rate)
                won = np.random.random() < 0.55
                
                if won:
                    bankroll += bet * (2.0 - 1)
                else:
                    bankroll -= bet
        
        # Should not be busted (fractional Kelly protects)
        assert bankroll > 0
        # May or may not be profitable due to variance


class TestSafeguardIntegration:
    """Test safeguards integrate with analysis."""
    
    def test_edge_analysis_with_safeguards(self):
        """Test edge analysis respects safeguards."""
        from ml.safeguards import validate_edge, validate_sample_size
        
        # Simulate analysis results
        analysis_results = [
            {'edge': 0.02, 'n_bets': 30, 'win_rate': 0.52},   # Small edge, small sample
            {'edge': 0.15, 'n_bets': 50, 'win_rate': 0.65},   # Big edge, small sample
            {'edge': 0.05, 'n_bets': 300, 'win_rate': 0.55},  # Medium edge, good sample
        ]
        
        for result in analysis_results:
            # Check sample size
            is_valid, msg = validate_sample_size(result['n_bets'], 'overall_edge')
            
            # Validate edge
            validation = validate_edge(
                edge_pct=result['edge'],
                n_bets=result['n_bets'],
                win_rate=result['win_rate'],
                expected_rate=0.50
            )
            
            # First two should be flagged
            if result['n_bets'] < 200:
                assert not is_valid or len(validation['warnings']) > 0
    
    def test_nationality_analysis_with_safeguards(self):
        """Test nationality analysis respects safeguards."""
        from ml.safeguards import validate_nationality_edge
        
        # Simulate country analysis results
        countries = [
            {'country': 'USA', 'n_fights': 800, 'win_rate': 0.49, 'p_value': 0.30},
            {'country': 'Brazil', 'n_fights': 400, 'win_rate': 0.52, 'p_value': 0.15},
            {'country': 'Dagestan', 'n_fights': 45, 'win_rate': 0.72, 'p_value': 0.01},
        ]
        
        for country in countries:
            result = validate_nationality_edge(
                country=country['country'],
                n_fights=country['n_fights'],
                win_rate=country['win_rate'],
                expected_rate=0.50,
                p_value=country['p_value']
            )
            
            # Dagestan should be flagged for small sample
            if country['country'] == 'Dagestan':
                assert not result['is_valid']
                assert any('INSUFFICIENT' in w for w in result['warnings'])


class TestEndToEndPrediction:
    """Test end-to-end prediction flow."""
    
    def test_prediction_generation(self):
        """Test generating a prediction from scratch."""
        from ml.realistic_odds_estimator import elo_to_probability, estimate_market_odds
        from ml.safeguards import safe_kelly, validate_edge
        
        # Fighter A: ELO 1650 (favored)
        # Fighter B: ELO 1450 (underdog)
        f1_elo, f2_elo = 1650, 1450
        elo_diff = f1_elo - f2_elo
        
        # Step 1: Model probability
        model_prob = elo_to_probability(elo_diff)
        assert model_prob > 0.5
        
        # Step 2: Market odds (simulated)
        odds = estimate_market_odds(elo_diff)
        
        # Step 3: Implied probability
        implied_f1 = odds['implied_prob_a']
        
        # Step 4: Edge
        edge = model_prob - implied_f1
        
        # Step 5: Validate edge
        # (In real scenario, would need historical data)
        
        # Step 6: Kelly sizing
        if edge > 0.02:
            kelly = safe_kelly(
                estimated_edge=edge,
                decimal_odds=odds['decimal_a'],
                bankroll=10000
            )
            
            # Should be conservative bet
            if kelly['should_bet']:
                assert kelly['bet_percentage'] <= 0.05


class TestRealWorldScenarios:
    """Test realistic scenarios."""
    
    def test_favorite_vs_underdog_prediction(self):
        """Test predicting a typical favorite vs underdog fight."""
        from ml.realistic_odds_estimator import elo_to_probability, probability_to_american
        
        # Heavy favorite: ELO 1700 (top 10)
        # Underdog: ELO 1400 (unranked)
        f1_elo, f2_elo = 1700, 1400
        
        prob = elo_to_probability(f1_elo, f2_elo)
        american = probability_to_american(prob)
        
        # Should be heavy favorite
        assert prob > 0.75
        assert american < -200
    
    def test_even_matchup_prediction(self):
        """Test predicting an even matchup."""
        from ml.realistic_odds_estimator import elo_to_probability, probability_to_american
        
        # Close ELOs
        f1_elo, f2_elo = 1550, 1530
        
        prob = elo_to_probability(f1_elo, f2_elo)
        
        # Should be close to 50-50
        assert 0.48 < prob < 0.55
    
    def test_upset_value_detection(self):
        """Test detecting value on upset potential."""
        from ml.realistic_odds_estimator import elo_to_probability, estimate_market_odds
        
        # Model: Underdog is better than market thinks
        # Market might be biased toward the popular fighter
        
        # True ELOs (unknown to market)
        true_elo_diff = 1500 - 1480  # 20 point diff
        true_prob_f1 = elo_to_probability(true_elo_diff)
        
        # Market perception: F1 is heavily favored
        perceived_elo_diff = 1600 - 1400  # 200 point diff
        odds = estimate_market_odds(perceived_elo_diff)
        
        implied_f2 = odds['implied_prob_b']
        
        # True probability for F2
        true_prob_f2 = 1 - true_prob_f1
        
        # Edge on underdog
        edge_f2 = true_prob_f2 - implied_f2
        
        # Should have significant edge on underdog
        assert edge_f2 > 0.10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

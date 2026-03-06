"""
Tests for edge analysis and market inefficiency detection.

Tests:
- Value bet identification
- ROI calculations
- Statistical significance tests
- Betting simulation logic
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))


class TestValueBetIdentification:
    """Test value bet identification logic."""
    
    def test_positive_edge_identified(self):
        """Test that positive edge is correctly identified."""
        model_prob = 0.60  # Model says 60% chance
        implied_prob = 0.50  # Market implies 50%
        
        edge = model_prob - implied_prob
        assert edge > 0
        assert abs(edge - 0.10) < 0.001
    
    def test_no_edge_when_model_agrees_with_market(self):
        """Test no edge when model agrees with market."""
        model_prob = 0.55
        implied_prob = 0.55
        
        edge = model_prob - implied_prob
        assert abs(edge) < 0.001
    
    def test_minimum_edge_threshold(self):
        """Test minimum edge threshold for betting."""
        MIN_EDGE = 0.05  # 5% minimum
        
        # 4% edge - should not bet
        small_edge = 0.04
        assert small_edge < MIN_EDGE
        
        # 6% edge - should bet
        good_edge = 0.06
        assert good_edge >= MIN_EDGE
    
    def test_edge_both_directions(self):
        """Test edge calculation for both fighters."""
        model_prob_a = 0.45
        implied_prob_a = 0.50
        implied_prob_b = 0.50  # 50% each with no vig
        
        edge_a = model_prob_a - implied_prob_a
        edge_b = (1 - model_prob_a) - implied_prob_b
        
        # If A is undervalued, A has negative edge, B has positive
        assert edge_a < 0
        assert edge_b > 0
        
        # Edges should be opposite
        assert abs(edge_a + edge_b) < 0.001


class TestROICalculations:
    """Test ROI calculation logic."""
    
    def test_roi_winning_bet(self):
        """Test ROI for a winning bet."""
        stake = 1.0
        decimal_odds = 2.0
        won = True
        
        if won:
            profit = stake * (decimal_odds - 1)
        else:
            profit = -stake
        
        roi = profit / stake
        assert abs(roi - 1.0) < 0.001  # 100% ROI on winning even money
    
    def test_roi_losing_bet(self):
        """Test ROI for a losing bet."""
        stake = 1.0
        won = False
        
        profit = -stake
        roi = profit / stake
        
        assert abs(roi - (-1.0)) < 0.001  # -100% ROI on loss
    
    def test_roi_multiple_bets(self):
        """Test ROI calculation for multiple bets."""
        # 3 bets: 2 wins at 2.0, 1 loss
        bets = [
            {'decimal_odds': 2.0, 'won': True},
            {'decimal_odds': 2.0, 'won': True},
            {'decimal_odds': 2.0, 'won': False},
        ]
        
        total_wagered = len(bets)
        total_profit = sum(
            (b['decimal_odds'] - 1) if b['won'] else -1
            for b in bets
        )
        
        roi = total_profit / total_wagered
        
        # 2 wins (each +1.0) + 1 loss (-1.0) = +1.0 profit
        # ROI = 1.0 / 3 = 33.3%
        assert abs(roi - (1.0 / 3)) < 0.001
    
    def test_break_even_roi(self):
        """Test break-even point for -110 odds."""
        # At -110 odds (decimal 1.909), need 52.4% win rate to break even
        decimal_odds = 1.909
        break_even_win_rate = 1 / decimal_odds
        
        assert abs(break_even_win_rate - 0.524) < 0.01
    
    def test_roi_with_vig(self):
        """Test that vig reduces ROI."""
        # Even odds without vig: decimal 2.0
        # Even odds with 5% vig: decimal ~1.91
        
        # 100 bets at 50% win rate
        np.random.seed(42)
        wins = 50
        losses = 50
        
        # Without vig
        profit_no_vig = wins * (2.0 - 1) - losses * 1
        roi_no_vig = profit_no_vig / 100
        
        # With vig (-110 both sides)
        profit_with_vig = wins * (1.909 - 1) - losses * 1
        roi_with_vig = profit_with_vig / 100
        
        # ROI with vig should be worse
        assert roi_with_vig < roi_no_vig


class TestStatisticalSignificance:
    """Test statistical significance calculations."""
    
    def test_binomial_test_significant(self):
        """Test that large deviation is significant."""
        from scipy.stats import binomtest
        
        # 70 wins out of 100 when expected is 50%
        result = binomtest(70, 100, 0.5, alternative='greater')
        
        # Should be highly significant
        assert result.pvalue < 0.01
    
    def test_binomial_test_not_significant(self):
        """Test that small deviation is not significant."""
        from scipy.stats import binomtest
        
        # 52 wins out of 100 when expected is 50%
        result = binomtest(52, 100, 0.5, alternative='greater')
        
        # Should NOT be significant at p<0.05
        assert result.pvalue > 0.05
    
    def test_sample_size_affects_significance(self):
        """Test that larger samples give more significance."""
        from scipy.stats import binomtest
        
        # Same 55% win rate, different sample sizes
        result_small = binomtest(55, 100, 0.5, alternative='greater')
        result_large = binomtest(550, 1000, 0.5, alternative='greater')
        
        # Larger sample should have smaller p-value
        assert result_large.pvalue < result_small.pvalue
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        # ROI of 10% +/- some margin
        mean_roi = 0.10
        std_roi = 0.20
        n_bets = 100
        
        se = std_roi / np.sqrt(n_bets)
        ci_low = mean_roi - 1.96 * se
        ci_high = mean_roi + 1.96 * se
        
        # 95% CI should include true value
        assert ci_low < mean_roi < ci_high
        
        # CI should be narrower with more samples
        se_more = std_roi / np.sqrt(1000)
        ci_low_more = mean_roi - 1.96 * se_more
        ci_high_more = mean_roi + 1.96 * se_more
        
        width_100 = ci_high - ci_low
        width_1000 = ci_high_more - ci_low_more
        
        assert width_1000 < width_100


class TestBettingSimulation:
    """Test betting simulation logic."""
    
    def test_simulate_bet_win(self):
        """Test simulating a winning bet."""
        bet_on_f1 = True
        fighter1_won = True
        decimal_odds = 1.8
        
        if bet_on_f1 == fighter1_won:
            profit = decimal_odds - 1
        else:
            profit = -1
        
        assert profit > 0
        assert abs(profit - 0.8) < 0.001
    
    def test_simulate_bet_loss(self):
        """Test simulating a losing bet."""
        bet_on_f1 = True
        fighter1_won = False
        decimal_odds = 1.8
        
        if bet_on_f1 == fighter1_won:
            profit = decimal_odds - 1
        else:
            profit = -1
        
        assert profit == -1
    
    def test_simulate_bet_underdog(self):
        """Test simulating underdog bet."""
        bet_on_f1 = False  # Bet on fighter 2
        fighter1_won = False  # Fighter 2 won
        decimal_odds = 3.0  # Underdog odds
        
        if bet_on_f1 == fighter1_won:
            profit = decimal_odds - 1
        else:
            profit = -1
        
        # Fighter 2 won, we bet on fighter 2
        # But our condition is checking if we bet on f1 == f1 won
        # Let's fix this logic
        bet_won = (bet_on_f1 and fighter1_won) or (not bet_on_f1 and not fighter1_won)
        
        if bet_won:
            profit = decimal_odds - 1
        else:
            profit = -1
        
        assert profit > 0
        assert abs(profit - 2.0) < 0.001
    
    def test_kelly_criterion_basic(self):
        """Test Kelly criterion for bet sizing."""
        # Kelly formula: f* = (bp - q) / b
        # where b = decimal odds - 1, p = win prob, q = 1 - p
        
        p = 0.55  # 55% win probability
        decimal_odds = 2.0
        b = decimal_odds - 1
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Should bet ~10% of bankroll
        assert kelly_fraction > 0
        assert kelly_fraction < 0.20


class TestEdgeFinder:
    """Test EdgeFinder class methods."""
    
    def test_decimal_to_implied(self):
        """Test decimal odds to implied probability conversion."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder.__new__(EdgeFinder)
        finder.model = None
        
        # 2.0 decimal = 50% implied
        assert abs(finder.decimal_to_implied(2.0) - 0.5) < 0.001
        
        # 1.5 decimal = 66.7% implied
        assert abs(finder.decimal_to_implied(1.5) - 0.667) < 0.01
        
        # 4.0 decimal = 25% implied
        assert abs(finder.decimal_to_implied(4.0) - 0.25) < 0.001
    
    def test_simulate_market_odds_symmetry(self):
        """Test that simulated odds are symmetric for even ELO."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder.__new__(EdgeFinder)
        finder.model = None
        
        odds_a, odds_b = finder.simulate_market_odds(0)  # Even fight
        
        # Both fighters should have similar odds
        assert abs(odds_a - odds_b) < 0.1
    
    def test_simulate_market_odds_favorite(self):
        """Test that favorite gets lower odds."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder.__new__(EdgeFinder)
        finder.model = None
        
        odds_a, odds_b = finder.simulate_market_odds(100)  # A is favored
        
        # Favorite (A) should have lower decimal odds
        assert odds_a < odds_b


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

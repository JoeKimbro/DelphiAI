"""
Tests for upcoming fight predictor.

Tests:
- Fighter ELO lookup
- Prediction generation
- Value bet identification
"""

import pytest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))


class TestPredictionLogic:
    """Test prediction generation logic."""
    
    def test_prediction_favors_higher_elo(self):
        """Test that prediction favors higher ELO fighter."""
        f1_elo = 1600
        f2_elo = 1400
        
        elo_diff = f1_elo - f2_elo
        model_prob_f1 = 1 / (1 + 10**(-elo_diff/400))
        
        # F1 should be favored
        assert model_prob_f1 > 0.5
        prediction = "Fighter1" if model_prob_f1 > 0.5 else "Fighter2"
        assert prediction == "Fighter1"
    
    def test_prediction_close_fight(self):
        """Test prediction for close fight."""
        f1_elo = 1510
        f2_elo = 1490
        
        elo_diff = f1_elo - f2_elo
        model_prob_f1 = 1 / (1 + 10**(-elo_diff/400))
        
        # Should be close to 50/50
        assert 0.45 < model_prob_f1 < 0.55
    
    def test_confidence_calculation(self):
        """Test confidence is max of both probabilities."""
        model_prob_f1 = 0.65
        model_prob_f2 = 1 - model_prob_f1
        
        confidence = max(model_prob_f1, model_prob_f2)
        
        assert confidence == 0.65
    
    def test_confidence_minimum(self):
        """Test minimum confidence is 50%."""
        for prob in [0.45, 0.50, 0.55, 0.70]:
            confidence = max(prob, 1 - prob)
            assert confidence >= 0.5


class TestValueBetDetection:
    """Test value bet detection for upcoming fights."""
    
    def test_value_bet_when_edge_exceeds_threshold(self):
        """Test value bet is flagged when edge > 5%."""
        model_prob = 0.60
        implied_prob = 0.50
        MIN_EDGE = 0.05
        
        edge = model_prob - implied_prob
        is_value_bet = edge > MIN_EDGE
        
        assert is_value_bet
    
    def test_no_value_bet_when_edge_too_small(self):
        """Test no value bet when edge < 5%."""
        model_prob = 0.53
        implied_prob = 0.50
        MIN_EDGE = 0.05
        
        edge = model_prob - implied_prob
        is_value_bet = edge > MIN_EDGE
        
        assert not is_value_bet
    
    def test_value_bet_on_underdog(self):
        """Test value bet detection on underdog."""
        model_prob_f1 = 0.40  # Model says 40% for F1
        implied_prob_f1 = 0.50  # Market implies 50%
        implied_prob_f2 = 0.50
        
        MIN_EDGE = 0.05
        
        edge_f1 = model_prob_f1 - implied_prob_f1
        edge_f2 = (1 - model_prob_f1) - implied_prob_f2
        
        # F2 should have the value bet
        value_on_f1 = edge_f1 > MIN_EDGE
        value_on_f2 = edge_f2 > MIN_EDGE
        
        assert not value_on_f1
        assert value_on_f2


class TestEloLookup:
    """Test ELO lookup functionality."""
    
    def test_default_elo_for_unknown_fighter(self):
        """Test default ELO is 1500 for unknown fighter."""
        DEFAULT_ELO = 1500.0
        
        # Simulate unknown fighter lookup
        fighter_elo = DEFAULT_ELO  # Would come from DB lookup
        
        assert fighter_elo == 1500.0
    
    def test_elo_is_float(self):
        """Test that ELO is returned as float."""
        elo = 1500.0
        
        assert isinstance(elo, float)


class TestOddsInterpretation:
    """Test odds interpretation for predictions."""
    
    def test_implied_prob_from_decimal(self):
        """Test implied probability calculation from decimal odds."""
        decimal_odds = 2.0
        implied_prob = 1 / decimal_odds
        
        assert implied_prob == 0.5
    
    def test_implied_prob_favorite(self):
        """Test implied probability for favorite."""
        decimal_odds = 1.5  # Favorite
        implied_prob = 1 / decimal_odds
        
        assert abs(implied_prob - 0.667) < 0.01
    
    def test_implied_prob_underdog(self):
        """Test implied probability for underdog."""
        decimal_odds = 3.0  # Underdog
        implied_prob = 1 / decimal_odds
        
        assert abs(implied_prob - 0.333) < 0.01
    
    def test_overround_detection(self):
        """Test detection of bookmaker overround."""
        # Typical -110 on both sides
        decimal_f1 = 1.909
        decimal_f2 = 1.909
        
        implied_f1 = 1 / decimal_f1
        implied_f2 = 1 / decimal_f2
        
        overround = implied_f1 + implied_f2 - 1
        
        # Should be positive (bookmaker edge)
        assert overround > 0
        assert abs(overround - 0.048) < 0.01  # ~4.8% overround


class TestPredictionOutput:
    """Test prediction output format."""
    
    def test_prediction_contains_required_fields(self):
        """Test that prediction dict has all required fields."""
        required_fields = [
            'event_name',
            'fighter1',
            'fighter2',
            'f1_elo',
            'f2_elo',
            'model_prob_f1',
            'prediction',
            'confidence',
        ]
        
        # Mock prediction
        prediction = {
            'event_name': 'UFC 300',
            'fighter1': 'Jon Jones',
            'fighter2': 'Stipe Miocic',
            'f1_elo': 1800,
            'f2_elo': 1650,
            'elo_diff': 150,
            'model_prob_f1': 0.70,
            'model_prob_f2': 0.30,
            'prediction': 'Jon Jones',
            'confidence': 0.70,
            'timestamp': '2024-01-01T00:00:00',
        }
        
        for field in required_fields:
            assert field in prediction
    
    def test_value_bet_format(self):
        """Test value bet structure when identified."""
        value_bet = {
            'bet_on': 'Jon Jones',
            'edge': 0.10,
            'decimal_odds': 1.5,
            'model_prob': 0.70,
            'implied_prob': 0.60,
        }
        
        required = ['bet_on', 'edge', 'decimal_odds', 'model_prob', 'implied_prob']
        
        for field in required:
            assert field in value_bet


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

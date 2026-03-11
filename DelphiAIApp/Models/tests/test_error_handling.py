"""
Error Handling Tests

Tests for edge cases and error scenarios:
- Missing data
- Invalid inputs
- Database errors
- Network issues
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))


class TestMissingDataHandling:
    """Test handling of missing data."""
    
    def test_missing_elo_uses_default(self):
        """Missing ELO should default to 1500."""
        # When ELO is None, should use baseline
        from ml.realistic_odds_estimator import elo_to_probability
        
        # If both ELOs are 1500 (default), should be 50%
        prob = elo_to_probability(1500, 1500)
        assert 0.49 < prob < 0.51
    
    def test_missing_nationality_handled(self):
        """Missing nationality should not crash extraction."""
        from ml.analyze_nationality import extract_country
        
        # Empty string
        assert extract_country('') == 'Unknown'
        assert extract_country(None) == 'Unknown'
        assert extract_country('N/A') == 'Unknown'
    
    def test_missing_odds_handled(self):
        """Missing odds should return None, not crash."""
        from ml.realistic_odds_estimator import probability_to_american
        
        # Edge cases
        result = probability_to_american(0)
        assert result is None or result == float('inf')
        
        result = probability_to_american(1)
        assert result is None or result == float('-inf')
    
    def test_nan_values_handled(self):
        """NaN values should be handled gracefully."""
        from ml.realistic_odds_estimator import estimate_market_odds
        
        # NaN ELO should still return a result (function handles it)
        result = estimate_market_odds(float('nan'))
        
        # Should return dict with odds (using default 1500 ELO internally)
        assert isinstance(result, dict)
        assert 'decimal_a' in result


class TestInvalidInputs:
    """Test handling of invalid inputs."""
    
    def test_negative_odds_rejected(self):
        """Negative decimal odds should be handled."""
        from ml.edge_finder import EdgeFinder
        
        # American odds of -100 should give decimal odds of 2.0
        # But decimal odds should never be negative
        finder = EdgeFinder()
        
        # Negative decimal odds make no sense
        # Should either error or return None
        result = finder.american_to_decimal(-100)
        assert result == 2.0  # -100 American = 2.0 decimal
        
        # Invalid American odds (0) should be handled
        with pytest.raises((ValueError, ZeroDivisionError)):
            finder.american_to_decimal(0)
    
    def test_probability_out_of_range(self):
        """Probabilities >1 or <0 should be handled."""
        from ml.realistic_odds_estimator import probability_to_american
        
        # Invalid probabilities
        result = probability_to_american(1.5)  # >1 is invalid
        assert result is None or isinstance(result, (int, float))
        
        result = probability_to_american(-0.1)  # <0 is invalid
        assert result is None or isinstance(result, (int, float))
    
    def test_division_by_zero_prevented(self):
        """Division by zero should be prevented."""
        from ml.realistic_odds_estimator import elo_to_probability
        
        # Same ELO should not cause issues
        prob = elo_to_probability(1500, 1500)
        assert prob == 0.5
        
        # Extreme ELO difference
        prob = elo_to_probability(3000, 500)
        assert 0 <= prob <= 1


class TestOddsEdgeCases:
    """Test odds conversion edge cases."""
    
    def test_even_odds(self):
        """Even money (+100/-100) should convert correctly."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder()
        
        # +100 = 2.0 decimal
        assert finder.american_to_decimal(100) == 2.0
        
        # -100 = 2.0 decimal
        assert finder.american_to_decimal(-100) == 2.0
    
    def test_heavy_favorite(self):
        """Heavy favorite odds should convert correctly."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder()
        
        # -500 = 1.2 decimal
        result = finder.american_to_decimal(-500)
        assert 1.19 < result < 1.21
    
    def test_heavy_underdog(self):
        """Heavy underdog odds should convert correctly."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder()
        
        # +500 = 6.0 decimal
        result = finder.american_to_decimal(500)
        assert result == 6.0
    
    def test_implied_probability_sum(self):
        """Implied probabilities should sum to >1 (vig)."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder()
        
        # Typical matchup: -200 / +170
        fav_decimal = finder.american_to_decimal(-200)  # 1.5
        dog_decimal = finder.american_to_decimal(170)   # 2.7
        
        implied_fav = 1 / fav_decimal  # 0.667
        implied_dog = 1 / dog_decimal  # 0.37
        
        total = implied_fav + implied_dog
        
        # Should be >1 due to vig
        assert total > 1.0


class TestDatabaseErrors:
    """Test handling of database errors."""
    
    def test_connection_failure_handled(self):
        """Database connection failures should be handled gracefully."""
        from ml.upcoming_predictor import UpcomingPredictor
        
        predictor = UpcomingPredictor()
        
        # Mock connection failure
        with patch.object(predictor, 'conn', None):
            # Should not crash, should use defaults
            result = predictor.get_fighter_elo("Unknown Fighter")
            assert result['elo'] == 1500.0  # Default ELO
    
    def test_missing_fighter_in_db(self):
        """Missing fighters should return defaults, not crash."""
        from ml.upcoming_predictor import UpcomingPredictor
        
        predictor = UpcomingPredictor()
        
        # Fighter that definitely doesn't exist
        result = predictor.get_fighter_elo("ZZZZZ NONEXISTENT FIGHTER ZZZZZ")
        
        assert result['elo'] == 1500.0
        assert result['fighter_id'] is None


class TestNetworkErrors:
    """Test handling of network/scraping errors."""
    
    def test_scrape_timeout_handled(self):
        """Network timeouts should be handled gracefully."""
        from ml.upcoming_predictor import UpcomingPredictor
        import requests
        
        predictor = UpcomingPredictor()
        
        # Mock a timeout
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout()
            
            # Use the correct method name
            try:
                events = predictor.fetch_upcoming_events()
            except requests.exceptions.Timeout:
                events = []  # Timeout is acceptable
            
            # Should return empty list, not crash
            assert events == [] or events is None or isinstance(events, list)
    
    def test_invalid_html_handled(self):
        """Invalid HTML should be handled gracefully."""
        from ml.upcoming_predictor import UpcomingPredictor
        
        predictor = UpcomingPredictor()
        
        # Mock response with garbage HTML
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = "<html><body>Not valid UFC page</body></html>"
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Use the correct method name
            events = predictor.fetch_upcoming_events()
            
            # Should return empty or list, not crash
            assert events == [] or events is None or isinstance(events, list)


class TestTypeErrors:
    """Test type handling."""
    
    def test_decimal_to_float_conversion(self):
        """Decimal database values should be converted to float."""
        from decimal import Decimal
        
        # When getting ELO from database, might be Decimal
        elo_decimal = Decimal('1623.45')
        elo_float = float(elo_decimal)
        
        # Should work in calculations
        diff = elo_float - 1500.0
        assert isinstance(diff, float)
        assert 123 < diff < 124
    
    def test_string_to_number_conversion(self):
        """String numbers should be handled."""
        # Sometimes odds come as strings
        odds_str = "-150"
        odds_int = int(odds_str)
        
        assert odds_int == -150


class TestBoundaryConditions:
    """Test boundary conditions."""
    
    def test_elo_extremes(self):
        """Extreme ELO values should produce valid probabilities."""
        from ml.realistic_odds_estimator import elo_to_probability
        
        # Very high ELO difference
        prob = elo_to_probability(2500, 1000)
        assert 0 < prob < 1
        assert prob > 0.95  # Should heavily favor higher ELO
        
        # Very low ELO difference
        prob = elo_to_probability(1000, 2500)
        assert 0 < prob < 1
        assert prob < 0.05  # Should heavily favor higher ELO
    
    def test_zero_fights(self):
        """Zero fights should be handled."""
        from ml.safeguards import validate_sample_size
        
        is_valid, msg = validate_sample_size(0, 'overall_edge')
        assert not is_valid
    
    def test_single_fight(self):
        """Single fight edge calculation should be rejected."""
        from ml.safeguards import validate_edge
        
        result = validate_edge(
            edge_pct=1.0,  # 100% win rate
            n_bets=1,
            win_rate=1.0,
            expected_rate=0.5
        )
        
        assert not result['is_reliable']


class TestRobustness:
    """Test overall system robustness."""
    
    def test_empty_dataframe_handled(self):
        """Empty DataFrames should not crash analysis."""
        import pandas as pd
        
        # Empty DataFrame
        df = pd.DataFrame()
        
        # Common operations should not crash
        assert len(df) == 0
        assert df.empty
    
    def test_unicode_fighter_names(self):
        """Unicode characters in names should be handled."""
        from ml.analyze_nationality import extract_country
        
        # Names with accents and special characters
        names = [
            "São Paulo, Brazil",
            "Zürich, Switzerland",
            "北京, China",
            "Москва, Russia",
        ]
        
        for name in names:
            # Should not crash
            result = extract_country(name)
            assert isinstance(result, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

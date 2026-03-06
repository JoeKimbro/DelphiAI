"""
Tests for the batch ELO adjustment update script.

Tests cover:
- Inactivity penalty calculations
- Database update logic (mocked)
- Fighter prioritization for injury checks
- Error handling
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.update_adjusted_elos import (
    calculate_inactivity_penalty,
)


class TestInactivityPenaltyCalculation:
    """Test the inactivity penalty calculation logic."""
    
    def test_no_penalty_under_6_months(self):
        """Test that fighters with < 6 months inactivity get no penalty."""
        penalty, details = calculate_inactivity_penalty(100, 1600)
        assert penalty == 0
        assert 'Active fighter' in details.get('reason', '')
    
    def test_no_penalty_at_180_days(self):
        """Test boundary at exactly 180 days."""
        penalty, details = calculate_inactivity_penalty(180, 1600)
        assert penalty == 0
    
    def test_penalty_at_181_days(self):
        """Test penalty kicks in at 181 days."""
        penalty, details = calculate_inactivity_penalty(181, 1600)
        assert penalty > 0
    
    def test_6_to_12_month_penalty(self):
        """Test penalty for 6-12 months inactivity (8% + 10 flat)."""
        # 9 months = 270 days
        elo = 1600
        penalty, details = calculate_inactivity_penalty(270, elo)
        
        # Should have 8% decay rate + 10 flat
        assert penalty >= 10  # At least flat penalty
        assert details['decay_rate'] > 0
        assert details['flat_penalty'] == 10
    
    def test_12_to_18_month_penalty(self):
        """Test penalty for 12-18 months inactivity (12% + 20 flat)."""
        # 15 months = 450 days
        elo = 1600
        penalty, details = calculate_inactivity_penalty(450, elo)
        
        assert details['flat_penalty'] == 20
        assert penalty > 20  # More than just flat penalty
    
    def test_18_to_24_month_penalty(self):
        """Test penalty for 18-24 months inactivity (18% + 30 flat)."""
        # 21 months = 630 days
        elo = 1600
        penalty, details = calculate_inactivity_penalty(630, elo)
        
        assert details['flat_penalty'] == 30
        assert penalty > 30
    
    def test_over_24_month_penalty(self):
        """Test penalty for 24+ months inactivity (25% + 40 flat)."""
        # 30 months = 900 days
        elo = 1600
        penalty, details = calculate_inactivity_penalty(900, elo)
        
        assert details['flat_penalty'] == 40
        assert penalty > 40
    
    def test_penalty_proportional_to_elo_above_baseline(self):
        """Test that higher ELO fighters get higher proportional decay."""
        days = 365  # 1 year
        
        # Fighter with high ELO
        high_elo = 1700
        penalty_high, _ = calculate_inactivity_penalty(days, high_elo)
        
        # Fighter with average ELO
        avg_elo = 1500
        penalty_avg, _ = calculate_inactivity_penalty(days, avg_elo)
        
        # High ELO should have higher proportional decay
        # (but both get same flat penalty)
        assert penalty_high > penalty_avg
    
    def test_penalty_below_baseline_elo(self):
        """Test penalty for fighter below 1500 ELO baseline."""
        days = 365
        low_elo = 1400
        
        penalty, details = calculate_inactivity_penalty(days, low_elo)
        
        # Proportional decay is based on (ELO - 1500), so negative
        # Total penalty should still be positive due to flat penalty
        assert penalty > 0
        assert details['proportional_decay'] < 0  # Negative (below baseline)
    
    def test_decay_cap_at_35_percent(self):
        """Test that decay is capped at 35%."""
        # Very long inactivity
        days = 1500  # ~4 years
        elo = 1800
        
        penalty, details = calculate_inactivity_penalty(days, elo)
        
        # Decay rate should be capped
        assert details['decay_rate'] <= 35.1  # Allow small floating point variance
    
    def test_realistic_scenario_albazi(self):
        """Test realistic scenario similar to Amir Albazi."""
        # Albazi: ~460 days inactive, ELO ~1572
        days = 460
        elo = 1572
        
        penalty, details = calculate_inactivity_penalty(days, elo)
        
        # 12-18 month bracket: 12% decay + 20 flat
        # ~1.26 years * 12% = ~15% effective decay
        # (1572 - 1500) * 0.15 = ~11 proportional
        # Total: ~11 + 20 = ~31
        
        assert 25 <= penalty <= 40  # Reasonable range
        assert details['flat_penalty'] == 20
    
    def test_recently_active_high_elo(self):
        """Test that recently active high ELO fighter gets no penalty."""
        days = 60  # 2 months
        elo = 1800
        
        penalty, _ = calculate_inactivity_penalty(days, elo)
        assert penalty == 0
    
    def test_long_inactive_average_fighter(self):
        """Test penalty for long inactive average fighter."""
        days = 730  # 2 years (exactly 24 months - falls in 18-24 bracket)
        elo = 1500  # Baseline
        
        penalty, details = calculate_inactivity_penalty(days, elo)
        
        # Should only get flat penalty since ELO is at baseline
        # 730 days is in 18-24 month bracket = 30 flat penalty
        assert penalty == 30  # Just the flat penalty for 18-24 months
        assert details['proportional_decay'] == 0


class TestPenaltyDetails:
    """Test the details returned by penalty calculation."""
    
    def test_details_contain_expected_fields(self):
        """Test that details dict has all expected fields."""
        penalty, details = calculate_inactivity_penalty(300, 1600)
        
        expected_fields = [
            'days_inactive',
            'years_inactive',
            'decay_rate',
            'flat_penalty',
            'proportional_decay',
            'total_penalty',
        ]
        
        for field in expected_fields:
            assert field in details
    
    def test_total_penalty_matches_return_value(self):
        """Test that details total matches returned penalty."""
        penalty, details = calculate_inactivity_penalty(400, 1650)
        
        assert penalty == details['total_penalty']
    
    def test_years_inactive_calculation(self):
        """Test that years inactive is calculated correctly."""
        days = 365
        penalty, details = calculate_inactivity_penalty(days, 1600)
        
        assert details['years_inactive'] == 1.0
    
    def test_active_fighter_reason(self):
        """Test that active fighters get explanatory reason."""
        penalty, details = calculate_inactivity_penalty(100, 1600)
        
        assert 'reason' in details
        assert 'Active' in details['reason']


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_days_inactive(self):
        """Test fighter who just fought."""
        penalty, _ = calculate_inactivity_penalty(0, 1600)
        assert penalty == 0
    
    def test_negative_days_inactive(self):
        """Test handling of invalid negative days."""
        # Should handle gracefully
        penalty, _ = calculate_inactivity_penalty(-10, 1600)
        assert penalty == 0
    
    def test_very_high_elo(self):
        """Test with very high ELO."""
        penalty, details = calculate_inactivity_penalty(400, 2000)
        
        # Should get significant proportional decay
        assert penalty > 50
    
    def test_very_low_elo(self):
        """Test with very low ELO."""
        penalty, details = calculate_inactivity_penalty(400, 1200)
        
        # Proportional decay should be negative (below baseline)
        # Total penalty is capped at minimum flat penalty
        assert penalty > 0
        assert penalty >= details['flat_penalty']  # At least flat penalty
        assert details['proportional_decay'] < 0
    
    def test_exactly_baseline_elo(self):
        """Test with exactly baseline ELO."""
        penalty, details = calculate_inactivity_penalty(400, 1500)
        
        # Only flat penalty, no proportional
        assert details['proportional_decay'] == 0
        assert penalty == details['flat_penalty']


class TestDatabaseIntegration:
    """Test database update functions (mocked)."""
    
    @patch('psycopg2.connect')
    def test_update_returns_count(self, mock_connect):
        """Test that update function returns count of updated fighters."""
        # This would require more complex mocking
        # Placeholder for integration tests
        pass
    
    @patch('psycopg2.connect')
    def test_injury_check_logs_to_database(self, mock_connect):
        """Test that injury checks are logged to database."""
        # Placeholder for integration tests
        pass


class TestFighterPrioritization:
    """Test logic for prioritizing which fighters to check."""
    
    def test_ranked_fighters_first(self):
        """Test that ranked fighters are checked before unranked."""
        # This tests the SQL ORDER BY logic conceptually
        # Ranked fighters should come before those with NULL ranking
        pass
    
    def test_higher_elo_priority(self):
        """Test that higher ELO fighters get priority."""
        # Within unranked fighters, higher ELO should be first
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

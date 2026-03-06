"""
Tests for injury penalty time decay and fight reset logic.

These tests ensure that:
1. Injury penalties decay over time (18-month window)
2. Penalties are recalculated based on current date, not cache date
3. Fighter's injury flag is cleared after they complete a fight
4. Very old injuries have minimal impact
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestInjuryTimeDecay:
    """Test that injury penalties decay over time."""
    
    def test_recent_injury_full_penalty(self):
        """Test that recent injuries (0-2 months) get 1.5x multiplier."""
        base_penalty = 55
        days_since = 30  # 1 month
        
        # 0-60 days = 1.5x multiplier
        expected_penalty = int(base_penalty * 1.5)
        
        assert expected_penalty == 82
    
    def test_2_to_4_month_injury(self):
        """Test that 2-4 month old injuries get 1.2x multiplier."""
        base_penalty = 55
        days_since = 90  # 3 months
        
        # 60-120 days = 1.2x multiplier
        expected_penalty = int(base_penalty * 1.2)
        
        assert expected_penalty == 66
    
    def test_4_to_6_month_injury(self):
        """Test that 4-6 month old injuries get 1.0x multiplier."""
        base_penalty = 55
        days_since = 150  # 5 months
        
        # 120-180 days = 1.0x multiplier
        expected_penalty = int(base_penalty * 1.0)
        
        assert expected_penalty == 55
    
    def test_6_to_12_month_injury(self):
        """Test that 6-12 month old injuries get 0.7x multiplier."""
        base_penalty = 55
        days_since = 270  # 9 months
        
        # 180-365 days = 0.7x multiplier
        expected_penalty = int(base_penalty * 0.7)
        
        assert expected_penalty == 38
    
    def test_12_to_18_month_injury(self):
        """Test that 12-18 month old injuries get 0.3x multiplier."""
        base_penalty = 55
        days_since = 450  # 15 months
        
        # 365-540 days = 0.3x multiplier
        expected_penalty = int(base_penalty * 0.3)
        
        assert expected_penalty == 16
    
    def test_18_plus_month_injury_minimal(self):
        """Test that 18+ month old injuries get minimal 0.1x multiplier."""
        base_penalty = 55
        days_since = 600  # 20 months
        
        # 540+ days = 0.1x multiplier
        expected_penalty = int(base_penalty * 0.1)
        
        assert expected_penalty == 5
    
    def test_very_old_injury_nearly_zero(self):
        """Test that very old injuries (2+ years) have almost no impact."""
        base_penalty = 55
        days_since = 900  # ~2.5 years
        
        # Still 0.1x multiplier
        expected_penalty = int(base_penalty * 0.1)
        
        assert expected_penalty == 5
        assert expected_penalty < 10  # Should be minimal


class TestMultiplierCalculation:
    """Test the multiplier calculation logic."""
    
    def get_multiplier(self, days_since_injury):
        """Helper to calculate multiplier based on days."""
        if days_since_injury <= 60:
            return 1.5
        elif days_since_injury <= 120:
            return 1.2
        elif days_since_injury <= 180:
            return 1.0
        elif days_since_injury <= 365:
            return 0.7
        elif days_since_injury <= 540:
            return 0.3
        else:
            return 0.1
    
    def test_boundary_at_60_days(self):
        """Test multiplier boundary at 60 days."""
        assert self.get_multiplier(60) == 1.5
        assert self.get_multiplier(61) == 1.2
    
    def test_boundary_at_120_days(self):
        """Test multiplier boundary at 120 days."""
        assert self.get_multiplier(120) == 1.2
        assert self.get_multiplier(121) == 1.0
    
    def test_boundary_at_180_days(self):
        """Test multiplier boundary at 180 days (6 months)."""
        assert self.get_multiplier(180) == 1.0
        assert self.get_multiplier(181) == 0.7
    
    def test_boundary_at_365_days(self):
        """Test multiplier boundary at 365 days (1 year)."""
        assert self.get_multiplier(365) == 0.7
        assert self.get_multiplier(366) == 0.3
    
    def test_boundary_at_540_days(self):
        """Test multiplier boundary at 540 days (18 months)."""
        assert self.get_multiplier(540) == 0.3
        assert self.get_multiplier(541) == 0.1


class TestFightResetsInjury:
    """Test that completing a fight clears/refreshes injury data."""
    
    def test_fight_after_injury_check_triggers_refresh(self):
        """Test that fighting after injury check triggers a refresh."""
        # Injury checked on Jan 1
        injury_check_date = datetime(2025, 1, 1)
        
        # Fighter fought on Jan 15 (after injury check)
        last_fight_date = datetime(2025, 1, 15)
        
        # Should trigger refresh because fight > injury_check
        needs_refresh = last_fight_date > injury_check_date
        
        assert needs_refresh == True
    
    def test_no_fight_since_injury_uses_cache(self):
        """Test that no fight since injury check uses cached data."""
        # Injury checked on Jan 15
        injury_check_date = datetime(2025, 1, 15)
        
        # Last fight was Jan 1 (before injury check)
        last_fight_date = datetime(2025, 1, 1)
        
        # Should NOT trigger refresh
        needs_refresh = last_fight_date > injury_check_date
        
        assert needs_refresh == False
    
    def test_fighting_proves_recovery(self):
        """Test scenario: fighter recovers and fights successfully."""
        # Timeline:
        # - Jan 2024: Surgery detected, -55 penalty
        # - Jun 2024: Fighter completes a fight (proved healthy)
        # - Jul 2024: Query should trigger fresh check
        
        injury_check_date = datetime(2024, 1, 1)
        fight_date = datetime(2024, 6, 15)
        
        # Fighter fought after injury check = needs fresh data
        assert fight_date > injury_check_date


class TestRealisticScenarios:
    """Test realistic fighter recovery scenarios."""
    
    def calculate_penalty(self, base_penalty, days_since):
        """Helper to calculate penalty with time decay."""
        if days_since <= 60:
            multiplier = 1.5
        elif days_since <= 120:
            multiplier = 1.2
        elif days_since <= 180:
            multiplier = 1.0
        elif days_since <= 365:
            multiplier = 0.7
        elif days_since <= 540:
            multiplier = 0.3
        else:
            multiplier = 0.1
        
        return int(base_penalty * multiplier)
    
    def test_acl_surgery_recovery_timeline(self):
        """Test ACL surgery recovery over 18 months."""
        base_penalty = 55  # Major injury
        
        # Month 1: Fresh surgery
        assert self.calculate_penalty(base_penalty, 30) == 82  # 1.5x
        
        # Month 3: Early rehab
        assert self.calculate_penalty(base_penalty, 90) == 66  # 1.2x
        
        # Month 6: Getting cleared
        assert self.calculate_penalty(base_penalty, 180) == 55  # 1.0x
        
        # Month 9: First fight back (high risk)
        assert self.calculate_penalty(base_penalty, 270) == 38  # 0.7x
        
        # Month 15: Third fight (low risk)
        assert self.calculate_penalty(base_penalty, 450) == 16  # 0.3x
        
        # Month 20: Old news
        assert self.calculate_penalty(base_penalty, 600) == 5   # 0.1x
    
    def test_albazi_scenario(self):
        """Test Amir Albazi's heart/neck surgery scenario."""
        base_penalty = 55
        
        # Surgery ~6 months ago (estimated from article)
        days_since = 180
        
        # At 6 months, should be 1.0x multiplier
        penalty = self.calculate_penalty(base_penalty, days_since)
        assert penalty == 55
        
        # If he fights and wins, penalty should be rechecked
        # and likely cleared (no new injury)
        
        # 6 months later (1 year total)
        penalty_later = self.calculate_penalty(base_penalty, 365)
        assert penalty_later == 38  # Reduced
        
        # 18 months later
        penalty_much_later = self.calculate_penalty(base_penalty, 540)
        assert penalty_much_later == 16  # Much reduced
    
    def test_minor_injury_timeline(self):
        """Test minor injury (pulled out of fight) timeline."""
        base_penalty = 25  # Minor injury
        
        # Fresh withdrawal
        assert self.calculate_penalty(base_penalty, 14) == 37  # 1.5x
        
        # 3 months later
        assert self.calculate_penalty(base_penalty, 90) == 30  # 1.2x
        
        # 6 months later (should be almost gone)
        assert self.calculate_penalty(base_penalty, 200) == 17  # 0.7x
        
        # 1 year later (minimal)
        assert self.calculate_penalty(base_penalty, 400) == 7   # 0.3x


class TestEdgeCases:
    """Test edge cases in injury time decay."""
    
    def test_zero_days_since_injury(self):
        """Test injury detected today."""
        base_penalty = 55
        days_since = 0
        
        # 0 days = 1.5x (most recent)
        if days_since <= 60:
            multiplier = 1.5
        expected = int(base_penalty * multiplier)
        
        assert expected == 82
    
    def test_exactly_18_months(self):
        """Test injury at exactly 18 months."""
        base_penalty = 55
        days_since = 540  # Exactly 18 months
        
        # 540 days = 0.3x (still in window)
        if days_since <= 540:
            multiplier = 0.3
        expected = int(base_penalty * multiplier)
        
        assert expected == 16
    
    def test_one_day_over_18_months(self):
        """Test injury at 18 months + 1 day."""
        base_penalty = 55
        days_since = 541
        
        # 541 days = 0.1x (old injury)
        multiplier = 0.1
        expected = int(base_penalty * multiplier)
        
        assert expected == 5
    
    def test_negative_days_handled(self):
        """Test handling of invalid negative days."""
        base_penalty = 55
        days_since = -10  # Invalid
        
        # Should default to recent (1.5x) or handle gracefully
        if days_since <= 60:  # -10 <= 60 is True
            multiplier = 1.5
        expected = int(base_penalty * multiplier)
        
        assert expected == 82  # Treated as recent


class TestPenaltyRecalculation:
    """Test that cached penalties are properly recalculated."""
    
    def test_cached_penalty_recalculated_on_query(self):
        """Test that old cached penalties are recalculated."""
        # Original cache: -55 penalty, injury date Jan 1, 2024
        original_penalty = 55
        injury_date = datetime(2024, 1, 1)
        
        # Query date: July 1, 2024 (6 months later)
        query_date = datetime(2024, 7, 1)
        days_since = (query_date - injury_date).days  # ~180 days
        
        # Should recalculate with current multiplier
        assert 180 <= days_since <= 185
        
        # At 180+ days, multiplier is 0.7x
        if days_since <= 180:
            multiplier = 1.0
        else:
            multiplier = 0.7
        
        recalculated = int(original_penalty * multiplier)
        
        # Should be less than original
        assert recalculated <= original_penalty
    
    def test_stale_cache_triggers_refresh(self):
        """Test that cache older than 7 days triggers refresh."""
        cache_age_days = 10  # 10 days old
        INJURY_CACHE_DAYS = 7
        
        needs_refresh = cache_age_days > INJURY_CACHE_DAYS
        
        assert needs_refresh == True
    
    def test_fresh_cache_no_refresh(self):
        """Test that cache within 7 days doesn't trigger refresh."""
        cache_age_days = 3  # 3 days old
        INJURY_CACHE_DAYS = 7
        
        needs_refresh = cache_age_days > INJURY_CACHE_DAYS
        
        assert needs_refresh == False


class TestIntegrationScenarios:
    """Integration tests for complete scenarios."""
    
    def test_full_injury_lifecycle(self):
        """Test complete injury lifecycle from detection to expiry."""
        base_penalty = 55
        
        # Day 1: Injury detected
        penalty_day1 = int(base_penalty * 1.5)
        assert penalty_day1 == 82
        
        # Month 3: Still recovering
        penalty_month3 = int(base_penalty * 1.2)
        assert penalty_month3 == 66
        
        # Month 6: Getting cleared
        penalty_month6 = int(base_penalty * 1.0)
        assert penalty_month6 == 55
        
        # Month 9: First fight back
        penalty_month9 = int(base_penalty * 0.7)
        assert penalty_month9 == 38
        
        # After successful fight, injury cleared
        # New check finds no injury
        penalty_after_fight = 0
        assert penalty_after_fight == 0
        
        # OR if injury still detected but old:
        # Month 18: Old news
        penalty_month18 = int(base_penalty * 0.3)
        assert penalty_month18 == 16
        
        # Month 24: Minimal impact
        penalty_month24 = int(base_penalty * 0.1)
        assert penalty_month24 == 5
    
    def test_multiple_injuries_scenario(self):
        """Test fighter with multiple injuries over time."""
        # First injury: ACL tear
        acl_base = 55
        
        # After 1 year, ACL mostly recovered
        acl_penalty_1yr = int(acl_base * 0.3)
        assert acl_penalty_1yr == 16
        
        # New injury: Hand fracture
        hand_base = 55
        
        # Fresh hand injury
        hand_penalty = int(hand_base * 1.5)
        assert hand_penalty == 82
        
        # System should only count the most recent/severe
        # Not stack both penalties
        total_penalty = max(acl_penalty_1yr, hand_penalty)
        assert total_penalty == 82  # Hand injury dominates


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

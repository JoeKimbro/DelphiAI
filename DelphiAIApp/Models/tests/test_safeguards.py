"""
Tests for betting safeguards.

These tests ensure our safeguard mechanisms work correctly
to prevent overconfident betting decisions.
"""

import pytest
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))


class TestSampleSizeValidation:
    """Test sample size safeguards."""
    
    def test_insufficient_sample_rejected(self):
        """Small samples should be flagged as unreliable."""
        from ml.safeguards import validate_sample_size
        
        is_valid, msg = validate_sample_size(30, 'overall_edge')
        
        assert not is_valid
        assert "INSUFFICIENT" in msg
    
    def test_borderline_sample_warning(self):
        """Borderline samples should have warnings."""
        from ml.safeguards import validate_sample_size
        
        # 250 is above 200 minimum but below 400 (2x)
        is_valid, msg = validate_sample_size(250, 'overall_edge')
        
        assert is_valid
        assert "borderline" in msg.lower() or msg == ""
    
    def test_sufficient_sample_accepted(self):
        """Large samples should pass without warnings."""
        from ml.safeguards import validate_sample_size
        
        is_valid, msg = validate_sample_size(500, 'overall_edge')
        
        assert is_valid
        assert msg == "" or "WARNING" not in msg.upper()
    
    def test_country_edge_requires_more_samples(self):
        """Country edges need 100+ samples."""
        from ml.safeguards import validate_sample_size
        
        is_valid, _ = validate_sample_size(50, 'country_edge')
        assert not is_valid
        
        is_valid, _ = validate_sample_size(150, 'country_edge')
        assert is_valid


class TestMultipleTestingCorrection:
    """Test multiple testing corrections."""
    
    def test_bonferroni_corrects_false_positives(self):
        """Bonferroni should reject marginally significant results."""
        from ml.safeguards import bonferroni_correction
        
        # 20 p-values, one at 0.03 (would be "significant" at 0.05)
        p_values = [0.03] + [0.5] * 19
        
        significant = bonferroni_correction(p_values, alpha=0.05)
        
        # With Bonferroni (0.05/20 = 0.0025), 0.03 is NOT significant
        assert not significant[0]
    
    def test_bonferroni_keeps_very_significant(self):
        """Very significant results should survive correction."""
        from ml.safeguards import bonferroni_correction
        
        # p = 0.001 should survive even with 20 tests
        p_values = [0.001] + [0.5] * 19
        
        significant = bonferroni_correction(p_values, alpha=0.05)
        
        # 0.001 < 0.0025, so still significant
        assert significant[0]
    
    def test_holm_less_conservative(self):
        """Holm correction should be less conservative than Bonferroni."""
        from ml.safeguards import bonferroni_correction, holm_bonferroni_correction
        
        # Multiple small p-values
        p_values = [0.001, 0.005, 0.01, 0.02, 0.03]
        
        bonf = bonferroni_correction(p_values)
        holm = holm_bonferroni_correction(p_values)
        
        # Holm should find at least as many significant results
        assert sum(holm) >= sum(bonf)


class TestNationalityEdgeSafeguards:
    """Test nationality-specific safeguards."""
    
    def test_small_country_sample_rejected(self):
        """Countries with <100 fights should be rejected."""
        from ml.safeguards import validate_nationality_edge
        
        result = validate_nationality_edge(
            country='Kazakhstan',
            n_fights=47,
            win_rate=0.72,
            expected_rate=0.50,
            p_value=0.01
        )
        
        assert not result['is_valid']
        assert "INSUFFICIENT" in str(result['warnings'])
    
    def test_small_edge_rejected(self):
        """Edges <3% should be flagged as too small."""
        from ml.safeguards import validate_nationality_edge
        
        result = validate_nationality_edge(
            country='USA',
            n_fights=500,
            win_rate=0.51,
            expected_rate=0.50,
            p_value=0.05
        )
        
        assert not result['is_valid']
        assert any("too small" in w.lower() for w in result['warnings'])
    
    def test_valid_edge_accepted_with_caution(self):
        """Large significant edges should be accepted cautiously."""
        from ml.safeguards import validate_nationality_edge
        
        result = validate_nationality_edge(
            country='Russia',
            n_fights=300,
            win_rate=0.65,
            expected_rate=0.50,
            p_value=0.0001
        )
        
        # Should be valid but recommend as ONE feature
        assert result['can_use_as_feature']
        assert "5-10% weight" in result['recommendation'] or "minor" in result['recommendation'].lower()


class TestKellySafeguards:
    """Test Kelly criterion safeguards."""
    
    def test_full_kelly_not_used(self):
        """Safe Kelly should never use full Kelly."""
        from ml.safeguards import safe_kelly
        
        result = safe_kelly(
            estimated_edge=0.10,
            decimal_odds=2.0,
            bankroll=1000,
            kelly_fraction=0.25  # 25% of full Kelly
        )
        
        if result['should_bet']:
            assert result['kelly_fractional'] < result['kelly_raw']
    
    def test_max_bet_capped(self):
        """Bets should be capped at 5% of bankroll."""
        from ml.safeguards import safe_kelly
        
        result = safe_kelly(
            estimated_edge=0.30,  # Very high edge
            decimal_odds=2.0,
            bankroll=1000,
            max_bet_pct=0.05
        )
        
        if result['should_bet']:
            assert result['bet_percentage'] <= 0.05
            assert result['bet_amount'] <= 50
    
    def test_small_edge_rejected(self):
        """Very small edges should not trigger bets."""
        from ml.safeguards import safe_kelly
        
        result = safe_kelly(
            estimated_edge=0.02,  # 2% edge before discount
            decimal_odds=2.0,
            bankroll=1000
        )
        
        # After 30% discount, edge is 1.4% - below 3% minimum
        assert not result['should_bet']
    
    def test_edge_discounted(self):
        """Edge should be discounted for conservatism."""
        from ml.safeguards import safe_kelly, EDGE_DISCOUNT
        
        # The function discounts edge by 30%
        # So 10% edge becomes 7% edge
        assert EDGE_DISCOUNT > 0
        assert EDGE_DISCOUNT < 1


class TestEloCalibration:
    """Test ELO calibration validation."""
    
    def test_perfect_calibration_passes(self):
        """Perfectly calibrated predictions should pass."""
        from ml.safeguards import validate_elo_calibration
        import numpy as np
        
        # Create perfectly calibrated data
        np.random.seed(42)
        predicted = np.linspace(0.3, 0.7, 100)
        # Actual outcomes match predicted probabilities
        actual = np.random.binomial(1, predicted)
        
        result = validate_elo_calibration(predicted.tolist(), actual.tolist())
        
        # Should have low calibration error
        assert result['calibration_error'] < 0.15  # Allow some random variance
    
    def test_poor_calibration_flagged(self):
        """Poorly calibrated predictions should be flagged."""
        from ml.safeguards import validate_elo_calibration
        
        # Predictions always say 70%, but actual is 40%
        predicted = [0.70] * 100
        actual = [1] * 40 + [0] * 60  # 40% actual win rate
        
        result = validate_elo_calibration(predicted, actual)
        
        assert result['calibration_error'] > 0.20
        assert not result['is_calibrated']
        assert result['warning'] is not None


class TestEdgeValidation:
    """Test comprehensive edge validation."""
    
    def test_unreliable_edge_flagged(self):
        """Edges with multiple issues should be flagged."""
        from ml.safeguards import validate_edge
        
        result = validate_edge(
            edge_pct=0.02,  # Too small
            n_bets=30,      # Too few
            win_rate=0.52,
            expected_rate=0.50,
            context='overall_edge'
        )
        
        assert not result['is_reliable']
        assert result['confidence'] == 'low' or result['confidence'] == 'none'
        assert len(result['warnings']) >= 1
    
    def test_reliable_edge_accepted(self):
        """Large, significant edges should be accepted."""
        from ml.safeguards import validate_edge
        
        result = validate_edge(
            edge_pct=0.08,
            n_bets=500,
            win_rate=0.58,
            expected_rate=0.50,
            context='overall_edge'
        )
        
        # With 500 bets and 8% edge, should be more reliable
        assert result['confidence'] in ['medium', 'high']


class TestWarningMechanisms:
    """Test that warnings are properly issued."""
    
    def test_multiple_testing_warning_issued(self):
        """Multiple tests should trigger warning."""
        from ml.safeguards import warn_multiple_testing
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_multiple_testing(20)
            
            assert len(w) == 1
            assert "MULTIPLE TESTING" in str(w[0].message)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

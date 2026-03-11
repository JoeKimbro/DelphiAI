"""
Tests for odds conversion and estimation functions.

Tests:
- American to decimal odds conversion
- Probability to odds conversion
- ELO to probability conversion
- Market odds estimation with vig
"""

import pytest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))


class TestOddsConversion:
    """Test odds conversion functions."""
    
    def test_american_to_decimal_favorite(self):
        """Test converting favorite American odds to decimal."""
        # -200 means bet $200 to win $100, so decimal = 1.5
        american = -200
        expected_decimal = 1.5
        
        # Import from scrape_odds
        from data.scrapers.scrape_odds import BestFightOddsScraper
        scraper = BestFightOddsScraper()
        
        result = scraper.parse_american_odds(str(american))
        assert result is not None
        assert abs(result - expected_decimal) < 0.01
    
    def test_american_to_decimal_underdog(self):
        """Test converting underdog American odds to decimal."""
        # +200 means bet $100 to win $200, so decimal = 3.0
        american = "+200"
        expected_decimal = 3.0
        
        from data.scrapers.scrape_odds import BestFightOddsScraper
        scraper = BestFightOddsScraper()
        
        result = scraper.parse_american_odds(american)
        assert result is not None
        assert abs(result - expected_decimal) < 0.01
    
    def test_american_to_decimal_even_money(self):
        """Test converting even money odds."""
        # -100 or +100 = 2.0 decimal
        from data.scrapers.scrape_odds import BestFightOddsScraper
        scraper = BestFightOddsScraper()
        
        result_neg = scraper.parse_american_odds("-100")
        result_pos = scraper.parse_american_odds("+100")
        
        assert result_neg is not None
        assert abs(result_neg - 2.0) < 0.01
        assert result_pos is not None
        assert abs(result_pos - 2.0) < 0.01
    
    def test_american_to_decimal_heavy_favorite(self):
        """Test heavy favorite odds."""
        # -500 means bet $500 to win $100, so decimal = 1.2
        from data.scrapers.scrape_odds import BestFightOddsScraper
        scraper = BestFightOddsScraper()
        
        result = scraper.parse_american_odds("-500")
        expected = 1.2
        assert result is not None
        assert abs(result - expected) < 0.01
    
    def test_american_to_decimal_big_underdog(self):
        """Test big underdog odds."""
        # +500 means bet $100 to win $500, so decimal = 6.0
        from data.scrapers.scrape_odds import BestFightOddsScraper
        scraper = BestFightOddsScraper()
        
        result = scraper.parse_american_odds("+500")
        expected = 6.0
        assert result is not None
        assert abs(result - expected) < 0.01
    
    def test_american_to_decimal_invalid(self):
        """Test invalid odds input."""
        from data.scrapers.scrape_odds import BestFightOddsScraper
        scraper = BestFightOddsScraper()
        
        assert scraper.parse_american_odds("") is None
        assert scraper.parse_american_odds(None) is None
        assert scraper.parse_american_odds("abc") is None


class TestEloToProbability:
    """Test ELO to probability conversion."""
    
    def test_equal_elo(self):
        """Equal ELO should give 50% probability."""
        from ml.realistic_odds_estimator import elo_to_probability
        
        prob = elo_to_probability(0)  # ELO diff = 0
        assert abs(prob - 0.5) < 0.001
    
    def test_positive_elo_diff(self):
        """Positive ELO diff means fighter A is favored."""
        from ml.realistic_odds_estimator import elo_to_probability
        
        # +100 ELO diff should give ~64% probability
        prob = elo_to_probability(100)
        assert prob > 0.5
        assert abs(prob - 0.64) < 0.02
    
    def test_negative_elo_diff(self):
        """Negative ELO diff means fighter A is underdog."""
        from ml.realistic_odds_estimator import elo_to_probability
        
        prob = elo_to_probability(-100)
        assert prob < 0.5
        assert abs(prob - 0.36) < 0.02
    
    def test_large_elo_diff(self):
        """Large ELO diff should approach 1 or 0."""
        from ml.realistic_odds_estimator import elo_to_probability
        
        # +400 ELO diff should give ~90% probability
        prob_high = elo_to_probability(400)
        assert prob_high > 0.9
        
        prob_low = elo_to_probability(-400)
        assert prob_low < 0.1
    
    def test_symmetry(self):
        """Probability should be symmetric around 0.5."""
        from ml.realistic_odds_estimator import elo_to_probability
        
        prob_pos = elo_to_probability(150)
        prob_neg = elo_to_probability(-150)
        
        assert abs((prob_pos + prob_neg) - 1.0) < 0.001


class TestMarketOddsEstimation:
    """Test market odds estimation with vig."""
    
    def test_vig_adds_overround(self):
        """Vig should make implied probabilities sum to > 100%."""
        from ml.realistic_odds_estimator import estimate_market_odds
        
        odds = estimate_market_odds(0)  # Even fight
        
        overround = odds['implied_prob_a'] + odds['implied_prob_b'] - 1
        assert overround > 0  # Should have positive overround
        # The function uses ~4.5% vig split between both sides, resulting in ~2.25% overround
        assert overround > 0.01  # At least 1% overround
        assert overround < 0.10  # No more than 10% overround
    
    def test_favorite_gets_worse_odds(self):
        """Favorite should get worse decimal odds than underdog."""
        from ml.realistic_odds_estimator import estimate_market_odds
        
        odds = estimate_market_odds(100)  # Fighter A favored
        
        assert odds['decimal_a'] < odds['decimal_b']
    
    def test_reasonable_odds_range(self):
        """Odds should be in reasonable range."""
        from ml.realistic_odds_estimator import estimate_market_odds
        
        for elo_diff in [-200, -100, 0, 100, 200]:
            odds = estimate_market_odds(elo_diff)
            
            assert odds['decimal_a'] >= 1.01
            assert odds['decimal_a'] <= 20.0
            assert odds['decimal_b'] >= 1.01
            assert odds['decimal_b'] <= 20.0


class TestExpectedValue:
    """Test expected value calculations."""
    
    def test_ev_positive_when_edge(self):
        """EV should be positive when model prob > implied prob."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder.__new__(EdgeFinder)
        finder.model = None
        
        # If we think fighter has 60% chance but odds imply 50%
        ev = finder.calculate_expected_value(0.60, 2.0)  # 2.0 decimal = 50% implied
        assert ev > 0
    
    def test_ev_negative_when_no_edge(self):
        """EV should be negative when betting on unfavorable odds."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder.__new__(EdgeFinder)
        finder.model = None
        
        # If we think fighter has 40% chance but bet on them anyway
        ev = finder.calculate_expected_value(0.40, 2.0)
        assert ev < 0
    
    def test_ev_zero_at_break_even(self):
        """EV should be ~0 when model prob equals implied prob."""
        from ml.edge_finder import EdgeFinder
        
        finder = EdgeFinder.__new__(EdgeFinder)
        finder.model = None
        
        # 50% chance at 2.0 odds = break even
        ev = finder.calculate_expected_value(0.50, 2.0)
        assert abs(ev) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

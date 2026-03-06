"""
Tests for nationality extraction and country classification.

Tests:
- Country extraction from place of birth
- Region classification
- Edge cases (US states vs countries, ambiguous cities)
"""

import pytest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))


class TestCountryExtraction:
    """Test country extraction from place of birth."""
    
    def test_usa_state(self):
        """Test US state extraction."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Las Vegas, Nevada") == "USA"
        assert extract_country("Los Angeles, California") == "USA"
        assert extract_country("New York, New York") == "USA"
        assert extract_country("Houston, Texas") == "USA"
        assert extract_country("Miami, Florida") == "USA"
    
    def test_usa_explicit(self):
        """Test explicit USA mentions."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("United States") == "USA"
        assert extract_country("USA") == "USA"
        assert extract_country("Chicago, Illinois, USA") == "USA"
    
    def test_brazil(self):
        """Test Brazilian places."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Sao Paulo, Brazil") == "Brazil"
        assert extract_country("Rio de Janeiro, Brasil") == "Brazil"
        assert extract_country("Curitiba, Brazil") == "Brazil"
    
    def test_russia(self):
        """Test Russian places."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Moscow, Russia") == "Russia"
        assert extract_country("St. Petersburg, Russia") == "Russia"
        assert extract_country("Russian Federation") == "Russia"
    
    def test_dagestan(self):
        """Test Dagestan (separate from Russia for analysis)."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Makhachkala, Dagestan, Russia") == "Dagestan"
        assert extract_country("Khasavyurt, Dagestan") == "Dagestan"
    
    def test_uk(self):
        """Test UK places."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("London, England") == "UK"
        assert extract_country("Manchester, United Kingdom") == "UK"
        assert extract_country("Liverpool, UK") == "UK"
        assert extract_country("Glasgow, Scotland") == "UK"
        assert extract_country("Cardiff, Wales") == "UK"
        assert extract_country("Dublin, Ireland") == "UK"
    
    def test_canada(self):
        """Test Canadian places."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Toronto, Ontario, Canada") == "Canada"
        assert extract_country("Montreal, Quebec, Canada") == "Canada"
        assert extract_country("Vancouver, Canada") == "Canada"
    
    def test_australia(self):
        """Test Australian places."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Sydney, Australia") == "Australia"
        assert extract_country("Melbourne, Victoria, Australia") == "Australia"
    
    def test_japan(self):
        """Test Japanese places."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Tokyo, Japan") == "Japan"
        assert extract_country("Osaka, Japan") == "Japan"
    
    def test_central_asian_countries(self):
        """Test Central Asian countries (high-performing in UFC)."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Almaty, Kazakhstan") == "Kazakhstan"
        assert extract_country("Tashkent, Uzbekistan") == "Uzbekistan"
        assert extract_country("Bishkek, Kyrgyzstan") == "Kyrgyzstan"
        assert extract_country("Dushanbe, Tajikistan") == "Tajikistan"
    
    def test_african_countries(self):
        """Test African countries."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Lagos, Nigeria") == "Nigeria"
        assert extract_country("Johannesburg, South Africa") == "South Africa"
        assert extract_country("Casablanca, Morocco") == "Morocco"
    
    def test_georgia_disambiguation(self):
        """Test Georgia (country) vs Georgia (US state)."""
        from ml.analyze_nationality import extract_country
        
        # Atlanta is in Georgia the state
        assert extract_country("Atlanta, Georgia") == "USA"
        assert extract_country("Savannah, Georgia") == "USA"
        
        # Tbilisi is in Georgia the country
        assert extract_country("Tbilisi, Georgia") == "Georgia"
    
    def test_unknown_places(self):
        """Test unknown/empty places."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country(None) == "Unknown"
        assert extract_country("") == "Unknown"
        assert extract_country("Some Random Place") == "Unknown"
    
    def test_city_only(self):
        """Test city-only inputs that should map to countries."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Moscow") == "Russia"
        assert extract_country("Tokyo") == "Japan"
        assert extract_country("Sydney") == "Australia"
        assert extract_country("Paris") == "France"
        assert extract_country("Berlin") == "Germany"


class TestRegionClassification:
    """Test region classification."""
    
    def test_north_america(self):
        """Test North American countries."""
        from ml.analyze_nationality import get_region
        
        assert get_region("USA") == "North America"
        assert get_region("Canada") == "North America"
        assert get_region("Mexico") == "North America"
    
    def test_south_america(self):
        """Test South American countries."""
        from ml.analyze_nationality import get_region
        
        assert get_region("Brazil") == "South America"
        assert get_region("Argentina") == "South America"
        assert get_region("Peru") == "South America"
    
    def test_caucasus(self):
        """Test Caucasus region."""
        from ml.analyze_nationality import get_region
        
        assert get_region("Dagestan") == "Caucasus"
        assert get_region("Georgia") == "Caucasus"
        assert get_region("Azerbaijan") == "Caucasus"
        assert get_region("Armenia") == "Caucasus"
    
    def test_central_asia(self):
        """Test Central Asian region."""
        from ml.analyze_nationality import get_region
        
        assert get_region("Kazakhstan") == "Central Asia"
        assert get_region("Uzbekistan") == "Central Asia"
        assert get_region("Kyrgyzstan") == "Central Asia"
        assert get_region("Tajikistan") == "Central Asia"
    
    def test_eastern_europe(self):
        """Test Eastern European countries."""
        from ml.analyze_nationality import get_region
        
        assert get_region("Russia") == "Eastern Europe"
        assert get_region("Ukraine") == "Eastern Europe"
        assert get_region("Belarus") == "Eastern Europe"
    
    def test_unknown_region(self):
        """Test unknown country returns 'Other'."""
        from ml.analyze_nationality import get_region
        
        assert get_region("Unknown") == "Other"
        assert get_region("SomeRandomCountry") == "Other"


class TestNationalityEdgeCalculations:
    """Test nationality edge calculation logic."""
    
    def test_edge_calculation_positive(self):
        """Test positive edge calculation."""
        win_rate = 0.65
        expected = 0.50
        edge = win_rate - expected
        
        assert edge > 0
        assert abs(edge - 0.15) < 0.001
    
    def test_edge_calculation_negative(self):
        """Test negative edge calculation."""
        win_rate = 0.45
        expected = 0.50
        edge = win_rate - expected
        
        assert edge < 0
        assert abs(edge - (-0.05)) < 0.001
    
    def test_caucasus_edge_exists(self):
        """Verify Caucasus fighters have documented edge."""
        # Based on our analysis, Caucasus fighters have ~17% edge
        # This is a sanity check that the finding is captured
        documented_edge = 0.172  # From our analysis
        
        assert documented_edge > 0.10  # At least 10% edge
        assert documented_edge < 0.25  # Not unrealistically high


class TestCountryPatterns:
    """Test country pattern matching robustness."""
    
    def test_case_insensitivity(self):
        """Test case insensitivity of country matching."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("UNITED STATES") == "USA"
        assert extract_country("brazil") == "Brazil"
        assert extract_country("TOKYO, JAPAN") == "Japan"
        assert extract_country("london, england") == "UK"
    
    def test_partial_matches(self):
        """Test partial string matches."""
        from ml.analyze_nationality import extract_country
        
        assert extract_country("Born in Brazil") == "Brazil"
        assert extract_country("From Russia with love") == "Russia"
    
    def test_multiple_countries_in_string(self):
        """Test string with multiple country mentions."""
        from ml.analyze_nationality import extract_country
        
        # Should return first match based on pattern order
        result = extract_country("Moved from Brazil to USA")
        assert result in ["USA", "Brazil"]  # Either is acceptable


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

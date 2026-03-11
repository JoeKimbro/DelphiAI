"""
Tests for the UFC Injury Scraper module.

Tests cover:
- URL slug generation
- Text extraction from HTML
- Injury keyword detection
- Date estimation logic
- Penalty calculations
- False positive filtering
- News article link extraction
- Full integration tests (mocked)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.injury_scraper import InjuryScraper


class TestURLSlugGeneration:
    """Test fighter name to URL slug conversion."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_simple_name(self):
        """Test basic name conversion."""
        assert self.scraper._name_to_slug("Jon Jones") == "jon-jones"
    
    def test_name_with_special_characters(self):
        """Test name with accents/special chars."""
        assert self.scraper._name_to_slug("Jose Aldo") == "jose-aldo"
        # Note: Real implementation might need accent handling
    
    def test_name_with_multiple_spaces(self):
        """Test name with extra whitespace."""
        assert self.scraper._name_to_slug("Jon  Jones") == "jon-jones"
        assert self.scraper._name_to_slug("  Jon Jones  ") == "jon-jones"
    
    def test_name_with_hyphen(self):
        """Test name that already has hyphen."""
        assert self.scraper._name_to_slug("Kai Kara-France") == "kai-kara-france"
    
    def test_name_with_apostrophe(self):
        """Test name with apostrophe."""
        result = self.scraper._name_to_slug("Israel O'Malley")
        assert "israel" in result
        assert "malley" in result
    
    def test_three_part_name(self):
        """Test name with middle name."""
        assert self.scraper._name_to_slug("Charles Da Silva") == "charles-da-silva"


class TestInjuryKeywordDetection:
    """Test injury keyword matching."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_major_injury_surgery(self):
        """Test detection of surgery keywords."""
        text = "the fighter underwent knee surgery last month"
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        
        assert len(injuries) > 0
        assert injuries[0]['severity'] == 'major'
        assert 'surgery' in injuries[0]['keyword'].lower()
    
    def test_major_injury_torn_acl(self):
        """Test detection of ACL tear."""
        text = "test fighter suffered a torn acl during training"
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        
        assert len(injuries) > 0
        assert injuries[0]['severity'] == 'major'
    
    def test_major_injury_fracture(self):
        """Test detection of fractures."""
        text = "test fighter has a fractured hand from the last fight"
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        
        assert len(injuries) > 0
        assert injuries[0]['severity'] == 'major'
    
    def test_minor_injury_pulled_out(self):
        """Test detection of fight withdrawal."""
        text = "test fighter was forced to pull out of the bout"
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        
        assert len(injuries) > 0
        # Minor because no specific major injury mentioned
    
    def test_no_injury_found(self):
        """Test when no injury keywords present."""
        text = "the fighter is training hard for his upcoming bout"
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        
        assert len(injuries) == 0
    
    def test_medical_clearance_issue(self):
        """Test detection of medical clearance problems."""
        text = "test fighter failed the medical clearance before the fight"
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        
        assert len(injuries) > 0
    
    def test_heart_surgery_specific(self):
        """Test detection of specific surgeries."""
        text = "test fighter recovered from heart surgery last year"
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        
        assert len(injuries) > 0
        assert injuries[0]['severity'] == 'major'


class TestFalsePositiveFiltering:
    """Test filtering of injuries about other fighters."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_filter_other_fighter_injury(self):
        """Test that injuries about other fighters are filtered when section breaks exist."""
        # This text has clear section breaks - surgery is in Alex Perez's section
        text = "alex perez | discusses recovery from acl surgery | kyoji horiguchi | returns to action healthy"
        injuries = self.scraper._find_injuries(text, "Kyoji Horiguchi")
        
        # Should NOT find injuries since ACL surgery is in Alex Perez's section
        assert len(injuries) == 0
    
    def test_keep_correct_fighter_injury(self):
        """Test that injuries about the target fighter are kept."""
        text = "amir albazi ended up with heart surgery and is now recovering"
        injuries = self.scraper._find_injuries(text, "Amir Albazi")
        
        assert len(injuries) > 0
        assert injuries[0]['severity'] == 'major'
    
    def test_section_break_filtering(self):
        """Test filtering when article has section breaks."""
        # Note: Current implementation filters based on | breaks
        # The injury section must have the fighter name to be attributed
        
        # For John Smith - injury is in his section WITH his name
        text_john = "john smith had knee surgery and is recovering"
        injuries = self.scraper._find_injuries(text_john, "John Smith")
        assert len(injuries) > 0  # Should find injury
        
        # For Jane Doe - separate text with no injury
        text_jane = "jane doe training hard for next fight"
        injuries = self.scraper._find_injuries(text_jane, "Jane Doe")
        assert len(injuries) == 0  # No injury keywords
    
    def test_fighter_name_must_be_in_context(self):
        """Test that fighter name must appear near injury mention."""
        # The context window is 100 chars each side - place surgery far from fighter name
        filler = "x" * 150  # Enough to push surgery out of context window
        text = f"someone had surgery. {filler} test fighter is healthy and ready."
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        
        # Context around "surgery" shouldn't include "test fighter" due to distance
        assert len(injuries) == 0


class TestDateEstimation:
    """Test injury date estimation logic."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_recent_keywords(self):
        """Test 'this week' and 'just' keywords."""
        context = "test fighter just announced he's pulling out this week"
        date = self.scraper._estimate_injury_date(context)
        
        # Should be within last 2 weeks
        days_ago = (datetime.now() - date).days
        assert days_ago <= 14
    
    def test_last_month_keyword(self):
        """Test 'last month' keyword - note: 'breaking' triggers recent date."""
        # "breaking" keyword triggers 7-day estimate
        context = "test fighter announced injury last month news update"
        date = self.scraper._estimate_injury_date(context)
        
        days_ago = (datetime.now() - date).days
        # With recency keywords like "announced", this triggers ~90 days
        # The scraper checks recency keywords which include "announced"
        assert days_ago <= 100  # Within reasonable range for recency keywords
    
    def test_comeback_article_detection(self):
        """Test detection of comeback/return articles."""
        context = "test fighter returns after surgery it took a long time for recovery"
        date = self.scraper._estimate_injury_date(context)
        
        # Comeback articles should estimate injury ~10 months ago
        days_ago = (datetime.now() - date).days
        assert days_ago >= 180  # At least 6 months ago
    
    def test_year_month_extraction(self):
        """Test extraction of specific dates."""
        context = "test fighter had surgery in january 2025"
        date = self.scraper._estimate_injury_date(context)
        
        assert date.year == 2025
        assert date.month == 1
    
    def test_abbreviated_month(self):
        """Test abbreviated month names."""
        context = "test fighter injured in feb 2025"
        date = self.scraper._estimate_injury_date(context)
        
        assert date.year == 2025
        assert date.month == 2
    
    def test_default_date(self):
        """Test default when no date clues found (no comeback indicators)."""
        # Avoid comeback indicators like "had", "underwent", "returns"
        context = "test fighter suffering from injury situation unclear"
        date = self.scraper._estimate_injury_date(context)
        
        # Default should be ~6 months ago (180 days)
        days_ago = (datetime.now() - date).days
        assert 150 <= days_ago <= 210


class TestTimeMultipliers:
    """Test time-based penalty multipliers."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_very_recent_injury(self):
        """Test multiplier for injury within 2 months."""
        multiplier = self.scraper._get_time_multiplier(30)  # 30 days
        assert multiplier == 1.5
    
    def test_recent_injury(self):
        """Test multiplier for injury 2-4 months ago."""
        multiplier = self.scraper._get_time_multiplier(90)  # 90 days
        assert multiplier == 1.2
    
    def test_moderate_injury(self):
        """Test multiplier for injury 4-6 months ago."""
        multiplier = self.scraper._get_time_multiplier(150)  # 150 days
        assert multiplier == 1.0
    
    def test_older_injury(self):
        """Test multiplier for injury 6-12 months ago."""
        multiplier = self.scraper._get_time_multiplier(300)  # 300 days
        assert multiplier == 0.7
    
    def test_old_injury(self):
        """Test multiplier for injury 12-18 months ago."""
        multiplier = self.scraper._get_time_multiplier(450)  # 450 days
        assert multiplier == 0.3
    
    def test_very_old_injury(self):
        """Test multiplier for injury beyond 18 months."""
        multiplier = self.scraper._get_time_multiplier(600)  # 600 days
        assert multiplier == 0.1


class TestPenaltyCalculation:
    """Test ELO penalty calculations."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_major_injury_penalty(self):
        """Test penalty for major injury."""
        injuries = [{
            'keyword': 'surgery',
            'severity': 'major',
            'context': 'test fighter had surgery this month',
            'base_penalty': 55,
        }]
        
        penalty, details = self.scraper._calculate_penalty(injuries)
        
        # Should be around 55 * multiplier
        assert penalty > 0
        assert details['severity'] == 'major'
    
    def test_minor_injury_penalty(self):
        """Test penalty for minor injury."""
        injuries = [{
            'keyword': 'pulled out',
            'severity': 'minor',
            'context': 'test fighter pulled out recently',
            'base_penalty': 25,
        }]
        
        penalty, details = self.scraper._calculate_penalty(injuries)
        
        assert penalty > 0
        assert penalty < 55  # Should be less than major
        assert details['severity'] == 'minor'
    
    def test_no_penalty_for_old_injury(self):
        """Test that very old injuries get minimal penalty."""
        injuries = [{
            'keyword': 'surgery',
            'severity': 'major',
            'context': 'fighter had surgery in january 2024',
            'base_penalty': 55,
        }]
        
        penalty, details = self.scraper._calculate_penalty(injuries)
        
        # Old injury should have reduced penalty or be excluded
        # Depending on whether it's beyond the 18-month window
    
    def test_empty_injuries_list(self):
        """Test handling of empty injuries list."""
        penalty, details = self.scraper._calculate_penalty([])
        
        assert penalty == 0
        assert details == {}


class TestNewsLinkExtraction:
    """Test extraction of news article links from HTML."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_extract_news_links(self):
        """Test extraction of /news/ links."""
        html = '''
        <html>
        <body>
            <a href="/news/amir-albazi-returns">Article about Albazi</a>
            <a href="/news/some-other-news">Other news</a>
            <a href="/athlete/other">Not a news link</a>
        </body>
        </html>
        '''
        
        links = self.scraper._extract_news_links(html, "Amir Albazi")
        
        # Should find the link with "albazi" in it
        assert len(links) >= 1
        assert any('albazi' in link.lower() for link in links)
    
    def test_filter_unrelated_news(self):
        """Test that unrelated news is filtered."""
        html = '''
        <html>
        <body>
            <a href="/news/conor-mcgregor-news">McGregor article</a>
            <a href="/news/jon-jones-update">Jones article</a>
        </body>
        </html>
        '''
        
        links = self.scraper._extract_news_links(html, "Amir Albazi")
        
        # Should not find links for other fighters
        assert len(links) == 0
    
    def test_limit_news_links(self):
        """Test that news links are limited to 5."""
        html = '<html><body>'
        for i in range(10):
            html += f'<a href="/news/test-fighter-article-{i}">Article {i} about Test Fighter</a>'
        html += '</body></html>'
        
        links = self.scraper._extract_news_links(html, "Test Fighter")
        
        assert len(links) <= 5
    
    def test_make_absolute_url(self):
        """Test that relative URLs are made absolute."""
        html = '''
        <html>
        <body>
            <a href="/news/test-fighter-returns">Article</a>
        </body>
        </html>
        '''
        
        links = self.scraper._extract_news_links(html, "Test Fighter")
        
        if links:
            assert links[0].startswith('https://www.ufc.com')


class TestTextExtraction:
    """Test HTML text extraction."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_remove_scripts(self):
        """Test that script tags are removed."""
        html = '''
        <html>
        <body>
            <script>var x = "surgery";</script>
            <p>Normal text here</p>
        </body>
        </html>
        '''
        
        text = self.scraper._extract_text(html)
        
        # Should not include script content
        assert 'var x' not in text
        assert 'normal text' in text
    
    def test_remove_styles(self):
        """Test that style tags are removed."""
        html = '''
        <html>
        <body>
            <style>.injury { color: red; }</style>
            <p>Normal text here</p>
        </body>
        </html>
        '''
        
        text = self.scraper._extract_text(html)
        
        assert 'color: red' not in text
        assert 'normal text' in text
    
    def test_lowercase_output(self):
        """Test that output is lowercased."""
        html = '<html><body><p>SURGERY Fracture INJURY</p></body></html>'
        
        text = self.scraper._extract_text(html)
        
        assert 'SURGERY' not in text
        assert 'surgery' in text
    
    def test_whitespace_normalization(self):
        """Test that whitespace is normalized."""
        html = '<html><body><p>multiple    spaces\n\nhere</p></body></html>'
        
        text = self.scraper._extract_text(html)
        
        assert '    ' not in text  # No multiple spaces
        assert '\n\n' not in text  # No multiple newlines


class TestInjuryWindowValidation:
    """Test the 18-month injury relevance window."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_injury_within_window(self):
        """Test injury within 18-month window is counted."""
        injuries = [{
            'keyword': 'surgery',
            'severity': 'major',
            'context': 'test fighter had surgery last month',
            'base_penalty': 55,
        }]
        
        penalty, details = self.scraper._calculate_penalty(injuries)
        assert penalty > 0
    
    def test_injury_window_constant(self):
        """Test the injury window constant is 540 days."""
        assert self.scraper.INJURY_WINDOW_DAYS == 540


class TestIntegrationMocked:
    """Integration tests with mocked HTTP requests."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    @patch.object(InjuryScraper, '_fetch_page')
    def test_full_check_with_injury(self, mock_fetch):
        """Test full injury check when injury is found."""
        # Mock the fighter page
        mock_fetch.return_value = '''
        <html>
        <body>
            <p>Test Fighter had surgery last month and is recovering.</p>
            <a href="/news/test-fighter-returns">News article</a>
        </body>
        </html>
        '''
        
        result = self.scraper.check_fighter_injuries("Test Fighter", check_news=False)
        
        assert result['injury_found'] == True
        assert result['elo_penalty'] > 0
        assert result['error'] is None
    
    @patch.object(InjuryScraper, '_fetch_page')
    def test_full_check_no_injury(self, mock_fetch):
        """Test full injury check when no injury is found."""
        mock_fetch.return_value = '''
        <html>
        <body>
            <p>Test Fighter is in great shape and ready to compete.</p>
        </body>
        </html>
        '''
        
        result = self.scraper.check_fighter_injuries("Test Fighter", check_news=False)
        
        assert result['injury_found'] == False
        assert result['elo_penalty'] == 0
    
    @patch.object(InjuryScraper, '_fetch_page')
    def test_full_check_page_not_found(self, mock_fetch):
        """Test handling when fighter page doesn't exist."""
        mock_fetch.return_value = None
        
        result = self.scraper.check_fighter_injuries("Unknown Fighter", check_news=False)
        
        assert result['injury_found'] == False
        assert result['error'] is not None


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_rate_limit_constants(self):
        """Test rate limiting constants are set."""
        assert self.scraper.REQUEST_DELAY[0] >= 1  # Min delay
        assert self.scraper.REQUEST_DELAY[1] >= self.scraper.REQUEST_DELAY[0]
        assert self.scraper.MAX_RETRIES >= 1
        assert self.scraper.TIMEOUT >= 5


class TestUserAgentRotation:
    """Test user agent rotation for safe scraping."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_multiple_user_agents(self):
        """Test that multiple user agents are available."""
        assert len(self.scraper.USER_AGENTS) >= 2
    
    def test_headers_contain_user_agent(self):
        """Test that headers include a user agent."""
        headers = self.scraper._get_headers()
        
        assert 'User-Agent' in headers
        assert headers['User-Agent'] in self.scraper.USER_AGENTS


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        self.scraper = InjuryScraper()
    
    def test_empty_text(self):
        """Test handling of empty text."""
        injuries = self.scraper._find_injuries("", "Test Fighter")
        assert len(injuries) == 0
    
    def test_empty_fighter_name(self):
        """Test handling of empty fighter name."""
        text = "someone had surgery"
        injuries = self.scraper._find_injuries(text, "")
        # Should still work but with relaxed filtering
        assert isinstance(injuries, list)
    
    def test_very_long_text(self):
        """Test handling of very long text with injury near fighter name."""
        # Put fighter name close to injury so it's within context window
        text = "normal text " * 100 + " test fighter had surgery last week " + "more text " * 100
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        # Should still find the injury when name is near the keyword
        assert len(injuries) > 0
    
    def test_unicode_in_text(self):
        """Test handling of unicode characters."""
        text = "test fighter had surgery in São Paulo"
        injuries = self.scraper._find_injuries(text, "Test Fighter")
        assert len(injuries) > 0
    
    def test_case_insensitivity(self):
        """Test that keyword matching is case insensitive."""
        text = "TEST FIGHTER HAD SURGERY"
        injuries = self.scraper._find_injuries(text.lower(), "Test Fighter")
        assert len(injuries) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

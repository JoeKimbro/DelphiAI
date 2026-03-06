"""
UFC Injury Scraper

Scrapes UFC.com fighter pages for injury keywords and calculates ELO adjustments.

Features:
- Safe scraping with rate limiting and retries
- Major/minor injury classification
- Time-based penalty multipliers
- 18-month relevance window

Usage:
    from ml.injury_scraper import InjuryScraper
    scraper = InjuryScraper()
    result = scraper.check_fighter_injuries("Amir Albazi")
"""

import re
import time
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from urllib.parse import quote
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InjuryScraper:
    """Scrape UFC.com for fighter injury information."""
    
    # Base URL for UFC athlete pages
    UFC_BASE_URL = "https://www.ufc.com/athlete/"
    
    # Safe scraping settings
    REQUEST_DELAY = (2, 5)  # Random delay between 2-5 seconds
    MAX_RETRIES = 3
    TIMEOUT = 15
    
    # User agent rotation for safe scraping
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    ]
    
    # Injury relevance window (18 months = 540 days)
    INJURY_WINDOW_DAYS = 540
    
    # =========================================================================
    # INJURY KEYWORDS
    # =========================================================================
    
    # Major injuries (50-60 ELO penalty base)
    MAJOR_INJURY_KEYWORDS = [
        # Specific surgeries (common in UFC articles)
        'heart surgery', 'neck surgery', 'knee surgery', 'shoulder surgery',
        'back surgery', 'hip surgery', 'ankle surgery', 'elbow surgery',
        'wrist surgery', 'acl surgery', 'mcl surgery',
        
        # General surgery terms
        'surgery', 'underwent surgery', 'surgical', 'operated on',
        'reconstructive', 'medical procedure', 'operation',
        'another surgery', 'had surgery',
        
        # Medical clearance issues (specific to combat sports)
        'medical clearance', 'medical test', 'failed medical',
        'not medically cleared', 'medical suspension', 'medically suspended',
        'not cleared to fight', 'medical issues',
        
        # Tears and ruptures
        'torn', 'tear', 'ruptured', 'acl tear', 'mcl tear', 'acl injury',
        'meniscus', 'rotator cuff', 'labrum', 'ligament tear',
        
        # Breaks and fractures
        'broken', 'fracture', 'fractured', 'shattered',
        'broken hand', 'broken foot', 'broken orbital', 'broken jaw',
        'broken nose', 'broken rib', 'broken arm', 'broken leg',
        
        # Severe conditions
        'detached retina', 'herniated disc', 'slipped disc',
        'out indefinitely', 'sidelined for months', 'career threatening',
        'serious injury', 'major injury', 'significant injury',
        'major health issues', 'health issues',
    ]
    
    # Minor injuries (20-30 ELO penalty base)
    MINOR_INJURY_KEYWORDS = [
        # General injury terms
        'injured', 'injury', 'hurt', 'sidelined',
        
        # Strains and pulls
        'strain', 'strained', 'pulled', 'pulled muscle', 'muscle tear',
        'sprain', 'sprained', 'tweaked',
        
        # Withdrawal (common in UFC articles)
        'pulled out', 'withdrew', 'withdrawal', 'forced out',
        'forced to withdraw', 'had to pull out', 'pull out',
        'forced to pull out', 'off the card', 'removed from',
        'bout cancelled', 'fight cancelled', 'fight scrapped',
        
        # Medical issues
        'medical issue', 'undisclosed injury', 'undisclosed illness',
        
        # Recovery
        'recovering', 'recovery', 'rehabilitation', 'rehab',
        'treatment', 'physical therapy',
        
        # Minor conditions
        'bruised', 'bruise', 'swollen', 'swelling', 'inflammation',
        'minor procedure', 'precautionary', 'nagging injury',
        'back issue', 'knee issue', 'shoulder issue',
    ]
    
    # Context keywords (help confirm recency)
    RECENCY_KEYWORDS = [
        'recent', 'recently', 'this week', 'last week',
        'this month', 'last month', 'announced', 'confirmed',
        'revealed', 'disclosed', 'reported', 'sources say',
        'breaking', 'just', 'update', 'news',
    ]
    
    # Month names for date extraction
    MONTHS = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    
    # Year patterns
    YEARS = ['2024', '2025', '2026']
    
    # =========================================================================
    # TIME-BASED MULTIPLIERS
    # =========================================================================
    
    # Time since injury -> multiplier
    # More recent = higher penalty
    TIME_MULTIPLIERS = [
        (60, 1.5),    # 0-2 months: 1.5x
        (120, 1.2),   # 2-4 months: 1.2x
        (180, 1.0),   # 4-6 months: 1.0x (base)
        (365, 0.7),   # 6-12 months: 0.7x
        (540, 0.3),   # 12-18 months: 0.3x
    ]
    
    # Base penalties
    MAJOR_INJURY_PENALTY = 55  # Base: 50-60
    MINOR_INJURY_PENALTY = 25  # Base: 20-30
    
    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.last_request_time = 0
    
    def _get_headers(self) -> Dict[str, str]:
        """Get randomized headers for safe scraping."""
        return {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        min_delay, max_delay = self.REQUEST_DELAY
        required_delay = random.uniform(min_delay, max_delay)
        
        if elapsed < required_delay:
            sleep_time = required_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _name_to_slug(self, name: str) -> str:
        """Convert fighter name to UFC URL slug."""
        # Remove special characters, convert to lowercase, replace spaces with hyphens
        slug = name.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars
        slug = re.sub(r'\s+', '-', slug)       # Spaces to hyphens
        slug = re.sub(r'-+', '-', slug)        # Multiple hyphens to single
        return slug
    
    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page with retries and rate limiting."""
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=self.TIMEOUT
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404:
                    logger.warning(f"Fighter page not found: {url}")
                    return None
                elif response.status_code == 429:
                    # Rate limited - back off
                    wait_time = (attempt + 1) * 30
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
            
            # Exponential backoff
            if attempt < self.MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 2)
        
        return None
    
    def _extract_text(self, html: str) -> str:
        """Extract relevant text from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator=' ')
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.lower()
    
    def _extract_news_links(self, html: str, fighter_name: str) -> List[str]:
        """Extract UFC news article links from fighter page."""
        soup = BeautifulSoup(html, 'html.parser')
        news_links = []
        
        # Find all links that point to /news/
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Check if it's a news article link
            if '/news/' in href:
                # Make absolute URL if relative
                if href.startswith('/'):
                    full_url = f"https://www.ufc.com{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                # Only include if it seems related to the fighter
                fighter_slug = self._name_to_slug(fighter_name)
                fighter_parts = fighter_name.lower().split()
                
                # Check if fighter name appears in URL or link text
                link_text = link.get_text().lower()
                href_lower = href.lower()
                
                if (any(part in href_lower for part in fighter_parts) or
                    any(part in link_text for part in fighter_parts) or
                    fighter_slug in href_lower):
                    if full_url not in news_links:
                        news_links.append(full_url)
        
        return news_links[:5]  # Limit to 5 most recent articles
    
    def _scrape_news_articles(self, news_links: List[str], debug: bool = False) -> str:
        """Scrape text from news articles."""
        all_text = []
        
        for url in news_links:
            logger.info(f"Checking news article: {url}")
            html = self._fetch_page(url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove scripts/styles/nav first
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
                    element.decompose()
                
                # Get all text from the page
                text = soup.get_text(separator=' ')
                text = re.sub(r'\s+', ' ', text)
                text_lower = text.lower()
                
                if debug:
                    # Show a sample of text containing key injury words
                    logger.info(f"Text length: {len(text_lower)}")
                    if 'surgery' in text_lower:
                        idx = text_lower.find('surgery')
                        logger.info(f"Found 'surgery' context: ...{text_lower[max(0,idx-50):idx+50]}...")
                
                all_text.append(text_lower)
        
        return ' '.join(all_text)
    
    def _find_injuries(self, text: str, fighter_name: str = None) -> List[Dict]:
        """
        Find injury mentions in text that are relevant to the fighter.
        
        Args:
            text: Text to search
            fighter_name: Fighter's name - only injuries near this name are considered
        """
        injuries = []
        
        # Get fighter name parts for proximity check
        fighter_parts = []
        if fighter_name:
            fighter_parts = [p.lower() for p in fighter_name.split() if len(p) > 2]
        
        def is_relevant_context(context: str) -> bool:
            """Check if the injury context is about our fighter, not someone else."""
            if not fighter_parts:
                return True  # No fighter name given, accept all
            
            context_lower = context.lower()
            
            # The fighter's name should appear in the context
            # Check if ANY part of their name appears
            name_in_context = any(part in context_lower for part in fighter_parts)
            
            if not name_in_context:
                return False  # Injury mention doesn't seem to be about our fighter
            
            # Additional check: Make sure another fighter's name doesn't appear
            # more prominently (they might be discussing another fighter's injury)
            # Look for common UFC fighter name patterns that aren't our fighter
            # This is a simple heuristic - look for "| " pattern indicating a section break
            if ' | ' in context_lower:
                # Split on section breaks and check which section has the injury
                parts = context_lower.split(' | ')
                for part in parts:
                    # If injury keyword is in this part, check if fighter name is too
                    if not any(fp in part for fp in fighter_parts):
                        # This part has injury but not our fighter's name
                        # Might be about another fighter
                        return False
            
            return True
        
        # Check for major injuries
        for keyword in self.MAJOR_INJURY_KEYWORDS:
            if keyword.lower() in text:
                # Find surrounding context (100 chars each side)
                pattern = re.compile(
                    r'.{0,100}' + re.escape(keyword.lower()) + r'.{0,100}',
                    re.IGNORECASE
                )
                matches = pattern.findall(text)
                
                for context in matches:
                    if is_relevant_context(context):
                        injuries.append({
                            'keyword': keyword,
                            'severity': 'major',
                            'context': context.strip(),
                            'base_penalty': self.MAJOR_INJURY_PENALTY,
                        })
        
        # Check for minor injuries (only if no major found)
        if not injuries:
            for keyword in self.MINOR_INJURY_KEYWORDS:
                if keyword.lower() in text:
                    pattern = re.compile(
                        r'.{0,100}' + re.escape(keyword.lower()) + r'.{0,100}',
                        re.IGNORECASE
                    )
                    matches = pattern.findall(text)
                    
                    for context in matches:
                        if is_relevant_context(context):
                            injuries.append({
                                'keyword': keyword,
                                'severity': 'minor',
                                'context': context.strip(),
                                'base_penalty': self.MINOR_INJURY_PENALTY,
                            })
        
        return injuries
    
    def _estimate_injury_date(self, context: str) -> Optional[datetime]:
        """Try to extract injury date from context."""
        context_lower = context.lower()
        now = datetime.now()
        
        # Check if this is a "return" or "comeback" article (injury is in the past)
        comeback_indicators = [
            'returns', 'return', 'comeback', 'back in action', 'cleared',
            'all clear', 'medically cleared', 'recovered', 'fully healed',
            'ready to fight', 'back in the cage', 'ended up with',  # past tense
            'i had', 'had surgery', 'underwent', 'went through',
            'after that surgery', 'after the surgery', 'cleared from',
            'since the fight', 'since that', 'it took a long time',
        ]
        
        is_comeback_article = any(kw in context_lower for kw in comeback_indicators)
        
        # Look for year + month combinations
        for year in self.YEARS:
            if year in context_lower:
                for i, month in enumerate(self.MONTHS):
                    if month in context_lower:
                        # Map abbreviated months to month number
                        month_num = (i % 12) + 1
                        try:
                            return datetime(int(year), month_num, 15)  # Assume mid-month
                        except:
                            pass
        
        # If comeback article, assume injury was 6-12 months ago (longer recovery)
        if is_comeback_article:
            return now - timedelta(days=300)  # ~10 months ago
        
        # Check for recency keywords
        if any(kw in context_lower for kw in ['this week', 'just', 'breaking']):
            return now - timedelta(days=7)
        elif any(kw in context_lower for kw in ['last week']):
            return now - timedelta(days=14)
        elif any(kw in context_lower for kw in ['this month', 'recently']):
            return now - timedelta(days=30)
        elif any(kw in context_lower for kw in ['last month']):
            return now - timedelta(days=60)
        
        # If recency keywords found but no date, assume recent
        if any(kw in context_lower for kw in self.RECENCY_KEYWORDS):
            return now - timedelta(days=90)
        
        # Default: assume 6 months ago if injury found but no date
        return now - timedelta(days=180)
    
    def _get_time_multiplier(self, days_since_injury: int) -> float:
        """Get penalty multiplier based on time since injury."""
        for max_days, multiplier in self.TIME_MULTIPLIERS:
            if days_since_injury <= max_days:
                return multiplier
        
        # Beyond 18 months - minimal impact
        return 0.1
    
    def _calculate_penalty(self, injuries: List[Dict]) -> Tuple[int, Dict]:
        """Calculate total ELO penalty from injuries."""
        if not injuries:
            return 0, {}
        
        # Take the most severe injury
        major_injuries = [i for i in injuries if i['severity'] == 'major']
        
        if major_injuries:
            injury = major_injuries[0]
        else:
            injury = injuries[0]
        
        # Estimate date
        injury_date = self._estimate_injury_date(injury['context'])
        
        if injury_date:
            days_since = (datetime.now() - injury_date).days
            
            # Check if within relevance window
            if days_since > self.INJURY_WINDOW_DAYS:
                return 0, {'reason': 'Injury too old (>18 months)', 'days_since': days_since}
            
            multiplier = self._get_time_multiplier(days_since)
            penalty = int(injury['base_penalty'] * multiplier)
        else:
            days_since = 180  # Default assumption
            multiplier = 1.0
            penalty = injury['base_penalty']
        
        return penalty, {
            'keyword': injury['keyword'],
            'severity': injury['severity'],
            'context': injury['context'][:200],
            'estimated_date': injury_date.strftime('%Y-%m-%d') if injury_date else 'Unknown',
            'days_since': days_since,
            'multiplier': multiplier,
            'base_penalty': injury['base_penalty'],
            'final_penalty': penalty,
        }
    
    def check_fighter_injuries(self, fighter_name: str, check_news: bool = True, debug: bool = False) -> Dict:
        """
        Check a fighter for injuries and calculate ELO adjustment.
        
        Args:
            fighter_name: Fighter's full name
            check_news: Also check UFC news articles (slower but more thorough)
            debug: Enable debug output to see what text is being scraped
            
        Returns:
            Dict with injury info and ELO penalty
        """
        result = {
            'fighter': fighter_name,
            'url': None,
            'injury_found': False,
            'elo_penalty': 0,
            'details': {},
            'news_articles_checked': 0,
            'error': None,
        }
        
        # Build URL
        slug = self._name_to_slug(fighter_name)
        url = f"{self.UFC_BASE_URL}{slug}"
        result['url'] = url
        
        logger.info(f"Checking injuries for {fighter_name}: {url}")
        
        # Fetch fighter page
        html = self._fetch_page(url)
        
        if not html:
            result['error'] = 'Could not fetch fighter page'
            return result
        
        # Extract text from fighter page
        all_text = self._extract_text(html)
        
        if debug:
            logger.info(f"Fighter page text length: {len(all_text)}")
            if 'surgery' in all_text:
                idx = all_text.find('surgery')
                logger.info(f"Found 'surgery' on fighter page: ...{all_text[max(0,idx-50):idx+50]}...")
        
        # Also check news articles linked from the page
        if check_news:
            news_links = self._extract_news_links(html, fighter_name)
            result['news_articles_checked'] = len(news_links)
            
            if news_links:
                logger.info(f"Found {len(news_links)} news articles to check")
                news_text = self._scrape_news_articles(news_links, debug=debug)
                all_text += ' ' + news_text
        
        if debug:
            logger.info(f"Total text length: {len(all_text)}")
            # Check for specific injury keywords
            for kw in ['surgery', 'heart surgery', 'neck surgery', 'medical']:
                if kw in all_text:
                    idx = all_text.find(kw)
                    logger.info(f"Found '{kw}': ...{all_text[max(0,idx-30):idx+50]}...")
        
        # Find injuries in combined text (only those relevant to this fighter)
        injuries = self._find_injuries(all_text, fighter_name)
        
        if debug:
            logger.info(f"Injuries found: {len(injuries)}")
        
        if injuries:
            result['injury_found'] = True
            penalty, details = self._calculate_penalty(injuries)
            result['elo_penalty'] = penalty
            result['details'] = details
            result['all_mentions'] = len(injuries)
        
        return result
    
    def get_adjusted_elo(
        self, 
        fighter_name: str, 
        current_elo: float,
        days_since_last_fight: int = 0
    ) -> Dict:
        """
        Get ELO adjusted for injuries and inactivity.
        
        Args:
            fighter_name: Fighter's name
            current_elo: Current ELO rating
            days_since_last_fight: Days since last fight (for ring rust)
            
        Returns:
            Dict with adjusted ELO and breakdown
        """
        result = {
            'fighter': fighter_name,
            'raw_elo': current_elo,
            'injury_penalty': 0,
            'inactivity_penalty': 0,
            'adjusted_elo': current_elo,
            'adjustments': [],
        }
        
        # Check for injuries
        injury_result = self.check_fighter_injuries(fighter_name)
        
        if injury_result['injury_found']:
            result['injury_penalty'] = injury_result['elo_penalty']
            result['adjustments'].append({
                'type': 'injury',
                'penalty': injury_result['elo_penalty'],
                'details': injury_result['details'],
            })
        
        # Calculate inactivity decay (ring rust)
        if days_since_last_fight > 180:  # More than 6 months
            # Decay toward 1500 baseline
            # Rate: ~5% per year of inactivity
            years_inactive = days_since_last_fight / 365
            decay_rate = 0.05 * years_inactive
            decay_rate = min(decay_rate, 0.25)  # Cap at 25% decay
            
            elo_diff_from_baseline = current_elo - 1500
            inactivity_decay = int(elo_diff_from_baseline * decay_rate)
            
            result['inactivity_penalty'] = inactivity_decay
            result['adjustments'].append({
                'type': 'inactivity',
                'days_inactive': days_since_last_fight,
                'years_inactive': round(years_inactive, 1),
                'decay_rate': f"{decay_rate*100:.1f}%",
                'penalty': inactivity_decay,
            })
        
        # Calculate final adjusted ELO
        total_penalty = result['injury_penalty'] + result['inactivity_penalty']
        result['adjusted_elo'] = current_elo - total_penalty
        result['total_penalty'] = total_penalty
        
        return result


def check_fighter(name: str, debug: bool = False):
    """Quick check for a single fighter."""
    scraper = InjuryScraper()
    result = scraper.check_fighter_injuries(name, check_news=True, debug=debug)
    
    print(f"\n{'='*60}")
    print(f"INJURY CHECK: {name}")
    print(f"{'='*60}")
    print(f"Fighter Page: {result['url']}")
    print(f"News Articles Checked: {result.get('news_articles_checked', 0)}")
    print(f"Injury Found: {result['injury_found']}")
    
    if result['injury_found']:
        print(f"\nELO Penalty: -{result['elo_penalty']} points")
        if result['details']:
            d = result['details']
            print(f"  Severity: {d.get('severity', 'N/A').upper()}")
            print(f"  Keyword: {d.get('keyword', 'N/A')}")
            print(f"  Estimated Date: {d.get('estimated_date', 'N/A')}")
            print(f"  Days Since: {d.get('days_since', 'N/A')}")
            print(f"  Multiplier: {d.get('multiplier', 'N/A')}x")
            print(f"\n  Context: ...{d.get('context', '')[:200]}...")
    elif result['error']:
        print(f"Error: {result['error']}")
    else:
        print("No injuries found on UFC.com page or news articles")
    
    return result


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Check UFC fighter for injuries')
    parser.add_argument('fighter_name', nargs='*', help='Fighter name')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    if args.fighter_name:
        fighter_name = ' '.join(args.fighter_name)
        check_fighter(fighter_name, debug=args.debug)
    else:
        # Demo with Amir Albazi
        print("Usage: python -m ml.injury_scraper \"Fighter Name\" [--debug]")
        print("\nRunning demo with Amir Albazi...")
        check_fighter("Amir Albazi", debug=True)

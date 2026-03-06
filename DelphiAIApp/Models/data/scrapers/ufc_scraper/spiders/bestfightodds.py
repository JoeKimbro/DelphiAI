"""
BestFightOdds.com Scraper

Scrapes historical betting odds for UFC fights.
This gives us real closing lines to calculate true ROI.

Data collected:
- Event name and date
- Fighter names
- Opening odds (when available)
- Closing odds (what we need for backtesting)
- Multiple sportsbook odds
"""

import re
import json
import scrapy
from datetime import datetime
from pathlib import Path


class BestFightOddsSpider(scrapy.Spider):
    name = 'bestfightodds'
    allowed_domains = ['bestfightodds.com']
    
    # Start with UFC events archive
    start_urls = ['https://www.bestfightodds.com/archive']
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2.0,  # Be respectful
        'CONCURRENT_REQUESTS': 1,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'FEEDS': {
            'output/historical_odds.json': {
                'format': 'json',
                'overwrite': True,
            },
        },
    }
    
    def __init__(self, start_year=2022, end_year=2026, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.events_scraped = 0
        self.fights_scraped = 0
    
    def parse(self, response):
        """Parse the archive page to find UFC events."""
        self.logger.info("Parsing archive page...")
        
        # Find all event links
        # BestFightOdds structure: /events/ufc-xxx-fighter-vs-fighter
        event_links = response.css('a[href*="/events/ufc-"]::attr(href)').getall()
        
        self.logger.info(f"Found {len(event_links)} UFC event links")
        
        for link in event_links:
            # Filter by year if possible from the page context
            full_url = response.urljoin(link)
            yield scrapy.Request(full_url, callback=self.parse_event)
    
    def parse_event(self, response):
        """Parse individual event page for odds."""
        
        # Get event info
        event_name = response.css('h1::text').get()
        if not event_name:
            event_name = response.css('.page-title::text').get()
        
        event_name = event_name.strip() if event_name else "Unknown Event"
        
        # Get event date
        date_text = response.css('.table-header-date::text').get()
        if not date_text:
            date_text = response.css('.event-date::text').get()
        
        event_date = self.parse_date(date_text) if date_text else None
        
        # Filter by year
        if event_date:
            year = event_date.year
            if year < self.start_year or year > self.end_year:
                self.logger.debug(f"Skipping {event_name} (year {year} outside range)")
                return
        
        self.logger.info(f"Parsing event: {event_name} ({event_date})")
        self.events_scraped += 1
        
        # Parse fight rows
        # BestFightOdds typically has a table with fights
        fight_rows = response.css('table.odds-table tbody tr')
        
        if not fight_rows:
            # Try alternative structure
            fight_rows = response.css('.event-holder .table-row')
        
        current_fight = {}
        
        for row in fight_rows:
            # Check if this is a fighter row (has odds)
            fighter_name = row.css('a.href-fighter-name::text').get()
            if not fighter_name:
                fighter_name = row.css('.name a::text').get()
            if not fighter_name:
                fighter_name = row.css('td:first-child a::text').get()
            
            if fighter_name:
                fighter_name = fighter_name.strip()
                
                # Get odds from various sportsbooks
                odds_cells = row.css('td.but-sg, td.odds-cell')
                odds = []
                
                for cell in odds_cells:
                    odd_text = cell.css('::text').get()
                    if odd_text:
                        odd_text = odd_text.strip()
                        parsed_odd = self.parse_american_odds(odd_text)
                        if parsed_odd:
                            odds.append(parsed_odd)
                
                # Also try data attributes
                if not odds:
                    for cell in row.css('td[data-odds]'):
                        odd_val = cell.attrib.get('data-odds')
                        if odd_val:
                            parsed = self.parse_american_odds(odd_val)
                            if parsed:
                                odds.append(parsed)
                
                # Determine best odds and closing line
                if odds:
                    best_odds = max(odds)  # Best odds for bettor
                    avg_odds = sum(odds) / len(odds)
                else:
                    best_odds = None
                    avg_odds = None
                
                # Build fight record
                if 'fighter1' not in current_fight:
                    current_fight = {
                        'event_name': event_name,
                        'event_date': event_date.isoformat() if event_date else None,
                        'fighter1': fighter_name,
                        'fighter1_odds': odds,
                        'fighter1_best_odds': best_odds,
                        'fighter1_avg_odds': avg_odds,
                    }
                else:
                    current_fight['fighter2'] = fighter_name
                    current_fight['fighter2_odds'] = odds
                    current_fight['fighter2_best_odds'] = best_odds
                    current_fight['fighter2_avg_odds'] = avg_odds
                    
                    # Yield complete fight
                    self.fights_scraped += 1
                    yield current_fight
                    current_fight = {}
        
        self.logger.info(f"Scraped {self.fights_scraped} fights from {self.events_scraped} events")
    
    def parse_date(self, date_str):
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        formats = [
            '%B %d, %Y',
            '%b %d, %Y', 
            '%m/%d/%Y',
            '%Y-%m-%d',
            '%d %B %Y',
            '%d %b %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def parse_american_odds(self, odds_str):
        """
        Parse American odds string to decimal odds.
        +150 -> 2.50
        -150 -> 1.67
        """
        if not odds_str:
            return None
        
        odds_str = str(odds_str).strip().replace(',', '')
        
        try:
            if odds_str.startswith('+'):
                american = int(odds_str[1:])
                return 1 + (american / 100)
            elif odds_str.startswith('-'):
                american = int(odds_str[1:])
                return 1 + (100 / american)
            elif odds_str.replace('.', '').isdigit():
                # Already decimal
                return float(odds_str)
            else:
                # Try parsing as integer
                val = int(odds_str)
                if val > 0:
                    return 1 + (val / 100)
                else:
                    return 1 + (100 / abs(val))
        except (ValueError, ZeroDivisionError):
            return None
    
    def closed(self, reason):
        """Called when spider closes."""
        self.logger.info(f"Spider closed: {reason}")
        self.logger.info(f"Total events scraped: {self.events_scraped}")
        self.logger.info(f"Total fights scraped: {self.fights_scraped}")

"""
UFCStats.com Spider

Crawls all UFC fighters from ufcstats.com:
1. Iterates through fighter listing pages (A-Z)
2. Extracts fighter detail page URLs
3. Scrapes fighter bio, career stats, and fight history

Usage:
    cd DelphiAIApp/Models/data/scrapers
    scrapy crawl ufcstats -o ../output/ufcstats_fighters.json
"""

import re
import string
from datetime import datetime

import scrapy
from ufc_scraper.items import FighterItem, CareerStatsItem, FightItem


class UFCStatsSpider(scrapy.Spider):
    name = "ufcstats"
    allowed_domains = ["ufcstats.com"]
    
    # Base URL for fighter listings
    base_url = "http://www.ufcstats.com/statistics/fighters"
    
    def start_requests(self):
        """
        Generate requests for fighter listing pages A-Z.
        """
        for char in string.ascii_lowercase:
            url = f"{self.base_url}?char={char}&page=all"
            yield scrapy.Request(url, callback=self.parse_fighter_list)
    
    def parse_fighter_list(self, response):
        """
        Parse fighter listing page and extract links to fighter detail pages.
        """
        # Find all fighter links in the table
        fighter_links = response.css('table.b-statistics__table tbody tr td.b-statistics__table-col a::attr(href)').getall()
        
        # Deduplicate (each fighter appears multiple times in the row)
        unique_links = list(set(fighter_links))
        
        self.logger.info(f"Found {len(unique_links)} fighters on {response.url}")
        
        for link in unique_links:
            if link and 'fighter-details' in link:
                yield scrapy.Request(link, callback=self.parse_fighter_detail)
    
    def parse_fighter_detail(self, response):
        """
        Parse individual fighter detail page.
        Yields: FighterItem, CareerStatsItem, and multiple FightItems
        """
        now = datetime.utcnow().isoformat()
        
        # === FIGHTER BIO ===
        fighter = FighterItem()
        
        # Name from the title
        fighter['name'] = self._clean_text(
            response.css('span.b-content__title-highlight::text').get()
        )
        fighter['fighter_url'] = response.url
        
        # Record (e.g., "Record: 27-8-0")
        record_text = self._clean_text(
            response.css('span.b-content__title-record::text').get()
        )
        wins, losses, draws = self._parse_record(record_text)
        fighter['wins'] = wins
        fighter['losses'] = losses
        fighter['draws'] = draws
        fighter['total_fights'] = wins + losses + draws if all(x is not None for x in [wins, losses, draws]) else None
        
        # Bio details (Height, Weight, Reach, Stance, DOB)
        bio_items = response.css('ul.b-list__box-list li.b-list__box-list-item')
        for item in bio_items:
            label = self._clean_text(item.css('i.b-list__box-item-title::text').get())
            value = self._clean_text(item.css('::text').getall()[-1]) if item.css('::text').getall() else None
            
            if label:
                label_lower = label.lower().replace(':', '').strip()
                if 'height' in label_lower:
                    fighter['height'] = value
                elif 'weight' in label_lower:
                    fighter['weight'] = value
                elif 'reach' in label_lower:
                    fighter['reach'] = value
                elif 'stance' in label_lower:
                    fighter['stance'] = value
                elif 'dob' in label_lower:
                    fighter['dob'] = value
        
        fighter['scraped_at'] = now
        fighter['source'] = 'ufcstats'
        
        # === PARSE FIGHT HISTORY FIRST to get last fight date ===
        fight_rows = response.css('table.b-fight-details__table tbody tr.b-fight-details__table-row')
        fights_to_yield = []
        most_recent_fight_date = None
        
        for row in fight_rows:
            # Skip header rows
            if row.css('th'):
                continue
                
            fight = FightItem()
            fight['fighter_name'] = fighter['name']
            fight['fighter_url'] = response.url
            
            # Get all cells in the row
            cells = row.css('td.b-fight-details__table-col')
            
            if len(cells) >= 8:
                # Column 0: W/L result (nested in i.b-flag__text)
                result_text = cells[0].css('i.b-flag__text::text').get()
                if not result_text:
                    # Fallback: try to get any text content
                    result_text = cells[0].css('a.b-flag ::text').get()
                fight['result'] = self._parse_result(result_text)
                
                # Column 1: Fighters (contains both fighter links)
                fighter_links = cells[1].css('a::attr(href)').getall()
                fighter_names = [self._clean_text(name) for name in cells[1].css('a::text').getall()]
                
                # Find opponent (the one that's not current fighter)
                for i, name in enumerate(fighter_names):
                    if name and name.lower() != fighter['name'].lower():
                        fight['opponent_name'] = name
                        if i < len(fighter_links):
                            fight['opponent_url'] = fighter_links[i]
                        break
                
                # Set winner_name based on result
                if fight['result'] == 'win':
                    fight['winner_name'] = fighter['name']
                elif fight['result'] == 'loss':
                    fight['winner_name'] = fight.get('opponent_name')
                # For draws/NC, winner_name stays None
                
                # Column 2: Knockdowns
                kd_values = cells[2].css('p::text').getall()
                fight['knockdowns'] = self._clean_text(kd_values[0]) if kd_values else None
                
                # Column 3: Strikes
                str_values = cells[3].css('p::text').getall()
                fight['sig_strikes'] = self._clean_text(str_values[0]) if str_values else None
                
                # Column 4: Takedowns
                td_values = cells[4].css('p::text').getall()
                fight['takedowns'] = self._clean_text(td_values[0]) if td_values else None
                
                # Column 5: Submissions
                sub_values = cells[5].css('p::text').getall()
                fight['sub_attempts'] = self._clean_text(sub_values[0]) if sub_values else None
                
                # Column 6: Event (contains link and date)
                event_link = cells[6].css('a::attr(href)').get()
                event_name = self._clean_text(cells[6].css('a::text').get())
                fight['event_url'] = event_link
                fight['event_name'] = event_name
                
                # Date is in a separate element
                date_text = cells[6].css('p.b-fight-details__table-text::text').getall()
                if date_text:
                    fight_date_str = self._clean_text(date_text[-1])
                    fight['date'] = fight_date_str
                    
                    # Track most recent fight date
                    fight_date = self._parse_fight_date(fight_date_str)
                    if fight_date and (most_recent_fight_date is None or fight_date > most_recent_fight_date):
                        most_recent_fight_date = fight_date
                
                # Check for title fight (belt icon)
                if cells[6].css('img[src*="belt"]'):
                    fight['is_title_fight'] = True
                else:
                    fight['is_title_fight'] = False
                
                # Column 7: Method
                method_text = cells[7].css('p.b-fight-details__table-text::text').getall()
                if method_text:
                    fight['method'] = self._clean_text(method_text[0])
                    if len(method_text) > 1:
                        fight['method_detail'] = self._clean_text(method_text[1])
                
                # Column 8: Round
                round_text = cells[8].css('p::text').get() if len(cells) > 8 else None
                fight['round'] = self._parse_int(round_text)
                
                # Column 9: Time
                time_text = cells[9].css('p::text').get() if len(cells) > 9 else None
                fight['time'] = self._clean_text(time_text)
            
            # Fight detail link (from first cell usually)
            fight_link = row.css('td a::attr(href)').get()
            if fight_link and 'fight-details' in fight_link:
                fight['fight_url'] = fight_link
            
            fight['scraped_at'] = now
            fight['source'] = 'ufcstats'
            
            fights_to_yield.append(fight)
        
        # === SET LAST FIGHT INFO ON FIGHTER ===
        if most_recent_fight_date:
            fighter['last_fight_date'] = most_recent_fight_date.strftime('%Y-%m-%d')
            days_since = (datetime.now() - most_recent_fight_date).days
            fighter['days_since_last_fight'] = days_since
            fighter['is_active'] = days_since <= (2 * 365)  # Active if fought in last 2 years
        else:
            fighter['is_active'] = False
        
        # Now yield the fighter with complete info
        yield fighter
        
        # === CAREER STATS ===
        stats = CareerStatsItem()
        stats['fighter_name'] = fighter['name']
        stats['fighter_url'] = response.url
        
        # Career statistics section
        stat_boxes = response.css('div.b-list__info-box-left ul.b-list__box-list li.b-list__box-list-item')
        for item in stat_boxes:
            label = self._clean_text(item.css('i.b-list__box-item-title::text').get())
            # Get the text after the label
            all_text = item.css('::text').getall()
            value = self._clean_text(all_text[-1]) if all_text else None
            
            if label:
                label_lower = label.lower().replace(':', '').replace('.', '').strip()
                if 'slpm' in label_lower:
                    stats['slpm'] = self._parse_decimal(value)
                elif 'str acc' in label_lower:
                    stats['str_acc'] = self._parse_percentage(value)
                elif 'sapm' in label_lower:
                    stats['sapm'] = self._parse_decimal(value)
                elif 'str def' in label_lower:
                    stats['str_def'] = self._parse_percentage(value)
                elif 'td avg' in label_lower:
                    stats['td_avg'] = self._parse_decimal(value)
                elif 'td acc' in label_lower:
                    stats['td_acc'] = self._parse_percentage(value)
                elif 'td def' in label_lower:
                    stats['td_def'] = self._parse_percentage(value)
                elif 'sub avg' in label_lower or 'sub. avg' in label_lower:
                    stats['sub_avg'] = self._parse_decimal(value)
        
        stats['scraped_at'] = now
        stats['source'] = 'ufcstats'
        
        yield stats
        
        # === YIELD FIGHTS (already parsed above) ===
        for fight in fights_to_yield:
            yield fight
    
    # === HELPER METHODS ===
    
    def _clean_text(self, text):
        """Remove extra whitespace and newlines."""
        if text:
            return ' '.join(text.split()).strip()
        return None
    
    def _parse_record(self, record_text):
        """Parse 'Record: 27-8-0' into (wins, losses, draws)."""
        if not record_text:
            return None, None, None
        
        match = re.search(r'(\d+)-(\d+)-(\d+)', record_text)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None, None, None
    
    def _parse_result(self, result_text):
        """Parse W/L/D/NC into standard format. Returns None for scheduled fights."""
        if not result_text:
            return None
        
        result_lower = result_text.lower().strip()
        if 'win' in result_lower or result_lower == 'w':
            return 'win'
        elif 'loss' in result_lower or result_lower == 'l':
            return 'loss'
        elif 'draw' in result_lower or result_lower == 'd':
            return 'draw'
        elif 'nc' in result_lower:
            return 'nc'
        elif 'next' in result_lower or 'scheduled' in result_lower:
            # Upcoming fight - not yet completed
            return None
        return result_text
    
    def _parse_decimal(self, value):
        """Parse a decimal value like '3.83'."""
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    
    def _parse_percentage(self, value):
        """Parse percentage like '47%' into decimal (47.0)."""
        if not value:
            return None
        try:
            # Remove % sign and convert
            clean = value.replace('%', '').strip()
            return float(clean)
        except ValueError:
            return None
    
    def _parse_int(self, value):
        """Parse integer value."""
        if not value:
            return None
        try:
            return int(self._clean_text(value))
        except (ValueError, TypeError):
            return None
    
    def _parse_fight_date(self, date_string):
        """Parse fight date string to datetime."""
        if not date_string or date_string == '--':
            return None
        
        date_formats = [
            "%b. %d, %Y",     # "Dec. 30, 1994"
            "%b %d, %Y",      # "Dec 30, 1994"
            "%B %d, %Y",      # "December 30, 1994"
            "%Y-%m-%d",       # "1994-12-30"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue
        
        return None

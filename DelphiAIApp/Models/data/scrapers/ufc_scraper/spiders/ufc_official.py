"""
UFC.com Official Website Spider (with UFCStats Merge)

Crawls UFC.com to get fighter data, then also looks up each fighter
on UFCStats to fill in any missing fields.

Data priority:
- UFC.com is primary for: weight_class, nickname, place_of_birth, leg_reach, stance
- UFCStats fills in: dob, height (if missing), weight (if missing), reach (if missing)
- Fight history and detailed stats come from UFCStats

This spider produces COMPLETE fighter records by combining both sources.

Usage:
    cd DelphiAIApp/Models/data/scrapers
    scrapy crawl ufc_official
"""

import re
import json
from datetime import datetime
from urllib.parse import urljoin, quote

import scrapy
from ufc_scraper.items import FighterItem, CareerStatsItem, FightItem


class UFCOfficialSpider(scrapy.Spider):
    name = "ufc_official"
    allowed_domains = ["ufc.com", "ufcstats.com"]
    
    # Base URL for athlete listings
    base_url = "https://www.ufc.com/athletes/all"
    ufcstats_base = "http://www.ufcstats.com"
    
    # Custom settings
    custom_settings = {
        'DOWNLOAD_DELAY': 1.5,
        'CONCURRENT_REQUESTS': 2,
        'ROBOTSTXT_OBEY': False,  # UFC.com blocks pagination URLs in robots.txt, but data is public
    }
    
    def start_requests(self):
        """
        Start with the athletes listing page.
        """
        yield scrapy.Request(
            self.base_url,
            callback=self.parse_athlete_list,
            meta={'page': 0}
        )
    
    def parse_athlete_list(self, response):
        """
        Parse the athlete listing page and extract links to individual profiles.
        Also handles pagination via "Load More" which uses AJAX.
        """
        # Extract athlete profile links from the current page
        athlete_links = response.css('a[href*="/athlete/"]::attr(href)').getall()
        
        # Deduplicate and filter
        unique_links = set()
        for link in athlete_links:
            if '/athlete/' in link and link not in unique_links:
                full_url = urljoin(response.url, link)
                unique_links.add(full_url)
        
        self.logger.info(f"Found {len(unique_links)} athletes on page {response.meta.get('page', 0)}")
        
        for link in unique_links:
            yield scrapy.Request(link, callback=self.parse_athlete_profile)
        
        # Check for pagination - UFC.com uses query parameter ?page=N
        current_page = response.meta.get('page', 0)
        
        # Only continue if we found athletes on this page
        if len(unique_links) > 0 and current_page < 500:  # Safety limit of 500 pages (~5500 fighters max)
            next_page = current_page + 1
            next_url = f"{self.base_url}?page={next_page}"
            
            self.logger.info(f"Requesting next page: {next_page}")
            
            yield scrapy.Request(
                next_url,
                callback=self.parse_athlete_list,
                meta={'page': next_page},
                dont_filter=True  # Ensure pagination requests aren't filtered
            )
    
    def parse_athlete_profile(self, response):
        """
        Parse individual athlete profile page.
        Yields FighterItem with supplementary data.
        """
        now = datetime.utcnow().isoformat()
        
        fighter = FighterItem()
        fighter['source'] = 'ufc_official'
        fighter['scraped_at'] = now
        fighter['fighter_url'] = response.url  # This is the UFC.com URL
        
        # Name - from the page title/header
        name = response.css('h1.hero-profile__name::text').get()
        if not name:
            name = response.css('.hero-profile__name::text').get()
        if not name:
            # Try meta title
            name = response.css('title::text').get()
            if name:
                name = name.split('|')[0].strip()
        
        if name:
            fighter['name'] = self._clean_text(name)
        else:
            self.logger.warning(f"Could not find name on {response.url}")
            return
        
        # Nickname
        nickname = response.css('.hero-profile__nickname::text').get()
        if nickname:
            fighter['nickname'] = self._clean_text(nickname).strip('"')
        
        # Weight class / Division
        weight_class = response.css('.hero-profile__division-title::text').get()
        if not weight_class:
            weight_class = response.css('.hero-profile__division::text').get()
        if weight_class:
            fighter['weight_class'] = self._normalize_weight_class(weight_class)
        
        # Record (W-L-D)
        record = response.css('.hero-profile__division-body::text').get()
        if record:
            wins, losses, draws = self._parse_record(record)
            fighter['wins'] = wins
            fighter['losses'] = losses
            fighter['draws'] = draws
            if all(v is not None for v in [wins, losses, draws]):
                fighter['total_fights'] = wins + losses + draws
        
        # Bio section - UFC.com has multiple possible structures
        # Try the standard c-bio structure first
        bio_fields = response.css('.c-bio__field')
        
        for field in bio_fields:
            label = self._clean_text(field.css('.c-bio__label::text').get())
            value = self._clean_text(field.css('.c-bio__text::text').get())
            
            if not label or not value:
                continue
            
            self._parse_bio_field(fighter, label, value)
        
        # Try the Info section table format (newer UFC.com layout)
        # This appears as definition list style or table format
        info_rows = response.css('.c-bio__row, .athlete-info__item, dl.c-bio dt, .field--name-field-athlete-bio')
        
        for row in info_rows:
            label = self._clean_text(row.css('.c-bio__label::text, dt::text, .field__label::text').get())
            value = self._clean_text(row.css('.c-bio__text::text, dd::text, .field__item::text').get())
            if label and value:
                self._parse_bio_field(fighter, label, value)
        
        # Try extracting from the Info tab content (structured as key-value pairs)
        # The bio appears in a tab section on UFC.com - search more broadly
        bio_text = response.css('.athlete-info, .c-bio, [class*="athlete-bio"], [class*="info"]').get()
        
        # If no bio section found, try the entire body
        if not bio_text:
            bio_text = response.text
        
        if bio_text:
            # Extract key-value pairs using regex patterns
            # These patterns handle various formats like "Age42" or "Age: 42" or "Age 42"
            patterns = [
                (r'Status\s*[:\-]?\s*(\w+)', 'status'),
                (r'Place of Birth\s*[:\-]?\s*([^<\n,]+(?:,\s*[^<\n]+)?)', 'place_of_birth'),
                (r'(?<![A-Za-z])Age\s*[:\-]?\s*(\d+)', 'age'),
                (r'(?<![A-Za-z])Height\s*[:\-]?\s*([\d.]+)', 'height'),
                (r'(?<![A-Za-z])Weight\s*[:\-]?\s*([\d.]+)', 'weight'),
                (r'(?<!Leg\s)(?<![A-Za-z])Reach\s*[:\-]?\s*([\d.]+)', 'reach'),
                (r'Leg\s*[Rr]each\s*[:\-]?\s*([\d.]+)', 'leg_reach'),
                (r'Fighting style\s*[:\-]?\s*([^<\n]+)', 'stance'),
            ]
            for pattern, field_name in patterns:
                match = re.search(pattern, bio_text, re.IGNORECASE)
                if match and not fighter.get(field_name):
                    value = match.group(1).strip()
                    # Clean HTML tags if present
                    value = re.sub(r'<[^>]+>', '', value).strip()
                    if not value:
                        continue
                    if field_name == 'status':
                        fighter['is_active'] = value.lower() == 'active'
                    elif field_name == 'age':
                        try:
                            fighter['age'] = int(value)
                        except ValueError:
                            pass
                    elif field_name == 'stance':
                        fighter['stance'] = value
                    else:
                        fighter[field_name] = value
        
        # Also try to get data from the page's JSON-LD or data attributes
        json_ld = response.css('script[type="application/ld+json"]::text').get()
        if json_ld:
            try:
                data = json.loads(json_ld)
                if isinstance(data, dict):
                    if data.get('birthPlace') and not fighter.get('place_of_birth'):
                        fighter['place_of_birth'] = data['birthPlace']
                    if data.get('height') and not fighter.get('height'):
                        fighter['height'] = str(data['height'])
                    if data.get('weight') and not fighter.get('weight'):
                        fighter['weight'] = str(data['weight'])
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Check for active status from hero section
        if fighter.get('is_active') is None:
            status_text = response.css('.hero-profile__tag::text').get()
            if status_text:
                fighter['is_active'] = 'active' in status_text.lower()
        
        # Extract average fight time from UFC.com
        # UFC.com format: <div class="c-stat-compare__number">09:27</div> followed by
        #                 <div class="c-stat-compare__label">Average fight time</div>
        avg_fight_time_match = re.search(
            r'c-stat-compare__number[^>]*>\s*(\d{1,2}:\d{2})\s*</div>\s*<div[^>]*c-stat-compare__label[^>]*>\s*Average\s*fight\s*time',
            response.text, re.IGNORECASE | re.DOTALL
        )
        if not avg_fight_time_match:
            # Fallback: Try simpler pattern in case of format variations
            avg_fight_time_match = re.search(r'(\d{1,2}:\d{2})\s*</[^>]+>\s*<[^>]+>\s*Average\s*fight\s*time', response.text, re.IGNORECASE)
        if not avg_fight_time_match:
            # Another fallback: Label before time
            avg_fight_time_match = re.search(r'Average\s*fight\s*time\s*[:\-]?\s*(\d{1,2}:\d{2})', response.text, re.IGNORECASE)
        
        if avg_fight_time_match:
            time_str = avg_fight_time_match.group(1)
            try:
                minutes, seconds = map(int, time_str.split(':'))
                # Convert to decimal minutes for database
                fighter['avg_fight_duration'] = round(minutes + seconds / 60, 2)
                self.logger.info(f"Scraped avg fight time from UFC.com for {fighter['name']}: {time_str} -> {fighter['avg_fight_duration']} min")
            except (ValueError, AttributeError):
                pass
        
        # Store UFC.com URL before we potentially overwrite fighter_url
        fighter['ufc_url'] = response.url
        
        # === NOW LOOK UP THIS FIGHTER ON UFCSTATS ===
        # UFCStats lists fighters by first name initial on the page
        # We search by first name initial, then match on full name
        name = fighter.get('name', '')
        if name:
            name_parts = name.split()
            if name_parts:
                # Get first name initial for search
                first_name = name_parts[0]
                first_char = first_name[0].lower()
                
                # Also get last name initial as fallback
                last_name = name_parts[-1] if len(name_parts) > 1 else first_name
                last_char = last_name[0].lower()
                
                # Request the UFCStats fighter list for first name initial
                ufcstats_url = f"{self.ufcstats_base}/statistics/fighters?char={first_char}&page=all"
                
                yield scrapy.Request(
                    ufcstats_url,
                    callback=self.find_fighter_on_ufcstats,
                    meta={
                        'ufc_fighter': dict(fighter),
                        'fighter_name': name,
                        'first_char': first_char,
                        'last_char': last_char,
                        'tried_last_name': False,
                    },
                    dont_filter=True,
                )
        else:
            # No name, just yield what we have
            yield fighter
        
        # Note: Career stats will be collected from UFCStats merge
        # UFC.com stats like wins by KO/SUB are captured in the merged data
    
    # === UFCSTATS LOOKUP METHODS ===
    
    def find_fighter_on_ufcstats(self, response):
        """
        Search UFCStats fighter listing page to find the fighter.
        Matches on both first AND last name.
        If not found, tries searching by last name initial as fallback.
        """
        ufc_fighter = response.meta['ufc_fighter']
        fighter_name = response.meta['fighter_name']
        first_char = response.meta.get('first_char', '')
        last_char = response.meta.get('last_char', '')
        tried_last_name = response.meta.get('tried_last_name', False)
        
        # Parse target name into first and last
        target_name = self._normalize_name_for_match(fighter_name)
        target_parts = target_name.split()
        
        if len(target_parts) < 2:
            # Single name, harder to match
            target_first = target_parts[0] if target_parts else ''
            target_last = target_first
        else:
            target_first = target_parts[0]
            target_last = target_parts[-1]
        
        # Find all fighter links on the page
        # UFCStats table has first name in first column, last name in second
        fighter_rows = response.css('table.b-statistics__table tbody tr')
        
        best_match_url = None
        best_match_score = 0
        
        for row in fighter_rows:
            # Get all links and names in this row
            links = row.css('td.b-statistics__table-col a::attr(href)').getall()
            names = row.css('td.b-statistics__table-col a::text').getall()
            
            if not links or not names:
                continue
            
            # UFCStats format: First name link, Last name link (both point to same page)
            fighter_link = None
            row_first = ''
            row_last = ''
            
            for i, link in enumerate(links):
                if link and 'fighter-details' in link:
                    fighter_link = link
                    if i < len(names):
                        row_first = self._normalize_name_for_match(names[i])
                    if i + 1 < len(names):
                        row_last = self._normalize_name_for_match(names[i + 1])
                    break
            
            if not fighter_link:
                continue
            
            # Also try parsing full name from single column
            if not row_last and names:
                full_name = self._normalize_name_for_match(' '.join(names[:2]))
                full_parts = full_name.split()
                if len(full_parts) >= 2:
                    row_first = full_parts[0]
                    row_last = full_parts[-1]
                elif full_parts:
                    row_first = full_parts[0]
                    row_last = row_first
            
            # Check for match: BOTH first AND last name MUST match (with fuzzy logic)
            first_match = self._names_match_fuzzy(target_first, row_first)
            last_match = self._names_match_fuzzy(target_last, row_last)
            
            # Also check exact match for higher priority
            exact_first = (target_first == row_first)
            exact_last = (target_last == row_last)
            
            if exact_first and exact_last:
                # Exact match on both - highest priority
                best_match_url = fighter_link
                best_match_score = 100
                self.logger.info(f"Found exact match: {row_first} {row_last} -> {fighter_link}")
                break
            elif first_match and last_match:
                # Fuzzy match on both - good enough
                if best_match_score < 90:
                    best_match_url = fighter_link
                    best_match_score = 90
                    self.logger.info(f"Found fuzzy match: '{target_first} {target_last}' ~ '{row_first} {row_last}' -> {fighter_link}")
            
            # Log near misses for debugging
            if (first_match or last_match) and not (first_match and last_match):
                self.logger.debug(f"Partial match skipped: target='{target_first} {target_last}', found='{row_first} {row_last}'")
        
        if best_match_url and best_match_score >= 90:
            # Found the fighter, now get their detail page
            yield scrapy.Request(
                best_match_url,
                callback=self.parse_ufcstats_and_merge,
                meta={'ufc_fighter': ufc_fighter},
                dont_filter=True,
            )
        elif not tried_last_name and last_char != first_char:
            # Try searching by last name initial as fallback
            ufcstats_url = f"{self.ufcstats_base}/statistics/fighters?char={last_char}&page=all"
            yield scrapy.Request(
                ufcstats_url,
                callback=self.find_fighter_on_ufcstats,
                meta={
                    'ufc_fighter': ufc_fighter,
                    'fighter_name': fighter_name,
                    'first_char': first_char,
                    'last_char': last_char,
                    'tried_last_name': True,
                },
                dont_filter=True,
            )
        else:
            # Fighter not found on UFCStats after trying both, yield UFC.com data only
            self.logger.info(f"Fighter not found on UFCStats: {fighter_name}")
            fighter = FighterItem(**ufc_fighter)
            fighter['source'] = 'ufc_official'
            yield fighter
    
    def parse_ufcstats_and_merge(self, response):
        """
        Parse UFCStats fighter page and merge with UFC.com data.
        UFC.com data takes priority, UFCStats fills in missing fields.
        """
        ufc_fighter = response.meta['ufc_fighter']
        now = datetime.utcnow().isoformat()
        
        # Create the final merged fighter item
        fighter = FighterItem()
        
        # Start with UFC.com data (priority)
        for key, value in ufc_fighter.items():
            if value is not None and value != '':
                fighter[key] = value
        
        # Override source to indicate merged data
        fighter['source'] = 'merged'
        fighter['scraped_at'] = now
        
        # Get UFCStats URL
        ufcstats_url = response.url
        fighter['fighter_url'] = ufcstats_url  # Use UFCStats URL as primary
        
        # === PARSE UFCSTATS DATA ===
        
        # Record (overwrite with UFCStats data as it's more accurate)
        record_text = self._clean_text(response.css('span.b-content__title-record::text').get())
        if record_text:
            wins, losses, draws = self._parse_record(record_text)
            fighter['wins'] = wins
            fighter['losses'] = losses
            fighter['draws'] = draws
            fighter['total_fights'] = (wins or 0) + (losses or 0) + (draws or 0) if any([wins, losses, draws]) else None
        
        # Bio details from UFCStats - fill in missing fields only
        bio_items = response.css('ul.b-list__box-list li.b-list__box-list-item')
        for item in bio_items:
            label = self._clean_text(item.css('i.b-list__box-item-title::text').get())
            all_text = item.css('::text').getall()
            value = self._clean_text(all_text[-1]) if all_text else None
            
            if label and value:
                label_lower = label.lower().replace(':', '').strip()
                
                # Only fill if UFC.com doesn't have the value
                if 'height' in label_lower and not fighter.get('height'):
                    fighter['height'] = value
                elif 'weight' in label_lower and not fighter.get('weight'):
                    fighter['weight'] = value
                elif 'reach' in label_lower and not fighter.get('reach'):
                    fighter['reach'] = value
                elif 'stance' in label_lower and not fighter.get('stance'):
                    fighter['stance'] = value
                elif 'dob' in label_lower:
                    # DOB only comes from UFCStats
                    fighter['dob'] = value
        
        # === PARSE FIGHT HISTORY FOR LAST FIGHT DATE ===
        fight_rows = response.css('table.b-fight-details__table tbody tr.b-fight-details__table-row')
        most_recent_fight_date = None
        fights_to_yield = []
        
        for row in fight_rows:
            if row.css('th'):
                continue
            
            cells = row.css('td.b-fight-details__table-col')
            if len(cells) >= 7:
                # Get fight date
                date_text = cells[6].css('p.b-fight-details__table-text::text').getall()
                if date_text:
                    fight_date_str = self._clean_text(date_text[-1])
                    fight_date = self._parse_fight_date(fight_date_str)
                    
                    if fight_date and (most_recent_fight_date is None or fight_date > most_recent_fight_date):
                        most_recent_fight_date = fight_date
                
                # Also create FightItem
                fight = FightItem()
                fight['fighter_name'] = fighter.get('name')
                fight['fighter_url'] = ufcstats_url
                
                # Result
                result_text = cells[0].css('i.b-flag__text::text').get()
                fight['result'] = self._parse_result(result_text)
                
                # Opponent
                fighter_names = [self._clean_text(n) for n in cells[1].css('a::text').getall()]
                fighter_links = cells[1].css('a::attr(href)').getall()
                
                for i, name in enumerate(fighter_names):
                    if name and name.lower() != fighter.get('name', '').lower():
                        fight['opponent_name'] = name
                        if i < len(fighter_links):
                            fight['opponent_url'] = fighter_links[i]
                        break
                
                # Winner
                if fight.get('result') == 'win':
                    fight['winner_name'] = fighter.get('name')
                elif fight.get('result') == 'loss':
                    fight['winner_name'] = fight.get('opponent_name')
                
                # Event info
                fight['event_name'] = self._clean_text(cells[6].css('a::text').get())
                fight['event_url'] = cells[6].css('a::attr(href)').get()
                if date_text:
                    fight['date'] = self._clean_text(date_text[-1])
                
                # Method
                method_text = cells[7].css('p.b-fight-details__table-text::text').getall() if len(cells) > 7 else []
                if method_text:
                    fight['method'] = self._clean_text(method_text[0])
                    if len(method_text) > 1:
                        fight['method_detail'] = self._clean_text(method_text[1])
                
                # Round and Time
                if len(cells) > 8:
                    fight['round'] = self._parse_int(cells[8].css('p::text').get())
                if len(cells) > 9:
                    fight['time'] = self._clean_text(cells[9].css('p::text').get())
                
                # Stats
                fight['knockdowns'] = self._clean_text(cells[2].css('p::text').get()) if len(cells) > 2 else None
                fight['sig_strikes'] = self._clean_text(cells[3].css('p::text').get()) if len(cells) > 3 else None
                fight['takedowns'] = self._clean_text(cells[4].css('p::text').get()) if len(cells) > 4 else None
                fight['sub_attempts'] = self._clean_text(cells[5].css('p::text').get()) if len(cells) > 5 else None
                
                # Title fight check
                fight['is_title_fight'] = bool(cells[6].css('img[src*="belt"]'))
                
                # Fight URL
                fight_link = row.css('td a::attr(href)').get()
                if fight_link and 'fight-details' in fight_link:
                    fight['fight_url'] = fight_link
                
                fight['scraped_at'] = now
                fight['source'] = 'merged'
                
                if fight.get('result'):  # Only include completed fights
                    fights_to_yield.append(fight)
        
        # Set last fight info
        if most_recent_fight_date:
            fighter['last_fight_date'] = most_recent_fight_date.strftime('%Y-%m-%d')
            days_since = (datetime.now() - most_recent_fight_date).days
            fighter['days_since_last_fight'] = days_since
            
            # Update is_active based on recent fights (if not already set by UFC.com)
            if fighter.get('is_active') is None:
                fighter['is_active'] = days_since <= (2 * 365)
        
        # Yield the merged fighter
        yield fighter
        
        # === CAREER STATS FROM UFCSTATS ===
        stats = CareerStatsItem()
        stats['fighter_name'] = fighter.get('name')
        stats['fighter_url'] = ufcstats_url
        stats['source'] = 'merged'
        stats['scraped_at'] = now
        
        # Use UFC.com avg fight duration if available (scraped earlier)
        ufc_avg_duration = ufc_fighter.get('avg_fight_duration')
        if ufc_avg_duration:
            stats['avg_fight_duration'] = ufc_avg_duration
        
        # Parse career stats from UFCStats
        stat_boxes = response.css('div.b-list__info-box-left ul.b-list__box-list li.b-list__box-list-item')
        for item in stat_boxes:
            label = self._clean_text(item.css('i.b-list__box-item-title::text').get())
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
        
        yield stats
        
        # Yield all fights
        for fight in fights_to_yield:
            yield fight
    
    def _normalize_name_for_match(self, name):
        """Normalize a name for matching between sources."""
        if not name:
            return ''
        
        import unicodedata
        
        # Normalize unicode
        name = unicodedata.normalize('NFKD', name)
        name = ''.join(c for c in name if not unicodedata.combining(c))
        
        # Lowercase, strip, remove extra spaces
        name = ' '.join(name.lower().split())
        
        return name
    
    def _names_match_fuzzy(self, name1, name2, threshold=0.85):
        """
        Check if two names match with fuzzy logic.
        Handles spelling variations like Abbassov/Abbasov.
        Returns True if names are similar enough.
        """
        if not name1 or not name2:
            return False
        
        # Exact match
        if name1 == name2:
            return True
        
        # One is substring of other (handles cases like "Jr" suffix)
        if name1 in name2 or name2 in name1:
            return True
        
        # Calculate simple similarity ratio
        # Count matching characters
        len1, len2 = len(name1), len(name2)
        if abs(len1 - len2) > 2:
            # Names differ by more than 2 characters in length
            return False
        
        # Simple Levenshtein-like check: count differences
        shorter, longer = (name1, name2) if len1 <= len2 else (name2, name1)
        differences = 0
        j = 0
        for i, c in enumerate(shorter):
            if j < len(longer):
                if c != longer[j]:
                    differences += 1
                    # Try to realign if there's an extra character
                    if j + 1 < len(longer) and c == longer[j + 1]:
                        j += 1
                j += 1
        
        # Add any remaining characters in longer string
        differences += len(longer) - j
        
        # Allow 1-2 character differences for names
        max_diff = 2 if len(shorter) > 4 else 1
        return differences <= max_diff
    
    def _parse_result(self, result_text):
        """Parse W/L/D/NC into standard format."""
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
        return None
    
    def _parse_decimal(self, value):
        """Parse a decimal value."""
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    
    def _parse_percentage(self, value):
        """Parse percentage like '47%' into decimal."""
        if not value:
            return None
        try:
            return float(value.replace('%', '').strip())
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
            "%b. %d, %Y",
            "%b %d, %Y",
            "%B %d, %Y",
            "%Y-%m-%d",
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue
        return None
    
    # === HELPER METHODS ===
    
    def _parse_bio_field(self, fighter, label, value):
        """Parse a bio field and set the appropriate fighter attribute."""
        if not label or not value:
            return
        
        label_lower = label.lower()
        
        if 'place of birth' in label_lower or 'hometown' in label_lower:
            if not fighter.get('place_of_birth'):
                fighter['place_of_birth'] = value
        elif 'age' in label_lower and 'average' not in label_lower:
            if not fighter.get('age'):
                try:
                    fighter['age'] = int(value.split()[0])  # Handle "42 years" format
                except (ValueError, IndexError):
                    pass
        elif 'height' in label_lower:
            if not fighter.get('height'):
                fighter['height'] = value
        elif 'weight' in label_lower:
            if not fighter.get('weight'):
                fighter['weight'] = value
        elif 'leg reach' in label_lower or 'leg_reach' in label_lower:
            if not fighter.get('leg_reach'):
                fighter['leg_reach'] = value
        elif 'reach' in label_lower:
            if not fighter.get('reach'):
                fighter['reach'] = value
        elif 'status' in label_lower:
            if fighter.get('is_active') is None:
                fighter['is_active'] = value.lower() == 'active'
        elif 'fighting style' in label_lower or 'style' in label_lower:
            if not fighter.get('stance'):
                fighter['stance'] = value
    
    def _clean_text(self, text):
        """Remove extra whitespace and newlines."""
        if text:
            return ' '.join(text.split()).strip()
        return None
    
    def _parse_record(self, record_text):
        """Parse 'W-L-D (W-L-D)' or '27-8-0' into (wins, losses, draws)."""
        if not record_text:
            return None, None, None
        
        match = re.search(r'(\d+)-(\d+)-(\d+)', record_text)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None, None, None
    
    def _normalize_weight_class(self, weight_class):
        """
        Normalize weight class names to standard format.
        """
        if not weight_class:
            return None
        
        weight_class = self._clean_text(weight_class)
        
        # Remove "Division" suffix if present
        weight_class = re.sub(r'\s*Division\s*$', '', weight_class, flags=re.IGNORECASE)
        
        # Normalize common variations
        weight_class_map = {
            'strawweight': 'Strawweight',
            "women's strawweight": "Women's Strawweight",
            'flyweight': 'Flyweight',
            "women's flyweight": "Women's Flyweight",
            'bantamweight': 'Bantamweight',
            "women's bantamweight": "Women's Bantamweight",
            'featherweight': 'Featherweight',
            "women's featherweight": "Women's Featherweight",
            'lightweight': 'Lightweight',
            'welterweight': 'Welterweight',
            'middleweight': 'Middleweight',
            'light heavyweight': 'Light Heavyweight',
            'heavyweight': 'Heavyweight',
            'catch weight': 'Catch Weight',
            'super heavyweight': 'Super Heavyweight',
        }
        
        normalized = weight_class_map.get(weight_class.lower(), weight_class)
        return normalized
    
    def _parse_fight_time(self, time_str):
        """
        Parse fight time string (MM:SS) to minutes as decimal.
        E.g., "11:01" -> 11.02
        """
        if not time_str:
            return None
        
        try:
            parts = time_str.strip().split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return round(minutes + seconds / 60, 2)
        except (ValueError, TypeError):
            pass
        
        return None

"""
Scrapy Items for UFC Fighter Data

These items map to the database schema defined in schemas.sql.
Raw scraped data is stored here, then cleaned/transformed before DB import.
"""

import scrapy


class FighterItem(scrapy.Item):
    """
    Fighter basic information - maps to FighterStats table.
    
    Scraped from UFCStats fighter detail pages.
    """
    # Core identification
    name = scrapy.Field()
    fighter_url = scrapy.Field()  # Source URL for reference
    
    # Physical attributes
    height = scrapy.Field()       # Raw: "5' 11\"" -> cleaned later
    weight = scrapy.Field()       # Raw: "145 lbs" -> cleaned later
    reach = scrapy.Field()        # Raw: "72\"" -> cleaned later
    stance = scrapy.Field()       # Orthodox, Southpaw, Switch
    dob = scrapy.Field()          # Raw: "Dec 30, 1994" -> DATE
    
    # Computed fields (calculated during cleaning/feature engineering)
    age = scrapy.Field()
    weight_class = scrapy.Field()
    days_since_last_fight = scrapy.Field()  # Ring rust indicator
    last_fight_date = scrapy.Field()        # Most recent fight date
    is_active = scrapy.Field()              # Fought within last 2 years
    
    # UFC.com supplementary fields
    nickname = scrapy.Field()               # Fighter nickname/alias
    place_of_birth = scrapy.Field()         # Hometown/birthplace
    leg_reach = scrapy.Field()              # Leg reach (UFC.com has this)
    ufc_url = scrapy.Field()                # UFC.com profile URL
    avg_fight_duration = scrapy.Field()     # Average fight duration from UFC.com
    
    # Record
    wins = scrapy.Field()
    losses = scrapy.Field()
    draws = scrapy.Field()
    total_fights = scrapy.Field()
    
    # Metadata
    scraped_at = scrapy.Field()
    source = scrapy.Field()       # 'ufcstats', 'ufc_official', etc.


class CareerStatsItem(scrapy.Item):
    """
    Career statistics - maps to CareerStats table.
    
    Scraped from UFCStats fighter detail pages.
    Note: Percentages stored as decimals (47% -> 47.0 or 0.47 depending on cleaning)
    """
    # Link to fighter
    fighter_name = scrapy.Field()  # Used to match with FighterItem
    fighter_url = scrapy.Field()
    
    # Striking stats (per minute)
    slpm = scrapy.Field()          # Significant Strikes Landed per Minute
    str_acc = scrapy.Field()       # Strike Accuracy (percentage)
    sapm = scrapy.Field()          # Significant Strikes Absorbed per Minute
    str_def = scrapy.Field()       # Strike Defense (percentage)
    
    # Grappling stats
    td_avg = scrapy.Field()        # Takedown Average (per 15 min)
    td_acc = scrapy.Field()        # Takedown Accuracy (percentage)
    td_def = scrapy.Field()        # Takedown Defense (percentage)
    sub_avg = scrapy.Field()       # Submission Average (per 15 min)
    
    # Derived stats (computed during feature engineering)
    win_streak_last3 = scrapy.Field()
    wins_by_ko_last5 = scrapy.Field()
    wins_by_sub_last5 = scrapy.Field()
    avg_fight_duration = scrapy.Field()
    first_round_finish_rate = scrapy.Field()
    decision_rate = scrapy.Field()
    ko_round1_pct = scrapy.Field()
    ko_round2_pct = scrapy.Field()
    ko_round3_pct = scrapy.Field()
    sub_round1_pct = scrapy.Field()
    sub_round2_pct = scrapy.Field()
    sub_round3_pct = scrapy.Field()
    elo_rating = scrapy.Field()
    
    # Metadata
    scraped_at = scrapy.Field()
    source = scrapy.Field()


class FightItem(scrapy.Item):
    """
    Individual fight record - maps to Fights table.
    
    Scraped from UFCStats fighter detail pages (fight history table).
    """
    # Fighter info
    fighter_name = scrapy.Field()
    fighter_url = scrapy.Field()
    
    # Opponent info
    opponent_name = scrapy.Field()
    opponent_url = scrapy.Field()
    
    # Fight result
    result = scrapy.Field()        # 'win', 'loss', 'draw', 'nc'
    winner_name = scrapy.Field()   # Name of winner (for verification)
    
    # Fight details
    event_name = scrapy.Field()
    event_url = scrapy.Field()
    date = scrapy.Field()          # Raw: "Jan. 31, 2026" -> DATE
    method = scrapy.Field()        # "KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"
    method_detail = scrapy.Field() # "Punches", "Rear Naked Choke", etc.
    round = scrapy.Field()         # 1-5
    time = scrapy.Field()          # "2:34" format
    
    # Fight stats (if available from detail page)
    knockdowns = scrapy.Field()
    sig_strikes = scrapy.Field()
    takedowns = scrapy.Field()
    sub_attempts = scrapy.Field()
    
    # Flags (computed or manual)
    is_title_fight = scrapy.Field()
    is_main_event = scrapy.Field()
    fluke_flag = scrapy.Field()    # Manual flag for upsets/flukes
    
    # Metadata
    fight_url = scrapy.Field()     # Link to detailed fight stats
    scraped_at = scrapy.Field()
    source = scrapy.Field()


class EventItem(scrapy.Item):
    """
    UFC Event information - useful for crawling all fighters.
    """
    event_name = scrapy.Field()
    event_url = scrapy.Field()
    date = scrapy.Field()
    location = scrapy.Field()
    
    # Metadata
    scraped_at = scrapy.Field()
    source = scrapy.Field()

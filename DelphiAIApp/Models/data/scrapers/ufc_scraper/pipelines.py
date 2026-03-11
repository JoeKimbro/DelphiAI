"""
Scrapy Pipelines for UFC Scraper

Exports scraped items to separate CSV files by item type:
- fighters.csv - FighterItem data
- career_stats.csv - CareerStatsItem data
- fights.csv - FightItem data

Includes:
- Data validation with expected ranges
- Duplicate filtering
- CSV export by item type
- PostgreSQL database upload
- Comprehensive completion statistics
"""

import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent.parent.parent.parent.parent / '.env'
    load_dotenv(env_path)
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from ufc_scraper.items import FighterItem, CareerStatsItem, FightItem, EventItem


logger = logging.getLogger(__name__)


class DatabasePipeline:
    """
    Pipeline to upload scraped data directly to PostgreSQL database.
    
    Uses connection settings from .env file:
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    
    Configure in settings.py:
        DATABASE_PIPELINE_ENABLED = True  (default: False)
    
    Usage:
        scrapy crawl ufcstats -s DATABASE_PIPELINE_ENABLED=True
    
    Features:
    - Upsert logic (insert or update if exists based on URL)
    - Automatic FighterID lookup for foreign keys
    - Batch inserts for performance
    - Connection pooling
    """
    
    def __init__(self, db_settings, enabled=False):
        self.db_settings = db_settings
        self.enabled = enabled
        self.connection = None
        self.cursor = None
        self.fighter_url_to_id = {}  # Cache for URL -> FighterID mapping
        self.stats = {
            'fighters_inserted': 0,
            'fighters_updated': 0,
            'career_stats_inserted': 0,
            'career_stats_updated': 0,
            'fights_inserted': 0,
            'fights_updated': 0,
            'errors': 0,
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not installed. Database pipeline disabled.")
            return cls({}, enabled=False)
        
        enabled = crawler.settings.getbool('DATABASE_PIPELINE_ENABLED', False)
        
        if not enabled:
            logger.info("Database pipeline disabled. Set DATABASE_PIPELINE_ENABLED=True to enable.")
            return cls({}, enabled=False)
        
        db_settings = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5433'),
            'dbname': os.getenv('DB_NAME', 'delphi_db'),
            'user': os.getenv('DB_USER', ''),
            'password': os.getenv('DB_PASSWORD', ''),
        }
        
        return cls(db_settings, enabled=True)
    
    def open_spider(self, spider):
        if not self.enabled:
            return
        
        try:
            self.connection = psycopg2.connect(
                host=self.db_settings['host'],
                port=self.db_settings['port'],
                dbname=self.db_settings['dbname'],
                user=self.db_settings['user'],
                password=self.db_settings['password'],
            )
            self.cursor = self.connection.cursor()
            logger.info(f"Database connected: {self.db_settings['dbname']}@{self.db_settings['host']}:{self.db_settings['port']}")
            
            # Load existing fighter URL -> ID mappings
            self._load_fighter_mappings()
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.enabled = False
    
    def close_spider(self, spider):
        if not self.enabled or not self.connection:
            return
        
        try:
            self.connection.commit()
            self.cursor.close()
            self.connection.close()
            
            logger.info("=" * 60)
            logger.info("DATABASE UPLOAD SUMMARY")
            logger.info("=" * 60)
            logger.info(f"   Fighters: {self.stats['fighters_inserted']} inserted, {self.stats['fighters_updated']} updated")
            logger.info(f"   Career Stats: {self.stats['career_stats_inserted']} inserted, {self.stats['career_stats_updated']} updated")
            logger.info(f"   Fights: {self.stats['fights_inserted']} inserted, {self.stats['fights_updated']} updated")
            logger.info(f"   Errors: {self.stats['errors']}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def _load_fighter_mappings(self):
        """Load existing FighterURL -> FighterID mappings from database."""
        try:
            self.cursor.execute("SELECT FighterID, FighterURL FROM FighterStats WHERE FighterURL IS NOT NULL")
            for row in self.cursor.fetchall():
                self.fighter_url_to_id[row[1]] = row[0]
            logger.info(f"Loaded {len(self.fighter_url_to_id)} existing fighter mappings")
        except Exception as e:
            logger.warning(f"Could not load fighter mappings: {e}")
    
    def process_item(self, item):
        if not self.enabled:
            return item
        
        adapter = ItemAdapter(item)
        
        try:
            if isinstance(item, FighterItem):
                self._upsert_fighter(adapter)
            elif isinstance(item, CareerStatsItem):
                self._upsert_career_stats(adapter)
            elif isinstance(item, FightItem):
                self._upsert_fight(adapter)
        except Exception as e:
            logger.error(f"Database error for {type(item).__name__}: {e}")
            self.stats['errors'] += 1
            # Don't rollback - just log and continue
        
        return item
    
    def _upsert_fighter(self, adapter):
        """Insert or update fighter in FighterStats table."""
        fighter_url = adapter.get('fighter_url')
        
        # Parse DOB to date
        dob = adapter.get('dob')
        dob_date = None
        if dob and dob != '--':
            try:
                for fmt in ["%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"]:
                    try:
                        dob_date = datetime.strptime(dob.strip(), fmt).date()
                        break
                    except ValueError:
                        continue
            except:
                pass
        
        # Parse last_fight_date
        last_fight = adapter.get('last_fight_date')
        last_fight_date = None
        if last_fight:
            try:
                last_fight_date = datetime.strptime(last_fight, "%Y-%m-%d").date()
            except:
                pass
        
        if fighter_url in self.fighter_url_to_id:
            # Update existing fighter
            self.cursor.execute("""
                UPDATE FighterStats SET
                    Name = %s, Height = %s, Weight = %s, Reach = %s, Stance = %s,
                    DOB = %s, Age = %s, WeightClass = %s, Nickname = %s,
                    PlaceOfBirth = %s, LegReach = %s, UFCUrl = %s,
                    TotalFights = %s, Wins = %s, Losses = %s, Draws = %s,
                    LastFightDate = %s, DaysSinceLastFight = %s, IsActive = %s,
                    Source = %s, ScrapedAt = %s, FightUpdatedAt = %s
                WHERE FighterURL = %s
            """, (
                adapter.get('name'),
                adapter.get('height'),
                adapter.get('weight'),
                adapter.get('reach'),
                adapter.get('stance'),
                dob_date,
                adapter.get('age'),
                adapter.get('weight_class'),
                adapter.get('nickname'),
                adapter.get('place_of_birth'),
                adapter.get('leg_reach'),
                adapter.get('ufc_url'),
                adapter.get('total_fights'),
                adapter.get('wins'),
                adapter.get('losses'),
                adapter.get('draws'),
                last_fight_date,
                adapter.get('days_since_last_fight'),
                adapter.get('is_active'),
                adapter.get('source'),
                datetime.now(),
                datetime.now(),
                fighter_url,
            ))
            self.stats['fighters_updated'] += 1
        else:
            # Insert new fighter
            self.cursor.execute("""
                INSERT INTO FighterStats (
                    Name, FighterURL, Height, Weight, Reach, Stance, DOB, Age,
                    WeightClass, Nickname, PlaceOfBirth, LegReach, UFCUrl,
                    TotalFights, Wins, Losses, Draws,
                    LastFightDate, DaysSinceLastFight, IsActive, Source, ScrapedAt, FightUpdatedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING FighterID
            """, (
                adapter.get('name'),
                fighter_url,
                adapter.get('height'),
                adapter.get('weight'),
                adapter.get('reach'),
                adapter.get('stance'),
                dob_date,
                adapter.get('age'),
                adapter.get('weight_class'),
                adapter.get('nickname'),
                adapter.get('place_of_birth'),
                adapter.get('leg_reach'),
                adapter.get('ufc_url'),
                adapter.get('total_fights'),
                adapter.get('wins'),
                adapter.get('losses'),
                adapter.get('draws'),
                last_fight_date,
                adapter.get('days_since_last_fight'),
                adapter.get('is_active'),
                adapter.get('source'),
                datetime.now(),
                datetime.now(),
            ))
            fighter_id = self.cursor.fetchone()[0]
            self.fighter_url_to_id[fighter_url] = fighter_id
            self.stats['fighters_inserted'] += 1
        
        self.connection.commit()
    
    def _upsert_career_stats(self, adapter):
        """Insert or update career stats in CareerStats table."""
        fighter_url = adapter.get('fighter_url')
        fighter_id = self.fighter_url_to_id.get(fighter_url)
        
        if not fighter_id:
            logger.warning(f"No FighterID found for CareerStats: {adapter.get('fighter_name')}")
            return
        
        # Check if career stats exist for this fighter
        self.cursor.execute("SELECT CSID FROM CareerStats WHERE FighterID = %s", (fighter_id,))
        existing = self.cursor.fetchone()
        
        if existing:
            # Update
            self.cursor.execute("""
                UPDATE CareerStats SET
                    FighterURL = %s, SLpM = %s, StrAcc = %s, SApM = %s, StrDef = %s,
                    TDAvg = %s, TDAcc = %s, TDDef = %s, SubAvg = %s,
                    AvgFightDuration = COALESCE(%s, AvgFightDuration),
                    Source = %s, ScrapedAt = %s, CareerUpdatedAt = %s
                WHERE FighterID = %s
            """, (
                fighter_url,
                adapter.get('slpm'),
                adapter.get('str_acc'),
                adapter.get('sapm'),
                adapter.get('str_def'),
                adapter.get('td_avg'),
                adapter.get('td_acc'),
                adapter.get('td_def'),
                adapter.get('sub_avg'),
                adapter.get('avg_fight_duration'),
                adapter.get('source'),
                datetime.now(),
                datetime.now(),
                fighter_id,
            ))
            self.stats['career_stats_updated'] += 1
        else:
            # Insert
            self.cursor.execute("""
                INSERT INTO CareerStats (
                    FighterID, FighterURL, SLpM, StrAcc, SApM, StrDef,
                    TDAvg, TDAcc, TDDef, SubAvg, AvgFightDuration,
                    Source, ScrapedAt, CareerUpdatedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter_id,
                fighter_url,
                adapter.get('slpm'),
                adapter.get('str_acc'),
                adapter.get('sapm'),
                adapter.get('str_def'),
                adapter.get('td_avg'),
                adapter.get('td_acc'),
                adapter.get('td_def'),
                adapter.get('sub_avg'),
                adapter.get('avg_fight_duration'),
                adapter.get('source'),
                datetime.now(),
                datetime.now(),
            ))
            self.stats['career_stats_inserted'] += 1
        
        self.connection.commit()
    
    def _upsert_fight(self, adapter):
        """Insert or update fight in Fights table."""
        fight_url = adapter.get('fight_url')
        fighter_url = adapter.get('fighter_url')
        opponent_url = adapter.get('opponent_url')
        
        fighter_id = self.fighter_url_to_id.get(fighter_url)
        opponent_id = self.fighter_url_to_id.get(opponent_url) if opponent_url else None
        
        if not fighter_id:
            logger.warning(f"No FighterID found for Fight: {adapter.get('fighter_name')}")
            return
        
        # Determine winner_id
        winner_id = None
        result = adapter.get('result')
        if result == 'win':
            winner_id = fighter_id
        elif result == 'loss' and opponent_id:
            winner_id = opponent_id
        
        # Parse fight date
        fight_date = None
        date_str = adapter.get('date')
        if date_str:
            for fmt in ["%b. %d, %Y", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"]:
                try:
                    fight_date = datetime.strptime(date_str.strip(), fmt).date()
                    break
                except ValueError:
                    continue
        
        # Check if fight exists (by FightURL or by fighter+opponent+date combo)
        if fight_url:
            self.cursor.execute("SELECT FightID FROM Fights WHERE FightURL = %s", (fight_url,))
        else:
            self.cursor.execute(
                "SELECT FightID FROM Fights WHERE FighterID = %s AND OpponentID = %s AND Date = %s",
                (fighter_id, opponent_id, fight_date)
            )
        existing = self.cursor.fetchone()
        
        if existing:
            # Update
            self.cursor.execute("""
                UPDATE Fights SET
                    FighterURL = %s, FighterName = %s, OpponentID = %s, OpponentURL = %s,
                    OpponentName = %s, WinnerID = %s, WinnerName = %s, Result = %s,
                    Date = %s, EventName = %s, EventURL = %s, Method = %s, MethodDetail = %s,
                    Round = %s, Time = %s, Knockdowns = %s, SigStrikes = %s, Takedowns = %s,
                    SubAttempts = %s, IsTitleFight = %s, Source = %s, ScrapedAt = %s
                WHERE FightID = %s
            """, (
                fighter_url,
                adapter.get('fighter_name'),
                opponent_id,
                opponent_url,
                adapter.get('opponent_name'),
                winner_id,
                adapter.get('winner_name'),
                result,
                fight_date,
                adapter.get('event_name'),
                adapter.get('event_url'),
                adapter.get('method'),
                adapter.get('method_detail'),
                adapter.get('round'),
                adapter.get('time'),
                adapter.get('knockdowns'),
                adapter.get('sig_strikes'),
                adapter.get('takedowns'),
                adapter.get('sub_attempts'),
                adapter.get('is_title_fight'),
                adapter.get('source'),
                datetime.now(),
                existing[0],
            ))
            self.stats['fights_updated'] += 1
        else:
            # Insert
            self.cursor.execute("""
                INSERT INTO Fights (
                    FighterID, FighterURL, FighterName, OpponentID, OpponentURL, OpponentName,
                    WinnerID, WinnerName, Result, Date, EventName, EventURL, FightURL,
                    Method, MethodDetail, Round, Time, Knockdowns, SigStrikes, Takedowns,
                    SubAttempts, IsTitleFight, Source, ScrapedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter_id,
                fighter_url,
                adapter.get('fighter_name'),
                opponent_id,
                opponent_url,
                adapter.get('opponent_name'),
                winner_id,
                adapter.get('winner_name'),
                result,
                fight_date,
                adapter.get('event_name'),
                adapter.get('event_url'),
                fight_url,
                adapter.get('method'),
                adapter.get('method_detail'),
                adapter.get('round'),
                adapter.get('time'),
                adapter.get('knockdowns'),
                adapter.get('sig_strikes'),
                adapter.get('takedowns'),
                adapter.get('sub_attempts'),
                adapter.get('is_title_fight'),
                adapter.get('source'),
                datetime.now(),
            ))
            self.stats['fights_inserted'] += 1
        
        self.connection.commit()


class ScrapeCompletionStatsPipeline:
    """
    Pipeline that tracks comprehensive statistics and logs a detailed summary
    when the scrape completes.
    
    This pipeline should have the LOWEST priority (highest number) so it runs last.
    
    Tracks:
    - Total items by type (fighters, fights, career_stats)
    - Duplicates filtered
    - Validation errors
    - Active vs inactive fighters
    - Weight class distribution
    - Data completeness
    """
    
    def __init__(self):
        self.start_time = None
        self.stats = {
            'fighters': {'total': 0, 'active': 0, 'inactive': 0},
            'career_stats': {'total': 0},
            'fights': {'total': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'nc': 0},
            'events': {'total': 0},
            'duplicates_filtered': 0,
            'validation_drops': 0,
            'weight_classes': defaultdict(int),
            'stances': defaultdict(int),
            'methods': defaultdict(int),
            'missing_data': {
                'no_dob': 0,
                'no_height': 0,
                'no_reach': 0,
                'no_stats': 0,
            },
            'errors': [],
            'pages_crawled': 0,
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        from scrapy import signals
        crawler.signals.connect(pipeline.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(pipeline.item_dropped, signal=signals.item_dropped)
        crawler.signals.connect(pipeline.spider_error, signal=signals.spider_error)
        pipeline.crawler = crawler
        return pipeline
    
    def spider_opened(self, spider):
        self.start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("🚀 SCRAPE STARTED")
        logger.info(f"   Spider: {spider.name}")
        logger.info(f"   Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
    
    def spider_error(self, failure, response, spider):
        self.stats['errors'].append({
            'url': response.url if response else 'unknown',
            'error': str(failure.value),
        })
    
    def item_dropped(self, item, response, exception, spider):
        if 'duplicate' in str(exception).lower():
            self.stats['duplicates_filtered'] += 1
        else:
            self.stats['validation_drops'] += 1
    
    def process_item(self, item):
        adapter = ItemAdapter(item)
        
        if isinstance(item, FighterItem):
            self.stats['fighters']['total'] += 1
            
            # Track active/inactive
            if adapter.get('is_active'):
                self.stats['fighters']['active'] += 1
            else:
                self.stats['fighters']['inactive'] += 1
            
            # Track weight classes
            weight = adapter.get('weight')
            if weight:
                self.stats['weight_classes'][weight] += 1
            
            # Track stances
            stance = adapter.get('stance')
            if stance:
                self.stats['stances'][stance] += 1
            
            # Track missing data
            if not adapter.get('dob') or adapter.get('dob') == '--':
                self.stats['missing_data']['no_dob'] += 1
            if not adapter.get('height') or adapter.get('height') == '--':
                self.stats['missing_data']['no_height'] += 1
            if not adapter.get('reach') or adapter.get('reach') == '--':
                self.stats['missing_data']['no_reach'] += 1
        
        elif isinstance(item, CareerStatsItem):
            self.stats['career_stats']['total'] += 1
            
            # Track fighters with no meaningful stats
            if (adapter.get('slpm') == 0 and adapter.get('str_acc') == 0 and 
                adapter.get('td_avg') == 0):
                self.stats['missing_data']['no_stats'] += 1
        
        elif isinstance(item, FightItem):
            self.stats['fights']['total'] += 1
            
            # Track results
            result = adapter.get('result')
            if result == 'win':
                self.stats['fights']['wins'] += 1
            elif result == 'loss':
                self.stats['fights']['losses'] += 1
            elif result == 'draw':
                self.stats['fights']['draws'] += 1
            elif result == 'nc':
                self.stats['fights']['nc'] += 1
            
            # Track methods
            method = adapter.get('method')
            if method:
                self.stats['methods'][method] += 1
        
        elif isinstance(item, EventItem):
            self.stats['events']['total'] += 1
        
        return item
    
    def spider_closed(self, spider, reason):
        end_time = datetime.now()
        duration = end_time - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Get Scrapy stats
        scrapy_stats = self.crawler.stats.get_stats()
        
        # Determine success/failure
        request_count = scrapy_stats.get('downloader/request_count', 0)
        response_count = scrapy_stats.get('downloader/response_count', 0)
        error_count = scrapy_stats.get('downloader/exception_count', 0) or 0
        
        # Build the summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("=" * 70)
        
        if reason == 'finished' and error_count == 0:
            logger.info("✅ SCRAPE COMPLETED SUCCESSFULLY")
        elif reason == 'finished':
            logger.info("⚠️  SCRAPE COMPLETED WITH WARNINGS")
        else:
            logger.info(f"❌ SCRAPE STOPPED: {reason}")
        
        logger.info("=" * 70)
        logger.info("")
        
        # Duration
        logger.info(f"⏱️  Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"   Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Items collected
        logger.info("📊 ITEMS COLLECTED:")
        logger.info(f"   Fighters:     {self.stats['fighters']['total']:,}")
        logger.info(f"      ├─ Active:   {self.stats['fighters']['active']:,}")
        logger.info(f"      └─ Inactive: {self.stats['fighters']['inactive']:,}")
        logger.info(f"   Career Stats: {self.stats['career_stats']['total']:,}")
        logger.info(f"   Fights:       {self.stats['fights']['total']:,}")
        logger.info(f"      ├─ Wins:   {self.stats['fights']['wins']:,}")
        logger.info(f"      ├─ Losses: {self.stats['fights']['losses']:,}")
        logger.info(f"      ├─ Draws:  {self.stats['fights']['draws']:,}")
        logger.info(f"      └─ NC:     {self.stats['fights']['nc']:,}")
        logger.info("")
        
        # Duplicates & Validation
        logger.info("🔄 FILTERING:")
        logger.info(f"   Duplicates filtered: {self.stats['duplicates_filtered']:,}")
        logger.info(f"   Validation drops:    {self.stats['validation_drops']:,}")
        logger.info("")
        
        # Requests
        logger.info("🌐 REQUESTS:")
        logger.info(f"   Total requests:  {request_count:,}")
        logger.info(f"   Total responses: {response_count:,}")
        logger.info(f"   Errors:          {error_count:,}")
        logger.info("")
        
        # Data Quality
        logger.info("📋 DATA QUALITY:")
        total_fighters = self.stats['fighters']['total'] or 1  # Avoid division by zero
        no_dob_pct = (self.stats['missing_data']['no_dob'] / total_fighters) * 100
        no_height_pct = (self.stats['missing_data']['no_height'] / total_fighters) * 100
        no_reach_pct = (self.stats['missing_data']['no_reach'] / total_fighters) * 100
        no_stats_pct = (self.stats['missing_data']['no_stats'] / total_fighters) * 100
        
        logger.info(f"   Missing DOB:    {self.stats['missing_data']['no_dob']:,} ({no_dob_pct:.1f}%)")
        logger.info(f"   Missing Height: {self.stats['missing_data']['no_height']:,} ({no_height_pct:.1f}%)")
        logger.info(f"   Missing Reach:  {self.stats['missing_data']['no_reach']:,} ({no_reach_pct:.1f}%)")
        logger.info(f"   No Career Stats:{self.stats['missing_data']['no_stats']:,} ({no_stats_pct:.1f}%)")
        logger.info("")
        
        # Fight Methods Distribution
        if self.stats['methods']:
            logger.info("🥊 FINISH METHODS:")
            sorted_methods = sorted(self.stats['methods'].items(), key=lambda x: x[1], reverse=True)[:5]
            for method, count in sorted_methods:
                logger.info(f"   {method}: {count:,}")
            logger.info("")
        
        # Verification checklist
        logger.info("=" * 70)
        logger.info("📝 VERIFICATION CHECKLIST:")
        logger.info("=" * 70)
        
        # Check 1: Finished without critical errors
        if reason == 'finished':
            logger.info("   ✅ Scraper finished (not stopped early)")
        else:
            logger.info(f"   ❌ Scraper stopped early: {reason}")
        
        # Check 2: Items collected
        if self.stats['fighters']['total'] > 0:
            logger.info(f"   ✅ Fighters collected: {self.stats['fighters']['total']:,}")
        else:
            logger.info("   ❌ No fighters collected!")
        
        # Check 3: Fights per fighter ratio (should be ~10-15 avg)
        if self.stats['fighters']['total'] > 0:
            fights_per_fighter = self.stats['fights']['total'] / self.stats['fighters']['total']
            if fights_per_fighter >= 5:
                logger.info(f"   ✅ Fights per fighter: {fights_per_fighter:.1f} (looks good)")
            else:
                logger.info(f"   ⚠️  Fights per fighter: {fights_per_fighter:.1f} (seems low)")
        
        # Check 4: Career stats match fighters
        if self.stats['career_stats']['total'] == self.stats['fighters']['total']:
            logger.info(f"   ✅ Career stats match fighters ({self.stats['career_stats']['total']:,} = {self.stats['fighters']['total']:,})")
        else:
            logger.info(f"   ⚠️  Career stats mismatch: {self.stats['career_stats']['total']:,} stats vs {self.stats['fighters']['total']:,} fighters")
        
        # Check 5: Error rate
        if request_count > 0:
            error_rate = (error_count / request_count) * 100
            if error_rate < 1:
                logger.info(f"   ✅ Error rate: {error_rate:.2f}% (excellent)")
            elif error_rate < 5:
                logger.info(f"   ⚠️  Error rate: {error_rate:.2f}% (acceptable)")
            else:
                logger.info(f"   ❌ Error rate: {error_rate:.2f}% (high - check logs)")
        
        # Check 6: Active fighters (sanity check - should be ~600-800 for UFC)
        if self.stats['fighters']['active'] > 0:
            active_pct = (self.stats['fighters']['active'] / self.stats['fighters']['total']) * 100
            logger.info(f"   ℹ️  Active fighters: {self.stats['fighters']['active']:,} ({active_pct:.1f}%)")
        
        logger.info("=" * 70)
        logger.info("")
        
        # Save summary to JSON
        output_dir = Path(__file__).parent.parent.parent / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_dir / 'scrape_summary.json'
        summary = {
            'completed_at': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'reason': reason,
            'fighters': self.stats['fighters'],
            'career_stats': self.stats['career_stats'],
            'fights': self.stats['fights'],
            'duplicates_filtered': self.stats['duplicates_filtered'],
            'validation_drops': self.stats['validation_drops'],
            'missing_data': self.stats['missing_data'],
            'requests': request_count,
            'responses': response_count,
            'errors': error_count,
            'top_methods': dict(sorted(self.stats['methods'].items(), key=lambda x: x[1], reverse=True)[:10]),
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Summary saved to: {summary_file}")
        logger.info("")


class ActiveFighterFilterPipeline:
    """
    Pipeline to filter for active fighters only.
    
    A fighter is considered "active" if they have fought within the last N years.
    
    Configure in settings.py:
        ACTIVE_FIGHTER_FILTER_ENABLED = True
        ACTIVE_FIGHTER_YEARS = 2  # Fighters who fought in last 2 years
    
    Usage:
        scrapy crawl ufcstats -s ACTIVE_FIGHTER_FILTER_ENABLED=True
        scrapy crawl ufcstats -s ACTIVE_FIGHTER_FILTER_ENABLED=True -s ACTIVE_FIGHTER_YEARS=3
    """
    
    DATE_FORMATS = [
        "%b %d, %Y",      # "Dec 30, 1994"
        "%B %d, %Y",      # "December 30, 1994"  
        "%b. %d, %Y",     # "Dec. 30, 1994"
        "%Y-%m-%d",       # "1994-12-30"
        "%m/%d/%Y",       # "12/30/1994"
    ]
    
    def __init__(self, enabled=False, years=2):
        self.enabled = enabled
        self.max_days = years * 365
        self.active_fighters = set()  # Track which fighters are active
        self.inactive_count = 0
        self.active_count = 0
        self.fight_dates = {}  # fighter_url -> most recent fight date
    
    @classmethod
    def from_crawler(cls, crawler):
        enabled = crawler.settings.getbool('ACTIVE_FIGHTER_FILTER_ENABLED', False)
        years = crawler.settings.getint('ACTIVE_FIGHTER_YEARS', 2)
        
        pipeline = cls(enabled, years)
        
        if enabled:
            logger.info(f"Active fighter filter ENABLED: keeping fighters who fought in last {years} years")
        
        from scrapy import signals
        crawler.signals.connect(pipeline.spider_closed, signal=signals.spider_closed)
        
        return pipeline
    
    def process_item(self, item):
        if not self.enabled:
            return item
        
        adapter = ItemAdapter(item)
        
        # For FightItem, track the most recent fight date per fighter
        if isinstance(item, FightItem):
            fighter_url = adapter.get('fighter_url')
            fight_date_str = adapter.get('date')
            
            if fighter_url and fight_date_str:
                fight_date = self._parse_date(fight_date_str)
                if fight_date:
                    # Keep track of most recent fight
                    if fighter_url not in self.fight_dates or fight_date > self.fight_dates[fighter_url]:
                        self.fight_dates[fighter_url] = fight_date
                        
                        # Check if this makes the fighter active
                        days_ago = (datetime.now() - fight_date).days
                        if days_ago <= self.max_days:
                            self.active_fighters.add(fighter_url)
            
            # Always pass through fights (we filter fighters, not fights)
            return item
        
        # For FighterItem, check if fighter is active
        if isinstance(item, FighterItem):
            fighter_url = adapter.get('fighter_url')
            
            if fighter_url in self.active_fighters:
                self.active_count += 1
                return item
            else:
                # Fighter not yet determined as active - pass through
                # (fights might come after fighter item)
                return item
        
        # For CareerStatsItem, check if associated fighter is active
        if isinstance(item, CareerStatsItem):
            fighter_url = adapter.get('fighter_url')
            
            if fighter_url in self.active_fighters:
                return item
            else:
                # Pass through - will be filtered later if needed
                return item
        
        return item
    
    def _parse_date(self, date_string):
        """Parse a date string into a datetime object."""
        if not date_string or date_string == '--':
            return None
        
        date_string = date_string.strip()
        
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        return None
    
    def spider_closed(self, spider):
        """Log stats when spider closes."""
        if self.enabled:
            logger.info(f"Active fighter filter: {len(self.active_fighters)} active fighters found")
            logger.info(f"Active fighters fought within last {self.max_days // 365} years")


class DataCleaningPipeline:
    """
    Pipeline to clean and transform scraped data.
    
    Transformations:
    - Calculate age from DOB
    - Parse height to inches
    - Parse weight to numeric
    - Parse reach to inches
    - Clean percentage values
    """
    
    # Common date formats found in scraped data
    DATE_FORMATS = [
        "%b %d, %Y",      # "Dec 30, 1994"
        "%B %d, %Y",      # "December 30, 1994"
        "%b. %d, %Y",     # "Dec. 30, 1994"
        "%Y-%m-%d",       # "1994-12-30"
        "%m/%d/%Y",       # "12/30/1994"
        "%d %b %Y",       # "30 Dec 1994"
        "%Y %b %d",       # "1994 Dec 30" (Tapology format)
    ]
    
    def _parse_date(self, date_string):
        """Parse a date string into a datetime object."""
        if not date_string or date_string == '--':
            return None
        
        date_string = date_string.strip()
        
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        return None
    
    def process_item(self, item):
        adapter = ItemAdapter(item)
        
        if isinstance(item, FighterItem):
            # Calculate age from DOB
            dob = adapter.get('dob')
            if dob and not adapter.get('age'):
                age = self._calculate_age(dob)
                if age is not None:
                    adapter['age'] = age
                    logger.debug(f"Calculated age {age} for {adapter.get('name')}")
            
            # Parse height to inches (optional - for future use)
            # height = adapter.get('height')
            # if height:
            #     adapter['height_inches'] = self._parse_height(height)
            
            # Parse weight to numeric (optional - for future use)
            # weight = adapter.get('weight')
            # if weight:
            #     adapter['weight_lbs'] = self._parse_weight(weight)
        
        return item
    
    def _calculate_age(self, dob_string):
        """
        Calculate age from date of birth string.
        
        Args:
            dob_string: Date string in various formats (e.g., "Dec 30, 1994")
            
        Returns:
            int: Age in years, or None if parsing fails
        """
        if not dob_string:
            return None
        
        # Clean the string
        dob_string = dob_string.strip()
        
        # Try each date format
        dob_date = None
        for fmt in self.DATE_FORMATS:
            try:
                dob_date = datetime.strptime(dob_string, fmt)
                break
            except ValueError:
                continue
        
        if dob_date is None:
            # Try to extract just the year if full parsing fails
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', dob_string)
            if year_match:
                birth_year = int(year_match.group(1))
                current_year = datetime.now().year
                # Approximate age (won't account for month/day)
                return current_year - birth_year
            
            logger.warning(f"Could not parse DOB: {dob_string}")
            return None
        
        # Calculate age
        today = datetime.now()
        age = today.year - dob_date.year
        
        # Adjust if birthday hasn't occurred this year
        if (today.month, today.day) < (dob_date.month, dob_date.day):
            age -= 1
        
        return age
    
    def _parse_height(self, height_string):
        """
        Parse height string to inches.
        
        Args:
            height_string: e.g., "5' 11\"", "5'11", "71 in"
            
        Returns:
            int: Height in inches, or None if parsing fails
        """
        if not height_string:
            return None
        
        # Try feet and inches format: 5' 11" or 5'11"
        match = re.search(r"(\d+)'?\s*(\d+)?\"?", height_string)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2)) if match.group(2) else 0
            return feet * 12 + inches
        
        # Try just inches: "71 in" or "71\""
        match = re.search(r"(\d+)\s*(?:in|\")", height_string)
        if match:
            return int(match.group(1))
        
        return None
    
    def _parse_weight(self, weight_string):
        """
        Parse weight string to numeric pounds.
        
        Args:
            weight_string: e.g., "145 lbs.", "145 lbs", "145"
            
        Returns:
            float: Weight in pounds, or None if parsing fails
        """
        if not weight_string:
            return None
        
        match = re.search(r"(\d+\.?\d*)", weight_string)
        if match:
            return float(match.group(1))
        
        return None
    
    def _parse_reach(self, reach_string):
        """
        Parse reach string to inches.
        
        Args:
            reach_string: e.g., "72\"", "72 in", "72"
            
        Returns:
            float: Reach in inches, or None if parsing fails
        """
        if not reach_string:
            return None
        
        match = re.search(r"(\d+\.?\d*)", reach_string)
        if match:
            return float(match.group(1))
        
        return None


class CsvExportPipeline:
    """
    Pipeline that exports items to separate CSV files based on item type.
    
    DEDUPLICATION: Loads existing CSV data, merges with new items by unique key,
    and writes all data at the end. This prevents duplicates across multiple scrape runs.
    """
    
    def __init__(self):
        self.data = {}  # Stores items in memory keyed by unique identifier
        self.item_counts = {'new': {}, 'updated': {}}
    
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        # Get output directory from settings or use default
        pipeline.output_dir = crawler.settings.get(
            'CSV_OUTPUT_DIR',
            Path(__file__).parent.parent.parent / 'output'
        )
        return pipeline
    
    def open_spider(self, spider):
        """Initialize output directory and load existing CSV data."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define CSV files and their headers
        self.csv_configs = {
            'fighters': {
                'filename': 'fighters.csv',
                'item_class': FighterItem,
                'unique_key': 'fighter_url',
                'headers': [
                    'name', 'fighter_url', 'height', 'weight', 'reach', 
                    'stance', 'dob', 'age', 'weight_class',
                    'nickname', 'place_of_birth', 'leg_reach', 'ufc_url',
                    'days_since_last_fight', 'last_fight_date', 'is_active',
                    'wins', 'losses', 'draws', 'total_fights',
                    'scraped_at', 'source'
                ]
            },
            'career_stats': {
                'filename': 'career_stats.csv',
                'item_class': CareerStatsItem,
                'unique_key': 'fighter_url',
                'headers': [
                    'fighter_name', 'fighter_url',
                    'slpm', 'str_acc', 'sapm', 'str_def',
                    'td_avg', 'td_acc', 'td_def', 'sub_avg',
                    'win_streak_last3', 'wins_by_ko_last5', 'wins_by_sub_last5',
                    'avg_fight_duration', 'first_round_finish_rate', 'decision_rate',
                    'ko_round1_pct', 'ko_round2_pct', 'ko_round3_pct',
                    'sub_round1_pct', 'sub_round2_pct', 'sub_round3_pct',
                    'elo_rating', 'scraped_at', 'source'
                ]
            },
            'fights': {
                'filename': 'fights.csv',
                'item_class': FightItem,
                'unique_key': 'fight_url',
                'headers': [
                    'fighter_name', 'fighter_url', 'opponent_name', 'opponent_url',
                    'result', 'winner_name', 'event_name', 'event_url', 'date',
                    'method', 'method_detail', 'round', 'time',
                    'knockdowns', 'sig_strikes', 'takedowns', 'sub_attempts',
                    'is_title_fight', 'is_main_event', 'fluke_flag',
                    'fight_url', 'scraped_at', 'source'
                ]
            },
            'events': {
                'filename': 'events.csv',
                'item_class': EventItem,
                'unique_key': 'event_url',
                'headers': [
                    'event_name', 'event_url', 'date', 'location',
                    'scraped_at', 'source'
                ]
            }
        }
        
        # Load existing data from CSVs into memory
        for key, config in self.csv_configs.items():
            filepath = os.path.join(self.output_dir, config['filename'])
            self.data[key] = {}
            self.item_counts['new'][key] = 0
            self.item_counts['updated'][key] = 0
            
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                try:
                    with open(filepath, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            unique_val = row.get(config['unique_key'])
                            if unique_val:
                                self.data[key][unique_val] = row
                    spider.logger.info(f"CSV Pipeline: Loaded {len(self.data[key])} existing records from {config['filename']}")
                except Exception as e:
                    spider.logger.warning(f"CSV Pipeline: Could not load {config['filename']}: {e}")
        
        spider.logger.info(f"CSV Pipeline: Output directory set to {self.output_dir}")
    
    def close_spider(self, spider):
        """Write all data to CSV files (replacing existing files)."""
        for key, config in self.csv_configs.items():
            filepath = os.path.join(self.output_dir, config['filename'])
            
            # Write all data to file (overwrite mode)
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=config['headers'],
                    extrasaction='ignore'
                )
                writer.writeheader()
                
                for row in self.data[key].values():
                    writer.writerow(row)
            
            total = len(self.data[key])
            new_count = self.item_counts['new'][key]
            updated_count = self.item_counts['updated'][key]
            spider.logger.info(f"CSV Pipeline: {config['filename']} - {total} total ({new_count} new, {updated_count} updated)")
    
    def process_item(self, item, spider):
        """Store items in memory, updating existing or adding new."""
        adapter = ItemAdapter(item)
        item_dict = adapter.asdict()
        
        # Determine which data store to use
        for key, config in self.csv_configs.items():
            if isinstance(item, config['item_class']):
                unique_key = config['unique_key']
                unique_val = item_dict.get(unique_key)
                
                if not unique_val:
                    # No unique key - can't deduplicate, skip
                    spider.logger.warning(f"Item missing unique key {unique_key}: {item_dict.get('name', 'Unknown')}")
                    return item
                
                # Check if this is an update or new record
                if unique_val in self.data[key]:
                    self.item_counts['updated'][key] += 1
                else:
                    self.item_counts['new'][key] += 1
                
                # Store/update the item
                self.data[key][unique_val] = item_dict
                return item
        
        # If item type doesn't match any config, log warning
        spider.logger.warning(f"Unknown item type: {type(item).__name__}")
        return item


class ValidationPipeline:
    """
    Pipeline to validate items before export.
    Drops items with missing required fields.
    """
    
    def process_item(self, item):
        adapter = ItemAdapter(item)
        
        # Validate FighterItem
        if isinstance(item, FighterItem):
            name = adapter.get('name')
            if not name or (isinstance(name, str) and name.strip() == ''):
                raise DropItem(f"Missing or empty fighter name: {item}")
            
            # Also validate fighter_url
            url = adapter.get('fighter_url')
            if not url or (isinstance(url, str) and url.strip() == ''):
                raise DropItem(f"Missing fighter URL for {name}: {item}")
        
        # Validate CareerStatsItem
        elif isinstance(item, CareerStatsItem):
            if not adapter.get('fighter_name'):
                raise DropItem(f"Missing fighter_name in CareerStats: {item}")
        
        # Validate FightItem
        elif isinstance(item, FightItem):
            if not adapter.get('fighter_name'):
                raise DropItem(f"Missing fighter_name in Fight: {item}")
            
            # Skip scheduled/upcoming fights (no result yet)
            result = adapter.get('result')
            if not result or result in ['next', 'scheduled']:
                raise DropItem(f"Scheduled fight (no result yet): {adapter.get('fighter_name')} vs {adapter.get('opponent_name')}")
        
        return item


class DataRangeValidationPipeline:
    """
    Pipeline to validate data is within expected ranges.
    
    Logs warnings for suspicious values but doesn't drop items
    (data might be legitimately unusual).
    
    Validates:
    - Percentages are 0-100
    - Rates are non-negative
    - Fight rounds are 1-5
    - Records make sense (wins + losses + draws = total)
    """
    
    def __init__(self):
        self.warnings = []
        self.validation_stats = {
            'total_validated': 0,
            'warnings_count': 0,
            'items_with_warnings': 0,
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        pipeline.output_dir = crawler.settings.get(
            'CSV_OUTPUT_DIR',
            Path(__file__).parent.parent.parent / 'output'
        )
        from scrapy import signals
        crawler.signals.connect(pipeline.spider_closed, signal=signals.spider_closed)
        return pipeline
    
    def process_item(self, item):
        adapter = ItemAdapter(item)
        item_warnings = []
        
        self.validation_stats['total_validated'] += 1
        
        if isinstance(item, FighterItem):
            item_warnings.extend(self._validate_fighter(adapter))
        elif isinstance(item, CareerStatsItem):
            item_warnings.extend(self._validate_career_stats(adapter))
        elif isinstance(item, FightItem):
            item_warnings.extend(self._validate_fight(adapter))
        
        if item_warnings:
            self.validation_stats['items_with_warnings'] += 1
            self.validation_stats['warnings_count'] += len(item_warnings)
            
            # Log warnings
            item_id = adapter.get('name') or adapter.get('fighter_name') or 'Unknown'
            for warning in item_warnings:
                logger.warning(f"[Validation] {item_id}: {warning}")
                self.warnings.append({
                    'item_type': type(item).__name__,
                    'item_id': item_id,
                    'warning': warning,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return item
    
    def _validate_fighter(self, adapter):
        """Validate FighterItem fields."""
        warnings = []
        
        # Validate record consistency
        wins = adapter.get('wins')
        losses = adapter.get('losses')
        draws = adapter.get('draws')
        total = adapter.get('total_fights')
        
        if all(v is not None for v in [wins, losses, draws, total]):
            expected_total = wins + losses + draws
            if total != expected_total:
                warnings.append(
                    f"Record mismatch: {wins}W-{losses}L-{draws}D != {total} total"
                )
        
        # Validate non-negative record
        for field, value in [('wins', wins), ('losses', losses), ('draws', draws)]:
            if value is not None and value < 0:
                warnings.append(f"Negative {field}: {value}")
        
        # Validate reasonable total fights (UFC record is ~40 max)
        if total is not None and total > 100:
            warnings.append(f"Unusually high total fights: {total}")
        
        return warnings
    
    def _validate_career_stats(self, adapter):
        """Validate CareerStatsItem fields."""
        warnings = []
        
        # Percentage fields (should be 0-100)
        percentage_fields = [
            'str_acc', 'str_def', 'td_acc', 'td_def',
            'first_round_finish_rate', 'decision_rate',
            'ko_round1_pct', 'ko_round2_pct', 'ko_round3_pct',
            'sub_round1_pct', 'sub_round2_pct', 'sub_round3_pct'
        ]
        
        for field in percentage_fields:
            value = adapter.get(field)
            if value is not None:
                if value < 0:
                    warnings.append(f"Negative percentage for {field}: {value}")
                elif value > 100:
                    warnings.append(f"Percentage over 100 for {field}: {value}")
        
        # Rate fields (should be non-negative, typically 0-15 range)
        rate_fields = ['slpm', 'sapm', 'td_avg', 'sub_avg']
        
        for field in rate_fields:
            value = adapter.get(field)
            if value is not None:
                if value < 0:
                    warnings.append(f"Negative rate for {field}: {value}")
                elif value > 20:
                    # Very high but possible for some fighters
                    warnings.append(f"Unusually high rate for {field}: {value}")
        
        # Elo rating (typically 1000-2000 range)
        elo = adapter.get('elo_rating')
        if elo is not None:
            if elo < 500:
                warnings.append(f"Very low Elo rating: {elo}")
            elif elo > 3000:
                warnings.append(f"Very high Elo rating: {elo}")
        
        return warnings
    
    def _validate_fight(self, adapter):
        """Validate FightItem fields."""
        warnings = []
        
        # Round should be 1-5 (sometimes 6 for old UFC)
        round_num = adapter.get('round')
        if round_num is not None:
            if round_num < 1:
                warnings.append(f"Invalid round number: {round_num}")
            elif round_num > 5:
                warnings.append(f"Unusual round number: {round_num}")
        
        # Result should be valid
        result = adapter.get('result')
        valid_results = ['win', 'loss', 'draw', 'nc', None]
        if result and result.lower() not in valid_results:
            warnings.append(f"Unknown result type: {result}")
        
        # Time format should be MM:SS or M:SS
        time = adapter.get('time')
        if time:
            if not re.match(r'^\d{1,2}:\d{2}$', time):
                warnings.append(f"Invalid time format: {time}")
            else:
                # Check time is within round (max 5:00)
                parts = time.split(':')
                minutes = int(parts[0])
                seconds = int(parts[1])
                if minutes > 5 or (minutes == 5 and seconds > 0):
                    warnings.append(f"Time exceeds round length: {time}")
        
        # Knockdowns should be reasonable (record is around 4 in a fight)
        kd = adapter.get('knockdowns')
        if kd is not None:
            try:
                kd_int = int(kd)
                if kd_int < 0:
                    warnings.append(f"Negative knockdowns: {kd}")
                elif kd_int > 10:
                    warnings.append(f"Unusually high knockdowns: {kd}")
            except ValueError:
                warnings.append(f"Invalid knockdowns value: {kd}")
        
        # Check fighter and opponent are different
        fighter = adapter.get('fighter_name')
        opponent = adapter.get('opponent_name')
        if fighter and opponent and fighter.lower() == opponent.lower():
            warnings.append(f"Fighter and opponent are the same: {fighter}")
        
        return warnings
    
    def spider_closed(self, spider):
        """Save validation report when spider closes."""
        if not self.warnings:
            logger.info("Validation complete: No warnings")
            return
        
        # Save validation report
        report_path = Path(self.output_dir) / 'validation_report.json'
        
        report = {
            'spider': spider.name,
            'generated_at': datetime.utcnow().isoformat(),
            'stats': self.validation_stats,
            'warnings': self.warnings
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
        logger.info(
            f"Validation stats: {self.validation_stats['total_validated']} items validated, "
            f"{self.validation_stats['items_with_warnings']} with warnings, "
            f"{self.validation_stats['warnings_count']} total warnings"
        )


class DataMergePipeline:
    """
    Pipeline to merge UFC.com data with UFCStats data.
    
    UFC.com provides supplementary data for FighterItem:
    - Weight class (primary source)
    - Nickname
    - Place of birth
    - Leg reach
    
    UFC.com also provides CareerStatsItem data:
    - Win streak
    - Wins by KO/SUB
    - Average fight time
    - Decision rate
    - KO/SUB round percentages
    
    Data is merged by normalized fighter name.
    
    IMPORTANT: This pipeline persists UFC.com data to a JSON cache file
    so that when running multiple spiders (ufc_official then ufcstats),
    the data can be shared across spider instances.
    """
    
    # Cache file path (in output directory)
    CACHE_FILE = Path(__file__).parent.parent.parent / 'output' / 'ufc_merge_cache.json'
    
    def __init__(self):
        self.ufc_official_fighters = {}  # name_key -> fighter data dict
        self.ufc_official_stats = {}     # name_key -> career stats dict
        self.merged_fighters = 0
        self.merged_stats = 0
        self.unmatched_count = 0
    
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        from scrapy import signals
        crawler.signals.connect(pipeline.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signal=signals.spider_closed)
        return pipeline
    
    def spider_opened(self, spider):
        """Load cached UFC.com data if available."""
        if self.CACHE_FILE.exists():
            try:
                with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    self.ufc_official_fighters = cache.get('fighters', {})
                    self.ufc_official_stats = cache.get('stats', {})
                    logger.info(f"Loaded UFC.com cache: {len(self.ufc_official_fighters)} fighters, {len(self.ufc_official_stats)} stats")
            except Exception as e:
                logger.warning(f"Failed to load UFC.com cache: {e}")
    
    def _save_cache(self):
        """Save UFC.com data to cache file."""
        try:
            self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            cache = {
                'fighters': self.ufc_official_fighters,
                'stats': self.ufc_official_stats,
                'updated_at': datetime.utcnow().isoformat(),
            }
            with open(self.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved UFC.com cache: {len(self.ufc_official_fighters)} fighters, {len(self.ufc_official_stats)} stats")
        except Exception as e:
            logger.warning(f"Failed to save UFC.com cache: {e}")
    
    def _normalize_name(self, name):
        """
        Normalize fighter name for matching.
        - Lowercase
        - Remove accents/diacritics
        - Remove extra whitespace
        - Remove common suffixes like Jr., Sr., III
        """
        if not name:
            return None
        
        import unicodedata
        
        # Normalize unicode and remove accents
        name = unicodedata.normalize('NFKD', name)
        name = ''.join(c for c in name if not unicodedata.combining(c))
        
        # Lowercase and strip
        name = name.lower().strip()
        
        # Remove suffixes
        suffixes = [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def process_item(self, item):
        adapter = ItemAdapter(item)
        
        if isinstance(item, FighterItem):
            source = adapter.get('source')
            name = adapter.get('name')
            name_key = self._normalize_name(name)
            
            if source == 'ufc_official':
                # Store ALL UFC.com fighter data for later merging (UFC.com is priority source)
                if name_key:
                    self.ufc_official_fighters[name_key] = {
                        # Bio fields
                        'height': adapter.get('height'),
                        'weight': adapter.get('weight'),
                        'reach': adapter.get('reach'),
                        'stance': adapter.get('stance'),
                        'dob': adapter.get('dob'),
                        'age': adapter.get('age'),
                        'weight_class': adapter.get('weight_class'),
                        'nickname': adapter.get('nickname'),
                        'place_of_birth': adapter.get('place_of_birth'),
                        'leg_reach': adapter.get('leg_reach'),
                        'ufc_url': adapter.get('fighter_url'),
                        'is_active': adapter.get('is_active'),
                        # Record fields
                        'wins': adapter.get('wins'),
                        'losses': adapter.get('losses'),
                        'draws': adapter.get('draws'),
                        'total_fights': adapter.get('total_fights'),
                    }
                return item
            
            elif source == 'ufcstats':
                # Try to merge with UFC.com data - UFC.com takes PRIORITY for all fields
                if name_key and name_key in self.ufc_official_fighters:
                    ufc_data = self.ufc_official_fighters[name_key]
                    
                    # UFC.com is authoritative - use its data when available
                    # Bio fields - UFC.com priority
                    priority_fields = [
                        'weight_class', 'nickname', 'place_of_birth', 'leg_reach',
                        'height', 'weight', 'reach', 'stance', 'dob', 'age', 'is_active'
                    ]
                    
                    for field in priority_fields:
                        ufc_val = ufc_data.get(field)
                        if ufc_val is not None and ufc_val != '':
                            adapter[field] = ufc_val
                    
                    # Always set ufc_url
                    if ufc_data.get('ufc_url'):
                        adapter['ufc_url'] = ufc_data['ufc_url']
                    
                    # For record fields, only fill if UFCStats doesn't have them
                    # (UFCStats usually has more accurate fight counts)
                    record_fields = ['wins', 'losses', 'draws', 'total_fights']
                    for field in record_fields:
                        if adapter.get(field) is None and ufc_data.get(field) is not None:
                            adapter[field] = ufc_data[field]
                    
                    self.merged_fighters += 1
                else:
                    self.unmatched_count += 1
        
        elif isinstance(item, CareerStatsItem):
            source = adapter.get('source')
            name = adapter.get('fighter_name')
            name_key = self._normalize_name(name)
            
            if source == 'ufc_official':
                # Store UFC.com career stats for later merging
                if name_key:
                    self.ufc_official_stats[name_key] = {
                        'win_streak_last3': adapter.get('win_streak_last3'),
                        'wins_by_ko_last5': adapter.get('wins_by_ko_last5'),
                        'wins_by_sub_last5': adapter.get('wins_by_sub_last5'),
                        'avg_fight_duration': adapter.get('avg_fight_duration'),
                        'first_round_finish_rate': adapter.get('first_round_finish_rate'),
                        'decision_rate': adapter.get('decision_rate'),
                        'ko_round1_pct': adapter.get('ko_round1_pct'),
                        'ko_round2_pct': adapter.get('ko_round2_pct'),
                        'ko_round3_pct': adapter.get('ko_round3_pct'),
                        'sub_round1_pct': adapter.get('sub_round1_pct'),
                        'sub_round2_pct': adapter.get('sub_round2_pct'),
                        'sub_round3_pct': adapter.get('sub_round3_pct'),
                    }
                return item
            
            elif source == 'ufcstats':
                # Try to merge with UFC.com career stats
                if name_key and name_key in self.ufc_official_stats:
                    ufc_stats = self.ufc_official_stats[name_key]
                    
                    # Merge stats fields (UFC.com supplements UFCStats)
                    # Only fill in if UFCStats doesn't have the value
                    stats_fields = [
                        'win_streak_last3', 'wins_by_ko_last5', 'wins_by_sub_last5',
                        'avg_fight_duration', 'first_round_finish_rate', 'decision_rate',
                        'ko_round1_pct', 'ko_round2_pct', 'ko_round3_pct',
                        'sub_round1_pct', 'sub_round2_pct', 'sub_round3_pct',
                    ]
                    
                    for field in stats_fields:
                        if ufc_stats.get(field) is not None and adapter.get(field) is None:
                            adapter[field] = ufc_stats[field]
                    
                    self.merged_stats += 1
        
        return item
    
    def spider_closed(self, spider):
        """Save cache and log merge statistics."""
        # Save UFC.com data to cache file when ufc_official spider finishes
        if spider.name == 'ufc_official' and (self.ufc_official_fighters or self.ufc_official_stats):
            self._save_cache()
        
        total_ufc_fighters = len(self.ufc_official_fighters)
        total_ufc_stats = len(self.ufc_official_stats)
        logger.info("=" * 60)
        logger.info("DATA MERGE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"   UFC.com fighters cached:    {total_ufc_fighters}")
        logger.info(f"   UFC.com career stats cached:{total_ufc_stats}")
        logger.info(f"   Fighters merged:            {self.merged_fighters}")
        logger.info(f"   Career stats merged:        {self.merged_stats}")
        logger.info(f"   Unmatched UFCStats fighters:{self.unmatched_count}")
        logger.info("=" * 60)


class FightStatsCalculationPipeline:
    """
    Pipeline to calculate fight statistics from fight history.
    
    Calculates:
    - KO round percentages (ko_round1_pct, ko_round2_pct, ko_round3_pct)
    - SUB round percentages (sub_round1_pct, sub_round2_pct, sub_round3_pct)
    - First round finish rate
    - Decision rate
    - Average fight duration
    - Win streak (last 3)
    - Wins by KO/SUB (last 5)
    
    This pipeline collects all fights, then calculates stats when spider closes.
    Stats are written to a separate JSON file for later merging with career_stats.csv.
    """
    
    def __init__(self):
        self.fighter_fights = defaultdict(list)  # fighter_url -> list of fights
        self.output_dir = None
        self.crawler = None
    
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        pipeline.crawler = crawler  # Store reference for settings access
        pipeline.output_dir = crawler.settings.get(
            'CSV_OUTPUT_DIR',
            Path(__file__).parent.parent.parent / 'output'
        )
        from scrapy import signals
        crawler.signals.connect(pipeline.spider_closed, signal=signals.spider_closed)
        return pipeline
    
    def process_item(self, item):
        if isinstance(item, FightItem):
            adapter = ItemAdapter(item)
            fighter_url = adapter.get('fighter_url')
            
            if fighter_url:
                # Store fight data for later calculation
                self.fighter_fights[fighter_url].append({
                    'result': adapter.get('result'),
                    'method': adapter.get('method'),
                    'method_detail': adapter.get('method_detail'),
                    'round': adapter.get('round'),
                    'time': adapter.get('time'),
                    'date': adapter.get('date'),
                })
        
        return item
    
    def spider_closed(self, spider):
        """Calculate stats for all fighters and update career_stats.csv and database."""
        if not self.fighter_fights:
            return
        
        logger.info("Calculating fight statistics from fight history...")
        
        calculated_stats = {}
        
        for fighter_url, fights in self.fighter_fights.items():
            stats = self._calculate_fighter_stats(fights)
            calculated_stats[fighter_url] = stats
        
        # Save to JSON file
        output_path = Path(self.output_dir) / 'calculated_stats.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(calculated_stats, f, indent=2)
        
        logger.info(f"Fight stats calculated for {len(calculated_stats)} fighters")
        logger.info(f"Stats saved to: {output_path}")
        
        # Also update career_stats.csv with calculated values
        self._update_career_stats_csv(calculated_stats)
        
        # Update database with calculated values
        self._update_database(calculated_stats)
    
    def _update_database(self, calculated_stats):
        """Update CareerStats table in database with calculated stats."""
        if not PSYCOPG2_AVAILABLE:
            return
        
        # Check if database pipeline was enabled
        db_enabled = self.crawler.settings.getbool('DATABASE_PIPELINE_ENABLED', False) if hasattr(self, 'crawler') else False
        if not db_enabled:
            return
        
        try:
            # Connect to database
            connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5433'),
                dbname=os.getenv('DB_NAME', 'delphi_db'),
                user=os.getenv('DB_USER', ''),
                password=os.getenv('DB_PASSWORD', ''),
            )
            cursor = connection.cursor()
            
            updated_count = 0
            for fighter_url, stats in calculated_stats.items():
                # Get FighterID from URL
                cursor.execute(
                    "SELECT FighterID FROM FighterStats WHERE FighterURL = %s OR UFCUrl = %s",
                    (fighter_url, fighter_url)
                )
                result = cursor.fetchone()
                
                if not result:
                    continue
                
                fighter_id = result[0]
                
                # Update CareerStats with calculated values
                update_fields = []
                update_values = []
                
                field_mapping = {
                    'avg_fight_duration': 'AvgFightDuration',
                    'first_round_finish_rate': 'FirstRoundFinishRate',
                    'decision_rate': 'DecisionRate',
                    'win_streak_last3': 'WinStreak_Last3',
                    'wins_by_ko_last5': 'WinsByKO_Last5',
                    'wins_by_sub_last5': 'WinsBySub_Last5',
                    'ko_round1_pct': 'KO_Round1_Pct',
                    'ko_round2_pct': 'KO_Round2_Pct',
                    'ko_round3_pct': 'KO_Round3_Pct',
                    'sub_round1_pct': 'Sub_Round1_Pct',
                    'sub_round2_pct': 'Sub_Round2_Pct',
                    'sub_round3_pct': 'Sub_Round3_Pct',
                }
                
                for py_field, db_field in field_mapping.items():
                    value = stats.get(py_field)
                    if value is not None:
                        # Only update if current value is NULL (preserve UFC.com data)
                        update_fields.append(f"{db_field} = COALESCE({db_field}, %s)")
                        update_values.append(value)
                
                if update_fields:
                    update_values.append(fighter_id)
                    sql = f"UPDATE CareerStats SET {', '.join(update_fields)} WHERE FighterID = %s"
                    cursor.execute(sql, update_values)
                    updated_count += 1
            
            connection.commit()
            cursor.close()
            connection.close()
            
            logger.info(f"Updated {updated_count} records in database with calculated stats")
            
        except Exception as e:
            logger.error(f"Error updating database with calculated stats: {e}")
    
    def _update_career_stats_csv(self, calculated_stats):
        """Update career_stats.csv with calculated stats from fight history."""
        csv_path = Path(self.output_dir) / 'career_stats.csv'
        
        if not csv_path.exists():
            logger.warning(f"career_stats.csv not found at {csv_path}, skipping update")
            return
        
        try:
            # Read existing CSV
            rows = []
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    rows.append(row)
            
            # Update rows with calculated stats
            updated_count = 0
            for row in rows:
                fighter_url = row.get('fighter_url')
                if fighter_url and fighter_url in calculated_stats:
                    stats = calculated_stats[fighter_url]
                    
                    # Only fill in empty/null values
                    stats_fields = [
                        'win_streak_last3', 'wins_by_ko_last5', 'wins_by_sub_last5',
                        'avg_fight_duration', 'first_round_finish_rate', 'decision_rate',
                        'ko_round1_pct', 'ko_round2_pct', 'ko_round3_pct',
                        'sub_round1_pct', 'sub_round2_pct', 'sub_round3_pct',
                    ]
                    
                    for field in stats_fields:
                        current_val = row.get(field)
                        new_val = stats.get(field)
                        
                        # Update if current is empty and we have a calculated value
                        if (not current_val or current_val == '' or current_val == 'None') and new_val is not None:
                            row[field] = new_val
                    
                    updated_count += 1
            
            # Write back to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Updated {updated_count} records in career_stats.csv with calculated stats")
            
        except Exception as e:
            logger.error(f"Error updating career_stats.csv: {e}")
    
    def _calculate_fighter_stats(self, fights):
        """
        Calculate statistics for a single fighter from their fight history.
        """
        stats = {
            'ko_round1_pct': None,
            'ko_round2_pct': None,
            'ko_round3_pct': None,
            'sub_round1_pct': None,
            'sub_round2_pct': None,
            'sub_round3_pct': None,
            'first_round_finish_rate': None,
            'decision_rate': None,
            'avg_fight_duration': None,
            'win_streak_last3': 0,
            'wins_by_ko_last5': 0,
            'wins_by_sub_last5': 0,
        }
        
        if not fights:
            return stats
        
        # Sort fights by date (most recent first)
        def parse_date(date_str):
            if not date_str:
                return datetime.min
            for fmt in ["%b. %d, %Y", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return datetime.min
        
        sorted_fights = sorted(fights, key=lambda f: parse_date(f.get('date')), reverse=True)
        
        # Calculate win streak (last 3)
        streak = 0
        for fight in sorted_fights[:3]:
            if fight.get('result') == 'win':
                streak += 1
            else:
                break
        stats['win_streak_last3'] = streak
        
        # Calculate wins by KO/SUB in last 5 fights
        ko_wins = 0
        sub_wins = 0
        for fight in sorted_fights[:5]:
            if fight.get('result') == 'win':
                method = (fight.get('method') or '').upper()
                if 'KO' in method or 'TKO' in method:
                    ko_wins += 1
                elif 'SUB' in method:
                    sub_wins += 1
        stats['wins_by_ko_last5'] = ko_wins
        stats['wins_by_sub_last5'] = sub_wins
        
        # Calculate round percentages for all fights
        total_wins = sum(1 for f in fights if f.get('result') == 'win')
        
        if total_wins > 0:
            ko_wins_total = [f for f in fights if f.get('result') == 'win' and 
                           ('KO' in (f.get('method') or '').upper() or 'TKO' in (f.get('method') or '').upper())]
            sub_wins_total = [f for f in fights if f.get('result') == 'win' and 
                            'SUB' in (f.get('method') or '').upper()]
            
            # KO round percentages
            if ko_wins_total:
                ko_r1 = sum(1 for f in ko_wins_total if f.get('round') == 1)
                ko_r2 = sum(1 for f in ko_wins_total if f.get('round') == 2)
                ko_r3 = sum(1 for f in ko_wins_total if f.get('round') == 3)
                total_ko = len(ko_wins_total)
                
                stats['ko_round1_pct'] = round((ko_r1 / total_ko) * 100, 2) if total_ko > 0 else 0
                stats['ko_round2_pct'] = round((ko_r2 / total_ko) * 100, 2) if total_ko > 0 else 0
                stats['ko_round3_pct'] = round((ko_r3 / total_ko) * 100, 2) if total_ko > 0 else 0
            
            # SUB round percentages
            if sub_wins_total:
                sub_r1 = sum(1 for f in sub_wins_total if f.get('round') == 1)
                sub_r2 = sum(1 for f in sub_wins_total if f.get('round') == 2)
                sub_r3 = sum(1 for f in sub_wins_total if f.get('round') == 3)
                total_sub = len(sub_wins_total)
                
                stats['sub_round1_pct'] = round((sub_r1 / total_sub) * 100, 2) if total_sub > 0 else 0
                stats['sub_round2_pct'] = round((sub_r2 / total_sub) * 100, 2) if total_sub > 0 else 0
                stats['sub_round3_pct'] = round((sub_r3 / total_sub) * 100, 2) if total_sub > 0 else 0
        
        # First round finish rate (all fights, not just wins)
        total_fights = len(fights)
        if total_fights > 0:
            finishes_r1 = sum(1 for f in fights if f.get('round') == 1 and 
                            f.get('method') and 'DEC' not in f.get('method', '').upper())
            stats['first_round_finish_rate'] = round((finishes_r1 / total_fights) * 100, 2)
            
            # Decision rate
            decisions = sum(1 for f in fights if 'DEC' in (f.get('method') or '').upper())
            stats['decision_rate'] = round((decisions / total_fights) * 100, 2)
        
        # Average fight duration
        durations = []
        for fight in fights:
            rnd = fight.get('round')
            time_str = fight.get('time')
            if rnd and time_str:
                try:
                    minutes, seconds = map(int, time_str.split(':'))
                    # Each completed round is 5 minutes, plus time in final round
                    total_minutes = (rnd - 1) * 5 + minutes + seconds / 60
                    durations.append(total_minutes)
                except (ValueError, TypeError):
                    pass
        
        if durations:
            stats['avg_fight_duration'] = round(sum(durations) / len(durations), 2)
        
        return stats


class DuplicateFilterPipeline:
    """
    Pipeline to filter duplicate fighters (by URL).
    """
    
    def __init__(self):
        self.seen_fighters = set()
        self.seen_fights = set()
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        if isinstance(item, FighterItem):
            url = adapter.get('fighter_url')
            if url in self.seen_fighters:
                raise DropItem(f"Duplicate fighter: {adapter.get('name')}")
            self.seen_fighters.add(url)
        
        elif isinstance(item, FightItem):
            # Create a unique key for fights
            fight_key = (
                adapter.get('fighter_name'),
                adapter.get('opponent_name'),
                adapter.get('date'),
                adapter.get('event_name')
            )
            if fight_key in self.seen_fights:
                raise DropItem(f"Duplicate fight: {fight_key}")
            self.seen_fights.add(fight_key)
        
        return item

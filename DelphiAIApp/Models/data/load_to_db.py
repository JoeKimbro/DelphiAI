"""
Load Scraped CSV Data into PostgreSQL Database

This script loads the scraped CSV files (fighters.csv, career_stats.csv, fights.csv)
into the PostgreSQL database without needing to re-scrape.

Usage:
    python load_to_db.py                    # Load all CSVs
    python load_to_db.py --fighters-only    # Load only fighters
    python load_to_db.py --clear            # Clear tables before loading
    python load_to_db.py --dry-run          # Preview without inserting

Requirements:
    pip install psycopg2-binary python-dotenv pandas
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Database connection settings
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'dbname': os.getenv('DB_NAME', 'delphi_db'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
}

# CSV file paths
OUTPUT_DIR = Path(__file__).parent / 'output'
CSV_FILES = {
    'fighters': OUTPUT_DIR / 'fighters.csv',
    'career_stats': OUTPUT_DIR / 'career_stats.csv',
    'fights': OUTPUT_DIR / 'fights.csv',
    # New ML feature tables
    'elo_history': OUTPUT_DIR / 'elo_history.csv',
    'pre_ufc_career': OUTPUT_DIR / 'pre_ufc_career.csv',
    'opponent_quality': OUTPUT_DIR / 'opponent_quality.csv',
    'matchup_features': OUTPUT_DIR / 'matchup_features.csv',
    'point_in_time_stats': OUTPUT_DIR / 'point_in_time_stats.csv',
}


def connect_db():
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"[OK] Connected to database: {DB_CONFIG['dbname']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
        return conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        sys.exit(1)


def clear_tables(conn, include_ml_tables=True):
    """Clear all tables (in correct order due to foreign keys)."""
    cursor = conn.cursor()
    
    # Core tables
    core_tables = ['Fights', 'CareerStats', 'FighterStats']
    
    # ML feature tables (can be cleared independently)
    ml_tables = ['PointInTimeStats', 'MatchupFeatures', 'OpponentQuality', 'PreUfcCareer', 'EloHistory']
    
    tables_to_clear = ml_tables + core_tables if include_ml_tables else core_tables
    
    for table in tables_to_clear:
        try:
            cursor.execute(f"DELETE FROM {table}")
            print(f"   Cleared {table}")
        except Exception as e:
            print(f"   [WARN] Could not clear {table}: {e}")
    
    conn.commit()
    cursor.close()
    print("[OK] Tables cleared")


def parse_date(date_str):
    """Parse various date formats to Python date object."""
    if pd.isna(date_str) or date_str == '--' or date_str == '':
        return None
    
    # Clean the string
    date_str = str(date_str).strip()
    
    # Handle datetime with time component (e.g., "1993-11-12 00:00:00")
    if ' ' in date_str and ':' in date_str:
        date_str = date_str.split(' ')[0]  # Take just the date part
    
    formats = [
        "%Y-%m-%d",
        "%b. %d, %Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    return None


def normalize_name(name):
    """Normalize fighter name for matching across sources."""
    import unicodedata
    
    if not name or pd.isna(name):
        return None
    
    # Normalize unicode and remove accents
    name = unicodedata.normalize('NFKD', str(name))
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


def build_fighter_name_to_id(conn):
    """Build normalized fighter name -> FighterID lookup."""
    cursor = conn.cursor()
    cursor.execute("SELECT FighterID, Name FROM FighterStats WHERE Name IS NOT NULL")
    name_to_id = {}

    for fighter_id, name in cursor.fetchall():
        key = normalize_name(name)
        if key and key not in name_to_id:
            name_to_id[key] = fighter_id

    cursor.close()
    return name_to_id


def resolve_fighter_id(url, name, fighter_url_to_id, fighter_name_to_id):
    """Resolve FighterID from URL first, then normalized name fallback."""
    fighter_id = None
    resolved_by_name = False

    if pd.notna(url):
        fighter_id = fighter_url_to_id.get(url)

    if not fighter_id and pd.notna(name):
        key = normalize_name(name)
        fighter_id = fighter_name_to_id.get(key) if key else None
        resolved_by_name = fighter_id is not None

    if fighter_id and pd.notna(url):
        fighter_url_to_id[url] = fighter_id

    return fighter_id, resolved_by_name


def merge_fighter_records(df):
    """
    Merge fighter records from UFC.com and UFCStats by normalized name.
    
    Priority:
    - UFC.com: weight_class, nickname, place_of_birth, leg_reach, ufc_url
    - UFCStats: height, weight, reach, stance, dob, all fight stats, fighter_url
    """
    print("[MERGE] Merging fighter data from both sources...")
    
    # Add normalized name column
    df['name_key'] = df['name'].apply(normalize_name)
    
    # Separate by source
    ufc_official = df[df['source'] == 'ufc_official'].copy()
    ufcstats = df[df['source'] == 'ufcstats'].copy()
    other = df[~df['source'].isin(['ufc_official', 'ufcstats'])].copy()
    
    print(f"   UFC.com records: {len(ufc_official)}")
    print(f"   UFCStats records: {len(ufcstats)}")
    
    if len(ufc_official) == 0 or len(ufcstats) == 0:
        # No merge needed, return as-is
        print("   No merge needed (single source)")
        return df.drop(columns=['name_key'])
    
    # Create lookup from UFC.com data
    ufc_lookup = {}
    for _, row in ufc_official.iterrows():
        name_key = row['name_key']
        if name_key:
            ufc_lookup[name_key] = row.to_dict()
    
    # Merge UFC.com data into UFCStats records - UFC.com has PRIORITY for all fields
    merged_rows = []
    merged_count = 0
    
    for _, stats_row in ufcstats.iterrows():
        name_key = stats_row['name_key']
        row_dict = stats_row.to_dict()
        
        if name_key and name_key in ufc_lookup:
            ufc_row = ufc_lookup[name_key]
            
            # UFC.com is the PRIORITY source for all bio fields
            ufc_priority_fields = [
                'weight_class', 'nickname', 'place_of_birth', 'leg_reach',
                'height', 'weight', 'reach', 'stance', 'dob', 'age', 'is_active'
            ]
            
            for field in ufc_priority_fields:
                ufc_val = ufc_row.get(field)
                # Use UFC.com value if it exists (regardless of UFCStats value)
                if pd.notna(ufc_val) and ufc_val != '' and ufc_val != 'NaN':
                    row_dict[field] = ufc_val
            
            # Always get ufc_url from UFC.com
            if pd.notna(ufc_row.get('fighter_url')):
                row_dict['ufc_url'] = ufc_row['fighter_url']
            
            # For record fields, UFCStats is usually more accurate (keep UFCStats values)
            # Only fill from UFC.com if UFCStats doesn't have the data
            record_fields = ['wins', 'losses', 'draws', 'total_fights', 
                           'last_fight_date', 'days_since_last_fight']
            for field in record_fields:
                stats_val = row_dict.get(field)
                ufc_val = ufc_row.get(field)
                if (pd.isna(stats_val) or stats_val == '') and pd.notna(ufc_val):
                    row_dict[field] = ufc_val
            
            merged_count += 1
            # Mark as merged and remove from ufc_lookup so we don't duplicate
            del ufc_lookup[name_key]
        
        merged_rows.append(row_dict)
    
    # Add remaining UFC.com fighters (not in UFCStats)
    for name_key, ufc_row in ufc_lookup.items():
        # Copy ufc_url from fighter_url for UFC.com-only records
        ufc_row['ufc_url'] = ufc_row.get('fighter_url')
        merged_rows.append(ufc_row)
    
    # Combine all records
    result_df = pd.DataFrame(merged_rows)
    
    # Add back any other source records
    if len(other) > 0:
        result_df = pd.concat([result_df, other], ignore_index=True)
    
    # Drop the temporary name_key column
    if 'name_key' in result_df.columns:
        result_df = result_df.drop(columns=['name_key'])
    
    print(f"   Merged records: {merged_count}")
    print(f"   Total records after merge: {len(result_df)}")
    
    return result_df


def load_fighters(conn, dry_run=False):
    """Load fighters.csv into FighterStats table."""
    csv_path = CSV_FILES['fighters']
    
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping fighters")
        return {}
    
    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} raw records from {csv_path.name}")
    
    # Merge records from different sources by name
    df = merge_fighter_records(df)
    print(f"[FILE] Loading {len(df)} fighters after merge")
    
    if dry_run:
        print(df.head())
        return {}
    
    cursor = conn.cursor()
    fighter_url_to_id = {}
    inserted = 0
    updated = 0
    skipped = 0
    
    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        
        # Skip records with missing name or fighter_url
        name = row.get('name')
        if pd.isna(name) or name == '' or pd.isna(fighter_url) or fighter_url == '':
            skipped += 1
            continue
        
        # Parse dates
        dob_date = parse_date(row.get('dob'))
        last_fight_date = parse_date(row.get('last_fight_date'))
        
        # Check if fighter exists
        cursor.execute("SELECT FighterID FROM FighterStats WHERE FighterURL = %s", (fighter_url,))
        existing = cursor.fetchone()
        
        if existing:
            # Update
            cursor.execute("""
                UPDATE FighterStats SET
                    Name = %s, Height = %s, Weight = %s, Reach = %s, Stance = %s,
                    DOB = %s, Age = %s, WeightClass = %s, Nickname = %s,
                    PlaceOfBirth = %s, LegReach = %s, UFCUrl = %s,
                    TotalFights = %s, Wins = %s, Losses = %s, Draws = %s,
                    LastFightDate = %s, DaysSinceLastFight = %s, IsActive = %s,
                    Source = %s, ScrapedAt = %s, FightUpdatedAt = %s
                WHERE FighterURL = %s
            """, (
                row.get('name'),
                row.get('height'),
                row.get('weight'),
                row.get('reach'),
                row.get('stance'),
                dob_date,
                row.get('age') if pd.notna(row.get('age')) else None,
                row.get('weight_class'),
                row.get('nickname') if pd.notna(row.get('nickname')) else None,
                row.get('place_of_birth') if pd.notna(row.get('place_of_birth')) else None,
                row.get('leg_reach') if pd.notna(row.get('leg_reach')) else None,
                row.get('ufc_url') if pd.notna(row.get('ufc_url')) else None,
                row.get('total_fights') if pd.notna(row.get('total_fights')) else None,
                row.get('wins') if pd.notna(row.get('wins')) else None,
                row.get('losses') if pd.notna(row.get('losses')) else None,
                row.get('draws') if pd.notna(row.get('draws')) else None,
                last_fight_date,
                row.get('days_since_last_fight') if pd.notna(row.get('days_since_last_fight')) else None,
                row.get('is_active') == True or row.get('is_active') == 'True',
                row.get('source'),
                datetime.now(),
                datetime.now(),
                fighter_url,
            ))
            fighter_url_to_id[fighter_url] = existing[0]
            updated += 1
        else:
            # Insert
            cursor.execute("""
                INSERT INTO FighterStats (
                    Name, FighterURL, Height, Weight, Reach, Stance, DOB, Age,
                    WeightClass, Nickname, PlaceOfBirth, LegReach, UFCUrl,
                    TotalFights, Wins, Losses, Draws,
                    LastFightDate, DaysSinceLastFight, IsActive, Source, ScrapedAt, FightUpdatedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING FighterID
            """, (
                row.get('name'),
                fighter_url,
                row.get('height'),
                row.get('weight'),
                row.get('reach'),
                row.get('stance'),
                dob_date,
                row.get('age') if pd.notna(row.get('age')) else None,
                row.get('weight_class'),
                row.get('nickname') if pd.notna(row.get('nickname')) else None,
                row.get('place_of_birth') if pd.notna(row.get('place_of_birth')) else None,
                row.get('leg_reach') if pd.notna(row.get('leg_reach')) else None,
                row.get('ufc_url') if pd.notna(row.get('ufc_url')) else None,
                row.get('total_fights') if pd.notna(row.get('total_fights')) else None,
                row.get('wins') if pd.notna(row.get('wins')) else None,
                row.get('losses') if pd.notna(row.get('losses')) else None,
                row.get('draws') if pd.notna(row.get('draws')) else None,
                last_fight_date,
                row.get('days_since_last_fight') if pd.notna(row.get('days_since_last_fight')) else None,
                row.get('is_active') == True or row.get('is_active') == 'True',
                row.get('source'),
                datetime.now(),
                datetime.now(),
            ))
            fighter_id = cursor.fetchone()[0]
            fighter_url_to_id[fighter_url] = fighter_id
            inserted += 1
    
    conn.commit()
    cursor.close()
    print(f"   [OK] Fighters: {inserted} inserted, {updated} updated, {skipped} skipped")
    
    return fighter_url_to_id


def load_career_stats(conn, fighter_url_to_id, dry_run=False):
    """Load career_stats.csv into CareerStats table."""
    csv_path = CSV_FILES['career_stats']
    
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping career stats")
        return
    
    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} career stats from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    cursor = conn.cursor()
    inserted = 0
    updated = 0
    skipped = 0
    
    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        
        # Skip records with missing fighter_url
        if pd.isna(fighter_url) or fighter_url == '':
            skipped += 1
            continue
        
        fighter_id = fighter_url_to_id.get(fighter_url)
        
        if not fighter_id:
            # Try to find by URL in database
            cursor.execute("SELECT FighterID FROM FighterStats WHERE FighterURL = %s", (fighter_url,))
            result = cursor.fetchone()
            if result:
                fighter_id = result[0]
                fighter_url_to_id[fighter_url] = fighter_id
        
        if not fighter_id:
            skipped += 1
            continue
        
        # Check if career stats exist
        cursor.execute("SELECT CSID FROM CareerStats WHERE FighterID = %s", (fighter_id,))
        existing = cursor.fetchone()
        
        def safe_float(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except:
                return None
        
        def safe_int(val):
            if pd.isna(val):
                return None
            try:
                return int(val)
            except:
                return None
        
        if existing:
            # Update
            cursor.execute("""
                UPDATE CareerStats SET
                    FighterURL = %s, SLpM = %s, StrAcc = %s, SApM = %s, StrDef = %s,
                    TDAvg = %s, TDAcc = %s, TDDef = %s, SubAvg = %s,
                    WinStreak_Last3 = %s, WinsByKO_Last5 = %s, WinsBySub_Last5 = %s,
                    AvgFightDuration = %s, FirstRoundFinishRate = %s, DecisionRate = %s,
                    KO_Round1_Pct = %s, KO_Round2_Pct = %s, KO_Round3_Pct = %s,
                    Sub_Round1_Pct = %s, Sub_Round2_Pct = %s, Sub_Round3_Pct = %s,
                    EloRating = %s, PeakEloRating = %s,
                    Source = %s, ScrapedAt = %s, CareerUpdatedAt = %s
                WHERE FighterID = %s
            """, (
                fighter_url,
                safe_float(row.get('slpm')),
                safe_float(row.get('str_acc')),
                safe_float(row.get('sapm')),
                safe_float(row.get('str_def')),
                safe_float(row.get('td_avg')),
                safe_float(row.get('td_acc')),
                safe_float(row.get('td_def')),
                safe_float(row.get('sub_avg')),
                safe_int(row.get('win_streak_last3')),
                safe_int(row.get('wins_by_ko_last5')),
                safe_int(row.get('wins_by_sub_last5')),
                safe_float(row.get('avg_fight_duration')),
                safe_float(row.get('first_round_finish_rate')),
                safe_float(row.get('decision_rate')),
                safe_float(row.get('ko_round1_pct')),
                safe_float(row.get('ko_round2_pct')),
                safe_float(row.get('ko_round3_pct')),
                safe_float(row.get('sub_round1_pct')),
                safe_float(row.get('sub_round2_pct')),
                safe_float(row.get('sub_round3_pct')),
                safe_float(row.get('elo_rating')),
                safe_float(row.get('peak_elo')),
                row.get('source'),
                datetime.now(),
                datetime.now(),
                fighter_id,
            ))
            updated += 1
        else:
            # Insert
            cursor.execute("""
                INSERT INTO CareerStats (
                    FighterID, FighterURL, SLpM, StrAcc, SApM, StrDef,
                    TDAvg, TDAcc, TDDef, SubAvg,
                    WinStreak_Last3, WinsByKO_Last5, WinsBySub_Last5,
                    AvgFightDuration, FirstRoundFinishRate, DecisionRate,
                    KO_Round1_Pct, KO_Round2_Pct, KO_Round3_Pct,
                    Sub_Round1_Pct, Sub_Round2_Pct, Sub_Round3_Pct,
                    EloRating, PeakEloRating,
                    Source, ScrapedAt, CareerUpdatedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter_id,
                fighter_url,
                safe_float(row.get('slpm')),
                safe_float(row.get('str_acc')),
                safe_float(row.get('sapm')),
                safe_float(row.get('str_def')),
                safe_float(row.get('td_avg')),
                safe_float(row.get('td_acc')),
                safe_float(row.get('td_def')),
                safe_float(row.get('sub_avg')),
                safe_int(row.get('win_streak_last3')),
                safe_int(row.get('wins_by_ko_last5')),
                safe_int(row.get('wins_by_sub_last5')),
                safe_float(row.get('avg_fight_duration')),
                safe_float(row.get('first_round_finish_rate')),
                safe_float(row.get('decision_rate')),
                safe_float(row.get('ko_round1_pct')),
                safe_float(row.get('ko_round2_pct')),
                safe_float(row.get('ko_round3_pct')),
                safe_float(row.get('sub_round1_pct')),
                safe_float(row.get('sub_round2_pct')),
                safe_float(row.get('sub_round3_pct')),
                safe_float(row.get('elo_rating')),
                safe_float(row.get('peak_elo')),
                row.get('source'),
                datetime.now(),
                datetime.now(),
            ))
            inserted += 1
    
    conn.commit()
    cursor.close()
    print(f"   [OK] Career Stats: {inserted} inserted, {updated} updated, {skipped} skipped")


def load_fights(conn, fighter_url_to_id, dry_run=False):
    """Load fights.csv into Fights table."""
    csv_path = CSV_FILES['fights']
    
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping fights")
        return
    
    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} fights from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    cursor = conn.cursor()
    inserted = 0
    updated = 0
    skipped = 0
    resolved_name_fighter = 0
    resolved_name_opponent = 0
    unresolved_examples = []
    
    # First, make sure we have all fighter URLs mapped (both UFCStats and UFC.com URLs)
    cursor.execute("""
        SELECT FighterID, FighterURL, UFCUrl
        FROM FighterStats
        WHERE FighterURL IS NOT NULL OR UFCUrl IS NOT NULL
    """)
    for fighter_id, fighter_url, ufc_url in cursor.fetchall():
        if fighter_url:
            fighter_url_to_id[fighter_url] = fighter_id
        if ufc_url:
            fighter_url_to_id[ufc_url] = fighter_id

    fighter_name_to_id = build_fighter_name_to_id(conn)
    
    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        opponent_url = row.get('opponent_url')
        fight_url = row.get('fight_url')

        fighter_name = row.get('fighter_name')
        opponent_name = row.get('opponent_name')

        fighter_id, fighter_by_name = resolve_fighter_id(
            fighter_url, fighter_name, fighter_url_to_id, fighter_name_to_id
        )
        opponent_id, opponent_by_name = resolve_fighter_id(
            opponent_url, opponent_name, fighter_url_to_id, fighter_name_to_id
        )
        if fighter_by_name:
            resolved_name_fighter += 1
        if opponent_by_name:
            resolved_name_opponent += 1
        
        if not fighter_id:
            skipped += 1
            if len(unresolved_examples) < 10:
                unresolved_examples.append({
                    'fighter_name': fighter_name,
                    'fighter_url': fighter_url,
                    'opponent_name': opponent_name,
                    'opponent_url': opponent_url,
                    'fight_url': fight_url,
                })
            continue
        
        # Determine winner_id
        winner_id = None
        result = row.get('result')
        if result == 'win':
            winner_id = fighter_id
        elif result == 'loss' and opponent_id:
            winner_id = opponent_id
        
        # Parse fight date
        fight_date = parse_date(row.get('date'))
        
        # Check if fight exists
        if pd.notna(fight_url):
            cursor.execute("SELECT FightID FROM Fights WHERE FightURL = %s", (fight_url,))
        else:
            cursor.execute(
                "SELECT FightID FROM Fights WHERE FighterID = %s AND OpponentID = %s AND Date = %s",
                (fighter_id, opponent_id, fight_date)
            )
        existing = cursor.fetchone()
        
        def safe_int(val):
            if pd.isna(val):
                return None
            try:
                return int(val)
            except:
                return None
        
        def safe_str(val):
            if pd.isna(val):
                return None
            return str(val)
        
        is_title = row.get('is_title_fight')
        is_title_bool = is_title == True or is_title == 'True' or is_title == 1
        
        if existing:
            # Update
            cursor.execute("""
                UPDATE Fights SET
                    FighterURL = %s, FighterName = %s, OpponentID = %s, OpponentURL = %s,
                    OpponentName = %s, WinnerID = %s, WinnerName = %s, Result = %s,
                    Date = %s, EventName = %s, EventURL = %s, Method = %s, MethodDetail = %s,
                    Round = %s, Time = %s, Knockdowns = %s, SigStrikes = %s, Takedowns = %s,
                    SubAttempts = %s, IsTitleFight = %s, Source = %s, ScrapedAt = %s
                WHERE FightID = %s
            """, (
                fighter_url,
                safe_str(row.get('fighter_name')),
                opponent_id,
                safe_str(opponent_url),
                safe_str(row.get('opponent_name')),
                winner_id,
                safe_str(row.get('winner_name')),
                safe_str(result),
                fight_date,
                safe_str(row.get('event_name')),
                safe_str(row.get('event_url')),
                safe_str(row.get('method')),
                safe_str(row.get('method_detail')),
                safe_int(row.get('round')),
                safe_str(row.get('time')),
                safe_str(row.get('knockdowns')),
                safe_str(row.get('sig_strikes')),
                safe_str(row.get('takedowns')),
                safe_str(row.get('sub_attempts')),
                is_title_bool,
                safe_str(row.get('source')),
                datetime.now(),
                existing[0],
            ))
            updated += 1
        else:
            # Insert
            cursor.execute("""
                INSERT INTO Fights (
                    FighterID, FighterURL, FighterName, OpponentID, OpponentURL, OpponentName,
                    WinnerID, WinnerName, Result, Date, EventName, EventURL, FightURL,
                    Method, MethodDetail, Round, Time, Knockdowns, SigStrikes, Takedowns,
                    SubAttempts, IsTitleFight, Source, ScrapedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter_id,
                fighter_url,
                safe_str(row.get('fighter_name')),
                opponent_id,
                safe_str(opponent_url),
                safe_str(row.get('opponent_name')),
                winner_id,
                safe_str(row.get('winner_name')),
                safe_str(result),
                fight_date,
                safe_str(row.get('event_name')),
                safe_str(row.get('event_url')),
                safe_str(fight_url),
                safe_str(row.get('method')),
                safe_str(row.get('method_detail')),
                safe_int(row.get('round')),
                safe_str(row.get('time')),
                safe_str(row.get('knockdowns')),
                safe_str(row.get('sig_strikes')),
                safe_str(row.get('takedowns')),
                safe_str(row.get('sub_attempts')),
                is_title_bool,
                safe_str(row.get('source')),
                datetime.now(),
            ))
            inserted += 1
    
    conn.commit()
    cursor.close()
    print(f"   [OK] Fights: {inserted} inserted, {updated} updated, {skipped} skipped")
    print(f"   [RESOLVE] Name fallback used -> fighter: {resolved_name_fighter}, opponent: {resolved_name_opponent}")
    if unresolved_examples:
        print("   [WARN] Sample unresolved fights (showing up to 10):")
        for ex in unresolved_examples:
            print(
                f"      fighter='{ex['fighter_name']}' ({ex['fighter_url']}) vs "
                f"opponent='{ex['opponent_name']}' ({ex['opponent_url']}) fight_url={ex['fight_url']}"
            )


def load_elo_history(conn, fighter_url_to_id, dry_run=False):
    """Load elo_history.csv into EloHistory table."""
    csv_path = CSV_FILES['elo_history']
    
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping ELO history")
        return
    
    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} ELO history records from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    
    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        opponent_url = row.get('opponent_url')
        
        fighter_id = fighter_url_to_id.get(fighter_url)
        opponent_id = fighter_url_to_id.get(opponent_url) if pd.notna(opponent_url) else None
        
        if not fighter_id:
            skipped += 1
            continue
        
        fight_date = parse_date(row.get('fight_date'))
        
        def safe_float(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except:
                return None
        
        # Skip records without valid fight_date
        if fight_date is None:
            skipped += 1
            continue
        
        try:
            cursor.execute("""
                INSERT INTO EloHistory (
                    FighterID, FighterURL, FightDate, OpponentID, OpponentURL,
                    EloBeforeFight, OpponentEloBeforeFight, EloAfterFight, EloChange,
                    Result, Method, ExpectedWinProb, EloSource, CalculatedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter_id,
                fighter_url,
                fight_date,
                opponent_id,
                opponent_url if pd.notna(opponent_url) else None,
                safe_float(row.get('elo_before_fight')),
                safe_float(row.get('opponent_elo_before_fight')),
                safe_float(row.get('elo_after_fight')),
                safe_float(row.get('elo_change')),
                row.get('result') if pd.notna(row.get('result')) else None,
                row.get('method') if pd.notna(row.get('method')) else None,
                safe_float(row.get('expected_win_prob')),
                row.get('elo_source', 'ufc_fights'),
                datetime.now(),
            ))
            inserted += 1
        except Exception as e:
            conn.rollback()  # Rollback to continue processing
            if skipped < 5:  # Only print first 5 errors
                print(f"   [ERROR] ELO History insert failed: {e}")
            skipped += 1
    
    conn.commit()
    cursor.close()
    print(f"   [OK] ELO History: {inserted} inserted, {skipped} skipped")


def load_pre_ufc_career(conn, fighter_url_to_id, dry_run=False):
    """Load pre_ufc_career.csv into PreUfcCareer table."""
    csv_path = CSV_FILES['pre_ufc_career']
    
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping pre-UFC career")
        return
    
    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} pre-UFC career records from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    cursor = conn.cursor()
    inserted = 0
    updated = 0
    skipped = 0
    
    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        fighter_id = fighter_url_to_id.get(fighter_url)
        
        if not fighter_id:
            skipped += 1
            continue
        
        def safe_float(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except:
                return None
        
        def safe_int(val):
            if pd.isna(val):
                return None
            try:
                return int(val)
            except:
                return None
        
        # Check if exists
        cursor.execute("SELECT PreUfcID FROM PreUfcCareer WHERE FighterID = %s", (fighter_id,))
        existing = cursor.fetchone()
        
        # Build breakdown JSON
        import json
        breakdown = {
            'record_adjustment': safe_float(row.get('record_adjustment')),
            'win_rate_bonus': safe_float(row.get('win_rate_bonus')),
            'career_efficiency_adj': safe_float(row.get('career_efficiency_adj')),
            'age_factor_adj': safe_float(row.get('age_factor_adj')),
            'recency_adj': safe_float(row.get('recency_adj')),
            'total_adjustment': safe_float(row.get('total_adjustment')),
        }
        
        if existing:
            cursor.execute("""
                UPDATE PreUfcCareer SET
                    FighterURL = %s, PreUfcWins = %s, PreUfcLosses = %s, PreUfcDraws = %s,
                    PreUfcTotalFights = %s, EstimatedInitialElo = %s, EloEstimationMethod = %s,
                    EloEstimationBreakdown = %s, OrgQualityTier = %s, PrimaryOrg = %s,
                    DataConfidence = %s, UpdatedAt = %s
                WHERE FighterID = %s
            """, (
                fighter_url,
                safe_int(row.get('pre_ufc_wins')),
                safe_int(row.get('pre_ufc_losses')),
                safe_int(row.get('pre_ufc_draws')),
                safe_int(row.get('pre_ufc_total_fights')),
                safe_float(row.get('estimated_initial_elo')),
                row.get('elo_estimation_method', 'enhanced'),
                json.dumps(breakdown),
                safe_int(row.get('org_quality_tier')),
                row.get('primary_org') if pd.notna(row.get('primary_org')) else None,
                row.get('data_confidence', 'medium'),
                datetime.now(),
                fighter_id,
            ))
            updated += 1
        else:
            cursor.execute("""
                INSERT INTO PreUfcCareer (
                    FighterID, FighterURL, PreUfcWins, PreUfcLosses, PreUfcDraws,
                    PreUfcTotalFights, EstimatedInitialElo, EloEstimationMethod,
                    EloEstimationBreakdown, OrgQualityTier, PrimaryOrg,
                    DataConfidence, CreatedAt, UpdatedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter_id,
                fighter_url,
                safe_int(row.get('pre_ufc_wins')),
                safe_int(row.get('pre_ufc_losses')),
                safe_int(row.get('pre_ufc_draws')),
                safe_int(row.get('pre_ufc_total_fights')),
                safe_float(row.get('estimated_initial_elo')),
                row.get('elo_estimation_method', 'enhanced'),
                json.dumps(breakdown),
                safe_int(row.get('org_quality_tier')),
                row.get('primary_org') if pd.notna(row.get('primary_org')) else None,
                row.get('data_confidence', 'medium'),
                datetime.now(),
                datetime.now(),
            ))
            inserted += 1
    
    conn.commit()
    cursor.close()
    print(f"   [OK] Pre-UFC Career: {inserted} inserted, {updated} updated, {skipped} skipped")


def load_opponent_quality(conn, fighter_url_to_id, dry_run=False):
    """Load opponent_quality.csv into OpponentQuality table."""
    csv_path = CSV_FILES['opponent_quality']
    
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping opponent quality")
        return
    
    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} opponent quality records from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    cursor = conn.cursor()
    inserted = 0
    updated = 0
    skipped = 0
    
    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        fighter_id = fighter_url_to_id.get(fighter_url)
        
        if not fighter_id:
            skipped += 1
            continue
        
        def safe_float(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except:
                return None
        
        def safe_int(val):
            if pd.isna(val):
                return None
            try:
                return int(val)
            except:
                return None
        
        # Check if exists
        cursor.execute("SELECT OQID FROM OpponentQuality WHERE FighterID = %s", (fighter_id,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute("""
                UPDATE OpponentQuality SET
                    FighterURL = %s, AvgOpponentElo = %s, AvgOpponentEloAtFightTime = %s,
                    EliteOpponentWins = %s, EliteOpponentLosses = %s,
                    GoodOpponentWins = %s, GoodOpponentLosses = %s,
                    AverageOpponentWins = %s, AverageOpponentLosses = %s,
                    BelowAverageWins = %s, BelowAverageLosses = %s,
                    EliteWinRate = %s, QualityWinIndex = %s,
                    RecentAvgOpponentElo = %s, RecentEliteWins = %s,
                    ScheduleStrengthRank = %s, ScheduleStrengthPercentile = %s,
                    LastCalculated = %s, FightsAnalyzed = %s
                WHERE FighterID = %s
            """, (
                fighter_url,
                safe_float(row.get('avg_opponent_elo')),
                safe_float(row.get('avg_opponent_elo_at_fight_time')),
                safe_int(row.get('elite_wins')),
                safe_int(row.get('elite_losses')),
                safe_int(row.get('good_wins')),
                safe_int(row.get('good_losses')),
                safe_int(row.get('average_wins')),
                safe_int(row.get('average_losses')),
                safe_int(row.get('below_average_wins')),
                safe_int(row.get('below_average_losses')),
                safe_float(row.get('elite_win_rate')),
                safe_float(row.get('quality_win_index')),
                safe_float(row.get('recent_avg_opponent_elo')),
                safe_int(row.get('recent_elite_wins')),
                safe_int(row.get('schedule_strength_rank')),
                safe_float(row.get('schedule_strength_percentile')),
                datetime.now(),
                safe_int(row.get('fights_analyzed')),
                fighter_id,
            ))
            updated += 1
        else:
            cursor.execute("""
                INSERT INTO OpponentQuality (
                    FighterID, FighterURL, AvgOpponentElo, AvgOpponentEloAtFightTime,
                    EliteOpponentWins, EliteOpponentLosses,
                    GoodOpponentWins, GoodOpponentLosses,
                    AverageOpponentWins, AverageOpponentLosses,
                    BelowAverageWins, BelowAverageLosses,
                    EliteWinRate, QualityWinIndex,
                    RecentAvgOpponentElo, RecentEliteWins,
                    ScheduleStrengthRank, ScheduleStrengthPercentile,
                    LastCalculated, FightsAnalyzed
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter_id,
                fighter_url,
                safe_float(row.get('avg_opponent_elo')),
                safe_float(row.get('avg_opponent_elo_at_fight_time')),
                safe_int(row.get('elite_wins')),
                safe_int(row.get('elite_losses')),
                safe_int(row.get('good_wins')),
                safe_int(row.get('good_losses')),
                safe_int(row.get('average_wins')),
                safe_int(row.get('average_losses')),
                safe_int(row.get('below_average_wins')),
                safe_int(row.get('below_average_losses')),
                safe_float(row.get('elite_win_rate')),
                safe_float(row.get('quality_win_index')),
                safe_float(row.get('recent_avg_opponent_elo')),
                safe_int(row.get('recent_elite_wins')),
                safe_int(row.get('schedule_strength_rank')),
                safe_float(row.get('schedule_strength_percentile')),
                datetime.now(),
                safe_int(row.get('fights_analyzed')),
            ))
            inserted += 1
    
    conn.commit()
    cursor.close()
    print(f"   [OK] Opponent Quality: {inserted} inserted, {updated} updated, {skipped} skipped")


def load_matchup_features(conn, fighter_url_to_id, dry_run=False):
    """Load matchup_features.csv into MatchupFeatures table."""
    csv_path = CSV_FILES['matchup_features']
    
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping matchup features")
        return
    
    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} matchup features from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    cursor = conn.cursor()
    inserted = 0
    updated = 0
    skipped = 0
    
    for _, row in df.iterrows():
        fighter1_url = row.get('fighter1_url')
        fighter2_url = row.get('fighter2_url')
        
        fighter1_id = fighter_url_to_id.get(fighter1_url)
        fighter2_id = fighter_url_to_id.get(fighter2_url)
        
        if not fighter1_id or not fighter2_id:
            skipped += 1
            continue
        
        def safe_float(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except:
                return None
        
        def safe_int(val):
            if pd.isna(val):
                return None
            try:
                return int(val)
            except:
                return None
        
        # Check if exists
        cursor.execute(
            "SELECT MatchupFeatureID FROM MatchupFeatures WHERE Fighter1ID = %s AND Fighter2ID = %s",
            (fighter1_id, fighter2_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute("""
                UPDATE MatchupFeatures SET
                    HeightDiff_cm = %s, ReachDiff_cm = %s, LegReachDiff_cm = %s, AgeDiff = %s,
                    EloDiff = %s, PeakEloDiff = %s,
                    SLpMDiff = %s, SApMDiff = %s, StrAccDiff = %s, StrDefDiff = %s,
                    TDAvgDiff = %s, TDAccDiff = %s, TDDefDiff = %s, SubAvgDiff = %s,
                    OpponentQualityDiff = %s, WinStreakDiff = %s,
                    DaysSinceLastFightDiff = %s, TotalFightsDiff = %s,
                    Fighter1Style = %s, Fighter2Style = %s, StyleMatchupAdvantage = %s,
                    CalculatedAt = %s, IsStale = %s
                WHERE MatchupFeatureID = %s
            """, (
                safe_float(row.get('height_diff_cm')),
                safe_float(row.get('reach_diff_cm')),
                safe_float(row.get('leg_reach_diff_cm')),
                safe_float(row.get('age_diff')),
                safe_float(row.get('elo_diff')),
                safe_float(row.get('peak_elo_diff')),
                safe_float(row.get('slpm_diff')),
                safe_float(row.get('sapm_diff')),
                safe_float(row.get('str_acc_diff')),
                safe_float(row.get('str_def_diff')),
                safe_float(row.get('td_avg_diff')),
                safe_float(row.get('td_acc_diff')),
                safe_float(row.get('td_def_diff')),
                safe_float(row.get('sub_avg_diff')),
                safe_float(row.get('opponent_quality_diff')),
                safe_int(row.get('win_streak_diff')),
                safe_int(row.get('days_since_fight_diff')),
                safe_int(row.get('total_fights_diff')),
                row.get('fighter1_style') if pd.notna(row.get('fighter1_style')) else None,
                row.get('fighter2_style') if pd.notna(row.get('fighter2_style')) else None,
                safe_int(row.get('style_matchup_advantage')),
                datetime.now(),
                False,
                existing[0],
            ))
            updated += 1
        else:
            cursor.execute("""
                INSERT INTO MatchupFeatures (
                    Fighter1ID, Fighter2ID,
                    HeightDiff_cm, ReachDiff_cm, LegReachDiff_cm, AgeDiff,
                    EloDiff, PeakEloDiff,
                    SLpMDiff, SApMDiff, StrAccDiff, StrDefDiff,
                    TDAvgDiff, TDAccDiff, TDDefDiff, SubAvgDiff,
                    OpponentQualityDiff, WinStreakDiff,
                    DaysSinceLastFightDiff, TotalFightsDiff,
                    Fighter1Style, Fighter2Style, StyleMatchupAdvantage,
                    CalculatedAt, IsStale
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter1_id,
                fighter2_id,
                safe_float(row.get('height_diff_cm')),
                safe_float(row.get('reach_diff_cm')),
                safe_float(row.get('leg_reach_diff_cm')),
                safe_float(row.get('age_diff')),
                safe_float(row.get('elo_diff')),
                safe_float(row.get('peak_elo_diff')),
                safe_float(row.get('slpm_diff')),
                safe_float(row.get('sapm_diff')),
                safe_float(row.get('str_acc_diff')),
                safe_float(row.get('str_def_diff')),
                safe_float(row.get('td_avg_diff')),
                safe_float(row.get('td_acc_diff')),
                safe_float(row.get('td_def_diff')),
                safe_float(row.get('sub_avg_diff')),
                safe_float(row.get('opponent_quality_diff')),
                safe_int(row.get('win_streak_diff')),
                safe_int(row.get('days_since_fight_diff')),
                safe_int(row.get('total_fights_diff')),
                row.get('fighter1_style') if pd.notna(row.get('fighter1_style')) else None,
                row.get('fighter2_style') if pd.notna(row.get('fighter2_style')) else None,
                safe_int(row.get('style_matchup_advantage')),
                datetime.now(),
                False,
            ))
            inserted += 1
    
    conn.commit()
    cursor.close()
    print(f"   [OK] Matchup Features: {inserted} inserted, {updated} updated, {skipped} skipped")


def load_point_in_time_stats(conn, fighter_url_to_id, dry_run=False):
    """Load point_in_time_stats.csv into PointInTimeStats table."""
    csv_path = CSV_FILES['point_in_time_stats']
    
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping point-in-time stats")
        return
    
    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} point-in-time stat records from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    
    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        fighter_id = fighter_url_to_id.get(fighter_url)
        
        if not fighter_id:
            skipped += 1
            continue
        
        fight_date = parse_date(row.get('fight_date'))
        
        if fight_date is None:
            skipped += 1
            continue
        
        def safe_float(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except:
                return None
        
        def safe_int(val):
            if pd.isna(val):
                return None
            try:
                return int(val)
            except:
                return None
        
        try:
            cursor.execute("""
                INSERT INTO PointInTimeStats (
                    FighterID, FighterURL, FightDate,
                    FightsBefore, WinsBefore, LossesBefore, WinRateBefore,
                    PIT_SLpM, PIT_StrAcc, PIT_TDAvg, PIT_SubAvg, PIT_KDRate,
                    RecentWinRate, AvgFightTime, FinishRate, HasPriorData, CalculatedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                fighter_id,
                fighter_url,
                fight_date,
                safe_int(row.get('fights_before')),
                safe_int(row.get('wins_before')),
                safe_int(row.get('losses_before')),
                safe_float(row.get('win_rate_before')),
                safe_float(row.get('pit_slpm')),
                safe_float(row.get('pit_str_acc')),
                safe_float(row.get('pit_td_avg')),
                safe_float(row.get('pit_sub_avg')),
                safe_float(row.get('pit_kd_rate')),
                safe_float(row.get('recent_win_rate')),
                safe_float(row.get('avg_fight_time')),
                safe_float(row.get('finish_rate')),
                row.get('has_prior_data', False),
                datetime.now(),
            ))
            inserted += 1
        except Exception as e:
            conn.rollback()
            if skipped < 5:
                print(f"   [ERROR] PIT Stats insert failed: {e}")
            skipped += 1
    
    conn.commit()
    cursor.close()
    print(f"   [OK] Point-in-Time Stats: {inserted} inserted, {skipped} skipped")


def show_summary(conn):
    """Show database summary after loading."""
    cursor = conn.cursor()
    
    print("\n" + "=" * 50)
    print("DATABASE SUMMARY")
    print("=" * 50)
    
    # Core tables
    print("\nCore Tables:")
    cursor.execute("SELECT COUNT(*) FROM FighterStats")
    print(f"   FighterStats:    {cursor.fetchone()[0]} records")
    
    cursor.execute("SELECT COUNT(*) FROM CareerStats")
    print(f"   CareerStats:     {cursor.fetchone()[0]} records")
    
    cursor.execute("SELECT COUNT(*) FROM Fights")
    print(f"   Fights:          {cursor.fetchone()[0]} records")
    
    # ML feature tables
    print("\nML Feature Tables:")
    
    ml_tables = [
        ('EloHistory', 'EloHistory'),
        ('PreUfcCareer', 'PreUfcCareer'),
        ('OpponentQuality', 'OpponentQuality'),
        ('MatchupFeatures', 'MatchupFeatures'),
        ('PointInTimeStats', 'PointInTimeStats'),
    ]
    
    for display_name, table_name in ml_tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   {display_name}: {count:>8} records")
        except Exception:
            print(f"   {display_name}: (table not found)")
    
    # Summary stats
    print("\nSummary:")
    cursor.execute("SELECT COUNT(*) FROM FighterStats WHERE IsActive = true")
    print(f"   Active Fighters: {cursor.fetchone()[0]}")
    
    print("=" * 50)
    cursor.close()


def main():
    parser = argparse.ArgumentParser(description='Load scraped CSV data into PostgreSQL')
    parser.add_argument('--fighters-only', action='store_true', help='Load only fighters')
    parser.add_argument('--core-only', action='store_true', help='Load only core tables (fighters, career, fights)')
    parser.add_argument('--ml-only', action='store_true', help='Load only ML feature tables')
    parser.add_argument('--clear', action='store_true', help='Clear tables before loading')
    parser.add_argument('--dry-run', action='store_true', help='Preview without inserting')
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("CSV TO DATABASE LOADER")
    print("=" * 50 + "\n")
    
    # Check CSV files exist
    for name, path in CSV_FILES.items():
        if path.exists():
            print(f"[OK] Found {path.name}")
        else:
            print(f"[--] Missing {path.name}")
    print()
    
    # Connect to database
    conn = connect_db()
    
    try:
        # Clear tables if requested
        if args.clear and not args.dry_run:
            print("\n[CLEAR] Clearing tables...")
            clear_tables(conn, include_ml_tables=not args.core_only)
        
        # Load data
        print("\n[LOAD] Loading data...\n")
        
        # Always need fighter URL to ID mapping
        fighter_url_to_id = {}
        
        if not args.ml_only:
            # Load core tables
            fighter_url_to_id = load_fighters(conn, args.dry_run)
            
            if not args.fighters_only:
                load_career_stats(conn, fighter_url_to_id, args.dry_run)
                load_fights(conn, fighter_url_to_id, args.dry_run)
        else:
            # Build fighter URL to ID map from existing data
            cursor = conn.cursor()
            cursor.execute("SELECT FighterID, FighterURL FROM FighterStats WHERE FighterURL IS NOT NULL")
            for row in cursor.fetchall():
                fighter_url_to_id[row[1]] = row[0]
            cursor.close()
            print(f"[INFO] Loaded {len(fighter_url_to_id)} fighter URL mappings from database")
        
        # Load ML feature tables
        if not args.fighters_only and not args.core_only:
            print("\n[ML] Loading ML feature tables...\n")
            load_elo_history(conn, fighter_url_to_id, args.dry_run)
            load_pre_ufc_career(conn, fighter_url_to_id, args.dry_run)
            load_opponent_quality(conn, fighter_url_to_id, args.dry_run)
            load_matchup_features(conn, fighter_url_to_id, args.dry_run)
            load_point_in_time_stats(conn, fighter_url_to_id, args.dry_run)
        
        # Show summary
        if not args.dry_run:
            show_summary(conn)
        
        print("\n[OK] Done!")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()

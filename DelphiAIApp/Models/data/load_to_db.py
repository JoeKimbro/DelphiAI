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
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Ensure the Models package root is importable so `db.*` helpers resolve
# regardless of the working directory this script is launched from.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from db.sql_identifiers import safe_identifier, ALLOWED_TABLES

# Database connection — DATABASE_URL takes priority (CI/production),
# individual DB_* vars are the local-dev fallback.
_DATABASE_URL = os.getenv('DATABASE_URL', '')
DB_CONFIG = {'dsn': _DATABASE_URL} if _DATABASE_URL else {
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


def _sf(val):
    if pd.isna(val): return None
    try: return float(val)
    except: return None

def _si(val):
    if pd.isna(val): return None
    try: return int(val)
    except: return None

def _ss(val):
    if pd.isna(val): return None
    return str(val)


def connect_db():
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        db_label = DB_CONFIG.get('dbname', DB_CONFIG.get('dsn', 'db')[:40])
        print(f"[OK] Connected to database: {db_label}")
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
            cursor.execute(f"DELETE FROM {safe_identifier(table, ALLOWED_TABLES)}")
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
    """Load fighters.csv into FighterStats table using batch upserts."""
    csv_path = CSV_FILES['fighters']
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping fighters")
        return {}

    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} raw records from {csv_path.name}")
    df = merge_fighter_records(df)
    print(f"[FILE] Loading {len(df)} fighters after merge")

    if dry_run:
        print(df.head())
        return {}

    now = datetime.now()
    rows = []
    fighter_urls = []
    skipped = 0

    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        name = row.get('name')
        if pd.isna(name) or name == '' or pd.isna(fighter_url) or fighter_url == '':
            skipped += 1
            continue
        rows.append((
            _ss(name), fighter_url,
            _ss(row.get('height')), _ss(row.get('weight')), _ss(row.get('reach')), _ss(row.get('stance')),
            parse_date(row.get('dob')), _si(row.get('age')),
            _ss(row.get('weight_class')), _ss(row.get('nickname')),
            _ss(row.get('place_of_birth')), _ss(row.get('leg_reach')), _ss(row.get('ufc_url')),
            _si(row.get('total_fights')), _si(row.get('wins')), _si(row.get('losses')), _si(row.get('draws')),
            parse_date(row.get('last_fight_date')), _si(row.get('days_since_last_fight')),
            row.get('is_active') in (True, 'True', 1),
            _ss(row.get('source')), now, now,
        ))
        fighter_urls.append(fighter_url)

    cursor = conn.cursor()
    execute_values(cursor, """
        INSERT INTO FighterStats (
            Name, FighterURL, Height, Weight, Reach, Stance, DOB, Age,
            WeightClass, Nickname, PlaceOfBirth, LegReach, UFCUrl,
            TotalFights, Wins, Losses, Draws,
            LastFightDate, DaysSinceLastFight, IsActive, Source, ScrapedAt, FightUpdatedAt
        ) VALUES %s
        ON CONFLICT (FighterURL) DO UPDATE SET
            Name = EXCLUDED.Name, Height = EXCLUDED.Height, Weight = EXCLUDED.Weight,
            Reach = EXCLUDED.Reach, Stance = EXCLUDED.Stance, DOB = EXCLUDED.DOB,
            Age = EXCLUDED.Age, WeightClass = EXCLUDED.WeightClass,
            Nickname = EXCLUDED.Nickname, PlaceOfBirth = EXCLUDED.PlaceOfBirth,
            LegReach = EXCLUDED.LegReach, UFCUrl = EXCLUDED.UFCUrl,
            TotalFights = EXCLUDED.TotalFights, Wins = EXCLUDED.Wins,
            Losses = EXCLUDED.Losses, Draws = EXCLUDED.Draws,
            LastFightDate = EXCLUDED.LastFightDate,
            DaysSinceLastFight = EXCLUDED.DaysSinceLastFight,
            IsActive = EXCLUDED.IsActive, Source = EXCLUDED.Source,
            ScrapedAt = EXCLUDED.ScrapedAt, FightUpdatedAt = EXCLUDED.FightUpdatedAt
    """, rows, page_size=200)
    conn.commit()

    cursor.execute(
        "SELECT FighterID, FighterURL FROM FighterStats WHERE FighterURL = ANY(%s)",
        (fighter_urls,)
    )
    fighter_url_to_id = {url: fid for fid, url in cursor.fetchall()}
    cursor.close()

    print(f"   [OK] Fighters: {len(rows)} upserted, {skipped} skipped")
    return fighter_url_to_id


def load_career_stats(conn, fighter_url_to_id, dry_run=False):
    """Load career_stats.csv into CareerStats table using batch upserts."""
    csv_path = CSV_FILES['career_stats']
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping career stats")
        return

    df = pd.read_csv(csv_path)
    print(f"[FILE] Loading {len(df)} career stats from {csv_path.name}")

    if dry_run:
        print(df.head())
        return

    now = datetime.now()
    rows = []
    skipped = 0

    # Bulk-resolve any URLs not yet in fighter_url_to_id
    missing_urls = [row.get('fighter_url') for _, row in df.iterrows()
                    if pd.notna(row.get('fighter_url')) and row.get('fighter_url') not in fighter_url_to_id]
    if missing_urls:
        cursor = conn.cursor()
        cursor.execute("SELECT FighterID, FighterURL FROM FighterStats WHERE FighterURL = ANY(%s)", (missing_urls,))
        for fid, url in cursor.fetchall():
            fighter_url_to_id[url] = fid
        cursor.close()

    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        if pd.isna(fighter_url) or fighter_url == '':
            skipped += 1
            continue
        fighter_id = fighter_url_to_id.get(fighter_url)
        if not fighter_id:
            skipped += 1
            continue
        rows.append((
            fighter_id, fighter_url,
            _sf(row.get('slpm')), _sf(row.get('str_acc')), _sf(row.get('sapm')), _sf(row.get('str_def')),
            _sf(row.get('td_avg')), _sf(row.get('td_acc')), _sf(row.get('td_def')), _sf(row.get('sub_avg')),
            _si(row.get('win_streak_last3')), _si(row.get('wins_by_ko_last5')), _si(row.get('wins_by_sub_last5')),
            _sf(row.get('avg_fight_duration')), _sf(row.get('first_round_finish_rate')), _sf(row.get('decision_rate')),
            _sf(row.get('ko_round1_pct')), _sf(row.get('ko_round2_pct')), _sf(row.get('ko_round3_pct')),
            _sf(row.get('sub_round1_pct')), _sf(row.get('sub_round2_pct')), _sf(row.get('sub_round3_pct')),
            _sf(row.get('elo_rating')), _sf(row.get('peak_elo')),
            _ss(row.get('source')), now, now,
        ))

    cursor = conn.cursor()
    execute_values(cursor, """
        INSERT INTO CareerStats (
            FighterID, FighterURL, SLpM, StrAcc, SApM, StrDef,
            TDAvg, TDAcc, TDDef, SubAvg,
            WinStreak_Last3, WinsByKO_Last5, WinsBySub_Last5,
            AvgFightDuration, FirstRoundFinishRate, DecisionRate,
            KO_Round1_Pct, KO_Round2_Pct, KO_Round3_Pct,
            Sub_Round1_Pct, Sub_Round2_Pct, Sub_Round3_Pct,
            EloRating, PeakEloRating,
            Source, ScrapedAt, CareerUpdatedAt
        ) VALUES %s
        ON CONFLICT (FighterID) DO UPDATE SET
            FighterURL = EXCLUDED.FighterURL,
            SLpM = EXCLUDED.SLpM, StrAcc = EXCLUDED.StrAcc,
            SApM = EXCLUDED.SApM, StrDef = EXCLUDED.StrDef,
            TDAvg = EXCLUDED.TDAvg, TDAcc = EXCLUDED.TDAcc,
            TDDef = EXCLUDED.TDDef, SubAvg = EXCLUDED.SubAvg,
            WinStreak_Last3 = EXCLUDED.WinStreak_Last3,
            WinsByKO_Last5 = EXCLUDED.WinsByKO_Last5,
            WinsBySub_Last5 = EXCLUDED.WinsBySub_Last5,
            AvgFightDuration = EXCLUDED.AvgFightDuration,
            FirstRoundFinishRate = EXCLUDED.FirstRoundFinishRate,
            DecisionRate = EXCLUDED.DecisionRate,
            KO_Round1_Pct = EXCLUDED.KO_Round1_Pct,
            KO_Round2_Pct = EXCLUDED.KO_Round2_Pct,
            KO_Round3_Pct = EXCLUDED.KO_Round3_Pct,
            Sub_Round1_Pct = EXCLUDED.Sub_Round1_Pct,
            Sub_Round2_Pct = EXCLUDED.Sub_Round2_Pct,
            Sub_Round3_Pct = EXCLUDED.Sub_Round3_Pct,
            EloRating = EXCLUDED.EloRating, PeakEloRating = EXCLUDED.PeakEloRating,
            Source = EXCLUDED.Source, ScrapedAt = EXCLUDED.ScrapedAt,
            CareerUpdatedAt = EXCLUDED.CareerUpdatedAt
    """, rows, page_size=200)
    conn.commit()
    cursor.close()
    print(f"   [OK] Career Stats: {len(rows)} upserted, {skipped} skipped")


def load_fights(conn, fighter_url_to_id, dry_run=False):
    """Load fights.csv into Fights table using batch upserts."""
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

    # Bulk-load all fighter URL mappings (UFCStats + UFC.com URLs)
    cursor.execute("SELECT FighterID, FighterURL, UFCUrl FROM FighterStats WHERE FighterURL IS NOT NULL OR UFCUrl IS NOT NULL")
    for fid, furl, uurl in cursor.fetchall():
        if furl: fighter_url_to_id[furl] = fid
        if uurl: fighter_url_to_id[uurl] = fid

    fighter_name_to_id = build_fighter_name_to_id(conn)

    now = datetime.now()
    rows_with_url = []   # fights that have a FightURL (can batch upsert)
    rows_without_url = []  # fights without FightURL (individual insert, rare)
    skipped = 0
    resolved_name_fighter = 0
    resolved_name_opponent = 0
    unresolved_examples = []

    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        opponent_url = row.get('opponent_url')
        fight_url = row.get('fight_url')

        fighter_id, fighter_by_name = resolve_fighter_id(
            fighter_url, row.get('fighter_name'), fighter_url_to_id, fighter_name_to_id)
        opponent_id, opponent_by_name = resolve_fighter_id(
            opponent_url, row.get('opponent_name'), fighter_url_to_id, fighter_name_to_id)

        if fighter_by_name: resolved_name_fighter += 1
        if opponent_by_name: resolved_name_opponent += 1

        if not fighter_id:
            skipped += 1
            if len(unresolved_examples) < 10:
                unresolved_examples.append(f"'{row.get('fighter_name')}' vs '{row.get('opponent_name')}' ({fight_url})")
            continue

        result = row.get('result')
        winner_id = fighter_id if result == 'win' else (opponent_id if result == 'loss' and opponent_id else None)
        is_title_bool = row.get('is_title_fight') in (True, 'True', 1)
        fight_date = parse_date(row.get('date'))

        t = (
            fighter_id, _ss(fighter_url), _ss(row.get('fighter_name')),
            opponent_id, _ss(opponent_url), _ss(row.get('opponent_name')),
            winner_id, _ss(row.get('winner_name')), _ss(result),
            fight_date, _ss(row.get('event_name')), _ss(row.get('event_url')), _ss(fight_url),
            _ss(row.get('method')), _ss(row.get('method_detail')),
            _si(row.get('round')), _ss(row.get('time')), _ss(row.get('knockdowns')),
            _ss(row.get('sig_strikes')), _ss(row.get('takedowns')), _ss(row.get('sub_attempts')),
            is_title_bool, _ss(row.get('source')), now,
        )

        if pd.notna(fight_url) and fight_url:
            rows_with_url.append(t)
        else:
            rows_without_url.append(t)

    COLS = """FighterID, FighterURL, FighterName, OpponentID, OpponentURL, OpponentName,
              WinnerID, WinnerName, Result, Date, EventName, EventURL, FightURL,
              Method, MethodDetail, Round, Time, Knockdowns, SigStrikes, Takedowns,
              SubAttempts, IsTitleFight, Source, ScrapedAt"""
    UPDATE_SET = """
        FighterURL = EXCLUDED.FighterURL, FighterName = EXCLUDED.FighterName,
        OpponentID = EXCLUDED.OpponentID, OpponentURL = EXCLUDED.OpponentURL,
        OpponentName = EXCLUDED.OpponentName, WinnerID = EXCLUDED.WinnerID,
        WinnerName = EXCLUDED.WinnerName, Result = EXCLUDED.Result,
        Date = EXCLUDED.Date, EventName = EXCLUDED.EventName, EventURL = EXCLUDED.EventURL,
        Method = EXCLUDED.Method, MethodDetail = EXCLUDED.MethodDetail,
        Round = EXCLUDED.Round, Time = EXCLUDED.Time, Knockdowns = EXCLUDED.Knockdowns,
        SigStrikes = EXCLUDED.SigStrikes, Takedowns = EXCLUDED.Takedowns,
        SubAttempts = EXCLUDED.SubAttempts, IsTitleFight = EXCLUDED.IsTitleFight,
        Source = EXCLUDED.Source, ScrapedAt = EXCLUDED.ScrapedAt"""

    if rows_with_url:
        execute_values(cursor,
            f"INSERT INTO Fights ({COLS}) VALUES %s ON CONFLICT (FightURL) DO UPDATE SET {UPDATE_SET}",
            rows_with_url, page_size=200)

    # Rare: fights without a FightURL — insert only if not already present
    for t in rows_without_url:
        fighter_id, _, _, opponent_id, _, _, _, _, _, fight_date = t[:10]
        cursor.execute(
            "SELECT FightID FROM Fights WHERE FighterID = %s AND OpponentID IS NOT DISTINCT FROM %s AND Date = %s",
            (fighter_id, opponent_id, fight_date))
        if not cursor.fetchone():
            cursor.execute(
                f"INSERT INTO Fights ({COLS}) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", t)

    conn.commit()
    cursor.close()

    print(f"   [OK] Fights: {len(rows_with_url) + len(rows_without_url)} upserted, {skipped} skipped")
    print(f"   [RESOLVE] Name fallback -> fighter: {resolved_name_fighter}, opponent: {resolved_name_opponent}")
    if unresolved_examples:
        print("   [WARN] Unresolved fights (up to 10):")
        for ex in unresolved_examples:
            print(f"      {ex}")


def load_elo_history(conn, fighter_url_to_id, dry_run=False):
    """Load elo_history.csv into EloHistory table (incremental: only new dates)."""
    csv_path = CSV_FILES['elo_history']

    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping ELO history")
        return

    df = pd.read_csv(csv_path)

    # Incremental load: skip dates already in the table
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(fightdate) FROM elohistory")
    max_date = cursor.fetchone()[0]
    cursor.close()
    if max_date is not None:
        df['fight_date'] = pd.to_datetime(df['fight_date'], errors='coerce')
        before = len(df)
        df = df[df['fight_date'] > pd.Timestamp(max_date)]
        print(f"[FILE] ELO history: {before} total rows, loading {len(df)} new rows after {max_date}")
    else:
        print(f"[FILE] Loading {len(df)} ELO history records from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    now = datetime.now()
    rows = []
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
        opponent_url = row.get('opponent_url')
        opponent_id = fighter_url_to_id.get(opponent_url) if pd.notna(opponent_url) else None
        rows.append((
            fighter_id, fighter_url, fight_date, opponent_id,
            opponent_url if pd.notna(opponent_url) else None,
            _sf(row.get('elo_before_fight')), _sf(row.get('opponent_elo_before_fight')),
            _sf(row.get('elo_after_fight')), _sf(row.get('elo_change')),
            row.get('result') if pd.notna(row.get('result')) else None,
            row.get('method') if pd.notna(row.get('method')) else None,
            _sf(row.get('expected_win_prob')),
            row.get('elo_source', 'ufc_fights'), now,
        ))

    cursor = conn.cursor()
    execute_values(cursor, """
        INSERT INTO EloHistory (
            FighterID, FighterURL, FightDate, OpponentID, OpponentURL,
            EloBeforeFight, OpponentEloBeforeFight, EloAfterFight, EloChange,
            Result, Method, ExpectedWinProb, EloSource, CalculatedAt
        ) VALUES %s
    """, rows, page_size=500)
    conn.commit()
    cursor.close()
    print(f"   [OK] ELO History: {len(rows)} inserted, {skipped} skipped")


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
    
    import json as _json
    now = datetime.now()
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        fighter_id = fighter_url_to_id.get(fighter_url)
        if not fighter_id:
            skipped += 1
            continue
        breakdown = _json.dumps({
            'record_adjustment': _sf(row.get('record_adjustment')),
            'win_rate_bonus': _sf(row.get('win_rate_bonus')),
            'career_efficiency_adj': _sf(row.get('career_efficiency_adj')),
            'age_factor_adj': _sf(row.get('age_factor_adj')),
            'recency_adj': _sf(row.get('recency_adj')),
            'total_adjustment': _sf(row.get('total_adjustment')),
        })
        rows.append((
            fighter_id, fighter_url,
            _si(row.get('pre_ufc_wins')), _si(row.get('pre_ufc_losses')),
            _si(row.get('pre_ufc_draws')), _si(row.get('pre_ufc_total_fights')),
            _sf(row.get('estimated_initial_elo')),
            row.get('elo_estimation_method', 'enhanced'),
            breakdown,
            _si(row.get('org_quality_tier')),
            row.get('primary_org') if pd.notna(row.get('primary_org')) else None,
            row.get('data_confidence', 'medium'),
            now, now,
        ))

    cursor = conn.cursor()
    execute_values(cursor, """
        INSERT INTO PreUfcCareer (
            FighterID, FighterURL, PreUfcWins, PreUfcLosses, PreUfcDraws,
            PreUfcTotalFights, EstimatedInitialElo, EloEstimationMethod,
            EloEstimationBreakdown, OrgQualityTier, PrimaryOrg,
            DataConfidence, CreatedAt, UpdatedAt
        ) VALUES %s
        ON CONFLICT (FighterID) DO UPDATE SET
            FighterURL = EXCLUDED.FighterURL,
            PreUfcWins = EXCLUDED.PreUfcWins, PreUfcLosses = EXCLUDED.PreUfcLosses,
            PreUfcDraws = EXCLUDED.PreUfcDraws, PreUfcTotalFights = EXCLUDED.PreUfcTotalFights,
            EstimatedInitialElo = EXCLUDED.EstimatedInitialElo,
            EloEstimationMethod = EXCLUDED.EloEstimationMethod,
            EloEstimationBreakdown = EXCLUDED.EloEstimationBreakdown,
            OrgQualityTier = EXCLUDED.OrgQualityTier, PrimaryOrg = EXCLUDED.PrimaryOrg,
            DataConfidence = EXCLUDED.DataConfidence, UpdatedAt = EXCLUDED.UpdatedAt
    """, rows, page_size=500)
    conn.commit()
    cursor.close()
    print(f"   [OK] Pre-UFC Career: {len(rows)} upserted, {skipped} skipped")


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
    
    now = datetime.now()
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        fighter_url = row.get('fighter_url')
        fighter_id = fighter_url_to_id.get(fighter_url)
        if not fighter_id:
            skipped += 1
            continue
        rows.append((
            fighter_id, fighter_url,
            _sf(row.get('avg_opponent_elo')), _sf(row.get('avg_opponent_elo_at_fight_time')),
            _si(row.get('elite_wins')), _si(row.get('elite_losses')),
            _si(row.get('good_wins')), _si(row.get('good_losses')),
            _si(row.get('average_wins')), _si(row.get('average_losses')),
            _si(row.get('below_average_wins')), _si(row.get('below_average_losses')),
            _sf(row.get('elite_win_rate')), _sf(row.get('quality_win_index')),
            _sf(row.get('recent_avg_opponent_elo')), _si(row.get('recent_elite_wins')),
            _si(row.get('schedule_strength_rank')), _sf(row.get('schedule_strength_percentile')),
            now, _si(row.get('fights_analyzed')),
        ))

    cursor = conn.cursor()
    execute_values(cursor, """
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
        ) VALUES %s
        ON CONFLICT (FighterID) DO UPDATE SET
            FighterURL = EXCLUDED.FighterURL,
            AvgOpponentElo = EXCLUDED.AvgOpponentElo,
            AvgOpponentEloAtFightTime = EXCLUDED.AvgOpponentEloAtFightTime,
            EliteOpponentWins = EXCLUDED.EliteOpponentWins,
            EliteOpponentLosses = EXCLUDED.EliteOpponentLosses,
            GoodOpponentWins = EXCLUDED.GoodOpponentWins,
            GoodOpponentLosses = EXCLUDED.GoodOpponentLosses,
            AverageOpponentWins = EXCLUDED.AverageOpponentWins,
            AverageOpponentLosses = EXCLUDED.AverageOpponentLosses,
            BelowAverageWins = EXCLUDED.BelowAverageWins,
            BelowAverageLosses = EXCLUDED.BelowAverageLosses,
            EliteWinRate = EXCLUDED.EliteWinRate,
            QualityWinIndex = EXCLUDED.QualityWinIndex,
            RecentAvgOpponentElo = EXCLUDED.RecentAvgOpponentElo,
            RecentEliteWins = EXCLUDED.RecentEliteWins,
            ScheduleStrengthRank = EXCLUDED.ScheduleStrengthRank,
            ScheduleStrengthPercentile = EXCLUDED.ScheduleStrengthPercentile,
            LastCalculated = EXCLUDED.LastCalculated,
            FightsAnalyzed = EXCLUDED.FightsAnalyzed
    """, rows, page_size=500)
    conn.commit()
    cursor.close()
    print(f"   [OK] Opponent Quality: {len(rows)} upserted, {skipped} skipped")


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
    
    now = datetime.now()
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        fighter1_url = row.get('fighter1_url')
        fighter2_url = row.get('fighter2_url')
        fighter1_id = fighter_url_to_id.get(fighter1_url)
        fighter2_id = fighter_url_to_id.get(fighter2_url)
        if not fighter1_id or not fighter2_id:
            skipped += 1
            continue
        rows.append((
            fighter1_id, fighter2_id,
            _sf(row.get('height_diff_cm')), _sf(row.get('reach_diff_cm')),
            _sf(row.get('leg_reach_diff_cm')), _sf(row.get('age_diff')),
            _sf(row.get('elo_diff')), _sf(row.get('peak_elo_diff')),
            _sf(row.get('slpm_diff')), _sf(row.get('sapm_diff')),
            _sf(row.get('str_acc_diff')), _sf(row.get('str_def_diff')),
            _sf(row.get('td_avg_diff')), _sf(row.get('td_acc_diff')),
            _sf(row.get('td_def_diff')), _sf(row.get('sub_avg_diff')),
            _sf(row.get('opponent_quality_diff')), _si(row.get('win_streak_diff')),
            _si(row.get('days_since_fight_diff')), _si(row.get('total_fights_diff')),
            row.get('fighter1_style') if pd.notna(row.get('fighter1_style')) else None,
            row.get('fighter2_style') if pd.notna(row.get('fighter2_style')) else None,
            _si(row.get('style_matchup_advantage')),
            now, False,
        ))

    cursor = conn.cursor()
    execute_values(cursor, """
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
        ) VALUES %s
        ON CONFLICT (Fighter1ID, Fighter2ID) DO UPDATE SET
            HeightDiff_cm = EXCLUDED.HeightDiff_cm,
            ReachDiff_cm = EXCLUDED.ReachDiff_cm,
            LegReachDiff_cm = EXCLUDED.LegReachDiff_cm,
            AgeDiff = EXCLUDED.AgeDiff,
            EloDiff = EXCLUDED.EloDiff, PeakEloDiff = EXCLUDED.PeakEloDiff,
            SLpMDiff = EXCLUDED.SLpMDiff, SApMDiff = EXCLUDED.SApMDiff,
            StrAccDiff = EXCLUDED.StrAccDiff, StrDefDiff = EXCLUDED.StrDefDiff,
            TDAvgDiff = EXCLUDED.TDAvgDiff, TDAccDiff = EXCLUDED.TDAccDiff,
            TDDefDiff = EXCLUDED.TDDefDiff, SubAvgDiff = EXCLUDED.SubAvgDiff,
            OpponentQualityDiff = EXCLUDED.OpponentQualityDiff,
            WinStreakDiff = EXCLUDED.WinStreakDiff,
            DaysSinceLastFightDiff = EXCLUDED.DaysSinceLastFightDiff,
            TotalFightsDiff = EXCLUDED.TotalFightsDiff,
            Fighter1Style = EXCLUDED.Fighter1Style,
            Fighter2Style = EXCLUDED.Fighter2Style,
            StyleMatchupAdvantage = EXCLUDED.StyleMatchupAdvantage,
            CalculatedAt = EXCLUDED.CalculatedAt, IsStale = FALSE
    """, rows, page_size=500)
    conn.commit()
    cursor.close()
    print(f"   [OK] Matchup Features: {len(rows)} upserted, {skipped} skipped")


def load_point_in_time_stats(conn, fighter_url_to_id, dry_run=False):
    """Load point_in_time_stats.csv into PointInTimeStats table (incremental: only new dates)."""
    csv_path = CSV_FILES['point_in_time_stats']

    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skipping point-in-time stats")
        return

    df = pd.read_csv(csv_path)

    # Incremental load: skip dates already in the table
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(fightdate) FROM pointintimestats")
    max_date = cursor.fetchone()[0]
    cursor.close()
    if max_date is not None:
        df['fight_date'] = pd.to_datetime(df['fight_date'], errors='coerce')
        before = len(df)
        df = df[df['fight_date'] > pd.Timestamp(max_date)]
        print(f"[FILE] PIT stats: {before} total rows, loading {len(df)} new rows after {max_date}")
    else:
        print(f"[FILE] Loading {len(df)} point-in-time stat records from {csv_path.name}")
    
    if dry_run:
        print(df.head())
        return
    
    now = datetime.now()
    rows = []
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
        rows.append((
            fighter_id, fighter_url, fight_date,
            _si(row.get('fights_before')), _si(row.get('wins_before')),
            _si(row.get('losses_before')), _sf(row.get('win_rate_before')),
            _sf(row.get('pit_slpm')), _sf(row.get('pit_str_acc')),
            _sf(row.get('pit_td_avg')), _sf(row.get('pit_sub_avg')),
            _sf(row.get('pit_kd_rate')), _sf(row.get('recent_win_rate')),
            _sf(row.get('avg_fight_time')), _sf(row.get('finish_rate')),
            row.get('has_prior_data', False), now,
        ))

    cursor = conn.cursor()
    execute_values(cursor, """
        INSERT INTO PointInTimeStats (
            FighterID, FighterURL, FightDate,
            FightsBefore, WinsBefore, LossesBefore, WinRateBefore,
            PIT_SLpM, PIT_StrAcc, PIT_TDAvg, PIT_SubAvg, PIT_KDRate,
            RecentWinRate, AvgFightTime, FinishRate, HasPriorData, CalculatedAt
        ) VALUES %s
    """, rows, page_size=500)
    conn.commit()
    cursor.close()
    print(f"   [OK] Point-in-Time Stats: {len(rows)} inserted, {skipped} skipped")


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
            cursor.execute(f"SELECT COUNT(*) FROM {safe_identifier(table_name, ALLOWED_TABLES)}")
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

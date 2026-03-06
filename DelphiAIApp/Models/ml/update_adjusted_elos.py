"""
Batch Update Adjusted ELOs

This script updates the database with adjusted ELO ratings that account for:
1. Inactivity decay (ring rust) - calculated for ALL fighters
2. Injury penalties - optionally checked for TOP fighters only (due to scraping time)

Usage:
    # Update inactivity decay only (fast, no scraping)
    python -m ml.update_adjusted_elos
    
    # Also check injuries for top 50 ranked fighters
    python -m ml.update_adjusted_elos --check-injuries --top-n 50
    
    # Check injuries for specific weight class
    python -m ml.update_adjusted_elos --check-injuries --weight-class Lightweight

Author: DelphiAI
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'dbname': os.getenv('DB_NAME', 'delphi_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}


def calculate_inactivity_penalty(days_inactive: int, current_elo: float) -> Tuple[int, Dict]:
    """
    Calculate ELO penalty for inactivity (ring rust).
    
    MMA-specific decay rates - fighters lose sharpness quickly.
    Decay is applied to ELO above/below 1500 baseline.
    
    Time brackets:
    - 6-12 months:  8% decay per year + 10 flat
    - 12-18 months: 12% decay per year + 20 flat
    - 18-24 months: 18% decay per year + 30 flat
    - 24+ months:   25% decay per year + 40 flat
    
    Args:
        days_inactive: Days since last fight
        current_elo: Current raw ELO rating
        
    Returns:
        Tuple of (penalty, details_dict)
    """
    if days_inactive <= 180:  # Less than 6 months - no penalty
        return 0, {'reason': 'Active fighter (< 6 months)'}
    
    years_inactive = days_inactive / 365
    
    # Progressive decay rate based on time off
    if days_inactive <= 365:  # 6-12 months
        decay_rate = 0.08
        flat_penalty = 10
    elif days_inactive <= 540:  # 12-18 months
        decay_rate = 0.12
        flat_penalty = 20
    elif days_inactive <= 730:  # 18-24 months
        decay_rate = 0.18
        flat_penalty = 30
    else:  # 24+ months
        decay_rate = 0.25
        flat_penalty = 40
    
    # Multiply by years for compounding effect
    effective_decay = decay_rate * years_inactive
    effective_decay = min(effective_decay, 0.35)  # Cap at 35% max
    
    elo_diff_from_baseline = current_elo - 1500
    proportional_decay = int(elo_diff_from_baseline * effective_decay)
    
    # Total penalty should never be negative (minimum is flat penalty)
    total_penalty = max(proportional_decay + flat_penalty, flat_penalty)
    
    return total_penalty, {
        'days_inactive': days_inactive,
        'years_inactive': round(years_inactive, 2),
        'decay_rate': round(effective_decay * 100, 1),
        'flat_penalty': flat_penalty,
        'proportional_decay': proportional_decay,
        'total_penalty': total_penalty,
    }


def update_all_inactivity_penalties(conn) -> int:
    """
    Update inactivity penalties for all fighters in the database.
    
    Args:
        conn: PostgreSQL connection
        
    Returns:
        Number of fighters updated
    """
    logger.info("Calculating inactivity penalties for all fighters...")
    
    cursor = conn.cursor()
    
    # Get all fighters with ELO ratings
    cursor.execute("""
        SELECT 
            cs.FighterID,
            fs.Name,
            cs.EloRating,
            COALESCE(fs.DaysSinceLastFight, 0) as DaysSinceLastFight
        FROM CareerStats cs
        JOIN FighterStats fs ON cs.FighterID = fs.FighterID
        WHERE cs.EloRating IS NOT NULL
    """)
    
    fighters = cursor.fetchall()
    logger.info(f"Found {len(fighters)} fighters with ELO ratings")
    
    # Calculate penalties
    updates = []
    penalties_applied = 0
    
    for fighter_id, name, elo, days_inactive in fighters:
        penalty, details = calculate_inactivity_penalty(days_inactive, float(elo))
        adjusted_elo = float(elo) - penalty
        
        updates.append((
            penalty,
            adjusted_elo,
            datetime.now(),
            fighter_id
        ))
        
        if penalty > 0:
            penalties_applied += 1
            if penalty >= 30:  # Log significant penalties
                logger.info(f"  {name}: -{penalty} ELO ({days_inactive} days inactive)")
    
    # Batch update
    cursor.executemany("""
        UPDATE CareerStats 
        SET 
            InactivityPenalty = %s,
            AdjustedEloRating = EloRating - %s + COALESCE(InactivityPenalty, 0) - COALESCE(InjuryPenalty, 0),
            AdjustmentsCalculatedAt = %s
        WHERE FighterID = %s
    """, [(p, p, t, fid) for p, _, t, fid in updates])
    
    # Actually, let's do a simpler update that recalculates adjusted ELO properly
    cursor.execute("""
        UPDATE CareerStats 
        SET AdjustedEloRating = EloRating - COALESCE(InactivityPenalty, 0) - COALESCE(InjuryPenalty, 0)
        WHERE EloRating IS NOT NULL
    """)
    
    conn.commit()
    
    logger.info(f"Updated {len(updates)} fighters, {penalties_applied} have inactivity penalties")
    
    return len(updates)


def get_top_fighters(conn, top_n: int = 50, weight_class: str = None) -> List[Dict]:
    """
    Get top fighters for injury checking.
    
    Prioritizes:
    1. Ranked fighters
    2. Recent activity
    3. Higher ELO
    
    Args:
        conn: PostgreSQL connection
        top_n: Number of fighters to return
        weight_class: Optional weight class filter
        
    Returns:
        List of fighter dicts
    """
    cursor = conn.cursor()
    
    where_clause = "WHERE cs.EloRating IS NOT NULL"
    params = []
    
    if weight_class:
        where_clause += " AND fs.WeightClass = %s"
        params.append(weight_class)
    
    cursor.execute(f"""
        SELECT 
            fs.FighterID,
            fs.Name,
            fs.WeightClass,
            cs.EloRating,
            fs.DaysSinceLastFight,
            fs.LastInjuryCheckDate,
            fs.UFCRanking
        FROM FighterStats fs
        JOIN CareerStats cs ON fs.FighterID = cs.FighterID
        {where_clause}
        ORDER BY 
            fs.UFCRanking NULLS LAST,
            cs.EloRating DESC,
            fs.DaysSinceLastFight ASC NULLS LAST
        LIMIT %s
    """, params + [top_n])
    
    columns = ['fighter_id', 'name', 'weight_class', 'elo', 'days_inactive', 
               'last_injury_check', 'ranking']
    
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def check_fighter_injuries_and_update(conn, fighter: Dict) -> Dict:
    """
    Check a fighter for injuries and update the database.
    
    Args:
        conn: PostgreSQL connection
        fighter: Fighter dict with id, name, etc.
        
    Returns:
        Result dict
    """
    # Import here to avoid circular imports
    try:
        from ml.injury_scraper import InjuryScraper
    except ModuleNotFoundError:
        from injury_scraper import InjuryScraper
    
    scraper = InjuryScraper()
    result = scraper.check_fighter_injuries(fighter['name'], check_news=True)
    
    cursor = conn.cursor()
    
    # Log the check
    cursor.execute("""
        INSERT INTO InjuryCheckLog 
        (FighterID, FighterName, InjuryFound, InjuryKeyword, InjurySeverity,
         EstimatedInjuryDate, DaysSinceInjury, ElopenaltyApplied, 
         NewsArticlesChecked, SourceURL, RawDetails, Error)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        fighter['fighter_id'],
        fighter['name'],
        result['injury_found'],
        result['details'].get('keyword') if result['injury_found'] else None,
        result['details'].get('severity') if result['injury_found'] else None,
        result['details'].get('estimated_date') if result['injury_found'] else None,
        result['details'].get('days_since') if result['injury_found'] else None,
        result['elo_penalty'] if result['injury_found'] else None,
        result.get('news_articles_checked', 0),
        result['url'],
        Json(result['details']) if result['details'] else None,
        result.get('error'),
    ))
    
    # Update fighter stats
    cursor.execute("""
        UPDATE FighterStats 
        SET 
            LastInjuryCheckDate = %s,
            InjuryDetails = %s
        WHERE FighterID = %s
    """, (
        datetime.now(),
        Json(result['details']) if result['injury_found'] else None,
        fighter['fighter_id'],
    ))
    
    # Update career stats with injury penalty
    if result['injury_found']:
        cursor.execute("""
            UPDATE CareerStats 
            SET 
                InjuryPenalty = %s,
                AdjustedEloRating = EloRating - COALESCE(InactivityPenalty, 0) - %s,
                AdjustmentsCalculatedAt = %s
            WHERE FighterID = %s
        """, (
            result['elo_penalty'],
            result['elo_penalty'],
            datetime.now(),
            fighter['fighter_id'],
        ))
    else:
        # Clear any previous injury penalty
        cursor.execute("""
            UPDATE CareerStats 
            SET 
                InjuryPenalty = 0,
                AdjustedEloRating = EloRating - COALESCE(InactivityPenalty, 0),
                AdjustmentsCalculatedAt = %s
            WHERE FighterID = %s
        """, (
            datetime.now(),
            fighter['fighter_id'],
        ))
    
    conn.commit()
    
    return result


def check_injuries_batch(conn, fighters: List[Dict], progress_callback=None) -> Dict:
    """
    Check injuries for a batch of fighters.
    
    Args:
        conn: PostgreSQL connection
        fighters: List of fighter dicts
        progress_callback: Optional callback(current, total, fighter_name)
        
    Returns:
        Summary dict
    """
    results = {
        'checked': 0,
        'injuries_found': 0,
        'errors': 0,
        'details': [],
    }
    
    for i, fighter in enumerate(fighters):
        if progress_callback:
            progress_callback(i + 1, len(fighters), fighter['name'])
        
        try:
            result = check_fighter_injuries_and_update(conn, fighter)
            results['checked'] += 1
            
            if result['injury_found']:
                results['injuries_found'] += 1
                results['details'].append({
                    'name': fighter['name'],
                    'penalty': result['elo_penalty'],
                    'keyword': result['details'].get('keyword'),
                })
                logger.info(f"  {fighter['name']}: INJURY FOUND (-{result['elo_penalty']} ELO)")
            
            if result.get('error'):
                results['errors'] += 1
                
        except Exception as e:
            logger.error(f"  {fighter['name']}: Error - {e}")
            results['errors'] += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Update adjusted ELO ratings in database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--check-injuries', '-i', action='store_true',
                        help='Also check UFC.com for injuries (slower)')
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top fighters to check for injuries (default: 50)')
    parser.add_argument('--weight-class', type=str,
                        help='Only check fighters in this weight class')
    parser.add_argument('--run-migration', action='store_true',
                        help='Run database migration first')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DELPHI AI - ADJUSTED ELO UPDATE")
    print("=" * 70)
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        sys.exit(1)
    
    try:
        # Run migration if requested
        if args.run_migration:
            logger.info("Running database migration...")
            migration_path = Path(__file__).parent.parent / 'db' / 'migrations' / '001_add_adjusted_elo_columns.sql'
            if migration_path.exists():
                cursor = conn.cursor()
                cursor.execute(migration_path.read_text())
                conn.commit()
                logger.info("Migration completed")
            else:
                logger.warning(f"Migration file not found: {migration_path}")
        
        # Step 1: Update inactivity penalties for ALL fighters
        print("\n" + "-" * 70)
        print("STEP 1: Calculating Inactivity Penalties")
        print("-" * 70)
        
        if args.dry_run:
            logger.info("DRY RUN - would update inactivity penalties")
        else:
            updated = update_all_inactivity_penalties(conn)
            print(f"\nUpdated {updated} fighters with inactivity penalties")
        
        # Step 2: Check injuries for top fighters (optional)
        if args.check_injuries:
            print("\n" + "-" * 70)
            print(f"STEP 2: Checking Injuries for Top {args.top_n} Fighters")
            print("-" * 70)
            
            fighters = get_top_fighters(conn, args.top_n, args.weight_class)
            logger.info(f"Found {len(fighters)} fighters to check")
            
            if args.dry_run:
                logger.info("DRY RUN - would check these fighters:")
                for f in fighters[:10]:
                    print(f"  - {f['name']} (ELO: {f['elo']}, Inactive: {f['days_inactive']} days)")
                if len(fighters) > 10:
                    print(f"  ... and {len(fighters) - 10} more")
            else:
                def progress(current, total, name):
                    print(f"  [{current}/{total}] Checking {name}...", end='\r')
                
                results = check_injuries_batch(conn, fighters, progress)
                
                print(f"\n\nInjury Check Results:")
                print(f"  Checked: {results['checked']}")
                print(f"  Injuries Found: {results['injuries_found']}")
                print(f"  Errors: {results['errors']}")
                
                if results['details']:
                    print("\n  Injuries Detected:")
                    for d in results['details']:
                        print(f"    - {d['name']}: -{d['penalty']} ELO ({d['keyword']})")
        
        # Summary
        print("\n" + "=" * 70)
        print("UPDATE COMPLETE")
        print("=" * 70)
        
        # Show sample of adjusted ELOs
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                fs.Name,
                cs.EloRating as RawElo,
                cs.InactivityPenalty,
                cs.InjuryPenalty,
                cs.AdjustedEloRating,
                fs.DaysSinceLastFight
            FROM CareerStats cs
            JOIN FighterStats fs ON cs.FighterID = fs.FighterID
            WHERE cs.AdjustedEloRating IS NOT NULL
            AND (cs.InactivityPenalty > 0 OR cs.InjuryPenalty > 0)
            ORDER BY cs.EloRating DESC
            LIMIT 15
        """)
        
        results = cursor.fetchall()
        if results:
            print("\nSample of Adjusted ELOs (top fighters with penalties):")
            print("-" * 70)
            print(f"{'Fighter':<25} {'Raw':>8} {'Inact':>6} {'Injury':>6} {'Adj':>8}")
            print("-" * 70)
            for name, raw, inact, injury, adj, days in results:
                inact = inact or 0
                injury = injury or 0
                print(f"{name[:24]:<25} {raw:>8.0f} {-inact:>6.0f} {-injury:>6.0f} {adj:>8.0f}")
        
    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == '__main__':
    main()

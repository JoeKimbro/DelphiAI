"""
Populate Fighter Styles & Style Matchup History

1. Classifies every fighter's style (striker/wrestler/grappler/balanced)
2. Builds style-vs-style record for each fighter from fight history

Usage:
    python -m ml.populate_styles                  # Populate all
    python -m ml.populate_styles --dry-run        # Preview without writing
    python -m ml.populate_styles --run-migration  # Run migration first
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

# Import canonical style classifier
try:
    from ml.style_classifier import classify_style, VALID_STYLES
except ModuleNotFoundError:
    from style_classifier import classify_style, VALID_STYLES

# Load environment
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'dbname': os.getenv('DB_NAME', 'delphi_db'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
}


def get_all_fighters_with_stats(conn):
    """Get all fighters with their career stats for style classification."""
    cur = conn.cursor()
    cur.execute('''
        SELECT 
            fs.fighterid, fs.name, fs.fighterurl,
            cs.slpm, cs.tdavg, cs.subavg, cs.strdef, cs.tddef
        FROM fighterstats fs
        LEFT JOIN careerstats cs ON fs.fighterid = cs.fighterid
        WHERE cs.slpm IS NOT NULL
    ''')
    
    fighters = []
    for row in cur.fetchall():
        fighters.append({
            'id': row[0],
            'name': row[1],
            'url': row[2],
            'slpm': float(row[3]) if row[3] else 0,
            'td_avg': float(row[4]) if row[4] else 0,
            'sub_avg': float(row[5]) if row[5] else 0,
            'str_def': float(row[6]) if row[6] else 0,
            'td_def': float(row[7]) if row[7] else 0,
        })
    
    cur.close()
    return fighters


def classify_all_fighters(fighters):
    """Classify style for every fighter."""
    results = []
    style_counts = defaultdict(int)
    
    for f in fighters:
        style = classify_style(
            slpm=f['slpm'],
            td_avg=f['td_avg'],
            sub_avg=f['sub_avg'],
            str_def=f['str_def'],
            td_def=f['td_def'],
        )
        f['style'] = style
        results.append(f)
        style_counts[style] += 1
    
    return results, style_counts


def update_fighter_styles(conn, fighters, dry_run=False):
    """Write FightingStyle to FighterStats for all fighters."""
    if dry_run:
        return
    
    cur = conn.cursor()
    for f in fighters:
        cur.execute(
            'UPDATE fighterstats SET fightingstyle = %s WHERE fighterid = %s',
            (f['style'], f['id'])
        )
    conn.commit()
    cur.close()


def build_style_lookup(fighters):
    """Build URL -> style lookup dict."""
    return {f['url']: f['style'] for f in fighters if f['url']}


def get_all_fights(conn):
    """Get all fights with method details."""
    cur = conn.cursor()
    cur.execute('''
        SELECT 
            f.fightername, f.fighterurl,
            f.opponentname, f.opponenturl,
            f.winnername, f.result,
            f.method, f.date,
            f.round, f.time
        FROM fights f
        WHERE f.result IN ('win', 'loss', 'draw')
          AND f.fighterurl IS NOT NULL
          AND f.opponenturl IS NOT NULL
        ORDER BY f.date
    ''')
    
    fights = []
    for row in cur.fetchall():
        # Parse method
        method = (row[6] or '').upper()
        if 'KO' in method or 'TKO' in method:
            method_type = 'ko'
        elif 'SUB' in method:
            method_type = 'sub'
        elif 'DEC' in method:
            method_type = 'dec'
        else:
            method_type = 'other'
        
        # Parse fight duration (round * 5 min - time remaining approximation)
        try:
            rnd = int(row[8]) if row[8] else 3
            time_parts = str(row[9] or '5:00').split(':')
            minutes = int(time_parts[0]) if time_parts else 5
            duration = (rnd - 1) * 5 + minutes
        except (ValueError, IndexError):
            duration = 15
        
        fights.append({
            'fighter_name': row[0],
            'fighter_url': row[1],
            'opponent_name': row[2],
            'opponent_url': row[3],
            'winner_name': row[4],
            'result': row[5],
            'method_type': method_type,
            'date': row[7],
            'duration': duration,
        })
    
    cur.close()
    return fights


def calculate_style_matchup_records(fighters, fights, style_lookup):
    """
    For each fighter, calculate their record vs each opponent style.
    
    Returns: dict of fighter_id -> {style -> record_dict}
    """
    # Build fighter URL -> ID lookup
    url_to_id = {f['url']: f['id'] for f in fighters if f['url']}
    
    # Initialize records: fighter_id -> opponent_style -> stats
    records = defaultdict(lambda: defaultdict(lambda: {
        'wins': 0, 'losses': 0, 'draws': 0,
        'ko_wins': 0, 'sub_wins': 0, 'dec_wins': 0,
        'ko_losses': 0, 'sub_losses': 0, 'dec_losses': 0,
        'total_slpm': 0, 'total_td': 0, 'total_sub_att': 0,
        'total_duration': 0, 'fight_count': 0,
    }))
    
    for fight in fights:
        fighter_url = fight['fighter_url']
        opponent_url = fight['opponent_url']
        
        # Skip if we can't classify opponent
        if opponent_url not in style_lookup:
            continue
        
        fighter_id = url_to_id.get(fighter_url)
        if not fighter_id:
            continue
        
        opp_style = style_lookup[opponent_url]
        rec = records[fighter_id][opp_style]
        rec['fight_count'] += 1
        rec['total_duration'] += fight['duration']
        
        result = fight['result']
        method = fight['method_type']
        
        if result == 'win':
            rec['wins'] += 1
            if method == 'ko':
                rec['ko_wins'] += 1
            elif method == 'sub':
                rec['sub_wins'] += 1
            elif method == 'dec':
                rec['dec_wins'] += 1
        elif result == 'loss':
            rec['losses'] += 1
            if method == 'ko':
                rec['ko_losses'] += 1
            elif method == 'sub':
                rec['sub_losses'] += 1
            elif method == 'dec':
                rec['dec_losses'] += 1
        elif result == 'draw':
            rec['draws'] += 1
    
    return records


def write_style_matchup_records(conn, records, fighters, dry_run=False):
    """Write style matchup records to database."""
    if dry_run:
        return 0
    
    cur = conn.cursor()
    url_lookup = {f['id']: f['url'] for f in fighters}
    
    rows_written = 0
    
    for fighter_id, style_records in records.items():
        fighter_url = url_lookup.get(fighter_id)
        
        for opp_style, rec in style_records.items():
            if rec['fight_count'] == 0:
                continue
            
            total = rec['wins'] + rec['losses'] + rec['draws']
            win_rate = (rec['wins'] / total * 100) if total > 0 else 0
            avg_duration = rec['total_duration'] / rec['fight_count'] if rec['fight_count'] > 0 else 0
            
            cur.execute('''
                INSERT INTO stylematchuprecord (
                    fighterid, fighterurl, opponentstyle,
                    wins, losses, draws, totalfights, winrate,
                    kowins, subwins, decwins,
                    kolosses, sublosses, declosses,
                    avgfightduration, calculatedat
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fighterid, opponentstyle) DO UPDATE SET
                    wins = EXCLUDED.wins,
                    losses = EXCLUDED.losses,
                    draws = EXCLUDED.draws,
                    totalfights = EXCLUDED.totalfights,
                    winrate = EXCLUDED.winrate,
                    kowins = EXCLUDED.kowins,
                    subwins = EXCLUDED.subwins,
                    decwins = EXCLUDED.decwins,
                    kolosses = EXCLUDED.kolosses,
                    sublosses = EXCLUDED.sublosses,
                    declosses = EXCLUDED.declosses,
                    avgfightduration = EXCLUDED.avgfightduration,
                    calculatedat = EXCLUDED.calculatedat
            ''', (
                fighter_id, fighter_url, opp_style,
                rec['wins'], rec['losses'], rec['draws'],
                total, round(win_rate, 2),
                rec['ko_wins'], rec['sub_wins'], rec['dec_wins'],
                rec['ko_losses'], rec['sub_losses'], rec['dec_losses'],
                round(avg_duration, 2), datetime.now()
            ))
            rows_written += 1
    
    conn.commit()
    cur.close()
    return rows_written


def run_migration(conn):
    """Run the style migration SQL."""
    migration_path = Path(__file__).parent.parent / 'db' / 'migrations' / '002_add_fighting_style.sql'
    if not migration_path.exists():
        print(f"[ERROR] Migration file not found: {migration_path}")
        return False
    
    print(f"  Running migration: {migration_path.name}")
    with open(migration_path, 'r') as f:
        sql = f.read()
    
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    print("  Migration complete.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Populate fighter styles and style matchup history',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m ml.populate_styles                  # Populate all
  python -m ml.populate_styles --dry-run        # Preview only
  python -m ml.populate_styles --run-migration  # Run migration first
        '''
    )
    parser.add_argument('--dry-run', action='store_true', help='Preview without writing to database')
    parser.add_argument('--run-migration', action='store_true', help='Run database migration first')
    
    args = parser.parse_args()
    
    # Connect
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        sys.exit(1)
    
    try:
        # Run migration if requested
        if args.run_migration:
            print("\n[1/5] Running migration...")
            if not run_migration(conn):
                sys.exit(1)
        else:
            print("\n[1/5] Skipping migration (use --run-migration to run)")
        
        # Step 1: Get all fighters
        print("[2/5] Loading fighters...")
        fighters = get_all_fighters_with_stats(conn)
        print(f"  Found {len(fighters)} fighters with stats")
        
        # Step 2: Classify all fighters
        print("[3/5] Classifying fighter styles...")
        fighters, style_counts = classify_all_fighters(fighters)
        
        print(f"\n  Style Distribution:")
        for style in VALID_STYLES:
            count = style_counts[style]
            pct = count / len(fighters) * 100 if fighters else 0
            bar = "#" * int(pct / 2)
            print(f"    {style:<10} {count:>5} ({pct:>5.1f}%)  {bar}")
        
        # Write styles to FighterStats
        if not args.dry_run:
            update_fighter_styles(conn, fighters)
            print(f"\n  Updated {len(fighters)} fighter styles in database")
        else:
            print(f"\n  [DRY RUN] Would update {len(fighters)} fighter styles")
        
        # Step 3: Build style lookup
        print("\n[4/5] Building style-vs-style matchup history...")
        style_lookup = build_style_lookup(fighters)
        
        # Get all fights
        fights = get_all_fights(conn)
        print(f"  Processing {len(fights)} fight records...")
        
        # Calculate records
        records = calculate_style_matchup_records(fighters, fights, style_lookup)
        
        # Count fighters with matchup data
        fighters_with_data = sum(1 for r in records.values() if any(s['fight_count'] > 0 for s in r.values()))
        total_records = sum(
            sum(1 for s in r.values() if s['fight_count'] > 0) 
            for r in records.values()
        )
        print(f"  {fighters_with_data} fighters with matchup data")
        print(f"  {total_records} total style matchup records")
        
        # Show sample
        print(f"\n  Sample matchup records:")
        sample_count = 0
        for f in fighters[:50]:  # Check first 50 for interesting ones
            fid = f['id']
            if fid in records:
                has_variety = sum(1 for s in records[fid].values() if s['fight_count'] > 0) >= 3
                if has_variety:
                    print(f"\n    {f['name']} ({f['style']}):")
                    for style in VALID_STYLES:
                        rec = records[fid].get(style, {})
                        total = rec.get('fight_count', 0)
                        if total > 0:
                            w = rec['wins']
                            l = rec['losses']
                            wr = w / (w + l) * 100 if (w + l) > 0 else 0
                            print(f"      vs {style:<10} {w}-{l}  ({wr:.0f}% win rate)")
                    sample_count += 1
                    if sample_count >= 3:
                        break
        
        # Step 4: Write to database
        print(f"\n[5/5] Writing to database...")
        rows = write_style_matchup_records(conn, records, fighters, dry_run=args.dry_run)
        
        if args.dry_run:
            print(f"  [DRY RUN] Would write {total_records} records")
        else:
            print(f"  Wrote {rows} style matchup records")
        
        print("\nDone!")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()

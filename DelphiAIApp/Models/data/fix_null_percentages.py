"""
Fix NULL percentage values in database.

For percentage/rate columns, NULL should be 0% (never happened)
rather than unknown. This ensures ML model interprets them correctly.
"""

import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

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


def main():
    print("="*60)
    print("FIXING NULL PERCENTAGE VALUES")
    print("="*60)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Columns to fix in CareerStats
    # These are percentages where NULL means "0%" not "unknown"
    career_stats_columns = [
        'KO_Round1_Pct',
        'KO_Round2_Pct', 
        'KO_Round3_Pct',
        'Sub_Round1_Pct',
        'Sub_Round2_Pct',
        'Sub_Round3_Pct',
        'FirstRoundFinishRate',
        'DecisionRate',
        'AvgFightDuration',
    ]
    
    print("\n[CareerStats Table]")
    print("-" * 40)
    
    # First, check current null counts
    for col in career_stats_columns:
        cursor.execute(f"SELECT COUNT(*) FROM CareerStats WHERE {col} IS NULL")
        null_count = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM CareerStats")
        total = cursor.fetchone()[0]
        print(f"   {col}: {null_count}/{total} NULLs ({null_count/total*100:.1f}%)")
    
    # Update NULLs to 0
    print("\n[UPDATING] Setting NULLs to 0...")
    
    for col in career_stats_columns:
        cursor.execute(f"UPDATE CareerStats SET {col} = 0 WHERE {col} IS NULL")
        updated = cursor.rowcount
        if updated > 0:
            print(f"   {col}: {updated} rows updated")
    
    conn.commit()
    
    # Verify
    print("\n[VERIFY] Checking for remaining NULLs...")
    remaining_nulls = 0
    for col in career_stats_columns:
        cursor.execute(f"SELECT COUNT(*) FROM CareerStats WHERE {col} IS NULL")
        null_count = cursor.fetchone()[0]
        if null_count > 0:
            print(f"   [!] {col}: {null_count} NULLs remaining")
            remaining_nulls += null_count
    
    if remaining_nulls == 0:
        print("   [OK] All percentage columns now have 0 instead of NULL")
    
    # Also check FighterStats for any similar columns
    print("\n[FighterStats Table]")
    print("-" * 40)
    
    fighter_stats_columns = [
        'Wins',
        'Losses', 
        'Draws',
        'TotalFights',
    ]
    
    for col in fighter_stats_columns:
        cursor.execute(f"SELECT COUNT(*) FROM FighterStats WHERE {col} IS NULL")
        null_count = cursor.fetchone()[0]
        if null_count > 0:
            print(f"   {col}: {null_count} NULLs - updating to 0...")
            cursor.execute(f"UPDATE FighterStats SET {col} = 0 WHERE {col} IS NULL")
            print(f"      Updated {cursor.rowcount} rows")
    
    conn.commit()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    cursor.execute("SELECT COUNT(*) FROM CareerStats")
    total_career = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM CareerStats 
        WHERE KO_Round1_Pct = 0 AND KO_Round2_Pct = 0 AND KO_Round3_Pct = 0
    """)
    no_ko_fighters = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM CareerStats 
        WHERE Sub_Round1_Pct = 0 AND Sub_Round2_Pct = 0 AND Sub_Round3_Pct = 0
    """)
    no_sub_fighters = cursor.fetchone()[0]
    
    print(f"\nTotal fighters with career stats: {total_career}")
    print(f"Fighters with 0% KO finishes (all rounds): {no_ko_fighters} ({no_ko_fighters/total_career*100:.1f}%)")
    print(f"Fighters with 0% Sub finishes (all rounds): {no_sub_fighters} ({no_sub_fighters/total_career*100:.1f}%)")
    
    # Show sample of KO percentage distribution
    print("\n[KO Round 1 % Distribution]")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN KO_Round1_Pct = 0 THEN '0%'
                WHEN KO_Round1_Pct <= 25 THEN '1-25%'
                WHEN KO_Round1_Pct <= 50 THEN '26-50%'
                WHEN KO_Round1_Pct <= 75 THEN '51-75%'
                ELSE '76-100%'
            END as bucket,
            COUNT(*) as count
        FROM CareerStats
        GROUP BY bucket
        ORDER BY bucket
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} fighters")
    
    cursor.close()
    conn.close()
    
    print("\n[OK] Done!")


if __name__ == '__main__':
    main()

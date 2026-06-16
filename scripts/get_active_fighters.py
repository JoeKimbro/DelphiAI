"""
Get UFC.com profile URLs for fighters who recently fought or have upcoming bouts.

Queries the database for:
  - Fighters who fought in the last 21 days (recent event window)
  - Fighters booked on upcoming events (unresolved PredictionTracking rows)

Output is written to a file (one URL per line) for use by scrape_all.py --fighters-file.

Usage:
    python scripts/get_active_fighters.py --output /tmp/active_fighters.txt
    python scripts/get_active_fighters.py          # prints to stdout
"""

import argparse
import os
from pathlib import Path

import psycopg2

# Support both DATABASE_URL and individual DB_* env vars
_DATABASE_URL = os.getenv("DATABASE_URL", "")
if not _DATABASE_URL:
    _root = Path(__file__).resolve().parents[1]
    _dotenv = _root / ".env"
    if _dotenv.exists():
        for line in _dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                _DATABASE_URL = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not _DATABASE_URL:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    name = os.getenv("DB_NAME", "delphi_db")
    user = os.getenv("DB_USER", "")
    pwd = os.getenv("DB_PASSWORD", "")
    _DATABASE_URL = f"postgresql://{user}:{pwd}@{host}:{port}/{name}"


def get_active_fighter_urls() -> list:
    conn = psycopg2.connect(_DATABASE_URL)
    try:
        cur = conn.cursor()
        urls: set[str] = set()

        # Fighters who fought in the last 21 days — their stats need refreshing.
        cur.execute("""
            SELECT DISTINCT fs.UFCUrl
            FROM Fights f
            JOIN FighterStats fs ON f.FighterID = fs.FighterID
            WHERE f.Date >= CURRENT_DATE - INTERVAL '21 days'
              AND fs.UFCUrl IS NOT NULL
              AND fs.UFCUrl <> ''
        """)
        urls.update(row[0] for row in cur.fetchall())

        # Fighters booked on upcoming events or who fought very recently, taken
        # from PredictionTracking (which the prefetch/resolve jobs keep current).
        #
        # This is essential: the recent-fights query above derives "active" from
        # the Fights table, but Fights only advances when the incremental scrape
        # runs against active fighters — a circular dependency. Once Fights fell
        # >21 days stale, that query returned zero fighters and the scrape stalled
        # permanently. PredictionTracking does NOT depend on Fights, so it seeds
        # the right fighters (upcoming cards + recently-resolved ones) regardless
        # of how stale Fights is, breaking the cycle.
        cur.execute("""
            SELECT DISTINCT fs.UFCUrl
            FROM PredictionTracking pt
            JOIN FighterStats fs
              ON fs.FighterID IN (pt.fighter1_id, pt.fighter2_id)
            WHERE (pt.was_correct IS NULL
                   OR pt.predicted_at >= CURRENT_DATE - INTERVAL '45 days')
              AND fs.UFCUrl IS NOT NULL
              AND fs.UFCUrl <> ''
        """)
        urls.update(row[0] for row in cur.fetchall())

        cur.close()
        return sorted(urls)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Output UFC.com URLs for fighters needing incremental scrape"
    )
    parser.add_argument(
        "--output", default="-",
        help="Output file path (default: stdout)"
    )
    args = parser.parse_args()

    urls = get_active_fighter_urls()

    if not urls:
        print("[get_active_fighters] No active fighters found — nothing to scrape.")
        return

    if args.output == "-":
        for url in urls:
            print(url)
    else:
        Path(args.output).write_text("\n".join(urls) + "\n", encoding="utf-8")
        print(f"[get_active_fighters] {len(urls)} fighter URLs → {args.output}")


if __name__ == "__main__":
    main()

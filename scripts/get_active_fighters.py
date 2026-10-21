"""
Get UFC.com profile URLs for fighters who recently fought or have upcoming bouts.

Queries the database for:
  - Fighters (and their opponents) who fought in the last N days (default 21)
  - Fighters on cards predicted or resolved in the last N days (PredictionTracking)

Fighters with no stored UFC.com URL get one built from their name.

Output is written to a file (one URL per line) for use by scrape_all.py --fighters-file.

Usage:
    python scripts/get_active_fighters.py --output /tmp/active_fighters.txt
    python scripts/get_active_fighters.py --days 130   # backfill a longer gap
    python scripts/get_active_fighters.py          # prints to stdout
"""

import argparse
import os
import re
import unicodedata
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


def _ufc_com_url(name: str) -> str:
    """Build a UFC.com athlete URL from a name (same rules as scrape_fighter_on_demand)."""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9-]", "", "-".join(n.lower().split()))
    return f"https://www.ufc.com/athlete/{slug}"


def get_active_fighter_urls(days: int = 21) -> list:
    conn = psycopg2.connect(_DATABASE_URL)
    try:
        cur = conn.cursor()

        # Fighters who fought in the window, plus their opponents — their stats need
        # refreshing. Opponents matter when Fights only partially covers a card.
        # Fighters on cards predicted or resolved in the window. Fights only gains
        # rows via this scrape, so relying on it alone stalls permanently once it
        # falls behind the window; PredictionTracking is filled independently.
        # predicted_at/resolved_at are used because event_date is often blank or
        # display-formatted. GREATEST ignores NULLs, so unresolved rows still count.
        cur.execute("""
            WITH ids AS (
                SELECT FighterID AS id FROM Fights
                WHERE Date >= CURRENT_DATE - make_interval(days => %(days)s)
                UNION
                SELECT OpponentID FROM Fights
                WHERE Date >= CURRENT_DATE - make_interval(days => %(days)s)
                UNION
                SELECT unnest(ARRAY[pt.fighter1_id, pt.fighter2_id])
                FROM PredictionTracking pt
                WHERE GREATEST(pt.predicted_at, pt.resolved_at)
                      >= CURRENT_DATE - make_interval(days => %(days)s)
            )
            SELECT DISTINCT fs.UFCUrl, fs.Name
            FROM ids
            JOIN FighterStats fs ON fs.FighterID = ids.id
        """, {"days": days})
        rows = cur.fetchall()

        cur.close()
        urls = set()
        for url, name in rows:
            if url:
                urls.add(url)
            elif name:
                urls.add(_ufc_com_url(name))
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
    parser.add_argument(
        "--days", type=int, default=21,
        help="Lookback window in days (default: 21). Raise it to backfill a longer gap."
    )
    args = parser.parse_args()

    urls = get_active_fighter_urls(args.days)

    if not urls:
        print("[get_active_fighters] No active fighters found — nothing to scrape.")
        return

    if args.output == "-":
        for url in urls:
            print(url)
    else:
        Path(args.output).write_text("\n".join(urls) + "\n", encoding="utf-8")
        print(f"[get_active_fighters] {len(urls)} fighter URLs (last {args.days} days) -> {args.output}")


if __name__ == "__main__":
    main()

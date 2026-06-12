"""
Auto-resolve predictions for events whose date has passed.
Queries PredictionTracking for unresolved 'live' predictions with a past
event_date, then runs update_results for each event URL.
"""
import os
import subprocess
import sys
from pathlib import Path

import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]
MODELS_DIR = Path(__file__).resolve().parents[1] / "DelphiAIApp" / "Models"


def get_unresolved_events():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT event_url, event_name
        FROM predictiontracking
        WHERE prediction_type = 'live'
          AND was_correct IS NULL
          AND event_url IS NOT NULL
          AND event_url <> ''
          AND event_date IS NOT NULL
          AND event_date::date < CURRENT_DATE
        ORDER BY event_url
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def main():
    events = get_unresolved_events()
    if not events:
        print("[resolve] No unresolved past events found.")
        return

    print(f"[resolve] Found {len(events)} unresolved event(s):")
    for url, name in events:
        print(f"  - {name} ({url})")

    for url, name in events:
        print(f"\n[resolve] Running update_results for: {name}")
        result = subprocess.run(
            [sys.executable, "-m", "ml.update_results", name, "--url", url],
            cwd=str(MODELS_DIR),
            env={**os.environ},
        )
        if result.returncode != 0:
            print(f"[resolve] WARNING: update_results failed for {name}")
        else:
            print(f"[resolve] OK: {name}")


if __name__ == "__main__":
    main()

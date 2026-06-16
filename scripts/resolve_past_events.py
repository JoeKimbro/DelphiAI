"""
Auto-resolve predictions for events whose date has passed.

For each unresolved 'live' event we decide "has this card happened?" using the
most reliable date signal available, then run update_results for the past ones.

The old version hard-required a parseable ISO event_date in the past. That
silently excluded two whole classes of event — they never resolved and so never
moved onto the Past Events page (which gates on was_correct IS NOT NULL):
  - branded/numbered cards stored with an empty event_date (e.g. "UFC Freedom
    250", "UFC 326"); and
  - any event whose event_date is a display string ("Sat, May 16 / 8:00 PM").

Date resolution, in order of trust:
  1. A definitive, explicit-year date — from the event URL slug
     (ufc-fight-night-june-20-2026) or an ISO event_date. If we have one, it
     alone decides past vs. future.
  2. Otherwise fall back to predicted_at. Predictions are written within days
     of the card, so a prediction that is no longer "today" means the event has
     effectively happened. We deliberately do NOT trust year-less display dates
     here ("Sat, Mar 7" with no year is ambiguous across year boundaries).

update_results is idempotent and safe to call on an event that hasn't happened
yet — it simply finds no posted results and leaves the rows unresolved — so the
occasional eager scrape of a not-quite-started card costs nothing but a GET.
"""
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]
MODELS_DIR = Path(__file__).resolve().parents[1] / "DelphiAIApp" / "Models"

_SLUG_DATE_RE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)[- ](\d{1,2})[- ](\d{4})",
    re.IGNORECASE,
)


def _definitive_event_date(event_url, event_date):
    """Return an explicit-year date for the event, or None if we don't have one.

    Trusts only sources that carry a year: the URL slug
    (ufc-fight-night-june-20-2026) and an ISO event_date (2026-05-16). A
    year-less display string like "Sat, Mar 7 / 9:00 PM" is intentionally
    treated as no signal — see module docstring.
    """
    slug = (event_url or "").rstrip("/").split("/event/")[-1]
    m = _SLUG_DATE_RE.search(slug)
    if m:
        try:
            return datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y"
            ).date()
        except ValueError:
            pass

    s = (event_date or "").strip().split("/")[0].strip().split("T")[0]
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            pass
    return None


def get_unresolved_events():
    """Return [(event_url, event_name)] for unresolved live cards that have happened."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT event_url,
               MAX(event_name)   AS event_name,
               MAX(event_date)   AS event_date,
               MAX(predicted_at) AS predicted_at
        FROM predictiontracking
        WHERE prediction_type = 'live'
          AND was_correct IS NULL
          AND event_url IS NOT NULL
          AND event_url <> ''
        GROUP BY event_url
        ORDER BY event_url
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    today = date.today()
    out = []
    for url, name, event_date, predicted_at in rows:
        definitive = _definitive_event_date(url, event_date)
        if definitive is not None:
            if definitive < today:
                out.append((url, name))
            continue  # definitive future date — not yet
        # No trustworthy date: use prediction recency as a proxy.
        if predicted_at is not None and predicted_at.date() < today:
            out.append((url, name))
    return out


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

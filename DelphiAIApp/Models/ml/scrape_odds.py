"""Scheduled BestFightOdds snapshot captures.

Captures the best available moneyline for every upcoming UFC fight at two
points in the event week:
  - 'wednesday': the Wednesday immediately before the event (within 6 days)
  - 'day_before': the calendar day before the event date

Running on the same day is idempotent (ON CONFLICT upsert). A BFO scrape
failure leaves existing rows untouched and is reported as a non-fatal exit 1.

Usage
-----
    # Automatic: capture all events due today
    python -m ml.scrape_odds

    # Force a specific event + label (testing / missed day)
    python -m ml.scrape_odds --event ufc-fight-night-march-22-2025 --label day_before

    # Dry-run: scrape + match + print, no DB write
    python -m ml.scrape_odds --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import psycopg2
import requests
from bs4 import BeautifulSoup

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from DelphiAIApp.Models.db.postgres import get_db_connection
from DelphiAIApp.Services.odds_math import american_to_decimal

logger = logging.getLogger(__name__)

_BFO_URL = "https://www.bestfightodds.com/"
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UpcomingEvent:
    event_url: str
    event_name: str
    event_date: date
    fights: list[tuple[str, str]] = field(default_factory=list)


class BfoFight(NamedTuple):
    fighter_a: str
    fighter_b: str
    american_a: int
    american_b: int
    decimal_a: float
    decimal_b: float


class MatchedFight(NamedTuple):
    fighter1_name: str   # our canonical name
    fighter2_name: str
    f1_best_american: int
    f2_best_american: int
    f1_best_decimal: float
    f2_best_decimal: float


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def upcoming_events_with_dates(conn) -> list[UpcomingEvent]:
    """Return upcoming events (future date, unresolved) with their fight pairs."""
    cur = conn.cursor()
    cur.execute("""
        SELECT event_url, event_name, event_date, fighter1_name, fighter2_name
        FROM PredictionTracking
        WHERE was_correct IS NULL
          AND event_date IS NOT NULL
          AND event_date ~ '^\\d{4}-\\d{2}-\\d{2}'
          AND event_date::date >= CURRENT_DATE
        ORDER BY event_date ASC, event_url, fighter1_name
    """)
    rows = cur.fetchall()
    cur.close()

    events: dict[str, UpcomingEvent] = {}
    for event_url, event_name, event_date_raw, f1, f2 in rows:
        if event_url not in events:
            try:
                ed = (event_date_raw if isinstance(event_date_raw, date)
                      else date.fromisoformat(str(event_date_raw)[:10]))
            except (ValueError, TypeError):
                continue
            events[event_url] = UpcomingEvent(
                event_url=event_url,
                event_name=event_name or event_url,
                event_date=ed,
            )
        events[event_url].fights.append((f1, f2))

    return list(events.values())


def already_captured(conn, event_url: str, label: str) -> bool:
    """True if we already have at least one fight_odds row for this event+label."""
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM fight_odds WHERE event_url = %s AND capture_label = %s LIMIT 1",
        (event_url, label),
    )
    found = cur.fetchone() is not None
    cur.close()
    return found


def store_odds(conn, event: UpcomingEvent, matched: list[MatchedFight], label: str) -> int:
    """Upsert matched fight odds into fight_odds. Returns rows written."""
    cur = conn.cursor()
    written = 0
    for m in matched:
        cur.execute("""
            INSERT INTO fight_odds
                (event_url, event_name, event_date,
                 fighter1_name, fighter2_name,
                 f1_best_american, f2_best_american,
                 f1_best_decimal, f2_best_decimal,
                 capture_label, source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'bestfightodds')
            ON CONFLICT (event_url, fighter1_name, fighter2_name, capture_label)
            DO UPDATE SET
                f1_best_american = EXCLUDED.f1_best_american,
                f2_best_american = EXCLUDED.f2_best_american,
                f1_best_decimal  = EXCLUDED.f1_best_decimal,
                f2_best_decimal  = EXCLUDED.f2_best_decimal,
                captured_at      = NOW()
        """, (
            event.event_url, event.event_name, event.event_date,
            m.fighter1_name, m.fighter2_name,
            m.f1_best_american, m.f2_best_american,
            m.f1_best_decimal, m.f2_best_decimal,
            label,
        ))
        written += cur.rowcount
    conn.commit()
    cur.close()
    return written


# ---------------------------------------------------------------------------
# Capture scheduling logic
# ---------------------------------------------------------------------------

def _wednesday_before(event_date: date) -> date | None:
    """The Wednesday immediately before event_date, within a 6-day window."""
    for delta in range(1, 7):
        candidate = event_date - timedelta(days=delta)
        if candidate.weekday() == 2:  # Wednesday
            return candidate
    return None


def due_captures(events: list[UpcomingEvent], today: date) -> list[tuple[UpcomingEvent, str]]:
    """Return (event, label) pairs that need a snapshot captured today."""
    result: list[tuple[UpcomingEvent, str]] = []
    for ev in events:
        day_before = ev.event_date - timedelta(days=1)
        wednesday = _wednesday_before(ev.event_date)

        if today == day_before:
            result.append((ev, "day_before"))
        elif wednesday and today == wednesday:
            result.append((ev, "wednesday"))
    return result


# ---------------------------------------------------------------------------
# BFO scraping + name matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str | None) -> str:
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", n.strip().lower())


def _surname(name: str) -> str:
    """Normalized last token of a name (used for fuzzy matching)."""
    parts = _normalize_name(name).split()
    return parts[-1] if parts else ""


def fetch_bfo_odds() -> list[BfoFight]:
    """Scrape the BFO homepage and return all fight lines found."""
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    resp = requests.get(_BFO_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    return _parse_bfo_html(resp.text)


def _parse_american(text: str | None) -> int | None:
    if not text:
        return None
    m = re.match(r"^([+\-]\d{2,5})", text.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_bfo_html(html: str) -> list[BfoFight]:
    soup = BeautifulSoup(html, "html.parser")
    fights: list[BfoFight] = []

    for event_div in soup.select("div.table-div"):
        body_table = event_div.select_one(".table-scroller table.odds-table")
        if body_table is None:
            continue

        pending: list[tuple[str, int]] = []
        for tr in body_table.find_all("tr"):
            anchor = tr.find("a", href=lambda h: bool(h) and "/fighters/" in h)
            if not anchor:
                continue
            name = anchor.get_text(strip=True)
            american: int | None = None
            for td in tr.find_all("td"):
                parsed = _parse_american(td.get_text(strip=True))
                if parsed is not None:
                    american = parsed
                    break
            if american is None:
                continue
            pending.append((name, american))

            if len(pending) == 2:
                (n_a, am_a), (n_b, am_b) = pending
                pending = []
                try:
                    dec_a = american_to_decimal(am_a)
                    dec_b = american_to_decimal(am_b)
                except ValueError:
                    continue
                fights.append(BfoFight(n_a, n_b, am_a, am_b, dec_a, dec_b))

    return fights


def match_fights(our_event: UpcomingEvent, bfo_fights: list[BfoFight]) -> list[MatchedFight]:
    """Map BFO fights to our canonical fighter names by normalized surname."""
    matched: list[MatchedFight] = []
    for our_f1, our_f2 in our_event.fights:
        s1 = _surname(our_f1)
        s2 = _surname(our_f2)
        if not s1 or not s2:
            continue

        hit: BfoFight | None = None
        orientation: str | None = None  # "normal" or "swapped"
        for bf in bfo_fights:
            ba = _surname(bf.fighter_a)
            bb = _surname(bf.fighter_b)
            if ba == s1 and bb == s2:
                hit, orientation = bf, "normal"
                break
            if ba == s2 and bb == s1:
                hit, orientation = bf, "swapped"
                break

        if hit is None:
            logger.warning("  no BFO match: %s vs %s", our_f1, our_f2)
            continue

        if orientation == "normal":
            matched.append(MatchedFight(
                our_f1, our_f2,
                hit.american_a, hit.american_b,
                hit.decimal_a, hit.decimal_b,
            ))
        else:
            matched.append(MatchedFight(
                our_f1, our_f2,
                hit.american_b, hit.american_a,
                hit.decimal_b, hit.decimal_a,
            ))

    return matched


# ---------------------------------------------------------------------------
# DB lookup used by event_service to prefer stored snapshots
# ---------------------------------------------------------------------------

def lookup_best_odds(conn, event_url: str, f1_name: str, f2_name: str) -> dict | None:
    """
    Return the best stored snapshot for a fight.

    Prefers 'day_before' over 'wednesday'. Returns None if no row exists.
    Result shape matches odds_provider rows so _enrich_with_odds can use it directly:
        {american_a, american_b, decimal_a, decimal_b, source, capture_label}
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT f1_best_american, f2_best_american,
               f1_best_decimal,  f2_best_decimal,
               capture_label
        FROM fight_odds
        WHERE event_url = %s
          AND fighter1_name = %s
          AND fighter2_name = %s
        ORDER BY CASE capture_label WHEN 'day_before' THEN 0 ELSE 1 END,
                 captured_at DESC
        LIMIT 1
    """, (event_url, f1_name, f2_name))
    row = cur.fetchone()
    cur.close()

    if row is None:
        return None

    am_a, am_b, dec_a, dec_b, label = row
    if am_a is None or am_b is None:
        return None

    return {
        "american_a": int(am_a),
        "american_b": int(am_b),
        "decimal_a": float(dec_a),
        "decimal_b": float(dec_b),
        "source": "bestfightodds",
        "capture_label": label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture BestFightOdds snapshots.")
    parser.add_argument("--event", help="Force a specific event URL slug (skips due_captures)")
    parser.add_argument("--label", choices=["wednesday", "day_before"],
                        help="Capture label when --event is set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scrape + match + print; no DB write")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[scrape_odds] %(message)s")

    try:
        bfo_fights = fetch_bfo_odds()
    except Exception as e:
        logger.error("BFO scrape failed: %s", e)
        return 1

    logger.info("BFO homepage: %d fights found", len(bfo_fights))

    with get_db_connection() as conn:
        if args.event:
            if not args.label:
                logger.error("--label required when --event is specified")
                return 1
            events = upcoming_events_with_dates(conn)
            ev = next((e for e in events if e.event_url.endswith(args.event)
                       or args.event in e.event_url), None)
            if ev is None:
                logger.error("Event not found in DB: %s", args.event)
                return 1
            to_capture = [(ev, args.label)]
        else:
            events = upcoming_events_with_dates(conn)
            today = date.today()
            to_capture = due_captures(events, today)
            logger.info("Events due today (%s): %d", today.isoformat(), len(to_capture))

        if not to_capture:
            logger.info("No captures due — nothing to do.")
            return 0

        total_written = 0
        for ev, label in to_capture:
            logger.info("Capturing %s [%s]", ev.event_name, label)
            matched = match_fights(ev, bfo_fights)
            logger.info("  matched %d / %d fights", len(matched), len(ev.fights))

            if args.dry_run:
                for m in matched:
                    logger.info("  DRY  %s vs %s  f1=%+d  f2=%+d",
                                m.fighter1_name, m.fighter2_name,
                                m.f1_best_american, m.f2_best_american)
                continue

            written = store_odds(conn, ev, matched, label)
            total_written += written
            logger.info("  stored %d rows", written)

        if not args.dry_run:
            logger.info("Done — %d rows total", total_written)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

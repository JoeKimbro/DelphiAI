"""
DelphiAI Upcoming Event Prefetcher

Pre-runs predictions for the next N upcoming UFC events so that when a user
clicks an event in the dashboard, the page renders from PredictionTracking
(sub-100ms) instead of running the full predict pipeline on demand
(which scrapes injury pages from UFC.com — several seconds per fighter).

When run repeatedly (e.g. via Windows Task Scheduler), each invocation:

  1. Scrapes UFC.com for upcoming events (uses predict_card.scrape_upcoming_events).
  2. Takes the top N events whose date is today or later.
  3. For each event, computes a SHA-256 fingerprint over every fighter-state
     field the model consumes (ELO, adjusted ELO, penalties, career stats,
     historical-feature rolling windows). The fingerprint is cheap — pure
     SELECTs against indexed columns; no UFC.com hit.
  4. Compares the fresh fingerprint against the one stored in PredictionTracking
     for that fight pair. If every fight matches, the event is skipped entirely
     (no injury scrape, no model call, no DB write).
  5. If anything changed (new fight added, ELO updated, days-since-last-fight
     ticked over, etc.) it runs the full predict pipeline; the existing
     save_predictions() UPSERTs by (event_name, fighter1, fighter2,
     prediction_type='live'); a final pass writes the new fingerprints.

Resolved events (was_correct IS NOT NULL) are left in place — they power
performance_summary, the accuracy KPIs, and the per-event history table.

Usage (from DelphiAIApp/Models/):
    python -m ml.prefetch_predictions             # top 4 upcoming events
    python -m ml.prefetch_predictions --top 6     # custom N
    python -m ml.prefetch_predictions --force     # ignore fingerprints, re-run

Exit codes: 0 on success, 1 on DB connection failure.
"""
import os
import sys
import io
import json
import hashlib
import logging
import argparse
import contextlib
from pathlib import Path
from datetime import datetime, date

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

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

try:
    from ml.predict_card import (
        scrape_upcoming_events,
        scrape_event_card,
        predict_full_card,
        _fuzzy_fighter_lookup,
    )
    from ml.predict_fight import get_fighter_data
    from ml.update_results import ensure_tracking_table
except (ModuleNotFoundError, ImportError):
    from predict_card import (
        scrape_upcoming_events,
        scrape_event_card,
        predict_full_card,
        _fuzzy_fighter_lookup,
    )
    from predict_fight import get_fighter_data
    from update_results import ensure_tracking_table

logger = logging.getLogger(__name__)


# ============================================================================
# DATE PARSING — UFC.com event date strings are inconsistent
# ============================================================================

def _parse_event_date(date_str):
    """Best-effort parse → datetime.date. Returns None if unparseable.

    Handles "Sat, Nov 15", "Sat, Nov 15 / 10:00 PM EST", "2026-01-18", etc.
    When the year is missing, assumes current year (or next year if the date
    would otherwise be in the past).
    """
    if not date_str:
        return None
    s = date_str.strip()
    s = s.split('/')[0].strip()
    if ',' in s:
        head, tail = s.split(',', 1)
        # If head looks like a weekday, drop it; otherwise keep both parts.
        if head.strip().lower()[:3] in {'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'}:
            s = tail.strip()
    try:
        return datetime.fromisoformat(s.split('T')[0]).date()
    except ValueError:
        pass
    today = date.today()
    for fmt in ('%b %d %Y', '%B %d %Y', '%b %d', '%B %d'):
        try:
            d = datetime.strptime(s, fmt).date()
            if '%Y' not in fmt:
                d = d.replace(year=today.year)
                if d < today:
                    d = d.replace(year=today.year + 1)
            return d
        except ValueError:
            continue
    return None


# ============================================================================
# FIGHTER-STATE FINGERPRINT
# ============================================================================

def _norm(v):
    """Stable serialization for JSON canonicalization."""
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, (int, float, bool)):
        return v
    return str(v)


_FP_FIELDS = (
    'elo', 'adj_elo', 'inactivity', 'injury',
    'slpm', 'stracc', 'tdavg', 'subavg',
    'streak3', 'r1finish',
    'days_inactive', 'last_fight',
    'elo_velocity', 'win_streak', 'avg_opp_elo',
)


def _compute_fight_fingerprint(conn, fighter1_id, fighter2_id, event_date_str, is_title):
    """SHA-256 over every fighter-state field that influences the prediction.

    Returns hex digest, or None if either fighter is missing from the DB.
    Cheap — two indexed SELECTs (one per fighter), no scraping.
    """
    if not fighter1_id or not fighter2_id:
        return None

    cur = conn.cursor()
    state = {'event_date': str(event_date_str or ''), 'is_title': bool(is_title)}
    try:
        for label, fid in (('f1', fighter1_id), ('f2', fighter2_id)):
            cur.execute("""
                SELECT
                    cs.EloRating, cs.AdjustedEloRating,
                    cs.InactivityPenalty, cs.InjuryPenalty,
                    cs.SLpM, cs.StrAcc, cs.TDAvg, cs.SubAvg,
                    cs.WinStreak_Last3, cs.FirstRoundFinishRate,
                    fs.DaysSinceLastFight, fs.LastFightDate,
                    hf.EloVelocity, hf.CurrentWinStreak, hf.AvgOpponentEloLast3
                FROM FighterStats fs
                LEFT JOIN CareerStats cs ON cs.FighterID = fs.FighterID
                LEFT JOIN LATERAL (
                    SELECT EloVelocity, CurrentWinStreak, AvgOpponentEloLast3
                    FROM FighterHistoricalFeatures
                    WHERE FighterID = fs.FighterID
                    ORDER BY FightDate DESC LIMIT 1
                ) hf ON TRUE
                WHERE fs.FighterID = %s
            """, (fid,))
            row = cur.fetchone()
            if not row:
                return None
            state[label] = {k: _norm(v) for k, v in zip(_FP_FIELDS, row)}
    finally:
        cur.close()

    canon = json.dumps(state, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()


# ============================================================================
# CACHED FINGERPRINT LOOKUP / STORAGE
# ============================================================================

def _fetch_cached_fingerprints(conn, event_name):
    """{(fighter1_name, fighter2_name): fingerprint_or_None} for unresolved rows."""
    cur = conn.cursor()
    cur.execute("""
        SELECT fighter1_name, fighter2_name, input_fingerprint
        FROM PredictionTracking
        WHERE event_name = %s
          AND prediction_type = 'live'
          AND was_correct IS NULL
    """, (event_name,))
    out = {(f1, f2): fp for (f1, f2, fp) in cur.fetchall()}
    cur.close()
    return out


def _store_fingerprints(conn, event_name, fingerprints):
    """Write per-fight fingerprints to the rows save_predictions just upserted."""
    rows = [(fp, event_name, f1, f2) for (f1, f2), fp in fingerprints.items() if fp]
    if not rows:
        return 0
    cur = conn.cursor()
    execute_values(cur, """
        UPDATE PredictionTracking AS pt SET input_fingerprint = data.fp
        FROM (VALUES %s) AS data(fp, event_name, f1, f2)
        WHERE pt.event_name = data.event_name
          AND pt.fighter1_name = data.f1
          AND pt.fighter2_name = data.f2
          AND pt.prediction_type = 'live'
    """, rows)
    written = cur.rowcount
    conn.commit()
    cur.close()
    return written


# ============================================================================
# FIGHTER LOOKUP — mirrors the fallback chain in predict_card.predict_full_card
# ============================================================================

def _resolve_fighter(conn, scraped_name):
    """Return (canonical_name, fighter_id) for a scraped name, or (scraped_name, None)
       if not in DB. Uses the same fuzzy fallback path as predict_card so we
       agree with it about both fighter identity AND the canonical name —
       predict_full_card writes the canonical name into PredictionTracking
       (predict_card.py:908), so our fingerprint UPDATE must key off the same."""
    f = get_fighter_data(conn, scraped_name)
    if not f and ' ' in scraped_name:
        parts = scraped_name.split()
        f = _fuzzy_fighter_lookup(
            conn, get_fighter_data,
            parts[-1],
            parts[0][0] if parts[0] else '',
            full_name=scraped_name,
        )
    if not f:
        return scraped_name, None
    canon = f.get('name') or f.get('Name') or scraped_name
    fid = f.get('fighterid') or f.get('FighterID') or f.get('id')
    return canon, fid


# ============================================================================
# PER-EVENT ORCHESTRATION
# ============================================================================

def _prefetch_one_event(conn, event_url, force=False):
    """Returns a one-line status string for logging."""
    card = scrape_event_card(event_url)
    if not card or not card.get('fights'):
        return "  SKIP (no card data)"

    event_name = card.get('event_name', '')
    event_date_str = card.get('event_date', '')
    event_date = _parse_event_date(event_date_str)
    fights = card.get('fights', [])

    if event_date and event_date < date.today():
        return f"  SKIP (past event: {event_date})"

    # Compute pre-prediction fingerprints for every fight on this card.
    # IMPORTANT: key every fingerprint dict on the CANONICAL DB name, not the
    # scraped name. predict_full_card writes the canonical name into
    # PredictionTracking, so any UPDATE/SELECT we do here must use the same
    # name or it won't find the row.
    #
    # Also: the prediction pipeline can mutate DB state (refreshes injury
    # data, recomputes AdjustedEloRating). We compare PRE-state against the
    # POST-state stored by the previous run — they should match because the
    # previous run's POST-state IS the current run's PRE-state (until a fight
    # resolves or weekly_update runs).
    fight_meta = []  # [(canon_f1, canon_f2, f1_id, f2_id, is_title), ...]
    pre_fps = {}
    missing_fighters = 0
    for fight in fights:
        canon_f1, f1_id = _resolve_fighter(conn, fight.get('fighter1', ''))
        canon_f2, f2_id = _resolve_fighter(conn, fight.get('fighter2', ''))
        is_title = fight.get('is_title', False)
        fp = _compute_fight_fingerprint(conn, f1_id, f2_id, event_date_str, is_title)
        pre_fps[(canon_f1, canon_f2)] = fp
        fight_meta.append((canon_f1, canon_f2, f1_id, f2_id, is_title))
        if not fp:
            missing_fighters += 1

    # Decide: are we re-predicting, skipping outright, or just backfilling
    # missing fingerprints (cheap UPDATE — no model call, no scrape)?
    if force:
        decision = "forced re-predict"
        cached_fps = None
    else:
        cached_fps = _fetch_cached_fingerprints(conn, event_name)
        # We only compare fights we could fingerprint (missing_fighters can't be).
        comparable = {k: v for k, v in pre_fps.items() if v}
        # A fight forces a re-predict only if its cached fingerprint exists AND
        # disagrees with the freshly-computed one (i.e. state actually changed).
        # A NULL cached fingerprint means "we predicted this fight but never
        # stored its hash" — no state change is implied, just backfill it.
        state_changed = any(
            cached_fps.get(k) is not None and cached_fps[k] != v
            for k, v in comparable.items()
        )
        # If any comparable fight is missing from cached entirely, the card
        # changed (new matchup added) → safer to re-predict.
        missing_from_cache = any(k not in cached_fps for k in comparable)
        backfillable = {
            k: v for k, v in comparable.items()
            if k in cached_fps and cached_fps[k] is None
        }
        if not state_changed and not missing_from_cache:
            if backfillable:
                written = _store_fingerprints(conn, event_name, backfillable)
                note = f" ({missing_fighters} missing)" if missing_fighters else ""
                return (f"  unchanged — kept {len(comparable)} cached fights, "
                        f"backfilled {written} fingerprints{note}")
            note = f" ({missing_fighters} missing)" if missing_fighters else ""
            return f"  unchanged — kept {len(comparable)} cached fights{note}"
        decision = "fingerprint changed" if cached_fps else "first prediction"

    # Run the full pipeline. predict_full_card prints heavily; capture stdout
    # so the prefetch log stays compact.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = predict_full_card(
            conn,
            fights,
            event_name=event_name,
            event_date=event_date_str,
            event_url=event_url,
        )
    if not results:
        return "  FAILED (pipeline returned no results)"

    # Recompute fingerprints from POST-prediction state — the pipeline may have
    # updated InjuryPenalty / AdjustedEloRating during calculate_elo_adjustments.
    # Storing the post-state hash makes the next run's pre-check a clean match.
    post_fps = {}
    for canon_f1, canon_f2, f1_id, f2_id, is_title in fight_meta:
        post_fps[(canon_f1, canon_f2)] = _compute_fight_fingerprint(
            conn, f1_id, f2_id, event_date_str, is_title
        )

    written = _store_fingerprints(conn, event_name, post_fps)
    note = f" ({missing_fighters} fighters missing from DB)" if missing_fighters else ""
    return f"  {decision} — {len(results)} fights predicted, {written} fingerprints stored{note}"


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pre-run predictions for the next N upcoming UFC events.',
    )
    parser.add_argument('--top', type=int, default=4,
                        help='Number of upcoming events to prefetch (default: 4)')
    parser.add_argument('--force', action='store_true',
                        help='Ignore fingerprints and re-run predictions for every event')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    started = datetime.now()
    print(f"[prefetch] Started {started.isoformat(timespec='seconds')} — top={args.top} force={args.force}")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[prefetch] ERROR: cannot connect to DB: {e}")
        return 1

    try:
        ensure_tracking_table(conn)

        print("[prefetch] Scraping UFC.com for upcoming events...")
        upcoming = scrape_upcoming_events() or []
        if not upcoming:
            print("[prefetch] No upcoming events found on UFC.com.")
            return 0

        processed = 0
        for ev in upcoming:
            if processed >= args.top:
                break
            url = ev.get('url')
            name = ev.get('name', '<unknown>')
            if not url:
                continue
            print(f"\n[prefetch] {name}")
            print(f"           {url}")
            try:
                status = _prefetch_one_event(conn, url, force=args.force)
            except Exception as e:
                logger.exception("prefetch failed for %s", url)
                status = f"  ERROR: {e}"
            print(status)
            processed += 1
    finally:
        conn.close()

    elapsed = (datetime.now() - started).total_seconds()
    print(f"\n[prefetch] Done in {elapsed:.1f}s.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

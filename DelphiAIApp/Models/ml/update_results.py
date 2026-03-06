"""
UFC Event Results Tracker & Performance Reporter

After an event completes, fetch actual fight results and compare
against stored predictions to validate model accuracy.

Usage:
    cd DelphiAIApp/Models

    # Update results for a specific event
    python -m ml.update_results "Strickland vs Hernandez"

    # Use a specific UFC.com URL for result scraping
    python -m ml.update_results "Strickland vs Hernandez" --url "https://www.ufc.com/event/..."

    # List all tracked events
    python -m ml.update_results --list

    # Re-fetch results even if already resolved
    python -m ml.update_results "Strickland vs Hernandez" --force
"""

import os
import sys
import re
import argparse
import time
import random
import logging
import unicodedata
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import psycopg2
import requests
from bs4 import BeautifulSoup
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

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]

logger = logging.getLogger(__name__)


# ============================================================================
# PREDICTION TRACKING TABLE
# ============================================================================

def ensure_tracking_table(conn):
    """Create PredictionTracking table if it doesn't exist."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS PredictionTracking (
            id SERIAL PRIMARY KEY,
            event_name VARCHAR(200) NOT NULL,
            event_date VARCHAR(50),
            event_url VARCHAR(255),
            fighter1_name VARCHAR(100) NOT NULL,
            fighter2_name VARCHAR(100) NOT NULL,
            fighter1_id INTEGER,
            fighter2_id INTEGER,
            is_title_fight BOOLEAN DEFAULT FALSE,
            pick_name VARCHAR(100) NOT NULL,
            pick_fighter_id INTEGER,
            pick_probability DECIMAL(5,4) NOT NULL,
            fighter1_probability DECIMAL(5,4) NOT NULL,
            confidence VARCHAR(10) NOT NULL,
            prob_source VARCHAR(50),
            predicted_method VARCHAR(50),
            predicted_ko DECIMAL(5,4),
            predicted_sub DECIMAL(5,4),
            predicted_dec DECIMAL(5,4),
            predicted_r1 DECIMAL(5,4),
            predicted_r2 DECIMAL(5,4),
            predicted_r3 DECIMAL(5,4),
            fighter1_elo DECIMAL(8,2),
            fighter2_elo DECIMAL(8,2),
            actual_winner_name VARCHAR(100),
            actual_winner_id INTEGER,
            actual_method VARCHAR(50),
            actual_round INTEGER,
            was_correct BOOLEAN,
            prediction_type VARCHAR(10) DEFAULT 'live',
            model_version VARCHAR(50) DEFAULT 'v3',
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
        )
    """)
    # Add prediction_type column if upgrading from older schema
    cur.execute("""
        ALTER TABLE PredictionTracking
        ADD COLUMN IF NOT EXISTS prediction_type VARCHAR(10) DEFAULT 'live'
    """)
    # Indexes
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pt_event_name
        ON PredictionTracking(event_name)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pt_was_correct
        ON PredictionTracking(was_correct) WHERE was_correct IS NOT NULL
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pt_prediction_type
        ON PredictionTracking(prediction_type)
    """)
    conn.commit()
    cur.close()


# ============================================================================
# SAVE PREDICTIONS (called from predict_card.py after generating predictions)
# ============================================================================

def save_predictions(conn, results, event_name, event_date='', event_url='',
                     prediction_type='live'):
    """
    Save card prediction results to the PredictionTracking table.

    Args:
        conn: Database connection
        results: List of result dicts from predict_full_card()
        event_name: Event name string
        event_date: Event date string
        event_url: UFC.com event URL (if available)
        prediction_type: 'live' for real-time predictions, 'backtest' for historical

    Returns:
        Number of predictions saved/updated
    """
    ensure_tracking_table(conn)
    cur = conn.cursor()

    saved = 0
    for r in results:
        pick_id = r.get('f1_id') if r['pick'] == r['fighter1'] else r.get('f2_id')

        # Dedup check: same event + same fighter pair + same type
        cur.execute("""
            SELECT id FROM PredictionTracking
            WHERE event_name = %s
              AND fighter1_name = %s AND fighter2_name = %s
              AND prediction_type = %s
        """, (event_name, r['fighter1'], r['fighter2'], prediction_type))

        existing = cur.fetchone()

        if existing:
            cur.execute("""
                UPDATE PredictionTracking SET
                    pick_name = %s, pick_fighter_id = %s,
                    pick_probability = %s, fighter1_probability = %s,
                    confidence = %s, prob_source = %s,
                    predicted_method = %s,
                    predicted_ko = %s, predicted_sub = %s, predicted_dec = %s,
                    predicted_r1 = %s, predicted_r2 = %s, predicted_r3 = %s,
                    fighter1_elo = %s, fighter2_elo = %s,
                    event_url = COALESCE(NULLIF(%s, ''), event_url),
                    predicted_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (
                r['pick'], pick_id,
                r['pick_pct'], r['f1_prob'],
                r['confidence'], r['prob_source'],
                r['method'],
                r['ko_pct'], r['sub_pct'], r['dec_pct'],
                r['r1_finish'], r['r2_finish'], r['r3_finish'],
                r['f1_elo'], r['f2_elo'],
                event_url or '',
                existing[0],
            ))
        else:
            cur.execute("""
                INSERT INTO PredictionTracking (
                    event_name, event_date, event_url,
                    fighter1_name, fighter2_name, fighter1_id, fighter2_id,
                    is_title_fight,
                    pick_name, pick_fighter_id, pick_probability, fighter1_probability,
                    confidence, prob_source,
                    predicted_method, predicted_ko, predicted_sub, predicted_dec,
                    predicted_r1, predicted_r2, predicted_r3,
                    fighter1_elo, fighter2_elo,
                    prediction_type
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s
                )
            """, (
                event_name, str(event_date) if event_date else '',
                event_url or '',
                r['fighter1'], r['fighter2'],
                r.get('f1_id'), r.get('f2_id'),
                r.get('is_title', False),
                r['pick'], pick_id,
                r['pick_pct'], r['f1_prob'],
                r['confidence'], r['prob_source'],
                r['method'],
                r['ko_pct'], r['sub_pct'], r['dec_pct'],
                r['r1_finish'], r['r2_finish'], r['r3_finish'],
                r['f1_elo'], r['f2_elo'],
                prediction_type,
            ))

        saved += 1

    conn.commit()
    cur.close()
    return saved


# ============================================================================
# FIND & RETRIEVE STORED PREDICTIONS
# ============================================================================

def list_tracked_events(conn):
    """List all events with stored predictions and their resolution status."""
    ensure_tracking_table(conn)
    cur = conn.cursor()
    cur.execute("""
        SELECT event_name, event_date,
               COUNT(*) as total,
               COUNT(CASE WHEN was_correct IS NOT NULL THEN 1 END) as resolved,
               COUNT(CASE WHEN was_correct = TRUE THEN 1 END) as correct,
               MIN(predicted_at) as first_predicted
        FROM PredictionTracking
        GROUP BY event_name, event_date
        ORDER BY first_predicted DESC NULLS LAST
    """)

    events = []
    for row in cur.fetchall():
        events.append({
            'event_name': row[0],
            'event_date': row[1],
            'total': row[2],
            'resolved': row[3],
            'correct': row[4],
            'predicted_at': row[5],
        })

    cur.close()
    return events


def find_event_predictions(conn, search_term):
    """
    Find stored predictions by event name (fuzzy keyword search).

    Returns: (event_name, list_of_prediction_dicts)
    """
    ensure_tracking_table(conn)
    cur = conn.cursor()

    # Extract keywords for flexible matching
    keywords = re.split(r'\s+(?:vs\.?|versus)\s+|\s+', search_term.strip())
    keywords = [k for k in keywords if len(k) > 1]

    if not keywords:
        cur.close()
        return None, []

    # Build ILIKE conditions
    conditions = []
    params = []
    for kw in keywords:
        conditions.append("event_name ILIKE %s")
        params.append(f'%{kw}%')

    where = " AND ".join(conditions)

    # Find matching event names
    cur.execute(f"""
        SELECT DISTINCT event_name
        FROM PredictionTracking
        WHERE {where}
        ORDER BY event_name
    """, params)

    event_names = [row[0] for row in cur.fetchall()]

    if not event_names:
        cur.close()
        return None, []

    # Use the first match (most relevant)
    chosen_event = event_names[0]

    # Get all predictions for this event
    cur.execute("""
        SELECT id, event_name, event_date, event_url,
               fighter1_name, fighter2_name, fighter1_id, fighter2_id,
               is_title_fight,
               pick_name, pick_fighter_id, pick_probability, fighter1_probability,
               confidence, prob_source,
               predicted_method, predicted_ko, predicted_sub, predicted_dec,
               predicted_r1, predicted_r2, predicted_r3,
               fighter1_elo, fighter2_elo,
               actual_winner_name, actual_winner_id, actual_method, actual_round,
               was_correct, predicted_at, resolved_at
        FROM PredictionTracking
        WHERE event_name = %s
        ORDER BY id
    """, (chosen_event,))

    columns = [desc[0] for desc in cur.description]
    predictions = [dict(zip(columns, row)) for row in cur.fetchall()]

    cur.close()
    return chosen_event, predictions


# ============================================================================
# FETCH ACTUAL RESULTS FROM DATABASE (FIGHTS TABLE)
# ============================================================================

def _fetch_results_from_db(conn, predictions):
    """
    Try to find actual results in the Fights table for each prediction.

    Returns: dict mapping prediction_id -> result_dict
    """
    cur = conn.cursor()
    results = {}

    for pred in predictions:
        f1_id = pred['fighter1_id']
        f2_id = pred['fighter2_id']

        row = None

        if f1_id and f2_id:
            # Match by fighter IDs (most reliable)
            cur.execute("""
                SELECT winnername, winnerid, method, round, eventname
                FROM fights
                WHERE (
                    (fighterid = %s AND opponentid = %s) OR
                    (fighterid = %s AND opponentid = %s)
                )
                ORDER BY date DESC
                LIMIT 1
            """, (f1_id, f2_id, f2_id, f1_id))
            row = cur.fetchone()

        if not row:
            # Fallback: match by fighter last names + event name keywords
            f1_last = pred['fighter1_name'].split()[-1]
            f2_last = pred['fighter2_name'].split()[-1]
            cur.execute("""
                SELECT winnername, winnerid, method, round, eventname
                FROM fights
                WHERE (
                    (fightername ILIKE %s AND opponentname ILIKE %s) OR
                    (fightername ILIKE %s AND opponentname ILIKE %s)
                )
                ORDER BY date DESC
                LIMIT 1
            """, (
                f'%{f1_last}%', f'%{f2_last}%',
                f'%{f2_last}%', f'%{f1_last}%',
            ))
            row = cur.fetchone()

        if row and row[0]:
            results[pred['id']] = {
                'winner_name': row[0],
                'winner_id': row[1],
                'method': row[2],
                'round': row[3],
                'source': 'database',
            }

    cur.close()
    return results


# ============================================================================
# SCRAPE RESULTS FROM UFC.COM
# ============================================================================

def _get_session():
    """Create a requests session with random user agent."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    return session


def scrape_event_results(event_url):
    """
    Scrape completed event results from a UFC.com event page.

    Returns: list of dicts with keys:
        fighter1, fighter2, winner, method, round, time
    """
    session = _get_session()
    time.sleep(random.uniform(1, 3))

    try:
        resp = session.get(event_url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {event_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    fight_results = []

    fight_cards = soup.select('.c-listing-fight')

    for card in fight_cards:
        corners = card.select('.c-listing-fight__corner')
        if len(corners) < 2:
            corners = [
                card.select_one('.c-listing-fight__corner--red'),
                card.select_one('.c-listing-fight__corner--blue'),
            ]
            corners = [c for c in corners if c]

        if len(corners) < 2:
            continue

        names = []
        winner_idx = None

        for ci, corner in enumerate(corners):
            # Extract fighter name
            name = None

            # Strategy A: athlete link href slug
            athlete_link = corner.select_one('a[href*="/athlete/"]')
            if athlete_link:
                href = athlete_link.get('href', '')
                slug = href.rstrip('/').split('/athlete/')[-1]
                slug_clean = re.sub(r'-\d+$', '', slug)
                text = athlete_link.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text).strip()
                if ' ' in text and 3 < len(text) < 40:
                    name = text
                elif slug_clean:
                    name = slug_clean.replace('-', ' ').title()

            # Strategy B: corner name elements
            if not name:
                name_el = corner.select_one('.c-listing-fight__corner-name')
                if name_el:
                    name = name_el.get_text(separator=' ', strip=True)
                    name = re.sub(r'\s+', ' ', name).strip()

            # Strategy C: given + family name spans
            if not name:
                given = corner.select_one('.c-listing-fight__corner-given-name')
                family = corner.select_one('.c-listing-fight__corner-family-name')
                if given and family:
                    name = f"{given.get_text(strip=True)} {family.get_text(strip=True)}"

            names.append(name or 'Unknown')

            # Check for winner indicator
            outcome_el = corner.select_one(
                '.c-listing-fight__outcome, '
                '[class*="outcome"]'
            )
            if outcome_el:
                outcome_text = outcome_el.get_text(strip=True).lower()
                if 'win' in outcome_text:
                    winner_idx = ci

            # Also check for winner-specific CSS classes
            corner_classes = ' '.join(corner.get('class', []))
            if 'winner' in corner_classes.lower():
                winner_idx = ci

        if len(names) < 2:
            continue

        # Extract method and round from fight details
        method = ''
        round_num = None
        fight_time = ''

        # Try multiple selectors for result text
        for selector in [
            '.c-listing-fight__result-text.method',
            '.c-listing-fight__result-text',
            '.c-listing-fight__result',
        ]:
            result_el = card.select_one(selector)
            if result_el:
                result_text = result_el.get_text(separator=' ', strip=True)
                if result_text and not method:
                    method = result_text

        # Try to get round info
        round_el = card.select_one(
            '.c-listing-fight__result-text.round, '
            '.c-listing-fight__result-round'
        )
        if round_el:
            round_text = round_el.get_text(strip=True)
            round_match = re.search(r'(\d+)', round_text)
            if round_match:
                round_num = int(round_match.group(1))

        # Try to get time
        time_el = card.select_one(
            '.c-listing-fight__result-text.time, '
            '.c-listing-fight__result-time'
        )
        if time_el:
            fight_time = time_el.get_text(strip=True)

        # If we couldn't find round from dedicated element, parse from method text
        if round_num is None and method:
            round_match = re.search(r'R(\d+)|Round\s*(\d+)', method, re.IGNORECASE)
            if round_match:
                round_num = int(round_match.group(1) or round_match.group(2))

        # Determine winner name
        winner_name = None
        if winner_idx is not None and winner_idx < len(names):
            winner_name = names[winner_idx]

        # If no explicit winner indicator, check for draw/NC in method
        is_draw = False
        if not winner_name and method:
            if any(kw in method.lower() for kw in ('draw', 'no contest', 'nc')):
                is_draw = True

        fight_results.append({
            'fighter1': names[0],
            'fighter2': names[1],
            'winner': winner_name,
            'method': method,
            'round': round_num,
            'time': fight_time,
            'is_draw': is_draw,
        })

    # Fallback: if c-listing-fight didn't work, try athlete link pairing
    if not fight_results:
        fight_results = _scrape_results_fallback(soup)

    return fight_results


def _scrape_results_fallback(soup):
    """Fallback result scraper using athlete links and page text."""
    results = []
    page_text = soup.get_text()

    vs_matches = re.findall(
        r'([A-Z][a-zA-Z\'\-\. ]+?)\s+(?:def\.?|defeated|vs\.?)\s+([A-Z][a-zA-Z\'\-\. ]+?)(?:\s+via\s+(.+?))?(?:\n|$|\s{2,})',
        page_text
    )

    seen = set()
    for m in vs_matches:
        winner, loser = m[0].strip(), m[1].strip()
        method = m[2].strip() if len(m) > 2 else ''
        pair_key = tuple(sorted([winner, loser]))
        if pair_key not in seen and len(winner) > 3 and len(loser) > 3:
            seen.add(pair_key)
            results.append({
                'fighter1': winner,
                'fighter2': loser,
                'winner': winner,
                'method': method,
                'round': None,
                'time': '',
                'is_draw': False,
            })

    return results


def _match_name(name_a, name_b):
    """Check if two fighter names likely refer to the same person."""
    def _norm(name):
        # Convert accents/diacritics to plain ASCII for robust matching.
        base = unicodedata.normalize('NFKD', (name or ''))
        base = base.encode('ascii', 'ignore').decode('ascii')
        return re.sub(r'[^a-z\s]', '', base.lower()).strip()

    a = _norm(name_a)
    b = _norm(name_b)

    if a == b:
        return True

    # Last name match
    a_last = a.split()[-1] if a.split() else ''
    b_last = b.split()[-1] if b.split() else ''
    if a_last == b_last and len(a_last) >= 3:
        return True

    # One name contained in the other
    if a_last in b or b_last in a:
        if len(a_last) >= 4 or len(b_last) >= 4:
            return True

    return False


def _match_scraped_to_predictions(predictions, scraped_results):
    """
    Match scraped UFC.com results to stored predictions.

    Returns: dict mapping prediction_id -> result_dict
    """
    matched = {}

    for pred in predictions:
        for si, sr in enumerate(scraped_results):
            # Check if both fighters match (in either order)
            f1_match = (_match_name(pred['fighter1_name'], sr['fighter1']) or
                        _match_name(pred['fighter1_name'], sr['fighter2']))
            f2_match = (_match_name(pred['fighter2_name'], sr['fighter1']) or
                        _match_name(pred['fighter2_name'], sr['fighter2']))

            if f1_match and f2_match:
                winner_name = sr['winner']
                winner_id = None

                # Map winner back to our prediction's fighter IDs
                if winner_name:
                    if _match_name(winner_name, pred['fighter1_name']):
                        winner_name = pred['fighter1_name']
                        winner_id = pred['fighter1_id']
                    elif _match_name(winner_name, pred['fighter2_name']):
                        winner_name = pred['fighter2_name']
                        winner_id = pred['fighter2_id']

                matched[pred['id']] = {
                    'winner_name': winner_name,
                    'winner_id': winner_id,
                    'method': sr['method'],
                    'round': sr['round'],
                    'source': 'ufc.com',
                    'is_draw': sr.get('is_draw', False),
                }
                break

    return matched


# ============================================================================
# RESOLVE: UPDATE PREDICTIONS WITH ACTUAL RESULTS
# ============================================================================

def resolve_event(conn, search_term, event_url=None, force=False):
    """
    Main function: find predictions, fetch results, update, return report data.

    Args:
        conn: Database connection
        search_term: Event name to search for
        event_url: Optional UFC.com URL for scraping results
        force: Re-fetch results even if already resolved

    Returns:
        (event_name, predictions) -- predictions list updated with results
    """
    event_name, predictions = find_event_predictions(conn, search_term)

    if not event_name or not predictions:
        print(f"  No stored predictions found for: \"{search_term}\"")
        print("  Run predict_card first to generate and save predictions:")
        print(f'    python -m ml.predict_card "{search_term}"')
        return None, []

    # Skip already-resolved predictions unless force=True
    pending = predictions if force else [p for p in predictions if p['was_correct'] is None]

    if not pending and not force:
        print(f"  All {len(predictions)} predictions for \"{event_name}\" are already resolved.")
        print("  Use --force to re-fetch results.")
        return event_name, predictions

    print(f"  Fetching results for: {event_name}")
    print(f"  Found {len(predictions)} predictions ({len(pending)} to resolve)")

    # Step 1: Try the Fights table (most reliable)
    db_results = _fetch_results_from_db(conn, pending)

    if db_results:
        print(f"  Found {len(db_results)}/{len(pending)} results in database")

    # Step 2: Try UFC.com scrape (even when DB has matches), then prefer scrape.
    unresolved_ids = set(p['id'] for p in pending) - set(db_results.keys())
    scraped_results = {}

    if pending:
        # Determine the UFC.com URL to scrape
        scrape_url = event_url
        if not scrape_url:
            # Check if we have a stored event_url
            for p in predictions:
                if p.get('event_url'):
                    scrape_url = p['event_url']
                    break

        if not scrape_url:
            # Try to search UFC.com for the event
            try:
                from ml.predict_card import search_ufc_event
            except (ModuleNotFoundError, ImportError):
                try:
                    from predict_card import search_ufc_event
                except (ModuleNotFoundError, ImportError):
                    search_ufc_event = None

            if search_ufc_event:
                print(f"  Searching UFC.com for event...")
                scrape_url = search_ufc_event(search_term)

        if scrape_url:
            print(f"  Fetching: {scrape_url}")
            event_results = scrape_event_results(scrape_url)
            if event_results:
                print(f"  Fetched {len(event_results)} fight results")
                scraped_results = _match_scraped_to_predictions(
                    pending, event_results
                )
            else:
                print("  [WARNING] Could not parse results from UFC.com")
        else:
            print("  [WARNING] No UFC.com URL available for scraping")

    # Merge results
    all_results = {**db_results, **scraped_results}

    # Step 3: Update predictions in the database
    cur = conn.cursor()
    updated_count = 0

    for pred in pending:
        result = all_results.get(pred['id'])
        if not result:
            continue

        winner_name = result['winner_name']
        winner_id = result.get('winner_id')
        method = result.get('method', '')
        round_num = result.get('round')
        is_draw = result.get('is_draw', False)

        # Determine correctness
        if is_draw:
            was_correct = False
        elif winner_name:
            was_correct = _match_name(pred['pick_name'], winner_name)
        else:
            was_correct = None

        cur.execute("""
            UPDATE PredictionTracking SET
                actual_winner_name = %s,
                actual_winner_id = %s,
                actual_method = %s,
                actual_round = %s,
                was_correct = %s,
                resolved_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (winner_name, winner_id, method, round_num, was_correct, pred['id']))

        # Update the in-memory prediction dict too
        pred['actual_winner_name'] = winner_name
        pred['actual_winner_id'] = winner_id
        pred['actual_method'] = method
        pred['actual_round'] = round_num
        pred['was_correct'] = was_correct

        updated_count += 1

    conn.commit()
    cur.close()

    # Print per-fight comparison
    print()
    for pred in predictions:
        f1_last = pred['fighter1_name'].split()[-1]
        f2_last = pred['fighter2_name'].split()[-1]
        matchup = f"{f1_last} vs {f2_last}"
        pick_pct = float(pred['pick_probability']) * 100

        if pred['was_correct'] is None:
            icon = "  "
            actual = "PENDING"
        elif pred['was_correct']:
            icon = "\u2705"
            actual = pred['actual_winner_name']
        else:
            icon = "\u274c"
            actual = pred['actual_winner_name'] or "Draw/NC"

        print(f"  {icon} {matchup:<28} Predicted {pred['pick_name'].split()[-1]} ({pick_pct:.0f}%), Actual: {actual}")

    print(f"\n  Updated {updated_count}/{len(predictions)} predictions")

    # Refresh predictions from DB to get latest state
    _, predictions = find_event_predictions(conn, event_name)
    return event_name, predictions


# ============================================================================
# PERFORMANCE REPORT
# ============================================================================

def print_event_report(predictions, event_name):
    """Print detailed performance report for a single event."""
    W = 80

    resolved = [p for p in predictions if p['was_correct'] is not None]
    if not resolved:
        print(f"\n  No resolved predictions for {event_name}")
        return

    correct = sum(1 for p in resolved if p['was_correct'])
    accuracy = correct / len(resolved) * 100 if resolved else 0

    print("\n" + "=" * W)
    print(f"{'PERFORMANCE REPORT: ' + event_name:^{W}}")
    print("=" * W)

    print(f"\n  OVERALL:")
    print(f"    Total Predictions: {len(resolved)}")
    print(f"    Correct:           {correct}")
    print(f"    Accuracy:          {accuracy:.1f}%")

    # By confidence tier
    tiers = {'HIGH': [], 'MED': [], 'LOW': [], 'TOSS': []}
    for p in resolved:
        tier = p.get('confidence', 'TOSS')
        if tier in tiers:
            tiers[tier].append(p)

    print(f"\n  BY CONFIDENCE TIER:")
    tier_order = ['HIGH', 'MED', 'LOW', 'TOSS']
    for tier in tier_order:
        preds = tiers[tier]
        if not preds:
            continue
        tier_correct = sum(1 for p in preds if p['was_correct'])
        tier_acc = tier_correct / len(preds) * 100 if preds else 0

        if tier_acc >= 65:
            icon = "\u2705"
        elif tier_acc >= 50:
            icon = "\u26a0\ufe0f"
        else:
            icon = "\u274c"

        print(f"    {icon} {tier:<4} : {tier_correct}/{len(preds)} ({tier_acc:.0f}%)")

    # High confidence breakdown (>=65% probability)
    high_conf = [p for p in resolved if float(p['pick_probability']) >= 0.65]
    if high_conf:
        hc_correct = sum(1 for p in high_conf if p['was_correct'])
        hc_acc = hc_correct / len(high_conf) * 100

        status = "EXCELLENT" if hc_acc >= 70 else "GOOD" if hc_acc >= 60 else "NEEDS REVIEW"
        status_icon = "\u2705" if hc_acc >= 65 else "\u26a0\ufe0f" if hc_acc >= 50 else "\u274c"

        print(f"\n  HIGH CONFIDENCE (\u226565%):")
        print(f"    Total:    {len(high_conf)}")
        print(f"    Correct:  {hc_correct}")
        print(f"    Accuracy: {hc_acc:.1f}%")
        print(f"    Status:   {status_icon} {status}")

    # Method accuracy
    method_preds = [p for p in resolved if p.get('predicted_method') and p.get('actual_method')]
    if method_preds:
        method_correct = 0
        for p in method_preds:
            pred_m = p['predicted_method'].upper()
            actual_m = p['actual_method'].upper()
            # Flexible matching: KO/TKO matches KO, SUB matches Submission, etc.
            if (pred_m in actual_m) or (actual_m in pred_m):
                method_correct += 1
            elif ('DEC' in pred_m and 'DEC' in actual_m):
                method_correct += 1
            elif ('KO' in pred_m and 'KO' in actual_m) or ('TKO' in pred_m and 'TKO' in actual_m):
                method_correct += 1
            elif ('SUB' in pred_m and 'SUB' in actual_m):
                method_correct += 1

        m_acc = method_correct / len(method_preds) * 100 if method_preds else 0
        print(f"\n  METHOD ACCURACY:")
        print(f"    Correct method: {method_correct}/{len(method_preds)} ({m_acc:.0f}%)")

    # Biggest misses (wrong predictions with highest confidence)
    wrong = [p for p in resolved if not p['was_correct']]
    if wrong:
        wrong_sorted = sorted(wrong, key=lambda x: float(x['pick_probability']), reverse=True)
        print(f"\n  BIGGEST MISSES:")
        for p in wrong_sorted[:5]:
            f1_last = p['fighter1_name'].split()[-1]
            f2_last = p['fighter2_name'].split()[-1]
            pct = float(p['pick_probability']) * 100
            print(f"    {f1_last} vs {f2_last}: Picked {p['pick_name'].split()[-1]} ({pct:.0f}%), "
                  f"Won: {p['actual_winner_name'] or 'N/A'}")

    # Paper trading (hypothetical ROI)
    _print_paper_trading(resolved)

    print("\n" + "=" * W)


def _print_paper_trading(resolved):
    """Calculate hypothetical ROI for high-confidence picks."""
    STAKE = 100
    high_conf = [p for p in resolved if float(p['pick_probability']) >= 0.65]

    if not high_conf:
        return

    total_wagered = 0
    total_returned = 0

    for p in high_conf:
        prob = float(p['pick_probability'])
        total_wagered += STAKE

        if p['was_correct']:
            # Calculate payout at fair odds (no vig)
            decimal_odds = 1.0 / prob
            profit = STAKE * (decimal_odds - 1)
            total_returned += STAKE + profit

    net_profit = total_returned - total_wagered
    roi = net_profit / total_wagered * 100 if total_wagered > 0 else 0

    print(f"\n  PAPER TRADING (65%+ picks, ${STAKE} flat bets, fair odds):")
    print(f"    Bets placed:    {len(high_conf)}")
    print(f"    Total wagered:  ${total_wagered:.0f}")
    print(f"    Total returned: ${total_returned:.0f}")
    print(f"    Net profit:     ${net_profit:+.0f}")
    print(f"    ROI:            {roi:+.1f}%")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UFC Event Results Tracker - Update predictions with actual results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Update results after an event
  python -m ml.update_results "Strickland vs Hernandez"

  # Use a specific UFC.com URL for result scraping
  python -m ml.update_results "Strickland vs Hernandez" --url "https://www.ufc.com/event/..."

  # List all tracked events
  python -m ml.update_results --list

  # Re-fetch results even if already resolved
  python -m ml.update_results "Strickland vs Hernandez" --force
        '''
    )
    parser.add_argument('event', nargs='?', help='Event name to update (e.g. "Strickland vs Hernandez")')
    parser.add_argument('--url', '-u', type=str, help='UFC.com event URL for scraping results')
    parser.add_argument('--list', '-l', action='store_true', help='List all tracked events')
    parser.add_argument('--force', '-f', action='store_true', help='Re-fetch results even if resolved')

    args = parser.parse_args()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        sys.exit(1)

    try:
        # Mode: list tracked events
        if args.list:
            events = list_tracked_events(conn)
            if not events:
                print("\n  No tracked events found.")
                print("  Run predict_card first to generate predictions:")
                print('    python -m ml.predict_card "Event Name"')
                return

            W = 80
            print("\n" + "=" * W)
            print(f"{'TRACKED EVENTS':^{W}}")
            print("=" * W)
            print(f"\n  {'EVENT':<40} {'DATE':<12} {'PRED':>4} {'DONE':>4} {'ACC':>6}")
            print("  " + "-" * (W - 4))

            for e in events:
                acc = ''
                if e['resolved'] > 0:
                    a = e['correct'] / e['resolved'] * 100
                    acc = f"{a:.0f}%"

                print(f"  {e['event_name'][:39]:<40} {(e['event_date'] or '')[:11]:<12} "
                      f"{e['total']:>4} {e['resolved']:>4} {acc:>6}")

            print(f"\n  Total: {len(events)} events")
            print("=" * W)
            return

        # Mode: update results
        if args.event:
            event_name, predictions = resolve_event(
                conn, args.event,
                event_url=args.url,
                force=args.force,
            )

            if event_name and predictions:
                print_event_report(predictions, event_name)
            return

        parser.print_help()

    finally:
        conn.close()


if __name__ == '__main__':
    main()

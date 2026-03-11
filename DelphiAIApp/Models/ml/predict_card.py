"""
UFC Fight Card Predictor

Predict all fights on a UFC event card at once.

Supports:
  1. Past events already in the database (fuzzy search by event name)
  2. Upcoming events scraped live from UFC.com
  3. Manual fight list fallback

Usage:
    # Search by event name (DB first, then UFC.com)
    python -m ml.predict_card "Strickland vs Hernandez"
    python -m ml.predict_card "UFC 312"
    python -m ml.predict_card "UFC Fight Night February 22 2025"

    # Show detailed per-fight breakdowns (not just summary)
    python -m ml.predict_card "UFC 312" --detail

    # Scrape UFC.com for a specific event URL
    python -m ml.predict_card --url "https://www.ufc.com/event/ufc-fight-night-february-22-2025"

    # List upcoming events
    python -m ml.predict_card --upcoming
"""

import os
import sys
import re
import argparse
import time
import random
import logging
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for fighter names with accents (e.g. Uroš Medić)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import psycopg2
import requests
from bs4 import BeautifulSoup
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

# Logging
logger = logging.getLogger(__name__)

# Safe scraping settings
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]


# ============================================================================
# DATABASE LOOKUP - Find event and fights in existing DB
# ============================================================================

def find_event_in_db(conn, search_term):
    """
    Fuzzy-search for an event in the Fights table.
    
    Returns list of matching events: [{'name': str, 'date': date, 'fight_count': int}]
    """
    cur = conn.cursor()
    
    # Normalize search term for flexible matching
    # "Strickland vs Hernandez" -> search for both names in event name
    keywords = re.split(r'\s+(?:vs\.?|versus)\s+|\s+', search_term.strip())
    keywords = [k for k in keywords if len(k) > 1]  # Drop tiny words
    
    # Build ILIKE conditions - all keywords must appear in event name
    conditions = []
    params = []
    for kw in keywords:
        conditions.append("eventname ILIKE %s")
        params.append(f'%{kw}%')
    
    if not conditions:
        return []
    
    where = " AND ".join(conditions)
    
    cur.execute(f'''
        SELECT eventname, date, COUNT(*) as fight_count
        FROM fights
        WHERE {where} AND eventname IS NOT NULL
        GROUP BY eventname, date
        ORDER BY date DESC
        LIMIT 10
    ''', params)
    
    results = []
    for row in cur.fetchall():
        results.append({
            'name': row[0],
            'date': row[1],
            'fight_count': row[2],
        })
    
    cur.close()
    return results


def get_event_fights(conn, event_name):
    """
    Get all unique fight matchups for a given event.
    
    Returns list of dicts: [{'fighter1': str, 'fighter2': str, 'is_title': bool}]
    Ordered by main event first, then card order.
    """
    cur = conn.cursor()
    
    cur.execute('''
        SELECT fightername, opponentname, istitlefight, ismainevent
        FROM fights
        WHERE eventname = %s
        ORDER BY 
            ismainevent DESC NULLS LAST,
            istitlefight DESC NULLS LAST,
            fightid ASC
    ''', (event_name,))
    
    # Deduplicate: we may have A vs B and B vs A
    seen = set()
    fights = []
    for row in cur.fetchall():
        f1, f2 = row[0], row[1]
        pair_key = tuple(sorted([f1, f2]))
        if pair_key not in seen:
            seen.add(pair_key)
            fights.append({
                'fighter1': f1,
                'fighter2': f2,
                'is_title': bool(row[2]),
                'is_main': bool(row[3]) if row[3] is not None else False,
            })
    
    cur.close()
    return fights


# ============================================================================
# UFC.COM SCRAPER - Fetch upcoming event cards
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


def scrape_upcoming_events():
    """
    Scrape UFC.com for a list of upcoming events.
    
    Returns: [{'name': str, 'url': str, 'full_text': str}]
    
    The full_text field includes fighter names and card text,
    useful for keyword matching (e.g. "Strickland vs Hernandez").
    """
    session = _get_session()
    events = []
    
    try:
        resp = session.get('https://www.ufc.com/events', timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Step 1: Collect unique event URLs
        seen_urls = set()
        event_urls = []
        for link in soup.select('a[href*="/event/"]'):
            href = link.get('href', '')
            if '/event/' not in href:
                continue
            url = href if href.startswith('http') else f'https://www.ufc.com{href}'
            url = url.split('?')[0].split('#')[0].rstrip('/')
            if 'ufc.com' not in url or url in seen_urls:
                continue
            seen_urls.add(url)
            event_urls.append(url)
        
        # Step 2: Collect "vs" headings (fighter matchup names)
        vs_headings = []
        for heading in soup.select('h3, h4'):
            text = heading.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            if 'vs' in text.lower() and len(text) > 3:
                vs_headings.append(text)
        
        # Step 3: Get the full page text around each event for context
        full_page_text = soup.get_text(separator=' ')
        
        # Step 4: Build events list - for each URL, try to find a matching heading
        used_headings = set()
        for url in event_urls:
            slug = url.split('/event/')[-1]
            url_name = slug.replace('-', ' ').title()
            
            event_name = url_name
            full_text = url_name
            
            # Match heading to URL:
            # For numbered UFCs (ufc-326), look for headings near that number
            # For dated events, try to match by proximity (first unused heading = first URL)
            matched = False
            for i, heading in enumerate(vs_headings):
                if i in used_headings:
                    continue
                # Direct match: heading keywords appear in slug
                h_words = [w.lower() for w in heading.split() if len(w) > 2 and w.lower() != 'vs']
                if any(w in slug for w in h_words):
                    event_name = heading
                    full_text = heading
                    used_headings.add(i)
                    matched = True
                    break
            
            # If no direct match, assign next unused heading
            # (assumes upcoming headings and URLs appear in same order)
            if not matched:
                for i, heading in enumerate(vs_headings):
                    if i not in used_headings:
                        event_name = heading
                        full_text = heading
                        used_headings.add(i)
                        break
            
            events.append({
                'name': event_name,
                'url': url,
                'full_text': full_text,
            })
        
    except Exception as e:
        logger.warning(f"Failed to scrape UFC.com events: {e}")
    
    return events[:20]





def scrape_event_card(event_url):
    """
    Scrape a specific UFC.com event page for the fight card.
    
    Args:
        event_url: Full URL to the UFC.com event page
        
    Returns: {
        'event_name': str,
        'event_date': str,
        'location': str,
        'fights': [{'fighter1': str, 'fighter2': str, 'is_title': bool, 'weight_class': str}]
    }
    """
    session = _get_session()
    time.sleep(random.uniform(1, 3))
    
    try:
        resp = session.get(event_url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch {event_url}: {e}")
        return None
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Extract event name from page title
    event_name = ''
    title_tag = soup.find('title')
    if title_tag:
        # e.g. "UFC Fight Night: Strickland vs Hernandez | UFC"
        event_name = title_tag.get_text(strip=True).split('|')[0].strip()
    if not event_name:
        name_el = soup.select_one('h1, .c-hero--full__headline, .field--name-node-title')
        if name_el:
            event_name = name_el.get_text(separator=' ', strip=True)

    
    # Extract event date
    event_date = ''
    date_el = soup.select_one('.c-hero--full__date, .c-event-fight-card-broadcaster__time')
    if date_el:
        event_date = date_el.get_text(strip=True)
    
    # Extract location
    location = ''
    loc_el = soup.select_one('.field--name-venue, .c-hero--full__detail-item--airing')
    if loc_el:
        location = loc_el.get_text(strip=True)
    
    fights = []
    
    # ---- METHOD 1: Parse .c-listing-fight elements (standard UFC.com structure) ----
    fight_cards = soup.select('.c-listing-fight')
    for card in fight_cards:
        # Extract fighter names - try multiple strategies
        names = []
        
        # Strategy A: Get name from athlete link href (most reliable)
        # Links like /athlete/sean-strickland -> "Sean Strickland"
        athlete_links_in_card = card.select('a[href*="/athlete/"]')
        seen_in_card = set()
        for alink in athlete_links_in_card:
            ahref = alink.get('href', '')
            aslug = ahref.rstrip('/').split('/athlete/')[-1]
            aslug_clean = re.sub(r'-\d+$', '', aslug)
            if aslug_clean and aslug_clean not in seen_in_card:
                seen_in_card.add(aslug_clean)
                # Prefer text content if it has a space (properly formatted)
                atext = alink.get_text(separator=' ', strip=True)
                atext = re.sub(r'\s+', ' ', atext).strip()
                if ' ' in atext and 3 < len(atext) < 40:
                    names.append(atext)
                else:
                    names.append(aslug_clean.replace('-', ' ').title())
        
        # Strategy B: Fallback to corner name text with separator
        if len(names) < 2:
            name_els = card.select('.c-listing-fight__corner-name')
            for nel in name_els:
                name_text = nel.get_text(separator=' ', strip=True)
                name_text = re.sub(r'\s+', ' ', name_text).strip()
                if name_text and name_text not in names:
                    names.append(name_text)
        
        if len(names) >= 2:
            is_title = False
            belt_el = card.select_one('.c-listing-fight__belt-icon, .c-listing-fight__class-text')
            if belt_el and any(kw in belt_el.get_text(strip=True).lower() for kw in ('title', 'belt', 'championship')):
                is_title = True
            belt_img = card.select_one('img[src*="belt"], img[alt*="belt"]')
            if belt_img:
                is_title = True
            
            weight_class = ''
            wc_el = card.select_one('.c-listing-fight__class-text, .c-listing-fight__class')
            if wc_el:
                weight_class = wc_el.get_text(strip=True)
            
            fights.append({
                'fighter1': names[0],
                'fighter2': names[1],
                'is_title': is_title,
                'weight_class': weight_class,
            })

    
    # ---- METHOD 2: Parse athlete links (works on newer UFC.com layouts) ----
    # UFC.com event pages list fighters as links to /athlete/ pages.
    # We pair them up: every two consecutive athlete links = one fight.
    if not fights:
        athlete_links = soup.select('a[href*="/athlete/"]')
        
        # Deduplicate by URL slug and build name list.
        # UFC.com pages have 2+ links per fighter (hero image, card text, footer).
        # Text content is unreliable (concatenated spans like "SeanStrickland"),
        # so we ALWAYS prefer the URL slug for the fighter name.
        # e.g. /athlete/sean-strickland -> "Sean Strickland"
        seen_slugs = set()
        unique_athletes = []
        for link in athlete_links:
            href = link.get('href', '')
            if '/athlete/' not in href:
                continue
            
            slug = href.rstrip('/').split('/athlete/')[-1]
            # Remove trailing numbers (UFC uses them for duplicate URLs like "spivak-0")
            clean_slug = re.sub(r'-\d+$', '', slug)
            
            if clean_slug in seen_slugs or not clean_slug:
                continue
            seen_slugs.add(clean_slug)
            
            # Always derive name from URL slug (most reliable)
            name = clean_slug.replace('-', ' ').title()
            
            # Override with text content ONLY if it looks well-formatted
            # (has a space and is reasonable length)
            text = link.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            if ' ' in text and 3 < len(text) < 40:
                name = text
            
            if name and len(name) > 2:
                unique_athletes.append(name)




        
        # Pair consecutive athletes as fights
        for idx in range(0, len(unique_athletes) - 1, 2):
            f1 = unique_athletes[idx]
            f2 = unique_athletes[idx + 1]
            # Skip if names are too short or look like navigation
            if len(f1) > 2 and len(f2) > 2:
                fights.append({
                    'fighter1': f1,
                    'fighter2': f2,
                    'is_title': idx == 0,  # First fight on page is typically the main event
                    'weight_class': '',
                })
    
    # ---- METHOD 3: Regex fallback - extract "Name vs Name" patterns from text ----
    if not fights:
        page_text = soup.get_text()
        vs_matches = re.findall(
            r'([A-Z][a-zA-Z\'\-\. ]+?)\s+vs\.?\s+([A-Z][a-zA-Z\'\-\. ]+?)(?:\n|$|\s{2,})',
            page_text
        )
        seen_pairs = set()
        for m in vs_matches:
            f1, f2 = m[0].strip(), m[1].strip()
            pair_key = tuple(sorted([f1, f2]))
            if pair_key not in seen_pairs and len(f1) > 3 and len(f2) > 3:
                seen_pairs.add(pair_key)
                fights.append({
                    'fighter1': f1,
                    'fighter2': f2,
                    'is_title': len(fights) == 0,
                    'weight_class': '',
                })
    
    if not fights:
        print(f"[WARNING] Could not parse any fights from {event_url}")
        print("  The page structure may have changed. Try --manual mode.")
    
    return {
        'event_name': event_name or event_url.split('/event/')[-1].replace('-', ' ').title(),
        'event_date': event_date,
        'location': location,
        'fights': fights,
    }



def search_ufc_event(search_term):
    """
    Search UFC.com events for a matching upcoming event.
    
    Strategy:
    1. Fetch upcoming events list from UFC.com/events
    2. Match keywords against event names AND URLs
    3. Fall back to direct URL construction
    
    Returns: event URL string or None
    """
    # Extract meaningful keywords (skip "vs", "ufc", small words)
    skip_words = {'vs', 'vs.', 'versus', 'ufc', 'fight', 'night', 'the', 'a', 'an'}
    keywords = re.split(r'\s+', search_term.strip().lower())
    keywords = [k for k in keywords if k not in skip_words and len(k) > 1]
    
    # Also keep the full search terms for broader matching
    all_words = re.split(r'\s+', search_term.strip().lower())
    all_words = [w for w in all_words if len(w) > 1]
    
    # Try fetching upcoming events and matching
    upcoming = scrape_upcoming_events()
    
    best_match = None
    best_score = 0
    
    for event in upcoming:
        event_text = f"{event['name']} {event['url']} {event.get('full_text', '')}".lower()
        
        # Score: how many keywords match
        score = sum(1 for kw in keywords if kw in event_text)

        
        # Bonus for matching fighter-specific keywords (longer words)
        for kw in keywords:
            if len(kw) >= 4 and kw in event_text:
                score += 2  # Extra weight for fighter names
        
        if score > best_score:
            best_score = score
            best_match = event
    
    # Require at least 1 meaningful keyword match
    if best_match and best_score >= 1:
        return best_match['url']
    
    # Fallback: construct likely URLs from the search term
    slug = '-'.join(all_words)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    
    possible_urls = [
        f'https://www.ufc.com/event/{slug}',
        f'https://www.ufc.com/event/ufc-fight-night-{slug}',
        f'https://www.ufc.com/event/ufc-{slug}',
    ]
    
    session = _get_session()
    for possible_url in possible_urls:
        try:
            resp = session.head(possible_url, timeout=10, allow_redirects=True)
            if resp.status_code == 200 and '/event/' in resp.url:
                return resp.url
        except Exception:
            pass
    
    return None




# ============================================================================
# CARD PREDICTION ENGINE
# ============================================================================

def _normalize_name(name):
    """Strip accents/diacritics for matching: Quiñonez -> Quinonez, Medić -> Medic."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', name)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def _fuzzy_fighter_lookup(conn, get_fighter_data_fn, last_name, first_initial='', full_name=''):
    """
    Smart fallback lookup when full name doesn't match DB.
    
    Tries progressively broader searches:
      1. Accent-normalized full name
      2. "% LastName" - matches "Zach Reese" but NOT "Tom Breese"
      3. "FirstInitial% LastName" - "Z% Reese" -> "Zach Reese"
      4. Accent-normalized last name variants
      5. Just last name (broadest) - validates first name initial matches
    
    This prevents e.g. "Cristian Quiñonez" matching "Jose Quinonez".
    """
    norm_last = _normalize_name(last_name)
    norm_full = _normalize_name(full_name) if full_name else ''

    # Try 1: Accent-normalized full name (Cristian Quinonez -> finds Cristian Quiñonez)
    if norm_full and norm_full != full_name:
        result = get_fighter_data_fn(conn, norm_full)
        if result:
            return result

    # Try 2: "% LastName" with original spelling
    result = get_fighter_data_fn(conn, f"% {last_name}")
    if result and _validate_first_initial(result, first_initial):
        return result
    
    # Try 3: "% LastName" with normalized spelling (ñ -> n, ć -> c)
    if norm_last != last_name:
        result = get_fighter_data_fn(conn, f"% {norm_last}")
        if result and _validate_first_initial(result, first_initial):
            return result
    
    # Try 4: First initial + wildcard + last name
    if first_initial:
        result = get_fighter_data_fn(conn, f"{first_initial}% {last_name}")
        if result:
            return result
        if norm_last != last_name:
            result = get_fighter_data_fn(conn, f"{first_initial}% {norm_last}")
            if result:
                return result
    
    # Try 5: Just last name (broadest) - but MUST validate first initial
    result = get_fighter_data_fn(conn, last_name)
    if result and _validate_first_initial(result, first_initial):
        return result
    if norm_last != last_name:
        result = get_fighter_data_fn(conn, norm_last)
        if result and _validate_first_initial(result, first_initial):
            return result

    return None


def _validate_first_initial(fighter_data, first_initial):
    """Verify the matched fighter's first name starts with the expected initial."""
    if not first_initial or not fighter_data:
        return True
    matched_name = fighter_data.get('name', '')
    if not matched_name:
        return True
    matched_first = _normalize_name(matched_name.split()[0]) if matched_name.split() else ''
    return matched_first and matched_first[0].upper() == first_initial.upper()



def _needs_injury_check(fighter_data, cache_days=7):
    """Return True when cached injury data is missing or stale."""
    if not fighter_data:
        return False
    last_check = fighter_data.get('last_injury_check')
    if not last_check:
        return True

    if isinstance(last_check, str):
        try:
            last_check = datetime.fromisoformat(last_check)
        except Exception:
            return True
    elif hasattr(last_check, 'year') and not isinstance(last_check, datetime):
        last_check = datetime.combine(last_check, datetime.min.time())

    if isinstance(last_check, datetime):
        return (datetime.now() - last_check).days > cache_days
    return True


def _skip_live_scrape(fighter_data):

    """
    Modify fighter data dict to prevent live injury scraping.
    Sets last_injury_check to today so the cache appears fresh.
    Use for card mode where we don't want 24 individual scrapes.
    """
    fd = dict(fighter_data)
    fd['last_injury_check'] = datetime.now()


    return fd


def predict_full_card(conn, fights, event_name='', event_date='', 
                      detail=False, force_refresh=False, event_url=''):
    """
    Run predictions for all fights on a card.
    
    Args:
        conn: Database connection
        fights: List of {'fighter1': str, 'fighter2': str, 'is_title': bool}
        event_name: Event name for display
        event_date: Event date for display
        detail: If True, show full analysis for each fight
        force_refresh: Force injury data refresh
        event_url: UFC.com event URL (stored for later result scraping)
    """

    # Import prediction functions
    try:
        from ml.predict_fight import (
            get_fighter_data, calculate_elo_adjustments, elo_to_probability,
            analyze_matchup, _get_ml_predictor, predict_method,
            recalibrate_probabilities,
            predict_sig_strikes, predict_takedowns
        )
    except ModuleNotFoundError:
        from predict_fight import (
            get_fighter_data, calculate_elo_adjustments, elo_to_probability,
            analyze_matchup, _get_ml_predictor, predict_method,
            recalibrate_probabilities,
            predict_sig_strikes, predict_takedowns
        )

    
    W = 95

    
    # Header
    print("\n" + "=" * W)
    print(f"{'DELPHI AI - FIGHT CARD PREDICTIONS':^{W}}")
    print("=" * W)
    if event_name:
        print(f"{'Event: ' + event_name:^{W}}")
    if event_date:
        print(f"{'Date: ' + str(event_date):^{W}}")
    print(f"{'Fights: ' + str(len(fights)):^{W}}")
    print("=" * W)
    
    # Collect results for summary
    results = []
    not_found = []
    
    for i, fight in enumerate(fights, 1):
        f1_name = fight['fighter1']
        f2_name = fight['fighter2']
        is_title = fight.get('is_title', False)
        
        # Look up fighters in DB (with fallback for name variations)
        print(f"  Loading fight {i}/{len(fights)}: {f1_name} vs {f2_name}...", end='', flush=True)
        f1 = get_fighter_data(conn, f1_name)
        f2 = get_fighter_data(conn, f2_name)
        
        # Retry with smarter fallbacks if full name not found.
        # Handles URL-based names like "Zachary Reese" when DB has "Zach Reese".
        # Strategy:
        #   1. Try "% LastName" (word boundary) - matches "Zach Reese" not "Tom Breese"
        #   2. Try first initial + last name - "Z% Reese"
        if not f1 and ' ' in f1_name:
            parts = f1_name.split()
            last = parts[-1]
            first_initial = parts[0][0] if parts[0] else ''
            f1 = _fuzzy_fighter_lookup(conn, get_fighter_data, last, first_initial, full_name=f1_name)
        if not f2 and ' ' in f2_name:
            parts = f2_name.split()
            last = parts[-1]
            first_initial = parts[0][0] if parts[0] else ''
            f2 = _fuzzy_fighter_lookup(conn, get_fighter_data, last, first_initial, full_name=f2_name)



        
        if not f1 or not f2:
            missing = []
            if not f1:
                missing.append(f1_name)
            if not f2:
                missing.append(f2_name)
            not_found.append({
                'fight_num': i,
                'fighter1': f1_name,
                'fighter2': f2_name,
                'missing': missing,
                'is_title': is_title,
            })
            print(f" MISSING: {', '.join(missing)}")
            continue
        
        print(" OK")
        
        # Check if injury scraping is needed for either fighter
        # (live scrapes UFC.com - takes ~3-5s per fighter with rate limiting)
        f1_needs_check = _needs_injury_check(f1)
        f2_needs_check = _needs_injury_check(f2)
        if f1_needs_check or f2_needs_check:
            names_checking = []
            if f1_needs_check:
                names_checking.append(f1['name'].split()[-1])
            if f2_needs_check:
                names_checking.append(f2['name'].split()[-1])
            print(f"    Checking injuries: {', '.join(names_checking)}...", end='', flush=True)
            # calculate_elo_adjustments will handle the scraping below
            print(" done")
        
        # Mark title fight

        if is_title:
            f1['is_title_fight'] = True
            f2['is_title_fight'] = True
        
        if detail:
            # Full detailed analysis for each fight
            label = "TITLE FIGHT" if is_title else f"FIGHT {i}"
            print(f"\n{'':>2}[{label}]")
            analyze_matchup(f1, f2, force_injury_refresh=force_refresh, conn=conn)

        
        # Always compute summary data
        f1_adj = calculate_elo_adjustments(f1, force_refresh)
        f2_adj = calculate_elo_adjustments(f2, force_refresh)
        
        elo_prob_f1 = elo_to_probability(f1_adj['adjusted_elo'], f2_adj['adjusted_elo'])
        
        # ML prediction with smart blending (same logic as analyze_matchup)
        ml_predictor = _get_ml_predictor()
        ml_result = None
        ml_used = False
        prob_source = "ELO"
        
        if ml_predictor and ml_predictor.is_available():
            f1_for_ml = dict(f1)
            f2_for_ml = dict(f2)
            f1_for_ml['final_elo'] = f1_adj['raw_elo']
            f2_for_ml['final_elo'] = f2_adj['raw_elo']
            try:
                ml_result = ml_predictor.predict(f1_for_ml, f2_for_ml, is_title_fight=is_title)
            except Exception:
                ml_result = None
        
        adj_prob_f1 = elo_prob_f1
        adj_prob_f2 = 1 - elo_prob_f1
        
        if ml_result:
            import math
            features = ml_result.get('features', {})
            nan_count = sum(1 for v in features.values()
                           if v is None or (isinstance(v, float) and math.isnan(v)))
            
            if nan_count <= 6:
                # Symmetry check: predict with swapped fighters
                f2_for_swap = dict(f1)
                f1_for_swap = dict(f2)
                f1_for_swap['final_elo'] = f2_adj['raw_elo']
                f2_for_swap['final_elo'] = f1_adj['raw_elo']
                try:
                    swap_result = ml_predictor.predict(f1_for_swap, f2_for_swap, is_title_fight=is_title)
                    swap_prob = swap_result['prob_a'] if swap_result else None
                except Exception:
                    swap_result = None
                    swap_prob = None
                
                if swap_prob is not None:
                    # If calibrated probs are saturated by clipping, use raw probs for symmetry.
                    fwd_prob = ml_result.get('raw_prob_a', ml_result['prob_a'])
                    rev_prob = swap_result.get('raw_prob_a', swap_prob) if swap_result else swap_prob
                    sym_error = abs(fwd_prob + rev_prob - 1.0)
                    corrected = (fwd_prob + (1.0 - rev_prob)) / 2.0
                    
                    # Graduated blending: allow moderate symmetry drift with lower ML weight.
                    if sym_error <= 0.60:
                        # Start from feature-quality weight, then discount for symmetry error.
                        ml_weight = 0.50 - nan_count * 0.08 - max(0.0, sym_error - 0.20) * 0.60
                        ml_weight = max(0.15, min(0.50, ml_weight))
                        blended = ml_weight * corrected + (1 - ml_weight) * elo_prob_f1
                        
                        # Injury/ring-rust shifts
                        f1_pen = f1_adj['inactivity_penalty'] + f1_adj['injury_penalty']
                        f2_pen = f2_adj['inactivity_penalty'] + f2_adj['injury_penalty']
                        f1_shift = min(f1_pen / 400 * 0.5, 0.10)
                        f2_shift = min(f2_pen / 400 * 0.5, 0.10)
                        blended = blended - f1_shift + f2_shift
                        blended = max(0.10, min(0.90, blended))
                        
                        adj_prob_f1 = blended
                        adj_prob_f2 = 1 - blended
                        ml_used = True
                        prob_source = f"ML+ELO ({ml_weight:.0%} ML)"
        
        if not ml_used:
            adj_prob_f1 = elo_to_probability(f1_adj['adjusted_elo'], f2_adj['adjusted_elo'])
            adj_prob_f2 = 1 - adj_prob_f1
            prob_source = "ELO"

        # Apply post-blend recalibration to counter confidence compression.
        adj_prob_f1 = recalibrate_probabilities(adj_prob_f1)
        adj_prob_f2 = 1 - adj_prob_f1



        
        # Determine pick
        if adj_prob_f1 > adj_prob_f2:
            pick, pick_pct = f1['name'], adj_prob_f1
            underdog, dog_pct = f2['name'], adj_prob_f2
        else:
            pick, pick_pct = f2['name'], adj_prob_f2
            underdog, dog_pct = f1['name'], adj_prob_f1
        
        # Confidence
        if pick_pct >= 0.70:
            confidence = "HIGH"
        elif pick_pct >= 0.60:
            confidence = "MED"
        elif pick_pct >= 0.55:
            confidence = "LOW"
        else:
            confidence = "TOSS"
        
        # Method prediction
        method = predict_method(f1, f2, adj_prob_f1, adj_prob_f2)
        all_methods = {'KO/TKO': method['ko'], 'Sub': method['sub'], 'Dec': method['dec']}
        best_method = max(all_methods, key=all_methods.get)
        
        # Notes (injuries/ring rust)
        notes = []
        if f1_adj['injury_penalty'] > 0:
            notes.append(f"{f1['name'].split()[-1]} inj")
        if f2_adj['injury_penalty'] > 0:
            notes.append(f"{f2['name'].split()[-1]} inj")
        if f1_adj['inactivity_penalty'] > 0:
            notes.append(f"{f1['name'].split()[-1]} rust")
        if f2_adj['inactivity_penalty'] > 0:
            notes.append(f"{f2['name'].split()[-1]} rust")
        
        results.append({
            'fight_num': i,
            'fighter1': f1['name'],
            'fighter2': f2['name'],
            'f1_id': f1.get('id'),
            'f2_id': f2.get('id'),

            'f1_record': f"{f1['wins']}-{f1['losses']}-{f1['draws']}",
            'f2_record': f"{f2['wins']}-{f2['losses']}-{f2['draws']}",
            'f1_elo': f1_adj['raw_elo'],
            'f2_elo': f2_adj['raw_elo'],
            'f1_adj_elo': f1_adj['adjusted_elo'],
            'f2_adj_elo': f2_adj['adjusted_elo'],
            'f1_prob': adj_prob_f1,
            'f2_prob': adj_prob_f2,
            'pick': pick,
            'pick_pct': pick_pct,
            'underdog': underdog,
            'dog_pct': dog_pct,
            'confidence': confidence,
            'method': best_method,
            'method_pct': all_methods[best_method],
            'prob_source': prob_source,
            'is_title': is_title,
            'notes': notes,
            # Method breakdown (overall fight outcome probabilities)
            'ko_pct': method['ko'],
            'sub_pct': method['sub'],
            'dec_pct': method['dec'],
            'goes_distance': method['goes_distance'],
            # Per-fighter method distribution (IF that fighter wins)
            'f1_method_dist': method['f1_method_dist'],
            'f2_method_dist': method['f2_method_dist'],
            # Round probabilities
            'r1_finish': method['r1_finish'],
            'r2_finish': method['r2_finish'],
            'r3_finish': method['r3_finish'],
            # Injury/rust penalties for display
            'f1_inj': f1_adj['injury_penalty'],
            'f2_inj': f2_adj['injury_penalty'],
            'f1_rust': f1_adj['inactivity_penalty'],
            'f2_rust': f2_adj['inactivity_penalty'],
        })

    
    # ======================== SUMMARY CARD ========================
    print("\n" + "=" * W)
    print(f"{'FIGHT CARD SUMMARY':^{W}}")
    print("=" * W)
    
    # Column headers
    header = f"  {'#':>2}  {'MATCHUP':<36} {'PICK':<16} {'%':>5} {'CONF':>5} {'METHOD':>7}"
    print(header)
    print("  " + "-" * (W - 4))
    
    for r in results:
        # Build matchup string
        matchup = f"{r['fighter1'].split()[-1]} vs {r['fighter2'].split()[-1]}"
        if r['is_title']:
            matchup = f"* {matchup}"  # * marker for title fights

        
        pick_last = r['pick']
        if len(pick_last) > 15:
            pick_last = r['pick'].split()[-1]
            if len(pick_last) > 15:
                pick_last = pick_last[:15]
        
        print(f"  {r['fight_num']:>2}  {matchup:<36} {pick_last:<16} {r['pick_pct']:>4.0%}  {r['confidence']:>5}  {r['method']:>7}")
    
    # Show not-found fights
    if not_found:
        print()
        for nf in not_found:
            matchup = f"{nf['fighter1'].split()[-1]} vs {nf['fighter2'].split()[-1]}"
            missing_str = ", ".join(nf['missing'])
            print(f"  {nf['fight_num']:>2}  {matchup:<36} {'NOT IN DB':<16} {'':>5} {'':>5}  {missing_str}")
    
    # ======================== DETAILED PER-FIGHT RESULTS ========================
    if not detail:
        print("\n" + "-" * W)
        print(f"{'FIGHT DETAILS':^{W}}")
        print("-" * W)
        
        for r in results:
            title_tag = " [TITLE]" if r['is_title'] else ""
            note_str = f"  [{', '.join(r['notes'])}]" if r['notes'] else ""
            
            print(f"\n  {r['fight_num']}. {r['fighter1']} vs {r['fighter2']}{title_tag}")
            
            # Fighter lines with ELO adjustments
            f1_elo_str = f"{r['f1_elo']:>5.0f}"
            f2_elo_str = f"{r['f2_elo']:>5.0f}"
            if r['f1_adj_elo'] != r['f1_elo']:
                f1_elo_str += f" -> {r['f1_adj_elo']:.0f}"
            if r['f2_adj_elo'] != r['f2_elo']:
                f2_elo_str += f" -> {r['f2_adj_elo']:.0f}"
            
            print(f"     {r['fighter1'][:20]:<20} ({r['f1_record']})  ELO {f1_elo_str}  {r['f1_prob']:>4.0%}")
            print(f"     {r['fighter2'][:20]:<20} ({r['f2_record']})  ELO {f2_elo_str}  {r['f2_prob']:>4.0%}")
            print(f"     PICK: {r['pick']} ({r['pick_pct']:.0%}) via {r['method']} ({r['method_pct']:.0%})")
            
            # Method of Victory breakdown
            print(f"     Method:  KO/TKO {r['ko_pct']:>4.0%}  |  Sub {r['sub_pct']:>4.0%}  |  Dec {r['dec_pct']:>4.0%}")
            
            # Round probabilities
            finish_total = r['r1_finish'] + r['r2_finish'] + r['r3_finish']
            print(f"     Rounds:  R1 Finish {r['r1_finish']:>4.0%}  |  R2 {r['r2_finish']:>4.0%}  |  R3 {r['r3_finish']:>4.0%}  |  Dec {r['dec_pct']:>4.0%}")
            
            # Per-fighter method distribution (IF they win)
            f1_md = r['f1_method_dist']
            f2_md = r['f2_method_dist']
            f1_short = r['fighter1'].split()[-1][:12]
            f2_short = r['fighter2'].split()[-1][:12]
            print(f"     If {f1_short:<12} wins:  KO {f1_md['ko']:>4.0%}  Sub {f1_md['sub']:>4.0%}  Dec {f1_md['dec']:>4.0%}")
            print(f"     If {f2_short:<12} wins:  KO {f2_md['ko']:>4.0%}  Sub {f2_md['sub']:>4.0%}  Dec {f2_md['dec']:>4.0%}")
            
            print(f"     Source: {r['prob_source']}{note_str}")

    
    # ======================== BETTING INSIGHTS ========================
    print("\n" + "-" * W)
    print(f"{'BETTING INSIGHTS':^{W}}")
    print("-" * W)
    
    # Best bets (highest confidence picks)
    high_conf = [r for r in results if r['confidence'] in ('HIGH', 'MED')]
    if high_conf:
        print("\n  STRONGEST PICKS:")
        for r in sorted(high_conf, key=lambda x: x['pick_pct'], reverse=True):
            # Convert probability to American odds
            odds = prob_to_american(r['pick_pct'])
            print(f"    {r['pick']:<25} {r['pick_pct']:>4.0%} (implied {odds})")
    
    # Upset candidates (close fights)
    toss_ups = [r for r in results if r['confidence'] in ('TOSS', 'LOW')]
    if toss_ups:
        print("\n  UPSET WATCH:")
        for r in toss_ups:
            dog_odds = prob_to_american(r['dog_pct'])
            print(f"    {r['underdog']:<25} {r['dog_pct']:>4.0%} (implied {dog_odds})")
    
    # Finish vs Decision summary
    finishes = [r for r in results if r['goes_distance'] < 0.40]
    decisions = [r for r in results if r['goes_distance'] >= 0.55]
    if finishes:
        print(f"\n  LIKELY FINISHES ({len(finishes)}):")
        for r in finishes:
            # Show the most likely finish method (KO or Sub, not Dec)
            finish_method = r['method'] if r['method'] != 'Dec' else 'KO/TKO or Sub'
            print(f"    {r['fighter1'].split()[-1]} vs {r['fighter2'].split()[-1]}: "
                  f"{1 - r['goes_distance']:.0%} finish chance via {finish_method}")

    if decisions:
        print(f"\n  LIKELY DECISIONS ({len(decisions)}):")
        for r in decisions:
            print(f"    {r['fighter1'].split()[-1]} vs {r['fighter2'].split()[-1]}: "
                  f"{r['goes_distance']:.0%} decision chance")
    
    # Count summary
    high_ct = sum(1 for r in results if r['confidence'] == 'HIGH')
    med_ct = sum(1 for r in results if r['confidence'] == 'MED')
    low_ct = sum(1 for r in results if r['confidence'] == 'LOW')
    toss_ct = sum(1 for r in results if r['confidence'] == 'TOSS')
    ml_ct = sum(1 for r in results if r['prob_source'].startswith('ML'))

    
    print(f"\n  Confidence: {high_ct} HIGH | {med_ct} MED | {low_ct} LOW | {toss_ct} TOSS")
    print(f"  ML model used: {ml_ct}/{len(results)} fights")
    if not_found:
        print(f"  Not found in DB: {len(not_found)} fights (fighters may not have UFC stats)")
    
    # Footer
    print("\n" + "=" * W)
    print(f"  Odds: -200=67% | -150=60% | +150=40% | +200=33%")
    print(f"  Disclaimer: Predictions are for analysis only. Bet responsibly.")
    print("=" * W)
    
    # Auto-save predictions to database for tracking
    if results:
        try:
            try:
                from ml.update_results import save_predictions
            except (ModuleNotFoundError, ImportError):
                from update_results import save_predictions
            
            saved = save_predictions(conn, results, event_name, event_date, event_url)
            print(f"\n  Predictions saved to database ({saved} fights)")
            print(f"  After event: python -m ml.update_results \"{event_name}\"")
        except Exception as e:
            logger.warning(f"Could not save predictions: {e}")
            print(f"\n  [NOTE] Could not save predictions for tracking: {e}")
    
    return results



def prob_to_american(prob):
    """Convert probability (0-1) to American odds string."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        return f"-{int(prob / (1 - prob) * 100)}"
    else:
        return f"+{int((1 - prob) / prob * 100)}"


# ============================================================================
# CLI MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UFC Fight Card Predictor - Analyze entire UFC event cards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Search by event name (DB first, then UFC.com scrape)
  python -m ml.predict_card "Strickland vs Hernandez"
  python -m ml.predict_card "UFC 312"
  
  # Show full detailed breakdowns per fight  
  python -m ml.predict_card "UFC 312" --detail
  
  # Scrape from a specific UFC.com URL
  python -m ml.predict_card --url "https://www.ufc.com/event/ufc-fight-night-february-22-2025"
  
  # List upcoming UFC events
  python -m ml.predict_card --upcoming
  
  # Manual fight list (when event isn't in DB or on UFC.com)
  python -m ml.predict_card --manual "Sean Strickland,Anthony Hernandez" "Zhang Weili,Tatiana Suarez"
        '''
    )
    parser.add_argument('event', nargs='?', help='Event name to search (e.g. "UFC 312", "Strickland vs Hernandez")')
    parser.add_argument('--url', '-u', type=str, help='UFC.com event URL to scrape')
    parser.add_argument('--upcoming', action='store_true', help='List upcoming UFC events')
    parser.add_argument('--detail', '-d', action='store_true', help='Show full analysis for each fight')
    parser.add_argument('--manual', '-m', nargs='+', metavar='FIGHT',
                        help='Manual fights: "Fighter1,Fighter2" "Fighter3,Fighter4"')
    parser.add_argument('--refresh', '-r', action='store_true', help='Force fresh injury check')
    
    args = parser.parse_args()
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        print("Make sure PostgreSQL is running and .env is configured.")
        sys.exit(1)
    
    try:
        # Mode: List upcoming events
        if args.upcoming:
            print("\n  Fetching upcoming UFC events from UFC.com...")
            events = scrape_upcoming_events()
            if events:
                print(f"\n  Found {len(events)} upcoming events:\n")
                for i, evt in enumerate(events, 1):
                    print(f"    {i}. {evt['name']}")
                    print(f"       {evt['url']}")
                print(f"\n  Use: python -m ml.predict_card --url <URL>")
            else:
                print("  No upcoming events found. UFC.com may have changed structure.")
            return
        
        # Mode: Scrape specific URL
        if args.url:
            print(f"\n  Scraping event card from: {args.url}")
            event_data = scrape_event_card(args.url)
            if event_data and event_data['fights']:
                fights = event_data['fights']
                predict_full_card(
                    conn, fights,
                    event_name=event_data['event_name'],
                    event_date=event_data['event_date'],
                    detail=args.detail,
                    force_refresh=args.refresh,
                    event_url=args.url,
                )

            else:
                print("[ERROR] No fights found at that URL.")
                print("  Try --manual mode to enter fights manually.")
            return
        
        # Mode: Manual fight list
        if args.manual:
            fights = []
            for fight_str in args.manual:
                if ',' in fight_str:
                    parts = fight_str.split(',', 1)
                    fights.append({
                        'fighter1': parts[0].strip(),
                        'fighter2': parts[1].strip(),
                        'is_title': False,
                    })
                else:
                    print(f"[WARNING] Skipping invalid format: '{fight_str}'. Use 'Fighter1,Fighter2'")
            
            if fights:
                predict_full_card(
                    conn, fights,
                    event_name='Manual Card',
                    detail=args.detail,
                    force_refresh=args.refresh,
                )
            return
        
        # Mode: Search by event name
        if args.event:
            search = args.event
            print(f"\n  Searching for: \"{search}\"")
            
            # Step 1: Check database
            db_events = find_event_in_db(conn, search)
            
            if db_events:
                # If multiple matches, show options
                if len(db_events) > 1:
                    print(f"\n  Found {len(db_events)} matching events:\n")
                    for i, evt in enumerate(db_events, 1):
                        print(f"    {i}. {evt['name']}  ({evt['date']})  [{evt['fight_count']} fights]")
                    
                    # Use the most recent one
                    chosen = db_events[0]
                    print(f"\n  Using most recent: {chosen['name']}")
                else:
                    chosen = db_events[0]
                    print(f"  Found: {chosen['name']}  ({chosen['date']})")
                
                # Get fights for this event
                fights = get_event_fights(conn, chosen['name'])
                if fights:
                    predict_full_card(
                        conn, fights,
                        event_name=chosen['name'],
                        event_date=str(chosen['date']),
                        detail=args.detail,
                        force_refresh=args.refresh,
                    )
                else:
                    print(f"[ERROR] No fights found for event: {chosen['name']}")
                return
            
            # Step 2: Not in DB - try UFC.com
            print("  Not found in database. Searching UFC.com...")
            event_url = search_ufc_event(search)
            
            if event_url:
                print(f"  Found: {event_url}")
                event_data = scrape_event_card(event_url)
                if event_data and event_data['fights']:
                    predict_full_card(
                        conn, event_data['fights'],
                        event_name=event_data['event_name'],
                        event_date=event_data['event_date'],
                        detail=args.detail,
                        force_refresh=args.refresh,
                        event_url=event_url,
                    )
                    return

            
            # Step 3: Not found anywhere
            print(f"\n  [NOT FOUND] Could not find event matching: \"{search}\"")
            print("  Try:")
            print("    - A more specific name: \"UFC 312\" or \"UFC Fight Night February 22 2025\"")
            print("    - The UFC.com URL: --url https://www.ufc.com/event/...")
            print("    - Manual mode: --manual \"Fighter1,Fighter2\" \"Fighter3,Fighter4\"")
            return
        
        # No arguments - show help
        parser.print_help()
    
    finally:
        conn.close()


if __name__ == '__main__':
    main()


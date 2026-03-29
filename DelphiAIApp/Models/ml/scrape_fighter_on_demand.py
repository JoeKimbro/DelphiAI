"""
On-demand fighter scraper for predict_card.py.

When a fighter is not found in the database during card prediction, this module
scrapes their bio and career stats from UFCStats (primary) and UFC.com (fallback/
supplement) and inserts them into the DB so the fight can still be predicted.

Source priority (mirrors the full pipeline in ufc_official.py):
  UFCStats  → career stats (SLpM, StrAcc, …), DOB, record, height/weight/reach
  UFC.com   → nickname, weight class, place of birth, leg reach, stance,
               avg fight duration, is_active
  Either    → fallback bio fields if the other source is missing
"""

import re
import time
import random
import logging
import unicodedata
from datetime import datetime, date

import requests
from bs4 import BeautifulSoup
import psycopg2

logger = logging.getLogger(__name__)

UFCSTATS_LISTING = "http://www.ufcstats.com/statistics/fighters?char={char}&page=all"
UFC_COM_ATHLETE   = "https://www.ufc.com/athlete/{slug}"

REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

_last_request_time = 0.0

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    delay = random.uniform(1.0, 2.5)
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_request_time = time.time()


def _fetch(url, allow_404=False):
    """Fetch a URL with retries. Returns (BeautifulSoup, status_code) or (None, code)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            _rate_limit()
            resp = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"Rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404 and allow_404:
                return None, 404
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                return None, resp.status_code
            return BeautifulSoup(resp.text, 'html.parser'), 200
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None, 0


def _clean_text(text):
    if text:
        return ' '.join(str(text).split()).strip()
    return None


def _normalize_name(name):
    """Lowercase + strip diacritics for matching."""
    if not name:
        return ''
    # Full unicode normalization (handles all diacritics)
    n = unicodedata.normalize('NFKD', name)
    n = ''.join(c for c in n if not unicodedata.combining(c))
    return n.lower().strip()


def _parse_record(record_text):
    if not record_text:
        return None, None, None
    m = re.search(r'(\d+)-(\d+)-(\d+)', record_text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def _parse_decimal(value):
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_percentage(value):
    """Parse '47%' → 47.0 (stored as DECIMAL in DB, not as string)."""
    if not value:
        return None
    try:
        return float(value.replace('%', '').strip())
    except ValueError:
        return None


def _parse_date(date_str):
    """Try several date formats, return datetime.date or None."""
    if not date_str or date_str == '--':
        return None
    for fmt in ('%b. %d, %Y', '%b %d, %Y', '%B %d, %Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# UFCStats scraping
# ---------------------------------------------------------------------------

def _find_ufcstats_url(fighter_name):
    """
    Search UFCStats listing page (indexed by last-name initial) for the fighter.
    Returns their detail URL or None.
    """
    parts = fighter_name.strip().split()
    if not parts:
        return None

    last_name = parts[-1]
    norm_last = _normalize_name(last_name)
    char = norm_last[0] if norm_last and norm_last[0].isalpha() else None
    if not char:
        return None

    soup, _ = _fetch(UFCSTATS_LISTING.format(char=char))
    if not soup:
        return None

    norm_target = _normalize_name(fighter_name)
    norm_target_last = norm_last
    first_initial = _normalize_name(parts[0])[0] if parts[0] else ''

    candidates = []
    for row in soup.select('table.b-statistics__table tbody tr'):
        cols = row.select('td.b-statistics__table-col')
        if len(cols) < 2:
            continue
        first_link = cols[0].select_one('a')
        last_link  = cols[1].select_one('a')
        if not first_link or not last_link:
            continue
        first = _clean_text(first_link.get_text()) or ''
        last  = _clean_text(last_link.get_text()) or ''
        detail_url = first_link.get('href', '')
        if not detail_url or 'fighter-details' not in detail_url:
            continue

        norm_full = _normalize_name(f"{first} {last}")
        # Tier 1: exact normalized full name
        if norm_full == norm_target:
            return detail_url
        # Tier 2: last name + first initial (collect unambiguous)
        if _normalize_name(last) == norm_target_last:
            norm_first = _normalize_name(first)
            if norm_first and norm_first[0] == first_initial:
                candidates.append(detail_url)

    return candidates[0] if len(candidates) == 1 else None


def _parse_ufcstats_page(soup):
    """
    Parse a UFCStats fighter detail page.
    Returns a dict; values may be None.
    """
    data = {}

    name_el = soup.select_one('span.b-content__title-highlight')
    data['name'] = _clean_text(name_el.get_text()) if name_el else None

    record_el = soup.select_one('span.b-content__title-record')
    wins, losses, draws = _parse_record(_clean_text(record_el.get_text()) if record_el else None)
    data['wins']   = wins
    data['losses'] = losses
    data['draws']  = draws

    # Bio (Height, Weight, Reach, Stance, DOB)
    for item in soup.select('ul.b-list__box-list li.b-list__box-list-item'):
        label_el = item.select_one('i.b-list__box-item-title')
        label = _clean_text(label_el.get_text()) if label_el else ''
        texts = [t for t in item.stripped_strings]
        value = _clean_text(texts[-1]) if texts else None
        ll = label.lower().replace(':', '').strip()
        if 'height' in ll:
            data['height'] = value
        elif 'weight' in ll:
            data['weight'] = value
        elif 'reach' in ll:
            data['reach']  = value
        elif 'stance' in ll:
            data['stance'] = value
        elif 'dob' in ll:
            data['dob']    = value

    # Career stats
    for item in soup.select('div.b-list__info-box-left ul.b-list__box-list li.b-list__box-list-item'):
        label_el = item.select_one('i.b-list__box-item-title')
        label = _clean_text(label_el.get_text()) if label_el else ''
        texts = [t for t in item.stripped_strings]
        value = _clean_text(texts[-1]) if texts else None
        ll = label.lower().replace(':', '').replace('.', '').strip()
        if 'slpm' in ll:
            data['slpm']    = _parse_decimal(value)
        elif 'str acc' in ll:
            data['str_acc'] = _parse_percentage(value)
        elif 'sapm' in ll:
            data['sapm']    = _parse_decimal(value)
        elif 'str def' in ll:
            data['str_def'] = _parse_percentage(value)
        elif 'td avg' in ll:
            data['td_avg']  = _parse_decimal(value)
        elif 'td acc' in ll:
            data['td_acc']  = _parse_percentage(value)
        elif 'td def' in ll:
            data['td_def']  = _parse_percentage(value)
        elif 'sub avg' in ll or 'sub. avg' in ll:
            data['sub_avg'] = _parse_decimal(value)

    # Last fight date from fight history
    most_recent = None
    for row in soup.select('table.b-fight-details__table tbody tr.b-fight-details__table-row'):
        if row.select('th'):
            continue
        cells = row.select('td.b-fight-details__table-col')
        if len(cells) >= 7:
            date_texts = cells[6].select('p.b-fight-details__table-text')
            if date_texts:
                fight_date = _parse_date(_clean_text(date_texts[-1].get_text()))
                if fight_date and (most_recent is None or fight_date > most_recent):
                    most_recent = fight_date

    if most_recent:
        data['last_fight_date']      = most_recent
        data['days_since_last_fight'] = (date.today() - most_recent).days
        data['is_active']             = data['days_since_last_fight'] <= (2 * 365)
    else:
        data['last_fight_date']      = None
        data['days_since_last_fight'] = None
        data['is_active']             = False

    return data


def _scrape_ufcstats(fighter_name):
    """Try UFCStats. Returns (detail_url, data_dict) or (None, None)."""
    detail_url = _find_ufcstats_url(fighter_name)
    if not detail_url:
        return None, None
    soup, _ = _fetch(detail_url)
    if not soup:
        return None, None
    data = _parse_ufcstats_page(soup)
    if not data.get('name'):
        return None, None
    return detail_url, data


# ---------------------------------------------------------------------------
# UFC.com scraping
# ---------------------------------------------------------------------------

def _ufc_com_slug(name):
    """Convert a fighter name to a UFC.com URL slug."""
    n = unicodedata.normalize('NFKD', name)
    n = ''.join(c for c in n if not unicodedata.combining(c))
    slug = '-'.join(n.lower().split())
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    return slug


def _find_ufc_com_url(fighter_name):
    """
    Try to find the fighter's UFC.com athlete page.
    Tries the direct slug, then drops middle names, then first-initial-last.
    Returns (soup, ufc_url) or (None, None).
    """
    parts = fighter_name.strip().split()

    slug_variants = []
    # Full name slug
    slug_variants.append(_ufc_com_slug(fighter_name))
    # Without middle name(s): first + last
    if len(parts) >= 3:
        slug_variants.append(_ufc_com_slug(f"{parts[0]} {parts[-1]}"))
    # First initial + last
    if len(parts) >= 2:
        slug_variants.append(_ufc_com_slug(f"{parts[0][0]} {parts[-1]}"))

    seen = set()
    for slug in slug_variants:
        if not slug or slug in seen:
            continue
        seen.add(slug)
        url = UFC_COM_ATHLETE.format(slug=slug)
        soup, status = _fetch(url, allow_404=True)
        if status == 200 and soup:
            # Verify this is actually a fighter page (not a generic error page)
            if soup.select_one('h1.hero-profile__name, .hero-profile__name'):
                return soup, url
        elif status == 404:
            continue  # Try next variant

    return None, None


def _parse_ufc_com_page(soup, ufc_url):
    """
    Parse a UFC.com athlete profile page.
    Returns a dict with UFC.com-specific fields; values may be None.
    """
    data = {'ufc_url': ufc_url}

    # Name
    name_el = (soup.select_one('h1.hero-profile__name')
                or soup.select_one('.hero-profile__name'))
    data['name'] = _clean_text(name_el.get_text()) if name_el else None

    # Nickname
    nick_el = soup.select_one('.hero-profile__nickname')
    if nick_el:
        data['nickname'] = _clean_text(nick_el.get_text()).strip('"') or None

    # Weight class
    wc_el = (soup.select_one('.hero-profile__division-title')
              or soup.select_one('.hero-profile__division'))
    if wc_el:
        data['weight_class'] = _clean_text(wc_el.get_text())

    # Record
    rec_el = soup.select_one('.hero-profile__division-body')
    if rec_el:
        wins, losses, draws = _parse_record(_clean_text(rec_el.get_text()))
        data['wins']   = wins
        data['losses'] = losses
        data['draws']  = draws

    # Is active (from status tag)
    status_el = soup.select_one('.hero-profile__tag')
    if status_el:
        data['is_active'] = 'active' in _clean_text(status_el.get_text()).lower()

    # Bio fields from .c-bio__field elements
    for field in soup.select('.c-bio__field'):
        label_el = field.select_one('.c-bio__label')
        value_el = field.select_one('.c-bio__text')
        label = _clean_text(label_el.get_text()) if label_el else None
        value = _clean_text(value_el.get_text()) if value_el else None
        if not label or not value:
            continue
        _apply_ufc_bio_field(data, label, value)

    # Regex fallback for bio fields not found via CSS
    page_text = soup.get_text(' ', strip=True)
    _regex_fill_bio(data, page_text)

    # Average fight time (from c-stat-compare section or regex)
    avg_time = _parse_avg_fight_time(str(soup))
    if avg_time is not None:
        data['avg_fight_duration'] = avg_time

    return data


def _apply_ufc_bio_field(data, label, value):
    ll = label.lower()
    if ('place of birth' in ll or 'hometown' in ll) and not data.get('place_of_birth'):
        data['place_of_birth'] = value
    elif 'age' in ll and 'average' not in ll and not data.get('age'):
        try:
            data['age'] = int(value.split()[0])
        except (ValueError, IndexError):
            pass
    elif 'leg reach' in ll and not data.get('leg_reach'):
        data['leg_reach'] = value
    elif 'reach' in ll and not data.get('reach'):
        data['reach'] = value
    elif 'height' in ll and not data.get('height'):
        data['height'] = value
    elif 'weight' in ll and not data.get('weight'):
        data['weight'] = value
    elif ('fighting style' in ll or 'style' in ll) and not data.get('stance'):
        data['stance'] = value
    elif 'status' in ll and data.get('is_active') is None:
        data['is_active'] = value.lower() == 'active'


def _regex_fill_bio(data, text):
    """Fill missing bio fields using regex on raw page text."""
    patterns = [
        (r'Place of Birth\s*[:\-]?\s*([^\n<,]{3,50})', 'place_of_birth'),
        (r'(?<![A-Za-z])Age\s*[:\-]?\s*(\d{1,2})',    'age_str'),
        (r'Leg\s*[Rr]each\s*[:\-]?\s*([\d.]+)',        'leg_reach'),
        (r'Fighting [Ss]tyle\s*[:\-]?\s*([^\n<]{3,30})', 'stance'),
        (r'Status\s*[:\-]?\s*(Active|Inactive)',        'status'),
    ]
    for pattern, key in patterns:
        if data.get(key if key != 'age_str' else 'age'):
            continue
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = re.sub(r'<[^>]+>', '', m.group(1)).strip()
            if not val:
                continue
            if key == 'age_str':
                try:
                    data['age'] = int(val)
                except ValueError:
                    pass
            elif key == 'status':
                data['is_active'] = val.lower() == 'active'
            else:
                data[key] = val


def _parse_avg_fight_time(html_text):
    """Extract average fight duration (minutes as float) from UFC.com HTML."""
    patterns = [
        r'c-stat-compare__number[^>]*>\s*(\d{1,2}:\d{2})\s*</[^>]+>\s*<[^>]*c-stat-compare__label[^>]*>\s*Average\s*fight\s*time',
        r'(\d{1,2}:\d{2})\s*</[^>]+>\s*<[^>]+>\s*Average\s*fight\s*time',
        r'Average\s*fight\s*time\s*[:\-]?\s*(\d{1,2}:\d{2})',
    ]
    for pattern in patterns:
        m = re.search(pattern, html_text, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                mins, secs = map(int, m.group(1).split(':'))
                return round(mins + secs / 60, 2)
            except (ValueError, AttributeError):
                continue
    return None


def _scrape_ufc_com(fighter_name):
    """Try UFC.com. Returns (ufc_url, data_dict) or (None, None)."""
    soup, ufc_url = _find_ufc_com_url(fighter_name)
    if not soup:
        return None, None
    data = _parse_ufc_com_page(soup, ufc_url)
    if not data.get('name'):
        return None, None
    return ufc_url, data


# ---------------------------------------------------------------------------
# Data merge
# ---------------------------------------------------------------------------

def _merge(ufcstats_data, ufc_data):
    """
    Merge UFCStats + UFC.com data.
    UFCStats is primary for career stats, DOB, record, and physical measurements.
    UFC.com fills in nickname, weight_class, place_of_birth, leg_reach, stance,
    avg_fight_duration, and is_active.
    """
    merged = {}

    # Start with UFCStats as base (primary)
    if ufcstats_data:
        merged.update(ufcstats_data)

    if ufc_data:
        # UFC.com-exclusive fields (always use if available)
        for key in ('nickname', 'weight_class', 'place_of_birth', 'leg_reach', 'ufc_url', 'avg_fight_duration'):
            if ufc_data.get(key) is not None:
                merged[key] = ufc_data[key]

        # Stance: UFC.com "fighting style" is more reliable than UFCStats
        if ufc_data.get('stance') and not merged.get('stance'):
            merged['stance'] = ufc_data['stance']

        # is_active: UFC.com is authoritative
        if ufc_data.get('is_active') is not None:
            merged['is_active'] = ufc_data['is_active']

        # Age: use UFC.com if UFCStats DOB didn't provide it
        if ufc_data.get('age') and not merged.get('age'):
            merged['age'] = ufc_data['age']

        # Physical measurements: fill gaps only
        for key in ('height', 'weight', 'reach'):
            if not merged.get(key) and ufc_data.get(key):
                merged[key] = ufc_data[key]

        # Record: fill gaps only (UFCStats is more accurate)
        if not ufcstats_data:
            for key in ('wins', 'losses', 'draws', 'name'):
                if ufc_data.get(key) is not None:
                    merged[key] = ufc_data[key]

    return merged


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------

def _insert_fighter(conn, fighter_url, scraped):
    """
    Insert FighterStats + CareerStats rows.
    Returns the existing or new FighterID.
    """
    cur = conn.cursor()

    # Avoid duplicates
    cur.execute("SELECT FighterID FROM FighterStats WHERE FighterURL = %s", (fighter_url,))
    existing = cur.fetchone()
    if existing:
        return existing[0]

    name = scraped.get('name')
    if not name:
        return None

    wins   = scraped.get('wins') or 0
    losses = scraped.get('losses') or 0
    draws  = scraped.get('draws') or 0

    dob_str  = scraped.get('dob')
    dob_date = _parse_date(dob_str)
    age = scraped.get('age')
    if age is None and dob_date:
        age = (date.today() - dob_date).days // 365

    now = datetime.utcnow()

    cur.execute("""
        INSERT INTO FighterStats (
            Name, FighterURL, UFCUrl,
            Height, Weight, Reach, LegReach, Stance, DOB, Age,
            WeightClass, Nickname, PlaceOfBirth,
            TotalFights, Wins, Losses, Draws,
            LastFightDate, DaysSinceLastFight, IsActive,
            Source, ScrapedAt, FightUpdatedAt
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        RETURNING FighterID
    """, (
        name, fighter_url, scraped.get('ufc_url'),
        scraped.get('height'), scraped.get('weight'),
        scraped.get('reach'), scraped.get('leg_reach'),
        scraped.get('stance'), dob_date, age,
        scraped.get('weight_class'), scraped.get('nickname'), scraped.get('place_of_birth'),
        wins + losses + draws, wins, losses, draws,
        scraped.get('last_fight_date'), scraped.get('days_since_last_fight'),
        scraped.get('is_active', False),
        'on_demand', now, now,
    ))
    fighter_id = cur.fetchone()[0]

    cur.execute("""
        INSERT INTO CareerStats (
            FighterID, FighterURL,
            SLpM, StrAcc, SApM, StrDef,
            TDAvg, TDAcc, TDDef, SubAvg,
            AvgFightDuration,
            EloRating, Source, ScrapedAt, CareerUpdatedAt
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (FighterID) DO UPDATE SET
            SLpM             = EXCLUDED.SLpM,
            StrAcc           = EXCLUDED.StrAcc,
            SApM             = EXCLUDED.SApM,
            StrDef           = EXCLUDED.StrDef,
            TDAvg            = EXCLUDED.TDAvg,
            TDAcc            = EXCLUDED.TDAcc,
            TDDef            = EXCLUDED.TDDef,
            SubAvg           = EXCLUDED.SubAvg,
            AvgFightDuration = EXCLUDED.AvgFightDuration,
            CareerUpdatedAt  = EXCLUDED.CareerUpdatedAt
    """, (
        fighter_id, fighter_url,
        scraped.get('slpm'), scraped.get('str_acc'),
        scraped.get('sapm'), scraped.get('str_def'),
        scraped.get('td_avg'), scraped.get('td_acc'),
        scraped.get('td_def'), scraped.get('sub_avg'),
        scraped.get('avg_fight_duration'),
        1500.0, 'on_demand', now, now,
    ))

    conn.commit()
    return fighter_id


# ---------------------------------------------------------------------------
# Return dict builder
# ---------------------------------------------------------------------------

def _build_fighter_dict(fighter_id, scraped):
    """
    Build a dict matching the format returned by get_fighter_data() in predict_fight.py.
    """
    nan = float('nan')

    slpm    = scraped.get('slpm')
    str_acc = scraped.get('str_acc')
    sapm    = scraped.get('sapm')
    str_def = scraped.get('str_def')
    td_avg  = scraped.get('td_avg')
    td_acc  = scraped.get('td_acc')
    td_def  = scraped.get('td_def')
    sub_avg = scraped.get('sub_avg')

    dob_date = _parse_date(scraped.get('dob')) if isinstance(scraped.get('dob'), str) else scraped.get('dob')
    age = scraped.get('age')
    if age is None and dob_date:
        age = float((date.today() - dob_date).days // 365)

    avg_fight_dur = scraped.get('avg_fight_duration') or 12.0

    return {
        'id':              fighter_id,
        'name':            scraped.get('name'),
        'nickname':        scraped.get('nickname'),
        'height':          scraped.get('height'),
        'weight':          scraped.get('weight'),
        'reach':           scraped.get('reach'),
        'stance':          scraped.get('stance'),
        'wins':            scraped.get('wins') or 0,
        'losses':          scraped.get('losses') or 0,
        'draws':           scraped.get('draws') or 0,
        'place_of_birth':  scraped.get('place_of_birth'),
        # Career stats — same fallback pattern as get_fighter_data()
        'slpm':    float(slpm)    if slpm    else 0,
        'str_acc': str_acc or '0%',
        'sapm':    float(sapm)    if sapm    else 0,
        'str_def': str_def or '0%',
        'td_avg':  float(td_avg)  if td_avg  else 0,
        'td_acc':  td_acc  or '0%',
        'td_def':  td_def  or '0%',
        'sub_avg': float(sub_avg) if sub_avg else 0,
        # Advanced stats not computable without fight history → defaults
        'ko_last5':               0,
        'sub_last5':              0,
        'decision_rate':          50.0,
        'avg_fight_duration':     avg_fight_dur,
        'first_round_finish_rate': 30.0,
        'ko_r1_pct': 40.0, 'ko_r2_pct': 30.0, 'ko_r3_pct': 30.0,
        'sub_r1_pct': 50.0, 'sub_r2_pct': 30.0, 'sub_r3_pct': 20.0,
        # ELO — default for new/unknown fighters
        'elo': 1500.0,
        # Recency
        'days_since_last_fight': scraped.get('days_since_last_fight') or 0,
        'last_fight':            scraped.get('last_fight_date'),
        # Injury (not scraped in this path)
        'last_injury_check': None,
        'injury_details':    None,
        # Historical features — not available for debut fighters
        'avg_opponent_elo_last_3':   nan,
        'elo_velocity':              nan,
        'current_win_streak':        0,
        'finish_rate_trending':      nan,
        'opponent_quality_trending': nan,
        # Demographics
        'dob': dob_date,
        'age': float(age) if age is not None else nan,
        # Live computed
        'finish_rate':  nan,
        'recent_form':  nan,
        # PIT fields — not available for debut fighters
        'kd_rate':             nan,
        'pit_slpm':            nan,
        'pit_stracc':          nan,
        'pit_tdavg':           nan,
        'pit_subavg':          nan,
        'pit_kdrate':          nan,
        'pit_recent_win_rate': nan,
        'pit_avg_fight_time':  nan,
        'pit_finish_rate':     nan,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_and_add_fighter(fighter_name, conn):
    """
    Scrape a fighter from UFCStats (primary) and UFC.com (supplement/fallback),
    insert into the DB, and return a fighter dict compatible with get_fighter_data().

    Strategy:
      1. Try UFCStats for career stats + bio
      2. Try UFC.com for nickname, weight class, place of birth, avg fight time, etc.
      3. Merge both sources (UFCStats wins on career stats; UFC.com wins on profile fields)
      4. If UFCStats fails but UFC.com succeeds, use UFC.com data alone (no career stats)
      5. If both fail, return None

    Returns None on any unrecoverable error; DB is rolled back cleanly.
    """
    try:
        # --- UFCStats ---
        ufcstats_url, ufcstats_data = _scrape_ufcstats(fighter_name)
        if ufcstats_data:
            logger.debug(f"[on-demand] Found '{fighter_name}' on UFCStats")
        else:
            logger.debug(f"[on-demand] '{fighter_name}' not found on UFCStats")

        # --- UFC.com ---
        ufc_com_url, ufc_com_data = _scrape_ufc_com(fighter_name)
        if ufc_com_data:
            logger.debug(f"[on-demand] Found '{fighter_name}' on UFC.com")
        else:
            logger.debug(f"[on-demand] '{fighter_name}' not found on UFC.com")

        # Need at least one source
        if not ufcstats_data and not ufc_com_data:
            return None

        # Merge — UFCStats URL is the canonical DB key when available
        merged = _merge(ufcstats_data, ufc_com_data)
        fighter_url = ufcstats_url or ufc_com_url

        if not merged.get('name'):
            return None

        # Insert into DB
        try:
            fighter_id = _insert_fighter(conn, fighter_url, merged)
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"[on-demand] DB error inserting '{fighter_name}': {e}")
            return None

        if fighter_id is None:
            return None

        merged['fighter_url'] = fighter_url
        return _build_fighter_dict(fighter_id, merged)

    except Exception as e:
        logger.error(f"[on-demand] Unexpected error for '{fighter_name}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None

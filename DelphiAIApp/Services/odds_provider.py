"""
In-process odds lookup. Two tiers:

  - Upcoming events: scrape bestfightodds.com once per TTL (15 min) and build
    a global { (fighter1_normalized, fighter2_normalized): {...} } index
    spanning every upcoming fight on the page (across promotions). Fast
    lookup, single HTTP request, no per-event URL mapping needed.

  - Past events: lazy-load the historical_odds.json produced by the
    BestFightOdds Scrapy spider; build the same name-pair-keyed index.

Both tiers return rows shaped:

    {
      'american_a': int,
      'american_b': int,
      'decimal_a': float,
      'decimal_b': float,
      'source': 'bestfightodds' | 'historical_archive',
    }

UFC scraper sometimes carries diacritics (`Uroš Medić`); BFO often does not.
We normalize names (lowercase, strip diacritics, collapse whitespace) on both
sides before keying.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from DelphiAIApp.Services.odds_math import american_to_decimal

logger = logging.getLogger(__name__)

_BFO_URL = "https://www.bestfightodds.com/"
_LIVE_TTL_SEC = 900.0  # 15 min
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

_HISTORICAL_PATH = (
    Path(__file__).resolve().parent.parent
    / "Models" / "data" / "scrapers" / "output" / "historical_odds.json"
)

# Module-level caches
_LIVE_CACHE: dict[str, object] = {"index": None, "ts": 0.0}
_HISTORICAL_INDEX: dict[tuple[str, str], dict] | None = None


def _normalize_name(name: str | None) -> str:
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", n.strip().lower())


def _parse_american(text: str | None) -> int | None:
    """Parse a string like '+150' / '-200' (with possible trailing arrow chars) into int."""
    if not text:
        return None
    m = re.match(r"^([+\-]\d{2,4})", text.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def get_event_odds(
    event_url: str | None, event_name: str | None, is_past: bool
) -> dict[tuple[str, str], dict]:
    """
    Return name-pair-keyed odds for every fight on this event card.

    The keys cover BOTH orderings (`(f1, f2)` and `(f2, f1)`) so callers can
    look up without caring which fighter was assigned f1 vs f2.
    """
    if is_past:
        return _historical_index()
    return _live_index()


# ----------------------------------------------------------------------------
# Live BestFightOdds scrape
# ----------------------------------------------------------------------------

def _live_index() -> dict[tuple[str, str], dict]:
    now = time.monotonic()
    cached_index = _LIVE_CACHE.get("index")
    if cached_index is not None and (now - float(_LIVE_CACHE["ts"])) < _LIVE_TTL_SEC:
        return cached_index  # type: ignore[return-value]

    try:
        index = _scrape_bestfightodds_homepage()
    except Exception as e:
        logger.warning("BestFightOdds scrape failed: %s", e)
        # Serve stale rather than break the UI — but only if we have a prior result.
        if cached_index is not None:
            return cached_index  # type: ignore[return-value]
        index = {}

    _LIVE_CACHE["index"] = index
    _LIVE_CACHE["ts"] = now
    return index


def _scrape_bestfightodds_homepage() -> dict[tuple[str, str], dict]:
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(_BFO_URL, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    index: dict[tuple[str, str], dict] = {}

    for event_div in soup.select("div.table-div"):
        body_table = event_div.select_one(".table-scroller table.odds-table")
        if body_table is None:
            continue

        # Pair successive fighter rows. Each fight = 2 consecutive rows that
        # contain `/fighters/...` anchors. Prop-bet rows in between have no
        # fighter anchor, so they're naturally skipped.
        pending: list[tuple[str, int | None]] = []
        for tr in body_table.find_all("tr"):
            anchor = tr.find("a", href=lambda h: bool(h) and "/fighters/" in h)
            if not anchor:
                continue
            name = anchor.get_text(strip=True)
            # First numeric td after the name cell is the headline American line.
            american = None
            for td in tr.find_all("td"):
                parsed = _parse_american(td.get_text(strip=True))
                if parsed is not None:
                    american = parsed
                    break
            pending.append((name, american))

            if len(pending) == 2:
                (n_a, am_a), (n_b, am_b) = pending
                pending = []
                if am_a is None or am_b is None:
                    continue
                try:
                    dec_a = american_to_decimal(am_a)
                    dec_b = american_to_decimal(am_b)
                except ValueError:
                    continue
                row = {
                    "american_a": am_a,
                    "american_b": am_b,
                    "decimal_a": dec_a,
                    "decimal_b": dec_b,
                    "source": "bestfightodds",
                }
                k_a = _normalize_name(n_a)
                k_b = _normalize_name(n_b)
                index[(k_a, k_b)] = row
                # Mirror under swapped key with a/b flipped so callers don't need
                # to care about ordering.
                index[(k_b, k_a)] = {
                    **row,
                    "american_a": am_b,
                    "american_b": am_a,
                    "decimal_a": dec_b,
                    "decimal_b": dec_a,
                }

    logger.info("BestFightOdds live index built: %d fight entries", len(index) // 2)
    return index


# ----------------------------------------------------------------------------
# Historical archive lookup
# ----------------------------------------------------------------------------

def _historical_index() -> dict[tuple[str, str], dict]:
    global _HISTORICAL_INDEX
    if _HISTORICAL_INDEX is not None:
        return _HISTORICAL_INDEX

    if not _HISTORICAL_PATH.exists():
        logger.warning(
            "historical_odds.json missing at %s; past-event odds will be unavailable",
            _HISTORICAL_PATH,
        )
        _HISTORICAL_INDEX = {}
        return _HISTORICAL_INDEX

    try:
        with _HISTORICAL_PATH.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception as e:
        logger.warning("Failed to load historical_odds.json: %s", e)
        _HISTORICAL_INDEX = {}
        return _HISTORICAL_INDEX

    index: dict[tuple[str, str], dict] = {}
    for r in rows:
        n_a = r.get("fighter1")
        n_b = r.get("fighter2")
        dec_a = r.get("fighter1_avg") or r.get("fighter1_best")
        dec_b = r.get("fighter2_avg") or r.get("fighter2_best")
        if not n_a or not n_b or not dec_a or not dec_b:
            continue
        try:
            from DelphiAIApp.Services.odds_math import decimal_to_american
            am_a = decimal_to_american(float(dec_a))
            am_b = decimal_to_american(float(dec_b))
        except (ValueError, TypeError):
            continue
        row = {
            "american_a": am_a,
            "american_b": am_b,
            "decimal_a": float(dec_a),
            "decimal_b": float(dec_b),
            "source": "historical_archive",
        }
        k_a = _normalize_name(n_a)
        k_b = _normalize_name(n_b)
        index[(k_a, k_b)] = row
        index[(k_b, k_a)] = {
            **row,
            "american_a": am_b,
            "american_b": am_a,
            "decimal_a": float(dec_b),
            "decimal_b": float(dec_a),
        }

    logger.info("Historical odds index built: %d fight entries", len(index) // 2)
    _HISTORICAL_INDEX = index
    return _HISTORICAL_INDEX

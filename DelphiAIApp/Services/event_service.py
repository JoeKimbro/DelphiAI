"""
Event service — bridges FastAPI to the ml/predict_card and ml/update_results pipelines.

Event IDs in this API are the UFC.com URL slug (e.g. "ufc-326-pereira-vs-ankalaev-2").
We round-trip slug ↔ URL via the UFC_EVENT_BASE prefix.
"""
from __future__ import annotations

import io
import time
import contextlib
from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from DelphiAIApp.Models.db.postgres import get_db_connection, execute_query

# UFC.com server-renders event times in US Eastern (the JS on the page then
# converts to the viewer's locale). When we scrape with `requests`, JS doesn't
# run, so the text we capture is always ET. We localize it here and emit a
# tz-aware ISO so the frontend can format it for any zone unambiguously.
_UFC_SOURCE_TZ = ZoneInfo("America/New_York")


def _event_date_to_iso(date_str: str | None, anchor: datetime | None) -> str | None:
    """
    Convert a display-format event date like "Sat, May 16 / 8:00 PM" to an
    ISO 8601 string. When a time is present, returns a tz-aware ISO with
    the event's ET offset ("2026-05-16T20:00:00-04:00"); otherwise date-only
    ("2026-05-16"). The string has no year, so we infer it from `anchor`
    (e.g. `predicted_at`, written within days of the event).

    Without this, the frontend does `new Date("Sat, May 16 / 8:00 PM")`,
    which V8 silently defaults to year 2001.
    """
    if not date_str or anchor is None:
        return None

    segments = [p.strip() for p in date_str.split('/')]
    date_part = segments[0]                                  # 'Sat, May 16'
    time_part = segments[1] if len(segments) > 1 else None   # '8:00 PM' or None
    if ',' in date_part:
        date_part = date_part.split(',', 1)[1].strip()       # 'May 16'

    for y_offset in (0, -1, 1):
        try:
            dt = datetime.strptime(f"{date_part} {anchor.year + y_offset}", "%b %d %Y")
        except ValueError:
            continue
        if abs((dt - anchor).days) >= 200:
            continue
        if time_part:
            try:
                t = datetime.strptime(time_part, "%I:%M %p")
                dt = dt.replace(hour=t.hour, minute=t.minute, tzinfo=_UFC_SOURCE_TZ)
                return dt.isoformat(timespec='seconds')
            except ValueError:
                pass
        return dt.strftime("%Y-%m-%d")
    return None

UFC_EVENT_BASE = "https://www.ufc.com/event/"

# In-process TTL cache for the UFC.com /events scrape. Single uvicorn worker
# assumption; expires every UPCOMING_TTL_SEC so a newly added card propagates
# within ~2 minutes without an explicit invalidation hook.
_UPCOMING_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}
_UPCOMING_TTL_SEC = 120.0


def _slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/event/")[-1]


def _url_from_slug(slug: str) -> str:
    return f"{UFC_EVENT_BASE}{slug.strip('/')}"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _slug_to_brand(slug: str) -> str:
    """Render a URL slug as a human brand name (e.g. 'ufc-freedom-250' → 'UFC Freedom 250')."""
    parts = slug.split("-")
    out = []
    for p in parts:
        if not p:
            continue
        if p.lower() == "ufc":
            out.append("UFC")
        elif p.isdigit():
            out.append(p)
        else:
            out.append(p.capitalize())
    return " ".join(out)


def _format_event_name(slug: str, heading: str) -> str:
    """Build a user-friendly event name from the slug and the per-card headline.

    The scraper now pulls a per-card headline directly. For most events the
    headline is the main-event matchup ("Allen vs Costa"); for branded cards
    UFC sometimes uses the brand itself ("UFC Freedom 250") with no matchup
    label. This helper:
      - drops "TBD vs TBD" placeholders in favour of a slug-derived label
      - prepends the slug brand to matchup headings on branded slugs so the
        user can tell which card it is (e.g. UFC 328: Chimaev vs Strickland)
      - never double-brands when the headline already starts with "UFC"
    """
    heading = (heading or "").strip()
    is_fight_night = slug.startswith("ufc-fight-night-")
    is_tbd = bool(heading) and "tbd" in heading.lower()

    if not heading or is_tbd:
        return _slug_to_brand(slug)

    if is_fight_night:
        return heading

    # Branded slug. Avoid double-branding if the headline already says UFC.
    if heading.lower().startswith("ufc"):
        return heading
    return f"{_slug_to_brand(slug)}: {heading}"


def _resolved_event_slugs(conn) -> set[str]:
    """Slugs of every event with at least one resolved row in PredictionTracking."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT event_url
        FROM PredictionTracking
        WHERE was_correct IS NOT NULL AND event_url IS NOT NULL AND event_url <> ''
        """
    )
    slugs = {_slug_from_url(row[0]) for row in cur.fetchall() if row[0]}
    cur.close()
    return slugs


def list_upcoming_events() -> list[dict]:
    """Scrape UFC.com for upcoming events, filtering out any that are already resolved.

    Cached in-process for `_UPCOMING_TTL_SEC` seconds. The UFC.com index updates
    at most a few times per day, and our prefetch job already populates the
    PredictionTracking-backed cache nightly — a 2-minute staleness window keeps
    repeat dashboard renders fast (the previous behaviour did a 500-700 ms
    UFC.com round-trip on every refresh).
    """
    now = time.monotonic()
    cached = _UPCOMING_CACHE["data"]
    if cached is not None and (now - _UPCOMING_CACHE["ts"]) < _UPCOMING_TTL_SEC:
        return cached

    fresh = _build_upcoming_events()
    _UPCOMING_CACHE["data"] = fresh
    _UPCOMING_CACHE["ts"] = now
    return fresh


def _build_upcoming_events() -> list[dict]:
    """Uncached upcoming-events build — see list_upcoming_events for caller contract."""
    from ml.predict_card import scrape_upcoming_events

    raw = scrape_upcoming_events() or []

    with get_db_connection() as conn:
        resolved = _resolved_event_slugs(conn)

    out = []
    for evt in raw:
        url = evt.get("url", "")
        if not url:
            continue
        slug = _slug_from_url(url)
        if slug in resolved:
            continue
        scraped_name = evt.get("name", "") or ""
        out.append(
            {
                "id": slug,
                "name": _format_event_name(slug, scraped_name),
                "url": url,
                "full_text": evt.get("full_text", "") or scraped_name,
            }
        )
    return out


def list_past_events(limit: int = 4) -> list[dict]:
    """Return the last `limit` resolved events ordered by actual event date.

    Aggregates PredictionTracking rows where `was_correct IS NOT NULL`,
    grouping by event so each event appears once with its record + accuracy.
    Pulls a wider candidate set than `limit` then sorts in Python by the
    parsed event date — `event_date` is a display string so we can't ORDER BY
    it in SQL and get chronological order.
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT event_name,
                   MAX(event_url)      AS event_url,
                   MAX(event_date)     AS event_date,
                   COUNT(*)            AS total,
                   COUNT(*) FILTER (WHERE was_correct) AS correct,
                   MAX(predicted_at)   AS last_predicted
            FROM PredictionTracking
            WHERE was_correct IS NOT NULL
            GROUP BY event_name
            ORDER BY MAX(predicted_at) DESC NULLS LAST
            LIMIT %s
            """,
            (max(limit * 4, 20),),
        )
        rows = cur.fetchall()
        cur.close()

    out = []
    for name, url, date, total, correct, last_predicted in rows:
        if not url:
            continue
        total = int(total or 0)
        correct = int(correct or 0)
        iso_date = _event_date_to_iso(date, last_predicted)
        out.append(
            {
                "id": _slug_from_url(url),
                "name": name,
                "date": iso_date,
                "url": url,
                "total": total,
                "correct": correct,
                "accuracy": (correct / total * 100.0) if total else 0.0,
                "_sort": iso_date or (last_predicted.strftime("%Y-%m-%d") if last_predicted else ""),
            }
        )

    out.sort(key=lambda e: e["_sort"], reverse=True)
    for e in out:
        e.pop("_sort", None)
    return out[:limit]


def get_past_event_details(event_id: str) -> dict | None:
    """Fetch a single past event's predictions + actuals.

    Looks up rows by `event_url` (the canonical, unique key) rather than
    `event_name`, which is often a generic string like "UFC Fight Night" that
    would ILIKE-match dozens of different events. Returns None if the event
    has no resolved rows at all (gate to keep upcoming events off the past
    page).
    """
    event_url = _url_from_slug(event_id)

    with get_db_connection() as conn:
        # Gate: must have at least one resolved row for this URL.
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1 FROM PredictionTracking
            WHERE event_url = %s AND was_correct IS NOT NULL
            LIMIT 1
            """,
            (event_url,),
        )
        has_resolved = cur.fetchone() is not None
        cur.close()
        if not has_resolved:
            return None

        cached = _fetch_cached_predictions_by_url(conn, event_url)
        if not cached:
            return None
        return _cached_response(event_id, event_url, cached, is_past=True)


def _enrich_with_odds(
    results: list[dict],
    event_url: str,
    event_name: str,
    is_past: bool,
) -> list[dict]:
    """Attach real American odds + edge_pct to each fight prediction.

    Priority for upcoming events:
      1. Stored DB snapshot from fight_odds (captured Wednesday / day_before).
      2. Live BestFightOdds homepage scrape (15-min in-process cache).
    Past events fall back to the historical_odds.json archive.
    Fights with no match get american=None and odds_source='unavailable'.
    """
    from DelphiAIApp.Services.odds_provider import get_event_odds, _normalize_name
    from DelphiAIApp.Services.odds_math import compute_edge_pct
    from DelphiAIApp.Models.ml.scrape_odds import lookup_best_odds

    # Pre-load live/historical odds map (used as fallback for upcoming fights
    # that don't have a stored snapshot yet, and as primary for past events).
    odds_map = get_event_odds(event_url, event_name, is_past=is_past) or {}

    enriched = []
    with get_db_connection() as conn:
        for r in results:
            item = dict(r)
            f1 = r.get("fighter1", "")
            f2 = r.get("fighter2", "")
            f1_prob = float(r.get("f1_prob", 0.5))

            m: dict | None = None

            # For upcoming events try the DB snapshot first.
            if not is_past:
                m = lookup_best_odds(conn, event_url, f1, f2)

            # Fall through to live BFO / historical archive.
            if m is None:
                key = (_normalize_name(f1), _normalize_name(f2))
                m = odds_map.get(key)

            if m and m.get("decimal_a") and m.get("decimal_b"):
                dec_a = float(m["decimal_a"])
                dec_b = float(m["decimal_b"])
                item["f1_american"] = int(m["american_a"])
                item["f2_american"] = int(m["american_b"])
                item["f1_edge_pct"] = round(compute_edge_pct(f1_prob, dec_a, dec_b), 2)
                item["f2_edge_pct"] = round(compute_edge_pct(1.0 - f1_prob, dec_b, dec_a), 2)
                item["odds_source"] = m.get("source", "bestfightodds")
            else:
                item["f1_american"] = None
                item["f2_american"] = None
                item["f1_edge_pct"] = 0.0
                item["f2_edge_pct"] = 0.0
                item["odds_source"] = "unavailable"

            enriched.append(_to_jsonable(item))
    return enriched


_CACHED_PREDICTIONS_COLUMNS = """
    event_name, event_date, event_url,
    fighter1_name, fighter2_name,
    pick_name, pick_probability, fighter1_probability,
    confidence, prob_source,
    predicted_method, predicted_ko, predicted_sub, predicted_dec,
    predicted_r1, predicted_r2, predicted_r3,
    is_title_fight, fighter1_elo, fighter2_elo,
    actual_winner_name, actual_method, actual_round, was_correct,
    predicted_at, resolved_at
"""


def _shape_cached_rows(rows: list[dict]) -> list[dict]:
    """Convert raw PredictionTracking dicts into the [meta, *fights] shape."""
    if not rows:
        return []
    out = []
    for p in rows:
        f1_prob = float(p["fighter1_probability"]) if p["fighter1_probability"] else 0.5
        out.append(
            {
                "fighter1": p["fighter1_name"],
                "fighter2": p["fighter2_name"],
                "f1_prob": f1_prob,
                "f2_prob": 1.0 - f1_prob,
                "pick": p["pick_name"],
                "pick_pct": float(p["pick_probability"]) if p["pick_probability"] else 0.5,
                "confidence": p["confidence"],
                "prob_source": p["prob_source"],
                "method": p["predicted_method"],
                "ko_pct": float(p["predicted_ko"]) if p["predicted_ko"] is not None else None,
                "sub_pct": float(p["predicted_sub"]) if p["predicted_sub"] is not None else None,
                "dec_pct": float(p["predicted_dec"]) if p["predicted_dec"] is not None else None,
                "r1_finish": float(p["predicted_r1"]) if p["predicted_r1"] is not None else None,
                "r2_finish": float(p["predicted_r2"]) if p["predicted_r2"] is not None else None,
                "r3_finish": float(p["predicted_r3"]) if p["predicted_r3"] is not None else None,
                "is_title": bool(p["is_title_fight"]),
                "f1_elo": float(p["fighter1_elo"]) if p["fighter1_elo"] is not None else None,
                "f2_elo": float(p["fighter2_elo"]) if p["fighter2_elo"] is not None else None,
                "actual_winner": p["actual_winner_name"],
                "actual_method": p["actual_method"],
                "actual_round": p["actual_round"],
                "was_correct": p["was_correct"],
                "notes": [],
            }
        )
    meta = {
        "event_name": rows[0]["event_name"],
        "event_date": _event_date_to_iso(
            rows[0]["event_date"], rows[0].get("predicted_at")
        ) or rows[0]["event_date"],
        "event_url": rows[0]["event_url"],
    }
    return [meta, *out]


def _fetch_cached_predictions(conn, event_name: str) -> list[dict]:
    """Read prior predictions for this event from PredictionTracking by event_name."""
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT {_CACHED_PREDICTIONS_COLUMNS}
        FROM PredictionTracking
        WHERE event_name ILIKE %s
        ORDER BY id
        """,
        (event_name,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return _shape_cached_rows(rows)


def _fetch_cached_predictions_by_url(conn, event_url: str) -> list[dict]:
    """Read prior predictions for this event from PredictionTracking by event_url.

    Used as the primary cache-hit path: a single indexed SELECT keyed on the
    canonical UFC.com URL avoids the UFC.com round-trip we used to do just to
    discover the event_name. Returns [] when no rows exist for the URL.
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT {_CACHED_PREDICTIONS_COLUMNS}
        FROM PredictionTracking
        WHERE event_url = %s
        ORDER BY id
        """,
        (event_url,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return _shape_cached_rows(rows)


def _run_predict_card(conn, event_url: str) -> dict | None:
    """Scrape the card from UFC.com and invoke the predict pipeline. Returns dict or None."""
    from ml.predict_card import scrape_event_card, predict_full_card

    # Single-shot user-facing request — skip the polite 1-3 s rate-limit sleep.
    card = scrape_event_card(event_url, delay=False)
    if not card or not card.get("fights"):
        return None

    # The pipeline prints heavily — capture stdout so the API stays clean.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = predict_full_card(
            conn,
            card["fights"],
            event_name=card.get("event_name", ""),
            event_date=card.get("event_date", ""),
            event_url=event_url,
        )
    return {
        "event_name": card.get("event_name", ""),
        "event_date": card.get("event_date", ""),
        "event_url": event_url,
        "results": results or [],
    }


def _cached_response(
    event_id: str, event_url: str, cached: list[dict], is_past: bool
) -> dict:
    meta, *fights = cached
    return {
        "id": event_id,
        "name": meta["event_name"],
        "date": meta["event_date"],
        "url": meta.get("event_url") or event_url,
        "fights": _enrich_with_odds(
            fights, event_url, meta.get("event_name", ""), is_past=is_past
        ),
        "source": "cache",
    }


def get_event_predictions(event_id: str, force_recompute: bool = False) -> dict | None:
    """
    Fetch predictions for a given event slug.

    Cache-first: a single indexed SELECT on `event_url` decides whether we can
    serve from PredictionTracking. We only scrape UFC.com + run the predict
    pipeline on a true cache miss (or when force_recompute is set), saving the
    ~2 s UFC.com round-trip on every cached read.
    """
    event_url = _url_from_slug(event_id)

    with get_db_connection() as conn:
        if not force_recompute:
            cached = _fetch_cached_predictions_by_url(conn, event_url)
            if cached:
                return _cached_response(event_id, event_url, cached, is_past=False)

        live = _run_predict_card(conn, event_url)
        if not live:
            return None
        return {
            "id": event_id,
            "name": live["event_name"],
            "date": live["event_date"],
            "url": live["event_url"],
            "fights": _enrich_with_odds(
                live["results"], event_url, live.get("event_name", ""), is_past=False
            ),
            "source": "live",
        }


def get_event_predictions_cached(event_id: str) -> dict | None:
    """Cache-only lookup — never scrapes UFC.com, never runs the live pipeline.

    Used by the dashboard main-event preview where we want sub-100 ms reads.
    Returns None if no cached rows exist (e.g. the prefetch cron hasn't run yet).
    """
    event_url = _url_from_slug(event_id)
    with get_db_connection() as conn:
        cached = _fetch_cached_predictions_by_url(conn, event_url)
        if not cached:
            return None
        return _cached_response(event_id, event_url, cached, is_past=False)


def resolve_event_results(event_name: str, force: bool = False) -> dict | None:
    """Run update_results.resolve_event and return a summary."""
    from ml.update_results import resolve_event

    with get_db_connection() as conn:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            resolved_name, predictions = resolve_event(conn, event_name, force=force)
        if not resolved_name:
            return None

        total = len(predictions)
        resolved = sum(1 for p in predictions if p.get("was_correct") is not None)
        correct = sum(1 for p in predictions if p.get("was_correct") is True)
        return {
            "event_name": resolved_name,
            "total": total,
            "resolved": resolved,
            "correct": correct,
            "accuracy_pct": (correct / resolved * 100) if resolved else None,
        }

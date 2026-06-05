"""
Automatic result resolution for past events.

Scans PredictionTracking for events whose `event_date` has passed but still
have rows with `was_correct IS NULL`, then calls update_results.resolve_event
on each. Designed to be invoked as a FastAPI BackgroundTask after a UI
request, so the user doesn't wait — the next dashboard render picks up the
newly-resolved rows.

A module-level TTL guard prevents thrashing: multiple dashboard refreshes
within `_TTL_SEC` skip the rescan entirely.
"""
from __future__ import annotations

import contextlib
import io
import logging
import threading
import time
from datetime import date, datetime

from DelphiAIApp.Models.db.postgres import get_db_connection

logger = logging.getLogger(__name__)

_TTL_SEC = 900.0  # 15 min — UFC.com updates results within hours of an event
_state_lock = threading.Lock()
_LAST_RUN_TS: float = 0.0
_RUNNING: bool = False


def _parse_event_date(date_str: str | None) -> date | None:
    """
    Best-effort parse of the PredictionTracking.event_date display string
    (e.g. "Sat, May 30 / 7:00 AM" or "2026-03-21") to a `date`. Mirrors
    `ml.prefetch_predictions._parse_event_date` so we don't drift.
    """
    if not date_str:
        return None
    s = date_str.strip()
    s = s.split("/")[0].strip()
    if "," in s:
        head, tail = s.split(",", 1)
        if head.strip().lower()[:3] in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}:
            s = tail.strip()
    try:
        return datetime.fromisoformat(s.split("T")[0]).date()
    except ValueError:
        pass
    today = date.today()
    for fmt in ("%b %d %Y", "%B %d %Y", "%b %d", "%B %d"):
        try:
            d = datetime.strptime(s, fmt).date()
            if "%Y" not in fmt:
                d = d.replace(year=today.year)
                if d < today.replace(month=1, day=1):
                    d = d.replace(year=today.year + 1)
            return d
        except ValueError:
            continue
    return None


def _find_pending_past_events(conn) -> list[tuple[str, str | None, str | None]]:
    """
    Return [(event_name, event_url, event_date_str)] for live-prediction
    events whose parsed event_date is strictly before today AND still have
    at least one unresolved row.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT event_name,
               MAX(event_url)  AS event_url,
               MAX(event_date) AS event_date
        FROM PredictionTracking
        WHERE COALESCE(prediction_type, 'live') = 'live'
        GROUP BY event_name
        HAVING COUNT(*) FILTER (WHERE was_correct IS NULL) > 0
        """
    )
    rows = cur.fetchall()
    cur.close()

    today = date.today()
    out: list[tuple[str, str | None, str | None]] = []
    for name, url, date_str in rows:
        parsed = _parse_event_date(date_str)
        if parsed is None:
            logger.debug("auto_resolve: skipping '%s' — unparseable date %r", name, date_str)
            continue
        if parsed >= today:
            continue  # event hasn't happened yet
        out.append((name, url, date_str))
    return out


def auto_resolve_pending(force_ttl: bool = False) -> dict:
    """
    Find every past-but-unresolved event and run update_results.resolve_event
    on each. Returns a summary dict suitable for the manual endpoint.

    Re-entrancy guard: skips if another invocation is in progress, and
    enforces a TTL between successful runs unless `force_ttl=True`.
    """
    global _LAST_RUN_TS, _RUNNING

    now = time.monotonic()
    with _state_lock:
        if _RUNNING:
            return {"status": "skipped", "reason": "already running"}
        if not force_ttl and (now - _LAST_RUN_TS) < _TTL_SEC:
            return {"status": "skipped", "reason": "ttl", "next_in_sec": int(_TTL_SEC - (now - _LAST_RUN_TS))}
        _RUNNING = True

    summary = {"status": "ok", "events": []}
    try:
        from ml.update_results import resolve_event

        with get_db_connection() as conn:
            pending = _find_pending_past_events(conn)
            if not pending:
                logger.info("auto_resolve: no past-but-unresolved events found")
                summary["events"] = []
                return summary

            logger.info("auto_resolve: %d events to attempt", len(pending))
            for name, url, date_str in pending:
                event_summary = {"event_name": name, "event_date": date_str}
                try:
                    # Capture noisy resolver output so the API/logs stay clean.
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        resolved_name, predictions = resolve_event(
                            conn, name, event_url=url, force=False
                        )
                    if not resolved_name:
                        event_summary["status"] = "no_match"
                    else:
                        total = len(predictions)
                        resolved = sum(1 for p in predictions if p.get("was_correct") is not None)
                        correct = sum(1 for p in predictions if p.get("was_correct") is True)
                        event_summary.update(
                            {
                                "status": "resolved",
                                "total": total,
                                "resolved": resolved,
                                "correct": correct,
                            }
                        )
                        logger.info(
                            "auto_resolve: %s → %d/%d resolved, %d correct",
                            resolved_name, resolved, total, correct,
                        )
                except Exception as e:
                    logger.warning("auto_resolve: %s failed: %s", name, e)
                    event_summary["status"] = "error"
                    event_summary["error"] = str(e)
                summary["events"].append(event_summary)
    finally:
        with _state_lock:
            _RUNNING = False
            _LAST_RUN_TS = time.monotonic()

    return summary

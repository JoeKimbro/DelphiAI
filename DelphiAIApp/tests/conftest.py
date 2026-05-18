"""Shared pytest fixtures + sys.path setup for DelphiAIApp tests.

Run with (from project root):
    python -m pytest DelphiAIApp/tests/ -v

The Services + ML modules use two different import roots:
- `from DelphiAIApp.Services...` expects the project root on sys.path.
- `from ml.predict_card import ...` (lazy import inside event_service) expects
  DelphiAIApp/Models on sys.path.

This conftest injects both so test files can stay clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]                  # .../DelphiAI/
_MODELS_DIR = _PROJECT_ROOT / "DelphiAIApp" / "Models"

for p in (_PROJECT_ROOT, _MODELS_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


@pytest.fixture(scope="session")
def db_available() -> bool:
    """True if the Postgres instance configured in .env is reachable.

    Lets DB-dependent tests skip cleanly when the user runs the suite without
    Docker up. Cached at session scope so we only probe once.
    """
    try:
        from DelphiAIApp.Models.db.postgres import get_db_connection

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
        return True
    except Exception:
        return False


@pytest.fixture
def require_db(db_available: bool) -> None:
    """Skip the test if Postgres is unreachable."""
    if not db_available:
        pytest.skip("Postgres not reachable; skipping DB-dependent test.")


@pytest.fixture
def known_resolved_event(require_db) -> dict:
    """A real resolved event from PredictionTracking, for read-only assertions.

    Returns {event_url, event_name, slug, fight_count}. Fails the test if no
    resolved events exist yet — that means update_results has never run, so
    the past-events code paths can't be exercised.
    """
    from DelphiAIApp.Models.db.postgres import get_db_connection
    from DelphiAIApp.Services.event_service import _slug_from_url

    with get_db_connection() as conn:
        cur = conn.cursor()
        # Step 1: find the most-recently-resolved event_url.
        cur.execute(
            """
            SELECT event_url, MAX(event_name)
            FROM PredictionTracking
            WHERE was_correct IS NOT NULL
              AND event_url IS NOT NULL AND event_url <> ''
            GROUP BY event_url
            ORDER BY MAX(resolved_at) DESC NULLS LAST
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            pytest.skip("No resolved events in PredictionTracking.")
        url, name = row
        # Step 2: count TOTAL fights for that URL (resolved + any stragglers).
        # `_fetch_cached_predictions_by_url` returns all rows, so we compare to
        # the same total — not just the resolved subset.
        cur.execute(
            "SELECT COUNT(*) FROM PredictionTracking WHERE event_url = %s",
            (url,),
        )
        total = int(cur.fetchone()[0])
        cur.close()
    return {
        "event_url": url,
        "event_name": name,
        "slug": _slug_from_url(url),
        "fight_count": total,
    }


@pytest.fixture
def known_cached_upcoming(require_db) -> dict | None:
    """An unresolved cached event in PredictionTracking, or None if none exist."""
    from DelphiAIApp.Models.db.postgres import get_db_connection
    from DelphiAIApp.Services.event_service import _slug_from_url

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT event_url, event_name, COUNT(*) AS n
            FROM PredictionTracking
            WHERE was_correct IS NULL
              AND event_url IS NOT NULL AND event_url <> ''
            GROUP BY event_url, event_name
            ORDER BY MAX(predicted_at) DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        cur.close()
    if not row:
        return None
    return {
        "event_url": row[0],
        "event_name": row[1],
        "slug": _slug_from_url(row[0]),
        "fight_count": int(row[2]),
    }

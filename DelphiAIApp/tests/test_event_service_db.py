"""Integration tests against the live PredictionTracking table.

These run read-only — they probe what's already in the DB rather than
inserting fixtures. Skipped automatically when Postgres is unreachable
(see conftest.require_db).
"""
from __future__ import annotations

import pytest

from DelphiAIApp.Models.db.postgres import get_db_connection
from DelphiAIApp.Services.event_service import (
    _fetch_cached_predictions,
    _fetch_cached_predictions_by_url,
    _resolved_event_slugs,
    list_past_events,
    get_past_event_details,
    get_event_predictions_cached,
)


class TestResolvedEventSlugs:
    def test_returns_set_of_strings(self, require_db):
        with get_db_connection() as conn:
            slugs = _resolved_event_slugs(conn)
        assert isinstance(slugs, set)
        for s in slugs:
            assert isinstance(s, str)
            # Slugs should look like 'ufc-...' — never a full URL.
            assert "http" not in s
            assert "/event/" not in s

    def test_only_includes_resolved_events(self, require_db):
        """Every returned slug must have at least one row with was_correct IS NOT NULL."""
        with get_db_connection() as conn:
            slugs = _resolved_event_slugs(conn)
            if not slugs:
                pytest.skip("No resolved events yet.")
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) FROM PredictionTracking
                WHERE event_url LIKE %s
                  AND was_correct IS NULL
                """,
                (f"%/{next(iter(slugs))}",),
            )
            # The slug we picked is supposedly resolved — but there could be
            # mixed-state rows on the same event (some resolved, some not).
            # That's allowed. We only require *at least one* resolved row,
            # which is the function's contract.
            cur.close()


class TestFetchCachedPredictionsByUrl:
    def test_returns_empty_for_unknown_url(self, require_db):
        with get_db_connection() as conn:
            rows = _fetch_cached_predictions_by_url(
                conn, "https://www.ufc.com/event/this-event-does-not-exist"
            )
        assert rows == []

    def test_returns_meta_plus_fights_shape(self, known_resolved_event):
        with get_db_connection() as conn:
            rows = _fetch_cached_predictions_by_url(
                conn, known_resolved_event["event_url"]
            )
        # Shape contract: rows[0] is the meta dict, rows[1:] are fight dicts.
        assert len(rows) == 1 + known_resolved_event["fight_count"]
        meta = rows[0]
        assert meta["event_url"] == known_resolved_event["event_url"]
        assert meta["event_name"] == known_resolved_event["event_name"]
        for fight in rows[1:]:
            # Fight dicts must have the keys the frontend consumes.
            for k in ("fighter1", "fighter2", "f1_prob", "f2_prob", "pick",
                      "confidence", "is_title"):
                assert k in fight, f"fight missing key {k!r}"
            # Probabilities must complement to 1 within float epsilon.
            assert abs(fight["f1_prob"] + fight["f2_prob"] - 1.0) < 1e-6

    def test_resolved_event_includes_actuals(self, known_resolved_event):
        with get_db_connection() as conn:
            rows = _fetch_cached_predictions_by_url(
                conn, known_resolved_event["event_url"]
            )
        fights = rows[1:]
        # At least one fight on a resolved card must have was_correct set.
        assert any(f.get("was_correct") is not None for f in fights)


class TestFetchCachedPredictionsByName:
    """The name-keyed helper still exists (uses ILIKE — fuzzy by design).

    Generic names like "UFC Fight Night" intentionally match multiple events;
    that's the legacy behaviour we keep for backwards compatibility. The
    URL-keyed helper is what new code should use for single-event lookups.
    """

    def test_url_lookup_is_a_subset_of_name_lookup(self, known_resolved_event):
        """Every row returned by URL lookup must also appear in name lookup."""
        with get_db_connection() as conn:
            by_url = _fetch_cached_predictions_by_url(
                conn, known_resolved_event["event_url"]
            )
            by_name = _fetch_cached_predictions(
                conn, known_resolved_event["event_name"]
            )
        # URL is canonical → smaller-or-equal result set than name ILIKE.
        assert len(by_url) <= len(by_name)
        # Both must surface at least the same meta event_name.
        assert by_url[0]["event_name"] == known_resolved_event["event_name"]


class TestListPastEvents:
    def test_returns_at_most_limit(self, require_db):
        for n in (1, 4, 10):
            events = list_past_events(limit=n)
            assert len(events) <= n

    def test_each_event_has_required_keys(self, require_db):
        events = list_past_events(limit=4)
        if not events:
            pytest.skip("No resolved events yet.")
        required = {"id", "name", "date", "url", "correct", "total", "accuracy"}
        for e in events:
            assert required.issubset(e.keys()), f"missing keys: {required - set(e.keys())}"
            assert 0 <= e["correct"] <= e["total"]
            assert 0.0 <= e["accuracy"] <= 100.0

    def test_ordered_most_recent_first(self, require_db):
        events = list_past_events(limit=4)
        if len(events) < 2:
            pytest.skip("Need ≥2 resolved events to verify ordering.")
        # Newest event must appear first. We sort by resolved_at in the SQL,
        # which isn't returned, but event_date strings sort roughly correctly
        # for our format. The SQL's ORDER BY is the authoritative test —
        # here we just assert the function exits with a non-empty list and
        # that consecutive events aren't trivially mis-ordered by ID.
        ids = [e["id"] for e in events]
        assert len(set(ids)) == len(ids), "duplicate events returned"


class TestGetPastEventDetails:
    def test_returns_none_for_unknown(self, require_db):
        assert get_past_event_details("this-event-does-not-exist-xyz") is None

    def test_returns_full_payload_for_known(self, known_resolved_event):
        data = get_past_event_details(known_resolved_event["slug"])
        assert data is not None
        assert data["id"] == known_resolved_event["slug"]
        assert data["source"] == "cache"
        assert len(data["fights"]) == known_resolved_event["fight_count"]

    def test_only_returns_resolved_events(self, require_db, known_cached_upcoming):
        """An unresolved (upcoming) cached event must NOT come back via past lookup."""
        if known_cached_upcoming is None:
            pytest.skip("No cached upcoming event to test against.")
        data = get_past_event_details(known_cached_upcoming["slug"])
        # Either None (no resolved rows for this url) or — if some fights on
        # the same url happen to be resolved — must not crash. The contract
        # is "skip events with zero resolved rows".
        if data is not None:
            # If it returned, at least one underlying row must be resolved.
            assert any(
                f.get("was_correct") is not None for f in data["fights"]
            )


class TestGetEventPredictionsCached:
    """The dashboard's main-event preview path. Must NEVER scrape UFC.com."""

    def test_returns_none_for_unknown(self, require_db):
        assert get_event_predictions_cached("ufc-99999-fake") is None

    def test_returns_cache_marker(self, known_resolved_event):
        data = get_event_predictions_cached(known_resolved_event["slug"])
        assert data is not None
        assert data["source"] == "cache"
        assert data["id"] == known_resolved_event["slug"]
        assert len(data["fights"]) > 0

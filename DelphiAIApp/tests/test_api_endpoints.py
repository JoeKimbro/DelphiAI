"""End-to-end API tests using FastAPI's TestClient.

Boots the real app + hits the real Postgres. Each test verifies status code
and response shape — the heavy logic is covered by the service-layer unit
tests; here we just confirm the wiring (routes registered, parameters bound,
shapes serialised correctly).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from DelphiAIApp.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"


class TestUpcomingEndpoint:
    def test_returns_events_array(self, client, require_db):
        r = client.get("/api/events/upcoming")
        assert r.status_code == 200
        body = r.json()
        assert "events" in body
        assert isinstance(body["events"], list)

    def test_event_objects_have_required_keys(self, client, require_db):
        r = client.get("/api/events/upcoming")
        for evt in r.json().get("events", []):
            assert {"id", "name", "url"}.issubset(evt.keys())
            assert isinstance(evt["id"], str) and evt["id"]
            assert "http" not in evt["id"]  # id is a slug, not a URL

    def test_resolved_events_not_in_upcoming(
        self, client, require_db, known_resolved_event
    ):
        """A resolved event's slug must NOT appear in the upcoming list."""
        r = client.get("/api/events/upcoming")
        ids = {e["id"] for e in r.json().get("events", [])}
        assert known_resolved_event["slug"] not in ids


class TestPastEventsEndpoint:
    def test_returns_at_most_limit(self, client, require_db):
        for n in (1, 4, 10):
            r = client.get(f"/api/events/past?limit={n}")
            assert r.status_code == 200
            assert len(r.json().get("events", [])) <= n

    def test_default_limit_capped_below_50(self, client, require_db):
        """Controller clamps limit to ≤50 to prevent runaway queries."""
        r = client.get("/api/events/past?limit=9999")
        assert r.status_code == 200
        assert len(r.json().get("events", [])) <= 50

    def test_event_shape(self, client, require_db):
        r = client.get("/api/events/past?limit=4")
        for e in r.json().get("events", []):
            for k in ("id", "name", "url", "correct", "total", "accuracy"):
                assert k in e
            assert 0 <= e["correct"] <= e["total"]
            assert 0.0 <= e["accuracy"] <= 100.0


class TestPastEventDetailEndpoint:
    def test_404_on_unknown_id(self, client, require_db):
        r = client.get("/api/events/past/this-event-id-definitely-does-not-exist")
        assert r.status_code == 404

    def test_known_past_event_returns_predictions(
        self, client, known_resolved_event
    ):
        r = client.get(f"/api/events/past/{known_resolved_event['slug']}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == known_resolved_event["slug"]
        assert body["source"] == "cache"
        assert len(body["fights"]) == known_resolved_event["fight_count"]
        # Past-event detail should include actuals on at least one fight.
        assert any(f.get("was_correct") is not None for f in body["fights"])


class TestPredictionsEndpoint:
    def test_cache_hit_returns_cache_source(self, client, known_resolved_event):
        r = client.get(f"/api/events/{known_resolved_event['slug']}/predictions")
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "cache"

    def test_cached_only_query_param_returns_cache_or_404(
        self, client, known_resolved_event
    ):
        """cached_only=true must never trigger a live prediction."""
        r = client.get(
            f"/api/events/{known_resolved_event['slug']}/predictions"
            "?cached_only=true"
        )
        # Either we get a cache hit, or 404 — never a "live" response.
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert r.json()["source"] == "cache"

    def test_unknown_slug_returns_404_or_502(self, client, require_db):
        # Unknown slug → cache miss → live scrape attempt → 404 from UFC.com
        # → bubbles up as 404 (or 502 if BeautifulSoup raises). Either is fine
        # as long as it's an error, not a 200.
        r = client.get("/api/events/ufc-this-does-not-exist-99999/predictions")
        assert r.status_code in (404, 502)


class TestRouteOrdering:
    """Make sure /past doesn't get caught by /{event_id}/predictions."""

    def test_past_route_is_not_treated_as_slug(self, client, require_db):
        # If /past/{id}/predictions matched /{event_id}/predictions with id=past,
        # we'd get a totally wrong handler. The endpoint must resolve to the
        # past handler and 404 on unknown IDs.
        r = client.get("/api/events/past/known-bad-slug")
        assert r.status_code == 404
        # The 404 body must come from the past handler, not the predictions one.
        detail = r.json().get("detail", "")
        assert "resolved predictions" in detail.lower() or "no" in detail.lower()

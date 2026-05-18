"""Cache-behaviour tests for event_service.

Verifies two key optimisations:
1. The TTL cache around scrape_upcoming_events skips a second UFC.com call.
2. get_event_predictions checks the DB first and never scrapes on cache hits.
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

from DelphiAIApp.Services import event_service as es


@pytest.fixture(autouse=True)
def reset_upcoming_cache():
    """Clear the module-level TTL cache before every test in this file."""
    es._UPCOMING_CACHE["data"] = None
    es._UPCOMING_CACHE["ts"] = 0.0
    yield
    es._UPCOMING_CACHE["data"] = None
    es._UPCOMING_CACHE["ts"] = 0.0


class TestUpcomingTTLCache:
    """list_upcoming_events caches the scrape for _UPCOMING_TTL_SEC."""

    def _fake_scrape(self):
        """One-shot mock that records call count + returns a deterministic list."""
        return [
            {"name": "Allen vs Costa", "url": "https://www.ufc.com/event/ufc-fight-night-may-16-2026", "full_text": "Allen vs Costa"},
        ]

    def test_first_call_scrapes(self, require_db):
        with patch.object(es, "_build_upcoming_events", wraps=lambda: self._fake_scrape()) as m:
            es.list_upcoming_events()
            assert m.call_count == 1

    def test_second_call_within_ttl_hits_cache(self, require_db):
        with patch.object(es, "_build_upcoming_events", side_effect=lambda: self._fake_scrape()) as m:
            es.list_upcoming_events()
            es.list_upcoming_events()
            es.list_upcoming_events()
            assert m.call_count == 1, "TTL cache should serve subsequent calls"

    def test_returns_identical_object_within_ttl(self, require_db):
        """The cached list reference itself should be returned, not a copy."""
        with patch.object(es, "_build_upcoming_events", side_effect=lambda: self._fake_scrape()):
            a = es.list_upcoming_events()
            b = es.list_upcoming_events()
        assert a is b

    def test_expired_cache_triggers_rebuild(self, require_db):
        with patch.object(es, "_build_upcoming_events", side_effect=lambda: self._fake_scrape()) as m:
            es.list_upcoming_events()
            # Fast-forward past the TTL by editing the timestamp.
            es._UPCOMING_CACHE["ts"] = time.monotonic() - (es._UPCOMING_TTL_SEC + 1)
            es.list_upcoming_events()
            assert m.call_count == 2, "Expired cache should re-scrape"

    def test_cache_survives_an_empty_scrape_result(self, require_db):
        """If UFC.com returns nothing (transient failure), don't poison the cache forever."""
        first_call = [True]

        def flaky():
            if first_call[0]:
                first_call[0] = False
                return []
            return self._fake_scrape()

        with patch.object(es, "_build_upcoming_events", side_effect=flaky):
            first = es.list_upcoming_events()
            # Force expiry; second call should be allowed to re-scrape.
            es._UPCOMING_CACHE["ts"] = time.monotonic() - (es._UPCOMING_TTL_SEC + 1)
            second = es.list_upcoming_events()
        # Both calls return *something* — first is the (empty) cached value
        # within TTL, second after expiry rebuilds.
        assert first == [] or first  # either is fine
        assert second  # after expiry we got the real list

    def test_ttl_constant_is_reasonable(self):
        # Guard against someone accidentally setting TTL to 0 or hours.
        assert 30.0 <= es._UPCOMING_TTL_SEC <= 600.0


class TestCacheFirstPredictionPath:
    """get_event_predictions must hit the DB *before* scraping UFC.com."""

    def test_cache_hit_does_not_call_scraper(self, require_db, known_resolved_event):
        """The killer test: a cached event must NOT trigger scrape_event_card."""
        scrape_calls = []

        def spy_scrape(*args, **kwargs):
            scrape_calls.append(args)
            return {"event_name": "X", "fights": []}

        # Patch the scraper at its source so any code path hitting it gets caught.
        with patch("ml.predict_card.scrape_event_card", side_effect=spy_scrape):
            data = es.get_event_predictions(
                known_resolved_event["slug"], force_recompute=False
            )

        assert data is not None
        assert data["source"] == "cache"
        assert scrape_calls == [], (
            "scrape_event_card was called on a cache hit — the optimisation regressed"
        )

    def test_force_recompute_bypasses_cache(self, require_db, known_resolved_event):
        """force_recompute=True must skip the cache and go to the live pipeline."""
        # _run_predict_card will call scrape_event_card; we intercept and short-circuit
        # to None so we don't actually run the model in a test.
        with patch.object(es, "_run_predict_card", return_value=None) as m:
            result = es.get_event_predictions(
                known_resolved_event["slug"], force_recompute=True
            )
        assert m.called, "force_recompute=True must call _run_predict_card"
        assert result is None  # because we stubbed _run_predict_card → None

    def test_cache_miss_falls_through_to_live_pipeline(self, require_db):
        """An event with no cached rows must fall through to the live pipeline."""
        with patch.object(es, "_run_predict_card", return_value=None) as m:
            result = es.get_event_predictions("this-slug-does-not-exist-12345")
        assert m.called, "Cache miss must invoke _run_predict_card"
        assert result is None

    def test_get_event_predictions_cached_never_scrapes(
        self, require_db, known_resolved_event
    ):
        """Even on a cache miss, the *_cached variant must NEVER call the scraper."""
        with patch("ml.predict_card.scrape_event_card") as scrape_mock, \
             patch.object(es, "_run_predict_card") as live_mock:
            # Known-cached event:
            es.get_event_predictions_cached(known_resolved_event["slug"])
            # Unknown event (cache miss):
            es.get_event_predictions_cached("this-slug-does-not-exist-99999")
        assert not scrape_mock.called
        assert not live_mock.called


class TestScrapeEventCardDelayFlag:
    """The delay=False path on scrape_event_card avoids the 1-3s sleep."""

    def test_delay_false_skips_sleep(self):
        from ml import predict_card

        with patch.object(predict_card.time, "sleep") as sleep_mock, \
             patch.object(predict_card, "_get_session") as session_mock:
            # Make the HTTP call fail quickly so we exit before doing real work.
            session_mock.return_value.get.side_effect = Exception("test short-circuit")
            predict_card.scrape_event_card(
                "https://www.ufc.com/event/x", delay=False
            )
        assert not sleep_mock.called, "delay=False must NOT call time.sleep"

    def test_delay_true_sleeps_by_default(self):
        from ml import predict_card

        with patch.object(predict_card.time, "sleep") as sleep_mock, \
             patch.object(predict_card, "_get_session") as session_mock:
            session_mock.return_value.get.side_effect = Exception("test short-circuit")
            predict_card.scrape_event_card("https://www.ufc.com/event/x")
        assert sleep_mock.called, "Default delay=True must call time.sleep"

"""Scraper tests using a captured UFC.com HTML snapshot.

The fixture file `fixtures/ufc_events_index.html` was downloaded once with the
HTML structure UFC.com served at the time. These tests guard against:

  1. UFC.com changing the HTML structure (.l-listing__item, etc.) in a way
     that would silently break scrape_upcoming_events.
  2. The positional-matching regression we fixed for UFC Freedom 250 — each
     event must get its OWN heading, not its neighbour's.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "ufc_events_index.html"


def _load_fixture() -> str:
    if not _FIXTURE.exists():
        pytest.skip(f"Fixture not found: {_FIXTURE}")
    return _FIXTURE.read_text(encoding="utf-8")


def _patched_scrape():
    """Run scrape_upcoming_events against the fixture HTML."""
    from ml import predict_card

    fake_response = MagicMock()
    fake_response.text = _load_fixture()
    fake_response.raise_for_status = MagicMock(return_value=None)

    fake_session = MagicMock()
    fake_session.get = MagicMock(return_value=fake_response)

    with patch.object(predict_card, "_get_session", return_value=fake_session):
        return predict_card.scrape_upcoming_events()


class TestScrapeReturnsEvents:
    def test_returns_non_empty_list(self):
        events = _patched_scrape()
        assert isinstance(events, list)
        assert len(events) > 0

    def test_each_event_has_required_keys(self):
        for e in _patched_scrape():
            assert "name" in e
            assert "url" in e
            assert "full_text" in e

    def test_urls_are_absolute_and_canonical(self):
        for e in _patched_scrape():
            url = e["url"]
            assert url.startswith("https://www.ufc.com/event/"), url
            assert "?" not in url and "#" not in url
            assert not url.endswith("/")

    def test_urls_are_unique(self):
        urls = [e["url"] for e in _patched_scrape()]
        assert len(urls) == len(set(urls)), "Duplicate URLs in scrape output"


class TestPerCardHeadingExtraction:
    """The UFC Freedom 250 regression: each URL must keep its own heading."""

    def test_freedom_250_gets_its_own_headline(self):
        events = _patched_scrape()
        by_slug = {e["url"].rsplit("/", 1)[-1]: e for e in events}
        if "ufc-freedom-250" not in by_slug:
            pytest.skip("Fixture doesn't include ufc-freedom-250 anymore.")
        # The fixture's per-card headline for freedom-250 is the brand itself
        # ("UFC Freedom 250"), NOT the next event's "Kape vs Horiguchi".
        name = by_slug["ufc-freedom-250"]["name"]
        assert "Kape" not in name, (
            "Freedom 250 stole Kape vs Horiguchi's heading — positional bug regressed"
        )

    def test_kape_horiguchi_event_keeps_its_own_headline(self):
        events = _patched_scrape()
        by_slug = {e["url"].rsplit("/", 1)[-1]: e for e in events}
        target = "ufc-fight-night-june-20-2026"
        if target not in by_slug:
            pytest.skip(f"Fixture missing {target}.")
        # Kape vs Horiguchi is the june-20 main event — not june-27.
        assert "Kape" in by_slug[target]["name"] or "Horiguchi" in by_slug[target]["name"]

    def test_fight_night_events_have_matchup_headings(self):
        """Generic fight-night cards should give us a 'X vs Y' headline."""
        events = _patched_scrape()
        # At least one fight-night card on a reasonable /events snapshot
        # should have a real matchup heading.
        fn_events = [
            e for e in events
            if "ufc-fight-night-" in e["url"]
        ]
        with_vs = [e for e in fn_events if "vs" in e["name"].lower()]
        assert len(with_vs) >= 1, "Expected at least one fight-night with 'vs' heading"

    def test_no_event_inherits_anothers_name(self):
        """Heuristic check: consecutive URLs shouldn't share a heading."""
        events = _patched_scrape()
        for prev, curr in zip(events, events[1:]):
            assert prev["name"] != curr["name"] or "TBD" in prev["name"].upper(), (
                f"Adjacent events share a heading: {prev['url']} and {curr['url']} "
                f"both labelled {prev['name']!r}"
            )


class TestScrapeReturnsAtMost20:
    """Hard cap from the scraper: events[:20]."""

    def test_no_more_than_twenty(self):
        events = _patched_scrape()
        assert len(events) <= 20

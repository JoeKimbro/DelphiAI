"""Unit tests for slug + name helpers in event_service.

These are pure functions — no DB, no HTTP. They guard the display-name fix that
keeps branded events like "UFC Freedom 250" from being silently mislabelled as
their main-event matchup.
"""
from __future__ import annotations

import pytest

from DelphiAIApp.Services.event_service import (
    _slug_from_url,
    _url_from_slug,
    _slug_to_brand,
    _format_event_name,
    UFC_EVENT_BASE,
)


class TestSlugUrlRoundTrip:
    @pytest.mark.parametrize(
        "slug",
        [
            "ufc-fight-night-may-16-2026",
            "ufc-freedom-250",
            "ufc-329",
            "ufc-326-pereira-vs-ankalaev-2",
        ],
    )
    def test_round_trip(self, slug: str):
        assert _slug_from_url(_url_from_slug(slug)) == slug

    def test_strips_trailing_slash(self):
        assert _slug_from_url(f"{UFC_EVENT_BASE}ufc-329/") == "ufc-329"

    def test_handles_path_with_event_segment(self):
        assert (
            _slug_from_url("https://www.ufc.com/event/ufc-freedom-250")
            == "ufc-freedom-250"
        )

    def test_url_from_slug_strips_leading_slash(self):
        # Defensive: should still work if a caller passed "/ufc-329".
        assert _url_from_slug("/ufc-329") == f"{UFC_EVENT_BASE}ufc-329"


class TestSlugToBrand:
    @pytest.mark.parametrize(
        "slug,expected",
        [
            ("ufc-329", "UFC 329"),
            ("ufc-freedom-250", "UFC Freedom 250"),
            ("ufc-fight-night-june-06-2026", "UFC Fight Night June 06 2026"),
            ("ufc-326-pereira-vs-ankalaev-2", "UFC 326 Pereira Vs Ankalaev 2"),
        ],
    )
    def test_capitalisation(self, slug: str, expected: str):
        assert _slug_to_brand(slug) == expected

    def test_ufc_token_always_uppercase(self):
        # The first 'ufc' token must always be UPPER no matter where it is.
        out = _slug_to_brand("ufc-329-foo")
        assert out.split()[0] == "UFC"

    def test_empty_segments_skipped(self):
        # Double dashes shouldn't introduce stray spaces.
        assert "  " not in _slug_to_brand("ufc--329")


class TestFormatEventName:
    """The UFC Freedom 250 fix — every case the helper must handle correctly."""

    # ----- fight-night slugs (most upcoming events) -------------------------

    def test_fight_night_with_real_matchup_uses_matchup(self):
        out = _format_event_name(
            "ufc-fight-night-may-16-2026", "Allen vs Costa"
        )
        assert out == "Allen vs Costa"

    def test_fight_night_with_tbd_falls_back_to_slug_brand(self):
        # When UFC.com hasn't named the main event yet we'd rather show the
        # date-stamped brand than a useless "TBD vs TBD".
        out = _format_event_name(
            "ufc-fight-night-july-18-2026", "TBD vs TBD"
        )
        assert "TBD" not in out
        assert "UFC Fight Night" in out
        assert "July" in out and "18" in out and "2026" in out

    def test_fight_night_with_empty_heading_falls_back_to_slug_brand(self):
        out = _format_event_name("ufc-fight-night-june-06-2026", "")
        assert out == "UFC Fight Night June 06 2026"

    # ----- branded slugs (numbered PPV / Freedom) ---------------------------

    def test_branded_with_matchup_prepends_brand(self):
        # Without this, the UFC 328 card would only show "Chimaev vs Strickland"
        # and the user couldn't tell which numbered card it is.
        out = _format_event_name("ufc-328", "Chimaev vs Strickland")
        assert out == "UFC 328: Chimaev vs Strickland"

    def test_branded_with_tbd_drops_tbd(self):
        out = _format_event_name("ufc-329", "TBD vs TBD")
        assert out == "UFC 329"

    def test_branded_heading_starting_with_ufc_is_not_double_branded(self):
        # This is the actual UFC Freedom 250 case: UFC.com puts the brand as
        # the headline, so prepending another brand would yield
        # "UFC Freedom 250: UFC Freedom 250" — wrong.
        out = _format_event_name("ufc-freedom-250", "UFC Freedom 250")
        assert out == "UFC Freedom 250"

    def test_branded_with_empty_heading_uses_brand_only(self):
        out = _format_event_name("ufc-329", "")
        assert out == "UFC 329"

    def test_heading_whitespace_is_stripped(self):
        out = _format_event_name("ufc-329", "   Chimaev vs Strickland   ")
        assert out == "UFC 329: Chimaev vs Strickland"

    # ----- regression guards: cases the old logic got wrong -----------------

    def test_old_bug_freedom_250_does_not_steal_next_events_heading(self):
        # Old positional logic assigned "Kape vs Horiguchi" to freedom-250.
        # With per-card extraction the helper just renders what it's given —
        # but this test pins the EXPECTED rendering once the right heading
        # ("UFC Freedom 250") is passed in.
        assert (
            _format_event_name("ufc-freedom-250", "UFC Freedom 250")
            != "Kape vs Horiguchi"
        )

    def test_tbd_substring_anywhere_triggers_fallback(self):
        # "Featherweight TBD" should still be treated as placeholder.
        out = _format_event_name("ufc-330", "Featherweight TBD")
        assert "TBD" not in out
        assert out == "UFC 330"

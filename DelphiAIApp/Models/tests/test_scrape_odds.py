"""Unit tests for ml/scrape_odds.py — no network, no DB required."""
import sys
from datetime import date, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from DelphiAIApp.Models.ml.scrape_odds import (
    UpcomingEvent,
    BfoFight,
    _normalize_name,
    _surname,
    _wednesday_before,
    due_captures,
    match_fights,
    _parse_bfo_html,
)


# ---------------------------------------------------------------------------
# _normalize_name / _surname
# ---------------------------------------------------------------------------

def test_normalize_strips_diacritics():
    assert _normalize_name("Uroš Medić") == "uros medic"


def test_normalize_collapses_whitespace():
    assert _normalize_name("  Jon   Jones  ") == "jon jones"


def test_surname_returns_last_token():
    assert _surname("Jon Jones") == "jones"
    assert _surname("Israel Adesanya") == "adesanya"


def test_surname_single_name():
    assert _surname("Overeem") == "overeem"


# ---------------------------------------------------------------------------
# _wednesday_before
# ---------------------------------------------------------------------------

def test_wednesday_before_saturday_card():
    # Saturday 2025-03-22 → Wednesday 2025-03-19
    saturday = date(2025, 3, 22)
    assert _wednesday_before(saturday) == date(2025, 3, 19)


def test_wednesday_before_sunday_card():
    # Sunday 2025-03-23 → Wednesday 2025-03-19
    sunday = date(2025, 3, 23)
    assert _wednesday_before(sunday) == date(2025, 3, 19)


def test_wednesday_before_thursday_card():
    # Thursday 2025-03-20 → Wednesday 2025-03-19 (1 day before)
    thursday = date(2025, 3, 20)
    assert _wednesday_before(thursday) == date(2025, 3, 19)


def test_wednesday_before_returns_none_when_too_far():
    # A Wednesday itself has no Wednesday within the 6-day window before it.
    wednesday = date(2025, 3, 19)
    result = _wednesday_before(wednesday)
    assert result is None


# ---------------------------------------------------------------------------
# due_captures
# ---------------------------------------------------------------------------

def _make_event(event_date: date) -> UpcomingEvent:
    return UpcomingEvent(
        event_url="https://ufc.com/event/test",
        event_name="Test Event",
        event_date=event_date,
        fights=[("Fighter A", "Fighter B")],
    )


def test_due_captures_saturday_card_on_wednesday():
    saturday = date(2025, 3, 22)
    ev = _make_event(saturday)
    wednesday = date(2025, 3, 19)
    result = due_captures([ev], wednesday)
    assert len(result) == 1
    assert result[0][1] == "wednesday"


def test_due_captures_saturday_card_on_friday():
    saturday = date(2025, 3, 22)
    ev = _make_event(saturday)
    friday = date(2025, 3, 21)
    result = due_captures([ev], friday)
    assert len(result) == 1
    assert result[0][1] == "day_before"


def test_due_captures_thursday_card_collision():
    # Thursday 2025-03-20: day_before = Wed 2025-03-19 = wednesday
    thursday = date(2025, 3, 20)
    ev = _make_event(thursday)
    wednesday = date(2025, 3, 19)
    result = due_captures([ev], wednesday)
    # Collision: only one capture, should be day_before
    assert len(result) == 1
    assert result[0][1] == "day_before"


def test_due_captures_not_due_today():
    saturday = date(2025, 3, 22)
    ev = _make_event(saturday)
    # Monday before (not Wednesday or Friday)
    monday = date(2025, 3, 17)
    assert due_captures([ev], monday) == []


# ---------------------------------------------------------------------------
# match_fights
# ---------------------------------------------------------------------------

def _bfo(name_a: str, name_b: str, am_a: int = -150, am_b: int = 130) -> BfoFight:
    from DelphiAIApp.Services.odds_math import american_to_decimal
    return BfoFight(name_a, name_b, am_a, am_b,
                    american_to_decimal(am_a), american_to_decimal(am_b))


def test_match_fights_correct_orientation():
    ev = _make_event(date(2025, 3, 22))
    ev.fights = [("Jon Jones", "Stipe Miocic")]
    bfo = [_bfo("Jon Jones", "Stipe Miocic", -300, 240)]
    result = match_fights(ev, bfo)
    assert len(result) == 1
    assert result[0].fighter1_name == "Jon Jones"
    assert result[0].f1_best_american == -300


def test_match_fights_swapped_orientation():
    ev = _make_event(date(2025, 3, 22))
    ev.fights = [("Jon Jones", "Stipe Miocic")]
    # BFO lists them in reverse order
    bfo = [_bfo("Stipe Miocic", "Jon Jones", 240, -300)]
    result = match_fights(ev, bfo)
    assert len(result) == 1
    # f1 = Jones, should get BFO's "b" line (-300)
    assert result[0].fighter1_name == "Jon Jones"
    assert result[0].f1_best_american == -300


def test_match_fights_no_match_logs_and_skips():
    ev = _make_event(date(2025, 3, 22))
    ev.fights = [("Jon Jones", "Stipe Miocic")]
    bfo = [_bfo("Khabib Nurmagomedov", "Conor McGregor")]
    result = match_fights(ev, bfo)
    assert result == []


# ---------------------------------------------------------------------------
# _parse_bfo_html (fixture HTML)
# ---------------------------------------------------------------------------

_FIXTURE_HTML = dedent("""
<html><body>
<div class="table-div">
  <div class="table-scroller">
    <table class="odds-table">
      <tr>
        <td><a href="/fighters/jon-jones">Jon Jones</a></td>
        <td>-300</td>
      </tr>
      <tr>
        <td><a href="/fighters/stipe-miocic">Stipe Miocic</a></td>
        <td>+240</td>
      </tr>
      <tr>
        <td><a href="/fighters/israel-adesanya">Israel Adesanya</a></td>
        <td>-175</td>
      </tr>
      <tr>
        <td><a href="/fighters/alex-pereira">Alex Pereira</a></td>
        <td>+145</td>
      </tr>
    </table>
  </div>
</div>
</body></html>
""")


def test_parse_bfo_html_extracts_fights():
    fights = _parse_bfo_html(_FIXTURE_HTML)
    assert len(fights) == 2


def test_parse_bfo_html_correct_lines():
    fights = _parse_bfo_html(_FIXTURE_HTML)
    jones_fight = next(f for f in fights if "Jones" in f.fighter_a)
    assert jones_fight.american_a == -300
    assert jones_fight.american_b == 240
    assert jones_fight.decimal_a == pytest.approx(1.333, rel=1e-2)

"""Tests for the resolver's event-name normalization + date resolution.

Root cause being fixed: the same event stored under variant spellings
('Adesanya vs Pyfer' vs 'Adesanya vs. Pyfer') and non-ISO display dates
('Sat, Mar 28 / 8:00 PM') caused duplicate batches and left whole batches
unresolved. Run from DelphiAIApp/Models:

    python -m pytest ml/tests/test_resolution_matching.py -v
"""
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml import update_results as ur


# --- _normalize_event_name (dedup + grouping key) ---------------------------

def test_normalize_collapses_vs_punctuation_and_case():
    a = ur._normalize_event_name("UFC Fight Night: Adesanya vs Pyfer")
    b = ur._normalize_event_name("UFC Fight Night: Adesanya vs. Pyfer")
    assert a == b


def test_normalize_distinguishes_different_events():
    a = ur._normalize_event_name("UFC Fight Night: Adesanya vs Pyfer")
    b = ur._normalize_event_name("UFC Fight Night: Evloev vs Murphy")
    assert a != b


# --- _select_variant_names (resolve ALL variants, not just the first) -------

def test_select_variant_names_groups_punctuation_variants():
    all_names = [
        "UFC Fight Night: Adesanya vs Pyfer",
        "UFC Fight Night: Adesanya vs. Pyfer",
        "UFC 300: Someone vs Other",
    ]
    chosen = "UFC Fight Night: Adesanya vs. Pyfer"
    variants = ur._select_variant_names(all_names, chosen)
    assert set(variants) == {
        "UFC Fight Night: Adesanya vs Pyfer",
        "UFC Fight Night: Adesanya vs. Pyfer",
    }


# --- _definitive_event_date (enable display-date DB resolution) -------------

def test_definitive_date_accepts_iso():
    assert ur._definitive_event_date("", "2026-03-28") == date(2026, 3, 28)


def test_definitive_date_infers_year_from_predicted_at_for_display_date():
    # 'Sat, Mar 28 / 8:00 PM' has no year; predicted_at anchors it to 2026.
    predicted = datetime(2026, 3, 27, 10, 0, 0)
    got = ur._definitive_event_date("", "Sat, Mar 28 / 8:00 PM", predicted_at=predicted)
    assert got == date(2026, 3, 28)


def test_definitive_date_handles_year_boundary():
    # Event 'Jan 2' predicted on Dec 30 belongs to the NEXT year.
    predicted = datetime(2025, 12, 30, 10, 0, 0)
    got = ur._definitive_event_date("", "Thu, Jan 2 / 8:00 PM", predicted_at=predicted)
    assert got == date(2026, 1, 2)


def test_definitive_date_still_none_without_any_year_anchor():
    # No URL year, non-ISO date, no predicted_at -> can't trust it (rematch guard).
    assert ur._definitive_event_date("", "Sat, Mar 28 / 8:00 PM") is None


# --- _event_sort_key: per-event history must sort by real date ---------------

def test_event_sort_key_parses_iso_date():
    from ml.performance_summary import _event_sort_key
    # ISO event_date must sort by the EVENT date, not fall back to predicted_at.
    got = _event_sort_key("2026-06-20", datetime(2026, 6, 18, 12, 0))
    assert got.date() == date(2026, 6, 20)


def test_event_sort_key_parses_display_date():
    from ml.performance_summary import _event_sort_key
    got = _event_sort_key("Sat, May 16 / 8:00 PM", datetime(2026, 5, 14, 12, 0))
    assert got.date() == date(2026, 5, 16)


def test_event_sort_key_falls_back_to_predicted_at():
    from ml.performance_summary import _event_sort_key
    anchor = datetime(2026, 5, 14, 12, 0)
    assert _event_sort_key("", anchor) == anchor


# --- _derive_event_name: label generic Fight Nights by their main event ------

def test_derive_event_name_adds_main_event_subtitle():
    from ml.predict_card import _derive_event_name
    fights = [{"fighter1": "Manel Kape", "fighter2": "Kyoji Horiguchi"}]
    assert _derive_event_name("UFC Fight Night", fights) == "UFC Fight Night: Kape vs Horiguchi"


def test_derive_event_name_keeps_existing_subtitle():
    from ml.predict_card import _derive_event_name
    fights = [{"fighter1": "Alex Pereira", "fighter2": "Magomed Ankalaev"}]
    assert _derive_event_name("UFC 320: Pereira vs Ankalaev", fights) == "UFC 320: Pereira vs Ankalaev"

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from DelphiAIApp.Services.odds_math import (
    american_to_decimal,
    decimal_to_american,
    decimal_to_implied,
    remove_vig,
    compute_edge_pct,
)


# american_to_decimal
def test_amd_favorite():
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_amd_underdog():
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_amd_zero_raises():
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_amd_none_raises():
    with pytest.raises((ValueError, TypeError)):
        american_to_decimal(None)


# decimal_to_american
def test_dta_favorite_roundtrip():
    assert decimal_to_american(american_to_decimal(-200)) == -200


def test_dta_underdog_roundtrip():
    assert decimal_to_american(american_to_decimal(150)) == 150


def test_dta_at_or_below_one_raises():
    with pytest.raises(ValueError):
        decimal_to_american(1.0)


def test_dta_none_raises():
    with pytest.raises((ValueError, TypeError)):
        decimal_to_american(None)


# decimal_to_implied
def test_dti_even_money():
    assert decimal_to_implied(2.0) == pytest.approx(0.5)


def test_dti_zero_or_negative_returns_zero():
    assert decimal_to_implied(0) == 0.0
    assert decimal_to_implied(-1) == 0.0


def test_dti_none_returns_zero():
    assert decimal_to_implied(None) == 0.0


# remove_vig
def test_remove_vig_sums_to_one():
    p1, p2 = remove_vig(0.55, 0.50)
    assert p1 + p2 == pytest.approx(1.0)


def test_remove_vig_zero_total():
    p1, p2 = remove_vig(0, 0)
    assert p1 == pytest.approx(0.5)
    assert p2 == pytest.approx(0.5)


# compute_edge_pct
def test_edge_positive_when_model_more_bullish():
    # market: -200 fav, +160 dog (after vig removal ≈ 63% / 37%)
    market_dec = american_to_decimal(-200)
    opp_dec = american_to_decimal(160)
    edge = compute_edge_pct(0.70, market_dec, opp_dec)
    assert edge > 0


def test_edge_negative_when_model_less_bullish():
    market_dec = american_to_decimal(-200)
    opp_dec = american_to_decimal(160)
    edge = compute_edge_pct(0.50, market_dec, opp_dec)
    assert edge < 0


def test_edge_none_inputs_return_zero():
    assert compute_edge_pct(0.6, None, 1.5) == 0.0
    assert compute_edge_pct(0.6, 1.5, None) == 0.0


def test_edge_decimal_lte_one_returns_zero():
    assert compute_edge_pct(0.6, 1.0, 2.0) == 0.0
    assert compute_edge_pct(0.6, 2.0, 0.5) == 0.0

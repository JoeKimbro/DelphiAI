import sys
from pathlib import Path
from decimal import Decimal

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from DelphiAIApp.Services.performance_service import _jsonable


def test_bare_decimal_becomes_float():
    result = _jsonable(Decimal("3.14"))
    assert result == pytest.approx(3.14)
    assert isinstance(result, float)


def test_dict_with_decimal_values():
    result = _jsonable({"roi": Decimal("12.5"), "n": 10})
    assert result == {"roi": pytest.approx(12.5), "n": 10}


def test_list_with_decimal_values():
    result = _jsonable([Decimal("1.0"), Decimal("2.0")])
    assert result == [pytest.approx(1.0), pytest.approx(2.0)]


def test_nested_structure():
    data = {"buckets": [{"profit": Decimal("5.25"), "bets": 3}]}
    result = _jsonable(data)
    assert result["buckets"][0]["profit"] == pytest.approx(5.25)
    assert result["buckets"][0]["bets"] == 3


def test_non_decimal_scalars_pass_through():
    assert _jsonable(42) == 42
    assert _jsonable("hello") == "hello"
    assert _jsonable(None) is None


def _fake_picks(n, prob=0.70, correct=True):
    return [
        {"pick_probability": prob, "was_correct": correct}
        for _ in range(n)
    ]


def test_paper_roi_not_reportable_below_threshold():
    from ml.performance_summary import _calculate_paper_trading, MIN_ROI_BETS

    result = _calculate_paper_trading(_fake_picks(MIN_ROI_BETS - 1))
    assert result["high_conf"]["bets"] == MIN_ROI_BETS - 1
    assert result["high_conf"]["reportable"] is False


def test_paper_roi_reportable_at_threshold():
    from ml.performance_summary import _calculate_paper_trading, MIN_ROI_BETS

    result = _calculate_paper_trading(_fake_picks(MIN_ROI_BETS))
    assert result["high_conf"]["bets"] == MIN_ROI_BETS
    assert result["high_conf"]["reportable"] is True

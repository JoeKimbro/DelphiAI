"""
Performance summary service — wraps ml/performance_summary.calculate_stats.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from DelphiAIApp.Models.db.postgres import get_db_connection


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def get_performance_summary(prediction_type: str = "live", year: int | None = None) -> dict:
    from ml.performance_summary import (
        get_resolved_predictions, calculate_stats, get_prediction_counts,
        _event_sort_key,
    )

    ptype = prediction_type if prediction_type in ("live", "backtest") else None

    with get_db_connection() as conn:
        predictions = get_resolved_predictions(conn, prediction_type=ptype)
        counts = get_prediction_counts(conn)

    # Distinct event years derived the same way the per-event sort does it
    # (parse event_date string with predicted_at as anchor, fall back to
    # predicted_at when the string is empty/unparseable).
    years_present: set[int] = set()
    for p in predictions:
        dt = _event_sort_key(p.get("event_date"), p.get("predicted_at"))
        if dt is not None:
            years_present.add(dt.year)
    available_years = sorted(years_present, reverse=True)

    if year is not None:
        predictions = [
            p for p in predictions
            if (dt := _event_sort_key(p.get("event_date"), p.get("predicted_at")))
            is not None and dt.year == year
        ]

    stats = calculate_stats(predictions) if predictions else None
    return _jsonable(
        {
            "prediction_type": prediction_type,
            "counts": counts,
            "stats": stats,
            "available_years": available_years,
            "selected_year": year,
        }
    )

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


def get_performance_summary(prediction_type: str = "live") -> dict:
    from ml.performance_summary import get_resolved_predictions, calculate_stats, get_prediction_counts

    ptype = prediction_type if prediction_type in ("live", "backtest") else None

    with get_db_connection() as conn:
        predictions = get_resolved_predictions(conn, prediction_type=ptype)
        counts = get_prediction_counts(conn)

    stats = calculate_stats(predictions) if predictions else None
    return _jsonable(
        {
            "prediction_type": prediction_type,
            "counts": counts,
            "stats": stats,
        }
    )

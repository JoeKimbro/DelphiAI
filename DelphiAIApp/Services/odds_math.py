"""
Single source of truth for sportsbook-odds conversions and edge math.

Decimal odds represent total payout per unit staked (1.5 means stake 1 → get 1.5
back). American odds are the US sportsbook display convention (-200 favorite,
+150 dog). Implied probabilities derived from decimal odds include the
sportsbook's vig; `remove_vig` normalizes a two-way market so the de-vigged
probabilities sum to 1.
"""
from __future__ import annotations


def american_to_decimal(american: int | float) -> float:
    if american is None:
        raise ValueError("american odds required")
    am = float(american)
    if am == 0:
        raise ValueError("american odds cannot be zero")
    if am > 0:
        return 1.0 + (am / 100.0)
    return 1.0 + (100.0 / -am)


def decimal_to_american(decimal: float) -> int:
    if decimal is None or decimal <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0 (got {decimal!r})")
    if decimal >= 2.0:
        # Underdog: +N where N = (decimal - 1) * 100
        return int(round((decimal - 1.0) * 100.0))
    # Favorite: -N where N = 100 / (decimal - 1)
    return -int(round(100.0 / (decimal - 1.0)))


def decimal_to_implied(decimal: float) -> float:
    if decimal is None or decimal <= 0:
        return 0.0
    return 1.0 / decimal


def remove_vig(p1: float, p2: float) -> tuple[float, float]:
    """Proportional vig removal — normalize a two-way market to sum=1."""
    total = p1 + p2
    if total <= 0:
        return 0.5, 0.5
    return p1 / total, p2 / total


def compute_edge_pct(
    model_prob: float, market_decimal: float, opponent_decimal: float
) -> float:
    """
    Edge in percentage points: model_prob minus the no-vig market implied
    probability for the same fighter, scaled to PP. Positive means the model
    is more bullish than the market; negative means less bullish.
    """
    if market_decimal is None or opponent_decimal is None:
        return 0.0
    if market_decimal <= 1.0 or opponent_decimal <= 1.0:
        return 0.0
    p_a = decimal_to_implied(market_decimal)
    p_b = decimal_to_implied(opponent_decimal)
    no_vig_a, _ = remove_vig(p_a, p_b)
    return (model_prob - no_vig_a) * 100.0

"""Allowlist guard for SQL identifiers interpolated into f-string queries.

These statements interpolate table/column NAMES (psycopg2 cannot parameterize
identifiers). All call sites pass hardcoded internal constants, but routing them
through this guard makes the safety explicit and fails loudly if that ever
changes.
"""
from __future__ import annotations

ALLOWED_TABLES: frozenset[str] = frozenset({
    "Fights", "CareerStats", "FighterStats",
    "PointInTimeStats", "MatchupFeatures", "OpponentQuality",
    "PreUfcCareer", "EloHistory",
})


def safe_identifier(name: str, allowed: frozenset[str]) -> str:
    """Return `name` if it is in `allowed`, else raise ValueError."""
    if name not in allowed:
        raise ValueError(f"Refusing to interpolate unknown SQL identifier: {name!r}")
    return name

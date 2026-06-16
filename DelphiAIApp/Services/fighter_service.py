"""
Fighter profile service — reads FighterStats + CareerStats + EloHistory + recent Fights.

Slugs accepted as input:
  - URL-style slug from FighterURL (preferred, stable)
  - Lowercased name with hyphens ("islam-makhachev") as a fallback
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any
from urllib.parse import unquote

from DelphiAIApp.Models.db.postgres import get_db_connection


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _rows(cur) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _find_fighter(cur, slug: str) -> dict | None:
    """Try multiple lookup strategies. Returns the FighterStats row + CareerStats join."""
    decoded = unquote(slug).strip()
    name_guess = decoded.replace("-", " ")

    # 1. Exact name (case-insensitive)
    cur.execute(
        """
        SELECT fs.*, cs.SLpM, cs.StrAcc, cs.SApM, cs.StrDef,
               cs.TDAvg, cs.TDAcc, cs.TDDef, cs.SubAvg,
               cs.AvgFightDuration, cs.FirstRoundFinishRate, cs.DecisionRate,
               cs.EloRating, cs.PeakEloRating
        FROM FighterStats fs
        LEFT JOIN CareerStats cs ON cs.FighterID = fs.FighterID
        WHERE LOWER(fs.Name) = LOWER(%s)
        LIMIT 1
        """,
        (name_guess,),
    )
    rows = _rows(cur)
    if rows:
        return rows[0]

    # 2. ILIKE wildcard
    cur.execute(
        """
        SELECT fs.*, cs.SLpM, cs.StrAcc, cs.SApM, cs.StrDef,
               cs.TDAvg, cs.TDAcc, cs.TDDef, cs.SubAvg,
               cs.AvgFightDuration, cs.FirstRoundFinishRate, cs.DecisionRate,
               cs.EloRating, cs.PeakEloRating
        FROM FighterStats fs
        LEFT JOIN CareerStats cs ON cs.FighterID = fs.FighterID
        WHERE fs.Name ILIKE %s
        ORDER BY fs.TotalFights DESC NULLS LAST
        LIMIT 1
        """,
        (f"%{name_guess}%",),
    )
    rows = _rows(cur)
    if rows:
        return rows[0]

    # 3. URL slug match
    cur.execute(
        """
        SELECT fs.*, cs.SLpM, cs.StrAcc, cs.SApM, cs.StrDef,
               cs.TDAvg, cs.TDAcc, cs.TDDef, cs.SubAvg,
               cs.AvgFightDuration, cs.FirstRoundFinishRate, cs.DecisionRate,
               cs.EloRating, cs.PeakEloRating
        FROM FighterStats fs
        LEFT JOIN CareerStats cs ON cs.FighterID = fs.FighterID
        WHERE fs.FighterURL ILIKE %s
        LIMIT 1
        """,
        (f"%{slug}%",),
    )
    rows = _rows(cur)
    if rows:
        return rows[0]

    # 4. Apostrophe-stripped match — handles slugs like "sean-omalley" → "Sean O'Malley"
    import re
    stripped = re.sub(r"[^a-z0-9]", "", name_guess.lower())
    if stripped:
        cur.execute(
            """
            SELECT fs.*, cs.SLpM, cs.StrAcc, cs.SApM, cs.StrDef,
                   cs.TDAvg, cs.TDAcc, cs.TDDef, cs.SubAvg,
                   cs.AvgFightDuration, cs.FirstRoundFinishRate, cs.DecisionRate,
                   cs.EloRating, cs.PeakEloRating
            FROM FighterStats fs
            LEFT JOIN CareerStats cs ON cs.FighterID = fs.FighterID
            WHERE LOWER(REGEXP_REPLACE(fs.Name, '[^a-zA-Z0-9]', '', 'g')) LIKE %s
            ORDER BY fs.TotalFights DESC NULLS LAST
            LIMIT 1
            """,
            (f"%{stripped}%",),
        )
        rows = _rows(cur)
        if rows:
            return rows[0]

    return None


def get_fighter_profile(slug: str, recent_n: int = 10, elo_n: int = 30) -> dict | None:
    """Return a complete fighter profile: stats, ELO history, recent fights."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        fighter = _find_fighter(cur, slug)
        if not fighter:
            cur.close()
            return None

        fighter_id = fighter["fighterid"]

        # ELO history (most recent N)
        cur.execute(
            """
            SELECT FightDate, EloBeforeFight, EloAfterFight, EloChange,
                   Result, Method, OpponentEloBeforeFight, ExpectedWinProb
            FROM EloHistory
            WHERE FighterID = %s
            ORDER BY FightDate DESC
            LIMIT %s
            """,
            (fighter_id, elo_n),
        )
        elo_history = _rows(cur)
        elo_history.reverse()  # chronological for charts

        # Recent fights
        cur.execute(
            """
            SELECT Date, OpponentName, Result, WinnerName, Method, MethodDetail,
                   Round, Time, EventName, IsTitleFight, IsMainEvent
            FROM Fights
            WHERE FighterID = %s AND Date IS NOT NULL
            ORDER BY Date DESC
            LIMIT %s
            """,
            (fighter_id, recent_n),
        )
        fights = _rows(cur)

        cur.close()

    return _jsonable(
        {
            "id": fighter_id,
            "name": fighter.get("name"),
            "nickname": fighter.get("nickname"),
            "url": fighter.get("fighterurl"),
            "weight_class": fighter.get("weightclass"),
            "stance": fighter.get("stance"),
            "height": fighter.get("height"),
            "weight": fighter.get("weight"),
            "reach": fighter.get("reach"),
            "leg_reach": fighter.get("legreach"),
            "dob": fighter.get("dob"),
            "age": fighter.get("age"),
            "place_of_birth": fighter.get("placeofbirth"),
            "record": {
                "wins": fighter.get("wins"),
                "losses": fighter.get("losses"),
                "draws": fighter.get("draws"),
                "total_fights": fighter.get("totalfights"),
            },
            "days_since_last_fight": fighter.get("dayssincelastfight"),
            "is_active": fighter.get("isactive"),
            "career_stats": {
                "slpm": fighter.get("slpm"),
                "str_acc": fighter.get("stracc"),
                "sapm": fighter.get("sapm"),
                "str_def": fighter.get("strdef"),
                "td_avg": fighter.get("tdavg"),
                "td_acc": fighter.get("tdacc"),
                "td_def": fighter.get("tddef"),
                "sub_avg": fighter.get("subavg"),
                "avg_fight_duration": fighter.get("avgfightduration"),
                "first_round_finish_rate": fighter.get("firstroundfinishrate"),
                "decision_rate": fighter.get("decisionrate"),
                "elo_rating": fighter.get("elorating"),
                "peak_elo": fighter.get("peakelorating"),
            },
            "elo_history": elo_history,
            "recent_fights": fights,
        }
    )

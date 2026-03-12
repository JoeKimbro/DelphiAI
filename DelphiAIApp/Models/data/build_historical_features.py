"""
Build FighterHistoricalFeatures from fight history in PostgreSQL.

These features are point-in-time safe: every row for (fighter, fight) is
computed using only fights strictly before that fight.
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'dbname': os.getenv('DB_NAME', 'delphi_db'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
}


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS FighterHistoricalFeatures (
    HistoricalFeatureID SERIAL PRIMARY KEY,
    FighterID INTEGER NOT NULL,
    FightID INTEGER NOT NULL,
    FightDate DATE NOT NULL,
    AvgOpponentEloLast3 DECIMAL(8,2),
    AvgOpponentEloLast5 DECIMAL(8,2),
    MaxOpponentEloLast5 DECIMAL(8,2),
    EloChangeLast3 DECIMAL(8,2),
    EloVelocity DECIMAL(8,2),
    FinishRateLast3 DECIMAL(6,3),
    FinishRateLast5 DECIMAL(6,3),
    FinishRateCareer DECIMAL(6,3),
    FinishRateTrending DECIMAL(6,3),
    CurrentWinStreak INTEGER DEFAULT 0,
    CurrentLossStreak INTEGER DEFAULT 0,
    OpponentQualityTrending DECIMAL(8,2),
    HasPriorData BOOLEAN DEFAULT FALSE,
    CalculatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (FighterID) REFERENCES FighterStats(FighterID) ON DELETE CASCADE,
    FOREIGN KEY (FightID) REFERENCES Fights(FightID) ON DELETE CASCADE,
    CONSTRAINT unique_fighter_fight_historical UNIQUE (FighterID, FightID)
);
CREATE INDEX IF NOT EXISTS idx_histfeat_fighter_date
    ON FighterHistoricalFeatures(FighterID, FightDate);
CREATE INDEX IF NOT EXISTS idx_histfeat_fightid
    ON FighterHistoricalFeatures(FightID);
"""


def _safe_mean(vals):
    if not vals:
        return None
    return float(np.mean(vals))


def _safe_max(vals):
    if not vals:
        return None
    return float(max(vals))


def _calc_pre_fight_features(state):
    opp_elo_hist = state['opp_elo_hist']
    elo_change_hist = state['elo_change_hist']
    finish_win_hist = state['finish_win_hist']
    wins_total = state['wins_total']
    finish_wins_total = state['finish_wins_total']
    fights_total = state['fights_total']

    last3_opp = opp_elo_hist[-3:]
    last5_opp = opp_elo_hist[-5:]
    prev_3_opp = opp_elo_hist[-6:-3] if len(opp_elo_hist) >= 6 else []
    last3_elo_change = elo_change_hist[-3:]
    last3_finish = finish_win_hist[-3:]
    last5_finish = finish_win_hist[-5:]

    avg_opp_last3 = _safe_mean(last3_opp)
    avg_opp_last5 = _safe_mean(last5_opp)
    max_opp_last5 = _safe_max(last5_opp)

    elo_change_last3 = float(sum(last3_elo_change)) if last3_elo_change else None
    elo_velocity = _safe_mean(last3_elo_change)

    finish_rate_last3 = _safe_mean(last3_finish)
    finish_rate_last5 = _safe_mean(last5_finish)
    finish_rate_career = (finish_wins_total / wins_total) if wins_total > 0 else 0.0
    finish_rate_trending = None
    if finish_rate_last5 is not None:
        finish_rate_trending = float(finish_rate_last5 - finish_rate_career)

    opp_quality_trending = None
    if last3_opp and prev_3_opp:
        opp_quality_trending = float(np.mean(last3_opp) - np.mean(prev_3_opp))

    return {
        'avg_opp_last3': avg_opp_last3,
        'avg_opp_last5': avg_opp_last5,
        'max_opp_last5': max_opp_last5,
        'elo_change_last3': elo_change_last3,
        'elo_velocity': elo_velocity,
        'finish_rate_last3': finish_rate_last3,
        'finish_rate_last5': finish_rate_last5,
        'finish_rate_career': float(finish_rate_career),
        'finish_rate_trending': finish_rate_trending,
        'current_win_streak': int(state['current_win_streak']),
        'current_loss_streak': int(state['current_loss_streak']),
        'opp_quality_trending': opp_quality_trending,
        'has_prior_data': fights_total > 0,
    }


def _update_state_after_fight(state, won, opp_elo_before, elo_change, finish_win):
    state['opp_elo_hist'].append(float(opp_elo_before))
    state['elo_change_hist'].append(float(elo_change))
    state['finish_win_hist'].append(1.0 if finish_win else 0.0)
    state['fights_total'] += 1
    if won:
        state['wins_total'] += 1
        if finish_win:
            state['finish_wins_total'] += 1
        state['current_win_streak'] += 1
        state['current_loss_streak'] = 0
    else:
        state['current_loss_streak'] += 1
        state['current_win_streak'] = 0


def _load_previous_fights_for_fighter(cur, fighter_id, fight_date, fight_id):
    """
    Load this fighter's prior fights (strictly before target fight in builder order).
    """
    cur.execute("""
        SELECT *
        FROM (
            SELECT DISTINCT ON (f.fightid)
                f.fightid,
                f.date,
                f.winnerid,
                f.result,
                f.method,
                CASE WHEN f.fighterid = %s THEN f.opponentid ELSE f.fighterid END AS opponent_id,
                eh.elobeforefight AS fighter_elo_before,
                eh.eloafterfight AS fighter_elo_after,
                eh_opp.elobeforefight AS opponent_elo_before
            FROM fights f
            LEFT JOIN elohistory eh
                ON eh.fighterid = %s AND eh.fightid = f.fightid
            LEFT JOIN elohistory eh_opp
                ON eh_opp.fighterid = CASE WHEN f.fighterid = %s THEN f.opponentid ELSE f.fighterid END
               AND eh_opp.fightid = f.fightid
            WHERE (f.fighterid = %s OR f.opponentid = %s)
              AND f.fighterid IS NOT NULL
              AND (f.winnerid IS NOT NULL OR f.result IN ('win', 'loss'))
              AND f.date IS NOT NULL
              AND (f.date < %s OR (f.date = %s AND f.fightid < %s))
            ORDER BY f.fightid,
                     CASE WHEN f.fighterid = f.winnerid THEN 0 ELSE 1 END,
                     f.date
        ) x
        ORDER BY x.date ASC, x.fightid ASC
    """, (fighter_id, fighter_id, fighter_id, fighter_id, fighter_id, fight_date, fight_date, fight_id))
    return cur.fetchall()


def _recompute_features_for_pair(cur, fighter_id, fight_id, fight_date):
    """
    Recompute expected historical features for (fighter_id, fight_id) from scratch.
    """
    prev_fights = _load_previous_fights_for_fighter(cur, fighter_id, fight_date, fight_id)
    state = {
        'opp_elo_hist': [],
        'elo_change_hist': [],
        'finish_win_hist': [],
        'wins_total': 0,
        'finish_wins_total': 0,
        'fights_total': 0,
        'current_win_streak': 0,
        'current_loss_streak': 0,
    }

    max_seen = None
    for pf in prev_fights:
        pf_id, pf_date, pf_winner_id, pf_result, pf_method, _opp_id, pf_elo_before, pf_elo_after, pf_opp_elo_before = pf
        if pf_winner_id is not None:
            winner_id = int(pf_winner_id)
        else:
            result_s = str(pf_result or "").strip().lower()
            # In this query's canonical row, fighter_id is always in f.fighterid OR f.opponentid.
            if result_s == 'win':
                winner_id = int(fighter_id)
            elif result_s == 'loss':
                winner_id = -1  # any non-matching id means fighter lost
            else:
                winner_id = None
        won = winner_id is not None and int(winner_id) == int(fighter_id)
        method_u = str(pf_method or "").upper()
        finish_win = won and (("KO" in method_u) or ("SUB" in method_u))
        elo_before = float(pf_elo_before) if pf_elo_before is not None else 1500.0
        elo_after = float(pf_elo_after) if pf_elo_after is not None else elo_before
        opp_elo_before = float(pf_opp_elo_before) if pf_opp_elo_before is not None else 1500.0
        elo_change = elo_after - elo_before

        _update_state_after_fight(state, won, opp_elo_before, elo_change, finish_win)
        max_seen = (pf_date, int(pf_id))

    # Leakage sentinel: recompute only uses fights strictly before current in builder order.
    no_future_data = True
    if max_seen is not None:
        no_future_data = (max_seen[0] < fight_date) or (
            max_seen[0] == fight_date and max_seen[1] < int(fight_id)
        )

    feats = _calc_pre_fight_features(state)
    feats['no_future_data'] = bool(no_future_data)
    return feats


def run_leakage_audit(sample_fights=50, tolerance=0.02):
    """
    Randomly sample fights and verify stored historical features by full
    chronological recomputation from scratch.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    print(f"Running leakage audit on {sample_fights} random fights...")

    # Recompute expected features for all fight-fighter pairs in builder order.
    cur.execute("""
        SELECT DISTINCT ON (f.fightid)
            f.fightid, f.date, f.fighterid, f.opponentid, f.winnerid, f.result, f.method
        FROM fights f
        WHERE f.date IS NOT NULL
          AND f.fighterid IS NOT NULL
          AND (f.winnerid IS NOT NULL OR f.result IN ('win', 'loss'))
        ORDER BY f.fightid,
                 CASE WHEN f.fighterid = f.winnerid THEN 0 ELSE 1 END,
                 f.date
    """)
    fights = cur.fetchall()
    fights.sort(key=lambda r: (r[1], r[0]))

    cur.execute("""
        SELECT fighterid, fightid, fightdate, elobeforefight, eloafterfight
        FROM elohistory
    """)
    elo_rows = cur.fetchall()
    elo_map_by_fight = {}
    elo_map_by_date = {}
    for fid, efight_id, fdate, before, after in elo_rows:
        if efight_id is not None:
            elo_map_by_fight[(int(fid), int(efight_id))] = (
                float(before) if before is not None else 1500.0,
                float(after) if after is not None else float(before) if before is not None else 1500.0,
            )
        elo_map_by_date[(int(fid), fdate)] = (
            float(before) if before is not None else 1500.0,
            float(after) if after is not None else float(before) if before is not None else 1500.0,
        )

    state = defaultdict(lambda: {
        'opp_elo_hist': [],
        'elo_change_hist': [],
        'finish_win_hist': [],
        'wins_total': 0,
        'finish_wins_total': 0,
        'fights_total': 0,
        'current_win_streak': 0,
        'current_loss_streak': 0,
    })

    expected = {}
    for fight_id, fight_date, f1_id, f2_id, winner_id, result, method in fights:
        f1_id = int(f1_id)
        f2_id = int(f2_id) if f2_id is not None else None
        if winner_id is not None:
            winner_id = int(winner_id)
        else:
            result_s = str(result or "").strip().lower()
            if result_s == 'win':
                winner_id = f1_id
            elif result_s == 'loss' and f2_id is not None:
                winner_id = f2_id
            else:
                winner_id = None

        fighter_ids = [f1_id] if (f2_id is None or f1_id == f2_id) else [f1_id, f2_id]
        for fighter_id in fighter_ids:
            expected[(fighter_id, int(fight_id))] = _calc_pre_fight_features(state[fighter_id])

        f1_before, f1_after = elo_map_by_fight.get(
            (f1_id, int(fight_id)),
            elo_map_by_date.get((f1_id, fight_date), (1500.0, 1500.0)),
        )
        if f2_id is not None:
            f2_before, f2_after = elo_map_by_fight.get(
                (f2_id, int(fight_id)),
                elo_map_by_date.get((f2_id, fight_date), (1500.0, 1500.0)),
            )
        else:
            f2_before, f2_after = (1500.0, 1500.0)
        method_u = str(method or "").upper()
        f1_won = winner_id is not None and winner_id == f1_id
        f2_won = winner_id is not None and winner_id == f2_id
        f1_finish_win = f1_won and (("KO" in method_u) or ("SUB" in method_u))
        f2_finish_win = f2_won and (("KO" in method_u) or ("SUB" in method_u))
        _update_state_after_fight(state[f1_id], f1_won, f2_before, f1_after - f1_before, f1_finish_win)
        if f2_id is not None and f1_id != f2_id:
            _update_state_after_fight(state[f2_id], f2_won, f1_before, f2_after - f2_before, f2_finish_win)

    if not expected:
        print("Audit aborted: no recomputed entries.")
        cur.close()
        conn.close()
        return False

    # Sample random fights and evaluate both fighters where available.
    fight_ids = sorted({fid for _, fid in expected.keys()})
    rng = np.random.RandomState(42)
    sample_n = min(sample_fights, len(fight_ids))
    sampled_fights = set(rng.choice(fight_ids, size=sample_n, replace=False))
    sampled_pairs = [k for k in expected.keys() if k[1] in sampled_fights]

    checked_pairs = 0
    mismatches = []

    compare_keys = [
        ('avg_opp_last3', 'avgopponentelolast3'),
        ('avg_opp_last5', 'avgopponentelolast5'),
        ('max_opp_last5', 'maxopponentelolast5'),
        ('elo_change_last3', 'elochangelast3'),
        ('elo_velocity', 'elovelocity'),
        ('finish_rate_last3', 'finishratelast3'),
        ('finish_rate_last5', 'finishratelast5'),
        ('finish_rate_career', 'finishratecareer'),
        ('finish_rate_trending', 'finishratetrending'),
        ('opp_quality_trending', 'opponentqualitytrending'),
    ]

    for fighter_id, fight_id in sampled_pairs:
        recomputed = expected[(fighter_id, fight_id)]
        cur.execute("""
                SELECT avgopponentelolast3, avgopponentelolast5, maxopponentelolast5,
                       elochangelast3, elovelocity,
                       finishratelast3, finishratelast5, finishratecareer, finishratetrending,
                       currentwinstreak, currentlossstreak, opponentqualitytrending, haspriordata
                FROM fighterhistoricalfeatures
                WHERE fighterid = %s AND fightid = %s
                LIMIT 1
            """, (fighter_id, fight_id))
        row = cur.fetchone()
        checked_pairs += 1

        if row is None:
            mismatches.append((fight_id, fighter_id, "missing_row", None, None))
            continue

        stored = {
            'avgopponentelolast3': float(row[0]) if row[0] is not None else None,
            'avgopponentelolast5': float(row[1]) if row[1] is not None else None,
            'maxopponentelolast5': float(row[2]) if row[2] is not None else None,
            'elochangelast3': float(row[3]) if row[3] is not None else None,
            'elovelocity': float(row[4]) if row[4] is not None else None,
            'finishratelast3': float(row[5]) if row[5] is not None else None,
            'finishratelast5': float(row[6]) if row[6] is not None else None,
            'finishratecareer': float(row[7]) if row[7] is not None else None,
            'finishratetrending': float(row[8]) if row[8] is not None else None,
            'currentwinstreak': int(row[9]) if row[9] is not None else 0,
            'currentlossstreak': int(row[10]) if row[10] is not None else 0,
            'opponentqualitytrending': float(row[11]) if row[11] is not None else None,
            'haspriordata': bool(row[12]) if row[12] is not None else False,
        }

        for recompute_key, stored_key in compare_keys:
            rv = recomputed[recompute_key]
            sv = stored[stored_key]
            if rv is None and sv is None:
                continue
            if (rv is None) != (sv is None):
                mismatches.append((fight_id, fighter_id, recompute_key, rv, sv))
                continue
            if abs(float(rv) - float(sv)) > tolerance:
                mismatches.append((fight_id, fighter_id, recompute_key, rv, sv))

        if int(recomputed['current_win_streak']) != int(stored['currentwinstreak']):
            mismatches.append((fight_id, fighter_id, 'current_win_streak',
                               recomputed['current_win_streak'], stored['currentwinstreak']))
        if int(recomputed['current_loss_streak']) != int(stored['currentlossstreak']):
            mismatches.append((fight_id, fighter_id, 'current_loss_streak',
                               recomputed['current_loss_streak'], stored['currentlossstreak']))
        if bool(recomputed['has_prior_data']) != bool(stored['haspriordata']):
            mismatches.append((fight_id, fighter_id, 'has_prior_data',
                               recomputed['has_prior_data'], stored['haspriordata']))

    print(f"  Checked fighter-fight pairs: {checked_pairs}")
    print(f"  Mismatches found: {len(mismatches)}")
    if mismatches:
        print("  First mismatches:")
        for m in mismatches[:10]:
            print(f"    fight_id={m[0]} fighter_id={m[1]} field={m[2]} recomputed={m[3]} stored={m[4]}")
    else:
        print("  ✅ Leakage audit passed: sampled rows match recomputation and no future data was used.")

    cur.close()
    conn.close()
    return len(mismatches) == 0


def build_historical_features(rebuild=True, batch_size=4000):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    print("Building FighterHistoricalFeatures...")

    cur.execute(CREATE_TABLE_SQL)
    if rebuild:
        cur.execute("TRUNCATE TABLE FighterHistoricalFeatures")
        conn.commit()

    # One canonical row per fight sorted by date.
    # Prefer winner-perspective rows when both directions exist, but do not require them.
    cur.execute("""
        SELECT DISTINCT ON (f.fightid)
            f.fightid, f.date, f.fighterid, f.opponentid, f.winnerid, f.result, f.method
        FROM fights f
        WHERE f.date IS NOT NULL
          AND f.fighterid IS NOT NULL
          AND (f.winnerid IS NOT NULL OR f.result IN ('win', 'loss'))
        ORDER BY f.fightid,
                 CASE WHEN f.fighterid = f.winnerid THEN 0 ELSE 1 END,
                 f.date
    """)
    fights = cur.fetchall()
    fights.sort(key=lambda r: (r[1], r[0]))

    cur.execute("""
        SELECT fighterid, fightid, fightdate, elobeforefight, eloafterfight
        FROM elohistory
    """)
    elo_rows = cur.fetchall()
    elo_map_by_fight = {}
    elo_map_by_date = {}
    for fid, efight_id, fdate, before, after in elo_rows:
        if efight_id is not None:
            elo_map_by_fight[(int(fid), int(efight_id))] = (
                float(before) if before is not None else 1500.0,
                float(after) if after is not None else float(before) if before is not None else 1500.0,
            )
        elo_map_by_date[(int(fid), fdate)] = (
            float(before) if before is not None else 1500.0,
            float(after) if after is not None else float(before) if before is not None else 1500.0,
        )

    state = defaultdict(lambda: {
        'opp_elo_hist': [],
        'elo_change_hist': [],
        'finish_win_hist': [],
        'wins_total': 0,
        'finish_wins_total': 0,
        'fights_total': 0,
        'current_win_streak': 0,
        'current_loss_streak': 0,
    })

    records = []
    insert_sql = """
        INSERT INTO FighterHistoricalFeatures (
            FighterID, FightID, FightDate,
            AvgOpponentEloLast3, AvgOpponentEloLast5, MaxOpponentEloLast5,
            EloChangeLast3, EloVelocity,
            FinishRateLast3, FinishRateLast5, FinishRateCareer, FinishRateTrending,
            CurrentWinStreak, CurrentLossStreak,
            OpponentQualityTrending, HasPriorData
        ) VALUES %s
        ON CONFLICT (FighterID, FightID) DO UPDATE SET
            FightDate = EXCLUDED.FightDate,
            AvgOpponentEloLast3 = EXCLUDED.AvgOpponentEloLast3,
            AvgOpponentEloLast5 = EXCLUDED.AvgOpponentEloLast5,
            MaxOpponentEloLast5 = EXCLUDED.MaxOpponentEloLast5,
            EloChangeLast3 = EXCLUDED.EloChangeLast3,
            EloVelocity = EXCLUDED.EloVelocity,
            FinishRateLast3 = EXCLUDED.FinishRateLast3,
            FinishRateLast5 = EXCLUDED.FinishRateLast5,
            FinishRateCareer = EXCLUDED.FinishRateCareer,
            FinishRateTrending = EXCLUDED.FinishRateTrending,
            CurrentWinStreak = EXCLUDED.CurrentWinStreak,
            CurrentLossStreak = EXCLUDED.CurrentLossStreak,
            OpponentQualityTrending = EXCLUDED.OpponentQualityTrending,
            HasPriorData = EXCLUDED.HasPriorData,
            CalculatedAt = CURRENT_TIMESTAMP
    """

    def flush():
        if not records:
            return
        execute_values(cur, insert_sql, records, page_size=1000)
        conn.commit()
        records.clear()

    for idx, (fight_id, fight_date, f1_id, f2_id, winner_id, result, method) in enumerate(fights, 1):
        f1_id = int(f1_id)
        f2_id = int(f2_id) if f2_id is not None else None
        if winner_id is not None:
            winner_id = int(winner_id)
        else:
            result_s = str(result or "").strip().lower()
            if result_s == 'win':
                winner_id = f1_id
            elif result_s == 'loss' and f2_id is not None:
                winner_id = f2_id
            else:
                winner_id = None

        # Compute pre-fight features for both sides before state update.
        # Guard against malformed rows where fighter_id == opponent_id.
        fighter_ids = [f1_id] if (f2_id is None or f1_id == f2_id) else [f1_id, f2_id]
        for fighter_id in fighter_ids:
            feats = _calc_pre_fight_features(state[fighter_id])
            records.append((
                fighter_id, int(fight_id), fight_date,
                feats['avg_opp_last3'], feats['avg_opp_last5'], feats['max_opp_last5'],
                feats['elo_change_last3'], feats['elo_velocity'],
                feats['finish_rate_last3'], feats['finish_rate_last5'],
                feats['finish_rate_career'], feats['finish_rate_trending'],
                feats['current_win_streak'], feats['current_loss_streak'],
                feats['opp_quality_trending'], feats['has_prior_data'],
            ))

        # Update states with this fight outcome.
        f1_before, f1_after = elo_map_by_fight.get(
            (f1_id, int(fight_id)),
            elo_map_by_date.get((f1_id, fight_date), (1500.0, 1500.0)),
        )
        if f2_id is not None:
            f2_before, f2_after = elo_map_by_fight.get(
                (f2_id, int(fight_id)),
                elo_map_by_date.get((f2_id, fight_date), (1500.0, 1500.0)),
            )
        else:
            f2_before, f2_after = (1500.0, 1500.0)
        f1_elo_change = f1_after - f1_before
        f2_elo_change = f2_after - f2_before
        f1_won = winner_id is not None and winner_id == f1_id
        f2_won = winner_id is not None and winner_id == f2_id
        method_u = str(method or "").upper()
        f1_finish_win = f1_won and (("KO" in method_u) or ("SUB" in method_u))
        f2_finish_win = f2_won and (("KO" in method_u) or ("SUB" in method_u))

        _update_state_after_fight(state[f1_id], f1_won, f2_before, f1_elo_change, f1_finish_win)
        if f2_id is not None and f1_id != f2_id:
            _update_state_after_fight(state[f2_id], f2_won, f1_before, f2_elo_change, f2_finish_win)

        if idx % batch_size == 0:
            flush()
            print(f"  processed {idx}/{len(fights)} fights...")

    flush()

    cur.execute("SELECT COUNT(*) FROM FighterHistoricalFeatures")
    total_rows = cur.fetchone()[0]
    print(f"Done. FighterHistoricalFeatures rows: {total_rows}")

    cur.close()
    conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build point-in-time fighter historical features")
    parser.add_argument('--no-rebuild', action='store_true',
                        help='Do not truncate existing FighterHistoricalFeatures rows first')
    parser.add_argument('--batch-size', type=int, default=4000,
                        help='Commit every N fights (default: 4000)')
    parser.add_argument('--audit-sample', type=int, default=0,
                        help='Run leakage audit on N random fights after build')
    parser.add_argument('--audit-only', action='store_true',
                        help='Skip build and run leakage audit only')
    parser.add_argument('--audit-tolerance', type=float, default=0.02,
                        help='Numeric tolerance for audit comparisons (default: 0.02)')
    args = parser.parse_args()
    if not args.audit_only:
        build_historical_features(rebuild=not args.no_rebuild, batch_size=args.batch_size)
    if args.audit_sample > 0 or args.audit_only:
        sample_n = args.audit_sample if args.audit_sample > 0 else 50
        ok = run_leakage_audit(sample_fights=sample_n, tolerance=args.audit_tolerance)
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

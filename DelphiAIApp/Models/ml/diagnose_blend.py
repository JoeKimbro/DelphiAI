"""
One-off diagnostic: isolate why the production ML+ELO blend underperforms
the direct model on the 2026 holdout set.

For every holdout fight, replays the exact production prediction path
(point-in-time ELO/PIT/historical-feature overrides + _predict_single_fight)
but also records the raw ML-only probability and the pure-ELO-only
probability that feed into the blend, so accuracy can be compared
side-by-side: ELO-only vs ML-only vs blended.

Run from DelphiAIApp/Models/:
    python -m ml.diagnose_blend
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.metrics import accuracy_score

from ml.train_model_v3 import connect_db
from ml.backtest import (
    get_historical_elo, get_pit_stats, get_historical_features,
    _apply_pit_override, _apply_historical_features_override,
)
from ml.predict_fight import (
    get_fighter_data, elo_to_probability, _get_ml_predictor,
    recalibrate_probabilities,
)


def load_holdout_fights(conn):
    query = """
        SELECT f.FightID as fightid, f.Date as fight_date,
               f.FighterID as fighter1_id, f.FighterName as fighter1_name,
               f.OpponentID as fighter2_id, f.OpponentName as fighter2_name,
               f.WinnerID as winnerid, f.IsTitleFight as istitlefight
        FROM Fights f
        WHERE f.Date >= '2026-01-01'
          AND f.Result = 'win'
          AND f.FighterID IS NOT NULL AND f.OpponentID IS NOT NULL
        ORDER BY f.Date
    """
    df = pd.read_sql(query, conn)
    df = df.drop_duplicates(subset=['fightid'])
    return df


def diagnose_fight(conn, fight):
    f1 = get_fighter_data(conn, fight['fighter1_name'])
    f2 = get_fighter_data(conn, fight['fighter2_name'])
    if not f1 or not f2:
        return None

    fight_date = fight['fight_date']
    hist_elo_f1 = get_historical_elo(conn, fight['fighter1_id'], fight_date)
    hist_elo_f2 = get_historical_elo(conn, fight['fighter2_id'], fight_date)
    if hist_elo_f1 is not None:
        f1['elo'] = hist_elo_f1
    if hist_elo_f2 is not None:
        f2['elo'] = hist_elo_f2

    pit_f1 = get_pit_stats(conn, fight['fighter1_id'], fight_date)
    pit_f2 = get_pit_stats(conn, fight['fighter2_id'], fight_date)
    _apply_pit_override(f1, pit_f1)
    _apply_pit_override(f2, pit_f2)
    hist_f1 = get_historical_features(conn, fight['fighter1_id'], fight['fightid'])
    hist_f2 = get_historical_features(conn, fight['fighter2_id'], fight['fightid'])
    _apply_historical_features_override(f1, hist_f1)
    _apply_historical_features_override(f2, hist_f2)

    is_title = bool(fight.get('istitlefight'))

    f1_elo = f1['elo']
    f2_elo = f2['elo']
    elo_prob_f1 = elo_to_probability(f1_elo, f2_elo)

    ml_predictor = _get_ml_predictor()
    ml_prob_f1 = None
    sym_error = None
    ml_weight = None
    nan_count = None
    total_features = None

    if ml_predictor and ml_predictor.is_available():
        f1_ml = dict(f1)
        f2_ml = dict(f2)
        f1_ml['final_elo'] = f1_elo
        f2_ml['final_elo'] = f2_elo
        try:
            ml_result = ml_predictor.predict(f1_ml, f2_ml, is_title_fight=is_title)
        except Exception:
            ml_result = None

        if ml_result:
            features = ml_result.get('features', {})
            nan_count = sum(1 for v in features.values()
                             if v is None or (isinstance(v, float) and math.isnan(v)))
            total_features = max(len(features), 1)

            f1_swap = dict(f2)
            f2_swap = dict(f1)
            f1_swap['final_elo'] = f2_elo
            f2_swap['final_elo'] = f1_elo
            try:
                swap_result = ml_predictor.predict(f1_swap, f2_swap, is_title_fight=is_title)
                swap_prob = swap_result['prob_a'] if swap_result else None
            except Exception:
                swap_result = None
                swap_prob = None

            if swap_prob is not None:
                fwd_prob = ml_result.get('raw_prob_a', ml_result['prob_a'])
                rev_prob = swap_result.get('raw_prob_a', swap_prob) if swap_result else swap_prob
                sym_error = abs(fwd_prob + rev_prob - 1.0)
                corrected = (fwd_prob + (1.0 - rev_prob)) / 2.0
                ml_prob_f1 = corrected

                if sym_error <= 0.60:
                    missing_ratio = nan_count / total_features
                    missing_penalty = min(0.35, missing_ratio * 0.50)
                    ml_weight = 0.50 - missing_penalty - max(0.0, sym_error - 0.20) * 0.60
                    ml_weight = max(0.15, min(0.50, ml_weight))

    if ml_weight is not None and ml_prob_f1 is not None:
        blended = ml_weight * ml_prob_f1 + (1 - ml_weight) * elo_prob_f1
        blended = max(0.10, min(0.90, blended))
    else:
        blended = elo_prob_f1

    blended = recalibrate_probabilities(blended)

    y_true = 1 if fight['winnerid'] == fight['fighter1_id'] else 0

    return {
        'fight_id': fight['fightid'],
        'fighter1': fight['fighter1_name'],
        'fighter2': fight['fighter2_name'],
        'y_true': y_true,
        'elo_prob_f1': elo_prob_f1,
        'ml_prob_f1': ml_prob_f1,
        'blended_prob_f1': blended,
        'ml_weight': ml_weight,
        'sym_error': sym_error,
        'nan_count': nan_count,
        'total_features': total_features,
    }


def main():
    conn = connect_db()
    holdout = load_holdout_fights(conn)
    print(f"Loaded {len(holdout)} holdout fights (2026+)")

    records = []
    skipped = 0
    for _, fight in holdout.iterrows():
        rec = diagnose_fight(conn, fight)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    conn.close()

    df = pd.DataFrame(records)
    print(f"\nEvaluated {len(df)} fights, skipped {skipped}")

    def acc(prob_col):
        sub = df.dropna(subset=[prob_col])
        if len(sub) == 0:
            return None, 0
        preds = (sub[prob_col] > 0.5).astype(int)
        return accuracy_score(sub['y_true'], preds), len(sub)

    elo_acc, elo_n = acc('elo_prob_f1')
    ml_acc, ml_n = acc('ml_prob_f1')
    blend_acc, blend_n = acc('blended_prob_f1')

    print("\n=== ACCURACY COMPARISON ===")
    print(f"  ELO-only:       {elo_acc:.3f}  (n={elo_n})")
    if ml_acc is not None:
        print(f"  ML-only (corr): {ml_acc:.3f}  (n={ml_n})")
    else:
        print("  ML-only (corr): N/A (no fights had usable ML predictions)")
    print(f"  Blended:        {blend_acc:.3f}  (n={blend_n})")

    has_ml = df.dropna(subset=['ml_prob_f1'])
    print(f"\nFights with usable ML prediction: {len(has_ml)}/{len(df)} "
          f"({len(has_ml) / len(df):.0%})")
    if len(has_ml) > 0:
        print(f"  Mean ml_weight:   {has_ml['ml_weight'].mean():.3f}")
        print(f"  Mean sym_error:   {has_ml['sym_error'].mean():.3f}")
        print(f"  Mean nan_ratio:   {(has_ml['nan_count'] / has_ml['total_features']).mean():.3f}")
        print(f"  ml_weight at floor (0.15): {(has_ml['ml_weight'] <= 0.15).sum()}")
        print(f"  ml_weight at ceil (0.50):  {(has_ml['ml_weight'] >= 0.50).sum()}")

    fell_back = df['ml_prob_f1'].isna().sum()
    print(f"\nFights that fell back to pure ELO (no ML prob at all): {fell_back}/{len(df)}")

    out_path = Path(__file__).parent / 'artifacts' / 'blend_diagnostic.csv'
    df.to_csv(out_path, index=False)
    print(f"\nFull per-fight records written to {out_path}")


if __name__ == '__main__':
    main()

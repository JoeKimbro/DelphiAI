"""
Historical Backtest Runner

Runs the prediction pipeline against past fights with known outcomes
to validate the model before trusting it with live predictions.

Data integrity measures:
  - Uses historical ELO from EloHistory (not current ELO)
  - Uses PointInTimeStats for career stats when available (no leakage)
  - Falls back to current stats with explicit leakage warnings
  - Pre-flight diagnostics check data coverage before running
  - Post-flight leakage detection flags suspiciously high accuracy

The model was trained on pre-2024 data, so 2025 is true out-of-sample.

Usage:
    cd DelphiAIApp/Models

    # Backtest all 2025 fights (default)
    python -m ml.backtest

    # Specific year
    python -m ml.backtest --year 2024

    # Multiple years with combined summary
    python -m ml.backtest --years 2023 2024 2025

    # Date range
    python -m ml.backtest --start 2025-01-01 --end 2025-07-01

    # Single event
    python -m ml.backtest --event "UFC 311"

    # Clear previous backtest data and re-run
    python -m ml.backtest --clear
"""

import os
import sys
import re
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import psycopg2
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

logger = logging.getLogger(__name__)

# Realistic accuracy ceiling for MMA prediction models.
# Anything above this almost certainly indicates data leakage.
LEAKAGE_THRESHOLD = 75.0


# ============================================================================
# PRE-FLIGHT DIAGNOSTICS
# ============================================================================

def _run_diagnostics(conn, start_date, end_date):
    """
    Check data availability before running backtest.
    Returns a diagnostics dict and prints warnings.
    """
    cur = conn.cursor()
    W = 80

    print(f"\n  PRE-FLIGHT DATA CHECK:")
    print(f"  {'-' * (W - 4)}")

    # 1. Count fights in range
    cur.execute("""
        SELECT COUNT(DISTINCT (LEAST(fighterid, opponentid),
                               GREATEST(fighterid, opponentid), date))
        FROM fights
        WHERE date >= %s AND date < %s
          AND result = 'win'
          AND fighterid IS NOT NULL AND opponentid IS NOT NULL
    """, (start_date, end_date))
    fight_count = cur.fetchone()[0] or 0

    # 2. EloHistory coverage
    cur.execute("""
        SELECT COUNT(DISTINCT fighterid), COUNT(*)
        FROM elohistory
        WHERE fightdate >= %s AND fightdate < %s
    """, (start_date, end_date))
    elo_row = cur.fetchone()
    elo_fighters = elo_row[0] or 0
    elo_records = elo_row[1] or 0

    # 3. PointInTimeStats coverage
    pit_fighters = 0
    pit_records = 0
    try:
        cur.execute("""
            SELECT COUNT(DISTINCT fighterid), COUNT(*)
            FROM pointintimestats
            WHERE fightdate >= %s AND fightdate < %s
        """, (start_date, end_date))
        pit_row = cur.fetchone()
        pit_fighters = pit_row[0] or 0
        pit_records = pit_row[1] or 0
    except Exception:
        conn.rollback()

    # 4. Check if model was trained on this period
    # Training data is pre-2024; 2024 is validation; 2025+ is holdout.
    try:
        year_start = int(start_date[:4])
    except (ValueError, IndexError):
        year_start = 2025

    trained_on_period = year_start < 2024

    cur.close()

    # Print results
    elo_pct = (elo_records / (fight_count * 2) * 100) if fight_count > 0 else 0
    pit_pct = (pit_records / (fight_count * 2) * 100) if fight_count > 0 else 0

    fight_icon = "\u2705" if fight_count >= 50 else "\u26a0\ufe0f" if fight_count >= 20 else "\u274c"
    elo_icon = "\u2705" if elo_pct >= 80 else "\u26a0\ufe0f" if elo_pct >= 40 else "\u274c"
    pit_icon = "\u2705" if pit_pct >= 80 else "\u26a0\ufe0f" if pit_pct >= 40 else "\u274c"

    print(f"    {fight_icon} Fights in range:     {fight_count}")
    print(f"    {elo_icon} EloHistory coverage:  {elo_records} records / {elo_fighters} fighters ({elo_pct:.0f}%)")
    print(f"    {pit_icon} PointInTimeStats:     {pit_records} records / {pit_fighters} fighters ({pit_pct:.0f}%)")

    if trained_on_period:
        print(f"    \u274c TRAINING DATA OVERLAP: {start_date[:4]} is in the training set (pre-2024)")
        print(f"       Results will be inflated! Use --year 2025 for valid out-of-sample test.")
    else:
        print(f"    \u2705 Out-of-sample:        {start_date[:4]} not in training data (trained pre-2024)")

    warnings = []

    if elo_pct < 40:
        warnings.append("EloHistory has low coverage - many fights will use CURRENT ELO (leakage)")
    if pit_pct < 10:
        warnings.append("PointInTimeStats missing - career stats will use CURRENT values (leakage)")
    if trained_on_period:
        warnings.append("Backtesting on training data - results are NOT valid for model evaluation")
    if fight_count < 20:
        warnings.append(f"Only {fight_count} fights - too few for reliable statistics")

    if warnings:
        print(f"\n  \u26a0\ufe0f  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")

    diag = {
        'fight_count': fight_count,
        'elo_fighters': elo_fighters,
        'elo_records': elo_records,
        'elo_coverage_pct': elo_pct,
        'pit_fighters': pit_fighters,
        'pit_records': pit_records,
        'pit_coverage_pct': pit_pct,
        'trained_on_period': trained_on_period,
        'warnings': warnings,
    }

    return diag


# ============================================================================
# HISTORICAL DATA RETRIEVAL
# ============================================================================

def get_historical_fights(conn, start_date, end_date, event_filter=None):
    """
    Get all completed fights in a date range from the Fights table.
    Returns one row per fight (winner's perspective) to avoid duplicates.
    """
    cur = conn.cursor()

    query = """
        SELECT f.fightid, f.fightername, f.opponentname, f.fighterid, f.opponentid,
               f.winnername, f.winnerid, f.method, f.round,
               f.date, f.eventname, f.istitlefight
        FROM fights f
        WHERE f.date >= %s AND f.date < %s
          AND f.result = 'win'
          AND f.fighterid IS NOT NULL
          AND f.opponentid IS NOT NULL
    """
    params = [start_date, end_date]

    if event_filter:
        query += " AND f.eventname ILIKE %s"
        params.append(f'%{event_filter}%')

    query += " ORDER BY f.date ASC, f.eventname, f.fightid"

    cur.execute(query, params)

    fights = []
    seen_pairs = set()
    for row in cur.fetchall():
        pair_key = (min(row[3], row[4]), max(row[3], row[4]), row[9])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        fights.append({
            'fight_id': row[0],
            'fighter1_name': row[1],
            'fighter2_name': row[2],
            'fighter1_id': row[3],
            'fighter2_id': row[4],
            'winner_name': row[5],
            'winner_id': row[6],
            'method': row[7],
            'round': row[8],
            'date': row[9],
            'event_name': row[10],
            'is_title': bool(row[11]) if row[11] is not None else False,
        })

    cur.close()
    return fights


def get_historical_elo(conn, fighter_id, fight_date):
    """Get a fighter's ELO from BEFORE a specific fight (point-in-time)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT elobeforefight FROM elohistory
        WHERE fighterid = %s AND fightdate = %s
        LIMIT 1
    """, (fighter_id, fight_date))
    row = cur.fetchone()
    cur.close()
    return float(row[0]) if row else None


def get_pit_stats(conn, fighter_id, fight_date):
    """
    Get PointInTimeStats for a fighter BEFORE a specific fight.
    These are career stats computed only from fights prior to this date,
    eliminating look-ahead bias in features like SLpM, TDAvg, etc.

    Returns: dict of stat overrides, or None if no PIT data exists.
    """
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT pit_slpm, pit_stracc, pit_tdavg, pit_subavg, pit_kdrate,
                   recentwinrate, avgfighttime, finishrate,
                   fightsbefore, winsbefore, lossesbefore, winratebefore,
                   haspriordata
            FROM pointintimestats
            WHERE fighterid = %s AND fightdate = %s
            LIMIT 1
        """, (fighter_id, fight_date))
        row = cur.fetchone()
    except Exception:
        conn.rollback()
        cur.close()
        return None

    cur.close()

    if not row or not row[12]:  # haspriordata = False means debut, no useful stats
        return None

    return {
        'slpm': float(row[0]) if row[0] is not None else None,
        'str_acc': float(row[1]) if row[1] is not None else None,
        'td_avg': float(row[2]) if row[2] is not None else None,
        'sub_avg': float(row[3]) if row[3] is not None else None,
        'kd_rate': float(row[4]) if row[4] is not None else None,
        'recent_form': float(row[5]) if row[5] is not None else None,
        'avg_fight_duration': float(row[6]) if row[6] is not None else None,
        'finish_rate': float(row[7]) if row[7] is not None else None,
        'total_fights': int(row[8]) if row[8] is not None else None,
        'wins': int(row[9]) if row[9] is not None else None,
        'losses': int(row[10]) if row[10] is not None else None,
    }


def get_historical_features(conn, fighter_id, fight_id):
    """
    Get FighterHistoricalFeatures for a fighter at this exact fight (point-in-time).
    """
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT avgopponentelolast3, elovelocity, currentwinstreak,
                   finishratetrending, opponentqualitytrending, haspriordata
            FROM fighterhistoricalfeatures
            WHERE fighterid = %s AND fightid = %s
            LIMIT 1
        """, (fighter_id, fight_id))
        row = cur.fetchone()
    except Exception:
        conn.rollback()
        cur.close()
        return None
    cur.close()

    if not row:
        return None

    return {
        'avg_opponent_elo_last_3': float(row[0]) if row[0] is not None else None,
        'elo_velocity': float(row[1]) if row[1] is not None else None,
        'current_win_streak': int(row[2]) if row[2] is not None else None,
        'finish_rate_trending': float(row[3]) if row[3] is not None else None,
        'opponent_quality_trending': float(row[4]) if row[4] is not None else None,
        'has_prior_data': bool(row[5]) if row[5] is not None else False,
    }


def _apply_pit_override(fighter_data, pit_stats):
    """
    Override current career stats with point-in-time stats.
    Only overrides fields where PIT data is available and non-null.
    """
    if not pit_stats:
        return False

    applied = False
    # Numeric fields that map directly
    for key in ['slpm', 'td_avg', 'sub_avg', 'kd_rate', 'recent_form',
                'avg_fight_duration', 'finish_rate']:
        if pit_stats.get(key) is not None:
            fighter_data[key] = pit_stats[key]
            applied = True

    # str_acc needs to match the expected format (could be Decimal or "XX%")
    if pit_stats.get('str_acc') is not None:
        fighter_data['str_acc'] = pit_stats['str_acc']
        applied = True

    # Win/loss record
    if pit_stats.get('wins') is not None:
        fighter_data['wins'] = pit_stats['wins']
        applied = True
    if pit_stats.get('losses') is not None:
        fighter_data['losses'] = pit_stats['losses']
        applied = True

    return applied


def _apply_historical_features_override(fighter_data, hist_features):
    """
    Apply point-in-time historical trend features to fighter_data.
    """
    if not hist_features:
        return False

    applied = False
    for key in [
        'avg_opponent_elo_last_3',
        'elo_velocity',
        'current_win_streak',
        'finish_rate_trending',
        'opponent_quality_trending',
    ]:
        if hist_features.get(key) is not None:
            fighter_data[key] = hist_features[key]
            applied = True
    return applied


# ============================================================================
# PREDICTION ENGINE (slim version of predict_card logic)
# ============================================================================

def _predict_single_fight(conn, f1_data, f2_data, is_title=False):
    """
    Run the ML+ELO prediction pipeline for a single fight.
    Mirrors the core blending logic from predict_card but without
    printing, injury scraping, or other side effects.
    """
    try:
        from ml.predict_fight import (
            elo_to_probability, _get_ml_predictor, predict_method,
            recalibrate_probabilities,
        )
    except (ModuleNotFoundError, ImportError):
        from predict_fight import (
            elo_to_probability, _get_ml_predictor, predict_method,
            recalibrate_probabilities,
        )

    f1_elo = f1_data['elo']
    f2_elo = f2_data['elo']
    elo_prob_f1 = elo_to_probability(f1_elo, f2_elo)

    ml_predictor = _get_ml_predictor()
    ml_used = False
    prob_source = "ELO"
    adj_prob_f1 = elo_prob_f1

    if ml_predictor and ml_predictor.is_available():
        f1_ml = dict(f1_data)
        f2_ml = dict(f2_data)
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
            max_allowed_nan = max(6, int(total_features * 0.45))

            if nan_count <= max_allowed_nan:
                f1_swap = dict(f2_data)
                f2_swap = dict(f1_data)
                f1_swap['final_elo'] = f2_elo
                f2_swap['final_elo'] = f1_elo
                try:
                    swap_result = ml_predictor.predict(f1_swap, f2_swap, is_title_fight=is_title)
                    swap_prob = swap_result['prob_a'] if swap_result else None
                except Exception:
                    swap_result = None
                    swap_prob = None

                if swap_prob is not None:
                    # Mirror predict_card: use raw probs for symmetry when available.
                    fwd_prob = ml_result.get('raw_prob_a', ml_result['prob_a'])
                    rev_prob = swap_result.get('raw_prob_a', swap_prob) if swap_result else swap_prob
                    sym_error = abs(fwd_prob + rev_prob - 1.0)
                    corrected = (fwd_prob + (1.0 - rev_prob)) / 2.0

                    # Mirror predict_card: tolerate moderate drift with reduced ML weight.
                    if sym_error <= 0.60:
                        missing_ratio = nan_count / total_features
                        missing_penalty = min(0.35, missing_ratio * 0.50)
                        ml_weight = 0.50 - missing_penalty - max(0.0, sym_error - 0.20) * 0.60
                        ml_weight = max(0.15, min(0.50, ml_weight))
                        blended = ml_weight * corrected + (1 - ml_weight) * elo_prob_f1
                        blended = max(0.10, min(0.90, blended))
                        adj_prob_f1 = blended
                        ml_used = True
                        prob_source = f"ML+ELO ({ml_weight:.0%} ML)"

    if not ml_used:
        adj_prob_f1 = elo_prob_f1

    adj_prob_f1 = recalibrate_probabilities(adj_prob_f1)
    adj_prob_f2 = 1 - adj_prob_f1

    if adj_prob_f1 > adj_prob_f2:
        pick, pick_pct = f1_data['name'], adj_prob_f1
    else:
        pick, pick_pct = f2_data['name'], adj_prob_f2

    if pick_pct >= 0.70:
        confidence = "HIGH"
    elif pick_pct >= 0.60:
        confidence = "MED"
    elif pick_pct >= 0.55:
        confidence = "LOW"
    else:
        confidence = "TOSS"

    method = predict_method(f1_data, f2_data, adj_prob_f1, adj_prob_f2)
    all_methods = {'KO/TKO': method['ko'], 'Sub': method['sub'], 'Dec': method['dec']}
    best_method = max(all_methods, key=all_methods.get)

    return {
        'fighter1': f1_data['name'],
        'fighter2': f2_data['name'],
        'f1_id': f1_data.get('id'),
        'f2_id': f2_data.get('id'),
        'f1_elo': f1_elo,
        'f2_elo': f2_elo,
        'f1_prob': adj_prob_f1,
        'f2_prob': adj_prob_f2,
        'pick': pick,
        'pick_pct': pick_pct,
        'confidence': confidence,
        'method': best_method,
        'prob_source': prob_source,
        'is_title': is_title,
        'ko_pct': method['ko'],
        'sub_pct': method['sub'],
        'dec_pct': method['dec'],
        'r1_finish': method['r1_finish'],
        'r2_finish': method['r2_finish'],
        'r3_finish': method['r3_finish'],
    }


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def run_backtest(conn, start_date, end_date, event_filter=None, clear=False):
    """
    Run the full backtest pipeline with data integrity checks.

    1. Pre-flight diagnostics (EloHistory, PIT coverage)
    2. For each fight: historical ELO + PIT stats override
    3. Run prediction, compare to actual
    4. Save to DB with type='backtest'
    5. Report with leakage detection
    """
    try:
        from ml.predict_fight import get_fighter_data
    except (ModuleNotFoundError, ImportError):
        from predict_fight import get_fighter_data

    try:
        from ml.update_results import ensure_tracking_table, save_predictions
    except (ModuleNotFoundError, ImportError):
        from update_results import ensure_tracking_table, save_predictions

    ensure_tracking_table(conn)

    if clear:
        cur = conn.cursor()
        cur.execute("DELETE FROM PredictionTracking WHERE prediction_type = 'backtest'")
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        if deleted:
            print(f"  Cleared {deleted} previous backtest predictions")

    W = 80
    print("\n" + "=" * W)
    print(f"{'DELPHI AI - HISTORICAL BACKTEST':^{W}}")
    print("=" * W)
    print(f"  Period: {start_date} to {end_date}")
    if event_filter:
        print(f"  Filter: {event_filter}")

    # Pre-flight diagnostics
    diag = _run_diagnostics(conn, start_date, end_date)

    if diag['fight_count'] == 0:
        print(f"\n  No fights found in date range. Aborting.")
        return []

    # Load historical fights
    fights = get_historical_fights(conn, start_date, end_date, event_filter)
    if not fights:
        print(f"\n  No fights matched filters.")
        return []

    events = defaultdict(list)
    for f in fights:
        events[f['event_name']].append(f)

    print(f"\n  Processing {len(fights)} fights across {len(events)} events...")
    print()

    # Coverage tracking
    coverage = {
        'elo_historical': 0,   # Used EloHistory (no leakage)
        'elo_current': 0,      # Fell back to current ELO (leakage!)
        'pit_used': 0,         # Used PIT stats (no leakage)
        'pit_missing': 0,      # Used current career stats (leakage!)
        'hist_used': 0,        # Used FighterHistoricalFeatures (no leakage)
        'hist_missing': 0,     # Missing historical trend features
    }

    all_results = []
    event_summaries = []
    skipped = 0

    for event_idx, (event_name, event_fights) in enumerate(events.items(), 1):
        event_date = event_fights[0]['date']
        event_correct = 0
        event_total = 0
        event_preds = []

        for fight in event_fights:
            f1 = get_fighter_data(conn, fight['fighter1_name'])
            f2 = get_fighter_data(conn, fight['fighter2_name'])

            if not f1 or not f2:
                skipped += 1
                continue

            # --- HISTORICAL ELO OVERRIDE ---
            hist_elo_f1 = get_historical_elo(conn, fight['fighter1_id'], fight['date'])
            hist_elo_f2 = get_historical_elo(conn, fight['fighter2_id'], fight['date'])

            if hist_elo_f1 is not None:
                f1['elo'] = hist_elo_f1
                coverage['elo_historical'] += 1
            else:
                coverage['elo_current'] += 1

            if hist_elo_f2 is not None:
                f2['elo'] = hist_elo_f2
                coverage['elo_historical'] += 1
            else:
                coverage['elo_current'] += 1

            # --- POINT-IN-TIME STATS OVERRIDE ---
            pit_f1 = get_pit_stats(conn, fight['fighter1_id'], fight['date'])
            pit_f2 = get_pit_stats(conn, fight['fighter2_id'], fight['date'])

            if _apply_pit_override(f1, pit_f1):
                coverage['pit_used'] += 1
            else:
                coverage['pit_missing'] += 1

            if _apply_pit_override(f2, pit_f2):
                coverage['pit_used'] += 1
            else:
                coverage['pit_missing'] += 1

            # --- HISTORICAL TREND FEATURES OVERRIDE ---
            hist_f1 = get_historical_features(conn, fight['fighter1_id'], fight['fight_id'])
            hist_f2 = get_historical_features(conn, fight['fighter2_id'], fight['fight_id'])
            if _apply_historical_features_override(f1, hist_f1):
                coverage['hist_used'] += 1
            else:
                coverage['hist_missing'] += 1
            if _apply_historical_features_override(f2, hist_f2):
                coverage['hist_used'] += 1
            else:
                coverage['hist_missing'] += 1

            # Run prediction
            pred = _predict_single_fight(conn, f1, f2, is_title=fight['is_title'])
            if not pred:
                skipped += 1
                continue

            # Compare to actual
            actual_winner = fight['winner_name']
            was_correct = False
            if actual_winner:
                pick_last = pred['pick'].lower().split()[-1]
                actual_last = actual_winner.lower().split()[-1]
                was_correct = (pick_last == actual_last)

            pred['actual_winner'] = actual_winner
            pred['actual_method'] = fight['method']
            pred['actual_round'] = fight['round']
            pred['was_correct'] = was_correct
            pred['event_name'] = event_name
            pred['event_date'] = str(event_date)

            event_preds.append(pred)
            all_results.append(pred)

            if was_correct:
                event_correct += 1
            event_total += 1

        if event_total > 0:
            acc = event_correct / event_total * 100
            icon = "\u2705" if acc >= 60 else "\u26a0\ufe0f" if acc >= 50 else "\u274c"
            short_name = event_name[:45] if len(event_name) > 45 else event_name
            print(f"  [{event_idx:>2}/{len(events)}] {short_name:<46} "
                  f"{event_correct}/{event_total} ({acc:.0f}%) {icon}")

            event_summaries.append({
                'name': event_name,
                'date': event_date,
                'total': event_total,
                'correct': event_correct,
                'accuracy': acc,
            })

        # Save to DB
        if event_preds:
            save_results = []
            for p in event_preds:
                save_results.append({
                    'fighter1': p['fighter1'],
                    'fighter2': p['fighter2'],
                    'f1_id': p.get('f1_id'),
                    'f2_id': p.get('f2_id'),
                    'f1_elo': p['f1_elo'],
                    'f2_elo': p['f2_elo'],
                    'f1_prob': p['f1_prob'],
                    'f2_prob': p['f2_prob'],
                    'pick': p['pick'],
                    'pick_pct': p['pick_pct'],
                    'confidence': p['confidence'],
                    'method': p['method'],
                    'prob_source': p['prob_source'],
                    'is_title': p['is_title'],
                    'ko_pct': p['ko_pct'],
                    'sub_pct': p['sub_pct'],
                    'dec_pct': p['dec_pct'],
                    'r1_finish': p['r1_finish'],
                    'r2_finish': p['r2_finish'],
                    'r3_finish': p['r3_finish'],
                })

            save_predictions(
                conn, save_results, event_name,
                event_date=str(event_date),
                prediction_type='backtest',
            )

            cur = conn.cursor()
            for p in event_preds:
                cur.execute("""
                    UPDATE PredictionTracking SET
                        actual_winner_name = %s,
                        actual_method = %s,
                        actual_round = %s,
                        was_correct = %s,
                        resolved_at = CURRENT_TIMESTAMP
                    WHERE event_name = %s
                      AND fighter1_name = %s
                      AND fighter2_name = %s
                      AND prediction_type = 'backtest'
                """, (
                    p['actual_winner'], p['actual_method'], p['actual_round'],
                    p['was_correct'],
                    event_name, p['fighter1'], p['fighter2'],
                ))
            conn.commit()
            cur.close()

    if skipped:
        print(f"\n  Skipped: {skipped} fights (fighters not in DB)")

    _print_backtest_report(all_results, event_summaries, start_date, end_date,
                           coverage, diag)

    return all_results


# ============================================================================
# BACKTEST REPORT
# ============================================================================

def _print_backtest_report(results, event_summaries, start_date, end_date,
                           coverage, diagnostics):
    """Print backtest report with data quality and leakage detection."""
    W = 80

    if not results:
        print(f"\n  No predictions were generated.")
        return

    total = len(results)
    correct = sum(1 for r in results if r['was_correct'])
    accuracy = correct / total * 100

    print("\n" + "=" * W)
    print(f"{'BACKTEST RESULTS':^{W}}")
    print("=" * W)

    # ---------- DATA QUALITY ----------
    total_fighters = coverage['elo_historical'] + coverage['elo_current']
    elo_clean_pct = coverage['elo_historical'] / total_fighters * 100 if total_fighters > 0 else 0
    total_stat_lookups = coverage['pit_used'] + coverage['pit_missing']
    pit_clean_pct = coverage['pit_used'] / total_stat_lookups * 100 if total_stat_lookups > 0 else 0
    total_hist_lookups = coverage['hist_used'] + coverage['hist_missing']
    hist_clean_pct = coverage['hist_used'] / total_hist_lookups * 100 if total_hist_lookups > 0 else 0

    print(f"\n  DATA QUALITY:")
    elo_icon = "\u2705" if elo_clean_pct >= 80 else "\u26a0\ufe0f" if elo_clean_pct >= 50 else "\u274c"
    pit_icon = "\u2705" if pit_clean_pct >= 80 else "\u26a0\ufe0f" if pit_clean_pct >= 50 else "\u274c"
    hist_icon = "\u2705" if hist_clean_pct >= 80 else "\u26a0\ufe0f" if hist_clean_pct >= 50 else "\u274c"

    print(f"    {elo_icon} ELO:   {coverage['elo_historical']} historical / {coverage['elo_current']} current ({elo_clean_pct:.0f}% clean)")
    print(f"    {pit_icon} Stats: {coverage['pit_used']} point-in-time / {coverage['pit_missing']} current ({pit_clean_pct:.0f}% clean)")
    print(f"    {hist_icon} Hist: {coverage['hist_used']} point-in-time / {coverage['hist_missing']} missing ({hist_clean_pct:.0f}% clean)")

    if coverage['elo_current'] > 0:
        print(f"         {coverage['elo_current']} fighters used current ELO (minor leakage)")
    if coverage['pit_missing'] > 0 and coverage['pit_used'] == 0:
        print(f"         All fighters used current career stats (leakage in features)")
        print(f"         Run PIT stats computation to fix: python -m data.build_pit_stats")
    if coverage['hist_missing'] > 0 and coverage['hist_used'] == 0:
        print(f"         Historical trend features missing (run builder script)")
        print(f"         Fix: python -m data.build_historical_features")

    # ---------- OVERALL ----------
    print(f"\n  OVERALL:")
    print(f"    Period:            {start_date} to {end_date}")
    print(f"    Events:            {len(event_summaries)}")
    print(f"    Total Predictions: {total}")
    print(f"    Correct:           {correct}")
    print(f"    Accuracy:          {accuracy:.1f}%")

    # ---------- LEAKAGE DETECTION ----------
    if accuracy > LEAKAGE_THRESHOLD:
        print(f"\n  \u274c LEAKAGE WARNING: {accuracy:.1f}% accuracy exceeds {LEAKAGE_THRESHOLD}% ceiling")
        print(f"    Real MMA prediction ceiling is ~68-72% for elite models.")
        print(f"    Likely causes:")
        if coverage['elo_current'] > total_fighters * 0.3:
            print(f"      - {coverage['elo_current']} fighters used CURRENT ELO (contains future results)")
        if coverage['pit_missing'] > total_stat_lookups * 0.5:
            print(f"      - {coverage['pit_missing']} fighters used CURRENT career stats (contains future fights)")
        if diagnostics.get('trained_on_period'):
            print(f"      - Backtesting on TRAINING DATA (model already saw these fights)")
        print(f"    Fix: Populate EloHistory and PointInTimeStats, or use --year 2025")

    # ---------- CONFIDENCE TIERS ----------
    tiers = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        tiers[r['confidence']]['total'] += 1
        if r['was_correct']:
            tiers[r['confidence']]['correct'] += 1

    print(f"\n  BY CONFIDENCE TIER:")
    for tier in ['HIGH', 'MED', 'LOW', 'TOSS']:
        t = tiers[tier]
        if t['total'] == 0:
            continue
        acc = t['correct'] / t['total'] * 100
        icon = "\u2705" if acc >= 65 else "\u26a0\ufe0f" if acc >= 50 else "\u274c"
        print(f"    {icon} {tier:<4} : {t['correct']}/{t['total']} ({acc:.1f}%)")

    # ---------- HIGH CONFIDENCE ----------
    high_conf = [r for r in results if r['pick_pct'] >= 0.65]
    hc_correct = 0
    hc_acc = 0
    if high_conf:
        hc_correct = sum(1 for r in high_conf if r['was_correct'])
        hc_acc = hc_correct / len(high_conf) * 100
        status = "EXCELLENT" if hc_acc >= 70 else "GOOD" if hc_acc >= 60 else "NEEDS REVIEW"
        status_icon = "\u2705" if hc_acc >= 65 else "\u26a0\ufe0f" if hc_acc >= 50 else "\u274c"

        print(f"\n  HIGH CONFIDENCE (\u226565%):")
        print(f"    Total:    {len(high_conf)}")
        print(f"    Correct:  {hc_correct}")
        print(f"    Accuracy: {hc_acc:.1f}%")
        print(f"    Status:   {status_icon} {status}")

    # ---------- CALIBRATION ----------
    buckets = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        prob = r['pick_pct']
        if prob >= 0.70:
            bucket = '70-100%'
        elif prob >= 0.65:
            bucket = '65-70%'
        elif prob >= 0.60:
            bucket = '60-65%'
        elif prob >= 0.55:
            bucket = '55-60%'
        else:
            bucket = '50-55%'
        buckets[bucket]['total'] += 1
        if r['was_correct']:
            buckets[bucket]['correct'] += 1

    print(f"\n  CALIBRATION CHECK:")
    print(f"    {'BUCKET':<12} {'PRED':>6} {'ACTUAL':>8} {'DELTA':>7} {'N':>4} {'STATUS':>8}")
    print(f"    {'-'*50}")
    for bucket in ['70-100%', '65-70%', '60-65%', '55-60%', '50-55%']:
        b = buckets[bucket]
        if b['total'] == 0:
            continue
        parts = bucket.replace('%', '').split('-')
        expected = (float(parts[0]) + float(parts[1])) / 2
        actual = b['correct'] / b['total'] * 100
        delta = actual - expected
        cal_ok = "GOOD" if abs(delta) <= 5 else ("OK" if abs(delta) <= 10 else "OFF")
        print(f"    {bucket:<12} {expected:>5.0f}% {actual:>7.1f}% {delta:>+6.1f}% {b['total']:>4} {cal_ok:>8}")

    # ---------- BY SOURCE ----------
    src_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        src = 'ML+ELO' if r['prob_source'].startswith('ML') else 'ELO'
        src_stats[src]['total'] += 1
        if r['was_correct']:
            src_stats[src]['correct'] += 1

    if len(src_stats) > 1:
        print(f"\n  BY PREDICTION SOURCE:")
        for src, d in src_stats.items():
            acc = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            print(f"    {src:<12}: {d['correct']}/{d['total']} ({acc:.1f}%)")

    # ---------- PAPER TRADING ----------
    STAKE = 100
    high_conf_bets = [r for r in results if r['pick_pct'] >= 0.65]
    total_wagered = len(high_conf_bets) * STAKE
    total_returned = 0
    for r in high_conf_bets:
        if r['was_correct']:
            decimal_odds = 1.0 / r['pick_pct'] if r['pick_pct'] > 0 else 1.0
            total_returned += STAKE * decimal_odds
    profit = total_returned - total_wagered
    roi = profit / total_wagered * 100 if total_wagered > 0 else 0

    print(f"\n  PAPER TRADING (65%+ picks, ${STAKE} flat bets, fair odds):")
    print(f"    Bets:     {len(high_conf_bets)}")
    print(f"    Wagered:  ${total_wagered:.0f}")
    print(f"    Returned: ${total_returned:.0f}")
    print(f"    Profit:   ${profit:+.0f}")
    print(f"    ROI:      {roi:+.1f}%")

    # ---------- MODEL STATUS ----------
    print(f"\n  MODEL STATUS:")

    if accuracy > LEAKAGE_THRESHOLD:
        print(f"    \u274c SUSPECT - {accuracy:.1f}% accuracy likely inflated by data leakage")
        print(f"    Action: Fix data coverage issues above, then re-run backtest")
    elif total >= 100:
        if accuracy >= 60 and (not high_conf or hc_acc >= 65):
            print(f"    \u2705 VALIDATED on {total} out-of-sample fights")
            print(f"    Ready for live tracking")
        elif accuracy >= 55:
            print(f"    \u26a0\ufe0f ACCEPTABLE - {accuracy:.1f}% on {total} fights")
            print(f"    Proceed with live tracking, monitor closely")
        else:
            print(f"    \u274c UNDERPERFORMING - {accuracy:.1f}% on {total} fights")
            print(f"    Consider retraining before live deployment")
    elif total >= 50:
        print(f"    Sample: {total} fights (good, 100+ preferred)")
        if accuracy >= 60:
            print(f"    \u2705 PROMISING - proceed to live tracking")
        else:
            print(f"    \u26a0\ufe0f INCONCLUSIVE - gather more data")
    else:
        print(f"    Sample: {total} fights (need 50+ for reliable stats)")

    # ---------- EXPECTED RANGES ----------
    print(f"\n  EXPECTED RANGES (production-ready model):")
    print(f"    {'METRIC':<28} {'EXPECTED':>12} {'ACTUAL':>12} {'STATUS':>8}")
    print(f"    {'-'*62}")

    checks = [
        ('Overall accuracy', '60-66%', accuracy, 60, 66),
        ('High conf accuracy', '70-78%', hc_acc if high_conf else 0, 70, 78),
        ('ROI (65%+ picks)', '+5% to +15%', roi, 5, 15),
    ]
    for label, expected_str, actual_val, lo, hi in checks:
        if lo <= actual_val <= hi:
            s = "\u2705"
        elif actual_val > hi:
            s = "\u26a0\ufe0f HIGH"
        else:
            s = "\u274c LOW"
        print(f"    {label:<28} {expected_str:>12} {actual_val:>11.1f}% {s:>8}")

    print(f"\n  Saved {total} backtest predictions to database")
    print("=" * W)


# ============================================================================
# MULTI-YEAR COMBINED SUMMARY
# ============================================================================

def _print_combined_summary(year_results):
    """
    Print a combined report across multiple backtest years.
    year_results: list of (year, results_list) tuples.
    """
    W = 80

    all_results = []
    for year, results in year_results:
        all_results.extend(results)

    if not all_results:
        return

    total = len(all_results)
    correct = sum(1 for r in all_results if r['was_correct'])
    accuracy = correct / total * 100

    years_str = ", ".join(str(y) for y, _ in year_results)

    print("\n" + "=" * W)
    print(f"{'COMBINED MULTI-YEAR BACKTEST SUMMARY':^{W}}")
    print("=" * W)

    # ---------- PER-YEAR BREAKDOWN ----------
    print(f"\n  PER-YEAR BREAKDOWN:")
    print(f"    {'YEAR':<8} {'FIGHTS':>7} {'CORRECT':>8} {'ACC':>7} {'HC ACC':>8} {'ROI':>7} {'STATUS':>10}")
    print(f"    {'-'*58}")

    for year, results in year_results:
        y_total = len(results)
        y_correct = sum(1 for r in results if r['was_correct'])
        y_acc = y_correct / y_total * 100 if y_total > 0 else 0

        y_hc = [r for r in results if r['pick_pct'] >= 0.65]
        y_hc_correct = sum(1 for r in y_hc if r['was_correct'])
        y_hc_acc = y_hc_correct / len(y_hc) * 100 if y_hc else 0

        STAKE = 100
        y_wagered = len(y_hc) * STAKE
        y_returned = 0
        for r in y_hc:
            if r['was_correct']:
                y_returned += STAKE * (1.0 / r['pick_pct'] if r['pick_pct'] > 0 else 1.0)
        y_roi = (y_returned - y_wagered) / y_wagered * 100 if y_wagered > 0 else 0

        trained_on = year < 2024
        if trained_on:
            status = "\u26a0\ufe0f TRAIN"
        elif y_acc > LEAKAGE_THRESHOLD:
            status = "\u274c LEAK?"
        elif y_acc >= 60:
            status = "\u2705 VALID"
        elif y_acc >= 55:
            status = "\u26a0\ufe0f OK"
        else:
            status = "\u274c LOW"

        print(f"    {year:<8} {y_total:>7} {y_correct:>8} {y_acc:>6.1f}% {y_hc_acc:>7.1f}% {y_roi:>+6.1f}% {status:>10}")

    # ---------- COMBINED OVERALL ----------
    print(f"\n  COMBINED ({years_str}):")
    print(f"    Total Predictions: {total}")
    print(f"    Correct:           {correct}")
    print(f"    Overall Accuracy:  {accuracy:.1f}%")

    # ---------- COMBINED TIERS ----------
    tiers = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in all_results:
        tiers[r['confidence']]['total'] += 1
        if r['was_correct']:
            tiers[r['confidence']]['correct'] += 1

    print(f"\n  COMBINED CONFIDENCE TIERS:")
    for tier in ['HIGH', 'MED', 'LOW', 'TOSS']:
        t = tiers[tier]
        if t['total'] == 0:
            continue
        acc = t['correct'] / t['total'] * 100
        icon = "\u2705" if acc >= 65 else "\u26a0\ufe0f" if acc >= 50 else "\u274c"
        print(f"    {icon} {tier:<4} : {t['correct']}/{t['total']} ({acc:.1f}%)")

    # ---------- COMBINED HIGH CONFIDENCE ----------
    hc = [r for r in all_results if r['pick_pct'] >= 0.65]
    if hc:
        hc_correct = sum(1 for r in hc if r['was_correct'])
        hc_acc = hc_correct / len(hc) * 100
        print(f"\n  COMBINED HIGH CONFIDENCE (\u226565%):")
        print(f"    Total:    {len(hc)}")
        print(f"    Correct:  {hc_correct}")
        print(f"    Accuracy: {hc_acc:.1f}%")

    # ---------- COMBINED CALIBRATION ----------
    buckets = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in all_results:
        prob = r['pick_pct']
        if prob >= 0.70:
            bucket = '70-100%'
        elif prob >= 0.65:
            bucket = '65-70%'
        elif prob >= 0.60:
            bucket = '60-65%'
        elif prob >= 0.55:
            bucket = '55-60%'
        else:
            bucket = '50-55%'
        buckets[bucket]['total'] += 1
        if r['was_correct']:
            buckets[bucket]['correct'] += 1

    print(f"\n  COMBINED CALIBRATION:")
    print(f"    {'BUCKET':<12} {'PRED':>6} {'ACTUAL':>8} {'DELTA':>7} {'N':>4} {'STATUS':>8}")
    print(f"    {'-'*50}")
    for bucket in ['70-100%', '65-70%', '60-65%', '55-60%', '50-55%']:
        b = buckets[bucket]
        if b['total'] == 0:
            continue
        parts = bucket.replace('%', '').split('-')
        expected = (float(parts[0]) + float(parts[1])) / 2
        actual = b['correct'] / b['total'] * 100
        delta = actual - expected
        cal_ok = "GOOD" if abs(delta) <= 5 else ("OK" if abs(delta) <= 10 else "OFF")
        print(f"    {bucket:<12} {expected:>5.0f}% {actual:>7.1f}% {delta:>+6.1f}% {b['total']:>4} {cal_ok:>8}")

    # ---------- COMBINED PAPER TRADING ----------
    STAKE = 100
    total_wagered = len(hc) * STAKE
    total_returned = 0
    for r in hc:
        if r['was_correct']:
            total_returned += STAKE * (1.0 / r['pick_pct'] if r['pick_pct'] > 0 else 1.0)
    profit = total_returned - total_wagered
    roi = profit / total_wagered * 100 if total_wagered > 0 else 0

    print(f"\n  COMBINED PAPER TRADING (65%+ picks, ${STAKE} flat bets):")
    print(f"    Bets:     {len(hc)}")
    print(f"    Wagered:  ${total_wagered:.0f}")
    print(f"    Returned: ${total_returned:.0f}")
    print(f"    Profit:   ${profit:+.0f}")
    print(f"    ROI:      {roi:+.1f}%")

    # ---------- OUT-OF-SAMPLE ONLY ----------
    oos_results = []
    for year, results in year_results:
        if year >= 2024:
            oos_results.extend(results)

    if oos_results and len(oos_results) < total:
        oos_total = len(oos_results)
        oos_correct = sum(1 for r in oos_results if r['was_correct'])
        oos_acc = oos_correct / oos_total * 100
        oos_hc = [r for r in oos_results if r['pick_pct'] >= 0.65]
        oos_hc_correct = sum(1 for r in oos_hc if r['was_correct'])
        oos_hc_acc = oos_hc_correct / len(oos_hc) * 100 if oos_hc else 0

        print(f"\n  OUT-OF-SAMPLE ONLY (2024+):")
        print(f"    Predictions: {oos_total}")
        print(f"    Accuracy:    {oos_acc:.1f}%")
        if oos_hc:
            print(f"    HC Accuracy: {oos_hc_acc:.1f}% ({oos_hc_correct}/{len(oos_hc)})")

    # ---------- VERDICT ----------
    print(f"\n  VERDICT:")

    oos_total = len(oos_results) if oos_results else 0
    oos_acc = sum(1 for r in oos_results if r['was_correct']) / oos_total * 100 if oos_total > 0 else 0

    consistent = True
    acc_range = []
    for year, results in year_results:
        if results:
            y_acc = sum(1 for r in results if r['was_correct']) / len(results) * 100
            acc_range.append(y_acc)
    if acc_range:
        spread = max(acc_range) - min(acc_range)
        if spread > 15:
            consistent = False

    if oos_total >= 100 and oos_acc >= 60 and consistent:
        print(f"    \u2705 MODEL VALIDATED across {len(year_results)} years ({total} total fights)")
        print(f"    Consistency: {min(acc_range):.1f}% - {max(acc_range):.1f}% (spread: {spread:.1f}%)")
        if roi > 0:
            print(f"    Paper Trading: Profitable ({roi:+.1f}% ROI)")
        print(f"    Status: READY FOR LIVE DEPLOYMENT")
    elif oos_total >= 50 and oos_acc >= 55:
        print(f"    \u26a0\ufe0f PROMISING but needs more data")
        print(f"    Consistency: {min(acc_range):.1f}% - {max(acc_range):.1f}%")
    else:
        print(f"    \u274c NEEDS REVIEW")
        if not consistent:
            print(f"    Inconsistent: {min(acc_range):.1f}% - {max(acc_range):.1f}% (spread: {spread:.1f}%)")

    print("=" * W)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Historical Backtest - Validate predictions against past fights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Backtest all 2025 fights (default)
  python -m ml.backtest

  # Specific year
  python -m ml.backtest --year 2024

  # Multiple years with combined summary
  python -m ml.backtest --years 2023 2024 2025

  # Date range
  python -m ml.backtest --start 2025-01-01 --end 2025-07-01

  # Single event
  python -m ml.backtest --event "UFC 311"

  # Clear previous backtest data first
  python -m ml.backtest --clear
        '''
    )
    parser.add_argument('--year', '-y', type=int, default=None,
                        help='Year to backtest (default: 2025)')
    parser.add_argument('--years', nargs='+', type=int,
                        help='Multiple years to backtest (e.g. --years 2023 2024 2025)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--event', '-e', type=str, help='Filter to a specific event name')
    parser.add_argument('--clear', '-c', action='store_true',
                        help='Clear previous backtest data before running')

    args = parser.parse_args()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        sys.exit(1)

    try:
        if args.years:
            year_results = []
            if args.clear:
                cur = conn.cursor()
                cur.execute("DELETE FROM PredictionTracking WHERE prediction_type = 'backtest'")
                deleted = cur.rowcount
                conn.commit()
                cur.close()
                if deleted:
                    print(f"  Cleared {deleted} previous backtest predictions")

            for year in sorted(args.years):
                results = run_backtest(
                    conn,
                    start_date=f'{year}-01-01',
                    end_date=f'{year + 1}-01-01',
                    event_filter=args.event,
                    clear=False,
                )
                year_results.append((year, results))

            _print_combined_summary(year_results)
        else:
            year = args.year or 2025

            if args.start:
                start_date = args.start
            else:
                start_date = f'{year}-01-01'

            if args.end:
                end_date = args.end
            else:
                end_date = f'{year + 1}-01-01'

            run_backtest(
                conn,
                start_date=start_date,
                end_date=end_date,
                event_filter=args.event,
                clear=args.clear,
            )
    finally:
        conn.close()


if __name__ == '__main__':
    main()

"""
Cumulative Performance Summary

Aggregates prediction results across all resolved events,
SEPARATED by prediction type (backtest vs live).

Usage:
    cd DelphiAIApp/Models

    # Full report (backtest + live shown separately)
    python -m ml.performance_summary

    # Only backtest results
    python -m ml.performance_summary --type backtest

    # Only live results
    python -m ml.performance_summary --type live

    # Last N events only
    python -m ml.performance_summary --last 5

    # Specific event
    python -m ml.performance_summary --event "Strickland vs Hernandez"
"""

import os
import sys
import argparse
from pathlib import Path
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


# ============================================================================
# DATA RETRIEVAL
# ============================================================================

def get_resolved_predictions(conn, prediction_type=None, last_n_events=None,
                             event_filter=None):
    """
    Get resolved predictions from the database.

    Args:
        conn: Database connection
        prediction_type: 'backtest', 'live', or None for all
        last_n_events: Limit to the N most recent events
        event_filter: Filter by event name (ILIKE)
    """
    cur = conn.cursor()

    query = """
        SELECT id, event_name, event_date,
               fighter1_name, fighter2_name,
               pick_name, pick_probability, fighter1_probability,
               confidence, prob_source,
               predicted_method, predicted_ko, predicted_sub, predicted_dec,
               actual_winner_name, actual_method, actual_round,
               was_correct, predicted_at, resolved_at,
               is_title_fight, fighter1_elo, fighter2_elo,
               COALESCE(prediction_type, 'live') as prediction_type
        FROM PredictionTracking
        WHERE was_correct IS NOT NULL
    """
    params = []

    if prediction_type:
        query += " AND COALESCE(prediction_type, 'live') = %s"
        params.append(prediction_type)

    if event_filter:
        query += " AND event_name ILIKE %s"
        params.append(f'%{event_filter}%')

    query += " ORDER BY predicted_at DESC, id"

    cur.execute(query, params)
    columns = [desc[0] for desc in cur.description]
    all_preds = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()

    if last_n_events and not event_filter:
        seen = []
        for p in all_preds:
            if p['event_name'] not in seen:
                seen.append(p['event_name'])
        allowed = set(seen[:last_n_events])
        all_preds = [p for p in all_preds if p['event_name'] in allowed]

    return all_preds


def get_prediction_counts(conn):
    """Get counts of backtest vs live predictions (including unresolved)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(prediction_type, 'live') as ptype,
               COUNT(*) as total,
               COUNT(CASE WHEN was_correct IS NOT NULL THEN 1 END) as resolved,
               COUNT(CASE WHEN was_correct = TRUE THEN 1 END) as correct
        FROM PredictionTracking
        GROUP BY COALESCE(prediction_type, 'live')
    """)
    counts = {}
    for row in cur.fetchall():
        counts[row[0]] = {'total': row[1], 'resolved': row[2], 'correct': row[3]}
    cur.close()
    return counts


# ============================================================================
# STATISTICS CALCULATION
# ============================================================================

def calculate_stats(predictions):
    """Calculate comprehensive performance statistics for a set of predictions."""
    if not predictions:
        return None

    total = len(predictions)
    correct = sum(1 for p in predictions if p['was_correct'])
    accuracy = correct / total * 100

    # By confidence tier
    tiers = defaultdict(lambda: {'total': 0, 'correct': 0})
    for p in predictions:
        tier = p.get('confidence', 'TOSS')
        tiers[tier]['total'] += 1
        if p['was_correct']:
            tiers[tier]['correct'] += 1

    tier_stats = {}
    for tier in ['HIGH', 'MED', 'LOW', 'TOSS']:
        t = tiers[tier]
        if t['total'] > 0:
            tier_stats[tier] = {
                'total': t['total'],
                'correct': t['correct'],
                'accuracy': t['correct'] / t['total'] * 100,
            }

    # By probability bucket (calibration)
    buckets = defaultdict(lambda: {'total': 0, 'correct': 0})
    for p in predictions:
        prob = float(p['pick_probability'])
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
        if p['was_correct']:
            buckets[bucket]['correct'] += 1

    bucket_stats = {}
    for b in ['70-100%', '65-70%', '60-65%', '55-60%', '50-55%']:
        d = buckets[b]
        if d['total'] > 0:
            bucket_stats[b] = {
                'total': d['total'],
                'correct': d['correct'],
                'accuracy': d['correct'] / d['total'] * 100,
            }

    # By source
    source_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for p in predictions:
        src = 'ML+ELO' if p.get('prob_source', '').startswith('ML') else 'ELO'
        source_stats[src]['total'] += 1
        if p['was_correct']:
            source_stats[src]['correct'] += 1
    for src in source_stats:
        d = source_stats[src]
        d['accuracy'] = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0

    # By event
    event_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'date': ''})
    for p in predictions:
        e = p['event_name']
        event_stats[e]['total'] += 1
        event_stats[e]['date'] = p.get('event_date', '')
        if p['was_correct']:
            event_stats[e]['correct'] += 1
    for e in event_stats:
        d = event_stats[e]
        d['accuracy'] = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0

    # High confidence
    high_conf = [p for p in predictions if float(p['pick_probability']) >= 0.65]
    hc_correct = sum(1 for p in high_conf if p['was_correct'])
    hc_accuracy = hc_correct / len(high_conf) * 100 if high_conf else 0

    # Paper trading
    paper_trading = _calculate_paper_trading(predictions)

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'tier_stats': tier_stats,
        'bucket_stats': bucket_stats,
        'source_stats': dict(source_stats),
        'event_stats': dict(event_stats),
        'high_conf_total': len(high_conf),
        'high_conf_correct': hc_correct,
        'high_conf_accuracy': hc_accuracy,
        'paper_trading': paper_trading,
        'num_events': len(event_stats),
    }


def _calculate_paper_trading(predictions):
    """Calculate hypothetical ROI for different strategies."""
    STAKE = 100
    results = {}

    for label, min_prob in [('all_picks', 0.50), ('med_conf', 0.60), ('high_conf', 0.65)]:
        filtered = [p for p in predictions if float(p['pick_probability']) >= min_prob]
        wagered = len(filtered) * STAKE
        returned = 0
        for p in filtered:
            if p['was_correct']:
                prob = float(p['pick_probability'])
                returned += STAKE * (1.0 / prob if prob > 0 else 1.0)
        results[label] = {
            'bets': len(filtered),
            'wagered': wagered,
            'returned': returned,
            'profit': returned - wagered,
            'roi': (returned - wagered) / wagered * 100 if wagered > 0 else 0,
        }

    return results


# ============================================================================
# REPORT PRINTING
# ============================================================================

def _print_section(stats, title, W=80):
    """Print a single performance section (used for both backtest and live)."""
    if not stats:
        print(f"\n  {title}:")
        print(f"    No resolved predictions yet.")
        print(f"    {'(Run backtest first: python -m ml.backtest)' if 'BACKTEST' in title else '(Predictions resolve after events complete)'}")
        return

    print(f"\n  {title}:")
    print(f"  {'-' * (W - 4)}")
    print(f"    Events:            {stats['num_events']}")
    print(f"    Total Predictions: {stats['total']}")
    print(f"    Correct:           {stats['correct']}")
    print(f"    Accuracy:          {stats['accuracy']:.1f}%")

    # High confidence
    if stats['high_conf_total'] > 0:
        hc_icon = "\u2705" if stats['high_conf_accuracy'] >= 65 else "\u26a0\ufe0f"
        print(f"    High Conf (\u226565%):  {stats['high_conf_accuracy']:.1f}% "
              f"({stats['high_conf_correct']}/{stats['high_conf_total']}) {hc_icon}")

    # By confidence tier
    print(f"\n    Confidence Tiers:")
    for tier in ['HIGH', 'MED', 'LOW', 'TOSS']:
        if tier not in stats['tier_stats']:
            continue
        t = stats['tier_stats'][tier]
        acc = t['accuracy']
        icon = "\u2705" if acc >= 65 else "\u26a0\ufe0f" if acc >= 50 else "\u274c"
        print(f"      {icon} {tier:<4} : {t['correct']}/{t['total']} ({acc:.1f}%)")

    # Calibration
    bucket_order = ['70-100%', '65-70%', '60-65%', '55-60%', '50-55%']
    has_buckets = any(b in stats['bucket_stats'] for b in bucket_order)
    if has_buckets:
        print(f"\n    Calibration:")
        print(f"      {'BUCKET':<12} {'PRED':>6} {'ACTUAL':>8} {'DELTA':>7}")
        print(f"      {'-'*35}")
        for bucket in bucket_order:
            if bucket not in stats['bucket_stats']:
                continue
            b = stats['bucket_stats'][bucket]
            parts = bucket.replace('%', '').split('-')
            expected = (float(parts[0]) + float(parts[1])) / 2
            actual = b['accuracy']
            delta = actual - expected
            print(f"      {bucket:<12} {expected:>5.0f}% {actual:>7.1f}% {delta:>+6.1f}%")

    # By source
    if len(stats['source_stats']) > 1:
        print(f"\n    By Source:")
        for src, d in stats['source_stats'].items():
            print(f"      {src:<12}: {d['correct']}/{d['total']} ({d['accuracy']:.1f}%)")

    # Paper trading (65%+ only for brevity)
    pt = stats['paper_trading']
    if pt['high_conf']['bets'] > 0:
        d = pt['high_conf']
        print(f"\n    Paper Trading (65%+ picks, $100 flat):")
        print(f"      {d['bets']} bets | ${d['profit']:+.0f} profit | {d['roi']:+.1f}% ROI")


def print_full_report(conn, last_n_events=None, event_filter=None,
                      type_filter=None):
    """
    Print the complete performance report with separate backtest and live sections.
    """
    W = 80

    print("\n" + "=" * W)
    title = 'DELPHI AI - PERFORMANCE SUMMARY'
    if event_filter:
        title += f' ({event_filter})'
    print(f"{title:^{W}}")
    print("=" * W)

    counts = get_prediction_counts(conn)

    # Show overview of what's tracked
    bt_counts = counts.get('backtest', {'total': 0, 'resolved': 0, 'correct': 0})
    live_counts = counts.get('live', {'total': 0, 'resolved': 0, 'correct': 0})

    print(f"\n  TRACKING OVERVIEW:")
    print(f"    Backtest predictions: {bt_counts['resolved']} resolved / {bt_counts['total']} total")
    print(f"    Live predictions:     {live_counts['resolved']} resolved / {live_counts['total']} total")

    # ====================== BACKTEST SECTION ======================
    if type_filter in (None, 'backtest'):
        bt_preds = get_resolved_predictions(
            conn, prediction_type='backtest',
            last_n_events=last_n_events, event_filter=event_filter,
        )
        bt_stats = calculate_stats(bt_preds)
        _print_section(bt_stats, 'BACKTEST (Historical Validation)', W)

        # Show per-event breakdown for backtest
        if bt_stats and bt_stats['num_events'] > 0:
            print(f"\n    Per Event:")
            print(f"      {'EVENT':<38} {'ACC':>12}")
            print(f"      {'-'*52}")
            sorted_events = sorted(bt_stats['event_stats'].items(),
                                   key=lambda x: x[1].get('date', ''))
            for name, d in sorted_events:
                short = name[:37]
                print(f"      {short:<38} {d['correct']}/{d['total']} ({d['accuracy']:.0f}%)")

    # ====================== LIVE SECTION ======================
    if type_filter in (None, 'live'):
        live_preds = get_resolved_predictions(
            conn, prediction_type='live',
            last_n_events=last_n_events, event_filter=event_filter,
        )
        live_stats = calculate_stats(live_preds)
        _print_section(live_stats, 'LIVE (Ongoing 2026 Tracking)', W)

        if live_stats and live_stats['num_events'] > 0:
            print(f"\n    Per Event:")
            print(f"      {'EVENT':<38} {'ACC':>12}")
            print(f"      {'-'*52}")
            sorted_events = sorted(live_stats['event_stats'].items(),
                                   key=lambda x: x[1].get('date', ''))
            for name, d in sorted_events:
                short = name[:37]
                print(f"      {short:<38} {d['correct']}/{d['total']} ({d['accuracy']:.0f}%)")

    # ====================== COMPARISON ======================
    if type_filter is None:
        bt_preds = get_resolved_predictions(conn, prediction_type='backtest')
        live_preds = get_resolved_predictions(conn, prediction_type='live')
        bt_stats = calculate_stats(bt_preds)
        live_stats = calculate_stats(live_preds)

        if bt_stats and live_stats:
            print(f"\n  BACKTEST vs LIVE COMPARISON:")
            print(f"  {'-' * (W - 4)}")
            print(f"    {'METRIC':<28} {'BACKTEST':>12} {'LIVE':>12}")
            print(f"    {'-'*54}")
            print(f"    {'Fights':<28} {bt_stats['total']:>12} {live_stats['total']:>12}")
            print(f"    {'Accuracy':<28} {bt_stats['accuracy']:>11.1f}% {live_stats['accuracy']:>11.1f}%")
            if bt_stats['high_conf_total'] > 0 and live_stats['high_conf_total'] > 0:
                print(f"    {'High Conf Accuracy':<28} {bt_stats['high_conf_accuracy']:>11.1f}% "
                      f"{live_stats['high_conf_accuracy']:>11.1f}%")

            bt_roi = bt_stats['paper_trading']['high_conf']['roi']
            live_roi = live_stats['paper_trading']['high_conf']['roi']
            print(f"    {'ROI (65%+ picks)':<28} {bt_roi:>+11.1f}% {live_roi:>+11.1f}%")

            delta = abs(bt_stats['accuracy'] - live_stats['accuracy'])
            if delta < 5:
                print(f"\n    \u2705 Backtest and live performance are consistent (delta: {delta:.1f}%)")
            elif delta < 10:
                print(f"\n    \u26a0\ufe0f Some divergence between backtest and live ({delta:.1f}%)")
            else:
                print(f"\n    \u274c Significant divergence ({delta:.1f}%) - investigate")

    # ====================== MODEL STATUS ======================
    all_preds = get_resolved_predictions(conn, prediction_type=type_filter)
    all_stats = calculate_stats(all_preds)

    if all_stats and all_stats['total'] >= 20:
        print(f"\n  MODEL STATUS:")
        print(f"  {'-' * (W - 4)}")

        # Check calibration
        calibrated = True
        for bucket in ['70-100%', '65-70%', '60-65%', '55-60%', '50-55%']:
            if bucket in all_stats['bucket_stats']:
                b = all_stats['bucket_stats'][bucket]
                parts = bucket.replace('%', '').split('-')
                expected = (float(parts[0]) + float(parts[1])) / 2
                if abs(b['accuracy'] - expected) > 15:
                    calibrated = False
                    break

        pt = all_stats['paper_trading']
        profitable = pt['high_conf']['roi'] > 0 if pt['high_conf']['bets'] > 0 else False

        if all_stats['accuracy'] >= 60 and calibrated:
            print(f"    Accuracy:    {all_stats['accuracy']:.1f}% \u2705 VALIDATED")
        elif all_stats['accuracy'] >= 55:
            print(f"    Accuracy:    {all_stats['accuracy']:.1f}% \u26a0\ufe0f ACCEPTABLE")
        else:
            print(f"    Accuracy:    {all_stats['accuracy']:.1f}% \u274c NEEDS RETRAINING")

        cal_icon = "\u2705" if calibrated else "\u274c"
        print(f"    Calibration: {'GOOD' if calibrated else 'MISCALIBRATED'} {cal_icon}")

        if profitable:
            print(f"    ROI (65%+):  {pt['high_conf']['roi']:+.1f}% \u2705 PROFITABLE")
        else:
            print(f"    ROI (65%+):  {pt['high_conf']['roi']:+.1f}% \u274c NOT PROFITABLE")

        has_live = live_counts['resolved'] > 0
        has_backtest = bt_counts['resolved'] > 0

        if has_backtest and has_live and all_stats['accuracy'] >= 60 and profitable:
            print(f"\n    Ready for betting: YES")
        elif has_backtest and not has_live:
            print(f"\n    Ready for betting: NOT YET - need live validation")
        else:
            print(f"\n    Ready for betting: CONTINUE TRACKING")

    print("\n" + "=" * W)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Performance Summary - Separate backtest vs live tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full report (backtest + live shown separately)
  python -m ml.performance_summary

  # Only backtest results
  python -m ml.performance_summary --type backtest

  # Only live results
  python -m ml.performance_summary --type live

  # Last 5 events
  python -m ml.performance_summary --last 5

  # Filter by event
  python -m ml.performance_summary --event "Strickland"
        '''
    )
    parser.add_argument('--type', '-t', choices=['backtest', 'live'],
                        help='Show only backtest or live results')
    parser.add_argument('--last', '-n', type=int, help='Only include the last N events')
    parser.add_argument('--event', '-e', type=str, help='Filter by event name')

    args = parser.parse_args()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        sys.exit(1)

    try:
        # Ensure the table exists (handles fresh installs)
        try:
            from ml.update_results import ensure_tracking_table
        except (ModuleNotFoundError, ImportError):
            from update_results import ensure_tracking_table
        ensure_tracking_table(conn)

        print_full_report(
            conn,
            last_n_events=args.last,
            event_filter=args.event,
            type_filter=args.type,
        )
    finally:
        conn.close()


if __name__ == '__main__':
    main()

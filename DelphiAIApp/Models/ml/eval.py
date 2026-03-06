"""
Prediction Evaluation Dashboard

Queries the PredictionLog table to show how the model is performing in production.
Shows running accuracy, calibration, ROI, and performance over time.

Usage:
    python -m ml.eval                    # Show overall stats
    python -m ml.eval --resolve          # Resolve unresolved predictions (update outcomes)
    python -m ml.eval --monthly          # Show monthly accuracy breakdown
    python -m ml.eval --calibration      # Show calibration analysis
    python -m ml.eval --roi              # Show ROI analysis
    python -m ml.eval --all              # Show everything
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import psycopg2
from dotenv import load_dotenv

# Load environment
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


def check_table_exists(conn):
    """Check if PredictionLog table exists."""
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'predictionlog'
        )
    """)
    exists = cur.fetchone()[0]
    cur.close()
    return exists


def get_prediction_stats(conn):
    """Get overall prediction statistics."""
    cur = conn.cursor()
    
    # Total predictions
    cur.execute("SELECT COUNT(*) FROM PredictionLog")
    total = cur.fetchone()[0]
    
    # Resolved predictions
    cur.execute("SELECT COUNT(*) FROM PredictionLog WHERE WasCorrect IS NOT NULL")
    resolved = cur.fetchone()[0]
    
    # Unresolved
    cur.execute("SELECT COUNT(*) FROM PredictionLog WHERE WasCorrect IS NULL")
    unresolved = cur.fetchone()[0]
    
    # Correct / Wrong
    cur.execute("SELECT COUNT(*) FROM PredictionLog WHERE WasCorrect = TRUE")
    correct = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM PredictionLog WHERE WasCorrect = FALSE")
    wrong = cur.fetchone()[0]
    
    # By model version
    cur.execute("""
        SELECT ModelVersion, 
               COUNT(*) as total,
               SUM(CASE WHEN WasCorrect = TRUE THEN 1 ELSE 0 END) as correct,
               SUM(CASE WHEN WasCorrect IS NOT NULL THEN 1 ELSE 0 END) as resolved
        FROM PredictionLog
        GROUP BY ModelVersion
        ORDER BY ModelVersion
    """)
    by_version = cur.fetchall()
    
    # By probability source (ML vs ELO)
    cur.execute("""
        SELECT ProbabilitySource,
               COUNT(*) as total,
               SUM(CASE WHEN WasCorrect = TRUE THEN 1 ELSE 0 END) as correct,
               SUM(CASE WHEN WasCorrect IS NOT NULL THEN 1 ELSE 0 END) as resolved
        FROM PredictionLog
        GROUP BY ProbabilitySource
        ORDER BY ProbabilitySource
    """)
    by_source = cur.fetchall()
    
    # By confidence level
    cur.execute("""
        SELECT ConfidenceLevel,
               COUNT(*) as total,
               SUM(CASE WHEN WasCorrect = TRUE THEN 1 ELSE 0 END) as correct,
               SUM(CASE WHEN WasCorrect IS NOT NULL THEN 1 ELSE 0 END) as resolved
        FROM PredictionLog
        GROUP BY ConfidenceLevel
        ORDER BY ConfidenceLevel
    """)
    by_confidence = cur.fetchall()
    
    cur.close()
    
    return {
        'total': total,
        'resolved': resolved,
        'unresolved': unresolved,
        'correct': correct,
        'wrong': wrong,
        'by_version': by_version,
        'by_source': by_source,
        'by_confidence': by_confidence,
    }


def get_monthly_stats(conn):
    """Get monthly prediction accuracy."""
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            DATE_TRUNC('month', CreatedAt) as month,
            COUNT(*) as total,
            SUM(CASE WHEN WasCorrect = TRUE THEN 1 ELSE 0 END) as correct,
            SUM(CASE WHEN WasCorrect IS NOT NULL THEN 1 ELSE 0 END) as resolved
        FROM PredictionLog
        GROUP BY DATE_TRUNC('month', CreatedAt)
        ORDER BY month DESC
    """)
    
    months = cur.fetchall()
    cur.close()
    return months


def get_calibration_data(conn):
    """Get calibration data (predicted vs actual win rates by bin)."""
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            ProbabilityA, ProbabilityB, WasCorrect,
            PredictedWinnerName, FighterAName
        FROM PredictionLog
        WHERE WasCorrect IS NOT NULL
    """)
    
    rows = cur.fetchall()
    cur.close()
    
    # Bin predictions
    bins = defaultdict(lambda: {'predicted_sum': 0, 'actual_sum': 0, 'count': 0})
    
    for prob_a, prob_b, was_correct, predicted, fighter_a in rows:
        prob_a = float(prob_a)
        prob_b = float(prob_b)
        
        # Use the probability of the predicted winner
        if predicted == fighter_a:
            conf = prob_a
        else:
            conf = prob_b
        
        # Bin by 5% intervals
        bin_key = round(conf * 20) / 20  # Round to nearest 5%
        bin_key = max(0.50, min(0.95, bin_key))
        
        bins[bin_key]['predicted_sum'] += conf
        bins[bin_key]['actual_sum'] += 1 if was_correct else 0
        bins[bin_key]['count'] += 1
    
    return dict(bins)


def get_roi_data(conn):
    """Calculate betting ROI from predictions."""
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            ProbabilityA, ProbabilityB, WasCorrect,
            PredictedWinnerName, FighterAName,
            EloProbabilityA, EloProbabilityB,
            ProbabilitySource
        FROM PredictionLog
        WHERE WasCorrect IS NOT NULL
    """)
    
    rows = cur.fetchall()
    cur.close()
    
    if not rows:
        return None
    
    # Betting parameters
    VIG_BREAKEVEN = 0.524
    DECIMAL_ODDS = 1.91  # -110
    
    # Track by edge threshold
    thresholds = [0.55, 0.57, 0.60, 0.65, 0.70]
    roi_data = {}
    
    for threshold in thresholds:
        bets = 0
        wins = 0
        profit = 0.0
        
        for prob_a, prob_b, was_correct, predicted, fighter_a, elo_a, elo_b, source in rows:
            prob_a = float(prob_a)
            prob_b = float(prob_b)
            
            if predicted == fighter_a:
                conf = prob_a
            else:
                conf = prob_b
            
            if conf >= threshold:
                bets += 1
                if was_correct:
                    wins += 1
                    profit += (DECIMAL_ODDS - 1)
                else:
                    profit -= 1
        
        roi_data[threshold] = {
            'bets': bets,
            'wins': wins,
            'win_rate': wins / bets if bets > 0 else 0,
            'profit': profit,
            'roi': profit / bets if bets > 0 else 0,
        }
    
    return roi_data


def resolve_predictions(conn):
    """
    Attempt to resolve unresolved predictions by matching with fight results.
    Looks up actual fight outcomes and updates the PredictionLog.
    """
    cur = conn.cursor()
    
    # Get unresolved predictions
    cur.execute("""
        SELECT PredictionID, FighterAName, FighterBName, PredictedWinnerName
        FROM PredictionLog
        WHERE WasCorrect IS NULL
    """)
    
    unresolved = cur.fetchall()
    
    if not unresolved:
        print("  No unresolved predictions to resolve.")
        return 0
    
    print(f"  Found {len(unresolved)} unresolved predictions")
    resolved_count = 0
    
    for pred_id, fighter_a, fighter_b, predicted in unresolved:
        # Try to find the fight result
        cur.execute("""
            SELECT WinnerName, Method, Date
            FROM Fights
            WHERE (
                (LOWER(FighterName) = LOWER(%s) AND LOWER(OpponentName) = LOWER(%s))
                OR (LOWER(FighterName) = LOWER(%s) AND LOWER(OpponentName) = LOWER(%s))
            )
            AND WinnerName IS NOT NULL AND WinnerName != ''
            ORDER BY Date DESC
            LIMIT 1
        """, (fighter_a, fighter_b, fighter_b, fighter_a))
        
        result = cur.fetchone()
        
        if result:
            actual_winner, method, fight_date = result
            was_correct = predicted.lower() == actual_winner.lower()
            
            cur.execute("""
                UPDATE PredictionLog
                SET ActualWinner = %s,
                    ActualMethod = %s,
                    FightDate = %s,
                    WasCorrect = %s,
                    ResolvedAt = CURRENT_TIMESTAMP
                WHERE PredictionID = %s
            """, (actual_winner, method, fight_date, was_correct, pred_id))
            
            resolved_count += 1
            status = "CORRECT" if was_correct else "WRONG"
            try:
                print(f"    [{status}] {fighter_a} vs {fighter_b} -> {actual_winner}")
            except UnicodeEncodeError:
                print(f"    [{status}] (fighter name encoding issue)")
    
    conn.commit()
    cur.close()
    
    print(f"\n  Resolved {resolved_count} / {len(unresolved)} predictions")
    return resolved_count


def print_overall_stats(stats):
    """Print overall prediction statistics."""
    print("\n" + "=" * 60)
    print(f"{'PREDICTION LOG DASHBOARD':^60}")
    print("=" * 60)
    
    total = stats['total']
    if total == 0:
        print("\n  No predictions logged yet.")
        print("  Run 'python -m ml.predict_fight ...' to make predictions.")
        print("  Run migration 003_prediction_log.sql first if needed.")
        return
    
    print(f"\n  Total Predictions:  {total}")
    print(f"  Resolved:           {stats['resolved']}")
    print(f"  Unresolved:         {stats['unresolved']}")
    
    if stats['resolved'] > 0:
        accuracy = stats['correct'] / stats['resolved'] * 100
        print(f"\n  ACCURACY: {accuracy:.1f}% ({stats['correct']}/{stats['resolved']})")
    
    # By model version
    if stats['by_version']:
        print(f"\n  BY MODEL VERSION:")
        print(f"  {'Version':<25} {'Total':>6} {'Resolved':>9} {'Accuracy':>9}")
        print("  " + "-" * 52)
        for version, total_v, correct_v, resolved_v in stats['by_version']:
            ver_str = str(version)[:25] if version else 'N/A'
            acc = f"{correct_v / resolved_v * 100:.1f}%" if resolved_v > 0 else "N/A"
            print(f"  {ver_str:<25} {total_v:>6} {resolved_v:>9} {acc:>9}")
    
    # By source (ML vs ELO)
    if stats['by_source']:
        print(f"\n  BY SOURCE:")
        print(f"  {'Source':<15} {'Total':>6} {'Resolved':>9} {'Accuracy':>9}")
        print("  " + "-" * 42)
        for source, total_s, correct_s, resolved_s in stats['by_source']:
            src = str(source) if source else 'unknown'
            acc = f"{correct_s / resolved_s * 100:.1f}%" if resolved_s > 0 else "N/A"
            print(f"  {src:<15} {total_s:>6} {resolved_s:>9} {acc:>9}")
    
    # By confidence
    if stats['by_confidence']:
        print(f"\n  BY CONFIDENCE LEVEL:")
        print(f"  {'Level':<15} {'Total':>6} {'Resolved':>9} {'Accuracy':>9}")
        print("  " + "-" * 42)
        for conf, total_c, correct_c, resolved_c in stats['by_confidence']:
            c = str(conf) if conf else 'unknown'
            acc = f"{correct_c / resolved_c * 100:.1f}%" if resolved_c > 0 else "N/A"
            print(f"  {c:<15} {total_c:>6} {resolved_c:>9} {acc:>9}")


def print_monthly_stats(months):
    """Print monthly accuracy breakdown."""
    print("\n" + "=" * 60)
    print(f"{'MONTHLY PERFORMANCE':^60}")
    print("=" * 60)
    
    if not months:
        print("\n  No data available.")
        return
    
    print(f"\n  {'Month':<15} {'Total':>6} {'Resolved':>9} {'Correct':>8} {'Accuracy':>9}")
    print("  " + "-" * 50)
    
    for month, total, correct, resolved in months:
        month_str = month.strftime('%Y-%m') if month else 'Unknown'
        if resolved > 0:
            acc_val = correct / resolved * 100
            acc = f"{acc_val:.1f}%"
            # Warn on small sample sizes
            if resolved < 20:
                acc += " *"
        else:
            acc = "N/A"
        print(f"  {month_str:<15} {total:>6} {resolved:>9} {correct:>8} {acc:>9}")
    
    print("\n  * = fewer than 20 resolved predictions (low confidence in accuracy)")


def print_calibration(cal_data):
    """Print calibration analysis."""
    print("\n" + "=" * 60)
    print(f"{'CALIBRATION ANALYSIS':^60}")
    print("=" * 60)
    
    if not cal_data:
        print("\n  No resolved predictions to analyze.")
        return
    
    print(f"\n  {'Predicted':>12} {'Actual':>10} {'Count':>8} {'Gap':>8}")
    print("  " + "-" * 42)
    
    for bin_key in sorted(cal_data.keys()):
        data = cal_data[bin_key]
        if data['count'] < 3:
            continue  # Skip bins with too few samples
        
        predicted_avg = data['predicted_sum'] / data['count']
        actual_avg = data['actual_sum'] / data['count']
        gap = actual_avg - predicted_avg
        
        print(f"  {predicted_avg:>11.1%} {actual_avg:>9.1%} {data['count']:>8d} {gap:>+7.1%}")
    
    print("\n  Interpretation:")
    print("  Positive gap = model is underconfident (actual > predicted)")
    print("  Negative gap = model is overconfident (predicted > actual)")
    print("  Good calibration: gaps close to 0%")


def print_roi(roi_data):
    """Print ROI analysis."""
    print("\n" + "=" * 60)
    print(f"{'ROI ANALYSIS (at -110 odds)':^60}")
    print("=" * 60)
    
    if not roi_data:
        print("\n  No resolved predictions to analyze.")
        return
    
    print(f"\n  Break-even at -110: 52.4%")
    print(f"\n  {'Threshold':>12} {'Bets':>6} {'Win Rate':>10} {'Profit':>8} {'ROI':>8}")
    print("  " + "-" * 48)
    
    for threshold in sorted(roi_data.keys()):
        data = roi_data[threshold]
        if data['bets'] == 0:
            continue
        print(f"  {threshold:>11.0%} {data['bets']:>6d} "
              f"{data['win_rate']:>9.1%} {data['profit']:>+7.1f}u {data['roi'] * 100:>+7.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate prediction performance from PredictionLog',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m ml.eval                    # Overall stats
  python -m ml.eval --resolve          # Resolve predictions with fight outcomes
  python -m ml.eval --monthly          # Monthly accuracy breakdown
  python -m ml.eval --calibration      # Calibration analysis
  python -m ml.eval --roi              # ROI analysis
  python -m ml.eval --all              # Show everything
        '''
    )
    parser.add_argument('--resolve', action='store_true',
                        help='Resolve unresolved predictions with actual fight results')
    parser.add_argument('--monthly', action='store_true',
                        help='Show monthly accuracy breakdown')
    parser.add_argument('--calibration', action='store_true',
                        help='Show calibration analysis')
    parser.add_argument('--roi', action='store_true',
                        help='Show ROI analysis')
    parser.add_argument('--all', action='store_true',
                        help='Show all analyses')
    
    args = parser.parse_args()
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        sys.exit(1)
    
    try:
        # Check if table exists
        if not check_table_exists(conn):
            print("[ERROR] PredictionLog table does not exist.")
            print("        Run migration: psql -d delphi_db -f db/migrations/003_prediction_log.sql")
            sys.exit(1)
        
        # Resolve first if requested
        if args.resolve:
            print("\n" + "=" * 60)
            print(f"{'RESOLVING PREDICTIONS':^60}")
            print("=" * 60)
            resolve_predictions(conn)
        
        # Always show overall stats
        stats = get_prediction_stats(conn)
        print_overall_stats(stats)
        
        # Monthly
        if args.monthly or args.all:
            months = get_monthly_stats(conn)
            print_monthly_stats(months)
        
        # Calibration
        if args.calibration or args.all:
            cal_data = get_calibration_data(conn)
            print_calibration(cal_data)
        
        # ROI
        if args.roi or args.all:
            roi_data = get_roi_data(conn)
            print_roi(roi_data)
        
        print()  # Final newline
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()

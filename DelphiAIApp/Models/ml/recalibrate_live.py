"""
Live Calibration Trainer

Fits a Platt Scaling (logistic regression) calibrator on resolved live
predictions from PredictionTracking and saves it to artifacts/live_calibrator.pkl.

Platt Scaling is preferred over IsotonicRegression for small samples (<200
data points): it fits a smooth sigmoid curve (2 parameters) that generalises
better than the coarse step function IsotonicRegression produces with ~40 OOS
fights.

Once saved, recalibrate_probabilities() in predict_fight.py will automatically
use it for all future predictions (predict_fight and predict_card).

Usage:
    cd DelphiAIApp/Models
    python -m ml.recalibrate_live
    python -m ml.recalibrate_live --include-backtest   # bootstrap from OOS backtest data
"""

import argparse
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import psycopg2
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# DB setup (mirrors predict_fight.py)
# ---------------------------------------------------------------------------
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

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
OUTPUT_PATH = ARTIFACTS_DIR / 'live_calibrator.pkl'

MIN_SAMPLES = 200  # Minimum resolved predictions required to fit calibrator (small samples fit noise)

BUCKET_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
BUCKET_LABELS = ['50-55%', '55-60%', '60-65%', '65-70%', '70%+']


def fetch_predictions(conn, include_backtest=False):
    """
    Fetch resolved predictions from PredictionTracking.

    Args:
        include_backtest: If True, use ONLY backtest predictions (prediction_type='backtest').
                         This bootstraps the calibrator from OOS backtest data using the
                         current model's probability scale — do NOT mix with live predictions
                         from older model runs, which have different probability distributions.
                         Once enough live predictions accumulate from the current model,
                         run without this flag to use live data only.
    """
    cur = conn.cursor()
    if include_backtest:
        cur.execute("""
            SELECT pick_probability, was_correct, prediction_type
            FROM PredictionTracking
            WHERE was_correct IS NOT NULL
              AND prediction_type = 'backtest'
            ORDER BY predicted_at
        """)
    else:
        cur.execute("""
            SELECT pick_probability, was_correct, prediction_type
            FROM PredictionTracking
            WHERE was_correct IS NOT NULL
              AND prediction_type = 'live'
            ORDER BY predicted_at
        """)
    rows = cur.fetchall()
    cur.close()
    return rows


def bucket_label(prob):
    for i, edge in enumerate(BUCKET_EDGES[1:]):
        if prob < edge:
            return BUCKET_LABELS[i]
    return BUCKET_LABELS[-1]


def print_calibration_table(rows, calibrator):
    buckets = defaultdict(lambda: {'probs': [], 'correct': 0, 'total': 0})
    for prob, correct, _ in rows:
        p = float(prob)
        label = bucket_label(p)
        buckets[label]['probs'].append(p)
        buckets[label]['total'] += 1
        if correct:
            buckets[label]['correct'] += 1

    print(f"\n{'Bucket':<10} {'N':>5} {'Avg Pred':>10} {'Actual Acc':>12} {'After Cal':>10} {'Error Before':>14} {'Error After':>12}")
    print("-" * 80)

    for label in BUCKET_LABELS:
        d = buckets[label]
        if d['total'] == 0:
            continue
        avg_pred = np.mean(d['probs'])
        actual = d['correct'] / d['total']
        mapped = float(calibrator.predict_proba([[avg_pred]])[0][1])
        err_before = avg_pred - actual
        err_after = mapped - actual
        print(
            f"{label:<10} {d['total']:>5} {avg_pred:>10.3f} {actual:>12.3f}"
            f" {mapped:>10.3f} {err_before:>+14.3f} {err_after:>+12.3f}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(description='DelphiAI Live Calibration Trainer')
    parser.add_argument(
        '--include-backtest',
        action='store_true',
        help='Also include resolved backtest predictions to bootstrap calibration. '
             'Use this when live predictions are insufficient (< 15). '
             'Only use OOS backtest data (run backtest --year 2026 first).'
    )
    args = parser.parse_args()

    print("=== DelphiAI Live Calibration Trainer ===\n")
    if args.include_backtest:
        print("[INFO] Including backtest predictions (--include-backtest flag set)")
        print("[INFO] Ensure the DB contains OOS-only backtest data (e.g. 2026)\n")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        print("Make sure PostgreSQL is running and .env is configured.")
        sys.exit(1)

    rows = fetch_predictions(conn, include_backtest=args.include_backtest)
    conn.close()

    live_count = sum(1 for r in rows if r[2] == 'live')
    backtest_count = sum(1 for r in rows if r[2] == 'backtest')

    if args.include_backtest:
        print(f"Found {backtest_count} resolved backtest predictions (OOS bootstrap mode).\n")
    else:
        print(f"Found {live_count} resolved live predictions.\n")

    if len(rows) < MIN_SAMPLES:
        print(f"[WARNING] Only {len(rows)} resolved predictions found. "
              f"Need at least {MIN_SAMPLES} for a reliable calibrator.")
        if not args.include_backtest:
            print("Try running with --include-backtest to use OOS backtest data.")
        print("Aborting.")
        sys.exit(1)

    X = np.array([float(r[0]) for r in rows])
    y = np.array([int(r[1]) for r in rows])

    # Platt Scaling: fit a logistic regression sigmoid through the probability space.
    # High C (low regularisation) lets the data shape the curve without shrinkage.
    calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrator.fit(X.reshape(-1, 1), y)

    print_calibration_table(rows, calibrator)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(calibrator, f)

    print(f"[OK] Calibrator saved to {OUTPUT_PATH}")
    print("Future predictions will automatically use this calibrator.")
    print("Re-run this script after each event to keep calibration current.\n")


if __name__ == '__main__':
    main()

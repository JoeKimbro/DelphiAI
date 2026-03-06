"""
Model weakness diagnostics focused on confidence-threshold behavior.

Usage:
    cd DelphiAIApp/Models
    python -m ml.model_weakness_tests --high-threshold 65
"""

import argparse
import math
import os
import random
import re
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5433"),
    "dbname": os.getenv("DB_NAME", "delphi_db"),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
}


def _stat(preds):
    total = len(preds)
    correct = sum(1 for p in preds if p["was_correct"])
    acc = (100.0 * correct / total) if total else 0.0
    return total, correct, acc


def _year_from_event_date(event_date):
    m = re.search(r"(20\d{2})", str(event_date or ""))
    return int(m.group(1)) if m else None


def _sort_key(pred):
    return (str(pred["event_date"] or ""), pred["id"])


def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    den = 1 + (z * z / n)
    center = (ph + (z * z / (2 * n))) / den
    half = z * math.sqrt((ph * (1 - ph) / n) + ((z * z) / (4 * n * n))) / den
    return center - half, center + half


def _fetch_predictions(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, event_name, event_date, COALESCE(prediction_type, 'live') AS ptype,
               pick_probability, was_correct, pick_fighter_id
        FROM PredictionTracking
        WHERE was_correct IS NOT NULL
        ORDER BY id
        """
    )
    cols = [
        "id",
        "event_name",
        "event_date",
        "ptype",
        "pick_probability",
        "was_correct",
        "pick_fighter_id",
    ]
    preds = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()

    for p in preds:
        p["pick_probability"] = (
            float(p["pick_probability"]) if p["pick_probability"] is not None else 0.0
        )
        p["year"] = _year_from_event_date(p["event_date"])
    return preds


def _print_header(threshold):
    print("\n" + "=" * 84)
    print(f"{'MODEL WEAKNESS TESTS @ ' + str(int(threshold * 100)) + '%+ THRESHOLD':^84}")
    print("=" * 84)


def _run_tests(conn, threshold):
    preds = _fetch_predictions(conn)
    back = [p for p in preds if p["ptype"] == "backtest"]
    live = [p for p in preds if p["ptype"] == "live"]

    back_hc = [p for p in back if p["pick_probability"] >= threshold]
    live_hc = [p for p in live if p["pick_probability"] >= threshold]

    print("\nCRITICAL TESTS")
    print("-" * 84)

    # Test 1: Year-by-year
    print("\n1) Year-by-year HIGH confidence (backtest)")
    for year in [2023, 2024, 2025]:
        ypred = [p for p in back_hc if p["year"] == year]
        n, c, a = _stat(ypred)
        print(f"   {year}: {c}/{n} ({a:.1f}%)")

    # Test 2: 2024 holdout
    print("\n2) 2024 holdout HIGH")
    p2024 = [p for p in back_hc if p["year"] == 2024]
    n, c, a = _stat(p2024)
    print(f"   2024 HIGH: {c}/{n} ({a:.1f}%)")

    # Test 3: Last 100 fights
    print("\n3) Last 100 backtest fights (recent proxy)")
    recent100 = sorted(back, key=_sort_key)[-100:]
    recent_hc = [p for p in recent100 if p["pick_probability"] >= threshold]
    n, c, a = _stat(recent_hc)
    print(f"   Last 100 HIGH: {c}/{n} ({a:.1f}%)")

    # Test 4: Monte Carlo / exact tail
    print("\n4) Monte Carlo vs backtest baseline")
    b_n, b_c, b_acc = _stat(back_hc)
    l_n, l_c, l_acc = _stat(live_hc)
    p_true = (b_c / b_n) if b_n else 0.0
    tail = 0.0
    if l_n > 0:
        for k in range(0, l_c + 1):
            tail += math.comb(l_n, k) * (p_true**k) * ((1 - p_true) ** (l_n - k))
    sims = 50000
    hit = 0
    if l_n > 0:
        for _ in range(sims):
            draw_correct = sum(1 for _ in range(l_n) if random.random() < p_true)
            if draw_correct <= l_c:
                hit += 1
    sim_tail = (hit / sims) if l_n > 0 else 0.0
    print(f"   Backtest baseline: {b_c}/{b_n} ({b_acc:.1f}%)")
    print(f"   Live observed:     {l_c}/{l_n} ({l_acc:.1f}%)")
    print(f"   P(X <= {l_c} | n={l_n}, p={p_true:.4f}): exact={tail:.4f}, sim={sim_tail:.4f}")

    print("\nDIAGNOSTIC TESTS")
    print("-" * 84)

    # Test 5: Age analysis
    print("\n5) High confidence age analysis")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(pt.prediction_type, 'live') AS ptype,
               AVG(fs.age::numeric) AS avg_age,
               COUNT(*) AS n
        FROM PredictionTracking pt
        JOIN FighterStats fs ON fs.FighterID = pt.pick_fighter_id
        WHERE pt.was_correct IS NOT NULL
          AND pt.pick_probability >= %s
          AND COALESCE(pt.prediction_type, 'live') IN ('backtest', 'live')
          AND fs.age IS NOT NULL
        GROUP BY COALESCE(pt.prediction_type, 'live')
        ORDER BY ptype
        """,
        (threshold,),
    )
    age_rows = cur.fetchall()
    for ptype, avg_age, n in age_rows:
        print(f"   {ptype:<8} avg_age={float(avg_age):.2f} (n={n})")

    # Test 6: Division
    print("\n6) High confidence by division")
    cur.execute(
        """
        SELECT COALESCE(pt.prediction_type, 'live') AS ptype,
               COALESCE(fs.weightclass, 'Unknown') AS weightclass,
               COUNT(*) AS n,
               SUM(CASE WHEN pt.was_correct THEN 1 ELSE 0 END) AS correct
        FROM PredictionTracking pt
        LEFT JOIN FighterStats fs ON fs.FighterID = pt.pick_fighter_id
        WHERE pt.was_correct IS NOT NULL
          AND pt.pick_probability >= %s
          AND COALESCE(pt.prediction_type, 'live') IN ('backtest', 'live')
        GROUP BY COALESCE(pt.prediction_type, 'live'), COALESCE(fs.weightclass, 'Unknown')
        ORDER BY ptype, n DESC
        """,
        (threshold,),
    )
    div_rows = cur.fetchall()
    for ptype in ["backtest", "live"]:
        print(f"   {ptype}:")
        sub = [r for r in div_rows if r[0] == ptype]
        for _, wc, n, corr in sub[:10]:
            acc = (100.0 * corr / n) if n else 0.0
            print(f"     - {wc}: {corr}/{n} ({acc:.1f}%)")

    # Test 7: Favorite vs underdog
    print("\n7) High confidence favorite vs underdog")
    print("   Not available: PredictionTracking does not store sportsbook odds columns.")

    # Test 8: Specific events
    print("\n8) Specific events (live HIGH)")
    for kw in ["Strickland", "Moreno", "Bautista"]:
        ep = [
            p
            for p in live_hc
            if kw.lower() in (p["event_name"] or "").lower()
        ]
        n, c, a = _stat(ep)
        print(f"   {kw}: {c}/{n} ({a:.1f}%)")

    print("\nTEMPORAL / STATISTICAL TESTS")
    print("-" * 84)

    # Test 9: Rolling windows
    print("\n9) Rolling 50-fight windows (backtest)")
    sorted_back = sorted(back, key=_sort_key)
    windows = []
    for i in range(0, len(sorted_back), 50):
        w = sorted_back[i : i + 50]
        wh = [p for p in w if p["pick_probability"] >= threshold]
        n, c, a = _stat(wh)
        windows.append((i + 1, i + len(w), n, c, a))
        print(f"   Fights {i+1:>3}-{i+len(w):>3}: HIGH {c}/{n} ({a:.1f}%)")
    points = [(idx + 1, w[4]) for idx, w in enumerate(windows) if w[2] > 0]
    if len(points) >= 2:
        mx = sum(x for x, _ in points) / len(points)
        my = sum(y for _, y in points) / len(points)
        num = sum((x - mx) * (y - my) for x, y in points)
        den = sum((x - mx) ** 2 for x, _ in points)
        slope = num / den if den else 0.0
        print(f"   Trend slope (pct points per window): {slope:+.2f}")

    # Test 10: Training overlap check
    print("\n10) Training overlap check")
    print("   Backtest config states: trained pre-2024; 2024 validation; 2025+ holdout.")
    p2025 = [p for p in back_hc if p["year"] == 2025]
    n, c, a = _stat(p2025)
    print(f"   2025 HIGH out-of-sample: {c}/{n} ({a:.1f}%)")

    # Test 11: Confidence intervals
    print("\n11) Confidence intervals")
    lo, hi = _wilson_ci(l_c, l_n) if l_n else (0.0, 0.0)
    print(
        f"   Live HIGH {l_c}/{l_n} => 95% CI: {100*lo:.1f}% to {100*hi:.1f}%"
        if l_n
        else "   No live HIGH picks for CI."
    )

    # Test 12: Power analysis
    print("\n12) Power analysis (detect drop from 75% to 60%)")
    p0, p1, alpha, target_power = 0.75, 0.60, 0.05, 0.80
    required = None
    for n in range(5, 400):
        cstar = None
        for c in range(n + 1):
            h0_tail = sum(
                math.comb(n, k) * (p0**k) * ((1 - p0) ** (n - k))
                for k in range(c + 1)
            )
            if h0_tail <= alpha:
                cstar = c
        if cstar is None:
            continue
        power = sum(
            math.comb(n, k) * (p1**k) * ((1 - p1) ** (n - k))
            for k in range(cstar + 1)
        )
        if power >= target_power:
            required = (n, cstar, power)
            break
    if required:
        n, cstar, power = required
        print(
            f"   Need about {n} HIGH picks (reject if <= {cstar} correct), power={power:.3f}"
        )
    else:
        print("   No solution found up to n=399.")

    cur.close()

    print("\nSUMMARY")
    print("-" * 84)
    print(f"Backtest {int(threshold*100)}%+: {b_acc:.1f}% ({b_c}/{b_n})")
    print(f"Live {int(threshold*100)}%+:     {l_acc:.1f}% ({l_c}/{l_n})")
    print(f"Monte Carlo tail (live or worse under backtest p): {tail:.4f}")
    print("=" * 84)


def main():
    parser = argparse.ArgumentParser(
        description="Run high-confidence weakness diagnostics on PredictionTracking data."
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=65.0,
        help="High-confidence threshold percentage (e.g. 65 for 65%%).",
    )
    args = parser.parse_args()

    threshold = args.high_threshold / 100.0
    if threshold <= 0 or threshold >= 1:
        print("[ERROR] --high-threshold must be between 0 and 100.")
        return

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as exc:
        print(f"[ERROR] Could not connect to database: {exc}")
        return

    try:
        _print_header(threshold)
        _run_tests(conn, threshold)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

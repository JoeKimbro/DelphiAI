"""
UFC Fight Prediction CLI

Simple command-line tool to analyze any UFC matchup.
Now automatically includes inactivity decay and injury adjustments.

Usage:
    python -m ml.predict_fight "Fighter A" "Fighter B"
    python -m ml.predict_fight "Islam Makhachev" "Arman Tsarukyan"
    python -m ml.predict_fight "Fighter A" "Fighter B" --refresh-injuries  # Force fresh injury check
    
    # Interactive mode:
    python -m ml.predict_fight
"""

import os
import sys
import math
import argparse
import logging
import json
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
import psycopg2
from dotenv import load_dotenv

# Suppress INFO logs from injury scraper (only show warnings/errors)
logging.getLogger('ml.injury_scraper').setLevel(logging.WARNING)

# How old can cached injury data be before we refresh (days)
INJURY_CACHE_DAYS = 7

# ML Model - lazy loaded
_ml_predictor = None

def _get_ml_predictor():
    """Get ML predictor singleton (lazy load). Logs errors instead of silencing."""
    global _ml_predictor
    if _ml_predictor is None:
        try:
            try:
                from ml.model_loader import MLPredictor
            except ModuleNotFoundError:
                from model_loader import MLPredictor
            _ml_predictor = MLPredictor()
            if not _ml_predictor.is_available():
                err = _ml_predictor.get_load_error()
                if err:
                    print(f"[ML] Model not available: {err}")
                    print("[ML] Run 'python -m ml.train_model_v3' to train a model")
                _ml_predictor = False
        except Exception as e:
            print(f"[ML] Failed to load predictor: {e}")
            _ml_predictor = False  # Mark as failed so we don't retry
    return _ml_predictor if _ml_predictor is not False else None






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


def get_fighter_data(conn, name):
    """Get all fighter data by name (fuzzy search)."""
    cur = conn.cursor()
    name_l = (name or "").lower()
    # Accent-insensitive fallback for common UFC name variants (e.g., Quinonez vs Quiñonez)
    name_norm = (
        name_l.replace("ñ", "n")
        .replace("ć", "c")
        .replace("č", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("’", "'")
    )
    
    # Try exact match first, then fuzzy
    # Get most recent ELO from EloHistory or CareerStats
    cur.execute('''
        SELECT 
            fs.fighterid, fs.name, fs.nickname,
            fs.height, fs.weight, fs.reach, fs.stance,
            fs.wins, fs.losses, fs.draws,
            fs.placeofbirth,
            cs.slpm, cs.stracc, cs.sapm, cs.strdef,
            cs.tdavg, cs.tdacc, cs.tddef, cs.subavg,
            cs.winsbyko_last5, cs.winsbysub_last5,
            cs.decisionrate, cs.avgfightduration, cs.firstroundfinishrate,
            cs.ko_round1_pct, cs.ko_round2_pct, cs.ko_round3_pct,
            cs.sub_round1_pct, cs.sub_round2_pct, cs.sub_round3_pct,
            COALESCE(
                (SELECT eloafterfight FROM elohistory eh 
                 WHERE eh.fighterid = fs.fighterid 
                 ORDER BY eh.fightdate DESC LIMIT 1),
                cs.elorating,
                1500
            ) as currentelo,
            fs.dayssincelastfight,
            fs.lastfightdate,
            fs.lastinjurycheckdate,
            fs.injurydetails,
            hf.avgopponentelolast3,
            hf.elovelocity,
            hf.currentwinstreak,
            hf.finishratetrending,
            hf.opponentqualitytrending
        FROM fighterstats fs
        LEFT JOIN careerstats cs ON fs.fighterid = cs.fighterid
        LEFT JOIN LATERAL (
            SELECT
                avgopponentelolast3,
                elovelocity,
                currentwinstreak,
                finishratetrending,
                opponentqualitytrending
            FROM fighterhistoricalfeatures h
            WHERE h.fighterid = fs.fighterid
            ORDER BY h.fightdate DESC, h.fightid DESC
            LIMIT 1
        ) hf ON TRUE
        WHERE fs.name ILIKE %s
           OR fs.name ILIKE %s
           OR REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(LOWER(fs.name),
                'ñ','n'),'ć','c'),'č','c'),'š','s'),'ž','z') ILIKE %s
           OR REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(LOWER(fs.name),
                'ñ','n'),'ć','c'),'č','c'),'š','s'),'ž','z') ILIKE %s
        ORDER BY 
            CASE WHEN fs.name ILIKE %s THEN 0 ELSE 1 END,
            fs.wins DESC
        LIMIT 1
    ''', (name, f'%{name}%', name_norm, f'%{name_norm}%', name))


    
    row = cur.fetchone()
    
    if not row:
        return None
    
    return {
        'id': row[0],
        'name': row[1],
        'nickname': row[2],
        'height': row[3],
        'weight': row[4],
        'reach': row[5],
        'stance': row[6],
        'wins': row[7] or 0,
        'losses': row[8] or 0,
        'draws': row[9] or 0,
        'place_of_birth': row[10],
        'slpm': float(row[11]) if row[11] else 0,
        'str_acc': row[12] or '0%',
        'sapm': float(row[13]) if row[13] else 0,
        'str_def': row[14] or '0%',
        'td_avg': float(row[15]) if row[15] else 0,
        'td_acc': row[16] or '0%',
        'td_def': row[17] or '0%',
        'sub_avg': float(row[18]) if row[18] else 0,
        'ko_last5': int(row[19]) if row[19] is not None else 0,
        'sub_last5': int(row[20]) if row[20] is not None else 0,
        'decision_rate': float(row[21]) if row[21] is not None else 50.0,
        'avg_fight_duration': float(row[22]) if row[22] is not None else 12.0,
        'first_round_finish_rate': float(row[23]) if row[23] is not None else 30.0,
        'ko_r1_pct': float(row[24]) if row[24] is not None else 40.0,
        'ko_r2_pct': float(row[25]) if row[25] is not None else 30.0,
        'ko_r3_pct': float(row[26]) if row[26] is not None else 30.0,
        'sub_r1_pct': float(row[27]) if row[27] is not None else 50.0,
        'sub_r2_pct': float(row[28]) if row[28] is not None else 30.0,
        'sub_r3_pct': float(row[29]) if row[29] is not None else 20.0,
        'elo': float(row[30]) if row[30] else 1500.0,
        'days_since_last_fight': int(row[31]) if row[31] is not None else 0,
        'last_fight': row[32],
        'last_injury_check': row[33],
        'injury_details': row[34],
        'avg_opponent_elo_last_3': float(row[35]) if row[35] is not None else float('nan'),
        'elo_velocity': float(row[36]) if row[36] is not None else float('nan'),
        'current_win_streak': int(row[37]) if row[37] is not None else 0,
        'finish_rate_trending': float(row[38]) if row[38] is not None else float('nan'),
        'opponent_quality_trending': float(row[39]) if row[39] is not None else float('nan'),
    }


def elo_to_probability(elo_a, elo_b):
    """Convert ELO ratings to win probability."""
    diff = elo_a - elo_b
    return 1 / (1 + 10 ** (-diff / 400))


def recalibrate_probabilities(prob):
    """
    Optional temperature scaling for calibration.

    Controlled by env var DELPHI_CALIBRATION_T (default 0.62 from backtest tuning).
    T < 1.0 increases confidence; T > 1.0 decreases confidence.
    """
    try:
        t = float(os.getenv("DELPHI_CALIBRATION_T", "0.62"))
    except Exception:
        t = 1.0

    # Disabled path (removes prior manual uplift).
    if abs(t - 1.0) < 1e-9:
        return prob

    # Numerically stable temperature scaling around logit space.
    p = max(1e-6, min(1 - 1e-6, float(prob)))
    logit = math.log(p / (1 - p))
    scaled = 1.0 / (1.0 + math.exp(-(logit / t)))
    return max(0.10, min(0.90, scaled))


def calculate_elo_adjustments(fighter_data, force_injury_refresh=False):
    """
    Build adjusted ELO using inactivity decay + optional injury scrape.

    Returns:
        {
            'raw_elo': float,
            'adjusted_elo': float,
            'injury_penalty': int,
            'inactivity_penalty': int,
        }
    """
    raw_elo = float(fighter_data.get('elo', 1500.0) or 1500.0)
    name = fighter_data.get('name', '')

    # Inactivity decay (ring rust): only decay ratings above baseline.
    days_since = fighter_data.get('days_since_last_fight', 0) or 0
    inactivity_penalty = 0
    if days_since > 180:
        years_inactive = days_since / 365.0
        decay_rate = min(0.05 * years_inactive, 0.25)
        inactivity_penalty = int(max(raw_elo - 1500.0, 0.0) * decay_rate)

    # Injury penalty, prefer cached value unless stale/forced.
    injury_penalty = 0
    last_check = fighter_data.get('last_injury_check')
    injury_details = fighter_data.get('injury_details')
    need_refresh = force_injury_refresh

    if not need_refresh:
        if not last_check:
            need_refresh = True
        else:
            if isinstance(last_check, str):
                try:
                    last_check = datetime.fromisoformat(last_check)
                except Exception:
                    need_refresh = True
            elif hasattr(last_check, "year") and not isinstance(last_check, datetime):
                # date -> datetime for age check
                last_check = datetime.combine(last_check, datetime.min.time())
            if isinstance(last_check, datetime):
                cache_age = (datetime.now() - last_check).days
                if cache_age > INJURY_CACHE_DAYS:
                    need_refresh = True

    if not need_refresh and injury_details:
        try:
            details = injury_details
            if isinstance(details, str):
                details = json.loads(details)
            if isinstance(details, dict):
                injury_penalty = int(
                    details.get('elo_penalty')
                    or details.get('final_penalty')
                    or details.get('penalty')
                    or 0
                )
        except Exception:
            injury_penalty = 0

    if need_refresh:
        try:
            try:
                from ml.injury_scraper import InjuryScraper
            except ModuleNotFoundError:
                from injury_scraper import InjuryScraper
            scraper = InjuryScraper()
            injury_result = scraper.check_fighter_injuries(name, check_news=True)
            if injury_result and injury_result.get('injury_found'):
                injury_penalty = int(injury_result.get('elo_penalty', 0) or 0)
        except Exception:
            # Never break prediction flow due to scrape issues.
            injury_penalty = 0

    adjusted_elo = raw_elo - inactivity_penalty - injury_penalty

    return {
        'raw_elo': raw_elo,
        'adjusted_elo': adjusted_elo,
        'injury_penalty': injury_penalty,
        'inactivity_penalty': inactivity_penalty,
    }


def log_prediction(conn, f1, f2, prob_f1, prob_f2, elo_prob_f1, elo_prob_f2,
                   f1_elo, f2_elo, model_version, prob_source, confidence, 
                   ml_features=None):
    """
    Log a prediction to the PredictionLog table for tracking accuracy over time.
    
    Prevents duplicate predictions for the same fighter pair within 1 hour.
    Silently fails if the table doesn't exist (migration not yet run).
    """
    try:
        import json as _json
        cur = conn.cursor()
        
        # Prevent duplicate predictions for same fight within 1 hour
        cur.execute('''
            SELECT COUNT(*) FROM PredictionLog
            WHERE FighterAName = %s AND FighterBName = %s
              AND CreatedAt > NOW() - INTERVAL '1 hour'
        ''', (f1['name'], f2['name']))
        if cur.fetchone()[0] > 0:
            cur.close()
            return  # Already logged recently
        
        # Determine predicted winner
        if prob_f1 > prob_f2:
            predicted_winner = f1['name']
        else:
            predicted_winner = f2['name']
        
        # Serialize features (truncate if too large to prevent JSONB bloat)
        features_json = None
        if ml_features:
            features_json = _json.dumps(ml_features)
            if len(features_json) > 10000:  # Safety limit
                features_json = _json.dumps({'truncated': True, 'reason': 'too_large'})
        
        cur.execute('''
            INSERT INTO PredictionLog 
                (FighterAName, FighterBName, FighterAID, FighterBID,
                 PredictedWinnerName, ProbabilityA, ProbabilityB,
                 EloProbabilityA, EloProbabilityB, EloA, EloB,
                 ModelVersion, ProbabilitySource, Features, ConfidenceLevel)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            f1['name'], f2['name'], f1.get('id'), f2.get('id'),
            predicted_winner, round(prob_f1, 4), round(prob_f2, 4),
            round(elo_prob_f1, 4), round(elo_prob_f2, 4),
            round(f1_elo, 2), round(f2_elo, 2),
            model_version, prob_source, features_json, confidence
        ))
        conn.commit()
    except Exception:
        # Silently fail - don't break predictions if logging fails
        # (e.g. table doesn't exist yet)
        try:
            conn.rollback()
        except Exception:
            pass



def parse_percentage(pct_str):

    """Parse percentage string to float."""
    if not pct_str:
        return 0.0
    try:
        return float(str(pct_str).replace('%', '')) / 100
    except:
        return 0.0


def predict_sig_strikes(f1, f2):
    """
    Predict who will land more significant strikes.
    
    Uses: SLpM (output) adjusted by opponent's StrDef (defense)
    """
    # Fighter 1's expected sig strikes vs Fighter 2's defense
    f1_def = parse_percentage(f1['str_def'])
    f2_def = parse_percentage(f2['str_def'])
    
    # Adjusted output = SLpM * (1 - opponent_defense)
    # Higher defense = fewer strikes land
    f1_adjusted = f1['slpm'] * (1 - f2_def * 0.5)  # Defense reduces by up to 50%
    f2_adjusted = f2['slpm'] * (1 - f1_def * 0.5)
    
    total = f1_adjusted + f2_adjusted
    if total == 0:
        return 0.5, 0.5
    
    f1_prob = f1_adjusted / total
    f2_prob = f2_adjusted / total
    
    return f1_prob, f2_prob


def predict_takedowns(f1, f2):
    """
    Predict who will land more takedowns.
    
    Uses: TDAvg (attempts) * TDAcc (success) adjusted by opponent's TDDef
    """
    f1_acc = parse_percentage(f1['td_acc'])
    f2_acc = parse_percentage(f2['td_acc'])
    f1_def = parse_percentage(f1['td_def'])
    f2_def = parse_percentage(f2['td_def'])
    
    # Expected TDs = attempts * accuracy * (1 - opponent_defense_factor)
    f1_expected = f1['td_avg'] * f1_acc * (1 - f2_def * 0.3)
    f2_expected = f2['td_avg'] * f2_acc * (1 - f1_def * 0.3)
    
    total = f1_expected + f2_expected
    if total == 0:
        return 0.5, 0.5
    
    return f1_expected / total, f2_expected / total


def predict_method(f1, f2, f1_win_prob, f2_win_prob):
    """
    Predict method of victory and round probabilities.
    
    Returns dict with KO%, SUB%, DEC% for each fighter and round-by-round finish odds.
    """
    total_fights = (f1['wins'] + f1['losses'] + f2['wins'] + f2['losses'])
    if total_fights == 0:
        total_fights = 1
    
    # ---- Method of Victory ----
    # KO power: based on KOs in last 5, strikes landed, and opponent absorbed
    f1_ko_power = (f1['ko_last5'] / 5) * 0.5 + (f1['slpm'] / 6) * 0.3 + (f2['sapm'] / 6) * 0.2
    f2_ko_power = (f2['ko_last5'] / 5) * 0.5 + (f2['slpm'] / 6) * 0.3 + (f1['sapm'] / 6) * 0.2
    
    # Sub threat: based on subs in last 5, sub avg, and opponent TD defense
    f2_td_def = parse_percentage(f2['td_def'])
    f1_td_def = parse_percentage(f1['td_def'])
    f1_sub_threat = (f1['sub_last5'] / 5) * 0.5 + (f1['sub_avg'] / 2) * 0.3 + (1 - f2_td_def) * 0.2
    f2_sub_threat = (f2['sub_last5'] / 5) * 0.5 + (f2['sub_avg'] / 2) * 0.3 + (1 - f1_td_def) * 0.2
    
    # --- Method split for each fighter (IF that fighter wins) ---
    # These show the distribution of HOW each fighter would win,
    # independent of their overall win probability.
    def _method_distribution(ko_power, sub_threat, decision_rate):
        """Return normalized method split that sums to 1.0 (how they win IF they win)."""
        dec_base = decision_rate / 100 if decision_rate else 0.5
        finish_base = 1 - dec_base
        
        ko_raw = ko_power * finish_base
        sub_raw = sub_threat * finish_base
        dec_raw = dec_base
        
        total = ko_raw + sub_raw + dec_raw
        if total == 0:
            return {'ko': 0, 'sub': 0, 'dec': 1.0}
        
        return {
            'ko': ko_raw / total,
            'sub': sub_raw / total,
            'dec': dec_raw / total,
        }
    
    # Method distributions normalized to 100% (how they win IF they win)
    f1_method_dist = _method_distribution(f1_ko_power, f1_sub_threat, f1['decision_rate'])
    f2_method_dist = _method_distribution(f2_ko_power, f2_sub_threat, f2['decision_rate'])
    
    # Weighted methods (each fighter's method * their win probability = overall fight outcome)
    f1_methods = {k: v * f1_win_prob for k, v in f1_method_dist.items()}
    f2_methods = {k: v * f2_win_prob for k, v in f2_method_dist.items()}
    
    # Combined probabilities (should sum to ~1.0)
    ko_total = f1_methods['ko'] + f2_methods['ko']
    sub_total = f1_methods['sub'] + f2_methods['sub']
    dec_total = f1_methods['dec'] + f2_methods['dec']

    
    # ---- Round Probabilities ----
    # Weighted average of both fighters' round finish tendencies
    avg_r1_finish = (f1['first_round_finish_rate'] + f2['first_round_finish_rate']) / 2 / 100
    
    # Avg fight duration gives us a sense of how deep fights go
    avg_duration = (f1['avg_fight_duration'] + f2['avg_fight_duration']) / 2
    
    # Higher finish rates = lower chance of going the distance
    finish_rate = 1 - (f1['decision_rate'] + f2['decision_rate']) / 200
    
    # Round-by-round KO distribution (weighted avg of both fighters)
    ko_r1 = (f1['ko_r1_pct'] * f1_win_prob + f2['ko_r1_pct'] * f2_win_prob) / 100
    ko_r2 = (f1['ko_r2_pct'] * f1_win_prob + f2['ko_r2_pct'] * f2_win_prob) / 100
    ko_r3 = (f1['ko_r3_pct'] * f1_win_prob + f2['ko_r3_pct'] * f2_win_prob) / 100
    
    # Round-by-round SUB distribution
    sub_r1 = (f1['sub_r1_pct'] * f1_win_prob + f2['sub_r1_pct'] * f2_win_prob) / 100
    sub_r2 = (f1['sub_r2_pct'] * f1_win_prob + f2['sub_r2_pct'] * f2_win_prob) / 100
    sub_r3 = (f1['sub_r3_pct'] * f1_win_prob + f2['sub_r3_pct'] * f2_win_prob) / 100
    
    # Total finish probability per round
    r1_finish = ko_r1 + sub_r1
    r2_finish = ko_r2 + sub_r2
    r3_finish = ko_r3 + sub_r3
    
    # Normalize: ensure round finishes + decision = ~1.0
    total_finish = r1_finish + r2_finish + r3_finish
    if total_finish > 0:
        scale = (1 - dec_total) / total_finish if total_finish > 0 else 1
        r1_finish *= scale
        r2_finish *= scale
        r3_finish *= scale
    
    return {
        'ko': ko_total,
        'sub': sub_total,
        'dec': dec_total,
        'f1_methods': f1_methods,       # weighted by win prob (for round calcs)
        'f2_methods': f2_methods,       # weighted by win prob (for round calcs)
        'f1_method_dist': f1_method_dist,  # IF fighter 1 wins, how (sums to 1.0)
        'f2_method_dist': f2_method_dist,  # IF fighter 2 wins, how (sums to 1.0)
        'r1_finish': r1_finish,
        'r2_finish': r2_finish,
        'r3_finish': r3_finish,
        'goes_distance': dec_total,
    }



def format_comparison(label, f1_val, f2_val, f1_name, f2_name, higher_better=True):

    """Format a stat comparison line."""
    if isinstance(f1_val, float):
        f1_str = f"{f1_val:.2f}"
        f2_str = f"{f2_val:.2f}"
    else:
        f1_str = str(f1_val)
        f2_str = str(f2_val)
    
    # Determine advantage
    try:
        f1_num = float(str(f1_val).replace('%', ''))
        f2_num = float(str(f2_val).replace('%', ''))
        if higher_better:
            adv = "<<<" if f1_num > f2_num else (">>>" if f2_num > f1_num else "   ")
        else:
            adv = "<<<" if f1_num < f2_num else (">>>" if f2_num < f1_num else "   ")
    except:
        adv = "   "
    
    return f"  {label:<20} {f1_str:>10} {adv:^5} {f2_str:<10}"


def analyze_matchup(f1, f2, force_injury_refresh=False, conn=None):
    """Run full matchup analysis with automatic ELO adjustments."""

    
    # Calculate ELO adjustments (uses database values + refreshes injuries if stale)
    f1_adj = calculate_elo_adjustments(f1, force_injury_refresh)
    f2_adj = calculate_elo_adjustments(f2, force_injury_refresh)

    
    # Header
    print("\n" + "=" * 70)
    print(f"{'FIGHT PREDICTION':^70}")
    print("=" * 70)
    
    # Fighter names
    f1_display = f"{f1['name']}"
    f2_display = f"{f2['name']}"
    if f1['nickname']:
        f1_display += f" \"{f1['nickname']}\""
    if f2['nickname']:
        f2_display += f" \"{f2['nickname']}\""
    
    print(f"\n  {f1_display}")
    print(f"  vs")
    print(f"  {f2_display}")
    
    # Records
    print(f"\n  {f1['wins']}-{f1['losses']}-{f1['draws']:<20} RECORD {f2['wins']}-{f2['losses']}-{f2['draws']:>20}")
    
    # Calculate win probability using adjusted ELOs
    adj_prob_f1 = elo_to_probability(f1_adj['adjusted_elo'], f2_adj['adjusted_elo'])
    adj_prob_f2 = 1 - adj_prob_f1
    adj_prob_f1 = recalibrate_probabilities(adj_prob_f1)
    adj_prob_f2 = 1 - adj_prob_f1
    
    # Simple ELO & Win Probability display
    print("\n" + "-" * 70)
    print(f"{'PREDICTION':^70}")
    print("-" * 70)
    
    # Show ELO (just the final number)
    print(f"\n  {'Fighter':<25} {'ELO':>8}   {'Win %':>8}")
    print("  " + "-" * 45)
    
    # Build adjustment notes for each fighter
    f1_notes = []
    if f1_adj['inactivity_penalty'] > 0:
        f1_notes.append(f"ring rust")
    if f1_adj['injury_penalty'] > 0:
        f1_notes.append(f"injury")
    
    f2_notes = []
    if f2_adj['inactivity_penalty'] > 0:
        f2_notes.append(f"ring rust")
    if f2_adj['injury_penalty'] > 0:
        f2_notes.append(f"injury")
    
    f1_note_str = f" ({', '.join(f1_notes)})" if f1_notes else ""
    f2_note_str = f" ({', '.join(f2_notes)})" if f2_notes else ""
    
    print(f"  {f1['name'][:25]:<25} {f1_adj['adjusted_elo']:>8.0f}   {adj_prob_f1:>7.1%}{f1_note_str}")
    print(f"  {f2['name'][:25]:<25} {f2_adj['adjusted_elo']:>8.0f}   {adj_prob_f2:>7.1%}{f2_note_str}")

    
    # Use adjusted values for the rest of the analysis
    win_prob_f1 = adj_prob_f1
    win_prob_f2 = adj_prob_f2


    
    # Physical comparison
    print("\n" + "-" * 70)
    print(f"{'PHYSICAL COMPARISON':^70}")
    print("-" * 70)
    print(f"\n  {'STAT':<20} {f1['name'][:10]:>10} {'':^5} {f2['name'][:10]:<10}")
    print("  " + "-" * 50)
    print(format_comparison("Height", f1['height'] or 'N/A', f2['height'] or 'N/A', f1['name'], f2['name']))
    print(format_comparison("Reach", f1['reach'] or 'N/A', f2['reach'] or 'N/A', f1['name'], f2['name']))
    print(format_comparison("Stance", f1['stance'] or 'N/A', f2['stance'] or 'N/A', f1['name'], f2['name']))
    
    # Striking comparison
    print("\n" + "-" * 70)
    print(f"{'STRIKING STATS':^70}")
    print("-" * 70)
    print(f"\n  {'STAT':<20} {f1['name'][:10]:>10} {'':^5} {f2['name'][:10]:<10}")
    print("  " + "-" * 50)
    print(format_comparison("Sig Str/Min", f1['slpm'], f2['slpm'], f1['name'], f2['name']))
    print(format_comparison("Str Accuracy", f1['str_acc'], f2['str_acc'], f1['name'], f2['name']))
    print(format_comparison("Absorbed/Min", f1['sapm'], f2['sapm'], f1['name'], f2['name'], higher_better=False))
    print(format_comparison("Str Defense", f1['str_def'], f2['str_def'], f1['name'], f2['name']))
    
    # Sig strike prediction
    ss_f1, ss_f2 = predict_sig_strikes(f1, f2)
    print(f"\n  +{'-' * 50}+")
    print(f"  |{'SIG STRIKE PREDICTION':^50}|")
    print(f"  |  {f1['name'][:20]:<20} {ss_f1:>6.1%}                  |")
    print(f"  |  {f2['name'][:20]:<20} {ss_f2:>6.1%}                  |")
    print(f"  +{'-' * 50}+")

    
    # Grappling comparison
    print("\n" + "-" * 70)
    print(f"{'GRAPPLING STATS':^70}")
    print("-" * 70)
    print(f"\n  {'STAT':<20} {f1['name'][:10]:>10} {'':^5} {f2['name'][:10]:<10}")
    print("  " + "-" * 50)
    print(format_comparison("TD/15min", f1['td_avg'], f2['td_avg'], f1['name'], f2['name']))
    print(format_comparison("TD Accuracy", f1['td_acc'], f2['td_acc'], f1['name'], f2['name']))
    print(format_comparison("TD Defense", f1['td_def'], f2['td_def'], f1['name'], f2['name']))
    print(format_comparison("Sub/15min", f1['sub_avg'], f2['sub_avg'], f1['name'], f2['name']))
    
    # Takedown prediction
    td_f1, td_f2 = predict_takedowns(f1, f2)
    print(f"\n  +{'-' * 50}+")
    print(f"  |{'TAKEDOWN PREDICTION':^50}|")
    print(f"  |  {f1['name'][:20]:<20} {td_f1:>6.1%}                  |")
    print(f"  |  {f2['name'][:20]:<20} {td_f2:>6.1%}                  |")
    print(f"  +{'-' * 50}+")

    
    # Clean Summary
    print("\n" + "=" * 70)
    print(f"{'FINAL PREDICTIONS':^70}")
    print("=" * 70)
    
    # Determine favorite
    if win_prob_f1 > win_prob_f2:
        favorite = f1['name']
        fav_pct = win_prob_f1
        underdog = f2['name']
        dog_pct = win_prob_f2
    else:
        favorite = f2['name']
        fav_pct = win_prob_f2
        underdog = f1['name']
        dog_pct = win_prob_f1
    
    print(f"\n  WINNER:      {favorite} ({fav_pct:.0%})")
    print(f"  SIG STRIKES: {f1['name'] if ss_f1 > ss_f2 else f2['name']} ({max(ss_f1, ss_f2):.0%})")
    print(f"  TAKEDOWNS:   {f1['name'] if td_f1 > td_f2 else f2['name']} ({max(td_f1, td_f2):.0%})")
    
    print("\n" + "-" * 70)
    print("  ODDS CONVERSION: -200 = 67% | -150 = 60% | +150 = 40% | +200 = 33%")
    print("=" * 70)



def interactive_mode(conn):

    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print(f"{'UFC FIGHT PREDICTOR':^60}")
    print("=" * 60)
    print("\nType two fighter names to analyze a matchup.")
    print("Type 'quit' or 'q' to exit.\n")
    
    while True:
        try:
            f1_input = input("Fighter 1: ").strip()
            if f1_input.lower() in ['quit', 'q', 'exit']:
                print("Goodbye!")
                break
            
            f2_input = input("Fighter 2: ").strip()
            if f2_input.lower() in ['quit', 'q', 'exit']:
                print("Goodbye!")
                break
            
            # Look up fighters
            f1 = get_fighter_data(conn, f1_input)
            f2 = get_fighter_data(conn, f2_input)
            
            if not f1:
                print(f"\n[ERROR] Fighter not found: '{f1_input}'")
                print("Try a different spelling or partial name.\n")
                continue
            
            if not f2:
                print(f"\n[ERROR] Fighter not found: '{f2_input}'")
                print("Try a different spelling or partial name.\n")
                continue
            
            # Run analysis
            analyze_matchup(f1, f2)

            
            print("\nEnter another matchup or 'quit' to exit.\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")



def main():
    parser = argparse.ArgumentParser(
        description='UFC Fight Prediction CLI - Automatically includes ELO adjustments for inactivity and injuries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m ml.predict_fight "Islam Makhachev" "Arman Tsarukyan"        # Simple output
  python -m ml.predict_fight "Jon Jones" "Stipe Miocic" --details       # Detailed breakdown
  python -m ml.predict_fight "Amir Albazi" "Kyoji Horiguchi" -d         # Short flag for details
  python -m ml.predict_fight "Amir Albazi" "Kyoji Horiguchi" --refresh  # Force fresh injury check
  python -m ml.predict_fight                                            # Interactive mode

Note: ELOs automatically include adjustments for inactivity (ring rust) and injuries.
        '''

    )
    parser.add_argument('fighter1', nargs='?', help='First fighter name')
    parser.add_argument('fighter2', nargs='?', help='Second fighter name')
    parser.add_argument('--refresh', '-r', action='store_true',
                        help='Force fresh injury check from UFC.com (slower)')



    
    args = parser.parse_args()
    force_refresh = args.refresh
    
    if force_refresh:
        print("\n[INFO] Forcing fresh injury check from UFC.com...")
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        print("Make sure PostgreSQL is running and .env is configured.")
        sys.exit(1)
    
    try:
        if args.fighter1 and args.fighter2:
            # Direct mode
            f1 = get_fighter_data(conn, args.fighter1)
            f2 = get_fighter_data(conn, args.fighter2)
            
            if not f1:
                print(f"[ERROR] Fighter not found: '{args.fighter1}'")
                sys.exit(1)
            if not f2:
                print(f"[ERROR] Fighter not found: '{args.fighter2}'")
                sys.exit(1)
            
            analyze_matchup(f1, f2, force_injury_refresh=force_refresh, conn=conn)


        else:
            # Interactive mode
            interactive_mode(conn)
    finally:
        conn.close()



if __name__ == '__main__':
    main()


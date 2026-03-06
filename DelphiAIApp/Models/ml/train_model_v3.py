"""
UFC Fight Prediction Model Training Pipeline V3

Improvements over V2:
1. Temporal filtering (2019+ only) - addresses MMA meta drift
2. Proper missing data handling (NaN for XGBoost, median for LR/RF)
3. Outlier winsorization at 1st/99th percentile
4. Correlation analysis - drops redundant features (>0.85)
5. Early stopping for XGBoost with validation set
6. Three-way chronological split: Train 70% / Validation 15% / Holdout 15%
7. Bootstrap confidence intervals on holdout
8. Feature importance (built-in + permutation)
9. Model versioning with timestamps

Usage:
    python -m ml.train_model_v3
    python -m ml.train_model_v3 --min-year 2020    # Override temporal filter
    python -m ml.train_model_v3 --no-filter         # Use all data (not recommended)
"""

import os
import sys
import json
import copy
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    brier_score_loss
)
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from scipy.stats import percentileofscore

# IsotonicCalibrator must be imported from model_loader so pickle can find it
try:
    from ml.model_loader import IsotonicCalibrator
except ModuleNotFoundError:
    from model_loader import IsotonicCalibrator

warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] xgboost not installed. Install with: pip install xgboost")

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

OUTPUT_DIR = Path(__file__).parent / 'artifacts'
OUTPUT_DIR.mkdir(exist_ok=True)

# Temporal filter - only train on modern MMA
MIN_YEAR_DEFAULT = 2019


def connect_db():
    return psycopg2.connect(**DB_CONFIG)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(min_year=None):
    """
    Load training data with point-in-time features.
    
    Uses EloHistory (ELO before fight) and PointInTimeStats (career stats
    before fight) to prevent data leakage.
    
    OpponentQuality is NOT used because it aggregates across ALL fights,
    which leaks future information.
    """
    print("\n" + "=" * 60)
    print("LOADING TRAINING DATA (V3)")
    print("=" * 60)

    conn = connect_db()

    fights_query = """
    SELECT 
        f.FightID, f.Date as fight_date,
        f.FighterID as fighter1_id, f.FighterURL as fighter1_url,
        f.FighterName as fighter1_name,
        f.OpponentID as fighter2_id, f.OpponentURL as fighter2_url,
        f.OpponentName as fighter2_name,
        f.WinnerID, f.Result, f.Method, f.Round, f.IsTitleFight, f.EventName,
        -- Physical stats (constant per fighter, no leakage)
        fs1.Height as f1_height, fs1.Reach as f1_reach, fs1.Stance as f1_stance,
        fs1.DOB as f1_dob,
        fs2.Height as f2_height, fs2.Reach as f2_reach, fs2.Stance as f2_stance,
        fs2.DOB as f2_dob,
        fs1.WeightClass,
        -- ELO at fight time (point-in-time, no leakage)
        e1.EloBeforeFight as f1_elo, e2.EloBeforeFight as f2_elo,
        -- Point-in-time stats (career stats BEFORE this fight, no leakage)
        pit1.FightsBefore as f1_fights, pit1.WinsBefore as f1_wins,
        pit1.LossesBefore as f1_losses, pit1.WinRateBefore as f1_win_rate,
        pit1.PIT_SLpM as f1_slpm, pit1.PIT_StrAcc as f1_stracc,
        pit1.PIT_TDAvg as f1_tdavg,
        pit1.PIT_SubAvg as f1_subavg, pit1.PIT_KDRate as f1_kd_rate,
        pit1.RecentWinRate as f1_recent_form, pit1.FinishRate as f1_finish_rate,
        pit1.AvgFightTime as f1_avg_fight_time,
        -- True historical trend features at this exact fight (point-in-time)
        hf1.AvgOpponentEloLast3 as f1_avg_opp_elo_last3,
        hf1.EloVelocity as f1_elo_velocity,
        hf1.CurrentWinStreak as f1_win_streak,
        hf1.FinishRateTrending as f1_finish_rate_trending,
        hf1.OpponentQualityTrending as f1_opp_quality_trending,
        pit2.FightsBefore as f2_fights, pit2.WinsBefore as f2_wins,
        pit2.LossesBefore as f2_losses, pit2.WinRateBefore as f2_win_rate,
        pit2.PIT_SLpM as f2_slpm, pit2.PIT_StrAcc as f2_stracc,
        pit2.PIT_TDAvg as f2_tdavg,
        pit2.PIT_SubAvg as f2_subavg, pit2.PIT_KDRate as f2_kd_rate,
        pit2.RecentWinRate as f2_recent_form, pit2.FinishRate as f2_finish_rate,
        pit2.AvgFightTime as f2_avg_fight_time,
        hf2.AvgOpponentEloLast3 as f2_avg_opp_elo_last3,
        hf2.EloVelocity as f2_elo_velocity,
        hf2.CurrentWinStreak as f2_win_streak,
        hf2.FinishRateTrending as f2_finish_rate_trending,
        hf2.OpponentQualityTrending as f2_opp_quality_trending
    FROM Fights f
    LEFT JOIN FighterStats fs1 ON f.FighterID = fs1.FighterID
    LEFT JOIN FighterStats fs2 ON f.OpponentID = fs2.FighterID
    LEFT JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    LEFT JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    LEFT JOIN PointInTimeStats pit1 ON f.FighterURL = pit1.FighterURL AND f.Date = pit1.FightDate
    LEFT JOIN PointInTimeStats pit2 ON f.OpponentURL = pit2.FighterURL AND f.Date = pit2.FightDate
    LEFT JOIN FighterHistoricalFeatures hf1 ON f.FighterID = hf1.FighterID AND f.FightID = hf1.FightID
    LEFT JOIN FighterHistoricalFeatures hf2 ON f.OpponentID = hf2.FighterID AND f.FightID = hf2.FightID
    WHERE f.Date IS NOT NULL
    ORDER BY f.Date
    """

    df = pd.read_sql(fights_query, conn)
    conn.close()

    total_loaded = len(df)
    unique_loaded = df['fightid'].nunique()
    print(f"   Loaded {total_loaded} total rows from database ({unique_loaded} unique fights)")

    # Deduplicate mirrored fight records before any split/augmentation.
    if total_loaded != unique_loaded:
        before = len(df)
        df = df.sort_values(['fightid', 'fight_date']).drop_duplicates(
            subset=['fightid'],
            keep='first',
        ).copy()
        removed = before - len(df)
        print(f"   Deduplicated by fight_id: removed {removed} duplicate rows")

    # Temporal filtering
    if min_year:
        df['fight_date'] = pd.to_datetime(df['fight_date'])
        before = len(df)
        df = df[df['fight_date'] >= f'{min_year}-01-01'].copy()
        filtered_out = before - len(df)
        print(f"   Filtered to {min_year}+: {len(df)} fights ({filtered_out} removed)")
    else:
        print(f"   [WARNING] No temporal filter applied - training on ALL data")

    # Check date range
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    print(f"   Date range: {df['fight_date'].min().date()} to {df['fight_date'].max().date()}")

    return df


# =============================================================================
# FEATURE ENGINEERING V3
# =============================================================================

def parse_height_inches(height_str):
    """Convert height string to inches. Returns NaN if missing."""
    if pd.isna(height_str) or height_str == '--':
        return np.nan
    try:
        height_str = str(height_str).replace('"', '').replace("'", ' ').strip()
        parts = height_str.split()
        if len(parts) >= 2:
            return int(parts[0]) * 12 + int(parts[1])
        return int(parts[0]) if parts else np.nan
    except (ValueError, IndexError):
        return np.nan


def parse_reach_inches(reach_str):
    """Convert reach string to inches. Returns NaN if missing."""
    if pd.isna(reach_str) or reach_str == '--':
        return np.nan
    try:
        return float(str(reach_str).replace('"', '').replace("'", '').strip())
    except (ValueError, TypeError):
        return np.nan


def calculate_age_at_fight(dob, fight_date):
    """Calculate fighter's age at the time of fight. Returns NaN if missing."""
    if pd.isna(dob) or pd.isna(fight_date):
        return np.nan
    try:
        dob = pd.to_datetime(dob)
        fight_date = pd.to_datetime(fight_date)
        return (fight_date - dob).days / 365.25
    except (ValueError, TypeError):
        return np.nan


def classify_fighting_style(slpm, tdavg, subavg):
    """Classify fighter style using canonical classifier."""
    if pd.isna(slpm):
        slpm = 3.0
    if pd.isna(tdavg):
        tdavg = 1.5
    if pd.isna(subavg):
        subavg = 0.5
    try:
        from ml.style_classifier import classify_style
    except ModuleNotFoundError:
        from style_classifier import classify_style
    return classify_style(slpm=slpm, td_avg=tdavg, sub_avg=subavg)


def get_style_advantage(style_a, style_b):
    """Style matchup advantage using canonical classifier."""
    try:
        from ml.style_classifier import get_style_matchup_advantage as _get_adv
    except ModuleNotFoundError:
        from style_classifier import get_style_matchup_advantage as _get_adv
    return _get_adv(style_a, style_b)


def engineer_features_v3(fights_df, augment_both=True, seed=42):
    """
    Feature engineering V3 - key changes from V2:
    - Returns NaN for missing values (not 0)
    - No OpponentQuality features (data leakage)
    - Adds strike accuracy differential
    - Adds avg fight time differential
    
    augment_both: If True, include BOTH orientations (A vs B AND B vs A) for
        every fight. This guarantees perfect positional symmetry in the training
        data, eliminating the "fighter A always wins" bias. Doubles the dataset.
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING V3")
    print("=" * 60)

    if augment_both:
        print("   Full augmentation ENABLED (both orientations per fight)")
    else:
        print("   No augmentation (single orientation per fight)")

    fights_df = fights_df.copy()
    fights_df['fight_date'] = pd.to_datetime(fights_df['fight_date'])
    features = []
    fight_count = 0
    skipped_no_winner = 0

    for idx, fight in fights_df.iterrows():
        if pd.isna(fight['winnerid']):
            skipped_no_winner += 1
            continue

        fight_count += 1

        # Include both orientations to guarantee positional symmetry.
        # For each fight, we create two training samples:
        #   (A vs B, target=A_won) and (B vs A, target=B_won)
        # All differential features negate, labels flip. This forces the
        # model to learn from fighter stats, not from position.
        orientations = [False, True] if augment_both else [False]

        for swap in orientations:
            fight_date = fight['fight_date']

            if swap:
                a_prefix, b_prefix = 'f2_', 'f1_'
                a_id, b_id = fight['fighter2_id'], fight['fighter1_id']
                a_name, b_name = fight['fighter2_name'], fight['fighter1_name']
            else:
                a_prefix, b_prefix = 'f1_', 'f2_'
                a_id, b_id = fight['fighter1_id'], fight['fighter2_id']
                a_name, b_name = fight['fighter1_name'], fight['fighter2_name']

            # Target: 1 if fighter A won, 0 otherwise
            target = 1 if fight['winnerid'] == a_id else 0

            # Helper: get value, return NaN if missing (not 0!)
            def get_val(col):
                val = fight.get(col)
                if pd.notna(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return np.nan
                return np.nan

            # === ELO (point-in-time) ===
            a_elo = get_val(f'{a_prefix}elo')
            b_elo = get_val(f'{b_prefix}elo')
            if np.isnan(a_elo):
                a_elo = 1500.0
            if np.isnan(b_elo):
                b_elo = 1500.0

            # === AGE AT FIGHT TIME ===
            a_age = calculate_age_at_fight(fight.get(f'{a_prefix}dob'), fight_date)
            b_age = calculate_age_at_fight(fight.get(f'{b_prefix}dob'), fight_date)

            # === PHYSICAL ===
            a_height = parse_height_inches(fight.get(f'{a_prefix}height'))
            b_height = parse_height_inches(fight.get(f'{b_prefix}height'))
            a_reach = parse_reach_inches(fight.get(f'{a_prefix}reach'))
            b_reach = parse_reach_inches(fight.get(f'{b_prefix}reach'))

            # === POINT-IN-TIME STATS ===
            a_slpm = get_val(f'{a_prefix}slpm')
            b_slpm = get_val(f'{b_prefix}slpm')
            a_stracc = get_val(f'{a_prefix}stracc')
            b_stracc = get_val(f'{b_prefix}stracc')
            a_tdavg = get_val(f'{a_prefix}tdavg')
            b_tdavg = get_val(f'{b_prefix}tdavg')
            a_subavg = get_val(f'{a_prefix}subavg')
            b_subavg = get_val(f'{b_prefix}subavg')
            a_kd_rate = get_val(f'{a_prefix}kd_rate')
            b_kd_rate = get_val(f'{b_prefix}kd_rate')
            a_win_rate = get_val(f'{a_prefix}win_rate')
            b_win_rate = get_val(f'{b_prefix}win_rate')
            a_recent = get_val(f'{a_prefix}recent_form')
            b_recent = get_val(f'{b_prefix}recent_form')
            a_finish = get_val(f'{a_prefix}finish_rate')
            b_finish = get_val(f'{b_prefix}finish_rate')
            a_fights = get_val(f'{a_prefix}fights')
            b_fights = get_val(f'{b_prefix}fights')
            a_wins = get_val(f'{a_prefix}wins')
            b_wins = get_val(f'{b_prefix}wins')
            a_losses = get_val(f'{a_prefix}losses')
            b_losses = get_val(f'{b_prefix}losses')
            a_avg_time = get_val(f'{a_prefix}avg_fight_time')
            b_avg_time = get_val(f'{b_prefix}avg_fight_time')
            a_opp_elo_last3 = get_val(f'{a_prefix}avg_opp_elo_last3')
            b_opp_elo_last3 = get_val(f'{b_prefix}avg_opp_elo_last3')
            a_elo_velocity = get_val(f'{a_prefix}elo_velocity')
            b_elo_velocity = get_val(f'{b_prefix}elo_velocity')
            a_win_streak = get_val(f'{a_prefix}win_streak')
            b_win_streak = get_val(f'{b_prefix}win_streak')
            a_finish_trending = get_val(f'{a_prefix}finish_rate_trending')
            b_finish_trending = get_val(f'{b_prefix}finish_rate_trending')
            a_opp_quality_trending = get_val(f'{a_prefix}opp_quality_trending')
            b_opp_quality_trending = get_val(f'{b_prefix}opp_quality_trending')

            # === STYLE CLASSIFICATION ===
            a_style = classify_fighting_style(a_slpm, a_tdavg, a_subavg)
            b_style = classify_fighting_style(b_slpm, b_tdavg, b_subavg)
            style_advantage = get_style_advantage(a_style, b_style)

            # === STANCE ===
            stance_map = {'Orthodox': 0, 'Southpaw': 1, 'Switch': 2}
            a_stance_val = stance_map.get(fight.get(f'{a_prefix}stance'), 0)
            b_stance_val = stance_map.get(fight.get(f'{b_prefix}stance'), 0)
            southpaw_adv = 1 if (a_stance_val == 1 and b_stance_val == 0) else (
                -1 if (a_stance_val == 0 and b_stance_val == 1) else 0
            )

            # === DIFFERENTIALS (NaN propagates correctly) ===
            elo_diff = a_elo - b_elo
            age_diff = a_age - b_age if (not np.isnan(a_age) and not np.isnan(b_age)) else np.nan
            height_diff = a_height - b_height
            reach_diff = a_reach - b_reach
            slpm_diff = a_slpm - b_slpm
            stracc_diff = a_stracc - b_stracc
            tdavg_diff = a_tdavg - b_tdavg
            subavg_diff = a_subavg - b_subavg
            kd_rate_diff = a_kd_rate - b_kd_rate
            win_rate_diff = a_win_rate - b_win_rate
            recent_diff = a_recent - b_recent
            finish_diff = a_finish - b_finish
            exp_diff = a_fights - b_fights
            avg_time_diff = a_avg_time - b_avg_time
            opp_elo_last3_diff = a_opp_elo_last3 - b_opp_elo_last3
            elo_velocity_diff = a_elo_velocity - b_elo_velocity
            win_streak_diff = a_win_streak - b_win_streak
            finish_trending_diff = a_finish_trending - b_finish_trending
            opp_quality_trending_diff = a_opp_quality_trending - b_opp_quality_trending

            # === NEW CONTEXT FEATURES (V3.1) ===
            avg_elo_level = (a_elo + b_elo) / 2.0
            opponent_elo = b_elo
            opponent_recent_form = b_recent
            opponent_finish_rate = b_finish

            # Prospect/veteran flags from point-in-time record and form.
            a_undefeated_prospect = 1 if (
                not np.isnan(a_fights) and not np.isnan(a_losses) and
                not np.isnan(a_wins) and a_fights >= 5 and a_fights <= 12 and
                a_losses == 0 and a_wins >= 5
            ) else 0
            b_undefeated_prospect = 1 if (
                not np.isnan(b_fights) and not np.isnan(b_losses) and
                not np.isnan(b_wins) and b_fights >= 5 and b_fights <= 12 and
                b_losses == 0 and b_wins >= 5
            ) else 0
            undefeated_prospect_diff = a_undefeated_prospect - b_undefeated_prospect

            a_rising_prospect = 1 if (
                not np.isnan(a_fights) and not np.isnan(a_recent) and not np.isnan(a_win_rate) and
                a_fights <= 10 and a_recent >= 0.67 and a_win_rate >= 0.65
            ) else 0
            b_rising_prospect = 1 if (
                not np.isnan(b_fights) and not np.isnan(b_recent) and not np.isnan(b_win_rate) and
                b_fights <= 10 and b_recent >= 0.67 and b_win_rate >= 0.65
            ) else 0
            rising_prospect_diff = a_rising_prospect - b_rising_prospect

            a_declining_veteran = 1 if (
                not np.isnan(a_fights) and not np.isnan(a_recent) and
                a_fights >= 12 and a_recent <= 0.40
            ) else 0
            b_declining_veteran = 1 if (
                not np.isnan(b_fights) and not np.isnan(b_recent) and
                b_fights >= 12 and b_recent <= 0.40
            ) else 0
            declining_veteran_diff = a_declining_veteran - b_declining_veteran

            # Style/composition features beyond generic style_advantage.
            striker_vs_grappler = 1 if (a_style == 'striker' and b_style == 'grappler') else (
                -1 if (a_style == 'grappler' and b_style == 'striker') else 0
            )
            finisher_vs_decision = 1 if (
                not np.isnan(a_finish) and not np.isnan(b_finish) and a_finish >= 0.60 and b_finish <= 0.35
            ) else (
                -1 if (
                    not np.isnan(a_finish) and not np.isnan(b_finish) and a_finish <= 0.35 and b_finish >= 0.60
                ) else 0
            )

            # "Velocity" proxies from point-in-time stats.
            if not np.isnan(a_recent) and not np.isnan(a_win_rate) and not np.isnan(b_recent) and not np.isnan(b_win_rate):
                form_velocity_diff = (a_recent - a_win_rate) - (b_recent - b_win_rate)
            else:
                form_velocity_diff = np.nan

            a_offense = (
                (a_slpm * max(a_stracc, 0.0)) if (not np.isnan(a_slpm) and not np.isnan(a_stracc)) else np.nan
            )
            b_offense = (
                (b_slpm * max(b_stracc, 0.0)) if (not np.isnan(b_slpm) and not np.isnan(b_stracc)) else np.nan
            )
            if not np.isnan(a_offense) and not np.isnan(b_offense):
                offensive_pressure_diff = a_offense - b_offense
            else:
                offensive_pressure_diff = np.nan

            # === POLYNOMIAL FEATURES ===
            elo_diff_sq = elo_diff ** 2 * np.sign(elo_diff)
            if not np.isnan(age_diff):
                age_diff_sq = age_diff ** 2 * np.sign(age_diff)
            else:
                age_diff_sq = np.nan

            # === INTERACTION FEATURES ===
            if not np.isnan(a_age) and not np.isnan(b_age):
                a_prime = 1.0 if 25 <= a_age <= 32 else 0.8
                b_prime = 1.0 if 25 <= b_age <= 32 else 0.8
                elo_prime_interaction = elo_diff * (a_prime - b_prime)
            else:
                elo_prime_interaction = np.nan

            # Reach x Striking (antisymmetric: abs(reach_diff) * slpm_diff)
            if not np.isnan(reach_diff) and not np.isnan(slpm_diff):
                reach_striking_interaction = abs(reach_diff) * slpm_diff / 10.0
            else:
                reach_striking_interaction = np.nan

            # === FIGHT CONTEXT ===
            is_title = 1 if fight.get('istitlefight') else 0
            is_debut_a = 1 if (not np.isnan(a_fights) and a_fights == 0) else 0
            is_debut_b = 1 if (not np.isnan(b_fights) and b_fights == 0) else 0
            debut_diff = is_debut_a - is_debut_b

            features.append({
                'fight_id': fight['fightid'],
                'fight_date': fight_date,
                'fighter_a_name': a_name,
                'fighter_b_name': b_name,
                'target': target,
                'swapped': swap,
                'elo_diff': elo_diff,
                'age_diff': age_diff,
                'height_diff': height_diff,
                'reach_diff': reach_diff,
                'slpm_diff': slpm_diff,
                'stracc_diff': stracc_diff,
                'tdavg_diff': tdavg_diff,
                'subavg_diff': subavg_diff,
                'kd_rate_diff': kd_rate_diff,
                'win_rate_diff': win_rate_diff,
                'recent_form_diff': recent_diff,
                'finish_rate_diff': finish_diff,
                'experience_diff': exp_diff,
                'avg_fight_time_diff': avg_time_diff,
                'opp_elo_last3_diff': opp_elo_last3_diff,
                'elo_velocity_diff': elo_velocity_diff,
                'win_streak_diff': win_streak_diff,
                'finish_trending_diff': finish_trending_diff,
                'opp_quality_trending_diff': opp_quality_trending_diff,
                'avg_elo_level': avg_elo_level,
                'opponent_elo': opponent_elo,
                'opponent_recent_form': opponent_recent_form,
                'opponent_finish_rate': opponent_finish_rate,
                'style_advantage': style_advantage,
                'southpaw_advantage': southpaw_adv,
                'undefeated_prospect_diff': undefeated_prospect_diff,
                'rising_prospect_diff': rising_prospect_diff,
                'declining_veteran_diff': declining_veteran_diff,
                'striker_vs_grappler': striker_vs_grappler,
                'finisher_vs_decision': finisher_vs_decision,
                'form_velocity_diff': form_velocity_diff,
                'offensive_pressure_diff': offensive_pressure_diff,
                'elo_diff_sq': elo_diff_sq,
                'age_diff_sq': age_diff_sq,
                'elo_prime_interaction': elo_prime_interaction,
                'reach_striking_interaction': reach_striking_interaction,
                'is_title_fight': is_title,
                'debut_diff': debut_diff,
            })

    df = pd.DataFrame(features)
    total = len(df)

    print(f"\n   Source fights: {fight_count}")
    print(f"   Skipped {skipped_no_winner} fights (no winner)")
    print(f"   Training samples: {total}" +
          (f" ({total // fight_count}x augmented)" if augment_both and fight_count > 0 else ""))
    print(f"   Class balance: {df['target'].mean() * 100:.1f}% Fighter A wins")

    return df


def get_feature_columns_v3():
    """
    Feature columns for V3 model.
    
    V3.1 feature set with additional context/prospect/style/velocity features.
    """
    return [
        # Base differentials
        'elo_diff', 'age_diff', 'height_diff', 'reach_diff',
        'slpm_diff', 'stracc_diff', 'tdavg_diff', 'subavg_diff',
        'kd_rate_diff', 'win_rate_diff', 'recent_form_diff',
        'finish_rate_diff', 'experience_diff', 'avg_fight_time_diff',
        'opp_elo_last3_diff', 'elo_velocity_diff', 'win_streak_diff',
        'finish_trending_diff', 'opp_quality_trending_diff',
        # Opponent quality / context
        'avg_elo_level', 'opponent_elo', 'opponent_recent_form', 'opponent_finish_rate',
        # Style/stance (2)
        'style_advantage', 'southpaw_advantage',
        # Prospect / momentum / style matchup
        'undefeated_prospect_diff', 'rising_prospect_diff', 'declining_veteran_diff',
        'striker_vs_grappler', 'finisher_vs_decision', 'form_velocity_diff',
        'offensive_pressure_diff',
        # Polynomial (2)
        'elo_diff_sq', 'age_diff_sq',
        # Interactions (2)
        'elo_prime_interaction', 'reach_striking_interaction',
        # Context (2)
        'is_title_fight', 'debut_diff',
    ]


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def analyze_missing_data(df, feature_cols):
    """Report missing data per feature."""
    print("\n" + "=" * 60)
    print("MISSING DATA ANALYSIS")
    print("=" * 60)

    missing_info = {}
    for col in feature_cols:
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        missing_info[col] = {'count': n_missing, 'pct': pct}
        if pct > 0:
            print(f"   {col:30s}: {n_missing:5d} missing ({pct:.1f}%)")

    total_any = df[feature_cols].isna().any(axis=1).sum()
    print(f"\n   Rows with ANY missing value: {total_any} ({total_any / len(df) * 100:.1f}%)")
    print(f"   Rows fully complete: {len(df) - total_any}")

    return missing_info


def winsorize_features(df, feature_cols, lower_pct=1, upper_pct=99):
    """
    Cap extreme values at percentile boundaries.
    Operates in-place on a copy. Returns the modified dataframe and bounds.
    """
    print("\n" + "=" * 60)
    print(f"WINSORIZATION ({lower_pct}th / {upper_pct}th percentile)")
    print("=" * 60)

    df = df.copy()
    bounds = {}
    total_capped = 0

    for col in feature_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        lower_bound = np.percentile(series, lower_pct)
        upper_bound = np.percentile(series, upper_pct)
        bounds[col] = {'lower': float(lower_bound), 'upper': float(upper_bound)}

        # Count values that will be capped
        n_low = (df[col] < lower_bound).sum()
        n_high = (df[col] > upper_bound).sum()
        n_capped = n_low + n_high

        if n_capped > 0:
            total_capped += n_capped
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"   {col:30s}: capped {n_capped} values "
                  f"(low={n_low}, high={n_high}) [{lower_bound:.2f}, {upper_bound:.2f}]")

    print(f"\n   Total values capped: {total_capped}")
    return df, bounds


def analyze_correlations(df, feature_cols, threshold=0.85):
    """
    Find and report highly correlated feature pairs.
    Returns list of features to drop.
    """
    print("\n" + "=" * 60)
    print(f"CORRELATION ANALYSIS (threshold={threshold})")
    print("=" * 60)

    # Compute correlation matrix (only on non-NaN values)
    corr_matrix = df[feature_cols].corr()

    # Find highly correlated pairs
    to_drop = set()
    high_corr_pairs = []

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > threshold:
                col_i = feature_cols[i]
                col_j = feature_cols[j]
                high_corr_pairs.append((col_i, col_j, corr))

                # Drop the one with lower univariate correlation with target
                corr_i = abs(df[col_i].corr(df['target']))
                corr_j = abs(df[col_j].corr(df['target']))

                drop_col = col_j if corr_i >= corr_j else col_i
                keep_col = col_i if corr_i >= corr_j else col_j
                to_drop.add(drop_col)

                print(f"   {col_i} <-> {col_j}: r={corr:.3f}")
                print(f"      Dropping '{drop_col}' (target corr={min(corr_i, corr_j):.3f}), "
                      f"keeping '{keep_col}' (target corr={max(corr_i, corr_j):.3f})")

    if not high_corr_pairs:
        print("   No highly correlated pairs found")

    print(f"\n   Features to drop: {list(to_drop) if to_drop else 'none'}")
    return list(to_drop)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_with_calibration(X_train, y_train, X_test, y_test, model_name, base_model):
    """Train model with probability calibration (isotonic regression)."""
    print(f"\n--- {model_name} ---")

    # Calibrate using isotonic regression with cross-validation
    calibrated = CalibratedClassifierCV(clone(base_model), method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)

    # Predictions
    train_probs = calibrated.predict_proba(X_train)[:, 1]
    test_probs = calibrated.predict_proba(X_test)[:, 1]
    train_preds = (train_probs > 0.5).astype(int)
    test_preds = (test_probs > 0.5).astype(int)

    # Metrics
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    train_ll = log_loss(y_train, train_probs)
    test_ll = log_loss(y_test, test_probs)
    test_brier = brier_score_loss(y_test, test_probs)

    # Also train uncalibrated for Brier comparison
    base_fitted = clone(base_model)
    base_fitted.fit(X_train, y_train)
    uncal_brier = brier_score_loss(y_test, base_fitted.predict_proba(X_test)[:, 1])

    gap = train_auc - test_auc

    print(f"   Train: Acc={train_acc:.3f}, AUC={train_auc:.3f}, LogLoss={train_ll:.4f}")
    print(f"   Test:  Acc={test_acc:.3f}, AUC={test_auc:.3f}, LogLoss={test_ll:.4f}")
    print(f"   Brier: {uncal_brier:.4f} -> {test_brier:.4f} (calibrated)")
    print(f"   Overfitting gap: {gap:.3f}", end="")
    if gap > 0.05:
        print(" [WARNING: >5%]")
    elif gap > 0.03:
        print(" [marginal]")
    else:
        print(" [OK]")

    return {
        'name': model_name,
        'model': calibrated,
        'base_model': base_fitted,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_logloss': train_ll,
        'test_logloss': test_ll,
        'brier_uncal': float(uncal_brier),
        'brier_cal': float(test_brier),
    }


def train_xgboost_early_stopping(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train XGBoost with early stopping on validation set.
    This prevents overfitting by stopping when validation loss stops improving.
    """
    if not HAS_XGBOOST:
        print("\n[SKIP] XGBoost not available")
        return None

    print(f"\n--- XGBoost (Early Stopping) ---")

    model = xgb.XGBClassifier(
        n_estimators=500,  # High ceiling, early stopping will cut it
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_round = model.best_iteration
    print(f"   Best iteration: {best_round} / 500")

    # Now calibrate the fitted model (manual isotonic, sklearn 1.6+ compatible)
    calibrated = IsotonicCalibrator(model)
    calibrated.fit(X_val, y_val)

    # Metrics on test set
    train_probs = calibrated.predict_proba(X_train)[:, 1]
    test_probs = calibrated.predict_proba(X_test)[:, 1]
    train_preds = (train_probs > 0.5).astype(int)
    test_preds = (test_probs > 0.5).astype(int)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    train_ll = log_loss(y_train, train_probs)
    test_ll = log_loss(y_test, test_probs)
    uncal_brier = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
    test_brier = brier_score_loss(y_test, test_probs)

    gap = train_auc - test_auc

    print(f"   Train: Acc={train_acc:.3f}, AUC={train_auc:.3f}, LogLoss={train_ll:.4f}")
    print(f"   Test:  Acc={test_acc:.3f}, AUC={test_auc:.3f}, LogLoss={test_ll:.4f}")
    print(f"   Brier: {uncal_brier:.4f} -> {test_brier:.4f} (calibrated)")
    print(f"   Overfitting gap: {gap:.3f}", end="")
    if gap > 0.05:
        print(" [WARNING: >5%]")
    elif gap > 0.03:
        print(" [marginal]")
    else:
        print(" [OK]")

    return {
        'name': 'XGBoost (EarlyStopping + Calibrated)',
        'model': calibrated,
        'base_model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_logloss': train_ll,
        'test_logloss': test_ll,
        'brier_uncal': float(uncal_brier),
        'brier_cal': float(test_brier),
        'best_iteration': best_round,
    }


def train_ensemble(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train stacking ensemble."""
    print("\n" + "=" * 60)
    print("STACKING ENSEMBLE")
    print("=" * 60)

    estimators = [
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=30,
            min_samples_split=50, random_state=42
        )),
    ]

    if HAS_XGBOOST:
        estimators.append(
            ('xgb', xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.03,
                min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0,
                subsample=0.6, colsample_bytree=0.6, random_state=42
            ))
        )

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.5, max_iter=1000),
        cv=5,
        passthrough=False
    )

    stacking.fit(X_train, y_train)

    # Predictions
    train_probs = stacking.predict_proba(X_train)[:, 1]
    test_probs = stacking.predict_proba(X_test)[:, 1]
    train_preds = (train_probs > 0.5).astype(int)
    test_preds = (test_probs > 0.5).astype(int)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    test_brier = brier_score_loss(y_test, test_probs)
    test_ll = log_loss(y_test, test_probs)
    train_ll = log_loss(y_train, train_probs)
    gap = train_auc - test_auc

    print(f"   Train: Acc={train_acc:.3f}, AUC={train_auc:.3f}")
    print(f"   Test:  Acc={test_acc:.3f}, AUC={test_auc:.3f}")
    print(f"   Brier Score: {test_brier:.4f}")
    print(f"   Overfitting gap: {gap:.3f}", end="")
    if gap > 0.05:
        print(" [WARNING: >5%]")
    else:
        print(" [OK]")

    return {
        'name': 'Stacking Ensemble',
        'model': stacking,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_logloss': train_ll,
        'test_logloss': test_ll,
        'brier_cal': float(test_brier),
    }


# =============================================================================
# EVALUATION
# =============================================================================

def bootstrap_confidence_interval(y_true, y_probs, n_bootstrap=1000, seed=42):
    """
    Bootstrap confidence intervals for accuracy and AUC.
    Returns dict with mean, lower, upper for each metric.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    accuracies = []
    aucs = []
    briers = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_probs[idx]

        # Skip degenerate samples (all same class)
        if len(np.unique(y_b)) < 2:
            continue

        preds_b = (p_b > 0.5).astype(int)
        accuracies.append(accuracy_score(y_b, preds_b))
        aucs.append(roc_auc_score(y_b, p_b))
        briers.append(brier_score_loss(y_b, p_b))

    return {
        'accuracy': {
            'mean': float(np.mean(accuracies)),
            'lower': float(np.percentile(accuracies, 2.5)),
            'upper': float(np.percentile(accuracies, 97.5)),
        },
        'auc': {
            'mean': float(np.mean(aucs)),
            'lower': float(np.percentile(aucs, 2.5)),
            'upper': float(np.percentile(aucs, 97.5)),
        },
        'brier': {
            'mean': float(np.mean(briers)),
            'lower': float(np.percentile(briers, 2.5)),
            'upper': float(np.percentile(briers, 97.5)),
        },
    }


def split_by_fight_id(features_df, train_frac=0.70, val_frac=0.15):
    """
    Chronological split by unique fight_id, not by augmented rows.
    """
    fights = (
        features_df[['fight_id', 'fight_date']]
        .drop_duplicates('fight_id')
        .sort_values(['fight_date', 'fight_id'])
        .reset_index(drop=True)
    )
    n_fights = len(fights)
    train_end = int(n_fights * train_frac)
    val_end = int(n_fights * (train_frac + val_frac))

    train_ids = set(fights.iloc[:train_end]['fight_id'])
    val_ids = set(fights.iloc[train_end:val_end]['fight_id'])
    holdout_ids = set(fights.iloc[val_end:]['fight_id'])

    train_df = features_df[features_df['fight_id'].isin(train_ids)].copy()
    val_df = features_df[features_df['fight_id'].isin(val_ids)].copy()
    holdout_df = features_df[features_df['fight_id'].isin(holdout_ids)].copy()
    return train_df, val_df, holdout_df


def _collapse_holdout_to_fight_level(holdout_meta, probs):
    """
    Collapse augmented rows into one probability per fight.
    """
    meta = holdout_meta[['fight_id', 'fight_date', 'swapped', 'target']].copy().reset_index(drop=True)
    meta['prob_raw'] = probs
    meta['prob_f1'] = np.where(meta['swapped'], 1.0 - meta['prob_raw'], meta['prob_raw'])

    rows = []
    for fight_id, g in meta.groupby('fight_id'):
        non_swapped = g[g['swapped'] == False]
        if len(non_swapped) > 0:
            y_true = int(round(float(non_swapped['target'].mean())))
        else:
            y_true = int(round(float(g['target'].mean())))

        p_f1 = float(g['prob_f1'].mean())
        rows.append({
            'fight_id': int(fight_id),
            'fight_date': g['fight_date'].min(),
            'y_true': y_true,
            'p_fighter1': p_f1,
            'pred': 1 if p_f1 > 0.5 else 0,
            'row_count': int(len(g)),
        })

    fight_df = pd.DataFrame(rows).sort_values(['fight_date', 'fight_id']).reset_index(drop=True)
    return fight_df


def evaluate_holdout_production_pipeline(conn, holdout_fights, sample_count=5):
    """
    Evaluate holdout using the same prediction path as backtest/production.
    """
    try:
        from ml.backtest import (
            get_historical_elo, get_pit_stats, get_historical_features,
            _apply_pit_override, _apply_historical_features_override, _predict_single_fight
        )
        from ml.predict_fight import get_fighter_data
    except ModuleNotFoundError:
        from backtest import (
            get_historical_elo, get_pit_stats, get_historical_features,
            _apply_pit_override, _apply_historical_features_override, _predict_single_fight
        )
        from predict_fight import get_fighter_data

    records = []
    skipped = 0

    for _, fight in holdout_fights.sort_values('fight_date').iterrows():
        f1 = get_fighter_data(conn, fight['fighter1_name'])
        f2 = get_fighter_data(conn, fight['fighter2_name'])
        if not f1 or not f2:
            skipped += 1
            continue

        fight_date = fight['fight_date']

        # Point-in-time overrides exactly like backtest.
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

        pred = _predict_single_fight(conn, f1, f2, is_title=bool(fight.get('istitlefight')))
        if not pred:
            skipped += 1
            continue

        y_true = 1 if fight['winnerid'] == fight['fighter1_id'] else 0
        p_f1 = float(pred['f1_prob'])
        records.append({
            'fight_id': int(fight['fightid']),
            'fight_date': fight_date,
            'fighter1_name': fight['fighter1_name'],
            'fighter2_name': fight['fighter2_name'],
            'p_fighter1': p_f1,
            'y_true': int(y_true),
            'pred': 1 if p_f1 > 0.5 else 0,
            'pick': pred['pick'],
            'prob_source': pred['prob_source'],
        })

    if not records:
        return {'accuracy': None, 'fight_count': 0, 'skipped': skipped, 'samples': []}

    prod_df = pd.DataFrame(records).sort_values('fight_date').reset_index(drop=True)
    acc = accuracy_score(prod_df['y_true'].values, prod_df['pred'].values)
    samples = prod_df.head(sample_count).to_dict('records')
    return {
        'accuracy': float(acc),
        'fight_count': int(len(prod_df)),
        'skipped': int(skipped),
        'samples': samples,
    }


def evaluate_on_holdout(model, X_holdout, y_holdout, feature_cols, holdout_meta, scaler=None):
    """
    Final evaluation on holdout set.
    Canonical metrics are fight-level (one prediction per fight).
    """
    print("\n" + "=" * 60)
    print("HOLDOUT SET EVALUATION (Final, Untouched)")
    print("=" * 60)

    probs = model.predict_proba(X_holdout)[:, 1]
    row_preds = (probs > 0.5).astype(int)
    row_acc = accuracy_score(y_holdout, row_preds)

    holdout_fights = _collapse_holdout_to_fight_level(holdout_meta, probs)
    y_fight = holdout_fights['y_true'].values
    p_fight = holdout_fights['p_fighter1'].values
    pred_fight = holdout_fights['pred'].values

    acc = accuracy_score(y_fight, pred_fight)
    auc = roc_auc_score(y_fight, p_fight)
    ll = log_loss(y_fight, p_fight)
    brier = brier_score_loss(y_fight, p_fight)

    print(f"\n   Holdout Accuracy (fight-level): {acc:.3f}")
    print(f"   Holdout AUC:       {auc:.3f}")
    print(f"   Holdout Log Loss:  {ll:.4f}")
    print(f"   Holdout Brier:     {brier:.4f}")
    print(f"   Holdout Row Accuracy (diagnostic): {row_acc:.3f}")

    # Bootstrap CI
    print("\n   Bootstrap Confidence Intervals (95%, n=1000):")
    ci = bootstrap_confidence_interval(y_fight, p_fight)

    acc_ci = ci['accuracy']
    auc_ci = ci['auc']
    brier_ci = ci['brier']

    print(f"   Accuracy: {acc_ci['mean']:.3f} [{acc_ci['lower']:.3f}, {acc_ci['upper']:.3f}]")
    print(f"   AUC:      {auc_ci['mean']:.3f} [{auc_ci['lower']:.3f}, {auc_ci['upper']:.3f}]")
    print(f"   Brier:    {brier_ci['mean']:.4f} [{brier_ci['lower']:.4f}, {brier_ci['upper']:.4f}]")

    print("\n   Calibration Check:")
    print(f"   {'Predicted':>12} {'Actual':>10} {'Count':>8} {'Gap':>8}")
    print("   " + "-" * 42)

    bins = [(0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.0)]
    for low, high in bins:
        mask = (p_fight >= low) & (p_fight < high)
        if mask.sum() > 0:
            actual = y_fight[mask].mean()
            predicted = p_fight[mask].mean()
            gap = actual - predicted
            print(f"   {predicted:>11.1%} {actual:>9.1%} {mask.sum():>8d} {gap:>+7.1%}")

    print("\n   Betting Simulation (holdout):")
    VIG_BREAKEVEN = 0.524
    MIN_EDGE = 0.05
    MIN_PROB = VIG_BREAKEVEN + MIN_EDGE
    DECIMAL_ODDS = 1.91

    bets = 0
    wins = 0
    profit = 0.0

    for i in range(len(y_fight)):
        prob_a = p_fight[i]
        prob_b = 1 - prob_a

        if prob_a >= MIN_PROB:
            bets += 1
            if y_fight[i] == 1:
                wins += 1
                profit += (DECIMAL_ODDS - 1)
            else:
                profit -= 1
        elif prob_b >= MIN_PROB:
            bets += 1
            if y_fight[i] == 0:
                wins += 1
                profit += (DECIMAL_ODDS - 1)
            else:
                profit -= 1

    if bets > 0:
        roi = profit / bets
        win_rate = wins / bets
        print(f"   Threshold: {MIN_PROB:.1%} (break-even: {VIG_BREAKEVEN:.1%})")
        print(f"   Bets: {bets} ({bets / len(y_fight) * 100:.0f}% of fights)")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   ROI: {roi * 100:+.1f}%")
        if roi > 0.15:
            print("   [!] ROI > 15% - verify no leakage")
        elif roi > 0:
            print("   [OK] Positive ROI")
        else:
            print("   [!] Negative ROI")
    else:
        print("   No bets placed above threshold")
        roi = 0.0

    return {
        'accuracy': float(acc),
        'row_accuracy': float(row_acc),
        'auc': float(auc),
        'log_loss': float(ll),
        'brier': float(brier),
        'confidence_intervals': ci,
        'roi': float(roi) if bets > 0 else None,
        'bets': bets,
        'holdout_size': int(len(y_fight)),
        'holdout_row_count': int(len(y_holdout)),
        'sample_predictions': holdout_fights[['fight_id', 'fight_date', 'p_fighter1', 'pred', 'y_true']]
            .head(5)
            .to_dict('records'),
    }


def analyze_feature_importance(model, X_test, y_test, feature_cols):
    """
    Analyze feature importance using multiple methods.
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    importance_data = {}

    # Method 1: Built-in importance (for tree models)
    base = model
    if hasattr(model, 'calibrated_classifiers_'):
        # CalibratedClassifierCV wraps the base model
        base = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, 'base_estimator'):
        # IsotonicCalibrator wraps the base model
        base = model.base_estimator
    elif hasattr(model, 'estimator'):
        base = model.estimator

    if hasattr(base, 'feature_importances_'):
        importances = base.feature_importances_
        print("\n   Built-in Feature Importance:")
        sorted_idx = np.argsort(importances)[::-1]
        for i in sorted_idx:
            bar = "#" * int(importances[i] * 100)
            print(f"   {feature_cols[i]:30s}: {importances[i]:.4f} {bar}")
            importance_data[feature_cols[i]] = float(importances[i])
    
    # Method 2: Permutation importance (model-agnostic)
    print("\n   Permutation Importance (on test set):")
    try:
        perm_result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=42, scoring='roc_auc'
        )
        sorted_idx = perm_result.importances_mean.argsort()[::-1]
        for i in sorted_idx:
            mean = perm_result.importances_mean[i]
            std = perm_result.importances_std[i]
            if mean > 0.001:  # Only show meaningful features
                print(f"   {feature_cols[i]:30s}: {mean:.4f} (+/- {std:.4f})")
    except Exception as e:
        print(f"   [WARNING] Permutation importance failed: {e}")

    return importance_data


# =============================================================================
# MODEL SAVING
# =============================================================================

def save_model(model, scaler, feature_cols, metrics, winsorize_bounds, 
               dropped_features, min_year, median_values):
    """
    Save model with versioning and full metadata.
    """
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_version = f"v3_{timestamp}"

    # Save model pickle
    model_filename = f"model_{model_version}.pkl"
    model_path = OUTPUT_DIR / model_filename
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'version': model_version,
            'winsorize_bounds': winsorize_bounds,
            'dropped_features': dropped_features,
            'min_year': min_year,
            'median_values': median_values,
        }, f)
    print(f"   Model saved: {model_path}")

    # Also save as model_latest.pkl for easy loading
    latest_path = OUTPUT_DIR / 'model_latest.pkl'
    with open(latest_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'version': model_version,
            'winsorize_bounds': winsorize_bounds,
            'dropped_features': dropped_features,
            'min_year': min_year,
            'median_values': median_values,
        }, f)
    print(f"   Latest symlink: {latest_path}")

    # Save training info JSON (comprehensive metadata for reproducibility)
    info = {
        'model_version': model_version,
        'model_filename': model_filename,
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'min_year_filter': min_year,
        'dropped_features': dropped_features,
        'winsorize_bounds': winsorize_bounds,
        'median_values': {k: float(v) for k, v in median_values.items()},
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
        # Environment info for reproducibility (pickle/version mismatch prevention)
        'python_version': sys.version,
        'xgboost_version': xgb.__version__ if HAS_XGBOOST else None,
        'sklearn_version': __import__('sklearn').__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        # Hyperparameters for the best model
        'hyperparameters': {
            'xgboost': {
                'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.03,
                'min_child_weight': 10, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
                'subsample': 0.7, 'colsample_bytree': 0.7, 'early_stopping_rounds': 20,
            },
            'logistic_regression': {'C': 0.05, 'max_iter': 1000},
            'random_forest': {
                'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 30,
                'min_samples_split': 60, 'max_features': 'sqrt',
            },
        },
    }

    info_path = OUTPUT_DIR / f'training_info_{model_version}.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    print(f"   Training info: {info_path}")

    # Also save as latest info
    latest_info_path = OUTPUT_DIR / 'training_info_latest.json'
    with open(latest_info_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)

    return model_version


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train UFC prediction model V3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ml.train_model_v3                  # Default (2019+ data)
  python -m ml.train_model_v3 --min-year 2020  # Only 2020+ data
  python -m ml.train_model_v3 --no-filter      # All historical data
        """
    )
    parser.add_argument('--min-year', type=int, default=MIN_YEAR_DEFAULT,
                        help=f'Minimum year for training data (default: {MIN_YEAR_DEFAULT})')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable temporal filtering (not recommended)')
    parser.add_argument('--diagnose-holdout', action='store_true',
                        help='Print extra holdout diagnostics and sample predictions')
    args = parser.parse_args()

    min_year = None if args.no_filter else args.min_year

    print("=" * 60)
    print("UFC PREDICTION MODEL V3")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    if min_year:
        print(f"Temporal filter: {min_year}+")
    else:
        print("Temporal filter: DISABLED")

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    fights_df = load_training_data(min_year=min_year)

    if len(fights_df) < 500:
        print(f"\n[ERROR] Only {len(fights_df)} fights loaded. Need at least 500.")
        sys.exit(1)

    # =========================================================================
    # 2. FEATURE ENGINEERING
    # =========================================================================
    features_df = engineer_features_v3(fights_df, augment_both=True)
    feature_cols = get_feature_columns_v3()

    # =========================================================================
    # 3. CHRONOLOGICAL SPLIT (MUST happen BEFORE winsorization/correlation)
    #    to prevent data leakage from holdout/validation into preprocessing
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHRONOLOGICAL THREE-WAY SPLIT")
    print("=" * 60)

    features_df = features_df.sort_values('fight_date').reset_index(drop=True)
    train_df, val_df, holdout_df = split_by_fight_id(features_df)

    train_ids = set(train_df['fight_id'].unique())
    val_ids = set(val_df['fight_id'].unique())
    holdout_ids = set(holdout_df['fight_id'].unique())
    overlap_train_val = len(train_ids & val_ids)
    overlap_train_holdout = len(train_ids & holdout_ids)
    overlap_val_holdout = len(val_ids & holdout_ids)

    print(f"   Train:    {len(train_df):5d} rows / {len(train_ids):4d} fights "
          f"({train_df['fight_date'].min().date()} to {train_df['fight_date'].max().date()})")
    print(f"   Valid:    {len(val_df):5d} rows / {len(val_ids):4d} fights "
          f"({val_df['fight_date'].min().date()} to {val_df['fight_date'].max().date()})")
    print(f"   Holdout:  {len(holdout_df):5d} rows / {len(holdout_ids):4d} fights "
          f"({holdout_df['fight_date'].min().date()} to {holdout_df['fight_date'].max().date()})")
    print(f"   Overlap check (fight_id): train∩val={overlap_train_val}, "
          f"train∩holdout={overlap_train_holdout}, val∩holdout={overlap_val_holdout}")
    assert overlap_train_val == 0 and overlap_train_holdout == 0 and overlap_val_holdout == 0, \
        "FIGHT_ID LEAK: same fight appears across splits!"

    # Verify temporal ordering (no future fights in training)
    assert train_df['fight_date'].max() <= val_df['fight_date'].min(), \
        "TEMPORAL LEAK: Training data overlaps with validation!"
    assert val_df['fight_date'].max() <= holdout_df['fight_date'].min(), \
        "TEMPORAL LEAK: Validation data overlaps with holdout!"

    # Check minimum split sizes
    if len(val_ids) < 100:
        print(f"   [WARNING] Validation set very small ({len(val_ids)} fights)")
    if len(holdout_ids) < 100:
        print(f"   [WARNING] Holdout set very small ({len(holdout_ids)} fights)")

    holdout_fights_unique = holdout_df[['fight_id', 'fight_date']].drop_duplicates('fight_id')
    holdout_year_counts = holdout_fights_unique['fight_date'].dt.year.value_counts().sort_index().to_dict()
    print(f"   Holdout years: {holdout_year_counts}")

    # =========================================================================
    # 4. MISSING DATA ANALYSIS (on training set only)
    # =========================================================================
    missing_info = analyze_missing_data(train_df, feature_cols)

    # =========================================================================
    # 5. WINSORIZATION (computed from TRAINING set only, applied to all)
    # =========================================================================
    # Compute bounds from training data ONLY to prevent holdout leakage
    train_df, winsorize_bounds = winsorize_features(train_df, feature_cols)

    # Apply TRAINING bounds to validation and holdout
    for col in feature_cols:
        if col in winsorize_bounds:
            bounds = winsorize_bounds[col]
            val_df[col] = val_df[col].clip(lower=bounds['lower'], upper=bounds['upper'])
            holdout_df[col] = holdout_df[col].clip(lower=bounds['lower'], upper=bounds['upper'])

    # =========================================================================
    # 6. CORRELATION ANALYSIS (on TRAINING set only)
    # =========================================================================
    dropped_features = analyze_correlations(train_df, feature_cols, threshold=0.85)
    if dropped_features:
        feature_cols = [c for c in feature_cols if c not in dropped_features]
        print(f"\n   Final feature count: {len(feature_cols)} (dropped {len(dropped_features)})")

    print(f"   Features: {len(feature_cols)}")
    print(f"   Ratio:    {len(train_df) // max(len(feature_cols), 1)} samples per feature")

    # =========================================================================
    # 7. PREPARE FEATURE MATRICES
    # =========================================================================

    # For XGBoost: keep NaN (it handles them natively)
    X_train_raw = train_df[feature_cols].values
    X_val_raw = val_df[feature_cols].values
    X_holdout_raw = holdout_df[feature_cols].values

    y_train = train_df['target'].values
    y_val = val_df['target'].values
    y_holdout = holdout_df['target'].values

    # Compute medians from TRAINING set only (for imputation and prediction time)
    median_values = {}
    for col in feature_cols:
        med = train_df[col].median()
        median_values[col] = float(med) if not np.isnan(med) else 0.0

    # For LR/RF: impute with median from training set
    X_train_imputed = train_df[feature_cols].copy()
    X_val_imputed = val_df[feature_cols].copy()
    X_holdout_imputed = holdout_df[feature_cols].copy()

    for col in feature_cols:
        med = median_values[col]
        X_train_imputed[col] = X_train_imputed[col].fillna(med)
        X_val_imputed[col] = X_val_imputed[col].fillna(med)
        X_holdout_imputed[col] = X_holdout_imputed[col].fillna(med)

    X_train_imputed = X_train_imputed.values
    X_val_imputed = X_val_imputed.values
    X_holdout_imputed = X_holdout_imputed.values

    # Scale for LR (fit on TRAINING only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_holdout_scaled = scaler.transform(X_holdout_imputed)

    # Combine train + val for final models (after hyperparameter selection)
    X_trainval_raw = np.vstack([X_train_raw, X_val_raw])
    X_trainval_imputed = np.vstack([X_train_imputed, X_val_imputed])
    X_trainval_scaled = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])

    # =========================================================================
    # 8. BASELINE
    # =========================================================================
    print("\n" + "=" * 60)
    print("BASELINE (Higher ELO Wins)")
    print("=" * 60)

    elo_col_idx = feature_cols.index('elo_diff')
    elo_test = X_holdout_raw[:, elo_col_idx]
    baseline_preds = (np.nan_to_num(elo_test, 0) > 0).astype(int)
    holdout_baseline = holdout_df[['fight_id', 'swapped', 'target']].copy().reset_index(drop=True)
    holdout_baseline['pred_raw'] = baseline_preds
    holdout_baseline['pred_f1'] = np.where(
        holdout_baseline['swapped'],
        1 - holdout_baseline['pred_raw'],
        holdout_baseline['pred_raw'],
    )
    baseline_fight = holdout_baseline.groupby('fight_id').apply(
        lambda g: pd.Series({
            'target': int(round(float(g[g['swapped'] == False]['target'].mean())))
            if (g['swapped'] == False).any()
            else int(round(float(g['target'].mean()))),
            'pred': int(round(float(g['pred_f1'].mean()))),
        }),
    )
    baseline_acc = accuracy_score(baseline_fight['target'].values, baseline_fight['pred'].values)
    print(f"   Holdout Accuracy (fight-level): {baseline_acc:.3f}")

    # =========================================================================
    # 9. TRAIN MODELS
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    results = []

    # Logistic Regression (uses scaled, imputed data)
    lr_model = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
    lr_result = train_with_calibration(
        X_train_scaled, y_train, X_val_scaled, y_val,
        "Logistic Regression (Calibrated)", lr_model
    )
    results.append(lr_result)

    # Random Forest (uses imputed data, no scaling needed)
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=30,
        min_samples_split=60, max_features='sqrt', random_state=42
    )
    rf_result = train_with_calibration(
        X_train_imputed, y_train, X_val_imputed, y_val,
        "Random Forest (Calibrated)", rf_model
    )
    results.append(rf_result)

    # XGBoost with early stopping (uses raw data with NaN)
    xgb_result = train_xgboost_early_stopping(
        X_train_raw, y_train, X_val_raw, y_val, X_val_raw, y_val
    )
    if xgb_result:
        results.append(xgb_result)

    # Stacking Ensemble (uses imputed data)
    ensemble_result = train_ensemble(
        X_train_imputed, y_train, X_val_imputed, y_val, X_val_imputed, y_val
    )
    results.append(ensemble_result)

    # =========================================================================
    # 10. MODEL COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (on validation set)")
    print("=" * 60)

    print(f"\n   {'Model':<40} {'Val Acc':>8} {'Val AUC':>8} {'Brier':>8} {'Gap':>6}")
    print("   " + "-" * 72)
    for r in results:
        gap = r.get('train_auc', 0) - r.get('test_auc', 0)
        brier = r.get('brier_cal', r.get('test_brier', 0))
        print(f"   {r['name']:<40} {r['test_acc']:>8.3f} {r['test_auc']:>8.3f} "
              f"{brier:>8.4f} {gap:>+6.3f}")

    # Select best model by validation AUC
    best_result = max(results, key=lambda x: x['test_auc'])
    best_name = best_result['name']
    print(f"\n   Best model: {best_name} (AUC={best_result['test_auc']:.3f})")

    # =========================================================================
    # 11. RETRAIN BEST MODEL ON TRAIN+VAL, EVALUATE ON HOLDOUT
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"RETRAINING BEST MODEL ON TRAIN+VAL DATA")
    print("=" * 60)
    print(f"   Model: {best_name}")
    print(f"   Train+Val: {len(y_trainval)} samples")

    # Determine which data format to use based on best model
    is_xgb = 'XGBoost' in best_name
    is_lr = 'Logistic' in best_name

    if is_xgb and HAS_XGBOOST:
        # Retrain XGBoost on train+val with NaN support
        final_model = xgb.XGBClassifier(
            n_estimators=best_result.get('best_iteration', 200),
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=2.0,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
        )
        final_model.fit(X_trainval_raw, y_trainval)

        # Calibrate on the validation portion (manual isotonic, sklearn 1.6+ compatible)
        calibrated_final = IsotonicCalibrator(final_model)
        calibrated_final.fit(X_val_raw, y_val)

        X_holdout_eval = X_holdout_raw
    elif is_lr:
        # Retrain LR on scaled data
        final_base = clone(best_result['base_model'])
        final_base.fit(X_trainval_scaled, y_trainval)
        calibrated_final = IsotonicCalibrator(final_base)
        calibrated_final.fit(X_val_scaled, y_val)
        X_holdout_eval = X_holdout_scaled
    else:
        # RF or ensemble - use imputed data
        if 'Stacking' in best_name:
            # Stacking needs to be retrained
            estimators = [
                ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
                ('rf', RandomForestClassifier(
                    n_estimators=100, max_depth=5, min_samples_leaf=30,
                    min_samples_split=50, random_state=42
                )),
            ]
            if HAS_XGBOOST:
                estimators.append(
                    ('xgb', xgb.XGBClassifier(
                        n_estimators=100, max_depth=3, learning_rate=0.03,
                        min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0,
                        subsample=0.6, colsample_bytree=0.6, random_state=42
                    ))
                )
            calibrated_final = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(C=0.5, max_iter=1000),
                cv=5, passthrough=False
            )
            calibrated_final.fit(X_trainval_imputed, y_trainval)
        else:
            final_base = clone(best_result['base_model'])
            final_base.fit(X_trainval_imputed, y_trainval)
            calibrated_final = IsotonicCalibrator(final_base)
            calibrated_final.fit(X_val_imputed, y_val)
        X_holdout_eval = X_holdout_imputed

    # =========================================================================
    # 12. HOLDOUT EVALUATION
    # =========================================================================
    holdout_metrics = evaluate_on_holdout(
        calibrated_final, X_holdout_eval, y_holdout, feature_cols, holdout_df
    )

    holdout_prod_metrics = {
        'accuracy': None,
        'fight_count': 0,
        'skipped': 0,
        'samples': [],
    }
    try:
        conn = connect_db()
        holdout_fights_for_prod = (
            fights_df[fights_df['fightid'].isin(holdout_ids)]
            .sort_values('fight_date')
            .drop_duplicates('fightid')
            .copy()
        )
        holdout_prod_metrics = evaluate_holdout_production_pipeline(conn, holdout_fights_for_prod)
        conn.close()
        if holdout_prod_metrics['accuracy'] is not None:
            delta = holdout_metrics['accuracy'] - holdout_prod_metrics['accuracy']
            print(f"   Direct-vs-production delta: {delta:+.3f}")
    except Exception as e:
        print(f"   [WARNING] Production-pipeline holdout eval failed: {e}")

    # =========================================================================
    # 13. FEATURE IMPORTANCE
    # =========================================================================
    importance = analyze_feature_importance(
        calibrated_final, X_holdout_eval, y_holdout, feature_cols
    )

    # =========================================================================
    # 14. TIME SERIES CROSS-VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("TIME SERIES CROSS-VALIDATION (5-fold)")
    print("=" * 60)

    if is_xgb and HAS_XGBOOST:
        cv_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0,
            subsample=0.7, colsample_bytree=0.7, random_state=42
        )
        cv_X = X_trainval_raw
    elif is_lr:
        cv_model = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
        cv_X = X_trainval_scaled
    else:
        cv_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=30,
            min_samples_split=60, max_features='sqrt', random_state=42
        )
        cv_X = X_trainval_imputed

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(cv_model, cv_X, y_trainval, cv=tscv, scoring='roc_auc')
    print(f"   AUC scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"   Mean AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # =========================================================================
    # 15. SAVE MODEL
    # =========================================================================
    all_metrics = {
        'best_model': best_name,
        'baseline_accuracy': float(baseline_acc),
        'validation': {
            'accuracy': best_result['test_acc'],
            'auc': best_result['test_auc'],
            'brier': best_result.get('brier_cal', 0),
        },
        'holdout': holdout_metrics,
        'holdout_production': holdout_prod_metrics,
        'split_diagnostics': {
            'train_fights': int(len(train_ids)),
            'val_fights': int(len(val_ids)),
            'holdout_fights': int(len(holdout_ids)),
            'overlap_train_val': int(overlap_train_val),
            'overlap_train_holdout': int(overlap_train_holdout),
            'overlap_val_holdout': int(overlap_val_holdout),
            'holdout_year_counts': {str(k): int(v) for k, v in holdout_year_counts.items()},
        },
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'feature_importance': importance,
        'all_models': [
            {
                'name': r['name'],
                'val_acc': r['test_acc'],
                'val_auc': r['test_auc'],
                'gap': r.get('train_auc', 0) - r.get('test_auc', 0),
            }
            for r in results
        ],
    }

    # Validate feature list matches what model_loader knows about
    try:
        from ml.model_loader import ALL_V3_FEATURES
    except ModuleNotFoundError:
        from model_loader import ALL_V3_FEATURES
    unknown = set(feature_cols) - set(ALL_V3_FEATURES)
    if unknown:
        print(f"\n   [CRITICAL WARNING] Features not in model_loader.ALL_V3_FEATURES: {unknown}")
        print("   The model_loader.build_feature_dict() MUST be updated to produce these features!")
    missing_from_training = set(ALL_V3_FEATURES) - set(feature_cols) - set(dropped_features)
    if missing_from_training:
        # This is expected if correlation analysis dropped some
        print(f"   Features available but not used: {missing_from_training}")

    model_version = save_model(
        model=calibrated_final,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics=all_metrics,
        winsorize_bounds=winsorize_bounds,
        dropped_features=dropped_features,
        min_year=min_year,
        median_values=median_values,
    )

    # =========================================================================
    # 16. SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"   Model version:     {model_version}")
    print(f"   Best model:        {best_name}")
    print(f"   Features:          {len(feature_cols)}")
    print(f"   Training samples:  {len(y_trainval)}")
    print(f"   Holdout rows:      {len(y_holdout)}")
    print(f"   Holdout fights:    {holdout_metrics['holdout_size']}")
    print(f"   Baseline accuracy: {baseline_acc:.1%}")
    print(f"   Holdout accuracy:  {holdout_metrics['accuracy']:.1%}")
    print(f"   Holdout row acc:   {holdout_metrics['row_accuracy']:.1%} (diagnostic)")
    print(f"   Holdout AUC:       {holdout_metrics['auc']:.3f}")
    ci = holdout_metrics['confidence_intervals']
    print(f"   95% CI (accuracy): [{ci['accuracy']['lower']:.1%}, {ci['accuracy']['upper']:.1%}]")
    print(f"   95% CI (AUC):      [{ci['auc']['lower']:.3f}, {ci['auc']['upper']:.3f}]")
    if holdout_prod_metrics.get('accuracy') is not None:
        print(f"   Production holdout:{holdout_prod_metrics['accuracy']:.1%} "
              f"({holdout_prod_metrics['fight_count']} fights)")
    if holdout_metrics.get('roi') is not None:
        print(f"   Holdout ROI:       {holdout_metrics['roi'] * 100:+.1f}%")

    if args.diagnose_holdout:
        print("\n" + "=" * 60)
        print("HOLDOUT DIAGNOSTICS")
        print("=" * 60)
        print(f"   Date range: {holdout_df['fight_date'].min().date()} to {holdout_df['fight_date'].max().date()}")
        print(f"   Year counts: {holdout_year_counts}")
        print(f"   Overlap check (fight_id): train/val={overlap_train_val}, "
              f"train/holdout={overlap_train_holdout}, val/holdout={overlap_val_holdout}")
        print("\n   Sample direct holdout predictions:")
        for row in holdout_metrics.get('sample_predictions', []):
            print(f"   fight_id={row['fight_id']} date={pd.to_datetime(row['fight_date']).date()} "
                  f"p_f1={row['p_fighter1']:.3f} pred={row['pred']} actual={row['y_true']}")
        if holdout_prod_metrics.get('samples'):
            print("\n   Sample production holdout predictions:")
            for row in holdout_prod_metrics['samples']:
                print(f"   fight_id={row['fight_id']} {row['fighter1_name']} vs {row['fighter2_name']} "
                      f"p_f1={row['p_fighter1']:.3f} pred={row['pred']} actual={row['y_true']} "
                      f"src={row['prob_source']}")
    print(f"\n   Completed: {datetime.now()}")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
UFC Fight Prediction Model Training Pipeline

This script follows ML best practices for sports betting prediction:
1. No data leakage - uses point-in-time ELO ratings
2. Chronological train/test split
3. Feature engineering with differentials
4. Baseline comparison
5. Multiple model evaluation
6. Probability calibration
7. Backtest simulation
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, 
    classification_report, confusion_matrix,
    brier_score_loss, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

warnings.filterwarnings('ignore')

# Try importing xgboost (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed. Install with: pip install xgboost")

# Load environment
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Database config
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'dbname': os.getenv('DB_NAME', 'delphi_db'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
}

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'artifacts'
OUTPUT_DIR.mkdir(exist_ok=True)


def connect_db():
    """Connect to database."""
    return psycopg2.connect(**DB_CONFIG)


def load_training_data():
    """
    Load and prepare training data with NO DATA LEAKAGE.
    
    Key: We use point-in-time stats - what each fighter's stats were
    BEFORE each fight, not their current career stats.
    """
    print("\n" + "="*60)
    print("LOADING TRAINING DATA (POINT-IN-TIME)")
    print("="*60)
    
    conn = connect_db()
    
    # Load fights with fighter info, ELO at fight time, AND point-in-time stats
    # All features are calculated using ONLY data available BEFORE the fight
    fights_query = """
    SELECT 
        f.FightID,
        f.Date as fight_date,
        f.FighterID as fighter1_id,
        f.FighterURL as fighter1_url,
        f.FighterName as fighter1_name,
        f.OpponentID as fighter2_id,
        f.OpponentURL as fighter2_url,
        f.OpponentName as fighter2_name,
        f.WinnerID,
        f.Result,
        f.Method,
        f.Round,
        f.IsTitleFight,
        f.EventName,
        -- Fighter 1 physical stats (these are OK - physical stats don't change)
        fs1.Height as f1_height,
        fs1.Reach as f1_reach,
        fs1.Age as f1_age,
        fs1.Stance as f1_stance,
        fs1.WeightClass,
        -- Fighter 2 physical stats
        fs2.Height as f2_height,
        fs2.Reach as f2_reach,
        fs2.Age as f2_age,
        fs2.Stance as f2_stance,
        -- ELO at fight time (point-in-time)
        e1.EloBeforeFight as f1_elo_at_fight,
        e2.EloBeforeFight as f2_elo_at_fight,
        -- POINT-IN-TIME career stats for Fighter 1 (calculated from prior fights only!)
        pit1.FightsBefore as f1_fights_before,
        pit1.WinsBefore as f1_wins_before,
        pit1.WinRateBefore as f1_win_rate_before,
        pit1.PIT_SLpM as f1_pit_slpm,
        pit1.PIT_TDAvg as f1_pit_tdavg,
        pit1.PIT_SubAvg as f1_pit_subavg,
        pit1.PIT_KDRate as f1_pit_kd_rate,
        pit1.RecentWinRate as f1_recent_form,
        pit1.FinishRate as f1_finish_rate,
        pit1.HasPriorData as f1_has_prior,
        -- POINT-IN-TIME career stats for Fighter 2
        pit2.FightsBefore as f2_fights_before,
        pit2.WinsBefore as f2_wins_before,
        pit2.WinRateBefore as f2_win_rate_before,
        pit2.PIT_SLpM as f2_pit_slpm,
        pit2.PIT_TDAvg as f2_pit_tdavg,
        pit2.PIT_SubAvg as f2_pit_subavg,
        pit2.PIT_KDRate as f2_pit_kd_rate,
        pit2.RecentWinRate as f2_recent_form,
        pit2.FinishRate as f2_finish_rate,
        pit2.HasPriorData as f2_has_prior
    FROM Fights f
    LEFT JOIN FighterStats fs1 ON f.FighterID = fs1.FighterID
    LEFT JOIN FighterStats fs2 ON f.OpponentID = fs2.FighterID
    LEFT JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    LEFT JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    LEFT JOIN PointInTimeStats pit1 ON f.FighterURL = pit1.FighterURL AND f.Date = pit1.FightDate
    LEFT JOIN PointInTimeStats pit2 ON f.OpponentURL = pit2.FighterURL AND f.Date = pit2.FightDate
    WHERE f.Date IS NOT NULL
    ORDER BY f.Date
    """
    
    fights_df = pd.read_sql(fights_query, conn)
    print(f"[OK] Loaded {len(fights_df)} fights")
    
    # Check ELO coverage
    has_f1_elo = fights_df['f1_elo_at_fight'].notna().sum()
    has_f2_elo = fights_df['f2_elo_at_fight'].notna().sum()
    has_both_elo = (fights_df['f1_elo_at_fight'].notna() & fights_df['f2_elo_at_fight'].notna()).sum()
    print(f"   ELO coverage: F1={has_f1_elo} ({has_f1_elo/len(fights_df)*100:.1f}%), "
          f"F2={has_f2_elo} ({has_f2_elo/len(fights_df)*100:.1f}%), "
          f"Both={has_both_elo} ({has_both_elo/len(fights_df)*100:.1f}%)")
    
    # Check point-in-time stats coverage
    has_f1_pit = fights_df['f1_has_prior'].notna().sum()
    has_f2_pit = fights_df['f2_has_prior'].notna().sum()
    has_both_pit = (fights_df['f1_has_prior'].notna() & fights_df['f2_has_prior'].notna()).sum()
    print(f"   PIT stats coverage: F1={has_f1_pit} ({has_f1_pit/len(fights_df)*100:.1f}%), "
          f"F2={has_f2_pit} ({has_f2_pit/len(fights_df)*100:.1f}%), "
          f"Both={has_both_pit} ({has_both_pit/len(fights_df)*100:.1f}%)")
    
    # No separate career stats needed - all stats are in the main query now
    career_df = pd.DataFrame()
    elo_df = pd.DataFrame()
    
    print(f"[OK] All features are point-in-time (no data leakage)")
    
    conn.close()
    
    return fights_df, career_df, elo_df


def parse_height_inches(height_str):
    """Convert height string to inches."""
    if pd.isna(height_str) or height_str == '--':
        return None
    try:
        # Format: "5' 10\"" or "5'10\""
        height_str = str(height_str).replace('"', '').replace("'", ' ').strip()
        parts = height_str.split()
        if len(parts) >= 2:
            feet = int(parts[0])
            inches = int(parts[1]) if parts[1] else 0
            return feet * 12 + inches
        elif len(parts) == 1:
            return int(parts[0])
    except:
        pass
    return None


def parse_reach_inches(reach_str):
    """Convert reach string to inches."""
    if pd.isna(reach_str) or reach_str == '--':
        return None
    try:
        # Format: "72\"" or "72"
        reach_str = str(reach_str).replace('"', '').replace("'", '').strip()
        return float(reach_str)
    except:
        pass
    return None


def get_elo_at_fight_time(fighter_id, fight_date, elo_df):
    """
    Get fighter's ELO rating at the time of a specific fight.
    This prevents data leakage by using historical ELO, not current.
    """
    if pd.isna(fighter_id) or pd.isna(fight_date):
        return 1500.0  # Default starting ELO
    
    # Find the ELO record for this fighter on or before this fight date
    fighter_elo = elo_df[
        (elo_df['fighterid'] == fighter_id) & 
        (elo_df['fightdate'] <= fight_date)
    ].sort_values('fightdate', ascending=False)
    
    if len(fighter_elo) > 0:
        return fighter_elo.iloc[0]['elobeforefight']
    
    return 1500.0  # Default for fighters with no prior history


def engineer_features(fights_df, career_df, elo_df, random_swap=True, seed=42):
    """
    Create features for ML model using POINT-IN-TIME stats.
    
    KEY PRINCIPLE: All features use ONLY information available
    BEFORE the fight occurred. No data leakage!
    
    IMPORTANT: We randomly swap fighter order to eliminate positional bias.
    Without this, Fighter1 wins 60.8% due to data artifact, not skill.
    
    Point-in-time stats (PIT) are calculated from a fighter's
    prior fights only - not their current career totals.
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING (POINT-IN-TIME)")
    print("="*60)
    
    if random_swap:
        print("   [!] Random fighter swap ENABLED (eliminates positional bias)")
        np.random.seed(seed)
    
    # Convert dates
    fights_df['fight_date'] = pd.to_datetime(fights_df['fight_date'])
    
    features = []
    swapped_count = 0
    
    for idx, fight in fights_df.iterrows():
        if idx % 1000 == 0:
            print(f"   Processing fight {idx}/{len(fights_df)}...")
        
        # Skip if no valid winner
        if pd.isna(fight['winnerid']):
            continue
        
        # === RANDOM SWAP to eliminate positional bias ===
        # Without this, Fighter1 wins 60.8% (data artifact)
        # With swap, target becomes ~50/50 (fair evaluation)
        swap = random_swap and (np.random.random() < 0.5)
        
        if swap:
            swapped_count += 1
            # Swap fighter 1 and fighter 2
            f1_id = fight['fighter2_id']
            f2_id = fight['fighter1_id']
            f1_name = fight['fighter2_name']
            f2_name = fight['fighter1_name']
            # Swap all the stats (we'll handle this in feature extraction)
            swap_prefix = True
        else:
            f1_id = fight['fighter1_id']
            f2_id = fight['fighter2_id']
            f1_name = fight['fighter1_name']
            f2_name = fight['fighter2_name']
            swap_prefix = False
        
        fight_date = fight['fight_date']
        
        # Target: Did our "fighter A" (potentially swapped) win?
        target = 1 if fight['winnerid'] == f1_id else 0
        
        # Helper to safely get float values
        def safe_float(val, default=0.0):
            if pd.isna(val):
                return default
            return float(val)
        
        # === EXTRACT FEATURES (respecting swap) ===
        # When swap=True, we read f2 columns as "our fighter A" and f1 as "our fighter B"
        if swap:
            # Swapped: original F2 becomes our A, original F1 becomes our B
            a_elo = safe_float(fight.get('f2_elo_at_fight'), 1500.0)
            b_elo = safe_float(fight.get('f1_elo_at_fight'), 1500.0)
            a_height = parse_height_inches(fight['f2_height'])
            b_height = parse_height_inches(fight['f1_height'])
            a_reach = parse_reach_inches(fight['f2_reach'])
            b_reach = parse_reach_inches(fight['f1_reach'])
            a_age = fight['f2_age'] if pd.notna(fight['f2_age']) else 30
            b_age = fight['f1_age'] if pd.notna(fight['f1_age']) else 30
            a_stance = fight['f2_stance']
            b_stance = fight['f1_stance']
            # PIT stats
            a_slpm = safe_float(fight.get('f2_pit_slpm'))
            a_tdavg = safe_float(fight.get('f2_pit_tdavg'))
            a_subavg = safe_float(fight.get('f2_pit_subavg'))
            a_kd_rate = safe_float(fight.get('f2_pit_kd_rate'))
            a_win_rate = safe_float(fight.get('f2_win_rate_before'), 0.5)
            a_recent_form = safe_float(fight.get('f2_recent_form'), 0.5)
            a_finish_rate = safe_float(fight.get('f2_finish_rate'))
            a_fights = safe_float(fight.get('f2_fights_before'))
            b_slpm = safe_float(fight.get('f1_pit_slpm'))
            b_tdavg = safe_float(fight.get('f1_pit_tdavg'))
            b_subavg = safe_float(fight.get('f1_pit_subavg'))
            b_kd_rate = safe_float(fight.get('f1_pit_kd_rate'))
            b_win_rate = safe_float(fight.get('f1_win_rate_before'), 0.5)
            b_recent_form = safe_float(fight.get('f1_recent_form'), 0.5)
            b_finish_rate = safe_float(fight.get('f1_finish_rate'))
            b_fights = safe_float(fight.get('f1_fights_before'))
        else:
            # Not swapped: original F1 is our A, original F2 is our B
            a_elo = safe_float(fight.get('f1_elo_at_fight'), 1500.0)
            b_elo = safe_float(fight.get('f2_elo_at_fight'), 1500.0)
            a_height = parse_height_inches(fight['f1_height'])
            b_height = parse_height_inches(fight['f2_height'])
            a_reach = parse_reach_inches(fight['f1_reach'])
            b_reach = parse_reach_inches(fight['f2_reach'])
            a_age = fight['f1_age'] if pd.notna(fight['f1_age']) else 30
            b_age = fight['f2_age'] if pd.notna(fight['f2_age']) else 30
            a_stance = fight['f1_stance']
            b_stance = fight['f2_stance']
            # PIT stats
            a_slpm = safe_float(fight.get('f1_pit_slpm'))
            a_tdavg = safe_float(fight.get('f1_pit_tdavg'))
            a_subavg = safe_float(fight.get('f1_pit_subavg'))
            a_kd_rate = safe_float(fight.get('f1_pit_kd_rate'))
            a_win_rate = safe_float(fight.get('f1_win_rate_before'), 0.5)
            a_recent_form = safe_float(fight.get('f1_recent_form'), 0.5)
            a_finish_rate = safe_float(fight.get('f1_finish_rate'))
            a_fights = safe_float(fight.get('f1_fights_before'))
            b_slpm = safe_float(fight.get('f2_pit_slpm'))
            b_tdavg = safe_float(fight.get('f2_pit_tdavg'))
            b_subavg = safe_float(fight.get('f2_pit_subavg'))
            b_kd_rate = safe_float(fight.get('f2_pit_kd_rate'))
            b_win_rate = safe_float(fight.get('f2_win_rate_before'), 0.5)
            b_recent_form = safe_float(fight.get('f2_recent_form'), 0.5)
            b_finish_rate = safe_float(fight.get('f2_finish_rate'))
            b_fights = safe_float(fight.get('f2_fights_before'))
        
        # === CALCULATE DIFFERENTIALS (A - B) ===
        elo_diff = a_elo - b_elo
        height_diff = (a_height - b_height) if (a_height and b_height) else 0
        reach_diff = (a_reach - b_reach) if (a_reach and b_reach) else 0
        age_diff = a_age - b_age
        slpm_diff = a_slpm - b_slpm
        tdavg_diff = a_tdavg - b_tdavg
        subavg_diff = a_subavg - b_subavg
        kd_rate_diff = a_kd_rate - b_kd_rate
        win_rate_diff = a_win_rate - b_win_rate
        recent_form_diff = a_recent_form - b_recent_form
        finish_rate_diff = a_finish_rate - b_finish_rate
        experience_diff = a_fights - b_fights
        
        # === STANCE MATCHUP ===
        stance_map = {'Orthodox': 0, 'Southpaw': 1, 'Switch': 2}
        a_stance_num = stance_map.get(a_stance, 0)
        b_stance_num = stance_map.get(b_stance, 0)
        
        # Southpaw vs Orthodox matchup indicator
        southpaw_vs_orthodox = 1 if (a_stance_num == 1 and b_stance_num == 0) else (
            -1 if (a_stance_num == 0 and b_stance_num == 1) else 0
        )
        
        # === FIGHT CONTEXT ===
        is_title_fight = 1 if fight['istitlefight'] else 0
        
        features.append({
            'fight_id': fight['fightid'],
            'fight_date': fight_date,
            'fighter1_id': f1_id,
            'fighter2_id': f2_id,
            'fighter1_name': f1_name,
            'fighter2_name': f2_name,
            'weight_class': fight['weightclass'],
            'is_title_fight': is_title_fight,
            'target': target,
            'swapped': swap,
            
            # ELO features (point-in-time)
            'f1_elo': a_elo,
            'f2_elo': b_elo,
            'elo_diff': elo_diff,
            
            # Physical features (static)
            'height_diff': height_diff,
            'reach_diff': reach_diff,
            'age_diff': age_diff,
            
            # Point-in-time performance features
            'slpm_diff': slpm_diff,
            'tdavg_diff': tdavg_diff,
            'subavg_diff': subavg_diff,
            'kd_rate_diff': kd_rate_diff,
            'win_rate_diff': win_rate_diff,
            'recent_form_diff': recent_form_diff,
            'finish_rate_diff': finish_rate_diff,
            'experience_diff': experience_diff,
            
            # Matchup features
            'southpaw_advantage': southpaw_vs_orthodox,
        })
    
    features_df = pd.DataFrame(features)
    print(f"\n[OK] Created {len(features_df)} training samples")
    
    if random_swap:
        print(f"   Swapped: {swapped_count} ({swapped_count/len(features_df)*100:.1f}%)")
        print(f"   Class balance after swap: {features_df['target'].mean()*100:.1f}% wins for Fighter A")
    
    # Show feature distributions to verify no leakage
    print("\nPoint-in-time feature distributions:")
    for col in ['slpm_diff', 'win_rate_diff', 'recent_form_diff']:
        print(f"   {col}: mean={features_df[col].mean():.3f}, std={features_df[col].std():.3f}")
    
    return features_df


def check_data_quality(df):
    """Check for data quality issues."""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Date range: {df['fight_date'].min()} to {df['fight_date'].max()}")
    
    # Class balance
    class_counts = df['target'].value_counts()
    print(f"\nClass balance:")
    print(f"   Fighter 1 wins: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"   Fighter 2 wins: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    # Missing values
    print("\nMissing values per feature:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"   {col}: {null_count} ({null_count/len(df)*100:.1f}%)")
    
    # ELO distribution
    print(f"\nELO distribution:")
    print(f"   Fighter 1 ELO: mean={df['f1_elo'].mean():.0f}, std={df['f1_elo'].std():.0f}")
    print(f"   Fighter 2 ELO: mean={df['f2_elo'].mean():.0f}, std={df['f2_elo'].std():.0f}")
    print(f"   ELO diff range: {df['elo_diff'].min():.0f} to {df['elo_diff'].max():.0f}")
    
    return df


def chronological_split(df, test_ratio=0.2):
    """
    Split data chronologically - CRITICAL for sports prediction.
    
    We train on past fights and test on future fights.
    Random splitting would cause data leakage!
    """
    print("\n" + "="*60)
    print("CHRONOLOGICAL TRAIN/TEST SPLIT")
    print("="*60)
    
    # Sort by date
    df = df.sort_values('fight_date').reset_index(drop=True)
    
    # Find split point
    split_idx = int(len(df) * (1 - test_ratio))
    split_date = df.iloc[split_idx]['fight_date']
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\nTrain set: {len(train_df)} fights")
    print(f"   Date range: {train_df['fight_date'].min().date()} to {train_df['fight_date'].max().date()}")
    
    print(f"\nTest set: {len(test_df)} fights")
    print(f"   Date range: {test_df['fight_date'].min().date()} to {test_df['fight_date'].max().date()}")
    
    return train_df, test_df, split_date


def get_feature_columns():
    """
    Define which columns are features for training.
    
    All features are point-in-time (no data leakage):
    - ELO at fight time
    - Physical attributes (static)
    - Performance stats calculated from prior fights only
    """
    return [
        # ELO (point-in-time)
        'elo_diff',
        # Physical (static)
        'height_diff',
        'reach_diff',
        'age_diff',
        # Point-in-time performance
        'slpm_diff',          # Strikes landed per minute (from prior fights)
        'tdavg_diff',         # Takedown average (from prior fights)
        'subavg_diff',        # Submission average (from prior fights)
        'kd_rate_diff',       # Knockdown rate (from prior fights)
        'win_rate_diff',      # Historical win rate
        'recent_form_diff',   # Last 3 fights win rate
        'finish_rate_diff',   # % of wins by finish
        'experience_diff',    # Number of fights
        # Matchup context
        'southpaw_advantage',
        'is_title_fight',
    ]


def train_baseline(train_df, test_df):
    """
    Baseline model: Always predict higher ELO fighter wins.
    
    This is the simplest possible model and sets the floor
    for what our ML model needs to beat.
    
    Expected accuracy: ~61% (higher ELO wins 61% when there's a difference)
    """
    print("\n" + "="*60)
    print("BASELINE MODEL: Higher ELO Wins")
    print("="*60)
    
    # Predict: fighter A wins if elo_diff > 0 (A has higher ELO)
    # For ties (elo_diff = 0), predict 0.5 probability
    train_preds = (train_df['elo_diff'] > 0).astype(int)
    test_preds = (test_df['elo_diff'] > 0).astype(int)
    
    # Probabilities based on ELO difference (standard chess formula)
    def elo_to_prob(elo_diff):
        return 1 / (1 + 10 ** (-elo_diff / 400))
    
    train_probs = train_df['elo_diff'].apply(elo_to_prob)
    test_probs = test_df['elo_diff'].apply(elo_to_prob)
    
    train_acc = accuracy_score(train_df['target'], train_preds)
    test_acc = accuracy_score(test_df['target'], test_preds)
    
    train_auc = roc_auc_score(train_df['target'], train_probs)
    test_auc = roc_auc_score(test_df['target'], test_probs)
    
    train_logloss = log_loss(train_df['target'], train_probs)
    test_logloss = log_loss(test_df['target'], test_probs)
    
    print(f"\nTrain: Accuracy={train_acc:.3f}, AUC={train_auc:.3f}, LogLoss={train_logloss:.3f}")
    print(f"Test:  Accuracy={test_acc:.3f}, AUC={test_auc:.3f}, LogLoss={test_logloss:.3f}")
    
    # Diagnostic info
    train_non_tied = train_df[train_df['elo_diff'] != 0]
    test_non_tied = test_df[test_df['elo_diff'] != 0]
    
    if len(train_non_tied) > 0:
        train_higher_elo_wins = (
            ((train_non_tied['elo_diff'] > 0) & (train_non_tied['target'] == 1)) |
            ((train_non_tied['elo_diff'] < 0) & (train_non_tied['target'] == 0))
        ).mean()
        print(f"\n   Train (non-tied): Higher ELO wins {train_higher_elo_wins*100:.1f}% of time")
    
    if len(test_non_tied) > 0:
        test_higher_elo_wins = (
            ((test_non_tied['elo_diff'] > 0) & (test_non_tied['target'] == 1)) |
            ((test_non_tied['elo_diff'] < 0) & (test_non_tied['target'] == 0))
        ).mean()
        print(f"   Test (non-tied): Higher ELO wins {test_higher_elo_wins*100:.1f}% of time")
    
    print(f"   Tied ELO fights: Train={len(train_df) - len(train_non_tied)}, Test={len(test_df) - len(test_non_tied)}")
    
    return {
        'name': 'Baseline (ELO)',
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_logloss': train_logloss,
        'test_logloss': test_logloss,
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train logistic regression model."""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION")
    print("="*60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    train_logloss = log_loss(y_train, train_probs)
    test_logloss = log_loss(y_test, test_probs)
    
    print(f"\nTrain: Accuracy={train_acc:.3f}, AUC={train_auc:.3f}, LogLoss={train_logloss:.3f}")
    print(f"Test:  Accuracy={test_acc:.3f}, AUC={test_auc:.3f}, LogLoss={test_logloss:.3f}")
    
    # Feature importance
    print("\nFeature Importance (coefficients):")
    feature_cols = get_feature_columns()
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    for _, row in coef_df.iterrows():
        print(f"   {row['feature']}: {row['coefficient']:.4f}")
    
    return {
        'name': 'Logistic Regression',
        'model': model,
        'scaler': scaler,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_logloss': train_logloss,
        'test_logloss': test_logloss,
        'feature_importance': coef_df,
    }


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model with regularization to prevent overfitting."""
    print("\n" + "="*60)
    print("RANDOM FOREST (Regularized)")
    print("="*60)
    
    # Increased regularization to reduce overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,              # Reduced from 10 (less overfitting)
        min_samples_leaf=20,      # Increased from 5 (more regularization)
        min_samples_split=40,     # Added (more regularization)
        max_features='sqrt',      # Limit features per tree
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    train_logloss = log_loss(y_train, train_probs)
    test_logloss = log_loss(y_test, test_probs)
    
    print(f"\nTrain: Accuracy={train_acc:.3f}, AUC={train_auc:.3f}, LogLoss={train_logloss:.3f}")
    print(f"Test:  Accuracy={test_acc:.3f}, AUC={test_auc:.3f}, LogLoss={test_logloss:.3f}")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_cols = get_feature_columns()
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return {
        'name': 'Random Forest',
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_logloss': train_logloss,
        'test_logloss': test_logloss,
        'feature_importance': importance_df,
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model with regularization."""
    if not HAS_XGBOOST:
        print("\n[SKIP] XGBoost not available")
        return None
    
    print("\n" + "="*60)
    print("XGBOOST (Regularized)")
    print("="*60)
    
    # Added regularization to reduce overfitting
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,              # Reduced from 5
        learning_rate=0.05,       # Reduced from 0.1 (slower learning)
        subsample=0.7,            # Reduced from 0.8
        colsample_bytree=0.7,     # Reduced from 0.8
        min_child_weight=5,       # Added (regularization)
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    train_logloss = log_loss(y_train, train_probs)
    test_logloss = log_loss(y_test, test_probs)
    
    print(f"\nTrain: Accuracy={train_acc:.3f}, AUC={train_auc:.3f}, LogLoss={train_logloss:.3f}")
    print(f"Test:  Accuracy={test_acc:.3f}, AUC={test_auc:.3f}, LogLoss={test_logloss:.3f}")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_cols = get_feature_columns()
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return {
        'name': 'XGBoost',
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_logloss': train_logloss,
        'test_logloss': test_logloss,
        'feature_importance': importance_df,
    }


def check_probability_calibration(model, X_test, y_test, model_name):
    """Check if predicted probabilities are well-calibrated."""
    print(f"\n--- Probability Calibration: {model_name} ---")
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        return
    
    # Brier score (lower is better, 0 is perfect)
    brier = brier_score_loss(y_test, probs)
    print(f"Brier Score: {brier:.4f}")
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    
    print("\nCalibration (predicted vs actual):")
    for i in range(len(prob_true)):
        diff = prob_true[i] - prob_pred[i]
        indicator = "OK" if abs(diff) < 0.05 else ("over" if diff < 0 else "under")
        print(f"   Predicted ~{prob_pred[i]:.2f}: Actual {prob_true[i]:.2f} [{indicator}]")


def backtest_simulation(test_df, model, scaler, feature_cols):
    """
    Simulate betting on test set with REALISTIC assumptions.
    
    Key assumptions for realistic backtesting:
    1. Use -110 odds (standard vig) - break-even at 52.4%
    2. Only bet when model edge > vig (prob > 0.524)
    3. Require minimum edge of 5% over implied odds (prob > 0.574)
    4. Track unit profit/loss, not just win rate
    """
    print("\n" + "="*60)
    print("BACKTEST SIMULATION (Realistic)")
    print("="*60)
    
    X_test = test_df[feature_cols].values
    
    if scaler:
        X_test = scaler.transform(X_test)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    # Betting parameters
    MIN_EDGE = 0.574      # Minimum probability to bet (includes ~5% edge over break-even)
    DECIMAL_ODDS = 1.91   # -110 American odds = 1.91 decimal
    
    # Track bets
    bets = []
    total_units_wagered = 0
    total_units_profit = 0
    wins = 0
    
    for i, (_, fight) in enumerate(test_df.iterrows()):
        prob_a = probs[i]  # Prob fighter A wins
        prob_b = 1 - prob_a
        
        # Bet 1 unit on fighter with edge
        if prob_a >= MIN_EDGE:
            total_units_wagered += 1
            if fight['target'] == 1:  # Fighter A won
                wins += 1
                profit = DECIMAL_ODDS - 1  # Win pays 0.91 units
                total_units_profit += profit
                bets.append({'result': 'win', 'prob': prob_a, 'profit': profit})
            else:
                total_units_profit -= 1  # Lose 1 unit
                bets.append({'result': 'loss', 'prob': prob_a, 'profit': -1})
        elif prob_b >= MIN_EDGE:
            total_units_wagered += 1
            if fight['target'] == 0:  # Fighter B won
                wins += 1
                profit = DECIMAL_ODDS - 1
                total_units_profit += profit
                bets.append({'result': 'win', 'prob': prob_b, 'profit': profit})
            else:
                total_units_profit -= 1
                bets.append({'result': 'loss', 'prob': prob_b, 'profit': -1})
    
    if total_units_wagered > 0:
        win_rate = wins / total_units_wagered
        roi = total_units_profit / total_units_wagered
        
        print(f"\nBetting threshold: {MIN_EDGE:.1%} probability")
        print(f"Bets placed: {total_units_wagered} ({total_units_wagered/len(test_df)*100:.0f}% of fights)")
        print(f"Wins: {wins}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Break-even: 52.4%")
        print(f"\nTotal wagered: {total_units_wagered:.1f} units")
        print(f"Total profit: {total_units_profit:.2f} units")
        print(f"ROI: {roi*100:.1f}%")
        
        # Reality check
        if roi > 0.15:
            print("\n[!] WARNING: ROI > 15% is likely unrealistic.")
            print("    Industry-leading models achieve 5-15% ROI.")
            print("    Verify model isn't overfitting or has data leakage.")
        elif roi > 0.05:
            print(f"\n[OK] ROI of {roi*100:.1f}% is competitive (good model!)")
        elif roi > 0:
            print(f"\n[OK] ROI of {roi*100:.1f}% shows some edge (marginal)")
        else:
            print(f"\n[!] Negative ROI - model underperforms vig")
    else:
        print(f"\nNo bets placed (no predictions with >{MIN_EDGE:.0%} confidence)")
    
    return bets


def compare_models(results):
    """Compare all model results."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison = []
    for r in results:
        if r:
            comparison.append({
                'Model': r['name'],
                'Train Acc': f"{r['train_acc']:.3f}",
                'Test Acc': f"{r['test_acc']:.3f}",
                'Train AUC': f"{r['train_auc']:.3f}",
                'Test AUC': f"{r['test_auc']:.3f}",
                'Test LogLoss': f"{r['test_logloss']:.3f}",
            })
    
    comparison_df = pd.DataFrame(comparison)
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # Find best model
    best_model = max([r for r in results if r], key=lambda x: x['test_auc'])
    print(f"\nBest model by Test AUC: {best_model['name']}")
    
    # Check for overfitting
    print("\nOverfitting check (Train AUC - Test AUC):")
    for r in results:
        if r:
            gap = r['train_auc'] - r['test_auc']
            status = "OK" if gap < 0.05 else "OVERFITTING" if gap > 0.1 else "WARNING"
            print(f"   {r['name']}: {gap:.3f} [{status}]")
    
    return comparison_df


def save_model(model_result, features_df):
    """Save the best model and training info."""
    import pickle
    
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Save model
    model_path = OUTPUT_DIR / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model_result.get('model'),
            'scaler': model_result.get('scaler'),
            'feature_columns': get_feature_columns(),
            'model_name': model_result['name'],
            'metrics': {
                'train_acc': model_result['train_acc'],
                'test_acc': model_result['test_acc'],
                'train_auc': model_result['train_auc'],
                'test_auc': model_result['test_auc'],
            },
            'trained_at': datetime.now().isoformat(),
        }, f)
    print(f"[OK] Model saved to: {model_path}")
    
    # Save training data info
    info_path = OUTPUT_DIR / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump({
            'total_samples': len(features_df),
            'date_range': {
                'start': features_df['fight_date'].min().isoformat(),
                'end': features_df['fight_date'].max().isoformat(),
            },
            'feature_columns': get_feature_columns(),
            'model_name': model_result['name'],
            'metrics': {
                'train_acc': model_result['train_acc'],
                'test_acc': model_result['test_acc'],
                'train_auc': model_result['train_auc'],
                'test_auc': model_result['test_auc'],
                'test_logloss': model_result['test_logloss'],
            },
            'trained_at': datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"[OK] Training info saved to: {info_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("UFC FIGHT PREDICTION MODEL TRAINING")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # 1. Load data
    fights_df, career_df, elo_df = load_training_data()
    
    # 2. Feature engineering
    features_df = engineer_features(fights_df, career_df, elo_df)
    
    # 3. Data quality check
    features_df = check_data_quality(features_df)
    
    # 4. Chronological split (NO random splitting!)
    train_df, test_df, split_date = chronological_split(features_df, test_ratio=0.2)
    
    # 5. Prepare features
    feature_cols = get_feature_columns()
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['target']
    
    print(f"\nFeature matrix shape: Train={X_train.shape}, Test={X_test.shape}")
    
    # 6. Train models
    results = []
    
    # Baseline
    baseline = train_baseline(train_df, test_df)
    results.append(baseline)
    
    # Logistic Regression
    lr_result = train_logistic_regression(X_train, y_train, X_test, y_test)
    results.append(lr_result)
    
    # Random Forest
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_result)
    
    # XGBoost
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
    results.append(xgb_result)
    
    # 7. Compare models
    compare_models(results)
    
    # 8. Check calibration of best model
    best_model_result = max([r for r in results if r and 'model' in r], key=lambda x: x['test_auc'])
    if 'model' in best_model_result:
        X_test_scaled = X_test
        if 'scaler' in best_model_result and best_model_result['scaler']:
            X_test_scaled = best_model_result['scaler'].transform(X_test)
        check_probability_calibration(
            best_model_result['model'], 
            X_test_scaled, 
            y_test, 
            best_model_result['name']
        )
    
    # 9. Backtest simulation
    if 'model' in best_model_result:
        backtest_simulation(
            test_df, 
            best_model_result['model'],
            best_model_result.get('scaler'),
            feature_cols
        )
    
    # 10. Save model
    save_model(best_model_result, features_df)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now()}")


if __name__ == '__main__':
    main()

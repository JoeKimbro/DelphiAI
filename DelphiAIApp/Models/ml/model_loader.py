"""
ML Model Loader - Load trained model and generate predictions.

Bridges between the trained model (train_model_v3.py) and live predictions
(predict_fight.py). Builds feature vectors from fighter data at prediction time.

CRITICAL INVARIANTS:
- Feature order MUST match what the model was trained on (stored in model pickle)
- Preprocessing (winsorize, impute, scale) MUST match training pipeline exactly
- NaN handling: ALWAYS impute then scale, because the model is wrapped in
  CalibratedClassifierCV which cannot handle NaN even if the base model can

Usage:
    from ml.model_loader import MLPredictor
    
    predictor = MLPredictor()
    prob_a = predictor.predict(fighter_a_data, fighter_b_data)
"""

import os
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime, date

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression

# Try to import style classifier
try:
    from ml.style_classifier import classify_style, get_style_matchup_advantage
except ModuleNotFoundError:
    from style_classifier import classify_style, get_style_matchup_advantage


class IsotonicCalibrator(BaseEstimator, ClassifierMixin):
    """
    Manually calibrate a pre-fitted classifier using isotonic regression.
    
    Replaces CalibratedClassifierCV(cv='prefit') which was removed in 
    scikit-learn >= 1.6.
    
    This class MUST live in model_loader.py because pickle stores the full
    module path. Both train_model_v3.py and backtest.py import from here.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
        self.classes_ = None

    def fit(self, X, y):
        """Fit isotonic calibration on held-out data using base model's raw probs."""
        self.classes_ = np.unique(y)
        raw_probs = self.base_estimator.predict_proba(X)[:, 1]
        self.calibrator_.fit(raw_probs, y)
        return self

    def predict_proba(self, X):
        """Return calibrated probabilities, capped to realistic fight range."""
        raw_probs = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.calibrator_.predict(raw_probs)
        # Cap at 10%-90%. Wide enough for symmetry correction to work,
        # tight enough to prevent unrealistic extreme confidence in MMA.
        calibrated = np.clip(calibrated, 0.10, 0.90)
        return np.column_stack([1 - calibrated, calibrated])




    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]



ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'

# All 22 possible features V3 can produce (before correlation dropping).
# The model's actual feature_cols may be a subset if correlation analysis dropped some.
ALL_V3_FEATURES = [
    'elo_diff', 'age_diff', 'height_diff', 'reach_diff',
    'slpm_diff', 'stracc_diff', 'tdavg_diff', 'subavg_diff',
    'kd_rate_diff', 'win_rate_diff', 'recent_form_diff',
    'finish_rate_diff', 'experience_diff', 'avg_fight_time_diff',
    'opp_elo_last3_diff', 'elo_velocity_diff', 'win_streak_diff',
    'finish_trending_diff', 'opp_quality_trending_diff',
    'avg_elo_level', 'opponent_elo', 'opponent_recent_form', 'opponent_finish_rate',
    'style_advantage', 'southpaw_advantage',
    'undefeated_prospect_diff', 'rising_prospect_diff', 'declining_veteran_diff',
    'striker_vs_grappler', 'finisher_vs_decision', 'form_velocity_diff',
    'offensive_pressure_diff',
    'elo_diff_sq', 'age_diff_sq',
    'elo_prime_interaction', 'reach_striking_interaction',
    'is_title_fight', 'debut_diff',
]


def build_feature_dict(fighter_a, fighter_b, is_title_fight=False):
    """
    Build a feature dictionary from two fighter data dicts.
    
    This is the SINGLE SOURCE OF TRUTH for feature engineering at prediction time.
    Both model_loader.predict() and backtest.run_ml_backtest() must use this function
    to ensure consistency with the training pipeline.
    
    Args:
        fighter_a: dict with fighter stats (from get_fighter_data or backtest row)
        fighter_b: dict with fighter stats
        is_title_fight: whether this is a title fight
        
    Returns:
        dict mapping feature name -> float value (may contain NaN)
    """
    def get_f(d, key, default=np.nan):
        """Get float value from dict, with fallback."""
        val = d.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def parse_height(h):
        """Convert height string like 6' 2\" to inches. Returns NaN if missing."""
        if h is None or (isinstance(h, float) and np.isnan(h)):
            return np.nan
        if isinstance(h, (int, float)):
            return float(h)  # Already numeric
        h = str(h)
        if not h or h == '--':
            return np.nan
        try:
            h = h.replace('"', '').replace("'", ' ').strip()
            parts = h.split()
            if len(parts) >= 2:
                return int(parts[0]) * 12 + int(parts[1])
            return float(parts[0]) if parts else np.nan
        except (ValueError, IndexError):
            return np.nan

    def parse_reach(r):
        """Convert reach string to inches. Returns NaN if missing."""
        if r is None or (isinstance(r, float) and np.isnan(r)):
            return np.nan
        if isinstance(r, (int, float)):
            return float(r)
        r = str(r)
        if not r or r == '--':
            return np.nan
        try:
            return float(r.replace('"', '').replace("'", '').strip())
        except (ValueError, TypeError):
            return np.nan

    def parse_pct(p):
        """Parse percentage string like '55%' to float. Returns NaN if missing."""
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return np.nan
        if isinstance(p, (int, float)):
            return float(p)  # Already numeric
        p = str(p)
        if not p or p == '--':
            return np.nan
        try:
            return float(p.replace('%', '').strip())
        except (ValueError, TypeError):
            return np.nan

    def calc_age(dob_val):
        """Calculate current age from DOB. Handles multiple formats."""
        if dob_val is None:
            return np.nan
        try:
            if isinstance(dob_val, (date, datetime)):
                return (datetime.now() - datetime(dob_val.year, dob_val.month, dob_val.day)).days / 365.25
            if isinstance(dob_val, str):
                # Try multiple date formats
                for fmt in ('%b %d, %Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y'):
                    try:
                        parsed = datetime.strptime(dob_val.strip(), fmt)
                        return (datetime.now() - parsed).days / 365.25
                    except ValueError:
                        continue
            return np.nan
        except (ValueError, TypeError, AttributeError):
            return np.nan

    def calc_age_at_date(dob_val, fight_date):
        """Calculate age at a specific date (for backtest use)."""
        if dob_val is None or fight_date is None:
            return np.nan
        try:
            if isinstance(dob_val, str):
                for fmt in ('%b %d, %Y', '%Y-%m-%d', '%m/%d/%Y'):
                    try:
                        dob_val = datetime.strptime(dob_val.strip(), fmt)
                        break
                    except ValueError:
                        continue
            if isinstance(dob_val, (date, datetime)) and isinstance(fight_date, (date, datetime)):
                dob_dt = datetime(dob_val.year, dob_val.month, dob_val.day)
                fight_dt = datetime(fight_date.year, fight_date.month, fight_date.day)
                return (fight_dt - dob_dt).days / 365.25
            return np.nan
        except (ValueError, TypeError, AttributeError):
            return np.nan

    # === ELO ===
    # Priority: final_elo (from ELO adjustment) > adjusted_elo > elo > 1500
    a_elo = get_f(fighter_a, 'final_elo',
                   get_f(fighter_a, 'adjusted_elo',
                          get_f(fighter_a, 'elo', 1500)))
    b_elo = get_f(fighter_b, 'final_elo',
                   get_f(fighter_b, 'adjusted_elo',
                          get_f(fighter_b, 'elo', 1500)))

    # === AGE ===
    # For live predictions: use current age
    # For backtest: use age at fight_date if provided
    fight_date = fighter_a.get('_fight_date') or fighter_b.get('_fight_date')
    if fight_date:
        a_age = calc_age_at_date(fighter_a.get('dob'), fight_date)
        b_age = calc_age_at_date(fighter_b.get('dob'), fight_date)
    else:
        a_age = calc_age(fighter_a.get('dob'))
        b_age = calc_age(fighter_b.get('dob'))
        # Live fallback when DOB is missing but scraped age exists.
        if np.isnan(a_age):
            a_age = get_f(fighter_a, 'age')
        if np.isnan(b_age):
            b_age = get_f(fighter_b, 'age')

    # === PHYSICAL ===
    a_height = parse_height(fighter_a.get('height'))
    b_height = parse_height(fighter_b.get('height'))
    a_reach = parse_reach(fighter_a.get('reach'))
    b_reach = parse_reach(fighter_b.get('reach'))

    # === CAREER STATS ===
    a_slpm = get_f(fighter_a, 'slpm')
    b_slpm = get_f(fighter_b, 'slpm')
    a_stracc = parse_pct(fighter_a.get('str_acc', fighter_a.get('stracc')))
    b_stracc = parse_pct(fighter_b.get('str_acc', fighter_b.get('stracc')))
    a_tdavg = get_f(fighter_a, 'td_avg', get_f(fighter_a, 'tdavg'))
    b_tdavg = get_f(fighter_b, 'td_avg', get_f(fighter_b, 'tdavg'))
    a_subavg = get_f(fighter_a, 'sub_avg', get_f(fighter_a, 'subavg'))
    b_subavg = get_f(fighter_b, 'sub_avg', get_f(fighter_b, 'subavg'))
    a_kd_rate = get_f(fighter_a, 'kd_rate')
    b_kd_rate = get_f(fighter_b, 'kd_rate')
    a_win_rate = get_f(fighter_a, 'win_rate')
    b_win_rate = get_f(fighter_b, 'win_rate')
    a_recent = get_f(fighter_a, 'recent_form', get_f(fighter_a, 'recent_win_rate'))
    b_recent = get_f(fighter_b, 'recent_form', get_f(fighter_b, 'recent_win_rate'))
    a_finish = get_f(fighter_a, 'finish_rate')
    b_finish = get_f(fighter_b, 'finish_rate')
    a_fights = get_f(fighter_a, 'fights', get_f(fighter_a, 'total_fights', 0))
    b_fights = get_f(fighter_b, 'fights', get_f(fighter_b, 'total_fights', 0))
    a_wins = get_f(fighter_a, 'wins', 0)
    b_wins = get_f(fighter_b, 'wins', 0)
    a_losses = get_f(fighter_a, 'losses', 0)
    b_losses = get_f(fighter_b, 'losses', 0)
    a_avg_time = get_f(fighter_a, 'avg_fight_duration', get_f(fighter_a, 'avg_fight_time'))
    b_avg_time = get_f(fighter_b, 'avg_fight_duration', get_f(fighter_b, 'avg_fight_time'))
    a_opp_elo_last3 = get_f(fighter_a, 'avg_opponent_elo_last_3')
    b_opp_elo_last3 = get_f(fighter_b, 'avg_opponent_elo_last_3')
    a_elo_velocity = get_f(fighter_a, 'elo_velocity')
    b_elo_velocity = get_f(fighter_b, 'elo_velocity')
    a_win_streak = get_f(fighter_a, 'current_win_streak', 0)
    b_win_streak = get_f(fighter_b, 'current_win_streak', 0)
    a_finish_trending = get_f(fighter_a, 'finish_rate_trending')
    b_finish_trending = get_f(fighter_b, 'finish_rate_trending')
    a_opp_quality_trending = get_f(fighter_a, 'opponent_quality_trending')
    b_opp_quality_trending = get_f(fighter_b, 'opponent_quality_trending')

    # If win_rate not provided, compute from wins/losses
    if np.isnan(a_win_rate):
        a_wins = get_f(fighter_a, 'wins', 0)
        a_losses = get_f(fighter_a, 'losses', 0)
        a_total = a_wins + a_losses
        a_win_rate = a_wins / a_total if a_total > 0 else np.nan
        a_fights = a_total if np.isnan(a_fights) or a_fights == 0 else a_fights
    if np.isnan(b_win_rate):
        b_wins = get_f(fighter_b, 'wins', 0)
        b_losses = get_f(fighter_b, 'losses', 0)
        b_total = b_wins + b_losses
        b_win_rate = b_wins / b_total if b_total > 0 else np.nan
        b_fights = b_total if np.isnan(b_fights) or b_fights == 0 else b_fights

    # === STYLE ===
    a_style = classify_style(
        slpm=a_slpm if not np.isnan(a_slpm) else 3.0,
        td_avg=a_tdavg if not np.isnan(a_tdavg) else 1.5,
        sub_avg=a_subavg if not np.isnan(a_subavg) else 0.5
    )
    b_style = classify_style(
        slpm=b_slpm if not np.isnan(b_slpm) else 3.0,
        td_avg=b_tdavg if not np.isnan(b_tdavg) else 1.5,
        sub_avg=b_subavg if not np.isnan(b_subavg) else 0.5
    )
    style_advantage = get_style_matchup_advantage(a_style, b_style)

    # === STANCE ===
    stance_map = {'Orthodox': 0, 'Southpaw': 1, 'Switch': 2}
    a_stance = stance_map.get(fighter_a.get('stance'), 0)
    b_stance = stance_map.get(fighter_b.get('stance'), 0)
    southpaw_adv = 1 if (a_stance == 1 and b_stance == 0) else (
        -1 if (a_stance == 0 and b_stance == 1) else 0
    )

    # === DIFFERENTIALS ===
    elo_diff = a_elo - b_elo
    age_diff = a_age - b_age if (not np.isnan(a_age) and not np.isnan(b_age)) else np.nan
    height_diff = a_height - b_height
    reach_diff = a_reach - b_reach
    slpm_diff = a_slpm - b_slpm
    stracc_diff = a_stracc - b_stracc
    tdavg_diff = a_tdavg - b_tdavg
    subavg_diff = a_subavg - b_subavg
    kd_rate_diff = a_kd_rate - b_kd_rate
    win_rate_diff = a_win_rate - b_win_rate if (not np.isnan(a_win_rate) and not np.isnan(b_win_rate)) else np.nan
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

    # === POLYNOMIAL ===
    elo_diff_sq = elo_diff ** 2 * np.sign(elo_diff)
    age_diff_sq = age_diff ** 2 * np.sign(age_diff) if not np.isnan(age_diff) else np.nan

    # === INTERACTIONS ===
    if not np.isnan(a_age) and not np.isnan(b_age):
        a_prime = 1.0 if 25 <= a_age <= 32 else 0.8
        b_prime = 1.0 if 25 <= b_age <= 32 else 0.8
        elo_prime_interaction = elo_diff * (a_prime - b_prime)
    else:
        elo_prime_interaction = np.nan

    if not np.isnan(reach_diff) and not np.isnan(slpm_diff):
        reach_striking_interaction = abs(reach_diff) * slpm_diff / 10.0
    else:
        reach_striking_interaction = np.nan




    # === CONTEXT ===
    is_title = 1 if is_title_fight else 0
    debut_a = 1 if (not np.isnan(a_fights) and a_fights == 0) else 0
    debut_b = 1 if (not np.isnan(b_fights) and b_fights == 0) else 0
    debut_diff = debut_a - debut_b

    return {
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
    }


class MLPredictor:
    """
    Load a trained model and make predictions from fighter data dicts.
    
    The model was trained on differential features (fighter A - fighter B).
    At prediction time, we build the same feature vector from live database data.
    """

    def __init__(self, model_path=None):
        """
        Load the trained model.
        
        Args:
            model_path: Path to model pickle. Defaults to model_latest.pkl.
        """
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.version = None
        self.winsorize_bounds = None
        self.median_values = None
        self.loaded = False
        self._load_error = None

        if model_path is None:
            model_path = ARTIFACTS_DIR / 'model_latest.pkl'

        self._load_model(model_path)

    def _load_model(self, model_path):
        """Load model from pickle file."""
        model_path = Path(model_path)
        if not model_path.exists():
            self._load_error = f"No model found at {model_path}"
            return

        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            self.model = data['model']
            self.scaler = data.get('scaler')
            self.feature_cols = data['feature_cols']
            self.version = data.get('version', 'unknown')
            self.winsorize_bounds = data.get('winsorize_bounds', {})
            self.median_values = data.get('median_values', {})
            self.loaded = True

            # Validate feature columns are a subset of known features
            unknown = set(self.feature_cols) - set(ALL_V3_FEATURES)
            if unknown:
                print(f"[ML WARNING] Unknown features in model: {unknown}")

        except Exception as e:
            self._load_error = str(e)
            print(f"[ML] Error loading model: {e}")

    def is_available(self):
        """Check if model is loaded and ready."""
        return self.loaded and self.model is not None

    def get_version(self):
        """Get model version string."""
        return self.version if self.loaded else None

    def get_load_error(self):
        """Get the error message if model failed to load."""
        return self._load_error

    def _apply_preprocessing(self, features_dict):
        """
        Apply the same preprocessing used during training:
        1. Select only the features the model expects (in EXACT order from training)
        2. Winsorize to training bounds
        3. Impute missing with training medians
        4. Scale if scaler was used
        
        IMPORTANT: Always impute NaN because the model is wrapped in 
        CalibratedClassifierCV which uses sklearn internals that cannot handle NaN,
        even when the base model is XGBoost.
        """
        # Build array in the EXACT order the model expects
        values = []
        for col in self.feature_cols:
            val = features_dict.get(col, np.nan)
            if val is None:
                val = np.nan

            # Winsorize to training bounds
            if col in self.winsorize_bounds and not np.isnan(val):
                bounds = self.winsorize_bounds[col]
                val = max(bounds['lower'], min(bounds['upper'], val))

            # Impute NaN with training median
            if np.isnan(val):
                val = self.median_values.get(col, 0.0)

            values.append(val)

        X = np.array(values, dtype=float).reshape(1, -1)

        # Scale if scaler was saved with model
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def predict(self, fighter_a, fighter_b, is_title_fight=False):
        """
        Predict win probability for fighter A vs fighter B.
        
        Args:
            fighter_a: dict from get_fighter_data()
            fighter_b: dict from get_fighter_data()
            is_title_fight: whether this is a title fight
            
        Returns:
            dict with:
                'prob_a': float (probability fighter A wins, i.e. class 1)
                'prob_b': float (probability fighter B wins)
                'raw_prob_a': float (pre-calibration probability from base model)
                'model_version': str
                'features': dict (the raw feature values before preprocessing)
            
            Returns None if model not available.
        """
        if not self.is_available():
            return None

        # Build feature vector using shared function (single source of truth)
        features = build_feature_dict(fighter_a, fighter_b, is_title_fight)

        # Apply preprocessing (winsorize, impute, scale) in training order
        X = self._apply_preprocessing(features)

        try:
            probs = self.model.predict_proba(X)[0]
            # Class 1 = fighter A wins (same as training target)
            prob_a = float(probs[1])
            prob_b = 1.0 - prob_a

            # Get raw (pre-calibration) probability from the base model.
            # The calibrated model clips to [0.15, 0.85] which can destroy
            # signal for the symmetry correction. The raw probability
            # preserves the relative difference between orderings.
            raw_prob_a = prob_a  # default: same as calibrated
            try:
                if hasattr(self.model, 'base_estimator'):
                    raw_probs = self.model.base_estimator.predict_proba(X)[:, 1]
                    raw_prob_a = float(raw_probs[0])
            except Exception:
                pass  # Fall back to calibrated prob

            # Sanity check: probabilities should be in [0, 1]
            prob_a = max(0.0, min(1.0, prob_a))
            prob_b = 1.0 - prob_a
        except Exception as e:
            print(f"[ML] Prediction error: {e}")
            return None

        # Serialize features for logging (convert NaN to None for JSON)
        features_safe = {}
        for k, v in features.items():
            if isinstance(v, float) and np.isnan(v):
                features_safe[k] = None
            else:
                features_safe[k] = float(v) if isinstance(v, (int, float, np.floating)) else v

        return {
            'prob_a': prob_a,
            'prob_b': prob_b,
            'raw_prob_a': raw_prob_a,
            'model_version': self.version,
            'features': features_safe,
        }



# Module-level singleton for convenience
_predictor = None


def get_predictor():
    """Get or create the module-level MLPredictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


def predict_fight(fighter_a_data, fighter_b_data, is_title_fight=False):
    """
    Convenience function to predict a fight.
    
    Returns dict with prob_a, prob_b, model_version, features.
    Returns None if model not available.
    """
    predictor = get_predictor()
    return predictor.predict(fighter_a_data, fighter_b_data, is_title_fight)



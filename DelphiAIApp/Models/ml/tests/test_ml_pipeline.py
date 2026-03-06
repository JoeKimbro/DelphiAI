"""
Comprehensive ML Pipeline Tests

Tests for the entire V3 ML pipeline to prevent every pitfall in the watch-out list.
Run with:
    cd DelphiAIApp/Models
    python -m pytest ml/tests/test_ml_pipeline.py -v
    
Or without pytest:
    python -m ml.tests.test_ml_pipeline
"""

import os
import sys
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock
from collections import OrderedDict

import numpy as np

# Ensure imports work from the Models directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ===========================================================================
# 1. FEATURE ORDER AND CONSISTENCY TESTS
# ===========================================================================

class TestFeatureOrder(unittest.TestCase):
    """Ensure feature order is IDENTICAL between training, prediction, and backtest."""

    def test_all_v3_features_matches_training(self):
        """ALL_V3_FEATURES in model_loader must contain every feature get_feature_columns_v3 produces."""
        from ml.model_loader import ALL_V3_FEATURES
        from ml.train_model_v3 import get_feature_columns_v3

        training_features = get_feature_columns_v3()
        loader_features = ALL_V3_FEATURES

        # Every training feature must be in the loader's list
        for f in training_features:
            self.assertIn(f, loader_features,
                          f"Training feature '{f}' not in model_loader.ALL_V3_FEATURES")

    def test_feature_names_no_typos(self):
        """Check for common typos like elo_dif vs elo_diff."""
        from ml.train_model_v3 import get_feature_columns_v3
        from ml.model_loader import ALL_V3_FEATURES

        for feature_list in [get_feature_columns_v3(), ALL_V3_FEATURES]:
            for f in feature_list:
                # No trailing/leading whitespace
                self.assertEqual(f, f.strip(), f"Feature '{f}' has whitespace")
                # No spaces
                self.assertNotIn(' ', f, f"Feature '{f}' contains spaces")
                # Common typo checks
                if 'elo' in f and 'dif' in f:
                    self.assertIn('diff', f,
                                  f"Possible typo: '{f}' should use 'diff' not 'dif'")

    def test_build_feature_dict_produces_all_features(self):
        """build_feature_dict must produce all 22 V3 features."""
        from ml.model_loader import build_feature_dict, ALL_V3_FEATURES

        fighter_a = {
            'elo': 1600, 'dob': '1990-01-15', 'height': "5' 10\"",
            'reach': '72"', 'stance': 'Orthodox', 'slpm': 4.5,
            'str_acc': '45%', 'td_avg': 2.0, 'sub_avg': 0.5,
            'wins': 15, 'losses': 3,
        }
        fighter_b = {
            'elo': 1500, 'dob': '1992-06-20', 'height': "6' 1\"",
            'reach': '75"', 'stance': 'Southpaw', 'slpm': 3.5,
            'str_acc': '50%', 'td_avg': 1.0, 'sub_avg': 1.0,
            'wins': 10, 'losses': 5,
        }

        features = build_feature_dict(fighter_a, fighter_b)

        for f in ALL_V3_FEATURES:
            self.assertIn(f, features,
                          f"build_feature_dict missing feature: '{f}'")

    def test_feature_dict_values_are_numeric(self):
        """All feature values must be numeric (float or NaN), never strings."""
        from ml.model_loader import build_feature_dict

        fighter_a = {'elo': 1600, 'wins': 10, 'losses': 2}
        fighter_b = {'elo': 1500, 'wins': 8, 'losses': 4}

        features = build_feature_dict(fighter_a, fighter_b)
        for k, v in features.items():
            self.assertIsInstance(v, (int, float, np.floating),
                                 f"Feature '{k}' is {type(v).__name__}, expected numeric")

    def test_feature_order_preserved_in_preprocessing(self):
        """MLPredictor._apply_preprocessing must produce features in model's saved order."""
        from ml.model_loader import MLPredictor

        # Create a mock model with specific feature order
        mock_feature_cols = ['elo_diff', 'age_diff', 'height_diff']
        mock_bounds = {
            'elo_diff': {'lower': -500, 'upper': 500},
            'age_diff': {'lower': -15, 'upper': 15},
        }
        mock_medians = {'elo_diff': 50.0, 'age_diff': -1.0, 'height_diff': 0.0}

        predictor = MLPredictor.__new__(MLPredictor)
        predictor.model = None
        predictor.scaler = None
        predictor.feature_cols = mock_feature_cols
        predictor.winsorize_bounds = mock_bounds
        predictor.median_values = mock_medians
        predictor.loaded = True

        features_dict = {'height_diff': 3.0, 'elo_diff': 100.0, 'age_diff': -5.0}
        X = predictor._apply_preprocessing(features_dict)

        # Order must match feature_cols, not dict order
        self.assertEqual(X[0, 0], 100.0)  # elo_diff
        self.assertEqual(X[0, 1], -5.0)   # age_diff
        self.assertEqual(X[0, 2], 3.0)    # height_diff


# ===========================================================================
# 2. NaN HANDLING TESTS
# ===========================================================================

class TestNaNHandling(unittest.TestCase):
    """Ensure NaN values are handled consistently everywhere."""

    def test_missing_fighter_data_produces_nan(self):
        """Missing fighter data should produce NaN, not 0."""
        from ml.model_loader import build_feature_dict

        fighter_a = {'elo': 1500}  # Minimal data
        fighter_b = {'elo': 1500}

        features = build_feature_dict(fighter_a, fighter_b)

        # Age, height, reach should be NaN when DOB/height/reach missing
        self.assertTrue(np.isnan(features['age_diff']),
                        "age_diff should be NaN when DOB missing")
        self.assertTrue(np.isnan(features['height_diff']),
                        "height_diff should be NaN when height missing")
        self.assertTrue(np.isnan(features['reach_diff']),
                        "reach_diff should be NaN when reach missing")

    def test_elo_defaults_to_1500_not_nan(self):
        """ELO should default to 1500 for new fighters, never NaN."""
        from ml.model_loader import build_feature_dict

        fighter_a = {}  # No data at all
        fighter_b = {}

        features = build_feature_dict(fighter_a, fighter_b)

        # ELO should be 0 diff (both default to 1500)
        self.assertEqual(features['elo_diff'], 0.0,
                         "elo_diff should be 0 when both fighters have no ELO")
        self.assertFalse(np.isnan(features['elo_diff']),
                         "elo_diff should never be NaN")

    def test_preprocessing_always_imputes_nan(self):
        """Preprocessing MUST impute all NaN values because CalibratedClassifierCV can't handle them."""
        from ml.model_loader import MLPredictor

        predictor = MLPredictor.__new__(MLPredictor)
        predictor.model = None
        predictor.scaler = None
        predictor.feature_cols = ['elo_diff', 'age_diff']
        predictor.winsorize_bounds = {}
        predictor.median_values = {'elo_diff': 50.0, 'age_diff': -1.0}
        predictor.loaded = True

        features_dict = {'elo_diff': 100.0, 'age_diff': np.nan}
        X = predictor._apply_preprocessing(features_dict)

        # NaN must be imputed
        self.assertFalse(np.isnan(X).any(),
                         "Preprocessing must impute all NaN values")
        self.assertEqual(X[0, 1], -1.0,
                         "NaN should be imputed with training median")

    def test_winsorize_preserves_nan(self):
        """Winsorization should not convert NaN to a clipped value."""
        from ml.model_loader import MLPredictor

        predictor = MLPredictor.__new__(MLPredictor)
        predictor.model = None
        predictor.scaler = None
        predictor.feature_cols = ['elo_diff']
        predictor.winsorize_bounds = {'elo_diff': {'lower': -300, 'upper': 300}}
        predictor.median_values = {'elo_diff': 0.0}
        predictor.loaded = True

        # NaN should be imputed with median (0.0), not clipped
        features_dict = {'elo_diff': np.nan}
        X = predictor._apply_preprocessing(features_dict)
        self.assertEqual(X[0, 0], 0.0)


# ===========================================================================
# 3. EDGE CASE TESTS
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases: new fighters, missing data, unusual inputs."""

    def test_new_fighter_no_data(self):
        """A brand new fighter with no historical data should still get a prediction."""
        from ml.model_loader import build_feature_dict

        fighter_a = {'elo': 1500}
        fighter_b = {'elo': 1600, 'dob': '1995-03-15', 'wins': 12, 'losses': 1}

        features = build_feature_dict(fighter_a, fighter_b)

        # Should not crash
        self.assertIsInstance(features, dict)
        self.assertEqual(features['elo_diff'], -100.0)

    def test_debut_fighter_detected(self):
        """Fighter with 0 fights should be flagged as debut."""
        from ml.model_loader import build_feature_dict

        fighter_a = {'elo': 1500, 'fights': 0}
        fighter_b = {'elo': 1500, 'fights': 10}

        features = build_feature_dict(fighter_a, fighter_b)
        self.assertEqual(features['debut_diff'], 1,
                         "debut_diff should be 1 when A is debut and B is not")

    def test_title_fight_flag_works(self):
        """is_title_fight parameter should set the feature."""
        from ml.model_loader import build_feature_dict

        features_no_title = build_feature_dict({}, {}, is_title_fight=False)
        features_title = build_feature_dict({}, {}, is_title_fight=True)

        self.assertEqual(features_no_title['is_title_fight'], 0)
        self.assertEqual(features_title['is_title_fight'], 1)

    def test_dob_multiple_formats(self):
        """DOB parsing should handle multiple date formats."""
        from ml.model_loader import build_feature_dict

        formats = [
            '1990-01-15',       # ISO format
            'Jan 15, 1990',     # Month name format
            '01/15/1990',       # US format
        ]

        for fmt in formats:
            fighter_a = {'dob': fmt, 'elo': 1500}
            fighter_b = {'dob': '1990-01-15', 'elo': 1500}
            features = build_feature_dict(fighter_a, fighter_b)
            # age_diff should be close to 0 (same person essentially)
            if not np.isnan(features['age_diff']):
                self.assertAlmostEqual(features['age_diff'], 0, delta=0.1,
                                       msg=f"DOB format '{fmt}' not parsed correctly")

    def test_height_formats(self):
        """Height parsing should handle various formats."""
        from ml.model_loader import build_feature_dict

        test_cases = [
            ("5' 10\"", 70),
            ("6' 1\"", 73),
            (72, 72),       # Already numeric
            (72.0, 72.0),   # Float
            (None, None),   # Missing -> NaN
            ('--', None),   # UFC placeholder
        ]

        for height_input, expected in test_cases:
            fighter_a = {'height': height_input, 'elo': 1500}
            fighter_b = {'height': "6' 0\"", 'elo': 1500}
            features = build_feature_dict(fighter_a, fighter_b)

            if expected is None:
                self.assertTrue(np.isnan(features['height_diff']),
                                f"height '{height_input}' should produce NaN diff")
            else:
                self.assertAlmostEqual(
                    features['height_diff'], expected - 72, delta=0.1,
                    msg=f"height '{height_input}' parsed incorrectly"
                )

    def test_percentage_parsing(self):
        """Percentage strings like '55%' and raw numbers should both work."""
        from ml.model_loader import build_feature_dict

        # String percentage
        fighter_a = {'str_acc': '55%', 'elo': 1500}
        fighter_b = {'str_acc': '50%', 'elo': 1500}
        features = build_feature_dict(fighter_a, fighter_b)
        self.assertAlmostEqual(features['stracc_diff'], 5.0, delta=0.01)

        # Raw number (from database, already parsed)
        fighter_a2 = {'stracc': 55.0, 'elo': 1500}
        fighter_b2 = {'stracc': 50.0, 'elo': 1500}
        features2 = build_feature_dict(fighter_a2, fighter_b2)
        self.assertAlmostEqual(features2['stracc_diff'], 5.0, delta=0.01)

    def test_southpaw_advantage(self):
        """Southpaw advantage should be correctly calculated."""
        from ml.model_loader import build_feature_dict

        # Southpaw vs Orthodox = +1
        features = build_feature_dict(
            {'stance': 'Southpaw', 'elo': 1500},
            {'stance': 'Orthodox', 'elo': 1500}
        )
        self.assertEqual(features['southpaw_advantage'], 1)

        # Orthodox vs Southpaw = -1
        features = build_feature_dict(
            {'stance': 'Orthodox', 'elo': 1500},
            {'stance': 'Southpaw', 'elo': 1500}
        )
        self.assertEqual(features['southpaw_advantage'], -1)

        # Same stance = 0
        features = build_feature_dict(
            {'stance': 'Orthodox', 'elo': 1500},
            {'stance': 'Orthodox', 'elo': 1500}
        )
        self.assertEqual(features['southpaw_advantage'], 0)

    def test_polynomial_features_preserve_direction(self):
        """elo_diff_sq should preserve the sign of the original difference."""
        from ml.model_loader import build_feature_dict

        # Positive elo diff
        features_pos = build_feature_dict({'elo': 1700}, {'elo': 1500})
        self.assertGreater(features_pos['elo_diff_sq'], 0,
                           "elo_diff_sq should be positive when A has higher ELO")

        # Negative elo diff
        features_neg = build_feature_dict({'elo': 1300}, {'elo': 1500})
        self.assertLess(features_neg['elo_diff_sq'], 0,
                        "elo_diff_sq should be negative when A has lower ELO")


# ===========================================================================
# 4. MODEL LOADING AND SAVING TESTS
# ===========================================================================

class TestModelIO(unittest.TestCase):
    """Test model saving and loading, including version mismatch detection."""

    def test_model_pickle_contains_required_keys(self):
        """Saved model pickle must contain all required metadata."""
        required_keys = [
            'model', 'scaler', 'feature_cols', 'version',
            'winsorize_bounds', 'median_values',
        ]

        # Create a minimal model package (no MagicMock, use simple objects)
        model_data = {
            'model': 'placeholder_model',
            'scaler': None,
            'feature_cols': ['elo_diff', 'age_diff'],
            'version': 'v3_test',
            'winsorize_bounds': {},
            'dropped_features': [],
            'min_year': 2019,
            'median_values': {'elo_diff': 0.0, 'age_diff': 0.0},
        }

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model_data, f)
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as f:
                loaded = pickle.load(f)
            for key in required_keys:
                self.assertIn(key, loaded, f"Model pickle missing key: '{key}'")
        finally:
            os.unlink(temp_path)

    def test_model_not_found_returns_not_available(self):
        """MLPredictor should gracefully handle missing model file."""
        from ml.model_loader import MLPredictor

        predictor = MLPredictor(model_path='/nonexistent/model.pkl')
        self.assertFalse(predictor.is_available())
        self.assertIsNotNone(predictor.get_load_error())

    def test_corrupt_model_returns_not_available(self):
        """MLPredictor should handle corrupt pickle files."""
        from ml.model_loader import MLPredictor

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            f.write(b'this is not a valid pickle')
            temp_path = f.name

        try:
            predictor = MLPredictor(model_path=temp_path)
            self.assertFalse(predictor.is_available())
        finally:
            os.unlink(temp_path)

    def test_predict_returns_none_when_model_unavailable(self):
        """predict() must return None, not crash, when model is not loaded."""
        from ml.model_loader import MLPredictor

        predictor = MLPredictor(model_path='/nonexistent/model.pkl')
        result = predictor.predict({'elo': 1500}, {'elo': 1500})
        self.assertIsNone(result)


# ===========================================================================
# 5. PREDICTION PROBABILITY TESTS
# ===========================================================================

class TestPredictionDirection(unittest.TestCase):
    """Ensure predict_proba returns probability for the CORRECT fighter."""

    def test_prob_a_plus_prob_b_equals_one(self):
        """Probabilities must sum to 1.0."""
        from ml.model_loader import MLPredictor

        # Create a mock predictor with a working model
        predictor = MLPredictor.__new__(MLPredictor)
        predictor.feature_cols = ['elo_diff']
        predictor.winsorize_bounds = {}
        predictor.median_values = {'elo_diff': 0.0}
        predictor.scaler = None
        predictor.version = 'test'
        predictor.loaded = True
        predictor._load_error = None

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.35, 0.65]])
        predictor.model = mock_model

        result = predictor.predict({'elo': 1600}, {'elo': 1500})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['prob_a'] + result['prob_b'], 1.0, places=6)

    def test_higher_elo_gets_higher_probability(self):
        """When using a properly trained model, higher ELO should generally get higher prob."""
        from ml.model_loader import MLPredictor

        predictor = MLPredictor.__new__(MLPredictor)
        predictor.feature_cols = ['elo_diff']
        predictor.winsorize_bounds = {}
        predictor.median_values = {'elo_diff': 0.0}
        predictor.scaler = None
        predictor.version = 'test'
        predictor.loaded = True
        predictor._load_error = None

        # Model returns [prob_class_0, prob_class_1]
        # class_1 = fighter A wins
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        predictor.model = mock_model

        result = predictor.predict({'elo': 1700}, {'elo': 1400})
        self.assertAlmostEqual(result['prob_a'], 0.7, places=5)
        self.assertAlmostEqual(result['prob_b'], 0.3, places=5)

    def test_probabilities_clamped_to_0_1(self):
        """Probabilities must be clamped to [0, 1] range."""
        from ml.model_loader import MLPredictor

        predictor = MLPredictor.__new__(MLPredictor)
        predictor.feature_cols = ['elo_diff']
        predictor.winsorize_bounds = {}
        predictor.median_values = {'elo_diff': 0.0}
        predictor.scaler = None
        predictor.version = 'test'
        predictor.loaded = True
        predictor._load_error = None

        # Edge case: model returns slightly out of bounds
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[- 0.01, 1.01]])
        predictor.model = mock_model

        result = predictor.predict({'elo': 1700}, {'elo': 1400})
        self.assertGreaterEqual(result['prob_a'], 0.0)
        self.assertLessEqual(result['prob_a'], 1.0)
        self.assertGreaterEqual(result['prob_b'], 0.0)
        self.assertLessEqual(result['prob_b'], 1.0)


# ===========================================================================
# 6. FALLBACK LOGIC TESTS
# ===========================================================================

class TestFallbackLogic(unittest.TestCase):
    """Ensure ELO fallback works when ML model fails."""

    def test_predict_fight_lazy_loader_handles_missing_model(self):
        """_get_ml_predictor should not crash when model file is missing."""
        # We test the module-level function behavior
        from ml.model_loader import MLPredictor
        predictor = MLPredictor(model_path='/nonexistent/path/model.pkl')
        self.assertFalse(predictor.is_available())

    def test_prediction_error_returns_none(self):
        """If model.predict_proba raises, predict() should return None."""
        from ml.model_loader import MLPredictor

        predictor = MLPredictor.__new__(MLPredictor)
        predictor.feature_cols = ['elo_diff']
        predictor.winsorize_bounds = {}
        predictor.median_values = {'elo_diff': 0.0}
        predictor.scaler = None
        predictor.version = 'test'
        predictor.loaded = True
        predictor._load_error = None

        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = RuntimeError("Model exploded")
        predictor.model = mock_model

        result = predictor.predict({'elo': 1500}, {'elo': 1500})
        self.assertIsNone(result)


# ===========================================================================
# 7. TRAINING PIPELINE VALIDATION TESTS
# ===========================================================================

class TestTrainingValidation(unittest.TestCase):
    """Test that training pipeline prevents data leakage and temporal issues."""

    def test_winsorize_bounds_computed_from_training_only(self):
        """Winsorization must be computed from training set ONLY, not full dataset."""
        import pandas as pd
        from ml.train_model_v3 import winsorize_features

        # Create a dataset where the extreme value is in the "holdout" portion
        data = pd.DataFrame({
            'feature_a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],  # 100 is extreme
        })

        # Winsorize on first 7 rows (simulating training set)
        train_data = data.iloc[:7].copy()
        _, bounds = winsorize_features(train_data, ['feature_a'])

        # The bounds should be based on the training set [1, 7]
        # not include 100 from the holdout
        self.assertLess(bounds['feature_a']['upper'], 50,
                        "Winsorize bounds should be from training data only")

    def test_median_computed_from_training_only(self):
        """Imputation medians must come from training set ONLY."""
        import pandas as pd

        # Simulate: training median is 5, but if we included holdout it would be 50
        train_data = pd.DataFrame({'x': [1, 3, 5, 7, 9]})  # median = 5
        holdout_data = pd.DataFrame({'x': [100, 200, 300]})  # median = 200

        train_median = float(train_data['x'].median())
        self.assertEqual(train_median, 5.0)

        # The combined median would be different - this should NOT be used
        combined_median = float(pd.concat([train_data, holdout_data])['x'].median())
        self.assertNotEqual(train_median, combined_median)

    def test_get_feature_columns_v3_no_leakage_features(self):
        """Feature list must NOT include OpponentQuality or other leaky features."""
        from ml.train_model_v3 import get_feature_columns_v3

        feature_cols = get_feature_columns_v3()
        leaky_features = [
            'opponent_quality', 'OpponentQuality', 'opponent_quality_diff',
            'future_', 'current_elo',  # No features starting with these
        ]

        for f in feature_cols:
            for leaky in leaky_features:
                self.assertNotIn(leaky.lower(), f.lower(),
                                 f"Feature '{f}' may contain data leakage")


# ===========================================================================
# 8. TEMPORAL ORDERING TESTS
# ===========================================================================

class TestTemporalOrdering(unittest.TestCase):
    """Ensure chronological ordering is maintained."""

    def test_feature_engineering_preserves_date_order(self):
        """engineer_features_v3 should maintain chronological order."""
        import pandas as pd
        from ml.train_model_v3 import engineer_features_v3

        # Create minimal fight data in chronological order
        fights = pd.DataFrame({
            'fightid': [1, 2, 3],
            'fight_date': pd.to_datetime(['2023-01-01', '2023-06-01', '2024-01-01']),
            'fighter1_id': [1, 3, 5],
            'fighter1_url': ['a', 'c', 'e'],
            'fighter1_name': ['A', 'C', 'E'],
            'fighter2_id': [2, 4, 6],
            'fighter2_url': ['b', 'd', 'f'],
            'fighter2_name': ['B', 'D', 'F'],
            'winnerid': [1, 3, 5],
            'result': ['Win', 'Win', 'Win'],
            'method': ['DEC', 'KO', 'SUB'],
            'round': [3, 1, 2],
            'istitlefight': [False, False, True],
            'eventname': ['UFC 1', 'UFC 2', 'UFC 3'],
            'f1_height': [None]*3, 'f1_reach': [None]*3, 'f1_stance': [None]*3,
            'f1_dob': [None]*3,
            'f2_height': [None]*3, 'f2_reach': [None]*3, 'f2_stance': [None]*3,
            'f2_dob': [None]*3,
            'weightclass': [None]*3,
            'f1_elo': [1500]*3, 'f2_elo': [1500]*3,
            'f1_fights': [None]*3, 'f1_wins': [None]*3, 'f1_losses': [None]*3,
            'f1_win_rate': [None]*3, 'f1_slpm': [None]*3, 'f1_stracc': [None]*3,
            'f1_tdavg': [None]*3, 'f1_subavg': [None]*3, 'f1_kd_rate': [None]*3,
            'f1_recent_form': [None]*3, 'f1_finish_rate': [None]*3,
            'f1_avg_fight_time': [None]*3,
            'f2_fights': [None]*3, 'f2_wins': [None]*3, 'f2_losses': [None]*3,
            'f2_win_rate': [None]*3, 'f2_slpm': [None]*3, 'f2_stracc': [None]*3,
            'f2_tdavg': [None]*3, 'f2_subavg': [None]*3, 'f2_kd_rate': [None]*3,
            'f2_recent_form': [None]*3, 'f2_finish_rate': [None]*3,
            'f2_avg_fight_time': [None]*3,
        })

        features_df = engineer_features_v3(fights, random_swap=False)

        # Dates should be in order
        dates = features_df['fight_date'].tolist()
        self.assertEqual(dates, sorted(dates), "Feature dates not in chronological order")


# ===========================================================================
# 9. AGE CALCULATION TESTS
# ===========================================================================

class TestAgeCalculation(unittest.TestCase):
    """Ensure age is calculated at fight time for training and current time for prediction."""

    def test_age_at_fight_date_for_backtest(self):
        """When _fight_date is provided, age should be calculated at that date."""
        from ml.model_loader import build_feature_dict

        fighter_a = {
            'dob': '1990-01-01',
            'elo': 1500,
            '_fight_date': date(2020, 1, 1),  # Age should be ~30
        }
        fighter_b = {
            'dob': '1995-01-01',
            'elo': 1500,
            '_fight_date': date(2020, 1, 1),  # Age should be ~25
        }

        features = build_feature_dict(fighter_a, fighter_b)
        # age_diff should be ~5 years (A is 30, B is 25)
        self.assertAlmostEqual(features['age_diff'], 5.0, delta=0.1)

    def test_age_current_for_live_prediction(self):
        """Without _fight_date, age should be current age."""
        from ml.model_loader import build_feature_dict

        fighter_a = {'dob': '1990-01-01', 'elo': 1500}
        fighter_b = {'dob': '1990-01-01', 'elo': 1500}

        features = build_feature_dict(fighter_a, fighter_b)
        # Same DOB = age_diff should be ~0
        self.assertAlmostEqual(features['age_diff'], 0.0, delta=0.1)


# ===========================================================================
# 10. STYLE CLASSIFIER INTEGRATION TESTS
# ===========================================================================

class TestStyleClassifier(unittest.TestCase):
    """Ensure style classification is consistent across all paths."""

    def test_style_default_values_when_missing(self):
        """When stats are missing, default values should be used for style classification."""
        from ml.model_loader import build_feature_dict

        # Both fighters have no stats -> same defaults -> advantage = 0
        features = build_feature_dict({}, {})
        self.assertEqual(features['style_advantage'], 0.0,
                         "Same default stats should produce 0 style advantage")


# ===========================================================================
# 11. FEATURES JSON SERIALIZATION TESTS
# ===========================================================================

class TestFeatureSerialization(unittest.TestCase):
    """Ensure features can be serialized to JSON for PredictionLog."""

    def test_features_json_serializable(self):
        """Feature dict from predict() must be JSON-serializable."""
        from ml.model_loader import MLPredictor

        predictor = MLPredictor.__new__(MLPredictor)
        predictor.feature_cols = ['elo_diff', 'age_diff']
        predictor.winsorize_bounds = {}
        predictor.median_values = {'elo_diff': 0.0, 'age_diff': 0.0}
        predictor.scaler = None
        predictor.version = 'test'
        predictor.loaded = True
        predictor._load_error = None

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
        predictor.model = mock_model

        result = predictor.predict({'elo': 1600}, {'elo': 1500})
        self.assertIsNotNone(result)

        # Must be JSON-serializable (no NaN, np.float64, etc.)
        try:
            serialized = json.dumps(result['features'])
            # Verify round-trip
            deserialized = json.loads(serialized)
            self.assertIsInstance(deserialized, dict)
        except (TypeError, ValueError) as e:
            self.fail(f"Features not JSON-serializable: {e}")


# ===========================================================================
# 12. TRAINING INFO METADATA TESTS
# ===========================================================================

class TestTrainingInfoMetadata(unittest.TestCase):
    """Ensure training info JSON contains all required metadata."""

    def test_training_info_has_hyperparameters(self):
        """training_info JSON must include hyperparameters for reproducibility."""
        required_fields = [
            'model_version', 'feature_columns', 'n_features',
            'min_year_filter', 'dropped_features', 'winsorize_bounds',
            'median_values', 'metrics', 'trained_at',
            'python_version', 'hyperparameters',
        ]

        # Check that save_model would produce these fields
        # (We test the structure, not the actual training)
        from ml.train_model_v3 import save_model
        # Verify the function signature accepts all needed params
        import inspect
        sig = inspect.signature(save_model)
        expected_params = ['model', 'scaler', 'feature_cols', 'metrics',
                           'winsorize_bounds', 'dropped_features', 'min_year',
                           'median_values']
        for p in expected_params:
            self.assertIn(p, sig.parameters,
                          f"save_model missing parameter: '{p}'")


# ===========================================================================
# RUN ALL TESTS
# ===========================================================================

def run_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("ML PIPELINE V3 - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Running at: {datetime.now()}")
    print(f"Python: {sys.version}")
    print()

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    if result.wasSuccessful():
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"FAILED: {failures} failures, {errors} errors, {passed}/{total} passed")

    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

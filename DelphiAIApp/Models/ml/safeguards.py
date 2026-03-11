"""
BETTING SAFEGUARDS AND VALIDATION

This module implements critical safeguards against:
1. Overfitting to historical data
2. Spurious correlations (especially nationality)
3. Multiple testing problems
4. Kelly criterion abuse
5. Small sample size decisions

IMPORTANT: Read and understand these safeguards before using the betting system!
"""

import numpy as np
from scipy import stats
from typing import Optional, Dict, List, Tuple
import warnings


# ============================================================================
# MINIMUM SAMPLE SIZE REQUIREMENTS
# ============================================================================

MIN_SAMPLES = {
    'overall_edge': 200,        # Minimum bets to claim any edge
    'country_edge': 100,        # Minimum fights per country
    'style_edge': 75,           # Minimum fights per style matchup
    'weight_class_edge': 100,   # Minimum fights per weight class
    'statistical_test': 50,     # Minimum for statistical significance
}


def validate_sample_size(n_samples: int, edge_type: str) -> Tuple[bool, str]:
    """
    Validate if sample size is sufficient for reliable inference.
    
    Returns (is_valid, warning_message)
    """
    min_required = MIN_SAMPLES.get(edge_type, 100)
    
    if n_samples < min_required:
        return False, (
            f"INSUFFICIENT SAMPLE SIZE: {n_samples} < {min_required} required for '{edge_type}'. "
            f"Results may be due to random chance. DO NOT bet based on this edge."
        )
    
    if n_samples < min_required * 2:
        return True, (
            f"WARNING: Sample size {n_samples} is borderline. "
            f"Edge estimates have high uncertainty."
        )
    
    return True, ""


# ============================================================================
# MULTIPLE TESTING CORRECTION (Bonferroni)
# ============================================================================

def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple testing.
    
    If you test N hypotheses, adjust alpha to alpha/N.
    This prevents false positives from testing many things.
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    return [p < adjusted_alpha for p in p_values]


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Holm-Bonferroni correction (less conservative than Bonferroni).
    
    Step-down procedure that's more powerful while controlling FWER.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    significant = [False] * n
    
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        adjusted_alpha = alpha / (n - i)
        if p < adjusted_alpha:
            significant[idx] = True
        else:
            break  # Stop at first non-significant
    
    return significant


def warn_multiple_testing(n_tests: int):
    """Issue warning about multiple testing problem."""
    if n_tests > 5:
        expected_false_positives = n_tests * 0.05
        warnings.warn(
            f"MULTIPLE TESTING WARNING: You are running {n_tests} statistical tests. "
            f"Expected false positives by chance: {expected_false_positives:.1f}. "
            f"Apply Bonferroni or Holm correction before claiming significance.",
            UserWarning
        )


# ============================================================================
# NATIONALITY EDGE SAFEGUARDS
# ============================================================================

NATIONALITY_WARNINGS = """
⚠️ NATIONALITY EDGE SAFEGUARDS ⚠️

DO NOT bet based solely on nationality. Reasons:

1. CONFOUNDING VARIABLES
   - Dagestan edge is likely WRESTLING skill, not nationality
   - Control for: fighting style, weight class, era, camp
   
2. SURVIVORSHIP BIAS
   - Only successful fighters from each country stay in UFC
   - You're seeing a biased sample of winners
   
3. SMALL SAMPLE SIZES
   - Most countries have <50 fighters total
   - Not enough data for reliable inference
   
4. CORRELATION ≠ CAUSATION
   - Nationality correlates with fighting culture
   - Fighting culture causes skill differences
   - Nationality itself has no effect on fight outcome

RECOMMENDATIONS:
- Use nationality as ONE feature with LOW weight (5-10%)
- Require 100+ fights per country before using
- Always combine with ELO, style, recent form
- Validate edge on out-of-sample data
"""


def validate_nationality_edge(
    country: str,
    n_fights: int,
    win_rate: float,
    expected_rate: float,
    p_value: float
) -> Dict:
    """
    Validate if a nationality edge is reliable.
    
    Returns validation result with warnings.
    """
    result = {
        'country': country,
        'is_valid': False,
        'can_use_as_feature': False,
        'warnings': [],
        'recommendation': None,
    }
    
    # Check sample size
    if n_fights < MIN_SAMPLES['country_edge']:
        result['warnings'].append(
            f"INSUFFICIENT DATA: {n_fights} fights < {MIN_SAMPLES['country_edge']} minimum"
        )
        result['recommendation'] = "DO NOT use this edge"
        return result
    
    # Check if edge is statistically significant with correction
    # Assume we test ~30 countries, so need much lower p-value
    adjusted_alpha = 0.05 / 30  # Bonferroni
    
    if p_value > adjusted_alpha:
        result['warnings'].append(
            f"NOT SIGNIFICANT after multiple testing correction (p={p_value:.4f} > {adjusted_alpha:.4f})"
        )
    
    # Check if edge is practically significant (>3% to overcome vig + variance)
    edge = win_rate - expected_rate
    if abs(edge) < 0.03:
        result['warnings'].append(
            f"Edge too small ({edge*100:.1f}%) - likely eaten by vig and variance"
        )
    
    # If passes all checks
    if n_fights >= MIN_SAMPLES['country_edge'] and p_value < adjusted_alpha and abs(edge) >= 0.05:
        result['is_valid'] = True
        result['can_use_as_feature'] = True
        result['recommendation'] = "Use as ONE feature with 5-10% weight, combine with other factors"
    else:
        result['can_use_as_feature'] = n_fights >= 50  # Can use as minor feature
        result['recommendation'] = "Use cautiously as minor feature only, or ignore"
    
    return result


# ============================================================================
# KELLY CRITERION SAFEGUARDS
# ============================================================================

MAX_KELLY_FRACTION = 0.25      # Never use more than 25% Kelly
MAX_BET_PERCENTAGE = 0.05      # Never bet more than 5% of bankroll
MIN_EDGE_FOR_BET = 0.03        # Need at least 3% edge to bet
EDGE_DISCOUNT = 0.30           # Reduce estimated edge by 30% (conservative)


def safe_kelly(
    estimated_edge: float,
    decimal_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 0.05,
    edge_discount: float = 0.30
) -> Dict:
    """
    Calculate safe Kelly bet size with multiple safeguards.
    
    Safeguards:
    1. Discounts edge estimate (assumes we're overconfident)
    2. Uses fractional Kelly (25% default, not full Kelly)
    3. Caps maximum bet at 5% of bankroll
    4. Requires minimum edge to bet at all
    """
    result = {
        'should_bet': False,
        'bet_amount': 0,
        'bet_percentage': 0,
        'warnings': [],
        'kelly_raw': 0,
        'kelly_fractional': 0,
    }
    
    # Discount the edge (assume we're overconfident)
    discounted_edge = estimated_edge * (1 - edge_discount)
    
    # Check minimum edge
    if discounted_edge < MIN_EDGE_FOR_BET:
        result['warnings'].append(
            f"Edge after discount ({discounted_edge*100:.1f}%) below minimum ({MIN_EDGE_FOR_BET*100:.0f}%)"
        )
        return result
    
    # Calculate win probability from edge and odds
    implied_prob = 1 / decimal_odds
    win_prob = implied_prob + discounted_edge
    
    # Kelly formula: f* = (bp - q) / b
    # where b = decimal_odds - 1, p = win_prob, q = 1 - win_prob
    b = decimal_odds - 1
    p = win_prob
    q = 1 - win_prob
    
    if b <= 0:
        result['warnings'].append("Invalid odds")
        return result
    
    kelly_raw = (b * p - q) / b
    result['kelly_raw'] = kelly_raw
    
    if kelly_raw <= 0:
        result['warnings'].append("Kelly suggests no bet (negative or zero)")
        return result
    
    # Apply fractional Kelly
    kelly_fractional = kelly_raw * kelly_fraction
    result['kelly_fractional'] = kelly_fractional
    
    # Cap at maximum bet percentage
    bet_percentage = min(kelly_fractional, max_bet_pct)
    
    if kelly_fractional > max_bet_pct:
        result['warnings'].append(
            f"Kelly ({kelly_fractional*100:.1f}%) exceeds max ({max_bet_pct*100:.0f}%), capped"
        )
    
    result['should_bet'] = True
    result['bet_percentage'] = bet_percentage
    result['bet_amount'] = bankroll * bet_percentage
    
    return result


# ============================================================================
# ELO CALIBRATION VALIDATION
# ============================================================================

def validate_elo_calibration(
    predicted_probs: List[float],
    actual_outcomes: List[int],
    n_bins: int = 10
) -> Dict:
    """
    Validate ELO probability calibration.
    
    Checks if predicted probabilities match actual outcomes.
    E.g., if we predict 60%, do fighters win 60% of the time?
    """
    predicted = np.array(predicted_probs)
    actual = np.array(actual_outcomes)
    
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_predicted = predicted[mask].mean()
            mean_actual = actual[mask].mean()
            count = mask.sum()
            calibration.append({
                'bin': i,
                'predicted': mean_predicted,
                'actual': mean_actual,
                'count': count,
                'error': abs(mean_predicted - mean_actual),
            })
    
    # Calculate calibration error
    total_error = sum(c['error'] * c['count'] for c in calibration) / len(predicted)
    
    result = {
        'calibration_error': total_error,
        'is_calibrated': total_error < 0.05,  # Within 5%
        'bins': calibration,
        'warning': None,
    }
    
    if total_error > 0.10:
        result['warning'] = (
            f"POOR CALIBRATION: Average error {total_error*100:.1f}%. "
            f"ELO probabilities are not matching actual outcomes. "
            f"Recalibrate your ELO→probability formula."
        )
    elif total_error > 0.05:
        result['warning'] = (
            f"MODERATE CALIBRATION ERROR: {total_error*100:.1f}%. "
            f"Consider recalibrating ELO formula."
        )
    
    return result


# ============================================================================
# EDGE VALIDATION
# ============================================================================

def validate_edge(
    edge_pct: float,
    n_bets: int,
    win_rate: float,
    expected_rate: float,
    context: str = "general"
) -> Dict:
    """
    Comprehensive edge validation with all safeguards.
    """
    result = {
        'edge': edge_pct,
        'is_reliable': False,
        'confidence': 'none',
        'warnings': [],
        'recommendation': None,
    }
    
    # 1. Sample size check
    is_valid, msg = validate_sample_size(n_bets, context)
    if not is_valid:
        result['warnings'].append(msg)
        result['recommendation'] = "DO NOT BET - insufficient data"
        return result
    if msg:
        result['warnings'].append(msg)
    
    # 2. Statistical significance with Bonferroni (assume 20 tests)
    from scipy.stats import binomtest
    test_result = binomtest(int(win_rate * n_bets), n_bets, expected_rate)
    p_value = test_result.pvalue
    
    adjusted_alpha = 0.05 / 20  # Conservative
    if p_value > adjusted_alpha:
        result['warnings'].append(
            f"Not significant after multiple testing correction (p={p_value:.4f})"
        )
    
    # 3. Practical significance (edge must overcome vig + variance)
    min_practical_edge = 0.03  # 3%
    if abs(edge_pct) < min_practical_edge:
        result['warnings'].append(
            f"Edge ({edge_pct*100:.1f}%) too small for practical use"
        )
    
    # 4. Confidence interval
    std_err = np.sqrt(win_rate * (1 - win_rate) / n_bets)
    ci_low = edge_pct - 1.96 * std_err
    ci_high = edge_pct + 1.96 * std_err
    
    if ci_low < 0:
        result['warnings'].append(
            f"95% CI includes zero [{ci_low*100:.1f}%, {ci_high*100:.1f}%] - edge may not exist"
        )
    
    # Determine reliability
    if len(result['warnings']) == 0:
        result['is_reliable'] = True
        result['confidence'] = 'high'
        result['recommendation'] = "Edge appears reliable. Use with fractional Kelly."
    elif len(result['warnings']) == 1:
        result['confidence'] = 'medium'
        result['recommendation'] = "Edge uncertain. Bet small or wait for more data."
    else:
        result['confidence'] = 'low'
        result['recommendation'] = "Edge unreliable. Do not bet."
    
    return result


# ============================================================================
# MAIN SAFEGUARD CHECK
# ============================================================================

def run_all_safeguards(
    edge_type: str,
    n_samples: int,
    edge_pct: float,
    win_rate: float,
    expected_rate: float,
    p_value: float = None
) -> None:
    """
    Run all safeguard checks and print comprehensive warnings.
    """
    print("\n" + "="*70)
    print("🛡️ SAFEGUARD CHECK")
    print("="*70)
    
    warnings_found = []
    
    # Sample size
    is_valid, msg = validate_sample_size(n_samples, edge_type)
    if msg:
        warnings_found.append(("Sample Size", msg))
    
    # Edge validation
    result = validate_edge(edge_pct, n_samples, win_rate, expected_rate, edge_type)
    for w in result['warnings']:
        warnings_found.append(("Edge Validation", w))
    
    # Kelly check
    if abs(edge_pct) > 0.10:
        warnings_found.append(("Kelly Risk", 
            f"Edge ({edge_pct*100:.1f}%) seems too high - possible overfit"))
    
    # Report
    if warnings_found:
        print("\n⚠️ WARNINGS FOUND:")
        for category, warning in warnings_found:
            print(f"\n  [{category}]")
            print(f"    {warning}")
    else:
        print("\n✅ All safeguard checks passed")
    
    print(f"\n📊 RECOMMENDATION: {result['recommendation']}")
    print("="*70)


if __name__ == '__main__':
    # Demo
    print(NATIONALITY_WARNINGS)
    
    # Example validation
    run_all_safeguards(
        edge_type='country_edge',
        n_samples=50,
        edge_pct=0.15,
        win_rate=0.65,
        expected_rate=0.50,
    )

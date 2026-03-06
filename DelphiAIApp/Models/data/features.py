"""
ELO Rating System for UFC Fighters

Based on FiveThirtyEight's methodology with enhancements:
- Phase 1: Base FiveThirtyEight ELO system
- Phase 2: Style matchups, reach, age, recent form modifiers
- Phase 3: Output as ML feature

FiveThirtyEight Reference:
https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
(Adapted for MMA context)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 1: FiveThirtyEight Base ELO System
# ============================================================================

class BaseEloSystem:
    """
    FiveThirtyEight-style ELO rating system.
    
    Key parameters:
    - Starting ELO: 1500
    - K-factor: Base 32, adjusted by fight importance
    - Mean reversion: Ratings drift toward 1500 over time
    - Margin of victory: Finish bonus
    """
    
    # Base parameters
    STARTING_ELO = 1500
    BASE_K_FACTOR = 32
    MEAN_REVERSION_RATE = 0.1  # 10% regression per year of inactivity
    
    # Fight importance multipliers for K-factor
    FIGHT_IMPORTANCE = {
        'title': 1.5,        # Title fights matter more
        'main_event': 1.25,  # Main events
        'regular': 1.0,      # Regular fights
    }
    
    # Finish method bonuses (margin of victory)
    FINISH_BONUS = {
        'KO/TKO': 1.3,       # Knockout shows dominance
        'SUB': 1.25,         # Submission shows skill
        'DEC': 1.0,          # Decision is baseline
        'DRAW': 0.5,         # Draw - both get partial
    }
    
    # Round finish bonus (earlier = more dominant)
    ROUND_BONUS = {
        1: 1.2,   # First round finish
        2: 1.1,   # Second round
        3: 1.05,  # Third round
        4: 1.02,  # Championship rounds
        5: 1.0,   # Fifth round
    }
    
    def __init__(self):
        self.elo_ratings = {}  # fighter_url -> current ELO
        self.elo_history = {}  # fighter_url -> list of (date, elo, opponent, result)
        self.last_fight_date = {}  # fighter_url -> last fight date
    
    def get_elo(self, fighter_url):
        """Get fighter's current ELO, or starting ELO if new."""
        return self.elo_ratings.get(fighter_url, self.STARTING_ELO)
    
    def expected_outcome(self, elo_a, elo_b):
        """
        Calculate expected win probability for fighter A.
        Uses logistic formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    def calculate_k_factor(self, fight_type, method, round_num):
        """
        Calculate dynamic K-factor based on fight importance and finish.
        """
        k = self.BASE_K_FACTOR
        
        # Fight importance
        k *= self.FIGHT_IMPORTANCE.get(fight_type, 1.0)
        
        # Method bonus
        method_upper = (method or '').upper()
        if 'KO' in method_upper or 'TKO' in method_upper:
            k *= self.FINISH_BONUS['KO/TKO']
        elif 'SUB' in method_upper:
            k *= self.FINISH_BONUS['SUB']
        elif 'DEC' in method_upper:
            k *= self.FINISH_BONUS['DEC']
        
        # Round bonus
        if round_num and round_num in self.ROUND_BONUS:
            k *= self.ROUND_BONUS[round_num]
        
        return k
    
    def apply_mean_reversion(self, fighter_url, current_date):
        """
        Apply mean reversion for inactive fighters.
        ELO drifts toward 1500 based on time since last fight.
        """
        if fighter_url not in self.last_fight_date:
            return
        
        last_date = self.last_fight_date[fighter_url]
        if not last_date or not current_date:
            return
        
        # Calculate days inactive
        try:
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, '%Y-%m-%d')
            if isinstance(current_date, str):
                current_date = datetime.strptime(current_date, '%Y-%m-%d')
            
            days_inactive = (current_date - last_date).days
            years_inactive = days_inactive / 365
            
            if years_inactive > 0.5:  # Only apply after 6 months
                current_elo = self.elo_ratings.get(fighter_url, self.STARTING_ELO)
                reversion_amount = (current_elo - self.STARTING_ELO) * self.MEAN_REVERSION_RATE * years_inactive
                self.elo_ratings[fighter_url] = current_elo - reversion_amount
        except (ValueError, TypeError):
            pass
    
    def update_elo(self, winner_url, loser_url, fight_date, fight_type='regular', 
                   method=None, round_num=None, is_draw=False):
        """
        Update ELO ratings after a fight.
        
        Returns: (winner_new_elo, loser_new_elo, elo_change)
        """
        # Apply mean reversion for inactive fighters
        self.apply_mean_reversion(winner_url, fight_date)
        self.apply_mean_reversion(loser_url, fight_date)
        
        # Get current ratings
        winner_elo = self.get_elo(winner_url)
        loser_elo = self.get_elo(loser_url)
        
        # Calculate expected outcomes
        winner_expected = self.expected_outcome(winner_elo, loser_elo)
        loser_expected = 1 - winner_expected
        
        # Calculate K-factor
        k = self.calculate_k_factor(fight_type, method, round_num)
        
        if is_draw:
            # Draw: both get partial credit
            winner_actual = 0.5
            loser_actual = 0.5
        else:
            winner_actual = 1.0
            loser_actual = 0.0
        
        # Calculate ELO changes
        winner_change = k * (winner_actual - winner_expected)
        loser_change = k * (loser_actual - loser_expected)
        
        # Update ratings
        new_winner_elo = winner_elo + winner_change
        new_loser_elo = loser_elo + loser_change
        
        self.elo_ratings[winner_url] = new_winner_elo
        self.elo_ratings[loser_url] = new_loser_elo
        
        # Update last fight dates
        self.last_fight_date[winner_url] = fight_date
        self.last_fight_date[loser_url] = fight_date
        
        # Record detailed history (for EloHistory table)
        if winner_url not in self.elo_history:
            self.elo_history[winner_url] = []
        if loser_url not in self.elo_history:
            self.elo_history[loser_url] = []
        
        # Winner history - includes ELO before fight for backtesting
        self.elo_history[winner_url].append({
            'date': fight_date,
            'elo_before': winner_elo,
            'elo_after': new_winner_elo,
            'elo_change': winner_change,
            'opponent_url': loser_url,
            'opponent_elo_before': loser_elo,
            'expected_win_prob': winner_expected,
            'result': 'draw' if is_draw else 'win',
            'method': method,
        })
        
        # Loser history
        self.elo_history[loser_url].append({
            'date': fight_date,
            'elo_before': loser_elo,
            'elo_after': new_loser_elo,
            'elo_change': loser_change,
            'opponent_url': winner_url,
            'opponent_elo_before': winner_elo,
            'expected_win_prob': loser_expected,
            'result': 'draw' if is_draw else 'loss',
            'method': method,
        })
        
        return new_winner_elo, new_loser_elo, winner_change


# ============================================================================
# PRE-UFC ELO ESTIMATION (Enhanced)
# ============================================================================

class PreUfcEloEstimator:
    """
    Estimates initial ELO for fighters with no UFC fight history.
    Uses their pre-UFC win-loss record to assign a reasonable starting ELO.
    
    Enhancements over basic record-based estimation:
    1. Logarithmic diminishing returns (better than linear for high fight counts)
    2. Career length penalty (long careers with mediocre records = weaker competition)
    3. Finish rate bonus (if available, rewards finishers)
    4. Activity/recency weighting (recent activity = higher confidence)
    5. Age-based career phase adjustment
    
    This prevents skewing the main ELO system by:
    - Using conservative estimates (regional competition varies)
    - Capping maximum/minimum adjustments
    - Penalizing inflated records from long regional careers
    """
    
    BASE_ELO = 1500
    
    # Logarithmic scaling constants
    WIN_COEFFICIENT = 15.0      # log(wins + 1) * coefficient
    LOSS_COEFFICIENT = 20.0     # log(losses + 1) * coefficient (losses hurt more)
    DRAW_COEFFICIENT = 3.0      # log(draws + 1) * coefficient
    
    # Win rate bonus thresholds (tiered)
    WIN_RATE_BONUS = {
        0.95: 60,   # 95%+ win rate - elite prospect
        0.90: 45,   # 90-95% win rate
        0.85: 35,   # 85-90% win rate  
        0.80: 25,   # 80-85% win rate
        0.75: 15,   # 75-80% win rate
        0.70: 8,    # 70-75% win rate
        0.60: 0,    # 60-70% win rate - baseline
    }
    
    # Career length penalty thresholds (fights per year of career)
    # Long career with mediocre record suggests weaker competition
    CAREER_EFFICIENCY = {
        # (min_fights_per_year, max_fights_per_year): modifier
        (4.0, float('inf')): 10,    # Very active = competitive
        (2.5, 4.0): 5,              # Active
        (1.5, 2.5): 0,              # Normal
        (0.5, 1.5): -10,            # Sparse activity
        (0, 0.5): -20,              # Very inactive (padding record?)
    }
    
    # Age-based modifiers (fighter's prime years)
    PRIME_AGE_START = 25
    PRIME_AGE_END = 32
    
    # Activity recency bonus
    RECENT_ACTIVITY_DAYS = 365    # Fought in last year
    STALE_ACTIVITY_DAYS = 1095    # 3+ years since last fight
    
    # Maximum adjustment from base (prevents unrealistic ELOs)
    MAX_POSITIVE_ADJUSTMENT = 220   # Max ~1720 starting ELO
    MAX_NEGATIVE_ADJUSTMENT = -180  # Min ~1320 starting ELO
    
    def estimate_elo(
        self, 
        wins: int, 
        losses: int, 
        draws: int = 0,
        age: float = None,
        days_since_last_fight: float = None,
        career_years: float = None,
        finish_rate: float = None,
    ) -> dict:
        """
        Estimate initial ELO based on pre-UFC record with enhancements.
        
        Args:
            wins: Number of pre-UFC wins
            losses: Number of pre-UFC losses
            draws: Number of pre-UFC draws
            age: Fighter's current age (optional)
            days_since_last_fight: Days since last fight (optional)
            career_years: Estimated career length in years (optional)
            finish_rate: Finish rate 0-1 (optional)
            
        Returns:
            dict with 'elo', 'breakdown' of components
        """
        total_fights = wins + losses + draws
        breakdown = {}
        
        if total_fights == 0:
            return {'elo': self.BASE_ELO, 'breakdown': {'base': self.BASE_ELO}}
        
        # =====================================================================
        # 1. LOGARITHMIC WIN/LOSS ADJUSTMENT
        # Using log scale provides natural diminishing returns
        # log(16 wins) = 2.77 vs log(8 wins) = 2.08 (not 2x)
        # =====================================================================
        win_adj = np.log(wins + 1) * self.WIN_COEFFICIENT if wins > 0 else 0
        loss_adj = -np.log(losses + 1) * self.LOSS_COEFFICIENT if losses > 0 else 0
        draw_adj = np.log(draws + 1) * self.DRAW_COEFFICIENT if draws > 0 else 0
        
        record_adjustment = win_adj + loss_adj + draw_adj
        breakdown['record_log'] = round(record_adjustment, 1)
        
        # =====================================================================
        # 2. WIN RATE BONUS
        # =====================================================================
        win_rate = wins / total_fights
        win_rate_bonus = 0
        for threshold, bonus in sorted(self.WIN_RATE_BONUS.items(), reverse=True):
            if win_rate >= threshold:
                win_rate_bonus = bonus
                break
        breakdown['win_rate_bonus'] = win_rate_bonus
        
        # =====================================================================
        # 3. CAREER LENGTH / EFFICIENCY PENALTY
        # A 48-year-old with 18-6 over 20 years = likely weak competition
        # vs 28-year-old with 18-6 over 5 years = more competitive
        # =====================================================================
        career_penalty = 0
        if career_years is None and age is not None:
            # Estimate career length: assume started at 20-22
            estimated_start_age = 21
            career_years = max(1, age - estimated_start_age)
        
        if career_years and career_years > 0:
            fights_per_year = total_fights / career_years
            for (min_fpy, max_fpy), modifier in self.CAREER_EFFICIENCY.items():
                if min_fpy <= fights_per_year < max_fpy:
                    career_penalty = modifier
                    break
            breakdown['career_efficiency'] = career_penalty
        
        # =====================================================================
        # 4. FINISH RATE BONUS
        # Finishers tend to be more dangerous
        # =====================================================================
        finish_bonus = 0
        if finish_rate is not None and finish_rate > 0:
            if finish_rate >= 0.80:
                finish_bonus = 20  # 80%+ finish rate
            elif finish_rate >= 0.60:
                finish_bonus = 10  # 60-80% finish rate
            elif finish_rate >= 0.40:
                finish_bonus = 5   # 40-60% finish rate
            breakdown['finish_bonus'] = finish_bonus
        
        # =====================================================================
        # 5. RECENCY / ACTIVITY BONUS
        # Recent fights = more reliable data
        # =====================================================================
        recency_bonus = 0
        if days_since_last_fight is not None:
            if days_since_last_fight <= self.RECENT_ACTIVITY_DAYS:
                recency_bonus = 15  # Active fighter
            elif days_since_last_fight <= self.RECENT_ACTIVITY_DAYS * 2:
                recency_bonus = 5   # Moderately active
            elif days_since_last_fight >= self.STALE_ACTIVITY_DAYS:
                recency_bonus = -15  # Long layoff, data may be stale
            breakdown['recency'] = recency_bonus
        
        # =====================================================================
        # 6. AGE-BASED PRIME ADJUSTMENT
        # Fighters in their prime get slight boost
        # =====================================================================
        age_bonus = 0
        if age is not None:
            if self.PRIME_AGE_START <= age <= self.PRIME_AGE_END:
                age_bonus = 10  # In prime
            elif age > 38:
                age_bonus = -15  # Past prime, likely declining
            elif age > 35:
                age_bonus = -5   # Starting to decline
            elif age < 23:
                age_bonus = 5    # Young prospect
            breakdown['age_factor'] = age_bonus
        
        # =====================================================================
        # TOTAL ADJUSTMENT
        # =====================================================================
        total_adjustment = (
            record_adjustment + 
            win_rate_bonus + 
            career_penalty + 
            finish_bonus + 
            recency_bonus + 
            age_bonus
        )
        
        # Cap the adjustment
        total_adjustment = max(
            self.MAX_NEGATIVE_ADJUSTMENT, 
            min(self.MAX_POSITIVE_ADJUSTMENT, total_adjustment)
        )
        
        final_elo = round(self.BASE_ELO + total_adjustment, 2)
        breakdown['total_adjustment'] = round(total_adjustment, 1)
        breakdown['final_elo'] = final_elo
        
        return {'elo': final_elo, 'breakdown': breakdown}
    
    def estimate_simple(self, wins: int, losses: int, draws: int = 0) -> float:
        """Simple estimation without extra data - returns just the ELO value."""
        result = self.estimate_elo(wins, losses, draws)
        return result['elo']
    
    def estimate_from_record_string(self, record: str) -> float:
        """Parse a record string like "15-3-1" and estimate ELO."""
        try:
            parts = record.replace(' ', '').split('-')
            wins = int(parts[0]) if len(parts) > 0 else 0
            losses = int(parts[1]) if len(parts) > 1 else 0
            draws = int(parts[2]) if len(parts) > 2 else 0
            return self.estimate_simple(wins, losses, draws)
        except (ValueError, IndexError):
            return self.BASE_ELO


# ============================================================================
# PHASE 2: Enhanced ELO with Additional Factors
# ============================================================================

class EnhancedEloSystem(BaseEloSystem):
    """
    Enhanced ELO system with additional modifiers:
    - Style matchup adjustments
    - Reach differential
    - Age factor (peak years vs decline)
    - Recent form (momentum)
    """
    
    # Age-related parameters
    PEAK_AGE_START = 27
    PEAK_AGE_END = 32
    AGE_DECLINE_RATE = 0.02  # 2% per year outside peak
    
    # Style matchup matrix (attacker style -> defender style -> modifier)
    # Positive = advantage for attacker, Negative = disadvantage
    STYLE_MATCHUPS = {
        'striker': {
            'striker': 0,
            'wrestler': -0.05,   # Wrestlers can take down strikers
            'grappler': -0.03,
            'balanced': 0,
        },
        'wrestler': {
            'striker': 0.05,    # Wrestlers control strikers
            'wrestler': 0,
            'grappler': -0.03,  # Grapplers can submit wrestlers
            'balanced': 0,
        },
        'grappler': {
            'striker': 0.03,
            'wrestler': 0.03,
            'grappler': 0,
            'balanced': 0,
        },
        'balanced': {
            'striker': 0,
            'wrestler': 0,
            'grappler': 0,
            'balanced': 0,
        },
    }
    
    def __init__(self, fighters_df=None):
        super().__init__()
        self.fighters_df = fighters_df
        self.fighter_styles = {}  # fighter_url -> style category
        self.fighter_ages = {}    # fighter_url -> age at time of calculation
        self.fighter_reach = {}   # fighter_url -> reach in inches
        self.recent_form = {}     # fighter_url -> recent performance score
        
        if fighters_df is not None:
            self._load_fighter_data(fighters_df)
    
    def _load_fighter_data(self, df):
        """Load fighter physical attributes for enhanced calculations."""
        for _, row in df.iterrows():
            url = row.get('fighter_url')
            if not url:
                continue
            
            # Determine style based on stance and stats
            self.fighter_styles[url] = self._determine_style(row)
            
            # Store reach
            reach = row.get('reach')
            if pd.notna(reach):
                try:
                    self.fighter_reach[url] = float(str(reach).replace('"', '').strip())
                except (ValueError, TypeError):
                    pass
            
            # Store age/DOB for age calculations
            dob = row.get('dob')
            if pd.notna(dob):
                try:
                    birth_date = pd.to_datetime(dob)
                    self.fighter_ages[url] = birth_date
                except:
                    pass
    
    def _determine_style(self, fighter_row):
        """
        Categorize fighter's style based on their stats.
        Uses canonical classifier from ml.style_classifier.
        Returns: 'striker', 'wrestler', 'grappler', or 'balanced'
        """
        try:
            from ml.style_classifier import classify_style_from_dict
        except ModuleNotFoundError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))
            from style_classifier import classify_style_from_dict
        return classify_style_from_dict(fighter_row)
    
    def get_age_factor(self, fighter_url, fight_date):
        """
        Calculate age-based modifier.
        Fighters in peak years (27-32) get no penalty.
        Outside peak: slight negative adjustment.
        """
        if fighter_url not in self.fighter_ages:
            return 1.0
        
        try:
            birth_date = self.fighter_ages[fighter_url]
            if isinstance(fight_date, str):
                fight_date = pd.to_datetime(fight_date)
            
            age = (fight_date - birth_date).days / 365.25
            
            if self.PEAK_AGE_START <= age <= self.PEAK_AGE_END:
                return 1.0  # Peak years
            elif age < self.PEAK_AGE_START:
                # Young - slight disadvantage (inexperience)
                years_under = self.PEAK_AGE_START - age
                return 1.0 - (years_under * 0.01)  # 1% per year under peak
            else:
                # Older - decline
                years_over = age - self.PEAK_AGE_END
                return 1.0 - (years_over * self.AGE_DECLINE_RATE)
        except:
            return 1.0
    
    def get_reach_advantage(self, fighter_url, opponent_url):
        """
        Calculate reach advantage modifier.
        Significant reach advantage gives slight ELO boost.
        """
        fighter_reach = self.fighter_reach.get(fighter_url)
        opponent_reach = self.fighter_reach.get(opponent_url)
        
        if fighter_reach is None or opponent_reach is None:
            return 0
        
        reach_diff = fighter_reach - opponent_reach
        
        # Every 2 inches of reach = ~1% advantage
        return reach_diff * 0.005
    
    def get_style_matchup(self, fighter_url, opponent_url):
        """
        Calculate style matchup modifier.
        Some styles have inherent advantages over others.
        """
        fighter_style = self.fighter_styles.get(fighter_url, 'balanced')
        opponent_style = self.fighter_styles.get(opponent_url, 'balanced')
        
        return self.STYLE_MATCHUPS.get(fighter_style, {}).get(opponent_style, 0)
    
    def get_recent_form(self, fighter_url):
        """
        Calculate recent form modifier based on last 3 fights.
        Winning streak = positive modifier, losing streak = negative.
        """
        history = self.elo_history.get(fighter_url, [])
        
        if len(history) < 1:
            return 0
        
        # Look at last 3 fights
        recent = history[-3:] if len(history) >= 3 else history
        
        form_score = 0
        for i, fight in enumerate(reversed(recent)):
            weight = 1.0 / (i + 1)  # More recent fights weighted higher
            if fight['result'] == 'win':
                form_score += 0.02 * weight
            elif fight['result'] == 'loss':
                form_score -= 0.02 * weight
        
        return form_score
    
    def calculate_enhanced_expected(self, fighter_url, opponent_url, fight_date):
        """
        Calculate expected outcome with all enhancements.
        """
        base_expected = self.expected_outcome(
            self.get_elo(fighter_url),
            self.get_elo(opponent_url)
        )
        
        # Apply modifiers
        age_mod = self.get_age_factor(fighter_url, fight_date)
        reach_mod = self.get_reach_advantage(fighter_url, opponent_url)
        style_mod = self.get_style_matchup(fighter_url, opponent_url)
        form_mod = self.get_recent_form(fighter_url)
        
        # Combine modifiers (multiplicative for age, additive for others)
        total_modifier = (age_mod - 1) + reach_mod + style_mod + form_mod
        
        # Adjust expected outcome
        enhanced_expected = base_expected + (total_modifier * 0.5)  # Scale factor
        
        # Clamp to valid probability range
        return max(0.05, min(0.95, enhanced_expected))
    
    def update_elo_enhanced(self, winner_url, loser_url, fight_date, fight_type='regular',
                           method=None, round_num=None, is_draw=False):
        """
        Update ELO with enhanced factors.
        """
        # Apply mean reversion
        self.apply_mean_reversion(winner_url, fight_date)
        self.apply_mean_reversion(loser_url, fight_date)
        
        # Get current ratings
        winner_elo = self.get_elo(winner_url)
        loser_elo = self.get_elo(loser_url)
        
        # Calculate enhanced expected outcomes
        winner_expected = self.calculate_enhanced_expected(winner_url, loser_url, fight_date)
        loser_expected = 1 - winner_expected
        
        # Calculate K-factor
        k = self.calculate_k_factor(fight_type, method, round_num)
        
        if is_draw:
            winner_actual = 0.5
            loser_actual = 0.5
        else:
            winner_actual = 1.0
            loser_actual = 0.0
        
        # Calculate ELO changes
        winner_change = k * (winner_actual - winner_expected)
        loser_change = k * (loser_actual - loser_expected)
        
        # Update ratings
        new_winner_elo = winner_elo + winner_change
        new_loser_elo = loser_elo + loser_change
        
        self.elo_ratings[winner_url] = new_winner_elo
        self.elo_ratings[loser_url] = new_loser_elo
        
        # Update last fight dates
        self.last_fight_date[winner_url] = fight_date
        self.last_fight_date[loser_url] = fight_date
        
        # Record history
        if winner_url not in self.elo_history:
            self.elo_history[winner_url] = []
        if loser_url not in self.elo_history:
            self.elo_history[loser_url] = []
        
        self.elo_history[winner_url].append({
            'date': str(fight_date),
            'elo': new_winner_elo,
            'change': winner_change,
            'opponent': loser_url,
            'result': 'draw' if is_draw else 'win'
        })
        self.elo_history[loser_url].append({
            'date': str(fight_date),
            'elo': new_loser_elo,
            'change': loser_change,
            'opponent': winner_url,
            'result': 'draw' if is_draw else 'loss'
        })
        
        return new_winner_elo, new_loser_elo, winner_change


# ============================================================================
# PHASE 3: ELO Calculator - Process All Fights
# ============================================================================

def calculate_all_elo_ratings(output_dir=None, use_enhanced=True):
    """
    Process all fights chronologically and calculate ELO ratings.
    
    Args:
        output_dir: Path to output directory with CSV files
        use_enhanced: Whether to use enhanced ELO system
    
    Returns:
        DataFrame with fighter URLs and their ELO ratings
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    else:
        output_dir = Path(output_dir)
    
    # Load data
    logger.info("Loading fight data...")
    fights_df = pd.read_csv(output_dir / 'fights.csv')
    fighters_df = pd.read_csv(output_dir / 'fighters.csv')
    career_stats_df = pd.read_csv(output_dir / 'career_stats.csv')
    
    logger.info(f"Loaded {len(fights_df)} fights, {len(fighters_df)} fighters")
    
    # Merge fighter stats for enhanced system
    if use_enhanced:
        # Merge career stats with fighters
        fighters_with_stats = fighters_df.merge(
            career_stats_df[['fighter_url', 'slpm', 'td_avg', 'sub_avg']],
            on='fighter_url',
            how='left'
        )
        elo_system = EnhancedEloSystem(fighters_with_stats)
        logger.info("Using Enhanced ELO system with style/age/reach modifiers")
    else:
        elo_system = BaseEloSystem()
        logger.info("Using Base FiveThirtyEight ELO system")
    
    # Parse dates and sort chronologically
    def parse_date(date_str):
        if pd.isna(date_str):
            return None
        for fmt in ['%b. %d, %Y', '%b %d, %Y', '%B %d, %Y', '%Y-%m-%d']:
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                continue
        return None
    
    fights_df['parsed_date'] = fights_df['date'].apply(parse_date)
    fights_df = fights_df.dropna(subset=['parsed_date'])
    fights_df = fights_df.sort_values('parsed_date')
    
    logger.info(f"Processing {len(fights_df)} fights with valid dates...")
    
    # Process each fight
    processed = 0
    for _, fight in fights_df.iterrows():
        fighter_url = fight.get('fighter_url')
        opponent_url = fight.get('opponent_url')
        result = fight.get('result')
        method = fight.get('method')
        round_num = fight.get('round')
        fight_date = fight.get('parsed_date')
        is_title = fight.get('is_title_fight', False)
        is_main = fight.get('is_main_event', False)
        
        if not fighter_url or not result:
            continue
        
        # Determine fight type
        if is_title:
            fight_type = 'title'
        elif is_main:
            fight_type = 'main_event'
        else:
            fight_type = 'regular'
        
        # Parse round
        try:
            round_num = int(round_num) if pd.notna(round_num) else None
        except (ValueError, TypeError):
            round_num = None
        
        # Determine winner/loser
        is_draw = result.lower() in ['draw', 'nc', 'no contest']
        
        if result.lower() == 'win':
            winner_url = fighter_url
            loser_url = opponent_url
        elif result.lower() == 'loss':
            winner_url = opponent_url
            loser_url = fighter_url
        elif is_draw:
            winner_url = fighter_url
            loser_url = opponent_url
        else:
            continue  # Skip unknown results
        
        if not loser_url:
            continue
        
        # Update ELO
        if use_enhanced:
            elo_system.update_elo_enhanced(
                winner_url, loser_url, fight_date,
                fight_type=fight_type, method=method,
                round_num=round_num, is_draw=is_draw
            )
        else:
            elo_system.update_elo(
                winner_url, loser_url, fight_date,
                fight_type=fight_type, method=method,
                round_num=round_num, is_draw=is_draw
            )
        
        processed += 1
    
    logger.info(f"Processed {processed} fights")
    
    # Create output DataFrame for fighters with UFC fights
    elo_results = []
    fighters_with_elo = set()
    
    for fighter_url, elo in elo_system.elo_ratings.items():
        history = elo_system.elo_history.get(fighter_url, [])
        # Handle both enhanced ('elo') and base ('elo_after') history formats
        if history:
            peak_elo = max([h.get('elo', h.get('elo_after', elo)) for h in history])
        else:
            peak_elo = elo
        
        elo_results.append({
            'fighter_url': fighter_url,
            'elo_rating': round(elo, 2),
            'peak_elo': round(peak_elo, 2),
            'fights_processed': len(history),
            'elo_source': 'ufc_fights',
        })
        fighters_with_elo.add(fighter_url)
    
    # =========================================================================
    # PRE-UFC ELO ESTIMATION for fighters without UFC fight history
    # =========================================================================
    logger.info("Estimating pre-UFC ELO for fighters without fight history...")
    
    pre_ufc_estimator = PreUfcEloEstimator()
    
    # Get all fighter URLs from career_stats
    all_career_urls = set(career_stats_df['fighter_url'].dropna())
    
    # Find fighters without ELO
    fighters_needing_estimate = all_career_urls - fighters_with_elo
    
    # Create a lookup for fighter data from fighters.csv (enhanced)
    fighter_data = {}
    for _, row in fighters_df.iterrows():
        url = row.get('fighter_url')
        if url:
            wins = row.get('wins', 0)
            losses = row.get('losses', 0)
            draws = row.get('draws', 0)
            age = row.get('age')
            days_since = row.get('days_since_last_fight')
            
            fighter_data[url] = {
                'wins': int(wins) if pd.notna(wins) else 0,
                'losses': int(losses) if pd.notna(losses) else 0,
                'draws': int(draws) if pd.notna(draws) else 0,
                'age': float(age) if pd.notna(age) else None,
                'days_since_last_fight': float(days_since) if pd.notna(days_since) else None,
            }
    
    pre_ufc_count = 0
    pre_ufc_details = []  # For logging
    
    for fighter_url in fighters_needing_estimate:
        data = fighter_data.get(fighter_url, {
            'wins': 0, 'losses': 0, 'draws': 0, 
            'age': None, 'days_since_last_fight': None
        })
        
        # Use enhanced estimation with all available data
        result = pre_ufc_estimator.estimate_elo(
            wins=data['wins'], 
            losses=data['losses'], 
            draws=data['draws'],
            age=data['age'],
            days_since_last_fight=data['days_since_last_fight'],
        )
        
        estimated_elo = result['elo']
        
        elo_results.append({
            'fighter_url': fighter_url,
            'elo_rating': estimated_elo,
            'peak_elo': estimated_elo,  # Peak is same as current for new fighters
            'fights_processed': 0,
            'elo_source': 'pre_ufc_estimate',
        })
        
        # Store details for logging
        pre_ufc_details.append({
            'url': fighter_url,
            'record': f"{data['wins']}-{data['losses']}-{data['draws']}",
            'age': data['age'],
            'elo': estimated_elo,
            'breakdown': result['breakdown'],
        })
        pre_ufc_count += 1
    
    logger.info(f"Estimated pre-UFC ELO for {pre_ufc_count} fighters (enhanced)")
    
    # Log a few examples
    if pre_ufc_details:
        logger.info("Sample pre-UFC estimates:")
        for detail in pre_ufc_details[:5]:
            age_str = f", age {detail['age']:.0f}" if detail['age'] else ""
            logger.info(f"  {detail['record']}{age_str} -> ELO {detail['elo']:.0f}")
    
    elo_df = pd.DataFrame(elo_results)
    
    # Save results
    elo_output_path = output_dir / 'elo_ratings.csv'
    elo_df.to_csv(elo_output_path, index=False)
    logger.info(f"Saved ELO ratings to {elo_output_path}")
    
    # Update career_stats.csv with ELO ratings
    logger.info("Updating career_stats.csv with ELO ratings...")
    
    # First, remove any existing elo columns to prevent merge suffix issues
    cols_to_remove = [col for col in career_stats_df.columns 
                      if col in ['elo_rating', 'peak_elo', 'elo_rating_old', 'peak_elo_old']]
    if cols_to_remove:
        career_stats_df = career_stats_df.drop(columns=cols_to_remove)
    
    # Now merge with fresh ELO data
    career_stats_df = career_stats_df.merge(
        elo_df[['fighter_url', 'elo_rating', 'peak_elo']],
        on='fighter_url',
        how='left'
    )
    
    career_stats_df.to_csv(output_dir / 'career_stats.csv', index=False)
    logger.info("Updated career_stats.csv with ELO ratings")
    
    # =========================================================================
    # SAVE ELO HISTORY AS CSV (for EloHistory table)
    # =========================================================================
    logger.info("Saving ELO history...")
    elo_history_records = []
    
    for fighter_url, history in elo_system.elo_history.items():
        for record in history:
            # Handle both enhanced and base history formats
            elo_after = record.get('elo', record.get('elo_after', 1500))
            elo_change = record.get('change', record.get('elo_change', 0))
            opponent_url = record.get('opponent', record.get('opponent_url'))
            elo_before = record.get('elo_before', elo_after - elo_change if elo_change else 1500)
            opponent_elo_before = record.get('opponent_elo_before', 1500)
            expected_prob = record.get('expected_win_prob', 0.5)
            
            elo_history_records.append({
                'fighter_url': fighter_url,
                'fight_date': record.get('date'),
                'opponent_url': opponent_url,
                'elo_before_fight': round(elo_before, 2),
                'opponent_elo_before_fight': round(opponent_elo_before, 2),
                'elo_after_fight': round(elo_after, 2),
                'elo_change': round(elo_change, 2),
                'result': record.get('result'),
                'method': record.get('method'),
                'expected_win_prob': round(expected_prob, 4),
                'elo_source': 'ufc_fights',
            })
    
    elo_history_df = pd.DataFrame(elo_history_records)
    elo_history_df.to_csv(output_dir / 'elo_history.csv', index=False)
    logger.info(f"Saved ELO history to {output_dir / 'elo_history.csv'} ({len(elo_history_records)} records)")
    
    # Also save JSON for detailed analysis
    history_json_output = output_dir / 'elo_history.json'
    with open(history_json_output, 'w') as f:
        serializable_history = {}
        for url, history in elo_system.elo_history.items():
            serializable_history[url] = history
        json.dump(serializable_history, f, indent=2, default=str)
    
    # =========================================================================
    # SAVE PRE-UFC CAREER DATA (for PreUfcCareer table)
    # =========================================================================
    logger.info("Saving pre-UFC career data...")
    pre_ufc_records = []
    
    for detail in pre_ufc_details:
        fighter_url = detail['url']
        record_parts = detail['record'].split('-')
        wins = int(record_parts[0]) if len(record_parts) > 0 else 0
        losses = int(record_parts[1]) if len(record_parts) > 1 else 0
        draws = int(record_parts[2]) if len(record_parts) > 2 else 0
        
        breakdown = detail.get('breakdown', {})
        
        pre_ufc_records.append({
            'fighter_url': fighter_url,
            'pre_ufc_wins': wins,
            'pre_ufc_losses': losses,
            'pre_ufc_draws': draws,
            'pre_ufc_total_fights': wins + losses + draws,
            'age_at_estimate': detail.get('age'),
            'estimated_initial_elo': detail['elo'],
            'elo_estimation_method': 'enhanced',
            # Breakdown factors
            'record_adjustment': breakdown.get('record_log'),
            'win_rate_bonus': breakdown.get('win_rate_bonus'),
            'career_efficiency_adj': breakdown.get('career_efficiency'),
            'age_factor_adj': breakdown.get('age_factor'),
            'recency_adj': breakdown.get('recency'),
            'total_adjustment': breakdown.get('total_adjustment'),
            # Org quality placeholder (to be filled manually or from other source)
            'org_quality_tier': None,
            'primary_org': None,
            'data_confidence': 'medium',
        })
    
    if pre_ufc_records:
        pre_ufc_df = pd.DataFrame(pre_ufc_records)
        pre_ufc_df.to_csv(output_dir / 'pre_ufc_career.csv', index=False)
        logger.info(f"Saved pre-UFC career data to {output_dir / 'pre_ufc_career.csv'} ({len(pre_ufc_records)} records)")
    
    # Print top fighters by ELO
    logger.info("\n" + "=" * 60)
    logger.info("TOP 20 FIGHTERS BY ELO RATING")
    logger.info("=" * 60)
    top_fighters = elo_df.nlargest(20, 'elo_rating')
    
    # Get fighter names
    name_map = dict(zip(fighters_df['fighter_url'], fighters_df['name']))
    
    for i, row in top_fighters.iterrows():
        name = name_map.get(row['fighter_url'], 'Unknown')
        logger.info(f"  {name}: {row['elo_rating']:.0f} (Peak: {row['peak_elo']:.0f})")
    
    return elo_df, elo_system


def main():
    """Main entry point for ELO calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate ELO ratings for UFC fighters')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Path to output directory with CSV files')
    parser.add_argument('--base-only', action='store_true',
                       help='Use base ELO system without enhancements')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    
    elo_df, elo_system = calculate_all_elo_ratings(
        output_dir=output_dir,
        use_enhanced=not args.base_only
    )
    
    print(f"\nELO ratings calculated for {len(elo_df)} fighters")
    print(f"Results saved to {output_dir}/elo_ratings.csv")
    print(f"Career stats updated with ELO in {output_dir}/career_stats.csv")


if __name__ == '__main__':
    main()

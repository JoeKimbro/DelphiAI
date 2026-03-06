"""
Calculate Matchup Features (differentials) for ML model performance.

Pre-calculates feature differentials between fighters to avoid
recalculating on every prediction request.

Usage:
    python calculate_matchup_features.py                    # Calculate for active fighters only
    python calculate_matchup_features.py --all-pairs        # Calculate for ALL fighter pairs (slow)
    python calculate_matchup_features.py --weight-class "Lightweight"  # Specific weight class
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from itertools import combinations
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style classification - import from canonical source
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))
try:
    from ml.style_classifier import classify_style_from_dict as classify_style, get_style_matchup_advantage
except ModuleNotFoundError:
    from style_classifier import classify_style_from_dict as classify_style, get_style_matchup_advantage


def parse_height_to_cm(height_str) -> float:
    """Parse height string to cm."""
    if pd.isna(height_str) or not height_str:
        return None
    
    height_str = str(height_str)
    
    # Try "5' 10"" format
    match = re.match(r"(\d+)'?\s*(\d+)\"?", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return round((feet * 12 + inches) * 2.54, 1)
    
    # Try just inches or cm
    try:
        val = float(re.sub(r'[^\d.]', '', height_str))
        if val > 100:  # Probably already cm
            return val
        else:  # Probably inches
            return round(val * 2.54, 1)
    except:
        return None


def parse_reach_to_cm(reach_str) -> float:
    """Parse reach string to cm."""
    if pd.isna(reach_str) or not reach_str:
        return None
    
    reach_str = str(reach_str)
    
    try:
        # Extract number
        val = float(re.sub(r'[^\d.]', '', reach_str))
        if val > 100:  # Already cm
            return val
        else:  # Inches
            return round(val * 2.54, 1)
    except:
        return None


def calculate_matchup_features(
    output_dir: Path = None,
    active_only: bool = True,
    weight_class: str = None,
    max_pairs: int = None
):
    """
    Calculate matchup features for fighter pairs.
    
    Args:
        output_dir: Directory containing CSV files
        active_only: Only calculate for active fighters
        weight_class: Filter to specific weight class
        max_pairs: Maximum number of pairs to calculate (for testing)
        
    Returns:
        DataFrame with matchup features
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    
    output_dir = Path(output_dir)
    
    # Load required data
    logger.info("Loading data...")
    fighters_df = pd.read_csv(output_dir / 'fighters.csv')
    career_df = pd.read_csv(output_dir / 'career_stats.csv')
    elo_df = pd.read_csv(output_dir / 'elo_ratings.csv')
    
    try:
        oq_df = pd.read_csv(output_dir / 'opponent_quality.csv')
        has_oq = True
    except FileNotFoundError:
        logger.warning("opponent_quality.csv not found, skipping quality metrics")
        has_oq = False
        oq_df = pd.DataFrame()
    
    logger.info(f"Loaded {len(fighters_df)} fighters, {len(career_df)} career stats, {len(elo_df)} ELO ratings")
    
    # Filter fighters
    if active_only:
        fighters_df = fighters_df[fighters_df['is_active'] == True].copy()
        logger.info(f"Filtered to {len(fighters_df)} active fighters")
    
    if weight_class:
        fighters_df = fighters_df[fighters_df['weight_class'] == weight_class].copy()
        logger.info(f"Filtered to {len(fighters_df)} fighters in {weight_class}")
    
    # Merge data
    fighter_data = fighters_df.merge(
        career_df[['fighter_url', 'slpm', 'str_acc', 'sapm', 'str_def', 
                   'td_avg', 'td_acc', 'td_def', 'sub_avg', 'elo_rating', 'peak_elo']],
        on='fighter_url',
        how='left'
    )
    
    if has_oq:
        fighter_data = fighter_data.merge(
            oq_df[['fighter_url', 'avg_opponent_elo_at_fight_time', 'quality_win_index']],
            on='fighter_url',
            how='left'
        )
    
    # Create lookup dictionaries for fast access
    fighter_lookup = {}
    for _, row in fighter_data.iterrows():
        url = row['fighter_url']
        fighter_lookup[url] = {
            'name': row.get('name'),
            'height_cm': parse_height_to_cm(row.get('height')),
            'reach_cm': parse_reach_to_cm(row.get('reach')),
            'leg_reach_cm': parse_reach_to_cm(row.get('leg_reach')),
            'age': row.get('age'),
            'weight_class': row.get('weight_class'),
            'total_fights': row.get('total_fights', 0),
            'days_since_last_fight': row.get('days_since_last_fight'),
            'win_streak': row.get('win_streak_last3', 0),
            # Stats
            'slpm': row.get('slpm'),
            'str_acc': row.get('str_acc'),
            'sapm': row.get('sapm'),
            'str_def': row.get('str_def'),
            'td_avg': row.get('td_avg'),
            'td_acc': row.get('td_acc'),
            'td_def': row.get('td_def'),
            'sub_avg': row.get('sub_avg'),
            # ELO
            'elo_rating': row.get('elo_rating'),
            'peak_elo': row.get('peak_elo'),
            # Quality
            'avg_opp_elo': row.get('avg_opponent_elo_at_fight_time') if has_oq else None,
            'quality_win_index': row.get('quality_win_index') if has_oq else None,
        }
        # Add style
        fighter_lookup[url]['style'] = classify_style(fighter_lookup[url])
    
    # Generate pairs (within same weight class for efficiency)
    logger.info("Generating fighter pairs...")
    
    # Group by weight class
    weight_classes = fighter_data['weight_class'].dropna().unique()
    all_pairs = []
    
    for wc in weight_classes:
        wc_fighters = [url for url, data in fighter_lookup.items() 
                       if data.get('weight_class') == wc]
        
        # Generate combinations within weight class
        pairs = list(combinations(wc_fighters, 2))
        all_pairs.extend(pairs)
    
    # Also add cross-weight class pairs for catch weight fights (optional)
    # For now, just same weight class
    
    logger.info(f"Generated {len(all_pairs)} fighter pairs")
    
    if max_pairs and len(all_pairs) > max_pairs:
        logger.info(f"Limiting to {max_pairs} pairs for testing")
        all_pairs = all_pairs[:max_pairs]
    
    # Calculate features for each pair
    logger.info("Calculating matchup features...")
    matchup_records = []
    
    for i, (url1, url2) in enumerate(all_pairs):
        if i > 0 and i % 10000 == 0:
            logger.info(f"  Processed {i}/{len(all_pairs)} pairs...")
        
        f1 = fighter_lookup.get(url1, {})
        f2 = fighter_lookup.get(url2, {})
        
        def safe_diff(val1, val2):
            """Calculate difference, handling None values."""
            if val1 is None or val2 is None:
                return None
            try:
                return round(float(val1) - float(val2), 2)
            except (ValueError, TypeError):
                return None
        
        record = {
            'fighter1_url': url1,
            'fighter2_url': url2,
            # Physical differentials
            'height_diff_cm': safe_diff(f1.get('height_cm'), f2.get('height_cm')),
            'reach_diff_cm': safe_diff(f1.get('reach_cm'), f2.get('reach_cm')),
            'leg_reach_diff_cm': safe_diff(f1.get('leg_reach_cm'), f2.get('leg_reach_cm')),
            'age_diff': safe_diff(f1.get('age'), f2.get('age')),
            # ELO differential
            'elo_diff': safe_diff(f1.get('elo_rating'), f2.get('elo_rating')),
            'peak_elo_diff': safe_diff(f1.get('peak_elo'), f2.get('peak_elo')),
            # Striking differentials
            'slpm_diff': safe_diff(f1.get('slpm'), f2.get('slpm')),
            'sapm_diff': safe_diff(f1.get('sapm'), f2.get('sapm')),
            'str_acc_diff': safe_diff(f1.get('str_acc'), f2.get('str_acc')),
            'str_def_diff': safe_diff(f1.get('str_def'), f2.get('str_def')),
            # Grappling differentials
            'td_avg_diff': safe_diff(f1.get('td_avg'), f2.get('td_avg')),
            'td_acc_diff': safe_diff(f1.get('td_acc'), f2.get('td_acc')),
            'td_def_diff': safe_diff(f1.get('td_def'), f2.get('td_def')),
            'sub_avg_diff': safe_diff(f1.get('sub_avg'), f2.get('sub_avg')),
            # Quality differentials
            'opponent_quality_diff': safe_diff(f1.get('avg_opp_elo'), f2.get('avg_opp_elo')),
            'win_streak_diff': safe_diff(f1.get('win_streak'), f2.get('win_streak')),
            # Activity differentials
            'days_since_fight_diff': safe_diff(f1.get('days_since_last_fight'), f2.get('days_since_last_fight')),
            'total_fights_diff': safe_diff(f1.get('total_fights'), f2.get('total_fights')),
            # Style info
            'fighter1_style': f1.get('style'),
            'fighter2_style': f2.get('style'),
            'style_matchup_advantage': get_style_matchup_advantage(
                f1.get('style', 'balanced'), 
                f2.get('style', 'balanced')
            ),
            # Metadata
            'calculated_at': datetime.now().isoformat(),
            'is_stale': False,
        }
        
        matchup_records.append(record)
    
    # Create DataFrame
    mf_df = pd.DataFrame(matchup_records)
    
    # Save results
    output_path = output_dir / 'matchup_features.csv'
    mf_df.to_csv(output_path, index=False)
    logger.info(f"Saved matchup features to {output_path} ({len(mf_df)} records)")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MATCHUP FEATURES SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total matchup pairs: {len(mf_df)}")
    logger.info(f"Unique fighters: {len(set(mf_df['fighter1_url']) | set(mf_df['fighter2_url']))}")
    
    # Sample matchups with biggest ELO differences
    if 'elo_diff' in mf_df.columns:
        biggest_mismatches = mf_df.dropna(subset=['elo_diff']).nlargest(5, 'elo_diff')
        logger.info("\nBIGGEST ELO MISMATCHES:")
        for _, row in biggest_mismatches.iterrows():
            f1_name = fighter_lookup.get(row['fighter1_url'], {}).get('name', 'Unknown')
            f2_name = fighter_lookup.get(row['fighter2_url'], {}).get('name', 'Unknown')
            logger.info(f"  {f1_name} vs {f2_name}: ELO diff {row['elo_diff']:.0f}")
    
    return mf_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate matchup features for ML')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Path to output directory with CSV files')
    parser.add_argument('--all-pairs', action='store_true',
                       help='Calculate for all fighters (not just active)')
    parser.add_argument('--weight-class', type=str, default=None,
                       help='Filter to specific weight class')
    parser.add_argument('--max-pairs', type=int, default=None,
                       help='Maximum pairs to calculate (for testing)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    
    mf_df = calculate_matchup_features(
        output_dir=output_dir,
        active_only=not args.all_pairs,
        weight_class=args.weight_class,
        max_pairs=args.max_pairs
    )
    
    print(f"\nMatchup features calculated for {len(mf_df)} pairs")


if __name__ == '__main__':
    main()

"""
Calculate Opponent Quality / Strength of Schedule metrics for each fighter.

This script analyzes each fighter's opponents to determine:
- Average opponent ELO (overall and at fight time)
- Quality breakdown (elite/good/average/below average wins/losses)
- Strength of schedule rankings

Usage:
    python calculate_opponent_quality.py
    python calculate_opponent_quality.py --output-dir ./output
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ELO tier thresholds
ELO_TIERS = {
    'elite': 1650,      # Championship level
    'good': 1550,       # Top contender
    'average': 1450,    # Mid-level
    # Below 1450 = below average
}


def calculate_opponent_quality(output_dir: Path = None):
    """
    Calculate opponent quality metrics for all fighters.
    
    Args:
        output_dir: Directory containing CSV files
        
    Returns:
        DataFrame with opponent quality metrics
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    
    output_dir = Path(output_dir)
    
    # Load required data
    logger.info("Loading data...")
    fights_df = pd.read_csv(output_dir / 'fights.csv')
    elo_df = pd.read_csv(output_dir / 'elo_ratings.csv')
    elo_history_df = pd.read_csv(output_dir / 'elo_history.csv')
    fighters_df = pd.read_csv(output_dir / 'fighters.csv')
    
    logger.info(f"Loaded {len(fights_df)} fights, {len(elo_df)} ELO ratings, {len(elo_history_df)} ELO history records")
    
    # Create ELO lookup
    current_elo = dict(zip(elo_df['fighter_url'], elo_df['elo_rating']))
    
    # Create name lookup
    name_lookup = dict(zip(fighters_df['fighter_url'], fighters_df['name']))
    
    # Get unique fighters
    all_fighter_urls = set(elo_df['fighter_url'].dropna())
    logger.info(f"Calculating opponent quality for {len(all_fighter_urls)} fighters...")
    
    opponent_quality_records = []
    
    for fighter_url in all_fighter_urls:
        # Get this fighter's fights
        fighter_fights = fights_df[fights_df['fighter_url'] == fighter_url].copy()
        
        if len(fighter_fights) == 0:
            continue
        
        # Get ELO history for this fighter (includes opponent ELO at fight time)
        fighter_elo_history = elo_history_df[elo_history_df['fighter_url'] == fighter_url].copy()
        
        # Initialize counters
        metrics = {
            'fighter_url': fighter_url,
            'fights_analyzed': 0,
            'opponent_elos': [],
            'opponent_elos_at_fight_time': [],
            'opponent_win_rates': [],
            # Quality breakdown
            'elite_wins': 0,
            'elite_losses': 0,
            'good_wins': 0,
            'good_losses': 0,
            'average_wins': 0,
            'average_losses': 0,
            'below_average_wins': 0,
            'below_average_losses': 0,
            # Recent (last 5)
            'recent_opponent_elos': [],
            'recent_elite_wins': 0,
        }
        
        # Sort fights by date for recency
        fighter_fights = fighter_fights.sort_values('date', ascending=False)
        
        for idx, (_, fight) in enumerate(fighter_fights.iterrows()):
            opponent_url = fight.get('opponent_url')
            result = fight.get('result', '').lower()
            
            if not opponent_url or result not in ['win', 'loss']:
                continue
            
            metrics['fights_analyzed'] += 1
            
            # Get opponent's current ELO
            opp_current_elo = current_elo.get(opponent_url, 1500)
            metrics['opponent_elos'].append(opp_current_elo)
            
            # Get opponent's ELO at fight time (from history)
            fight_date = fight.get('date')
            opp_elo_at_fight = None
            
            # Look up in ELO history
            if pd.notna(fight_date):
                history_match = fighter_elo_history[
                    (fighter_elo_history['fight_date'] == fight_date) &
                    (fighter_elo_history['opponent_url'] == opponent_url)
                ]
                if len(history_match) > 0:
                    opp_elo_at_fight = history_match.iloc[0].get('opponent_elo_before_fight')
            
            if opp_elo_at_fight is None:
                opp_elo_at_fight = opp_current_elo
            
            metrics['opponent_elos_at_fight_time'].append(opp_elo_at_fight)
            
            # Track recent (last 5 fights)
            if idx < 5:
                metrics['recent_opponent_elos'].append(opp_elo_at_fight)
            
            # Categorize by ELO tier
            if opp_elo_at_fight >= ELO_TIERS['elite']:
                tier = 'elite'
            elif opp_elo_at_fight >= ELO_TIERS['good']:
                tier = 'good'
            elif opp_elo_at_fight >= ELO_TIERS['average']:
                tier = 'average'
            else:
                tier = 'below_average'
            
            # Count wins/losses by tier
            if result == 'win':
                metrics[f'{tier}_wins'] += 1
                if idx < 5 and tier == 'elite':
                    metrics['recent_elite_wins'] += 1
            else:
                metrics[f'{tier}_losses'] += 1
        
        # Calculate averages
        if metrics['opponent_elos']:
            metrics['avg_opponent_elo'] = round(np.mean(metrics['opponent_elos']), 2)
        else:
            metrics['avg_opponent_elo'] = None
            
        if metrics['opponent_elos_at_fight_time']:
            metrics['avg_opponent_elo_at_fight_time'] = round(np.mean(metrics['opponent_elos_at_fight_time']), 2)
        else:
            metrics['avg_opponent_elo_at_fight_time'] = None
            
        if metrics['recent_opponent_elos']:
            metrics['recent_avg_opponent_elo'] = round(np.mean(metrics['recent_opponent_elos']), 2)
        else:
            metrics['recent_avg_opponent_elo'] = None
        
        # Calculate elite win rate
        elite_total = metrics['elite_wins'] + metrics['elite_losses']
        if elite_total > 0:
            metrics['elite_win_rate'] = round(100 * metrics['elite_wins'] / elite_total, 2)
        else:
            metrics['elite_win_rate'] = None
        
        # Calculate Quality Win Index (weighted score)
        # Elite wins worth 4, good wins 3, average 2, below average 1
        # Losses are negative
        qwi = (
            metrics['elite_wins'] * 4 - metrics['elite_losses'] * 4 +
            metrics['good_wins'] * 3 - metrics['good_losses'] * 3 +
            metrics['average_wins'] * 2 - metrics['average_losses'] * 2 +
            metrics['below_average_wins'] * 1 - metrics['below_average_losses'] * 1
        )
        metrics['quality_win_index'] = qwi
        
        # Clean up intermediate lists (not needed in output)
        del metrics['opponent_elos']
        del metrics['opponent_elos_at_fight_time']
        del metrics['opponent_win_rates']
        del metrics['recent_opponent_elos']
        
        opponent_quality_records.append(metrics)
    
    # Create DataFrame
    oq_df = pd.DataFrame(opponent_quality_records)
    
    # Calculate rankings
    if len(oq_df) > 0:
        # Rank by avg opponent ELO at fight time (higher = tougher schedule)
        oq_df['schedule_strength_rank'] = oq_df['avg_opponent_elo_at_fight_time'].rank(
            ascending=False, method='min', na_option='bottom'
        ).astype(int)
        
        # Percentile
        oq_df['schedule_strength_percentile'] = round(
            100 * (1 - oq_df['schedule_strength_rank'] / len(oq_df)), 2
        )
    
    # Add timestamp
    oq_df['last_calculated'] = datetime.now().isoformat()
    
    # Save results
    output_path = output_dir / 'opponent_quality.csv'
    oq_df.to_csv(output_path, index=False)
    logger.info(f"Saved opponent quality data to {output_path} ({len(oq_df)} records)")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("OPPONENT QUALITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Fighters analyzed: {len(oq_df)}")
    logger.info(f"Avg opponent ELO (all fighters): {oq_df['avg_opponent_elo_at_fight_time'].mean():.0f}")
    
    # Top 10 toughest schedules
    top_schedules = oq_df.nlargest(10, 'avg_opponent_elo_at_fight_time')
    logger.info("\nTOP 10 TOUGHEST SCHEDULES:")
    for _, row in top_schedules.iterrows():
        name = name_lookup.get(row['fighter_url'], 'Unknown')
        logger.info(f"  {name}: Avg opp ELO {row['avg_opponent_elo_at_fight_time']:.0f}, QWI: {row['quality_win_index']}")
    
    # Top 10 elite opponent records
    elite_fighters = oq_df[oq_df['elite_wins'] >= 3].nlargest(10, 'elite_wins')
    logger.info("\nTOP 10 MOST ELITE WINS:")
    for _, row in elite_fighters.iterrows():
        name = name_lookup.get(row['fighter_url'], 'Unknown')
        logger.info(f"  {name}: {row['elite_wins']}W-{row['elite_losses']}L vs elite")
    
    return oq_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate opponent quality metrics')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Path to output directory with CSV files')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    
    oq_df = calculate_opponent_quality(output_dir=output_dir)
    print(f"\nOpponent quality calculated for {len(oq_df)} fighters")


if __name__ == '__main__':
    main()

"""
Calculate Point-in-Time Career Stats

This script calculates what a fighter's career stats would have been
BEFORE each fight, eliminating data leakage in ML predictions.

For each fight, we compute:
- Strikes landed per minute (using only prior fights)
- Takedown average (using only prior fights)
- Win rate (using only prior fights)
- Recent form (last 3 fights)
- etc.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'output'


def parse_fight_time(round_num, time_str):
    """
    Parse fight duration in minutes.
    round_num: Which round the fight ended
    time_str: Time in the round (e.g., "4:32")
    """
    try:
        round_num = int(round_num) if pd.notna(round_num) else 3
        
        if pd.isna(time_str) or time_str == '':
            # Assume fight went the distance
            return round_num * 5.0
        
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
            time_in_round = minutes + seconds / 60.0
        else:
            time_in_round = float(time_str)
        
        # Total time = completed rounds + time in final round
        completed_rounds = round_num - 1
        total_minutes = (completed_rounds * 5.0) + time_in_round
        
        return max(total_minutes, 0.5)  # Minimum 30 seconds
    except:
        return 15.0  # Default 3 rounds


def calculate_pit_for_fighter(history, fighter_url, fight_date):
    """
    Calculate point-in-time stats for a fighter based on their prior fight history.
    """
    if len(history) == 0:
        # First fight - no prior data
        return {
            'fighter_url': fighter_url,
            'fight_date': fight_date,
            'fights_before': 0,
            'wins_before': 0,
            'losses_before': 0,
            'win_rate_before': 0.5,  # Default
            'pit_slpm': 0.0,
            'pit_str_acc': 0.0,
            'pit_td_avg': 0.0,
            'pit_sub_avg': 0.0,
            'pit_kd_rate': 0.0,
            'recent_win_rate': 0.5,
            'avg_fight_time': 10.0,
            'finish_rate': 0.0,
            'has_prior_data': False,
        }
    
    # Calculate from prior fights
    wins = sum(1 for h in history if h['result'] == 'win')
    losses = sum(1 for h in history if h['result'] == 'loss')
    total = len(history)
    
    # Calculate averages
    total_sig_strikes = sum(h['sig_strikes'] for h in history)
    total_takedowns = sum(h['takedowns'] for h in history)
    total_sub_attempts = sum(h['sub_attempts'] for h in history)
    total_knockdowns = sum(h['knockdowns'] for h in history)
    total_time = sum(h['fight_time'] for h in history)
    
    # Per-minute rates
    if total_time > 0:
        slpm = total_sig_strikes / total_time
        td_avg = (total_takedowns / total_time) * 15  # Per 15 min
        sub_avg = (total_sub_attempts / total_time) * 15
        kd_rate = total_knockdowns / total_time
    else:
        slpm = td_avg = sub_avg = kd_rate = 0.0
    
    # Win rate
    win_rate = wins / total if total > 0 else 0.5
    
    # Recent form (last 3 fights)
    recent = history[-3:] if len(history) >= 3 else history
    recent_wins = sum(1 for h in recent if h['result'] == 'win')
    recent_win_rate = recent_wins / len(recent) if recent else 0.5
    
    # Average fight time
    avg_fight_time = total_time / total if total > 0 else 10.0
    
    # Finish rate (KO/TKO or SUB wins)
    finishes = sum(1 for h in history if h['result'] == 'win' and 
                  ('KO' in str(h.get('method', '')).upper() or 
                   'SUB' in str(h.get('method', '')).upper()))
    finish_rate = finishes / wins if wins > 0 else 0.0
    
    return {
        'fighter_url': fighter_url,
        'fight_date': fight_date,
        'fights_before': total,
        'wins_before': wins,
        'losses_before': losses,
        'win_rate_before': win_rate,
        'pit_slpm': round(slpm, 2),
        'pit_str_acc': 0.0,
        'pit_td_avg': round(td_avg, 2),
        'pit_sub_avg': round(sub_avg, 2),
        'pit_kd_rate': round(kd_rate, 3),
        'recent_win_rate': round(recent_win_rate, 2),
        'avg_fight_time': round(avg_fight_time, 2),
        'finish_rate': round(finish_rate, 2),
        'has_prior_data': True,
    }


def calculate_point_in_time_stats():
    """
    Process all fights chronologically and calculate point-in-time stats.
    
    KEY: We create a PIT record for BOTH fighters in each fight,
    so opponent stats are also available for ML training.
    """
    print("="*70)
    print("CALCULATING POINT-IN-TIME CAREER STATS")
    print("="*70)
    
    # Load fights
    fights_df = pd.read_csv(OUTPUT_DIR / 'fights.csv')
    print(f"Loaded {len(fights_df)} fights")
    
    # Parse dates
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
    
    print(f"Fights with valid dates: {len(fights_df)}")
    print(f"Date range: {fights_df['parsed_date'].min()} to {fights_df['parsed_date'].max()}")
    
    # Track cumulative stats per fighter
    # Format: fighter_url -> list of fight records
    fighter_history = defaultdict(list)
    
    # Output: point-in-time stats for each fight (for BOTH fighters)
    pit_records = []
    
    # Track unique (fighter, date) pairs to avoid duplicates
    seen_records = set()
    
    print("\nProcessing fights chronologically...")
    
    for idx, fight in fights_df.iterrows():
        if len(pit_records) % 3000 == 0:
            print(f"   Processed {len(pit_records)} fight records...")
        
        fighter_url = fight['fighter_url']
        opponent_url = fight['opponent_url']
        fight_date = fight['parsed_date']
        result = fight['result']
        method = fight.get('method', '')
        
        # Parse per-fight stats
        def safe_int(val):
            if pd.isna(val):
                return 0
            try:
                return int(val)
            except:
                return 0
        
        fight_time = parse_fight_time(fight['round'], fight['time'])
        sig_strikes = safe_int(fight['sig_strikes'])
        takedowns = safe_int(fight['takedowns'])
        sub_attempts = safe_int(fight['sub_attempts'])
        knockdowns = safe_int(fight['knockdowns'])
        
        # === Process FIGHTER (if valid) ===
        if fighter_url and (fighter_url, fight_date) not in seen_records:
            history = fighter_history[fighter_url]
            pit_stats = calculate_pit_for_fighter(history, fighter_url, fight_date)
            pit_records.append(pit_stats)
            seen_records.add((fighter_url, fight_date))
            
            # Add this fight to fighter's history
            fighter_history[fighter_url].append({
                'date': fight_date,
                'result': result,
                'method': method,
                'sig_strikes': sig_strikes,
                'takedowns': takedowns,
                'sub_attempts': sub_attempts,
                'knockdowns': knockdowns,
                'fight_time': fight_time,
            })
        
        # === Process OPPONENT (if valid) ===
        if opponent_url and (opponent_url, fight_date) not in seen_records:
            history = fighter_history[opponent_url]
            pit_stats = calculate_pit_for_fighter(history, opponent_url, fight_date)
            pit_records.append(pit_stats)
            seen_records.add((opponent_url, fight_date))
            
            # Determine opponent's result (opposite of fighter's)
            opp_result = 'loss' if result == 'win' else ('win' if result == 'loss' else 'draw')
            
            # Add this fight to opponent's history
            # Note: We don't have opponent's individual stats, so use estimates
            fighter_history[opponent_url].append({
                'date': fight_date,
                'result': opp_result,
                'method': method,
                'sig_strikes': 0,  # Don't have opponent's stats
                'takedowns': 0,
                'sub_attempts': 0,
                'knockdowns': 0,
                'fight_time': fight_time,
            })
    
    print(f"\nCreated {len(pit_records)} point-in-time stat records")
    
    # Save to CSV
    pit_df = pd.DataFrame(pit_records)
    output_path = OUTPUT_DIR / 'point_in_time_stats.csv'
    pit_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    has_data = pit_df[pit_df['has_prior_data']]
    no_data = pit_df[~pit_df['has_prior_data']]
    
    print(f"\nRecords with prior fight data: {len(has_data)} ({len(has_data)/len(pit_df)*100:.1f}%)")
    print(f"Records with NO prior data (first fights): {len(no_data)} ({len(no_data)/len(pit_df)*100:.1f}%)")
    
    print(f"\nPoint-in-time SLpM distribution:")
    print(f"  Mean: {has_data['pit_slpm'].mean():.2f}")
    print(f"  Std: {has_data['pit_slpm'].std():.2f}")
    print(f"  Max: {has_data['pit_slpm'].max():.2f}")
    
    print(f"\nPoint-in-time win rate distribution:")
    print(f"  Mean: {has_data['win_rate_before'].mean():.2f}")
    print(f"  Std: {has_data['win_rate_before'].std():.2f}")
    
    return pit_df


if __name__ == '__main__':
    calculate_point_in_time_stats()
    print("\n[DONE]")

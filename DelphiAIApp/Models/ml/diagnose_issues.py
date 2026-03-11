"""
Diagnose issues with the training pipeline:
1. Why is ELO baseline only 47.7%?
2. Why does Fighter1 win 60.8%?
3. What's the relationship between position, ELO, and winning?
"""

import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

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


def main():
    print("="*70)
    print("DIAGNOSING ELO BASELINE AND CLASS IMBALANCE")
    print("="*70)
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Load fights with ELO
    query = """
    SELECT 
        f.FightID,
        f.Date as fight_date,
        f.FighterID as fighter1_id,
        f.FighterName as fighter1_name,
        f.OpponentID as fighter2_id,
        f.OpponentName as fighter2_name,
        f.WinnerID,
        e1.EloBeforeFight as f1_elo,
        e2.EloBeforeFight as f2_elo
    FROM Fights f
    LEFT JOIN EloHistory e1 ON f.FighterID = e1.FighterID AND f.Date = e1.FightDate
    LEFT JOIN EloHistory e2 ON f.OpponentID = e2.FighterID AND f.Date = e2.FightDate
    WHERE f.WinnerID IS NOT NULL
    ORDER BY f.Date
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\nTotal fights with winners: {len(df)}")
    
    # Fill missing ELO with 1500
    df['f1_elo'] = df['f1_elo'].fillna(1500)
    df['f2_elo'] = df['f2_elo'].fillna(1500)
    
    # === ANALYSIS 1: Class Balance ===
    print("\n" + "="*70)
    print("1. CLASS BALANCE ANALYSIS")
    print("="*70)
    
    df['fighter1_won'] = df['winnerid'] == df['fighter1_id']
    f1_wins = df['fighter1_won'].sum()
    f2_wins = len(df) - f1_wins
    
    print(f"\nFighter1 wins: {f1_wins} ({f1_wins/len(df)*100:.1f}%)")
    print(f"Fighter2 wins: {f2_wins} ({f2_wins/len(df)*100:.1f}%)")
    
    # === ANALYSIS 2: ELO by Position ===
    print("\n" + "="*70)
    print("2. ELO BY POSITION")
    print("="*70)
    
    print(f"\nFighter1 ELO: mean={df['f1_elo'].mean():.1f}, std={df['f1_elo'].std():.1f}")
    print(f"Fighter2 ELO: mean={df['f2_elo'].mean():.1f}, std={df['f2_elo'].std():.1f}")
    
    # Who has higher ELO?
    df['f1_higher_elo'] = df['f1_elo'] > df['f2_elo']
    df['elo_diff'] = df['f1_elo'] - df['f2_elo']
    
    f1_higher = df['f1_higher_elo'].sum()
    f2_higher = (df['f2_elo'] > df['f1_elo']).sum()
    tied = (df['f1_elo'] == df['f2_elo']).sum()
    
    print(f"\nFighter1 has higher ELO: {f1_higher} ({f1_higher/len(df)*100:.1f}%)")
    print(f"Fighter2 has higher ELO: {f2_higher} ({f2_higher/len(df)*100:.1f}%)")
    print(f"Tied ELO: {tied} ({tied/len(df)*100:.1f}%)")
    
    # === ANALYSIS 3: ELO Baseline Prediction ===
    print("\n" + "="*70)
    print("3. ELO BASELINE ANALYSIS")
    print("="*70)
    
    # Predict higher ELO fighter wins
    df['elo_pred_f1'] = df['f1_elo'] >= df['f2_elo']  # Predict F1 if higher or equal
    df['elo_correct'] = df['elo_pred_f1'] == df['fighter1_won']
    
    elo_accuracy = df['elo_correct'].mean()
    print(f"\nELO baseline accuracy (predict higher ELO): {elo_accuracy*100:.1f}%")
    
    # Break down by who has higher ELO
    f1_higher_df = df[df['f1_elo'] > df['f2_elo']]
    f2_higher_df = df[df['f2_elo'] > df['f1_elo']]
    tied_df = df[df['f1_elo'] == df['f2_elo']]
    
    if len(f1_higher_df) > 0:
        f1_higher_acc = (f1_higher_df['fighter1_won']).mean()
        print(f"\nWhen F1 has higher ELO ({len(f1_higher_df)} fights):")
        print(f"   F1 actually wins: {f1_higher_acc*100:.1f}%")
    
    if len(f2_higher_df) > 0:
        f2_higher_acc = (~f2_higher_df['fighter1_won']).mean()
        print(f"\nWhen F2 has higher ELO ({len(f2_higher_df)} fights):")
        print(f"   F2 actually wins: {f2_higher_acc*100:.1f}%")
    
    if len(tied_df) > 0:
        print(f"\nWhen ELO is tied ({len(tied_df)} fights):")
        print(f"   F1 wins: {tied_df['fighter1_won'].mean()*100:.1f}%")
    
    # === ANALYSIS 4: The Root Cause ===
    print("\n" + "="*70)
    print("4. ROOT CAUSE ANALYSIS")
    print("="*70)
    
    # Cross-tabulation
    print("\nCross-tab: ELO favorite vs Actual winner")
    print("-"*50)
    
    # Higher ELO fighter wins?
    df['higher_elo_won'] = (
        ((df['f1_elo'] > df['f2_elo']) & df['fighter1_won']) |
        ((df['f2_elo'] > df['f1_elo']) & ~df['fighter1_won'])
    )
    
    higher_elo_wins = df['higher_elo_won'].sum()
    total_non_tied = len(df) - tied
    
    print(f"\nHigher ELO fighter wins: {higher_elo_wins}/{total_non_tied} = {higher_elo_wins/total_non_tied*100:.1f}%")
    print("   [!] This is the TRUE ELO predictive power")
    
    # === ANALYSIS 5: Position Bias in Data ===
    print("\n" + "="*70)
    print("5. POSITION BIAS INVESTIGATION")
    print("="*70)
    
    # Is Fighter1 always the favorite?
    print("\nDoes Fighter1 position correlate with being favorite?")
    
    # ELO difference distribution
    print(f"\nELO difference (F1 - F2):")
    print(f"   Mean: {df['elo_diff'].mean():.1f}")
    print(f"   Std: {df['elo_diff'].std():.1f}")
    print(f"   >0 (F1 favored): {(df['elo_diff'] > 0).sum()} ({(df['elo_diff'] > 0).mean()*100:.1f}%)")
    print(f"   <0 (F2 favored): {(df['elo_diff'] < 0).sum()} ({(df['elo_diff'] < 0).mean()*100:.1f}%)")
    
    # === ANALYSIS 6: Recommended Fix ===
    print("\n" + "="*70)
    print("6. RECOMMENDED FIXES")
    print("="*70)
    
    print("""
    DIAGNOSIS:
    - Fighter1 position is biased (wins 60.8%)
    - But Fighter1 doesn't always have higher ELO
    - This creates a mismatch between ELO prediction and actual outcomes
    
    FIXES NEEDED:
    
    1. SWAP FIGHTER ORDER RANDOMLY (for training)
       - For each fight, randomly assign which fighter is "A" vs "B"
       - This eliminates positional bias
       - Target becomes 50/50
    
    2. USE HIGHER-ELO-WINS AS PROPER BASELINE
       - Don't predict F1 or F2
       - Predict "higher ELO fighter wins"
       - This should be ~59% accurate
    
    3. RE-FRAME THE PREDICTION TASK
       - Instead of "Does Fighter1 win?"
       - Use "Does the ELO favorite win?" or "Does Fighter A win?" (after random swap)
    """)
    
    # === What baseline SHOULD be ===
    print("\n" + "="*70)
    print("7. CORRECT BASELINE CALCULATION")
    print("="*70)
    
    # Exclude ties for proper baseline
    non_tied = df[df['f1_elo'] != df['f2_elo']]
    higher_elo_correct = (
        ((non_tied['f1_elo'] > non_tied['f2_elo']) & non_tied['fighter1_won']) |
        ((non_tied['f2_elo'] > non_tied['f1_elo']) & ~non_tied['fighter1_won'])
    ).sum()
    
    proper_baseline = higher_elo_correct / len(non_tied)
    print(f"\nProper ELO baseline (excluding ties): {proper_baseline*100:.1f}%")
    print(f"   (This is what the baseline SHOULD report)")


if __name__ == '__main__':
    main()

"""
Upset Potential Analyzer

Calculates the likelihood of an upset based on:
1. ELO Volatility - How much each fighter's ELO swings fight-to-fight
2. Upset History - How often a fighter loses as favorite / wins as underdog
3. Style Danger - Does the underdog's style counter the favorite?
4. Momentum - Recent win/loss streaks
5. Finish Volatility - High KO power = more unpredictable outcomes

Usage:
    from ml.upset_analyzer import analyze_upset_potential
"""

import os
import math
from pathlib import Path

try:
    from ml.style_classifier import classify_style, get_style_matchup_advantage
except ModuleNotFoundError:
    from style_classifier import classify_style, get_style_matchup_advantage


def get_elo_history(conn, fighter_id, limit=10):
    """Get recent ELO history for a fighter."""
    cur = conn.cursor()
    cur.execute('''
        SELECT elochange, expectedwinprob, result, method, fightdate
        FROM elohistory
        WHERE fighterid = %s
        ORDER BY fightdate DESC
        LIMIT %s
    ''', (fighter_id, limit))
    
    rows = cur.fetchall()
    cur.close()
    
    return [{
        'elo_change': float(r[0]) if r[0] else 0,
        'expected_win_prob': float(r[1]) if r[1] else 0.5,
        'result': r[2],
        'method': r[3] or '',
        'date': r[4],
    } for r in rows]


def get_style_record(conn, fighter_id, opponent_style):
    """Get a fighter's record vs a specific opponent style."""
    cur = conn.cursor()
    cur.execute('''
        SELECT wins, losses, totalfights, winrate
        FROM stylematchuprecord
        WHERE fighterid = %s AND opponentstyle = %s
    ''', (fighter_id, opponent_style))
    
    row = cur.fetchone()
    cur.close()
    
    if row:
        return {
            'wins': row[0] or 0,
            'losses': row[1] or 0,
            'total': row[2] or 0,
            'win_rate': float(row[3]) if row[3] else 0,
        }
    return {'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0}


def calc_elo_volatility(history):
    """
    Calculate ELO volatility (standard deviation of ELO changes).
    Higher volatility = less predictable fighter.
    Returns a score 0-1 where 1 = extremely volatile.
    """
    if len(history) < 2:
        return 0.3  # Default moderate volatility for unknowns
    
    changes = [h['elo_change'] for h in history]
    mean = sum(changes) / len(changes)
    variance = sum((c - mean) ** 2 for c in changes) / len(changes)
    std_dev = math.sqrt(variance)
    
    # Typical ELO changes: 5-30 points. 
    # std_dev > 25 is very volatile, < 10 is stable
    score = min(std_dev / 30.0, 1.0)
    return score


def calc_upset_history(history):
    """
    Calculate how often a fighter defies expectations.
    - Favorite (expected > 0.55) who loses = upset against them
    - Underdog (expected < 0.45) who wins = they caused an upset
    
    Returns: (upset_rate_as_favorite, upset_cause_rate_as_underdog)
    """
    fav_fights = 0
    fav_losses = 0
    dog_fights = 0
    dog_wins = 0
    
    for h in history:
        prob = h['expected_win_prob']
        result = (h['result'] or '').lower()
        
        if prob > 0.55:  # Was favorite
            fav_fights += 1
            if result == 'loss':
                fav_losses += 1
        elif prob < 0.45:  # Was underdog
            dog_fights += 1
            if result == 'win':
                dog_wins += 1
    
    fav_upset_rate = fav_losses / fav_fights if fav_fights > 0 else 0
    dog_win_rate = dog_wins / dog_fights if dog_fights > 0 else 0
    
    return fav_upset_rate, dog_win_rate


def calc_momentum(history):
    """
    Calculate momentum based on recent results.
    Returns a score from -1 (losing streak) to +1 (winning streak).
    """
    if not history:
        return 0
    
    streak = 0
    for h in history:
        result = (h['result'] or '').lower()
        if result == 'win':
            streak += 1
        elif result == 'loss':
            streak -= 1
            break  # Stop at first loss for win streak
        else:
            break
    
    if streak >= 0:
        # Count actual streak from start
        streak = 0
        for h in history:
            if (h['result'] or '').lower() == 'win':
                streak += 1
            else:
                break
    else:
        streak = 0
        for h in history:
            if (h['result'] or '').lower() == 'loss':
                streak -= 1
            else:
                break
    
    # Normalize: 5+ win streak = 1.0, 3+ loss streak = -1.0
    if streak > 0:
        return min(streak / 5.0, 1.0)
    else:
        return max(streak / 3.0, -1.0)


def calc_finish_volatility(f1, f2):
    """
    Calculate how volatile the fight is based on finish power.
    High KO rates on both sides = unpredictable.
    Returns 0-1 where 1 = very volatile (both fighters finish often).
    """
    f1_finish = (100 - (f1.get('decision_rate') or 50)) / 100
    f2_finish = (100 - (f2.get('decision_rate') or 50)) / 100
    
    # Combined finish threat
    combined = (f1_finish + f2_finish) / 2
    
    # KO power specifically (KOs are the most random finishes)
    f1_ko_power = f1.get('ko_last5', 0) / 5
    f2_ko_power = f2.get('ko_last5', 0) / 5
    ko_factor = (f1_ko_power + f2_ko_power) / 2
    
    # Weighted: KO power matters more than just finish rate
    return min(combined * 0.5 + ko_factor * 0.5, 1.0)


def analyze_upset_potential(conn, f1, f2, f1_win_prob, f2_win_prob):
    """
    Analyze the upset potential for a fight.
    
    Args:
        conn: Database connection
        f1: Fighter 1 data dict
        f2: Fighter 2 data dict
        f1_win_prob: Fighter 1 adjusted win probability
        f2_win_prob: Fighter 2 adjusted win probability
    
    Returns dict with:
        - upset_score: 0-100 (higher = more likely upset)
        - risk_level: LOW / MODERATE / HIGH / VERY HIGH
        - reasons: List of human-readable reason strings
        - components: Dict of individual factor scores
    """
    # Determine favorite and underdog
    if f1_win_prob >= f2_win_prob:
        fav, dog = f1, f2
        fav_prob, dog_prob = f1_win_prob, f2_win_prob
    else:
        fav, dog = f2, f1
        fav_prob, dog_prob = f2_win_prob, f1_win_prob
    
    reasons = []
    components = {}
    
    # If it's basically a coin flip, upset isn't meaningful
    edge = fav_prob - 0.5
    if edge < 0.03:
        return {
            'upset_score': 50,
            'risk_level': 'EVEN',
            'reasons': ['Fight is too close to call - no clear favorite'],
            'components': {},
        }
    
    # ---- 1. ELO Volatility ----
    fav_history = get_elo_history(conn, fav['id'])
    dog_history = get_elo_history(conn, dog['id'])
    
    fav_volatility = calc_elo_volatility(fav_history)
    dog_volatility = calc_elo_volatility(dog_history)
    
    # Volatile favorite = more upset risk. Volatile underdog = more upset potential.
    volatility_score = (fav_volatility * 0.6 + dog_volatility * 0.4)
    components['elo_volatility'] = round(volatility_score * 100)
    
    if fav_volatility > 0.5:
        reasons.append(f"{fav['name']} is inconsistent (high ELO swings)")
    if dog_volatility > 0.5:
        reasons.append(f"{dog['name']} is a wildcard (volatile performer)")
    
    # ---- 2. Upset History ----
    fav_upset_rate, _ = calc_upset_history(fav_history)
    _, dog_upset_cause = calc_upset_history(dog_history)
    
    upset_history_score = (fav_upset_rate * 0.6 + dog_upset_cause * 0.4)
    components['upset_history'] = round(upset_history_score * 100)
    
    if fav_upset_rate > 0.3 and len(fav_history) >= 3:
        reasons.append(f"{fav['name']} has been upset before ({fav_upset_rate:.0%} loss rate as favorite)")
    if dog_upset_cause > 0.4 and len(dog_history) >= 3:
        reasons.append(f"{dog['name']} has upset history ({dog_upset_cause:.0%} win rate as underdog)")
    
    # ---- 3. Style Danger ----
    fav_style = classify_style(slpm=fav['slpm'], td_avg=fav['td_avg'], sub_avg=fav['sub_avg'])
    dog_style = classify_style(slpm=dog['slpm'], td_avg=dog['td_avg'], sub_avg=dog['sub_avg'])
    
    style_adv = get_style_matchup_advantage(dog_style, fav_style)
    
    # Check underdog's record vs favorite's style
    dog_vs_fav_style = get_style_record(conn, dog['id'], fav_style)
    fav_vs_dog_style = get_style_record(conn, fav['id'], dog_style)
    
    style_score = 0
    if style_adv > 0:
        style_score = 0.6
        reasons.append(f"Style advantage: {dog['name']} ({dog_style}) counters {fav['name']} ({fav_style})")
    elif style_adv < 0:
        style_score = 0.0  # Favorite has style advantage - reduces upset risk
    
    # If underdog has great record vs favorite's style
    if dog_vs_fav_style['total'] >= 2 and dog_vs_fav_style['win_rate'] > 70:
        style_score = max(style_score, 0.5)
        reasons.append(f"{dog['name']} is {dog_vs_fav_style['wins']}-{dog_vs_fav_style['losses']} vs {fav_style}s")
    
    # If favorite struggles vs underdog's style
    if fav_vs_dog_style['total'] >= 2 and fav_vs_dog_style['win_rate'] < 50:
        style_score = max(style_score, 0.5)
        reasons.append(f"{fav['name']} struggles vs {dog_style}s ({fav_vs_dog_style['wins']}-{fav_vs_dog_style['losses']})")
    
    components['style_danger'] = round(style_score * 100)
    
    # ---- 4. Momentum ----
    fav_momentum = calc_momentum(fav_history)
    dog_momentum = calc_momentum(dog_history)
    
    # Bad momentum for favorite + good momentum for underdog = upset risk
    momentum_score = 0
    if fav_momentum < 0:
        momentum_score += abs(fav_momentum) * 0.5
        reasons.append(f"{fav['name']} on a losing streak")
    if dog_momentum > 0.4:
        momentum_score += dog_momentum * 0.3
        reasons.append(f"{dog['name']} on a win streak")
    
    momentum_score = min(momentum_score, 1.0)
    components['momentum'] = round(momentum_score * 100)
    
    # ---- 5. Finish Volatility ----
    finish_vol = calc_finish_volatility(fav, dog)
    components['finish_volatility'] = round(finish_vol * 100)
    
    if finish_vol > 0.5:
        reasons.append("High KO power on both sides (anything can happen)")
    
    # ---- Combine into final score ----
    # Weighted combination
    weights = {
        'elo_volatility': 0.20,
        'upset_history': 0.25,
        'style_danger': 0.25,
        'momentum': 0.15,
        'finish_volatility': 0.15,
    }
    
    raw_score = (
        volatility_score * weights['elo_volatility'] +
        upset_history_score * weights['upset_history'] +
        style_score * weights['style_danger'] +
        momentum_score * weights['momentum'] +
        finish_vol * weights['finish_volatility']
    )
    
    # Scale: also factor in how lopsided the fight is
    # Bigger favorite = lower base upset chance, but our factors can push it up
    base_upset = dog_prob  # The underdog's win prob is the base upset chance
    
    # Adjust base by our factors (can push up by up to 15%)
    adjustment = raw_score * 0.15
    upset_pct = min(base_upset + adjustment, 0.65)  # Cap at 65%
    upset_score = round(upset_pct * 100)
    
    # Risk level
    if upset_score >= 45:
        risk_level = "VERY HIGH"
    elif upset_score >= 35:
        risk_level = "HIGH"
    elif upset_score >= 25:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    # Ensure we have at least one reason
    if not reasons:
        if upset_score >= 35:
            reasons.append("Close matchup with competitive stats")
        else:
            reasons.append("Favorite has clear advantages across the board")
    
    return {
        'upset_score': upset_score,
        'risk_level': risk_level,
        'reasons': reasons[:4],  # Max 4 reasons to keep output clean
        'components': components,
        'fav_name': fav['name'],
        'dog_name': dog['name'],
    }

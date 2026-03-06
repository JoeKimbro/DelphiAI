"""
Fighter Style Classification - Single source of truth.

Classifies fighters as: striker, wrestler, grappler, or balanced
based on their career stats. All other files should import from here.

Usage:
    from ml.style_classifier import classify_style, get_style_matchup_advantage
"""


def classify_style(slpm=0, td_avg=0, sub_avg=0, str_def=0, td_def=0, **kwargs):
    """
    Classify a fighter's style based on their stats.
    
    Args:
        slpm: Significant strikes landed per minute
        td_avg: Takedown average per 15 minutes
        sub_avg: Submission average per 15 minutes
        str_def: Striking defense % (0-100)
        td_def: Takedown defense % (0-100)
    
    Returns: 'striker', 'wrestler', 'grappler', or 'balanced'
    
    Thresholds (consolidated from 4 prior implementations):
        Wrestler:  TD avg > 2.5 per 15 min
        Grappler:  Sub avg > 1.0 per 15 min (and not already wrestler)
        Striker:   SLpM > 4.0 and TD avg < 2.0 (offensive on feet, not grappling much)
        Balanced:  Everything else
    """
    try:
        slpm = float(slpm or 0)
        td_avg = float(td_avg or 0)
        sub_avg = float(sub_avg or 0)
    except (ValueError, TypeError):
        return 'balanced'
    
    # Wrestler: consistently takes people down
    if td_avg > 2.5:
        return 'wrestler'
    
    # Grappler: hunts submissions
    if sub_avg > 1.0:
        return 'grappler'
    
    # Striker: high output on feet + doesn't rely on takedowns
    if slpm > 4.0 and td_avg < 2.0:
        return 'striker'
    
    return 'balanced'


def classify_style_from_dict(fighter_stats: dict) -> str:
    """
    Convenience wrapper - classify from a stats dictionary.
    Handles various key naming conventions.
    """
    slpm = fighter_stats.get('slpm') or fighter_stats.get('SLpM') or 0
    td_avg = fighter_stats.get('td_avg') or fighter_stats.get('TDAvg') or fighter_stats.get('tdavg') or 0
    sub_avg = fighter_stats.get('sub_avg') or fighter_stats.get('SubAvg') or fighter_stats.get('subavg') or 0
    str_def = fighter_stats.get('str_def') or fighter_stats.get('StrDef') or fighter_stats.get('strdef') or 0
    td_def = fighter_stats.get('td_def') or fighter_stats.get('TDDef') or fighter_stats.get('tddef') or 0
    
    return classify_style(slpm=slpm, td_avg=td_avg, sub_avg=sub_avg, str_def=str_def, td_def=td_def)


def get_style_matchup_advantage(style1: str, style2: str) -> int:
    """
    Get style matchup advantage.
    
    Returns:
         1 = fighter 1 has advantage
        -1 = fighter 2 has advantage
         0 = neutral
    
    Classic MMA matchup theory:
        - Wrestlers beat strikers (close distance, control)
        - Grapplers beat wrestlers (submit off back or scrambles)
        - Strikers beat grapplers (keep distance, avoid clinch)
        - Same style = neutral
        - Balanced = neutral (no clear exploit)
    """
    s1 = style1.lower() if style1 else 'balanced'
    s2 = style2.lower() if style2 else 'balanced'
    
    matchups = {
        ('wrestler', 'striker'):   1,
        ('striker', 'wrestler'):  -1,
        ('grappler', 'wrestler'):  1,
        ('wrestler', 'grappler'): -1,
        ('striker', 'grappler'):   1,
        ('grappler', 'striker'):  -1,
    }
    return matchups.get((s1, s2), 0)


# All valid styles
VALID_STYLES = ('striker', 'wrestler', 'grappler', 'balanced')

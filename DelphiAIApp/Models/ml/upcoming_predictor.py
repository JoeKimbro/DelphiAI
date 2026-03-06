"""
Upcoming Fight Predictor

Tracks upcoming UFC events and makes predictions.
Designed to be run before each event to:
1. Scrape current odds from BestFightOdds
2. Generate model predictions
3. Identify value bets
4. Track performance over time

This is how we VALIDATE the model with REAL data.
"""

import os
import sys
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

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

OUTPUT_DIR = Path(__file__).parent / 'predictions'
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}


class UpcomingPredictor:
    """Generate predictions for upcoming UFC fights."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.conn = None
    
    def connect_db(self):
        """Connect to database."""
        if not self.conn:
            self.conn = psycopg2.connect(**DB_CONFIG)
    
    def close_db(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def fetch_upcoming_events(self):
        """Fetch upcoming UFC events from BestFightOdds."""
        print("Fetching upcoming events from BestFightOdds...")
        
        try:
            response = self.session.get('https://www.bestfightodds.com/', timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching page: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        events = []
        
        # Find UFC event links
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            
            if '/events/' in href and 'ufc' in text.lower():
                if not href.startswith('http'):
                    href = 'https://www.bestfightodds.com' + href
                
                events.append({
                    'url': href,
                    'name': text,
                })
        
        # Remove duplicates
        seen = set()
        unique_events = []
        for e in events:
            if e['url'] not in seen:
                seen.add(e['url'])
                unique_events.append(e)
        
        print(f"Found {len(unique_events)} upcoming UFC events")
        return unique_events
    
    def fetch_event_odds(self, event_url):
        """Fetch odds for a specific event."""
        print(f"  Fetching odds: {event_url}")
        
        try:
            response = self.session.get(event_url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"  Error: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get event name
        title = soup.find('h1')
        event_name = title.get_text(strip=True) if title else "Unknown Event"
        
        fights = []
        current_fight = {}
        
        # Parse table rows
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                
                if not cells:
                    continue
                
                # Find fighter name
                name_link = row.find('a', href=lambda x: x and '/fighters/' in x)
                
                if name_link:
                    fighter_name = name_link.get_text(strip=True)
                    
                    # Extract odds from cells
                    odds_list = []
                    
                    for cell in cells[1:]:
                        text = cell.get_text(strip=True)
                        if text and text[0] in ['+', '-'] and text[1:].isdigit():
                            american = int(text)
                            if american > 0:
                                decimal = 1 + (american / 100)
                            else:
                                decimal = 1 + (100 / abs(american))
                            odds_list.append({
                                'american': american,
                                'decimal': round(decimal, 3),
                            })
                    
                    # Build fight
                    if 'fighter1' not in current_fight:
                        current_fight = {
                            'event_name': event_name,
                            'fighter1': fighter_name,
                            'fighter1_odds': odds_list,
                        }
                    else:
                        current_fight['fighter2'] = fighter_name
                        current_fight['fighter2_odds'] = odds_list
                        
                        # Get best odds
                        if current_fight['fighter1_odds']:
                            decimals = [o['decimal'] for o in current_fight['fighter1_odds']]
                            current_fight['fighter1_best_decimal'] = max(decimals)
                            current_fight['fighter1_implied'] = 1 / max(decimals)
                        else:
                            current_fight['fighter1_best_decimal'] = None
                            current_fight['fighter1_implied'] = None
                        
                        if current_fight['fighter2_odds']:
                            decimals = [o['decimal'] for o in current_fight['fighter2_odds']]
                            current_fight['fighter2_best_decimal'] = max(decimals)
                            current_fight['fighter2_implied'] = 1 / max(decimals)
                        else:
                            current_fight['fighter2_best_decimal'] = None
                            current_fight['fighter2_implied'] = None
                        
                        fights.append(current_fight)
                        current_fight = {}
        
        print(f"    Found {len(fights)} fights")
        return fights
    
    def get_fighter_elo(self, fighter_name):
        """Get current ELO for a fighter."""
        self.connect_db()
        
        cur = self.conn.cursor()
        
        # Try exact match first
        cur.execute("""
            SELECT cs.EloRating, fs.FighterID, fs.Name
            FROM CareerStats cs
            JOIN FighterStats fs ON cs.FighterID = fs.FighterID
            WHERE LOWER(fs.Name) = LOWER(%s)
        """, (fighter_name,))
        
        result = cur.fetchone()
        
        if result:
            return {
                'elo': float(result[0]) if result[0] else 1500,
                'fighter_id': result[1],
                'name': result[2],
            }
        
        # Try partial match
        cur.execute("""
            SELECT cs.EloRating, fs.FighterID, fs.Name
            FROM CareerStats cs
            JOIN FighterStats fs ON cs.FighterID = fs.FighterID
            WHERE LOWER(fs.Name) LIKE %s
            LIMIT 1
        """, (f'%{fighter_name.lower()}%',))
        
        result = cur.fetchone()
        
        if result:
            return {
                'elo': float(result[0]) if result[0] else 1500,
                'fighter_id': result[1],
                'name': result[2],
            }
        
        return {'elo': 1500.0, 'fighter_id': None, 'name': fighter_name}
    
    def generate_predictions(self, fights):
        """Generate predictions for each fight."""
        predictions = []
        
        for fight in fights:
            f1_data = self.get_fighter_elo(fight['fighter1'])
            f2_data = self.get_fighter_elo(fight['fighter2'])
            
            # ELO-based probability
            elo_diff = f1_data['elo'] - f2_data['elo']
            model_prob_f1 = 1 / (1 + 10**(-elo_diff/400))
            
            # Market implied probability
            implied_f1 = fight.get('fighter1_implied') or 0.5
            implied_f2 = fight.get('fighter2_implied') or 0.5
            
            # Calculate edge
            edge_f1 = model_prob_f1 - implied_f1 if implied_f1 else 0
            edge_f2 = (1 - model_prob_f1) - implied_f2 if implied_f2 else 0
            
            # Determine value bet
            value_bet = None
            if edge_f1 > 0.05:
                value_bet = {
                    'bet_on': fight['fighter1'],
                    'edge': edge_f1,
                    'decimal_odds': fight.get('fighter1_best_decimal'),
                    'model_prob': model_prob_f1,
                    'implied_prob': implied_f1,
                }
            elif edge_f2 > 0.05:
                value_bet = {
                    'bet_on': fight['fighter2'],
                    'edge': edge_f2,
                    'decimal_odds': fight.get('fighter2_best_decimal'),
                    'model_prob': 1 - model_prob_f1,
                    'implied_prob': implied_f2,
                }
            
            pred = {
                'event_name': fight['event_name'],
                'fighter1': fight['fighter1'],
                'fighter2': fight['fighter2'],
                'f1_elo': f1_data['elo'],
                'f2_elo': f2_data['elo'],
                'elo_diff': elo_diff,
                'model_prob_f1': round(model_prob_f1, 3),
                'model_prob_f2': round(1 - model_prob_f1, 3),
                'implied_prob_f1': round(implied_f1, 3) if implied_f1 else None,
                'implied_prob_f2': round(implied_f2, 3) if implied_f2 else None,
                'f1_decimal_odds': fight.get('fighter1_best_decimal'),
                'f2_decimal_odds': fight.get('fighter2_best_decimal'),
                'edge_f1': round(edge_f1, 3),
                'edge_f2': round(edge_f2, 3),
                'value_bet': value_bet,
                'prediction': fight['fighter1'] if model_prob_f1 > 0.5 else fight['fighter2'],
                'confidence': max(model_prob_f1, 1 - model_prob_f1),
                'timestamp': datetime.now().isoformat(),
            }
            
            predictions.append(pred)
        
        return predictions
    
    def display_predictions(self, predictions):
        """Display predictions in readable format."""
        print("\n" + "="*80)
        print("UPCOMING FIGHT PREDICTIONS")
        print("="*80)
        
        if not predictions:
            print("\nNo predictions generated")
            return
        
        event_name = predictions[0]['event_name'] if predictions else "Unknown"
        print(f"\nEvent: {event_name}")
        print("-"*80)
        
        value_bets = []
        
        for pred in predictions:
            print(f"\n{pred['fighter1']} ({pred['f1_elo']:.0f}) vs {pred['fighter2']} ({pred['f2_elo']:.0f})")
            print(f"   ELO Diff: {pred['elo_diff']:+.0f}")
            print(f"   Model: {pred['fighter1']} {pred['model_prob_f1']:.1%} | {pred['fighter2']} {pred['model_prob_f2']:.1%}")
            
            if pred['implied_prob_f1'] and pred['implied_prob_f2']:
                print(f"   Market: {pred['fighter1']} {pred['implied_prob_f1']:.1%} | {pred['fighter2']} {pred['implied_prob_f2']:.1%}")
                print(f"   Edge: {pred['fighter1']} {pred['edge_f1']*100:+.1f}% | {pred['fighter2']} {pred['edge_f2']*100:+.1f}%")
            
            print(f"   >>> Prediction: {pred['prediction']} ({pred['confidence']:.1%} confidence)")
            
            if pred['value_bet']:
                vb = pred['value_bet']
                if vb['decimal_odds']:
                    print(f"   >>> VALUE BET: {vb['bet_on']} @ {vb['decimal_odds']:.2f} ({vb['edge']*100:.1f}% edge)")
                    value_bets.append(pred)
                else:
                    print(f"   >>> POTENTIAL VALUE: {vb['bet_on']} ({vb['edge']*100:.1f}% edge) - odds unavailable")
        
        # Summary of value bets
        if value_bets:
            print("\n" + "="*80)
            print("VALUE BETS SUMMARY")
            print("="*80)
            
            for pred in value_bets:
                vb = pred['value_bet']
                print(f"\n   {vb['bet_on']}")
                if vb['decimal_odds']:
                    print(f"   Odds: {vb['decimal_odds']:.2f}")
                print(f"   Model prob: {vb['model_prob']:.1%}")
                if vb['implied_prob']:
                    print(f"   Implied prob: {vb['implied_prob']:.1%}")
                print(f"   Edge: {vb['edge']*100:.1f}%")
    
    def save_predictions(self, predictions, event_name):
        """Save predictions to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = "".join(c if c.isalnum() else "_" for c in event_name)
        filename = f"{safe_name}_{timestamp}.json"
        
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"\nPredictions saved to: {filepath}")
    
    def run(self, event_url=None):
        """Run the prediction pipeline."""
        print("="*80)
        print("DELPHI AI - UPCOMING FIGHT PREDICTOR")
        print("="*80)
        
        if event_url:
            # Single event
            fights = self.fetch_event_odds(event_url)
            if fights:
                predictions = self.generate_predictions(fights)
                self.display_predictions(predictions)
                
                if predictions:
                    self.save_predictions(predictions, predictions[0]['event_name'])
        else:
            # All upcoming events
            events = self.fetch_upcoming_events()
            
            for event in events[:3]:  # Limit to next 3 events
                print(f"\n{'='*80}")
                fights = self.fetch_event_odds(event['url'])
                
                if fights:
                    predictions = self.generate_predictions(fights)
                    self.display_predictions(predictions)
                    
                    if predictions:
                        self.save_predictions(predictions, predictions[0]['event_name'])
        
        self.close_db()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions for upcoming UFC fights')
    parser.add_argument('--event', help='URL of specific event to predict')
    
    args = parser.parse_args()
    
    predictor = UpcomingPredictor()
    predictor.run(event_url=args.event)


if __name__ == '__main__':
    main()

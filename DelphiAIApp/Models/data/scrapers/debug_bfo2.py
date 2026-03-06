# -*- coding: utf-8 -*-
"""Debug BestFightOdds - find historical events."""

import requests
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# Check different sections
urls_to_check = [
    'https://www.bestfightodds.com/',
    'https://www.bestfightodds.com/archive',
    'https://www.bestfightodds.com/past-events',
    'https://www.bestfightodds.com/events',
]

for url in urls_to_check:
    print(f"\n{'='*60}")
    print(f"Checking: {url}")
    print('='*60)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"  Status: {response.status_code}")
            continue
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find UFC event links
        ufc_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            if '/events/' in href and 'ufc' in text.lower():
                ufc_links.append((href, text[:40]))
        
        print(f"  UFC events found: {len(ufc_links)}")
        for href, text in ufc_links[:10]:
            print(f"    {href} -> {text}")
        if len(ufc_links) > 10:
            print(f"    ... and {len(ufc_links) - 10} more")
    except Exception as e:
        print(f"  Error: {e}")

# Also check for a specific past UFC event
print(f"\n{'='*60}")
print("Trying to access a known past UFC event...")
print('='*60)

# Try UFC 309 (a known past event)
test_urls = [
    'https://www.bestfightodds.com/events/ufc-309-jones-vs-miocic',
    'https://www.bestfightodds.com/events/ufc-309',
    'https://www.bestfightodds.com/events/ufc-300',
]

for url in test_urls:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        print(f"\n  {url}")
        print(f"    Status: {response.status_code}")
        print(f"    Final URL: {response.url}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('h1')
            if title:
                print(f"    Title: {title.get_text(strip=True)}")
            
            # Find fighter names
            fighters = []
            for a in soup.find_all('a', href=True):
                if '/fighters/' in a['href']:
                    fighters.append(a.get_text(strip=True))
            print(f"    Fighters found: {len(fighters)}")
            if fighters:
                print(f"    First few: {fighters[:5]}")
    except Exception as e:
        print(f"    Error: {e}")

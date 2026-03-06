"""Debug BestFightOdds page structure."""

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# Check archive page
print("Checking archive page...")
response = requests.get('https://www.bestfightodds.com/archive', headers=HEADERS)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links
links = []
for a in soup.find_all('a', href=True):
    href = a['href']
    text = a.get_text(strip=True)
    if '/events/' in href and 'ufc' in href.lower():
        links.append((href, text))

print(f"\nFound {len(links)} UFC links:")
for href, text in links[:20]:
    print(f"  {href} -> {text}")

# Check the events page structure
print("\n\nChecking main page for upcoming/past events...")
response = requests.get('https://www.bestfightodds.com/', headers=HEADERS)
soup = BeautifulSoup(response.text, 'html.parser')

for a in soup.find_all('a', href=True):
    href = a['href']
    text = a.get_text(strip=True)[:50]
    if 'ufc' in href.lower() and ('/events/' in href or 'past' in href.lower()):
        print(f"  {href} -> {text}")

# Try to find a specific event and look at its structure
print("\n\nChecking a specific event page...")
# Get one of the event URLs we found
if links:
    event_url = links[0][0]
    if not event_url.startswith('http'):
        event_url = 'https://www.bestfightodds.com' + event_url
    
    print(f"URL: {event_url}")
    response = requests.get(event_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find tables
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables[:2]):
        print(f"\n  Table {i}:")
        rows = table.find_all('tr')[:5]
        for row in rows:
            cells = row.find_all(['td', 'th'])
            cell_texts = [c.get_text(strip=True)[:20] for c in cells[:5]]
            print(f"    {cell_texts}")

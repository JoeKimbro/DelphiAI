"""
Add Nationality Column to FighterStats

Extracts country from PlaceOfBirth and adds it as a new column.
This enables nationality-based edge analysis.
"""

import os
import sys
import re
import psycopg2
from pathlib import Path
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

# Country extraction patterns (same as analyze_nationality.py)
COUNTRY_PATTERNS = {
    'USA': ['United States', 'USA', 'U.S.A', 'America', 'American'],
    'Brazil': ['Brazil', 'Brasil'],
    'Russia': ['Russia', 'Russian Federation'],
    'Dagestan': ['Dagestan'],
    'UK': ['England', 'United Kingdom', 'UK', 'Great Britain', 'Scotland', 'Wales', 'Ireland', 'Northern Ireland'],
    'Canada': ['Canada'],
    'Mexico': ['Mexico'],
    'Australia': ['Australia'],
    'Japan': ['Japan'],
    'South Korea': ['South Korea', 'Korea'],
    'China': ['China'],
    'Poland': ['Poland'],
    'Netherlands': ['Netherlands', 'Holland'],
    'Sweden': ['Sweden'],
    'Germany': ['Germany'],
    'France': ['France'],
    'Italy': ['Italy'],
    'Spain': ['Spain'],
    'Argentina': ['Argentina'],
    'Peru': ['Peru'],
    'Chile': ['Chile'],
    'Colombia': ['Colombia'],
    'Ecuador': ['Ecuador'],
    'Venezuela': ['Venezuela'],
    'Cuba': ['Cuba'],
    'Jamaica': ['Jamaica'],
    'New Zealand': ['New Zealand'],
    'Thailand': ['Thailand'],
    'Philippines': ['Philippines'],
    'Indonesia': ['Indonesia'],
    'Kazakhstan': ['Kazakhstan'],
    'Uzbekistan': ['Uzbekistan'],
    'Azerbaijan': ['Azerbaijan'],
    'Georgia': ['Georgia'],
    'Ukraine': ['Ukraine'],
    'Belarus': ['Belarus'],
    'Czech Republic': ['Czech Republic', 'Czechia'],
    'Croatia': ['Croatia'],
    'Serbia': ['Serbia'],
    'Nigeria': ['Nigeria'],
    'Cameroon': ['Cameroon'],
    'South Africa': ['South Africa'],
    'Egypt': ['Egypt'],
    'Morocco': ['Morocco'],
    'Afghanistan': ['Afghanistan'],
    'Iran': ['Iran'],
    'Iraq': ['Iraq'],
    'Kyrgyzstan': ['Kyrgyzstan'],
    'Tajikistan': ['Tajikistan'],
    'Armenia': ['Armenia'],
    'Moldova': ['Moldova'],
}

US_STATES = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
    'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia', 'D.C.'
]

CITY_MAPPINGS = {
    'Moscow': 'Russia', 'St. Petersburg': 'Russia', 'Sochi': 'Russia',
    'Sao Paulo': 'Brazil', 'Rio de Janeiro': 'Brazil', 'Rio': 'Brazil',
    'London': 'UK', 'Manchester': 'UK', 'Liverpool': 'UK', 'Birmingham': 'UK',
    'Paris': 'France', 'Lyon': 'France', 'Marseille': 'France',
    'Berlin': 'Germany', 'Munich': 'Germany', 'Hamburg': 'Germany',
    'Tokyo': 'Japan', 'Osaka': 'Japan',
    'Seoul': 'South Korea', 'Busan': 'South Korea',
    'Beijing': 'China', 'Shanghai': 'China',
    'Sydney': 'Australia', 'Melbourne': 'Australia',
    'Toronto': 'Canada', 'Montreal': 'Canada', 'Vancouver': 'Canada',
    'Mexico City': 'Mexico', 'Guadalajara': 'Mexico', 'Tijuana': 'Mexico',
    'Warsaw': 'Poland', 'Krakow': 'Poland',
    'Amsterdam': 'Netherlands', 'Rotterdam': 'Netherlands',
    'Stockholm': 'Sweden', 'Gothenburg': 'Sweden',
    'Lagos': 'Nigeria', 'Abuja': 'Nigeria',
    'Makhachkala': 'Dagestan', 'Khasavyurt': 'Dagestan',
    'Grozny': 'Russia',
    'Tbilisi': 'Georgia',
    'Baku': 'Azerbaijan',
    'Yerevan': 'Armenia',
    'Almaty': 'Kazakhstan', 'Nur-Sultan': 'Kazakhstan',
    'Tashkent': 'Uzbekistan',
    'Kiev': 'Ukraine', 'Kyiv': 'Ukraine',
    'Minsk': 'Belarus',
    'Prague': 'Czech Republic',
    'Zagreb': 'Croatia',
    'Belgrade': 'Serbia',
    'Buenos Aires': 'Argentina',
    'Lima': 'Peru',
    'Santiago': 'Chile',
    'Bogota': 'Colombia',
    'Havana': 'Cuba',
    'Bangkok': 'Thailand',
    'Manila': 'Philippines',
    'Jakarta': 'Indonesia',
    'Tehran': 'Iran',
    'Kabul': 'Afghanistan',
}


def extract_country(place_of_birth):
    """Extract country from place of birth string."""
    if not place_of_birth:
        return 'Unknown'
    
    pob = str(place_of_birth).strip()
    
    # Check for US states first
    for state in US_STATES:
        if state.lower() in pob.lower():
            return 'USA'
    
    # Check for country patterns
    for country, patterns in COUNTRY_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in pob.lower():
                if country == 'Georgia' and any(s.lower() in pob.lower() for s in ['Atlanta', 'Savannah', 'Augusta']):
                    return 'USA'
                return country
    
    # Check city mappings
    for city, country in CITY_MAPPINGS.items():
        if city.lower() in pob.lower():
            return country
    
    return 'Unknown'


def add_nationality_column():
    """Add Nationality column to FighterStats table."""
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print("Adding Nationality column to FighterStats...")
    
    # Check if column exists
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'fighterstats' AND column_name = 'nationality'
    """)
    
    if not cur.fetchone():
        cur.execute("ALTER TABLE FighterStats ADD COLUMN Nationality VARCHAR(50)")
        conn.commit()
        print("  Column added")
    else:
        print("  Column already exists")
    
    # Get all fighters with place of birth
    cur.execute("SELECT FighterID, PlaceOfBirth FROM FighterStats")
    fighters = cur.fetchall()
    
    print(f"Processing {len(fighters)} fighters...")
    
    # Update nationalities
    updated = 0
    for fighter_id, pob in fighters:
        country = extract_country(pob)
        cur.execute(
            "UPDATE FighterStats SET Nationality = %s WHERE FighterID = %s",
            (country, fighter_id)
        )
        updated += 1
    
    conn.commit()
    print(f"  Updated {updated} fighters")
    
    # Show distribution
    cur.execute("""
        SELECT Nationality, COUNT(*) as cnt 
        FROM FighterStats 
        GROUP BY Nationality 
        ORDER BY cnt DESC 
        LIMIT 20
    """)
    
    print("\nNationality Distribution:")
    for row in cur.fetchall():
        print(f"   {row[0]}: {row[1]}")
    
    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    add_nationality_column()

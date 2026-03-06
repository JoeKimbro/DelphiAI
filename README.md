# DelphiAI
This is a Sports Betting stat analyzer that can help you predict the best course of action for your money!

Python
Pandas / NumPy
Scikit-learn
XGBoost or LightGBM
PostgreSQL
FastAPI
LLM (for explanations, not predictions)
Optional: FAISS for RAG

PostgreSQL → Pandas → XGBoost → Probabilities
                                ↓
                        Math (value detection)
                                ↓
                          LLM Explanation
                                ↓
                            FastAPI
                                ↓
                              User


┌──────── VIEW ────────┐
│  Dashboard / UI      │
│  Charts / Text       │
└────────▲─────────────┘
         │ JSON
┌────────┴─────────────┐
│     CONTROLLER       │
│  FastAPI Routes      │
│  Validation          │
│  Orchestration       │
└────────▲─────────────┘
         │ Calls
┌────────┴─────────────┐
│        MODEL         │
│  PostgreSQL          │
│  Pandas / NumPy      │
│  XGBoost / SKLearn   │
│  FAISS (optional)    │
└──────────────────────┘
## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the database
```bash
docker-compose up -d
```

### 3. Access pgAdmin
- URL: http://localhost:5050
- Email: `admin@delphi.local`
- Password: `admin123`

### 4. Connect to PostgreSQL in pgAdmin
- Host: `postgres`
- Port: `5432`
- Database: `delphi_db`
- Username: `delphi_user`
- Password: `delphi_password`

## Docker Commands

| Command | Description |
|---------|-------------|
| `docker-compose up -d` | Start all containers |
| `docker-compose down` | Stop all containers |
| `docker-compose restart` | Restart all containers |
| `docker-compose logs -f` | View live logs |
| `docker-compose logs postgres` | View database logs |
| `docker-compose ps` | List running containers |

## Database Commands

| Command | Description |
|---------|-------------|
| `docker-compose down -v` | Stop containers and delete all data |
| `docker-compose exec postgres psql -U delphi_user -d delphi_db` | Open PostgreSQL shell |

## Troubleshooting

**Reset the database (delete all data):**
```bash
docker-compose down -v
docker-compose up -d
```

**Check if containers are running:**
```bash
docker-compose ps
```

**View container logs:**
```bash
docker-compose logs -f
```
DelphiAIApp/Models/data/
├── scrapers/
│   ├── __init__.py
│   ├── scrapy.cfg
│   └── ufc_scraper/
│       ├── __init__.py
│       ├── items.py          # FighterItem, CareerStatsItem, FightItem
│       ├── middlewares.py    # Request handling
│       ├── pipelines.py      # CSV export to separate files
│       ├── settings.py       # Rate limiting, caching, user agent
│       └── spiders/
│           ├── __init__.py
│           ├── ufcstats.py   # UFCStats.com spider (working)
│           └── tapology.py   # Tapology.com spider
└── output/                   # Scraped data files
    ├── fighters.csv
    ├── career_stats.csv
    └── fights.csv


scrapy crawl ufcstats in terminal to scrap
# Full scrape (all ~4000+ fighters) - will take several hours with rate limiting
scrapy crawl ufcstats

# Or with logging to file for monitoring
scrapy crawl ufcstats --logfile=../output/scrape.log

Scraper  →  CSV files  →  (manual step needed)  →  PostgreSQL

python scrape_all.py              # Run both spiders
python scrape_all.py --ufc-only   # Only UFC.com
python scrape_all.py --stats-only # Only UFCStats
python scrape_all.py --test       # Test mode (limited pages)

1. UFC.com Spider → Scrapes stats from athlete profiles
2. UFCStats Spider → Scrapes detailed fight history
3. DataMergePipeline → Merges UFC.com data into UFCStats data
4. FightStatsCalculationPipeline → Calculates precise stats from fight history
5. CSV Export → Saves to career_stats.csv
6. FightStatsCalculationPipeline → Updates career_stats.csv with calculated values
7. load_to_db.py → Loads everything into PostgreSQL

## Data Scraping Notes

### Unified Spider (ufc_official)
The scraper uses a unified approach:
1. Scrapes fighter profile from UFC.com (primary source for bio data)
2. Looks up the same fighter on UFCStats (by first/last name)
3. Merges data from both sources, prioritizing UFC.com for bio fields

### Fuzzy Name Matching
The scraper uses fuzzy matching to handle minor spelling variations between sources:
- **Works for**: Abbassov/Abbasov, O'Brien/OBrien, José/Jose
- **Does NOT work for**: Nickname vs real name (e.g., "Tank Abbott" vs "David Abbott")

### Known Limitations
| Fighter | Issue | Result |
|---------|-------|--------|
| Tank Abbott | UFC.com uses "Tank" (nickname), UFCStats uses real name | No UFCStats data merged |
| Fighters with very different name spellings | Sources may have different romanizations | May not merge |

Fighters that can't be matched on UFCStats will only have UFC.com data (basic bio info, no detailed career stats like SLpM, StrAcc, etc.).

### Calculated Stats (from Fight History)
These stats are calculated from the fighter's fight history, not scraped directly:
- `win_streak_last3` - Consecutive wins in last 3 fights
- `wins_by_ko_last5` - KO/TKO wins in last 5 fights
- `wins_by_sub_last5` - SUB wins in last 5 fights
- `avg_fight_duration` - Average fight length in minutes
- `first_round_finish_rate` - % of fights finished in Round 1
- `decision_rate` - % of fights that went to decision
- `ko_round1_pct`, `ko_round2_pct`, `ko_round3_pct` - Distribution of KO wins by round
- `sub_round1_pct`, `sub_round2_pct`, `sub_round3_pct` - Distribution of SUB wins by round

*ISSUE THAT WILL NEED FIX LATER
Leg reach, not a lot of data, but could maybe use predictive model to assume leg reach basied off height

## ELO Rating System

The project includes a FiveThirtyEight-style ELO rating system with enhancements for UFC fighters.

### How to Run ELO Calculation
```bash
cd DelphiAIApp/Models/data
python features.py               # Enhanced ELO (recommended)
python features.py --base-only   # Base FiveThirtyEight ELO only
```

### Phase 1: Base FiveThirtyEight System
- Starting ELO: 1500
- K-factor: Base 32, adjusted by fight type and finish method
- Mean reversion: Ratings drift toward 1500 for inactive fighters
- Margin of victory: KO/TKO (1.3x), SUB (1.25x), DEC (1.0x)
- Round bonus: Earlier finishes get higher K-factor multipliers

### Phase 2: Enhancements
| Modifier | Description |
|----------|-------------|
| Style Matchup | Wrestlers get +5% vs strikers, grapplers vs wrestlers, etc. |
| Reach Differential | +0.5% per inch of reach advantage |
| Age Factor | 27-32 = peak years, decline penalty outside this range |
| Recent Form | Last 3 fights affect expected outcome (momentum) |

### Phase 3: ML Feature
ELO is stored in `career_stats.csv` and loaded to `CareerStats.EloRating` in PostgreSQL:
- `elo_rating` - Current strength estimate
- `peak_elo` - Highest ELO ever achieved (useful for decline detection)

### ELO Interpretation
| ELO Range | Meaning |
|-----------|---------|
| 1700+ | Elite/Championship level |
| 1600-1700 | Top contender |
| 1500-1600 | Above average |
| 1400-1500 | Average |
| Below 1400 | Below average/gatekeeper |

### Top Fighters by Peak ELO (Historical)
1. Daniel Cormier: 1814 (current: 1681)
2. Donald Cerrone: 1801 (current: 1482)
3. Tony Ferguson: 1775 (current: 1369)
4. Jon Fitch: 1768 (current: 1649)
5. TJ Dillashaw: 1768 (current: 1587)
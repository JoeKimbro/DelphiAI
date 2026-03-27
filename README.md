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

1. SCRAPE DATA
   scrapy crawl ufc_official

2. CALCULATE ELO + EXPORT NEW TABLES
   python features.py                    # ELO history + pre-UFC career

3. CALCULATE ML FEATURES  
   python calculate_opponent_quality.py  # Strength of schedule
   python calculate_matchup_features.py  # Pre-cached differentials

4. VALIDATE
   python validate_data.py

5. LOAD TO DATABASE
   python load_to_db.py                  # All tables
   python load_to_db.py --core-only      # Just fighters/fights/career
   python load_to_db.py --ml-only        # Just ML feature tables


SUMMARY:
DON'T wait for real fights to test!
✅ Do this NOW:

Calculate baseline accuracy on historical fights (2023-2024)
Train ML model on older fights (pre-2023)
Test on same 2023-2024 fights
Compare: Did ML beat baseline?

✅ Deploy to production ONLY IF:

ML consistently beats baseline by 3-5%+
Walk-forward validation confirms performance
Model is properly calibrated

✅ Then track real predictions:

Make predictions for upcoming events
Store in Matchups table
Update after fights occur
Monitor ongoing performance

MONITORING CHECKLIST FOR PRODUCTION:
Track these metrics for every event:
Pre-Event:

 Predictions generated and stored
 Betting lines recorded
 Bets identified (Kelly criterion)

Post-Event:

 Results updated in database
 Accuracy calculated
 ROI calculated
 Brier score calculated
 Feature drift checked

Monthly:

 Retrain model with new data
 Compare v2 vs v3 performance
 Update if new version better


RED FLAGS TO WATCH FOR:
🚨 Stop betting if you see:

Accuracy drops below 60% for 20+ consecutive bets
Losing streak of 10+ bets
ROI goes negative for 50+ bets
Brier score increases significantly (>0.25)
Distribution shift detected (features changing)

I WANT TO ADD PROP BETS MF LETS DO IT! (Later tho lets test it with paper and PAPA)

python -m ml.predict_fight "Islam Makhachev" "Charles Oliveira"
python -m ml.predict_fight "Jon Jones" "Stipe Miocic"
python -m ml.predict_fight "Alex Pereira" "Magomed Ankalaev"

TESTING 
python c:\Users\Silly\Desktop\DelphiAI\DelphiAI\DelphiAIApp\Models\ml\predict_fight.py "fighter1" "fighter2"






BETTING
🎯 The Simple Version
Your Fight:
Mario Bautista vs Vinicius Oliveira

Your model says:
Mario:    50.2% chance to win
Vinicius: 49.8% chance to win

Vegas odds:
Mario:    -175 
Vinicius: +150

Step 1: What do Vegas odds mean?
Mario -175 means:

Vegas thinks Mario has a 63.6% chance to win
You need to bet $175 to win $100

Vinicius +150 means:

Vegas thinks Vinicius has a 40% chance to win
If you bet $100, you win $150


Step 2: Compare to your model
Mario bet:
Your model:  50.2% chance
Vegas says:  63.6% chance

Vegas thinks Mario is MORE likely to win than you do
❌ DON'T BET MARIO
Vinicius bet:
Your model:  49.8% chance
Vegas says:  40.0% chance

You think Vinicius is MORE likely to win than Vegas does
✅ THIS IS A "VALUE BET"

Step 3: How much edge do you have?
Your model:  49.8%
Vegas:       40.0%
Difference:   9.8%

After accounting for fees (vig):
Real edge: About 6-8%
This is good! Most bettors look for 5%+ edge.

Example of a GOOD bet:
Fighter A: Your model says 65%, Vegas says 55%
→ You're confident AND you have edge ✅ BET IT

Fighter B: Your model says 50.2%, Vegas says 40%  
→ You have edge BUT you're not confident ⚠️ PASS

📝 Simple Rule:
Only bet when BOTH are true:
1. Your model disagrees with Vegas by 5%+  ✅
2. Your model is confident (at least 55%+) ⚠️ (yours is only 50.2%)

Your situation:
1. Edge: 8% ✅
2. Confidence: 50.2% ❌

Result: PASS

🎯 Bottom Line:
Math says: Bet Vinicius +150 (you have 6-8% edge)
Reality says: Pass (your model basically says "I don't know who wins")
Wait for a fight where you're more sure!

🎯 Updated Betting Strategy:
Your Personal Betting Rules:
IF model confidence ≥ 70%:
→ BET 2-3% of bankroll
→ Historical edge: 7.1%
→ Strong signal

IF model confidence 60-69%:
→ BET 1-2% of bankroll
→ Historical edge: 2.8%
→ Good signal

IF model confidence 55-59%:
→ BET 0.5-1% of bankroll
→ Historical edge: 3.1%
→ Moderate signal

IF model confidence 50-54%:
→ PASS (or 0.25% if underdog)
→ Historical edge: 0.1%
→ Weak signal

IF model confidence < 50%:
→ PASS always
→ Bet the other side if confident enough

📊 Albazi Specific Recommendation:
Based on Your Historical Performance:
Model: Horiguchi 58% (Albazi 42%)
Bracket: 55-60% confidence
Historical accuracy in this bracket: 55.5%

Betting rules:
55-60% confidence → Bet 0.5-1%

But wait - you're betting the UNDERDOG (Albazi):

Albazi at +300:
Your model (calibrated): 44.5%
Vegas: 25%
Edge: 19.5%

Break-even needed: 25%
Your expected: 44.5%
Margin: +19.5%

Recommendation:
✅ BET 1-1.5% (increase from normal 55-60% tier)

Why larger:
- Underdog odds give cushion
- 19.5% edge is huge even after calibration
- At +300, you can miss more and still profit

💰 Expected Value Calculation:
For Albazi +300 Bet:
Bet: $100
Odds: +300
Calibrated win probability: 44.5%

Expected outcome:
Win: 44.5% × $300 = $133.50
Loss: 55.5% × -$100 = -$55.50
Net EV: +$78 per $100 bet

ROI: 78% 🔥

Even with calibration, this is excellent!


✅ Bottom Line:
Your Model Quality: B+ (Good, Not Elite)
✅ 53.9% overall accuracy (beats random)
✅ 59.5% on high-confidence (very good)
✅ Well-calibrated across confidence levels
✅ Tested on 10,000+ fights (robust)

⚠️ Only 1.9% edge on average (thin)
⚠️ Slight overconfidence at all levels
⚠️ Famous upsets still hard to predict
Albazi Bet: STRONG YES ✅
Confidence: 58% (falls in 55-60% bracket)
Historical accuracy: 55.5%
Calibrated prob: 44.5%
Vegas: 25%
Edge: 19.5%

Bet size: 1-1.5% of bankroll
Expected ROI: 78%

This is one of your best edges!
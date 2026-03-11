# DelphiAI - UFC Fight Prediction System

A machine learning system for predicting UFC fight outcomes with **66.1% accuracy** on out-of-sample data (2024-2025).

---

## 🎯 Key Performance Metrics

- **Overall Accuracy**: 66.1% (436 fights, 2024-2025)
- **High Confidence Picks** (≥65%): 79.7% accuracy (74 picks)
- **Paper Trading ROI**: +14.5%
- **Validated**: Leakage audit passed, no future data contamination

### Performance by Year
- **2024**: 64.0% (211 fights)
- **2025**: 68.0% (225 fights)
- **Live 2026**: 62.5% (40 fights tracked)

---

## 🏗️ System Architecture

### Core Components

**Machine Learning Pipeline**
- XGBoost classifier with isotonic calibration
- Point-in-time feature engineering (no data leakage)
- Graduated ML/ELO blending based on data quality
- Historical rolling features (opponent quality, momentum, trends)

**ELO Rating System**
- Custom UFC ELO ratings (K-factor: 32)
- Division-specific adjustments
- Method-of-victory bonuses (KO/TKO, submission)
- Tracks all UFC fights from 2019+

**Data Sources**
- UFC Stats API
- Fighter career statistics
- Historical fight outcomes
- Point-in-time injury/layoff tracking

---

## 📊 Feature Engineering

### Historical Context Features (Key Innovation)
- **Opponent Quality**: Rolling average of opponent ELO (last 3, 5 fights)
- **Performance Velocity**: ELO change trajectory and momentum
- **Trending Stats**: Recent finish rate vs career average
- **Competition Level**: Opponent quality trending (harder/easier opponents)

### Fighter Attributes
- Striking metrics (SLpM, SS%, significant strikes)
- Grappling metrics (TD avg, TD%, submission attempts)
- Defensive metrics (SApM, TD defense)
- Physical attributes (reach, height, stance)
- Career metrics (win streaks, finish rates, experience)

### Matchup Features
- Style contrasts (striker vs grappler)
- Physical differentials (reach, height advantage)
- Experience gaps (UFC fights, age)
- Form differentials (recent performance)

### Contextual Factors
- Injury penalties (layoff length, frequency)
- Rust penalties (time since last fight)
- Age curves (prime vs declining)
- Division-specific adjustments

---

## 🎓 Model Training

### Data Strategy
- **Training**: 2020-2023 fights (~2,200 fights)
- **Validation**: 2024 fights (~210 fights)
- **Holdout**: 2025 fights (~225 fights)
- **Chronological splits**: No temporal leakage
- **Recency weighting**: Recent years weighted higher

### Model Configuration
- **Algorithm**: XGBoost (gradient boosting)
- **Calibration**: Isotonic regression
- **Probability range**: 5-95% (prevents overconfidence)
- **Cross-validation**: Temporal walk-forward

### Quality Assurance
- Fight-level deduplication (no augmentation leakage)
- Point-in-time feature validation
- Leakage audit on random sample
- Symmetry checks (prediction consistency)

---

## 🚀 Usage

### Predict a Fight Card

```bash
python -m ml.predict_card "UFC 313: Pereira vs Hill"
```

**Output**:
- Fight-by-fight predictions with confidence levels
- Win probabilities for each fighter
- Method of victory breakdown (KO/TKO, Submission, Decision)
- Round prediction distribution
- Betting insights and value picks

### Update Results After Event

```bash
python -m ml.update_results "UFC 313: Pereira vs Hill"
```

Automatically scrapes results and tracks prediction accuracy.

### View Performance Summary

```bash
python -m ml.performance_summary
```

**Displays**:
- Overall accuracy (backtest vs live)
- Performance by confidence tier
- ROI metrics
- Accuracy trends over time

### Run Historical Backtest

```bash
python -m ml.backtest --years 2024 2025 --clear
```

Validates model on historical data using point-in-time evaluation.

---

## 📁 Project Structure

```
DelphiAI/
├── Models/
│   ├── ml/
│   │   ├── train_model_v3.py       # Model training pipeline
│   │   ├── predict_card.py         # Event predictions
│   │   ├── predict_fight.py        # Single fight predictions
│   │   ├── backtest.py             # Historical validation
│   │   ├── update_results.py       # Result tracking
│   │   ├── performance_summary.py  # Analytics
│   │   ├── model_loader.py         # Feature engineering
│   │   └── model_latest.pkl        # Trained model artifact
│   ├── data/
│   │   ├── build_historical_features.py  # Feature builder
│   │   ├── load_to_db.py                 # Data ingestion
│   │   └── calculate_elo.py              # ELO rating system
│   └── db/
│       ├── schemas.sql                    # Database schema
│       └── database.db                    # SQLite database
└── README.md
```

---

## 🗄️ Database Schema

### Key Tables

**Fights**
- All UFC fights (2019+)
- Fighter matchups, results, methods
- ELO ratings before/after each fight

**FighterStats**
- Career statistics per fighter
- Physical attributes
- Current ELO rating

**FighterHistoricalFeatures**
- Point-in-time rolling features
- Opponent quality metrics
- Performance velocity/momentum
- Computed chronologically (no leakage)

**PredictionTracking**
- All predictions (backtest + live)
- Actual outcomes
- Confidence levels
- Used for performance analysis

**OddsLines** (optional)
- Vegas betting lines
- Closing line value tracking
- Edge calculation vs market

---

## 📈 Betting Strategy (Paper Trading Validated)

### Tier-Based Approach

**High Confidence (≥65%)**
- Accuracy: 79.7%
- Bet size: 1.5-2% of bankroll
- Expected ROI: +15-20%

**Medium Confidence (60-65%)**
- Accuracy: ~67%
- Bet size: 1-1.5% of bankroll
- Expected ROI: +10-15%

**Low Confidence (55-60%)**
- Accuracy: ~64%
- Bet size: 0.5-1% of bankroll
- Expected ROI: +5-10%

**Toss-ups (<55%)**
- Skip or minimal action
- High variance, low edge

### Underdog Value Strategy
- Model picks underdog: 69.9% win rate (historical)
- Target underdogs with 60%+ model confidence
- Higher expected value than favorites

### Risk Management
- Never exceed 2% on single fight
- Maximum 10% total exposure per event
- Stop-loss: 20% drawdown or 8+ losing streak

---

## 🔬 Model Validation

### Leakage Prevention
- Chronological data splits (no temporal leakage)
- Point-in-time features only (no future data)
- Historical features computed sequentially
- Audit verification: 0 mismatches on 96 fighter-fight pairs

### Evaluation Metrics
- **Out-of-sample accuracy**: 66.1% (2024-2025)
- **Production-path validation**: Same pipeline for training and deployment
- **Fight-level metrics**: One prediction per fight (no row-level inflation)
- **Overlap checks**: 0/0/0 (train/val/holdout isolation)

### Known Limitations
- Model conservative (relies 50% on ELO when ML data quality low)
- Age features incomplete (reduces ML weight by ~10-15%)
- Performance degrades on fighters with <3 UFC fights
- Late-replacement fights (short notice) less reliable

---

## 🛠️ Setup & Installation

### Requirements
- Python 3.8+
- SQLite 3
- XGBoost
- scikit-learn
- pandas, numpy

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DelphiAI.git
cd DelphiAI

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m data.apply_schema

# Build historical features
python -m data.build_historical_features

# Optional: Calculate ELO ratings
python -m data.calculate_elo
```

### Data Updates

```bash
# Weekly maintenance (after UFC events)
python -m data.update_fighters          # Scrape new fighter data
python -m data.calculate_elo            # Update ELO ratings
python -m data.build_historical_features # Rebuild rolling features
```

---

## 📊 Performance Tracking

### Live Tracking (2026)
Track predictions vs actual outcomes in real-time:

| Event | Date | Accuracy | HC Accuracy | Sample |
|-------|------|----------|-------------|--------|
| Strickland vs Imavov | Jan 11 | 57.1% | 50.0% | 14 fights |
| Moreno vs Albazi | Feb 1 | 76.9% | 100% | 13 fights |
| Bautista vs Figueiredo | Feb 22 | 53.8% | - | 13 fights |
| **Cumulative** | **-** | **62.5%** | **66.7%** | **40 fights** |

*Live tracking showing model generalizes well to 2026 data*

---

## 🎯 Roadmap

### Immediate Priorities
- [x] Fix historical feature loading in live predictions
- [x] Validate leakage prevention (audit passed)
- [ ] Backfill age/DOB data for all fighters
- [ ] Fix age-related features (currently missing)

### Short-term (Q1 2026)
- [ ] Improve ML weight to 40%+ (currently 26-30%)
- [ ] Add division-specific model variants
- [ ] Implement automated odds scraping
- [ ] Build live performance dashboard

### Medium-term (Q2-Q3 2026)
- [ ] Temporal ensemble (multiple model versions)
- [ ] Enhanced prospect detection
- [ ] Style-matchup neural network
- [ ] Automated result tracking

### Long-term
- [ ] Expand to other MMA organizations (Bellator, ONE)
- [ ] Real-time odds monitoring & alerts
- [ ] Multi-sportsbook integration
- [ ] Advanced visualization dashboard

---

## 🤝 Contributing

This is a personal research project. If you find issues or have suggestions:

1. Open an issue describing the problem/enhancement
2. Include relevant data/examples
3. For bugs: Steps to reproduce

**Note**: This project is for educational and research purposes. Gambling involves risk.

---

## ⚠️ Disclaimer

**For Educational Purposes Only**

This model is a research project and should not be used as the sole basis for gambling decisions. Sports betting involves risk, and past performance does not guarantee future results.

- No guarantees of accuracy or profitability
- Individual event variance is high (50-80% per card)
- Long-run edge requires large sample sizes (100+ fights)
- Responsible gambling practices strongly encouraged
- Not financial or betting advice

**The developers are not responsible for any financial losses incurred from using this system.**

---

## 📝 License

MIT License - See LICENSE file for details

---

## 📧 Contact

For questions or feedback: [Your contact info]

---

## 🙏 Acknowledgments

- UFC Stats for data access
- XGBoost and scikit-learn communities
- MMA analytics research community

---

**Built with Python, XGBoost, and a passion for MMA analytics** 🥊📊

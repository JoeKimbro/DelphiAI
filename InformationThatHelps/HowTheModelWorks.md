# How the Delphi AI Prediction Model Works

A complete explanation of how Delphi AI analyzes UFC fights and produces predictions.

---

## Overview

The system produces a fight prediction by blending two independent signals -- a **statistical ELO rating** and a **machine learning model** -- then layering on real-time adjustments for injuries and ring rust. The prediction flows through 9 stages from raw data to final output.

---

## Stage 1: Data Collection

When you ask for a prediction (say, Strickland vs Hernandez), the system queries the PostgreSQL database and pulls each fighter's full profile:

- **Identity**: name, DOB, stance (Orthodox / Southpaw / Switch)
- **Physical**: height, reach (in inches)
- **Career record**: wins, losses, draws
- **Performance stats**: significant strikes landed per minute (SLpM), striking accuracy (%), strikes absorbed per minute (SApM), striking defense (%), takedown average per 15 min, takedown accuracy / defense, submission average per 15 min
- **ELO rating**: pulled from the `EloHistory` table (the most recent `eloAfterFight` value, defaulting to 1500 for new fighters)
- **Derived stats**: knockdown rate (career knockdowns / total fights), recent form (win rate in last 5 fights), finish rate (1 - decision rate), round-by-round KO/Sub percentages

---

## Stage 2: ELO Adjustments (Ring Rust + Injuries)

Before anything else, each fighter's raw ELO gets adjusted for real-world factors.

### Inactivity Decay (Ring Rust)

If a fighter hasn't fought in 6+ months, their ELO is penalized. The penalty has two components:

| Time Off | Proportional Decay Rate | Flat Penalty |
|---|---|---|
| 6-12 months | 8% per year | -10 ELO |
| 12-18 months | 12% per year | -20 ELO |
| 18-24 months | 18% per year | -30 ELO |
| 24+ months | 25% per year | -40 ELO |

The proportional decay is applied to the fighter's ELO above or below the 1500 baseline. The maximum total decay is capped at 35%.

**Example**: A 1600-rated fighter who's been out 14 months loses approximately `(1600 - 1500) * 0.12 * 1.16 + 20 = ~34 ELO`, dropping to roughly 1566.

### Injury Scraping

The system checks UFC.com for injury news on each fighter. It searches recent articles for keywords like "injury," "surgery," "torn," "broken," and "out." Depending on severity:

- **Minor injury**: -17 ELO
- **Moderate injury**: -35 ELO
- **Severe injury**: -55 ELO

Results are cached in the database for 7 days to avoid repeated scraping.

After this stage, each fighter has an `adjusted_elo` that reflects their current real-world readiness, not just their last fight result.

---

## Stage 3: The ELO Probability (Baseline)

The adjusted ELO ratings are converted to a win probability using the standard Elo formula:

```
P(A wins) = 1 / (1 + 10^(-(ELO_A - ELO_B) / 400))
```

A 100-point ELO advantage translates to roughly a 64% win probability. This serves as the **baseline prediction** -- the system falls back to this when the ML model can't produce a reliable signal.

---

## Stage 4: Feature Engineering (ML Input)

The ML model doesn't see raw stats. It sees **20 differential features** -- always computed as `Fighter A's stat minus Fighter B's stat`. The model learns from *relative differences*, not absolute values.

### The 20 Features

| Feature | What It Measures |
|---|---|
| `elo_diff` | Skill gap (ELO A - ELO B) |
| `age_diff` | Age gap in years |
| `reach_diff` | Reach advantage in inches |
| `slpm_diff` | Striking output difference (sig. strikes landed per minute) |
| `stracc_diff` | Striking accuracy difference |
| `tdavg_diff` | Takedown volume difference |
| `subavg_diff` | Submission attempt difference |
| `kd_rate_diff` | Knockdown rate difference |
| `win_rate_diff` | Career win rate difference |
| `recent_form_diff` | Last-5-fights win rate difference |
| `finish_rate_diff` | Finish rate difference |
| `experience_diff` | Total fights difference |
| `avg_fight_time_diff` | Average fight duration difference |
| `style_advantage` | Rock-paper-scissors style matchup (+1, 0, -1) |
| `southpaw_advantage` | Southpaw vs Orthodox stance edge (+1, 0, -1) |
| `elo_prime_interaction` | ELO gap amplified when one fighter is in prime (25-32) and the other isn't |
| `reach_striking_interaction` | Reach gap amplifies striking output difference |
| `is_title_fight` | Whether it's a title fight (1 or 0) |
| `debut_diff` | Whether one fighter is debuting and the other isn't |
| `height_diff` | Height gap (often unavailable) |

### Style Classification

Each fighter is classified into one of four styles based on their career stats:

- **Wrestler**: Takedown average > 2.5 per 15 min
- **Grappler**: Submission average > 1.0 per 15 min (and not already a wrestler)
- **Striker**: SLpM > 4.0 and takedown average < 2.0 (offensive on feet, doesn't grapple much)
- **Balanced**: Everything else

The `style_advantage` feature follows classic MMA matchup theory:

- **Wrestlers beat Strikers** (close distance, control)
- **Grapplers beat Wrestlers** (submit off back or scrambles)
- **Strikers beat Grapplers** (keep distance, avoid clinch)
- Same style or balanced = neutral (0)

### Why Differentials Matter

Every feature is antisymmetric: if you swap Fighter A and B, every differential negates. If Strickland has +50 ELO over Hernandez, then Hernandez has -50 over Strickland. This property is what allows the symmetry correction (Stage 6) to work.

---

## Stage 5: The ML Model (XGBoost)

The model is an **XGBoost gradient-boosted tree classifier**, selected because it had the best validation AUC (0.760) among the four candidates trained:

- Logistic Regression
- Random Forest
- **XGBoost** (winner)
- Stacking Ensemble

### Training Data

- **4,138 UFC fights** from 2019 onward
- **Augmented to 8,276 samples** by including both orientations of every fight (A vs B AND B vs A)
- This guarantees the model can't learn positional bias -- it sees every matchup from both sides

### Training Pipeline

1. **Chronological split** (70/15/15): Training data is pre-2024, validation is 2024, holdout is 2025-2026. No future data leaks into training.
2. **Winsorization**: Caps extreme outlier values at the 1st and 99th percentile of training data
3. **Correlation analysis**: Drops features with r > 0.85 (dropped `elo_diff_sq` and `age_diff_sq` since they're nearly identical to `elo_diff` and `age_diff`)
4. **XGBoost training** with early stopping on the validation set (stopped at 350 of 500 rounds)
5. **Isotonic calibration**: After training, the raw XGBoost probabilities are mapped through an isotonic regression fit on the validation set. This converts XGBoost's confidence scores into actual probabilities that match real-world outcomes. Output is clipped to [0.10, 0.90] -- the system never claims more than 90% certainty because MMA is inherently chaotic.

### Holdout Performance

- **70.1% accuracy** on the untouched 2025-2026 test set
- **0.780 AUC** (area under the ROC curve)
- **+39.8% ROI** when simulating unit bets on the model's picks

### What XGBoost Actually Learns

XGBoost builds approximately 350 decision trees sequentially. Each tree learns from the errors of the previous ones. For a fight prediction, the trees collectively ask questions like:

- "Is the ELO gap large?"
- "Is the younger fighter in prime age?"
- "Does the wrestler face a striker?"

...and combine hundreds of these weak signals into a single probability.

### Most Important Features (by model ranking)

1. `age_diff` (19.1%) -- age is the single strongest predictor
2. `elo_diff` (9.6%) -- historical skill rating
3. `reach_diff` (8.1%) -- physical advantage
4. `debut_diff` (7.4%) -- debuting fighters are disadvantaged
5. `win_rate_diff` (5.9%) -- career win percentage

---

## Stage 6: Symmetry Correction

Even with augmented training, the model can have minor asymmetries due to the calibration layer. So the system runs a **symmetry check**:

1. Predict `prob(A wins)` with A in position 1, B in position 2
2. Predict `prob(A wins)` with B in position 1, A in position 2 (swap the inputs)
3. The **corrected probability** = average of both: `(prob_AB + (1 - prob_BA)) / 2`
4. The **symmetry error** = `|prob_AB + prob_BA - 1.0|`

**Why this matters**: If the model is working perfectly, `prob(A wins given A,B) + prob(A wins given B,A)` should equal 1.0 (the two predictions are mirror images). The symmetry error measures how far off this is.

- If symmetry error <= 0.30 → the model passes, and the corrected probability is used
- If symmetry error > 0.30 → the model is unreliable for that specific matchup, and the system falls back to pure ELO

---

## Stage 7: Blending ML + ELO

When the ML model passes the symmetry check, its corrected probability is **blended with the ELO probability**:

```
final_prob = ML_weight * ML_corrected + (1 - ML_weight) * ELO_prob
```

### ML Weight Calculation

- **Base ML weight**: 50%
- **Reduced by 8% for each missing (NaN) feature**
- Minimum ML weight: 20%

So a fight with 0 NaN features gets 50% ML weight. A fight with 3 NaN features gets 26% ML weight. The model is less trusted when it has incomplete data.

### After Blending

1. **Injury/ring-rust shifts** are applied: each fighter's total penalty (inactivity + injury) can shift the blended probability by up to 10% in either direction
2. The **final probability is capped** to [0.10, 0.90] -- no prediction can be more confident than 90%

---

## Stage 8: Method of Victory + Round Predictions

Once the win probability is determined, a **separate statistical model** (not XGBoost) predicts *how* the fight ends. This uses each fighter's career finishing tendencies.

### KO Power

A weighted combination of:
- KOs in last 5 fights (50% weight)
- Striking output / SLpM (30% weight)
- Opponent's strikes absorbed / SApM (20% weight -- chinny opponents get KO'd more)

### Submission Threat

A weighted combination of:
- Subs in last 5 fights (50%)
- Submission attempts per 15 min (30%)
- Opponent's takedown defense weakness (20% -- poor TD defense means easier path to the mat)

### Method Distribution

These are normalized into a **per-fighter method distribution**: "IF Fighter A wins, what's the probability it's by KO, Sub, or Decision?"

Then they're weighted by each fighter's win probability to give the **overall fight outcome probabilities**. For example: KO 34%, Sub 25%, Dec 41%.

### Round Probabilities

Round predictions use each fighter's historical round-by-round KO% and Sub% from the database, weighted by their win probability, then normalized so that:

```
R1 finish + R2 finish + R3 finish + Decision = 100%
```

---

## Stage 9: Card-Level Output

For full event predictions (using `predict_card`), this entire pipeline runs for every fight on the card:

1. Scrape UFC.com for the fight card (if it's an upcoming event not yet in the database)
2. Fuzzy-match each scraped fighter name to the database
3. Run the full prediction pipeline (Stages 1-8) for each matchup
4. Aggregate into a summary table with KO/Sub/Dec percentages, round probabilities, per-fighter method breakdowns, and betting insights

---

## Concrete Example: Strickland vs Hernandez

Here's how all the stages come together for a single fight:

1. **ELO**: Strickland 1625 (adjusted to ~1606 due to ring rust), Hernandez 1668 (adjusted to ~1643 due to injury). ELO alone gives Hernandez roughly 60%.

2. **ML Model**: Looks at the 20 differential features. Hernandez has a large age advantage (younger, in prime), higher recent ELO, and the age/elo interaction compounds these. The ML's symmetry-corrected probability is approximately 75-80% Hernandez.

3. **Blend**: `50% * ~78% ML + 50% * ~60% ELO = ~69% Hernandez`. Injury shift nudges it slightly. Final: **69% Hernandez**.

4. **Method**: Hernandez has high KO power AND high submission threat. If he wins: 45% KO, 40% Sub, 15% Dec. If Strickland wins: 17% KO, 3% Sub, 80% Dec (volume striker who grinds decisions).

5. **Overall outcome**: KO 36%, Sub 28%, Dec 36%. The fight is more likely to end by finish (64%) than go the distance.

---

## Three-Layer Architecture Summary

| Layer | Source | What It Captures | Fallback |
|---|---|---|---|
| **ELO** | Fight history (wins/losses weighted by opponent quality) | Long-term skill trajectory | Always available |
| **ML (XGBoost)** | 20 differential features from current stats | Stylistic matchups, physical advantages, form, age | Falls back to ELO if symmetry fails or more than 6 NaN features |
| **Method/Round** | Career finishing stats (KO%, Sub%, round distributions) | How the fight ends | Static calculation, always available |

The ELO provides a stable foundation. The ML model adds the nuance that ELO misses (style matchups, physical edges, age curves). The blending ensures that neither system can produce wildly overconfident predictions on its own.

---

## Key Design Decisions

- **Why blend instead of using ML alone?** ML models can be overconfident or biased. Blending with ELO acts as a regularizer -- it anchors the prediction to historical performance.
- **Why the symmetry check?** It catches cases where the model has no real signal (outputs the same probability regardless of who's in which position). Falling back to ELO for those fights is honest.
- **Why cap at [0.10, 0.90]?** MMA is the most unpredictable major sport. Even extreme mismatches can produce upsets. Capping prevents the system from ever saying "this is a lock."
- **Why full data augmentation?** Training with both orientations of every fight makes positional bias structurally impossible. The model can never learn "Fighter A tends to win" because it sees every fighter equally in both positions.
- **Why isotonic calibration?** XGBoost's raw probability outputs don't match real-world frequencies. A raw 0.70 might only win 62% of the time. Isotonic regression maps these to calibrated probabilities that you can trust at face value.

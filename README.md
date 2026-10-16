# Football Prediction & Market Efficiency Model

A quantitative football modelling and betting research pipeline for the **English Premier League** (and Championship). This project implements a time-decayed **Dixon-Coles** goal model, **XGBoost** machine learning classifiers/regressors built on **Understat xG** and match-context features, leakage-free **model stacking**, and rigorous **walk-forward cross-validation** against bookmaker opening and closing lines.

---

## Key Features & Capabilities

1. **Dixon-Coles Parametric Goal Model (`models/dixon_coles.py`)**
   - Unconstrained `L-BFGS-B` Maximum Likelihood Estimation (MLE) using sum-to-zero attack parameter reconstruction ($n - 1$ free attack parameters) for fast convergence in rolling backtests.
   - Exponential time-decay weighting: $w_i = \exp(-\xi \cdot \Delta t_i)$.
   - Low-scoreline dependency adjustment $\tau(x, y \mid \lambda, \mu, \rho)$ for $0\text{–}0, 1\text{–}0, 0\text{–}1, 1\text{–}1$ scorelines.
   - Full scoreline probability matrix pricing both **1X2** and **Over/Under $2.5$ Totals** markets simultaneously.
   - Cold-start prior seeding (`weakest`, `average`, or `percentile`) for newly promoted teams with no top-flight history.

2. **Expected Goals (xG) & Contextual Feature Engineering (`data/`, `features/`)**
   - Automated scraping and local CSV caching of match-level xG from **Understat** (via `penaltyblog`) back to `2014-15`, aligned with **football-data.co.uk** match and odds data.
   - **Rolling Form (`features/rolling_form.py`)**: Strict pre-match shifted rolling averages ($n=5$) of xG created, xG conceded, goals scored/conceded, and matchup attack/defence differentials.
   - **Structural Context (`features/context.py`)**: Leakage-safe features capturing signals Dixon-Coles structurally ignores:
     - Rest days and 14-day fixture congestion
     - Chronologically reconstructed pre-match league table position, PPG, and season progress
     - Promoted-team indicator flags
     - Rolling xG overperformance ($\text{Goals} - \text{xG}$, regression-to-the-mean signal) and schedule strength (recent opponents' PPG)

3. **Machine Learning & Stacking (`models/xgb_model.py`, `main_stack.py`)**
   - **xG-Form XGBoost**: Multi-class (`multi:softprob`) classifier predicting 1X2 probabilities from rolling xG features.
   - **Leakage-Free Stacked Ensemble**: Walk-forward out-of-fold Dixon-Coles probabilities (`dc_home`, `dc_draw`, `dc_away`, `dc_exp_home`, `dc_exp_away`) combined with rolling xG form and contextual features in an XGBoost meta-learner.

4. **Market Efficiency, CLV & Betting Simulation (`evaluation/`, `betting/`)**
   - Supports multiple bookmakers and aggregates (`bet365`, `pinnacle`, `market_avg`, `market_max`, and their `*_closing` equivalents) with `basic` or `power` margin removal (devigging).
   - **True Closing Line Value (CLV)** grading: Bets are placed at best available opening odds (`market_max`) and graded against best closing prices (`market_max_closing`).
   - **Staking Strategies**: Flat staking, fractional Kelly Criterion (25% Kelly, capped), and edge-proportional staking.
   - **Statistical Significance**: Expanding-window season-by-season walk-forward cross-validation (`2019-20` to `2025-26`, $2,660$ matches) with **10,000-resample paired bootstrap** confidence intervals on Ranked Probability Score (RPS) and Brier score.

---

## Project Structure

```text
Football-Prediction-Model/
├── config.py                  # Central configuration (seasons, DC decay xi, bookmakers, betting bank/edge rules)
├── main.py                    # Core pipeline: walk-forward Dixon-Coles backtest, bookmaker report & 1X2 betting sim
├── main_xgb.py                # Single-split comparison: XGBoost (xG form) vs Dixon-Coles vs Closing Market
├── main_stack.py              # Stacked model pipeline: Out-of-fold Dixon-Coles + Rolling xG Form -> XGBoost
├── main_cv.py                 # 7-season expanding-window walk-forward CV (2,660 matches) + paired bootstrap tests
├── main_tune_xi.py            # Nested rolling-origin tuning of Dixon-Coles time-decay rate (xi)
├── main_totals.py             # Over/Under 2.5 goals evaluation & value-betting simulation
├── main_clv_target.py         # Closing line movement regressor: predicting (P_close - P_open) directly
│
├── data/
│   ├── understat.py           # Understat xG scraper, team-name normaliser & local CSV caching
│   └── understat_cache/       # Cached EPL Understat xG CSVs (2014-15 through 2025-26)
│
├── features/
│   ├── rolling_form.py        # Shifted rolling xG & goals form features + matchup differentials
│   └── context.py             # Rest/congestion, live league standings, xG overperformance & opponent strength
│
├── models/
│   ├── dixon_coles.py         # DixonColesModel: L-BFGS-B MLE, scoreline matrix, totals pricing & promoted priors
│   └── xgb_model.py           # XGBResultModel: 1X2 classifier wrapper & gain-based feature importance
│
├── evaluation/
│   ├── backtester.py          # Walk-forward match-by-match Backtester for Dixon-Coles
│   ├── bookmaker.py           # BookmakerComparison: devigging, calibration, disagreement analysis & 4-panel plots
│   ├── walkforward.py         # Multi-season expanding-window CV engine & paired bootstrap significance testing
│   └── totals.py              # Two-way Over/Under 2.5 devigging, Brier/log-loss metrics & totals bet finder
│
├── betting/
│   └── strategy.py            # BettingStrategy: value bet scanner, CLV tracking, flat/Kelly/edge staking & plots
│
└── outputs/                   # Generated CSVs (predictions, CV tables, value bets) and performance plots (.png)
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Required packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib`, `penaltyblog`

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib penaltyblog
```

*(Note: Match results and odds are streamed directly from `football-data.co.uk`, while Understat xG data is already cached in `data/understat_cache/` for the `2014-15` to `2025-26` Premier League seasons.)*

---

## Usage & Pipelines

All parameters (active league, test season, time-decay $\xi$, bookmaker columns, bankroll, and edge thresholds) are centralised in `config.py`.

### 1. Core Dixon-Coles Backtest & 1X2 Betting Simulation
Runs the walk-forward Dixon-Coles model on `TEST_SEASON`, compares calibration and RPS against the bookmaker, simulates `flat`, `kelly`, and `edge` staking strategies, and saves plots and bets to `outputs/`:
```bash
python main.py
```

### 2. Multi-Season Walk-Forward Cross-Validation (Recommended Benchmark)
Evaluates all models (`Market (closing)`, `Dixon-Coles`, `Stacked + context`, `Stacked (DC + xG/XGB)`, `XGBoost (xG form)`) across 7 expanding-window test folds (`2019-20` to `2025-26`, $2,660$ matches) with paired bootstrap significance tests:
```bash
python main_cv.py
```

### 3. Single-Season xG-XGBoost & Stacked Model Runs
Compare pure xG-form XGBoost or the out-of-fold DC + xG stacked model on `TEST_SEASON`:
```bash
python main_xgb.py
python main_stack.py
```

### 4. Nested Hyperparameter Tuning for Time Decay ($\xi$)
Performs unbiased nested rolling-origin validation over candidate decay rates $\xi \in [0.0005, 0.015]$ (half-lives from ~46 days to ~3.8 years) to test parameter stability and out-of-sample impact:
```bash
python main_tune_xi.py
```

### 5. Over/Under 2.5 Goals Market Evaluation
Prices the 2.5 goals line directly from the walk-forward Dixon-Coles scoreline matrix and tests whether the two-way totals market offers softer value than 1X2:
```bash
python main_totals.py
```

### 6. Closing Line Movement Prediction (`CLV Target`)
Replaces noisy discrete match outcomes with continuous market line movement ($\Delta P = P_{\text{close}} - P_{\text{open}}$) using `XGBRegressor`, testing whether disagreements between Dixon-Coles and opening odds predict closing line drift:
```bash
python main_clv_target.py
```

---

## Summary of Empirical Findings

Pooled results across **2,660 Premier League matches** (`2019-20` to `2025-26` walk-forward cross-validation from `outputs/cv_pooled.csv`):

| Model | Matches ($N$) | Accuracy | Mean RPS (Lower = Better) | Log Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Market (closing)** | 2,660 | **54.74%** | **0.1967** | **0.9635** |
| **Dixon-Coles (walk-forward)** | 2,660 | 52.89% | 0.2034 | 0.9872 |
| **Stacked + Context** | 2,660 | 52.11% | 0.2063 | 1.0015 |
| **Stacked (DC + xG/XGB)** | 2,660 | 52.33% | 0.2065 | 1.0018 |
| **XGBoost (xG form only)** | 2,660 | 50.75% | 0.2156 | 1.0280 |

### Key Takeaways
- **Dixon-Coles vs. Tabular ML**: Parametric Dixon-Coles (`RPS 0.2034`) significantly outperforms standalone rolling-xG XGBoost (`RPS 0.2156`) and slightly outperforms stacked meta-models (`RPS 0.2063`), because global team-strength estimation via MLE handles opponent quality and long-horizon shrinkage more sample-efficiently than tree splits on ~1,500–4,000 matches.
- **Time-Decay ($\xi$) Flatness**: Nested validation (`outputs/xi_tuning_curve.csv`) shows a shallow optimum around $\xi \in [0.001, 0.00325]$ (half-life ~213 to ~693 days), with less than `0.0007` RPS difference across that plateau.
- **Line Shopping vs. Closing Efficiency**: Shopping across bookmakers (`market_max` vs `bet365`) reduces bookmaker overround from ~5.6% to ~2.3%, saving ~3.3% margin per bet—an execution advantage larger than most model-level improvements.

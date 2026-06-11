# World Cup 2026 Prediction Model — Poisson + LightGBM + Optuna

This repository contains a machine learning pipeline to predict FIFA World Cup 2026 group-stage match outcomes.

The notebook builds a hybrid model combining:

- **Poisson regression** to estimate expected goals (`lambda_home`, `lambda_away`) and scoreline probabilities.
- **LightGBM multiclass classification** to predict match outcome probabilities: home win, draw, away win.
- **Optuna hyperparameter optimization** to tune the LightGBM model.
- **Monte Carlo group-stage simulation** to estimate qualification and group-ranking probabilities.

The main notebook is:

```text
wc_prediction_poisson_lgbm_optuna_wc2022_test_wc2026.ipynb
```

---

## Project objective

The goal is to produce pre-match predictions for the FIFA World Cup 2026 group stage:

```text
p_home_win, p_draw, p_away_win
```

and to derive:

- expected goals for both teams;
- most likely scorelines;
- group-stage qualification probabilities;
- group-rank probabilities;
- model-based betting-value analysis.

The project focuses on **pre-match predictive features only**. Post-match variables such as shots, xG, corners, cards, or possession are deliberately excluded to avoid data leakage.

---

## Repository structure

Suggested repository layout:

```text
.
├── wc_prediction_poisson_lgbm_optuna_wc2022_test_wc2026.ipynb
├── train.csv
├── results.csv
├── eloratings.csv
├── WorldCup2026.xlsx
├── test_wc2022.csv
├── wc2026_group_fixtures_with_odds_fixed.csv
├── knockout_slots.csv
├── submission_wc2026_group_stage_poisson_lgbm_optuna.csv
├── wc2022_backtest_predictions_poisson_lgbm.csv
├── wc2026_group_qualification_probabilities.csv
├── feature_importance_lgbm_optuna.csv
└── README.md
```

---

## Data inputs

The notebook expects the following files in the project root.

| File | Description |
|---|---|
| `train.csv` | Historical international football matches used for model training. |
| `results.csv` | Additional historical international results. |
| `eloratings.csv` | External Elo ratings by date and team. |
| `WorldCup2026.xlsx` | Historical World Cup / qualification data and bookmaker odds when available. |
| `test_wc2022.csv` | FIFA World Cup 2022 fixtures used for backtesting. |
| `wc2026_group_fixtures_with_odds_fixed.csv` | FIFA World Cup 2026 group-stage fixtures with pre-match odds. |
| `knockout_slots.csv` | Optional knockout bracket mapping for future bracket simulation. |

---

## External data sources

The project can be reproduced with data from the following public sources or equivalent datasets:

### International match results

Historical international football results are commonly available from:

- Kaggle — International football results dataset  
  https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

This type of dataset usually contains:

- match date;
- home team;
- away team;
- home goals;
- away goals;
- tournament;
- neutral-site flag.

### Elo ratings

External Elo ratings are based on public national-team Elo ratings:

- World Football Elo Ratings  
  https://www.eloratings.net/

The notebook expects an `eloratings.csv` file with at least:

```text
date, team, rating
```

### Betting odds

Historical and pre-match odds can be collected from football odds datasets such as:

- Football-Data.co.uk  
  https://www.football-data.co.uk/

The notebook converts decimal odds into normalized implied probabilities:

```text
market_p_home_mean
market_p_draw_mean
market_p_away_mean
```

### FIFA World Cup 2026 fixtures

The 2026 fixtures and tournament structure should be aligned with official FIFA information:

- FIFA World Cup 2026 official website  
  https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026
- FIFA match schedule  
  https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/match-schedule

### Competition format

The FIFA World Cup 2026 format contains:

- 48 teams;
- 12 groups of 4 teams;
- top 2 teams from each group qualifying directly;
- 8 best third-placed teams also qualifying;
- knockout stage starting from the round of 32.

Sources:

- FIFA World Cup 2026 official information  
  https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026
- Reuters World Cup 2026 overview  
  https://www.reuters.com/sports/soccer/world-cup-2026-teams-qualified-key-players-fixtures-2026-06-11/

---

## Methodology

### 1. Data loading and cleaning

The notebook loads and harmonizes historical match datasets.

Main cleaning steps:

- standardize team names;
- parse dates;
- normalize score columns;
- remove duplicates;
- add result labels:
  - `H` = home win;
  - `D` = draw;
  - `A` = away win;
- add tournament importance;
- identify neutral venues.

Team-name harmonization is handled through a custom mapping dictionary, for example:

```python
TEAM_NAME_MAP = {
    "USA": "United States",
    "South Korea": "Korea Republic",
    "Czechia": "Czech Republic",
    ...
}
```

---

### 2. Market odds features

Decimal odds are converted into normalized implied probabilities.

For odds:

```text
H_odds, D_odds, A_odds
```

the implied probabilities are computed as:

```python
inv = 1 / odds
probabilities = inv / inv.sum()
```

The notebook then creates market-consensus features such as:

```text
market_p_home_mean
market_p_draw_mean
market_p_away_mean
market_favorite_prob
market_underdog_prob
market_home_minus_away
market_draw_gap
market_entropy
has_market_odds
```

Important: odds are treated as **pre-match market information** only.

---

### 3. Feature engineering

The notebook builds chronological features to avoid leakage.

Main features include:

#### Team strength

- internal Elo rating;
- external Elo rating;
- Elo difference;
- home advantage adjustment.

#### Recent form

Rolling windows over recent matches:

```text
last5_points_avg
last5_win_rate
last5_draw_rate
last5_gd_avg
last5_gf_avg
last5_ga_avg
last5_clean_sheet_rate
last5_failed_score_rate
```

Equivalent features are also computed for other windows such as 10, 20, or 40 matches depending on the configuration.

#### Head-to-head

Recent head-to-head features:

```text
h2h_last3_home_points_avg
h2h_last3_home_win_rate
h2h_last3_draw_rate
h2h_last3_home_gd_avg
h2h_last3_n
```

#### Confederation-based features

Team performance against opponents from the same confederation:

```text
lastN_points_avg_vs_conf
lastN_win_rate_vs_conf
lastN_gd_avg_vs_conf
```

#### Calendar and rest

The notebook also tracks:

- days since previous match;
- number of recent matches in the last N days;
- tournament importance;
- neutral-site flag.

---

## Models

### Poisson model

Two independent Poisson regressors are trained:

- one for home goals;
- one for away goals.

The expected goals are clipped to a reasonable range:

```python
lambda_home = clip(predicted_home_goals, 0.05, 5.5)
lambda_away = clip(predicted_away_goals, 0.05, 5.5)
```

The score matrix is then computed as:

```python
P(home_goals = i, away_goals = j)
```

for scorelines from 0 to `POISSON_MAX_GOALS`.

The score matrix is aggregated into:

```text
p_home_win
p_draw
p_away_win
```

---

### LightGBM model

A multiclass LightGBM model predicts:

```text
H, D, A
```

with the following objective:

```python
objective = "multiclass"
num_class = 3
metric = "multi_logloss"
```

The model is trained on engineered numeric features.

---

### Optuna tuning

Optuna is used to tune LightGBM hyperparameters, including:

```text
learning_rate
num_leaves
max_depth
min_data_in_leaf
feature_fraction
bagging_fraction
bagging_freq
lambda_l1
lambda_l2
min_gain_to_split
```

The optimization target is validation multiclass log loss.

Default configuration:

```python
N_OPTUNA_TRIALS = 80
LGBM_NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 150
RANDOM_STATE = 42
```

For stronger results, increase:

```python
N_OPTUNA_TRIALS = 200
```

or more if runtime allows.

---

## Blended prediction

The final prediction is a weighted blend of:

- LightGBM outcome probabilities;
- Poisson outcome probabilities.

The blend weight is optimized during the World Cup 2022 backtest:

```python
p_blend = w * p_lgbm + (1 - w) * p_poisson
```

The optimized weight is then reused for the World Cup 2026 inference.

---

## Backtest on FIFA World Cup 2022

The notebook performs a pre-tournament-style backtest:

1. train only on matches before the FIFA World Cup 2022;
2. predict all 64 World Cup 2022 matches;
3. compare predicted probabilities with actual results;
4. evaluate:
   - log loss;
   - accuracy;
   - model calibration qualitatively.

Generated output:

```text
wc2022_backtest_predictions_poisson_lgbm.csv
```

The backtest is designed as a sanity check before applying the final model to 2026.

---

## World Cup 2026 inference

The final model is trained on all available historical data before the start of World Cup 2026.

The notebook then predicts the group-stage fixtures from:

```text
wc2026_group_fixtures_with_odds_fixed.csv
```

Generated output:

```text
submission_wc2026_group_stage_poisson_lgbm_optuna.csv
```

The output contains:

```text
match_id
date
group
home_team
away_team
venue
lambda_home
lambda_away
p_home_win
p_draw
p_away_win
pred
```

Example interpretation:

```text
lambda_home = 2.40
lambda_away = 0.70
```

means the model expects the home team to score around 2.4 goals and the away team around 0.7 goals.

---

## Group-stage simulation

The notebook includes an optional Monte Carlo simulation of the World Cup 2026 group stage.

Simulation steps:

1. sample each match outcome from predicted probabilities;
2. sample scorelines conditionally from the Poisson lambdas;
3. update group tables;
4. rank teams by:
   - points;
   - goal difference;
   - goals scored;
   - wins;
   - random tie-breaker fallback;
5. qualify:
   - top 2 teams in each group;
   - 8 best third-placed teams.

Generated output:

```text
wc2026_group_qualification_probabilities.csv
```

Expected columns:

```text
team
group
qualification_probability
rank_1_probability
rank_2_probability
rank_3_probability
rank_4_probability
```

---

## Feature importance

The final LightGBM model exports feature importance:

```text
feature_importance_lgbm_optuna.csv
```

The notebook reports both:

- gain importance;
- split importance.

This helps understand whether the model relies mostly on:

- market odds;
- Elo ratings;
- recent form;
- goals for / goals against;
- rest days;
- head-to-head features.

---

## Installation

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn lightgbm optuna openpyxl matplotlib
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install numpy pandas scikit-learn lightgbm optuna openpyxl matplotlib
```

---

## Usage

Run the notebook from top to bottom:

```bash
jupyter notebook wc_prediction_poisson_lgbm_optuna_wc2022_test_wc2026.ipynb
```

or:

```bash
jupyter lab
```

Suggested execution order:

1. load libraries and constants;
2. load historical data;
3. clean and standardize team names;
4. build market odds features;
5. build chronological features;
6. run the World Cup 2022 backtest;
7. train final model;
8. predict World Cup 2026 group stage;
9. run optional group simulation;
10. export CSV outputs.

---

## Outputs

| Output file | Description |
|---|---|
| `submission_wc2026_group_stage_poisson_lgbm_optuna.csv` | Match-level probabilities and expected goals for World Cup 2026 group-stage fixtures. |
| `wc2022_backtest_predictions_poisson_lgbm.csv` | World Cup 2022 backtest predictions. |
| `wc2026_group_qualification_probabilities.csv` | Monte Carlo qualification and ranking probabilities. |
| `feature_importance_lgbm_optuna.csv` | LightGBM feature importance. |

---

## Betting interpretation

This project can be used to compare model probabilities against bookmaker odds.

For a given event with model probability:

```text
p_model
```

the fair decimal odds are:

```text
fair_odds = 1 / p_model
```

A bet is theoretically positive expected value if:

```text
bookmaker_odds > fair_odds
```

Example:

```text
p_model = 0.60
fair_odds = 1 / 0.60 = 1.67
```

If the bookmaker offers 2.00, the model sees potential value.

Important: this is not financial advice or betting advice. Football outcomes are high-variance, and model probabilities can be miscalibrated.

---

## Limitations

This project has several limitations:

1. **No player-level availability by default**  
   Injuries, suspensions, rotations, and squad announcements are not fully modeled unless manually added.

2. **Market odds can dominate**  
   Odds are powerful predictive features, but they may also reduce the model's independence from bookmaker expectations.

3. **World Cup context is special**  
   Neutral venues, travel, climate, pressure, squad freshness, and knockout incentives can all affect performance.

4. **Team strength changes over time**  
   National teams are volatile because player pools, coaches, and tactical systems change quickly.

5. **Poisson independence assumption**  
   The Poisson model assumes home and away goals are conditionally independent, which is a simplification.

6. **Scoreline probabilities are fragile**  
   Exact score predictions are much noisier than outcome probabilities.

7. **Group simulation tie-breakers are simplified**  
   FIFA tie-breakers are approximated; the simulation uses a practical ranking fallback.

---

## Future improvements

Potential extensions:

- add FIFA rankings;
- add player-level features;
- add squad market value from Transfermarkt-like data;
- add injury and suspension data;
- add travel distance and climate features;
- calibrate probabilities with isotonic regression or Platt scaling;
- compare against bookmaker closing odds;
- add Bayesian hierarchical Poisson models;
- add Dixon-Coles correction for low-score dependence;
- improve third-place qualification simulation using exact FIFA tie-breakers;
- build a Streamlit dashboard for interactive exploration.

---

## Disclaimer

This project is for educational and analytical purposes only.

The predictions are model-based estimates, not certainties. They should not be interpreted as guaranteed outcomes or financial advice.

---

## Credits and references

- FIFA World Cup 2026 official website:  
  https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026

- FIFA World Cup 2026 match schedule:  
  https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/match-schedule

- Reuters — World Cup 2026 overview:  
  https://www.reuters.com/sports/soccer/world-cup-2026-teams-qualified-key-players-fixtures-2026-06-11/

- Kaggle — International football results dataset:  
  https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

- World Football Elo Ratings:  
  https://www.eloratings.net/

- Football-Data.co.uk:  
  https://www.football-data.co.uk/

- LightGBM documentation:  
  https://lightgbm.readthedocs.io/

- Optuna documentation:  
  https://optuna.org/

- scikit-learn documentation:  
  https://scikit-learn.org/

- Poisson regression reference, scikit-learn:  
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html

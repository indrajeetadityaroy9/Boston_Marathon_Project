# Boston Marathon Finish Time Prediction (MATH 7343)

A multi-scale prediction framework for Boston Marathon finish times, progressively incorporating demographic, longitudinal, and in-race information. Uses 615,682 finisher records from 1897-2019.

## Research Questions

- **RQ1 (Pre-race demographic prediction):** How accurately can finish time be predicted from age, gender, and year alone? Adding a runner's historical mean finish time reduces test RMSE from 2,504s to 1,812s (R² from 0.17 to 0.51).
- **RQ2 (Pre-race personalized prediction):** For repeat runners, how much does a mixed-effects model improve prediction? In-sample conditional RMSE = 996s; out-of-sample = 1,593s. Honest value of personalization = 651s (~10.8 min).
- **RQ3 (In-race progressive prediction):** How does prediction improve at each checkpoint? Ridge regression RMSE drops from 1,259s at 5K to 760s at halfway to 124s at 40K. At early checkpoints (5K, 10K), knowing a runner's past race history gives a better prediction than knowing their current split time. But by 15K, the accumulating split data becomes more informative than the runner's historical profile — this is the crossover point.

## Setup

### Install dependencies

```bash
pip install pandas numpy scikit-learn scipy statsmodels scikit-posthocs matplotlib
```

The raw CSV files are included in `data/` (sourced from [adrian3/Boston-Marathon-Data-Project](https://github.com/adrian3/Boston-Marathon-Data-Project)). Wheelchair and diverted variants are automatically skipped during cleaning.

## Running the Pipeline

Scripts run from the project root. Steps 1-2 are prerequisites. Steps 3-4 can run in parallel. Step 5 depends on Step 4.

### Step 1: Clean the data

```bash
python data_scripts/01_data_cleaning.py
```

Reads all CSVs from `data/`, unifies them into a single schema, parses times, cleans gender and age values, imputes missing ages via KNN, validates, and writes `cleaned_data/boston_marathon_cleaned.csv` (615,682 rows, 39 columns).

### Step 2: Feature discovery EDA

```bash
python data_scripts/02_feature_discovery_eda.py
```

Seven analysis sections feeding into the prediction pipeline:

1. **Descriptive statistics** — center, spread, shape by gender and decade
2. **Normality tests** — Shapiro-Wilk on raw and log-transformed times
3. **Correlation analysis** — Pearson, Spearman, partial correlation (age vs finish time controlling for gender)
4. **Gender comparison** — Welch's t-test, Mann-Whitney U, Hedges' g effect size, decade trend
5. **Age group analysis** — Kruskal-Wallis, ANOVA with eta-squared, Tukey HSD, Dunn's post-hoc
6. **Split-time analysis** — checkpoint correlation matrix (2015-2017), pacing classification
7. **Repeat runner profiling** — name collision filtering, ICC, within-runner aging slopes

### Step 3: Demographic baseline prediction (RQ1)

```bash
python data_scripts/03_rq1_demographic_baseline.py
```

Fits three OLS models on the 2000+ non-imputed sample (n=427,258):

- **Linear OLS:** finish_time ~ centered_age + female + centered_year
- **Quadratic OLS:** adds centered_age² + age×female interaction
- **History OLS:** adds prior_appearances + prior_mean_time (repeat runners only, n=19,636 in test)

Temporal hold-out: train on 2000-2017, test on 2018-2019. Includes 5-fold cross-validation grouped by year and standardized feature importance.

### Step 4: Personalized mixed-effects prediction (RQ2)

```bash
python data_scripts/04_rq2_personalized_mixed_effects.py
```

Fits three nested models on 188,310 observations from 65,590 repeat runners (2000-2019):

- **OLS baseline:** no random effects
- **Random intercept:** per-runner baseline offset (captures individual ability)
- **Random intercept + slope:** per-runner aging rate (captures individual aging)

Includes ICC computation, likelihood ratio tests with boundary correction, Nakagawa-Schielzeth variance decomposition, residual diagnostics, and per-runner predicted offset export. Reports both in-sample and out-of-sample (temporal hold-out: train 2000-2016, test 2017-2019) evaluation.

### Step 5: Progressive checkpoint prediction (RQ3)

```bash
python data_scripts/05_rq3_progressive_checkpoint.py
```

At each of 9 checkpoints (5K through 40K), fits Ridge regression models on 79,038 runners (train 2015-2016, test 2017):

- **Naive baseline:** constant-pace extrapolation
- **Splits-only Ridge:** cumulative checkpoint times
- **Splits + demographics Ridge:** adds age and gender
- **Full Ridge:** adds per-runner predicted offsets from RQ2 (leak-free, 30% coverage)

Produces a prediction convergence curve and crossover analysis identifying where splits outperform personalized history (~15K).

## Cleaned Data Dictionary

`cleaned_data/boston_marathon_cleaned.csv` — 615,682 rows, 39 columns.

| Column | Type | Description |
|---|---|---|
| `year` | int | Race year (1897-2019) |
| `display_name` | string | Runner's full display name |
| `first_name` | string | First name (null for some legacy years) |
| `last_name` | string | Last name (null for some legacy years) |
| `age` | float | Runner's age; null for years lacking age data |
| `gender` | string | `M` or `F` |
| `residence` | string | Combined city, state, country string (post-~2001 only) |
| `city` | string | City of residence (post-~2001 only) |
| `state` | string | State of residence (post-~2001 only) |
| `country_residence` | string | Country of residence (post-~2001 only) |
| `country_citizenship` | string | Country of citizenship (post-~2001 only) |
| `bib` | float | Bib number (post-~2001 only) |
| `pace` | string | Pace as original time string (e.g. `7:08`) |
| `official_time` | string | Official finish time string (e.g. `2:55:10`) |
| `seconds` | float | Finish time in seconds |
| `overall` | int | Overall finishing place |
| `gender_result` | int | Finishing place within gender |
| `division_result` | int | Finishing place within age division |
| `5k`-`40k` | string | Original checkpoint time strings (2015-2017 only) |
| `half` | string | Original half-marathon time string (2015-2017 only) |
| `projected_time` | string | Projected finish time (2015-2017 only) |
| `pace_seconds_per_mile` | float | Pace in seconds per mile |
| `5k_seconds`-`40k_seconds` | float | Checkpoint times in seconds (2015-2017 only) |
| `half_seconds` | float | Half-marathon time in seconds (2015-2017 only) |
| `age_imputed` | bool | `True` if age was filled in by KNN imputation |

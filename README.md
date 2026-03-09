# Boston Marathon Statistical Analysis (MATH 7343) WIP!!!

Statistical analysis of Boston Marathon race results from 1897-2019. The pipeline cleans and unifies ~120 per-year CSV files, then runs exploratory data analysis with hypothesis testing.

## Setup

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn scipy statsmodels scikit-posthocs
```

The raw CSV files are included in `data/` (sourced from [adrian3/Boston-Marathon-Data-Project](https://github.com/adrian3/Boston-Marathon-Data-Project)). The wheelchair and diverted variants are automatically skipped during cleaning.

## Running the pipeline

### Step 1: Clean the data

```bash
python data_cleaning.py
```

This reads all CSVs from `data/`, unifies them into a single schema, parses times, cleans gender and age values, imputes missing ages via KNN, validates, and writes `cleaned_data/boston_marathon_cleaned.csv`.

### Step 2: Run the exploratory data analysis

```bash
python eda.py
```

This reads the cleaned CSV and prints results for 10 analysis sections:

1. Descriptive statistics (overall, by gender, by decade)
2. Percentile analysis
3. Normality and goodness-of-fit tests (Shapiro-Wilk, Anderson-Darling, Lilliefors)
4. Tail behavior and extreme value analysis (GEV fit, log-normal fit)
5. Correlation analysis (Pearson, Spearman, point-biserial, partial)
6. Time series metrics (yearly trends, winning time regression)
7. Gender comparison tests (Welch's t-test, Mann-Whitney U, Levene's, KS, Hedges' g)
8. Age group analysis (Kruskal-Wallis, ANOVA, Dunn's post-hoc)
9. Outlier detection (IQR, Z-score, modified Z-score/MAD)
10. Split-time pacing analysis (2015-2017 only)

## Cleaned data dictionary

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
| `seconds` | float | Finish time in seconds (derived from `seconds` or parsed from `official_time`) |
| `overall` | int | Overall finishing place |
| `gender_result` | int | Finishing place within gender |
| `division_result` | int | Finishing place within age division |
| `5k`-`40k` | string | Original checkpoint time strings (2015-2017 only) |
| `half` | string | Original half-marathon time string (2015-2017 only) |
| `projected_time` | string | Projected finish time (2015-2017 only) |
| `pace_seconds_per_mile` | float | Pace in seconds per mile (parsed or derived from `seconds / 26.2188`) |
| `5k_seconds`-`40k_seconds` | float | Checkpoint times in seconds (2015-2017 only) |
| `half_seconds` | float | Half-marathon time in seconds (2015-2017 only) |
| `age_imputed` | bool | `True` if age was filled in by KNN imputation |

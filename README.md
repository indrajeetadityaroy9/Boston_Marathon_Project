# Boston Marathon Statistical Analysis (MATH 7343)

Statistical analysis of Boston Marathon results from 1897-2019 using a two-stage approach: (1) statistical testing to determine whether finish times differ by age group and gender, and (2) mixed-effects modeling to track how individual runners' performances change with age over repeated appearances. The pipeline cleans and unifies ~120 per-year CSV files, runs exploratory data analysis with hypothesis testing, and includes a pacing strategy analysis.

## Setup

### Install dependencies

```bash
pip install pandas numpy scikit-learn scipy statsmodels scikit-posthocs matplotlib
```

The raw CSV files are included in `data/` (sourced from [adrian3/Boston-Marathon-Data-Project](https://github.com/adrian3/Boston-Marathon-Data-Project)). Wheelchair and diverted variants are automatically skipped during cleaning.

## Running the Pipeline

### Step 1: Clean the data

```bash
python data_cleaning.py
```

Reads all CSVs from `data/`, unifies them into a single schema, parses times, cleans gender and age values, imputes missing ages via KNN, validates, and writes `cleaned_data/boston_marathon_cleaned.csv`.

### Step 2: Run the exploratory data analysis

```bash
python eda.py
```

Prints results for 7 analysis sections:

1. **Descriptive statistics** — overall, by gender, by decade; skewness and kurtosis
2. **Normality tests** — Shapiro-Wilk on raw and log-transformed times (5000-row samples); shape comparison before/after log transform
3. **Correlation analysis** — Pearson, Spearman, and partial correlation (age vs. finish time, controlling for gender); excludes KNN-imputed ages
4. **Gender comparison** — Welch's t-test, Mann-Whitney U, Hedges' g effect size; gender gap by decade
5. **Age group analysis** — Kruskal-Wallis, one-way ANOVA with eta-squared, Dunn's post-hoc (Bonferroni), Tukey's HSD
6. **Split-time analysis** — checkpoint correlation matrix (2015-2017), pacing classification (negative/even/positive split) by year
7. **Repeat runner profiling** — name collision filtering, appearance counts, age span, ICC for random intercepts justification, within-runner aging slopes for random slopes justification

### Step 3: Run the pacing strategy analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook pacing_analysis.ipynb
```

Analyzes pacing strategies using 2015-2017 split data (~79,000 runners):

- Classifies runners as negative split (ratio < 0.99), even split (0.99-1.01), or positive split (> 1.01) based on second-half/first-half pace ratio
- **Kruskal-Wallis test** — tests whether finish time distributions differ across pacing groups (H = 1709.57, p < 0.001)
- **Dunn's post-hoc test** — all three pairwise comparisons significant (Bonferroni correction)
- **Effect size** — eta-squared = 0.019 (1.9% of finish time variance explained by pacing group)
- **Gender breakdown** — pacing profiles are nearly identical across genders (~93% positive split for both)
- Visualizations: pacing distribution bar chart, finish time box plots by pacing group, grouped bar chart by gender

## Research Questions

- **Q1 (Inference)**: Do finish times differ across age groups and gender? Are the differences practically meaningful, or only statistically significant because of the large sample size? Tested with both parametric (Welch's t-test, ANOVA, Tukey's HSD) and non-parametric (Mann-Whitney U, Kruskal-Wallis, Dunn's) methods, with effect sizes (Hedges' g, eta-squared) to assess practical significance.
- **Q2 (Longitudinal)**: How does an individual runner's performance change with age? Do all runners slow down at the same rate? Addressed with a linear mixed-effects model with per-runner random intercepts (justified by ICC = 0.69) and random slopes on age (justified by high variance in within-runner aging slopes). This accounts for the non-independence of repeat runners (~51.6% of the dataset) that Q1 ignores.

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

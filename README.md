# Boston Marathon Statistical Analysis (MATH 7343) WIP!!!

Statistical analysis of Boston Marathon race results from 1897-2019. The pipeline cleans and unifies ~120 per-year CSV files, then runs exploratory data analysis with hypothesis testing.

## Setup

### 1. Download the raw data

Clone or download the CSV files from the original data source:

```bash
git clone https://github.com/adrian3/Boston-Marathon-Data-Project.git
```

### 2. Place the CSV files

Copy all `results*.csv` files from the cloned repository into the `data/` directory at the root of this project:

```bash
cp Boston-Marathon-Data-Project/*.csv data/
```

The `data/` directory should contain files like:

```
data/
  results1897.csv
  results1898.csv
  ...
  results2019.csv
  results1993_includes-wheelchair.csv
  results1995_includes-wheelchair.csv
  results2013_without-diverted.csv
```

The wheelchair and diverted variants are automatically skipped during cleaning.

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn scipy statsmodels scikit-posthocs
```

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

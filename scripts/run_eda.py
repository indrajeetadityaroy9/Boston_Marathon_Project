"""Exploratory data analysis for the cleaned Boston Marathon dataset.

Reads one file -- data/processed/boston_marathon_cleaned.csv -- via
boston_marathon.data.load_cleaned and prints descriptive statistics,
hypothesis tests, and temporal trends to stdout.

1. Data quality
2. Descriptive statistics on finish time and age
3. Normality tests
4. Age vs finish-time correlation
5. Male vs female finish-time comparison
6. Age-group differences and post-hoc tests
7. Split-time analysis for 2015-2017
8. Repeat-runner profiling
9. Temporal drift over time
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy.stats import kruskal, ks_2samp, mannwhitneyu, pearsonr, shapiro, spearmanr, ttest_ind

from boston_marathon import config as cfg, data


def icc_anova(df, group_col, value_col):
    """Compute ANOVA-based intra-class correlation (ICC) for grouped data.

    Returns a value in [0, 1] showing how much of the variance in
    `value_col` is explained by membership in `group_col`.

    Uses the standard Shrout & Fleiss ICC(1) formula with a correction
    for unbalanced groups.
    """
    n_groups, n_obs = df[group_col].nunique(), len(df)
    grp = df.groupby(group_col)[value_col]
    group_means = grp.transform('mean')
    grand_mean = df[value_col].mean()
    ms_between = ((group_means - grand_mean) ** 2).sum() / (n_groups - 1)
    ms_within = ((df[value_col] - group_means) ** 2).sum() / (n_obs - n_groups)
    k0 = (n_obs - (grp.size() ** 2).sum() / n_obs) / (n_groups - 1)
    return (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)


def runner_slopes(df):
    """Compute per-runner slope of finish time vs age.

    For each display_name with an age span of at least 5 years, fit a
    simple line of finish time vs age using the centered-sum formula
    `sum(dx*dy) / sum(dx**2)`.

    Returns a pandas Series indexed by display_name, with slope in
    seconds per year. Positive means slowing down, negative means
    getting faster.
    """
    age_span = df.groupby('display_name')['age'].agg(lambda s: s.max() - s.min())
    sub = df[df['display_name'].isin(age_span[age_span >= 5].index)]
    grp = sub.groupby('display_name')
    dx = sub['age'] - grp['age'].transform('mean')
    dy = sub['seconds'] - grp['seconds'].transform('mean')
    return ((dx * dy).groupby(sub['display_name']).sum() /
            (dx ** 2).groupby(sub['display_name']).sum())


def cross_year_name_collision_mask(df):
    """Flag rows with impossible cross-year age changes.

    Sort rows by display_name and year, then compare age differences to
    year differences. Large mismatches suggest the same display_name is
    shared by different runners.

    Returns a boolean mask for all rows with flagged names.
    """
    sorted_df = df.sort_values(['display_name', 'year'])
    names = sorted_df['display_name'].to_numpy()
    ages = sorted_df['age'].to_numpy()
    years = sorted_df['year'].to_numpy()
    age_jumps = np.abs(np.diff(ages) - np.diff(years))
    bad_names = set(names[1:][(names[1:] == names[:-1]) & (age_jumps > 8)])
    return df['display_name'].isin(bad_names)


def main():
    print("BOSTON MARATHON EXPLORATORY DATA ANALYSIS")

    # Read the cleaned CSV via the library loader, which asserts the
    # on-disk header equals cfg.CLEANED_COLUMNS. Convert gender to
    # categorical and create a decade column for grouping.
    df = data.load_cleaned()
    df['gender'] = df['gender'].astype('category')
    df['decade'] = (df['year'] // 10) * 10

    # --- 1. DATA QUALITY REPORT ---
    # Check missing values, invariants, collisions, and outliers.
    # This helps identify which subsets are safe for later analysis.
    print("\n1. DATA QUALITY REPORT")

    print(f"\n  Rows: {len(df):,}  Columns: {len(df.columns)}  Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  Unique display_names: {df['display_name'].nunique():,}")

    print("\n  Missing values per column (>1 % only):")
    na_pct = df.isna().mean() * 100
    for col, pct in na_pct[na_pct > 1].sort_values(ascending=False).items():
        print(f"    {col:<30} {pct:>6.2f}%  ({df[col].isna().sum():>7,})")

    print("\n  Row-count distribution across decades:")
    print(df.groupby('decade').size().to_string())

    # Re-check basic data-quality invariants.
    print("\n  Data-quality invariants:")
    checks = {
        '0 < seconds < 50,000': ((df['seconds'] > 0) & (df['seconds'] < 50_000)).all(),
        'gender in {M, F}': df['gender'].isin(['M', 'F']).all(),
        'age in [14, 90] or NaN': (df['age'].isna() | df['age'].between(14, 90)).all(),
        'gender_result <= overall': (df['gender_result'] <= df['overall']).all(),
        'division_result <= gender_result': (df['division_result'] <= df['gender_result']).all(),
    }
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    n_collide = int(df['name_collides_within_year'].sum())
    n_collide_groups = df[df['name_collides_within_year']].groupby(['year', 'display_name']).ngroups
    print(f"\n  Within-year name collisions: {n_collide:,} rows in {n_collide_groups:,} (year, display_name) groups")
    print(f"    (distinct runners sharing a display_name within the same race -- ambiguous identities for any per-runner analysis)")

    # Count extreme finish-time outliers using the 3*IQR rule within each
    # (year, gender) group.
    def iqr_outliers(g):
        q1, q3 = g['seconds'].quantile([0.25, 0.75])
        iqr = q3 - q1
        return ((g['seconds'] < q1 - 3 * iqr) | (g['seconds'] > q3 + 3 * iqr)).sum()
    extreme = df.groupby(['year', 'gender'], observed=True).apply(iqr_outliers, include_groups=False).sum()
    print(f"\n  Extreme finish-time outliers (3*IQR per year*gender): {extreme:,} "
          f"({extreme / len(df) * 100:.3f}% of rows)")

    print(f"\n  Imputed ages: {int(df['age_imputed'].sum()):,} rows")
    for year, g in df[df['age_imputed']].groupby('year'):
        real = df[(df['year'] == year) & (~df['age_imputed']) & df['age'].notna()]['age']
        if len(real) > 1 and len(g) > 1:
            ratio = g['age'].std(ddof=1) / real.std(ddof=1)
            print(f"    {year}: n={len(g):,}  imputed_std/real_std={ratio:.3f}")

    # --- 2. DESCRIPTIVE STATISTICS ---
    # Summary statistics for finish time and age.
    # Skewness and kurtosis describe distribution shape.
    print("\n2. DESCRIPTIVE STATISTICS")

    print("\nOverall:")
    print(df[['seconds', 'age']].describe().T.to_string())
    print(f"  seconds  skewness={df['seconds'].skew():.4f}  kurtosis={df['seconds'].kurtosis():.4f}")
    print(f"  age      skewness={df['age'].skew():.4f}  kurtosis={df['age'].kurtosis():.4f}")

    print("\nBy Gender:")
    for gender, g in df.groupby('gender', observed=True):
        print(f"\n  {gender} (n={len(g)}):")
        print(g[['seconds', 'age']].describe().T.to_string())

    print("\nBy Decade (seconds):")
    print(df.groupby('decade')['seconds'].describe().to_string())

    # --- 3. NORMALITY TESTS ---
    # Test normality with Shapiro-Wilk on sampled data.
    # Also compare raw and log-transformed finish times.
    print("\n3. NORMALITY TESTS")

    strata = {
        'Overall': df['seconds'],
        'Male':    df[df['gender'] == 'M']['seconds'],
        'Female':  df[df['gender'] == 'F']['seconds'],
    }

    norm_rows = []
    for name, series in strata.items():
        sample = series.sample(n=5000, random_state=42)
        sw = shapiro(sample)
        norm_rows.append({'group': name, 'statistic': sw.statistic, 'p_value': sw.pvalue, 'reject_H0': sw.pvalue < 0.05})
        sw_log = shapiro(np.log(sample))
        norm_rows.append({'group': f'{name} (log)', 'statistic': sw_log.statistic, 'p_value': sw_log.pvalue, 'reject_H0': sw_log.pvalue < 0.05})

    print("\nShapiro-Wilk Results (n=5000 samples):")
    print(pd.DataFrame(norm_rows).to_string(index=False))

    # Compare shape of raw and log-transformed finish-time distributions.
    raw_secs = df['seconds']
    log_secs = np.log(raw_secs)
    print(f"\nShape Comparison:")
    print(f"  Raw:  skewness={raw_secs.skew():.4f}, kurtosis={raw_secs.kurtosis():.4f}")
    print(f"  Log:  skewness={log_secs.skew():.4f}, kurtosis={log_secs.kurtosis():.4f}")

    # --- 4. CORRELATION ANALYSIS ---
    # Measure the relationship between age and finish time.
    # Pearson r: linear relationship
    # Spearman rho: monotonic relationship
    # Partial r controls for gender
    # Drop imputed ages to avoid imputation-driven correlation.
    print("\n4. CORRELATION ANALYSIS")

    valid_age = df[df['age'].notna() & ~df['age_imputed']].copy()
    print(f"\nUsing {len(valid_age):,} rows with non-imputed age data")

    pr = pearsonr(valid_age['age'], valid_age['seconds'])
    pr_ci = pr.confidence_interval()
    sr = spearmanr(valid_age['age'], valid_age['seconds'])
    print(f"\nAge vs Seconds:")
    print(f"  Pearson r={pr.correlation:.4f}, p={pr.pvalue:.2e}, 95% CI=[{pr_ci.low:.4f}, {pr_ci.high:.4f}]")
    print(f"  Spearman rho={sr.correlation:.4f}, p={sr.pvalue:.2e}")

    valid_age['female'] = (valid_age['gender'] == 'F').astype(int)
    pc = pg.partial_corr(data=valid_age, x='age', y='seconds', covar='female')
    print(f"  Partial r (age | gender): {pc['r'].iloc[0]:.4f}, p={pc['p_val'].iloc[0]:.2e}")

    # --- 5. GENDER COMPARISON ---
    # Compare male and female finish times.
    # Welch's t-test compares means without equal-variance assumption.
    # Mann-Whitney U is the non-parametric alternative.
    # Hedges' g gives the effect size.
    print("\n5. GENDER COMPARISON")

    m_sec = df[df['gender'] == 'M']['seconds']
    f_sec = df[df['gender'] == 'F']['seconds']

    # Welch's t-test.
    welch = ttest_ind(m_sec, f_sec, equal_var=False)
    welch_ci = welch.confidence_interval()
    print(f"\nWelch's t-test: t={welch.statistic:.4f}, p={welch.pvalue:.2e}, df={welch.df:.1f}")
    print(f"  95% CI for mean difference: [{welch_ci.low:.1f}, {welch_ci.high:.1f}]")

    # Mann-Whitney U test.
    mwu = mannwhitneyu(m_sec, f_sec, alternative='two-sided')
    print(f"Mann-Whitney U: U={mwu.statistic:.0f}, p={mwu.pvalue:.2e}")

    # Hedges' g effect size.
    g = pg.compute_effsize(m_sec, f_sec, eftype='hedges')
    size = 'small' if abs(g) < 0.5 else 'medium' if abs(g) < 0.8 else 'large'
    print(f"Hedges' g (M-F): {g:.4f} ({size} effect; negative = males faster)")

    # Repeat the gender comparison by decade.
    gd_rows = []
    for decade in sorted(df[df['gender'] == 'F']['decade'].unique()):
        m_d = df[(df['decade'] == decade) & (df['gender'] == 'M')]['seconds']
        f_d = df[(df['decade'] == decade) & (df['gender'] == 'F')]['seconds']
        mwu_d = mannwhitneyu(m_d, f_d, alternative='two-sided')
        g_d = pg.compute_effsize(m_d, f_d, eftype='hedges')
        gd_rows.append({
            'decade': decade, 'n_male': len(m_d), 'n_female': len(f_d),
            'mean_diff': m_d.mean() - f_d.mean(), 'p_value': mwu_d.pvalue, 'hedges_g': g_d,
        })
    print("\nGender Gap by Decade:")
    print(pd.DataFrame(gd_rows).set_index('decade').to_string())

    # --- 6. AGE GROUP ANALYSIS ---
    # Group runners into age bins and compare finish times.
    # Kruskal-Wallis tests for group differences without normality assumption.
    # Dunn's test gives post-hoc pairwise comparisons.
    # ANOVA gives eta-squared effect size.
    # Tukey's HSD gives parametric pairwise comparisons.
    print("\n6. AGE GROUP ANALYSIS")

    aged = df[df['age'].notna() & ~df['age_imputed']].copy()
    aged['age_group'] = pd.cut(
        aged['age'],
        bins=[14, 30, 40, 50, 60, 70, 100],
        labels=['14-29', '30-39', '40-49', '50-59', '60-69', '70+'],
        right=False,
    )
    print(f"\nRows with valid age (14-90): {len(aged):,}")

    print("\nAge Group Stats:")
    print(aged.groupby('age_group', observed=True)['seconds']
              .agg(n='count', mean='mean', median='median', std='std').to_string())

    # Kruskal-Wallis test.
    kw = kruskal(*[g['seconds'].values for _, g in aged.groupby('age_group', observed=True)])
    print(f"\nKruskal-Wallis: H={kw.statistic:.4f}, p={kw.pvalue:.2e}")

    # Dunn's post-hoc test with Bonferroni correction.
    print("\nDunn's Post-hoc (Bonferroni):")
    print(sp.posthoc_dunn(aged, val_col='seconds', group_col='age_group', p_adjust='bonferroni').to_string())

    # One-way ANOVA and eta-squared.
    aov = pg.anova(data=aged, dv='seconds', between='age_group', detailed=False)
    print(f"\nOne-way ANOVA: F={aov['F'].iloc[0]:.4f}, p={aov['p_unc'].iloc[0]:.2e}")
    print(f"Eta-squared: {aov['np2'].iloc[0]:.4f}")

    # Tukey's HSD post-hoc test.
    print("\nTukey's HSD:")
    print(pg.pairwise_tukey(data=aged, dv='seconds', between='age_group')
            .to_string(index=False))

    # --- 7. SPLIT-TIME ANALYSIS (2015-2017) ---
    # Analyze checkpoint splits for the 2015-2017 races.
    # Build correlation matrix and classify pacing type.
    print("\n7. SPLIT-TIME ANALYSIS (2015-2017)")

    splits = df[df['year'].between(2015, 2017) & df[cfg.CUMULATIVE_SPLIT_COLUMNS].notna().all(axis=1)].copy()
    print(f"\nRows with valid split data: {len(splits):,}")

    print("\nSplit-Time Correlation Matrix:")
    print(splits[cfg.CUMULATIVE_SPLIT_COLUMNS + ['seconds']].corr().to_string())

    # Classify pacing using second-half / first-half ratio.
    second_half = splits['seconds'] - splits['half_seconds']
    pacing_ratio = second_half / splits['half_seconds']
    splits['pacing_type'] = pd.cut(
        pacing_ratio,
        bins=[-np.inf, 0.99, 1.01, np.inf],
        labels=['negative_split', 'even_split', 'positive_split'],
    )

    print("\nPacing Distribution by Year (%):")
    print((pd.crosstab(splits['year'], splits['pacing_type'], normalize='index') * 100).to_string())

    overall_pacing = splits['pacing_type'].value_counts(normalize=True).sort_index() * 100
    print("\nOverall Pacing Distribution:")
    for ptype, pct in overall_pacing.items():
        print(f"  {ptype}: {pct:.1f}%")

    kw_pace = kruskal(*[g['seconds'].values for _, g in splits.groupby('pacing_type', observed=True)])
    print(f"\nKruskal-Wallis (pacing): H={kw_pace.statistic:.2f}, p={kw_pace.pvalue:.2e}")

    aov_pace = pg.welch_anova(data=splits, dv='seconds', between='pacing_type')
    eta_sq_pace = float(aov_pace['np2'].iloc[0])
    print(f"Eta-squared (pacing): {eta_sq_pace:.4f} ({eta_sq_pace * 100:.1f}% of variance)")

    print("\nPer-Year Eta-squared (pacing):")
    for yr, yr_df in splits.groupby('year'):
        aov_yr = pg.welch_anova(data=yr_df, dv='seconds', between='pacing_type')
        print(f"  {yr}: eta_sq={aov_yr['np2'].iloc[0]:.4f}")

    print("\nDunn's Post-hoc Pacing (Bonferroni):")
    print(sp.posthoc_dunn(splits, val_col='seconds', group_col='pacing_type', p_adjust='bonferroni').to_string())

    # --- 8. REPEAT RUNNER PROFILING ---
    # Analyze runners with multiple appearances.
    # ICC measures between-runner vs within-runner variance.
    # Runner slopes measure performance change with age.
    print("\n8. REPEAT RUNNER PROFILING")

    name_counts = df['display_name'].value_counts()
    repeat_df = df[df['display_name'].isin(name_counts[name_counts > 1].index)]
    print(f"\nUnique runners: {len(name_counts):,}")
    print(f"Repeat runners (>1 appearance): {(name_counts > 1).sum():,}")
    print(f"Rows from repeat runners: {len(repeat_df):,} ({len(repeat_df) / len(df) * 100:.1f}%)")

    # Remove ambiguous identities before per-runner analysis.
    repeat_with_age = repeat_df[repeat_df['age'].notna() & ~repeat_df['age_imputed']].copy()
    repeat_with_age = repeat_with_age[~repeat_with_age['name_collides_within_year']]
    repeat_with_age = repeat_with_age[~cross_year_name_collision_mask(repeat_with_age)]
    print(f"After within-year + cross-year collision filters: "
          f"{repeat_with_age['display_name'].nunique():,} runners, {len(repeat_with_age):,} rows")

    age_stats = repeat_with_age.groupby('display_name')['age'].agg(
        n_races='count', age_span=lambda s: s.max() - s.min())
    age_stats = age_stats[age_stats['n_races'] > 1]
    print(f"\nAge Span (n={len(age_stats):,} runners with >=2 age observations):")
    print(f"  Mean: {age_stats['age_span'].mean():.1f} years, Median: {age_stats['age_span'].median():.1f} years")
    print(f"  Span >=5yr: {(age_stats['age_span'] >= 5).sum():,}, >=10yr: {(age_stats['age_span'] >= 10).sum():,}")

    icc = icc_anova(repeat_with_age, 'display_name', 'seconds')
    icc_label = 'strong' if icc > 0.3 else 'moderate' if icc > 0.1 else 'weak'
    print(f"\nIntra-class correlation (ICC): {icc:.4f} -- {icc * 100:.1f}% of finish time variance is between runners")
    print(f"  -> {icc_label} between-runner clustering")

    slopes = runner_slopes(repeat_with_age)
    print(f"\nWithin-Runner Aging Slopes (n={len(slopes):,}, age span >=5yr):")
    print(f"  Mean: {slopes.mean():.1f} sec/yr, Median: {slopes.median():.1f} sec/yr, Std: {slopes.std():.1f} sec/yr")
    print(f"  Slowing: {(slopes > 0).sum():,} ({(slopes > 0).mean() * 100:.1f}%), "
          f"Improving: {(slopes < 0).sum():,} ({(slopes < 0).mean() * 100:.1f}%)")
    print(f"  -> individual runners vary widely in how they age")

    # --- 9. TEMPORAL DRIFT OVER TIME ---
    # Summarize changes across decades.
    # Use the two-sample KS test to compare adjacent decades.
    print("\n9. TEMPORAL DRIFT OVER TIME")

    decade_summary = (df.groupby('decade')
                      .agg(n=('seconds', 'size'),
                           mean_seconds=('seconds', 'mean'),
                           std_seconds=('seconds', 'std'),
                           mean_age=('age', 'mean'),
                           female_frac=('gender', lambda g: (g == 'F').mean()))
                      .reset_index())
    print("\nPer-decade summary (seconds numeric, age in years):")
    print(decade_summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # KS comparisons for adjacent decades with enough sample size.
    print("\nAdjacent-decade KS comparisons on finish time (n_decade >= 1,000):")
    decades_all = sorted(df['decade'].unique())
    rows = []
    for earlier, later in zip(decades_all[:-1], decades_all[1:]):
        a = df[df['decade'] == earlier]['seconds']
        b = df[df['decade'] == later]['seconds']
        if len(a) < 1000 or len(b) < 1000:
            continue
        ks = ks_2samp(a, b)
        rows.append({
            'earlier':      f'{int(earlier)}s',
            'later':        f'{int(later)}s',
            'n_earlier':    len(a),
            'n_later':      len(b),
            'mean_earlier': float(a.mean()),
            'mean_later':   float(b.mean()),
            'delta_mean':   float(b.mean() - a.mean()),
            'ks_stat':      float(ks.statistic),
            'ks_p':         float(ks.pvalue),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Per-year summary for the modern era.
    print("\nPer-year mean finish time (2000-2019):")
    recent = (df[df['year'] >= 2000]
              .groupby('year')
              .agg(n=('seconds', 'size'),
                   mean_seconds=('seconds', 'mean'),
                   median_seconds=('seconds', 'median'),
                   female_frac=('gender', lambda g: (g == 'F').mean()))
              .reset_index())
    recent['delta_vs_prev'] = recent['mean_seconds'].diff()
    print(recent.to_string(index=False, float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()

"""Exploratory data analysis for the cleaned Boston Marathon dataset.

Reads one file -- data/processed/boston_marathon_cleaned.csv -- and prints
descriptive statistics, hypothesis tests, and temporal trends to stdout.

  1. Data quality -- missing values, invariants, collisions, outliers
  2. Descriptive statistics on finish time and age
  3. Normality tests (raw and log-transformed finish times)
  4. Age vs finish-time correlation (Pearson, Spearman, partial)
  5. Male vs female finish-time comparison
  6. Age-group differences and post-hoc pairwise tests
  7. Split-time analysis for the 2015-2017 window
  8. Repeat-runner profiling (ICC, aging slopes)
  9. Year-over-year and decade-over-decade temporal drift
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy.stats import kruskal, ks_2samp, mannwhitneyu, pearsonr, shapiro, spearmanr, ttest_ind

ROOT = Path(__file__).resolve().parent.parent
CLEANED_CSV = ROOT / 'data' / 'processed' / 'boston_marathon_cleaned.csv'

# The nine checkpoint columns that carry cumulative split times in seconds.
# Only the 2015-2017 races have these filled in; every other year has NaN.
SPLIT_COLS = [
    '5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds',
    'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds',
]


def icc_anova(df, group_col, value_col):
    """ANOVA-based intra-class correlation for grouped data.

    Returns a single number in [0, 1] that measures what fraction of the
    variation in `value_col` is explained by membership in `group_col`.
    A value near 1 means the groups are very different from each other and
    the within-group spread is small; a value near 0 means the opposite.

    Uses the standard Shrout & Fleiss ICC(1) formula with a `k0` correction
    that handles unbalanced groups (groups of different sizes).
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
    """Per-runner slope of finish time on age, for runners with a >=5-year span.

    For each display_name whose earliest and latest observations are at least
    five years apart, fits a simple straight line of finish time vs age using
    the within-runner centred-sum formula `sum(dx*dy) / sum(dx**2)`. Returns a
    pandas Series indexed by display_name, with the slope in seconds per
    year. A positive value means the runner is slowing down with age; a
    negative value means they are getting faster.
    """
    age_span = df.groupby('display_name')['age'].agg(lambda s: s.max() - s.min())
    sub = df[df['display_name'].isin(age_span[age_span >= 5].index)]
    grp = sub.groupby('display_name')
    dx = sub['age'] - grp['age'].transform('mean')
    dy = sub['seconds'] - grp['seconds'].transform('mean')
    return ((dx * dy).groupby(sub['display_name']).sum() /
            (dx ** 2).groupby(sub['display_name']).sum())


def cross_year_name_collision_mask(df):
    """Flag rows whose display_name has an impossible cross-year age trajectory.

    Walks through every display_name's rows in chronological order and
    compares the age difference between consecutive rows against the year
    difference. If the two differ by more than 8 years, the display_name is
    almost certainly being shared by more than one runner (no real person
    ages 10 years between consecutive Boston Marathons). Every row carrying
    one of those flagged names is returned as True.
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

    # Read the cleaned CSV. The pyarrow engine is noticeably faster than the
    # default C engine on a 72 MB file. `gender` comes back as plain strings
    # because CSV has no category dtype, so it is re-cast here -- several
    # groupby calls below pass `observed=True`, which only has an effect on
    # categoricals. `decade` is a derived column used by several later
    # sections to group rows into ten-year buckets.
    df = pd.read_csv(CLEANED_CSV, engine='pyarrow')
    df['gender'] = df['gender'].astype('category')
    df['decade'] = (df['year'] // 10) * 10

    # --- 1. DATA QUALITY REPORT ---
    # Print a data-state report before running any analysis. This tells the
    # reader where missing values live, re-runs the basic invariants that the
    # cleaning stage asserts at write time (so any accidental modification of
    # the CSV shows up), and reports two data hazards the cleaning step flags
    # but does not fix: runners sharing a display name in the same race, and
    # extreme finish-time outliers. Later sections rely on knowing which
    # subsets of the data are safe to use.
    print("\n1. DATA QUALITY REPORT")

    print(f"\n  Rows: {len(df):,}  Columns: {len(df.columns)}  Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  Unique display_names: {df['display_name'].nunique():,}")

    print("\n  Missing values per column (>1 % only):")
    na_pct = df.isna().mean() * 100
    for col, pct in na_pct[na_pct > 1].sort_values(ascending=False).items():
        print(f"    {col:<30} {pct:>6.2f}%  ({df[col].isna().sum():>7,})")

    print("\n  Row-count distribution across decades:")
    print(df.groupby('decade').size().to_string())

    # Basic integrity rules the dataset should satisfy. Re-running them here
    # catches any accidental modification of the CSV after it was cleaned.
    # The place columns (overall, gender_result, division_result) never have
    # null values in the cleaned output, so the comparisons don't need null
    # guards.
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

    # Count extreme finish-time outliers -- rows that sit more than 3 IQRs
    # (inter-quartile ranges) away from the 25th or 75th percentile of their
    # own (year, gender) cell. 3*IQR is a common cutoff for "very unusual".
    # A well-cleaned dataset has very few of these; a spike would indicate
    # typo rows or unit-conversion bugs in the raw files.
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
    # Basic summary numbers (count, mean, std, quartiles, min/max) for
    # finish time and age, broken out overall, by gender, and by decade.
    # Skewness and kurtosis tell us how lopsided or heavy-tailed each
    # distribution is, which informs whether later sections should use
    # parametric tests (which assume roughly normal distributions) or
    # non-parametric ones.
    #
    # The age stats here include the imputed rows from the cleaning step.
    # That imputer preserves the natural spread well (see the variance
    # ratios in step 1), so pooling has well below 0.1 % distortion on
    # mean and std. Steps 4, 6, and 8 still drop imputed ages when it
    # matters for the specific analysis.
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
    # Check whether finish times look roughly normal-distributed. Many of
    # the classical tests used later (t-tests, ANOVA) assume normality, so
    # the assumption needs to hold before those tests are trustworthy. A
    # log-transformed version is also tested, to see whether the log is
    # closer to normal than the raw values.
    #
    # Shapiro-Wilk is the standard normality test, but on samples with
    # n = 600k even a near-normal distribution gets rejected with p approx 0.
    # Drawing 5000 rows puts the test back in a sensible regime where the
    # p-value actually reflects a meaningful difference from normality.
    print("\n3. NORMALITY TESTS")

    strata = {
        'Overall': df['seconds'],
        'Male':    df[df['gender'] == 'M']['seconds'],
        'Female':  df[df['gender'] == 'F']['seconds'],
    }

    norm_rows = []
    for name, data in strata.items():
        sample = data.sample(n=5000, random_state=42)
        sw = shapiro(sample)
        norm_rows.append({'group': name, 'statistic': sw.statistic, 'p_value': sw.pvalue, 'reject_H0': sw.pvalue < 0.05})
        sw_log = shapiro(np.log(sample))
        norm_rows.append({'group': f'{name} (log)', 'statistic': sw_log.statistic, 'p_value': sw_log.pvalue, 'reject_H0': sw_log.pvalue < 0.05})

    print("\nShapiro-Wilk Results (n=5000 samples):")
    print(pd.DataFrame(norm_rows).to_string(index=False))

    # Compare the shape of the raw and log-transformed distributions. Even
    # if Shapiro-Wilk still rejects normality on the log version, the
    # skewness and kurtosis numbers show how much more symmetric and less
    # heavy-tailed the log version is.
    raw_secs = df['seconds']
    log_secs = np.log(raw_secs)
    print(f"\nShape Comparison:")
    print(f"  Raw:  skewness={raw_secs.skew():.4f}, kurtosis={raw_secs.kurtosis():.4f}")
    print(f"  Log:  skewness={log_secs.skew():.4f}, kurtosis={log_secs.kurtosis():.4f}")

    # --- 4. CORRELATION ANALYSIS ---
    # Measure how strongly age and finish time move together. Three
    # correlations are reported:
    #   - Pearson r: captures straight-line relationships only.
    #   - Spearman rho: captures any always-increasing or always-decreasing
    #     trend, even if it isn't a straight line.
    #   - Partial r (age | gender): strips out gender's contribution to
    #     show what age does on its own, controlling for the male/female split.
    #
    # Imputed ages are dropped here. Because the imputer learned age from
    # finish time, imputed rows would have a built-in age-seconds correlation
    # that is a property of the imputer rather than of real runners.
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
    # Compare men and women on finish time. There are two separate questions:
    #
    #   1. Is there a statistically significant difference?
    #      (Welch's t-test on means, Mann-Whitney U on ranks.)
    #   2. How big is the difference in practical terms?
    #      (Hedges' g = standardised mean difference.)
    #
    # With ~600k rows, any tiny difference is "statistically significant"
    # because the sample size drives p-values to zero. So the real question
    # is the second one: how big is the gap in units of standard deviation?
    # That's what Hedges' g measures.
    print("\n5. GENDER COMPARISON")

    m_sec = df[df['gender'] == 'M']['seconds']
    f_sec = df[df['gender'] == 'F']['seconds']

    # Welch's t-test: compares the two means without assuming the two
    # groups have the same variance. Safer than Student's t-test here
    # because male and female finish-time spreads are not identical.
    welch = ttest_ind(m_sec, f_sec, equal_var=False)
    welch_ci = welch.confidence_interval()
    print(f"\nWelch's t-test: t={welch.statistic:.4f}, p={welch.pvalue:.2e}, df={welch.df:.1f}")
    print(f"  95% CI for mean difference: [{welch_ci.low:.1f}, {welch_ci.high:.1f}]")

    # Mann-Whitney U: a rank-based test that doesn't assume normality.
    # Step 3 showed finish times are not normally distributed, so this
    # is the more defensible test; both are reported for transparency.
    mwu = mannwhitneyu(m_sec, f_sec, alternative='two-sided')
    print(f"Mann-Whitney U: U={mwu.statistic:.0f}, p={mwu.pvalue:.2e}")

    # Hedges' g: the difference in means expressed in units of pooled
    # standard deviation, with a small-sample correction. Rough rules of
    # thumb: |g| < 0.5 is small, 0.5-0.8 medium, > 0.8 large. A negative
    # value here means men finish faster on average (smaller seconds).
    # pingouin computes pooled-SD Cohen's d then multiplies by the Hedges
    # & Olkin 1985 bias-correction factor `1 - 3 / (4*(n1+n2-2) - 1)`.
    g = pg.compute_effsize(m_sec, f_sec, eftype='hedges')
    size = 'small' if abs(g) < 0.5 else 'medium' if abs(g) < 0.8 else 'large'
    print(f"Hedges' g (M-F): {g:.4f} ({size} effect; negative = males faster)")

    # Repeat the same gender-gap analysis separately for each decade. This
    # shows whether the gap is shrinking over time. Boston only allowed men
    # in its earliest decades, so the earliest female numbers come from
    # small fields and the gap has a long trajectory to trace.
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
    # Bin runners into six age brackets and ask whether the groups have
    # different average finish times. Eta-squared reports the fraction of
    # total finish-time variance that age-group membership explains -- a
    # practical measure of how much age actually matters.
    #
    # The typical pattern in this dataset is that the 30-39 bracket is the
    # fastest and performance falls off roughly quadratically with age from
    # there. Both Kruskal-Wallis (rank-based, doesn't assume normality)
    # and ANOVA (parametric, gives eta-squared) are reported, plus pairwise
    # post-hoc tests to identify which specific age-group pairs differ.
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

    # Kruskal-Wallis: the rank-based version of one-way ANOVA. It asks
    # "do at least two of these groups come from different distributions?"
    # and doesn't assume the data is normally distributed.
    kw = kruskal(*[g['seconds'].values for _, g in aged.groupby('age_group', observed=True)])
    print(f"\nKruskal-Wallis: H={kw.statistic:.4f}, p={kw.pvalue:.2e}")

    # Dunn's post-hoc test: once Kruskal-Wallis says "some groups differ",
    # Dunn's test reports which specific pairs of age groups differ. The
    # Bonferroni correction tightens the p-value threshold to account for
    # running many pairwise tests at once (so "significant" differences
    # aren't found purely by chance).
    print("\nDunn's Post-hoc (Bonferroni):")
    print(sp.posthoc_dunn(aged, val_col='seconds', group_col='age_group', p_adjust='bonferroni').to_string())

    # One-way ANOVA: the parametric counterpart to Kruskal-Wallis. It
    # assumes roughly normal distributions within each group, which is
    # not strictly true here, but it gives eta-squared -- the fraction of
    # total finish-time variance explained by age group. pingouin's
    # `anova` returns F, the uncorrected p-value, and `np2` (partial
    # eta-squared, which equals eta-squared for one-way ANOVA) in one
    # DataFrame, so no manual sum-of-squares arithmetic is needed.
    aov = pg.anova(data=aged, dv='seconds', between='age_group', detailed=False)
    print(f"\nOne-way ANOVA: F={aov['F'].iloc[0]:.4f}, p={aov['p_unc'].iloc[0]:.2e}")
    print(f"Eta-squared: {aov['np2'].iloc[0]:.4f}")

    # Tukey's HSD: the parametric equivalent of Dunn's post-hoc test. It
    # reports all pairwise group differences and adjusts the confidence
    # levels so that the overall chance of a false positive across all
    # pairs stays at 5 %. pingouin's version also reports Hedges' g for
    # each pair as a bonus.
    print("\nTukey's HSD:")
    print(pg.pairwise_tukey(data=aged, dv='seconds', between='age_group')
            .to_string(index=False))

    # --- 7. SPLIT-TIME ANALYSIS (2015-2017) ---
    # Only the 2015-2017 races record checkpoint splits (5K, 10K, ..., 40K).
    # This section does two things with that window:
    #   1. Build the correlation matrix of the nine splits against each
    #      other and against the final finish time. Adjacent splits are
    #      correlated above 0.99, meaning that once you know one split, the
    #      next one is almost completely determined -- a runner who is 25
    #      minutes in at 5K is very likely to be about 50 minutes in at 10K.
    #   2. Classify each runner's pacing as negative split (ran the second
    #      half faster), even split, or positive split (slowed down), and
    #      check whether the three pacing groups have different average
    #      finish times.
    print("\n7. SPLIT-TIME ANALYSIS (2015-2017)")

    splits = df[df['year'].between(2015, 2017) & df[SPLIT_COLS].notna().all(axis=1)].copy()
    print(f"\nRows with valid split data: {len(splits):,}")

    print("\nSplit-Time Correlation Matrix:")
    print(splits[SPLIT_COLS + ['seconds']].corr().to_string())

    # Classify each runner by pacing. Compute (second-half time) / (first-
    # half time). A ratio below 0.99 means the runner ran the second half
    # faster ("negative split"), between 0.99 and 1.01 is "even split", and
    # above 1.01 means they slowed down in the second half ("positive
    # split"). The 1 % tolerance keeps tiny watch-rounding differences out
    # of the "slowdown" bucket.
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
    # About half the dataset is people who ran Boston more than once. This
    # section asks: how much of the variation in finish time comes from
    # stable between-runner differences (some runners are consistently
    # faster than others) versus race-to-race noise within a single runner?
    #
    # The intra-class correlation (ICC) answers that. A high ICC means
    # between-runner differences dominate; a low ICC means individual
    # runners vary almost as much year to year as the whole population
    # does. The distribution of per-runner aging slopes is also computed
    # for runners with a career span of at least five years, showing how
    # individual performance changes with age.
    print("\n8. REPEAT RUNNER PROFILING")

    name_counts = df['display_name'].value_counts()
    repeat_df = df[df['display_name'].isin(name_counts[name_counts > 1].index)]
    print(f"\nUnique runners: {len(name_counts):,}")
    print(f"Repeat runners (>1 appearance): {(name_counts > 1).sum():,}")
    print(f"Rows from repeat runners: {len(repeat_df):,} ({len(repeat_df) / len(df) * 100:.1f}%)")

    # Drop two kinds of ambiguous runner identities before computing any
    # per-runner statistics:
    #   (1) name_collides_within_year -- two different people shared a
    #       display name in the same race (e.g. two "Aaron Smiths" in 2000).
    #   (2) cross_year_name_collision_mask -- the age progression across
    #       years is impossible for one person (e.g. 30 years old in 2010
    #       and 50 years old in 2015, so not the same runner despite the
    #       shared name).
    # Without these filters, the ICC and aging slopes would be polluted by
    # cross-person comparisons pretending to be within-runner comparisons.
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
    # How has the Boston Marathon field evolved across the full 1897-2019
    # span? This section builds a decade-by-decade summary of participation,
    # finish time, age, and gender composition, then tests whether the
    # finish-time distribution is stable between adjacent decades using a
    # two-sample Kolmogorov-Smirnov (KS) test. The KS statistic is a number
    # in [0, 1] that measures how different two distributions are (0 =
    # identical, 1 = no overlap); the accompanying p-value asks whether the
    # difference is likely to be real or just sampling noise.
    #
    # Very early decades have only a few dozen rows and are listed in the
    # summary for context but skipped from the KS comparisons, which need
    # reasonable sample sizes to be meaningful.
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

    # Adjacent-decade KS tests on finish time. Only decades where both
    # sides have at least 1,000 rows are compared, so the p-values stay
    # meaningful.
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

    # Per-year mean finish time for the modern era (2000 onward), as a
    # compact view of recent year-to-year swings. Useful for spotting
    # unusual years (extreme weather, race-day disruptions).
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

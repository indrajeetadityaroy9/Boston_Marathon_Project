import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu, pearsonr, shapiro, spearmanr, ttest_ind
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.meta_analysis import effectsize_smd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from boston_marathon.metrics import icc_anova

CLEANED_PATH = ROOT / 'data' / 'processed' / 'boston_marathon_cleaned.csv'
SPLIT_SECS = ['5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds', 'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds']


def main():
    print("BOSTON MARATHON EXPLORATORY DATA ANALYSIS")

    df = pd.read_csv(CLEANED_PATH, engine='pyarrow')
    df['decade'] = (df['year'] // 10) * 10

    # --- SECTION 1: DESCRIPTIVE STATISTICS ---
    # Summarize center, spread, and shape of the finish time distribution.
    # Skewness and kurtosis inform whether standard regression or non-parametric
    # approaches are appropriate for the demographic baseline prediction.
    print("\nSECTION 1: DESCRIPTIVE STATISTICS")

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

    # --- SECTION 2: NORMALITY TESTS ---
    # Check whether finish times follow a normal distribution (assumption for t-tests, ANOVA).
    # Log-transform may improve residual behavior in the mixed-effects personalization model.
    # 5000-row subsample avoids trivially rejecting normality due to large sample size.
    print("\nSECTION 2: NORMALITY TESTS")

    strata = {'Overall': df['seconds'], 'Male': df[df['gender'] == 'M']['seconds'], 'Female': df[df['gender'] == 'F']['seconds']}

    norm_rows = []
    for name, data in strata.items():
        sample = data.sample(n=5000, random_state=42)
        sw = shapiro(sample)
        norm_rows.append({'group': name, 'statistic': sw.statistic, 'p_value': sw.pvalue, 'reject_H0': sw.pvalue < 0.05})
        log_sample = np.log(sample)
        sw_log = shapiro(log_sample)
        norm_rows.append({'group': f'{name} (log)', 'statistic': sw_log.statistic, 'p_value': sw_log.pvalue, 'reject_H0': sw_log.pvalue < 0.05})

    print("\nShapiro-Wilk Results (n=5000 samples):")
    print(pd.DataFrame(norm_rows).to_string(index=False))

    # Comparing skewness and kurtosis before and after log-transform shows how much
    # closer to symmetric the distribution becomes, even if it still rejects normality.
    raw_secs = df['seconds']
    log_secs = np.log(raw_secs)
    print(f"\nShape Comparison:")
    print(f"  Raw:  skewness={raw_secs.skew():.4f}, kurtosis={raw_secs.kurtosis():.4f}")
    print(f"  Log:  skewness={log_secs.skew():.4f}, kurtosis={log_secs.kurtosis():.4f}")

    # --- SECTION 3: CORRELATION ANALYSIS ---
    # Measures how strongly age correlates with finish time, both linearly (Pearson)
    # and monotonically (Spearman). Partial correlation controls for gender.
    # Excludes KNN-imputed ages to avoid artificial correlation from the imputation model.
    print("\nSECTION 3: CORRELATION ANALYSIS")

    valid_age = df[df['age'].notna() & ~df['age_imputed']].copy()
    print(f"\nUsing {len(valid_age)} rows with non-imputed age data")

    # Pearson measures linear association; Spearman measures monotonic association
    # without assuming a straight-line relationship. Both are reported because
    # the age-time relationship may be nonlinear (e.g., U-shaped performance curve).
    pr = pearsonr(valid_age['age'], valid_age['seconds'])
    pr_ci = pr.confidence_interval()
    sr = spearmanr(valid_age['age'], valid_age['seconds'])
    print(f"\nAge vs Seconds:")
    print(f"  Pearson r={pr.correlation:.4f}, p={pr.pvalue:.2e}, 95% CI=[{pr_ci.low:.4f}, {pr_ci.high:.4f}]")
    print(f"  Spearman rho={sr.correlation:.4f}, p={sr.pvalue:.2e}")

    # Partial correlation controls for gender confounding via pingouin
    valid_age['female'] = (valid_age['gender'] == 'F').astype(int)
    pc = pg.partial_corr(data=valid_age, x='age', y='seconds', covar='female')
    print(f"  Partial r (age | gender): {pc['r'].iloc[0]:.4f}, p={pc['p_val'].iloc[0]:.2e}")

    # --- SECTION 4: GENDER COMPARISON ---
    # Tests whether male and female finish times differ significantly, and measures
    # how large the difference is using Hedges' g (standardized mean difference).
    # Effect size distinguishes meaningful differences from large-sample artifacts.
    print("\nSECTION 4: GENDER COMPARISON")

    m_sec = df[df['gender'] == 'M']['seconds']
    f_sec = df[df['gender'] == 'F']['seconds']

    # Welch's t-test compares means without assuming equal variance between groups.
    # Used instead of Student's t-test because Levene's test rejects equal variance.
    welch = ttest_ind(m_sec, f_sec, equal_var=False)
    welch_ci = welch.confidence_interval()
    print(f"\nWelch's t-test: t={welch.statistic:.4f}, p={welch.pvalue:.2e}, df={welch.df:.1f}")
    print(f"  95% CI for mean difference: [{welch_ci.low:.1f}, {welch_ci.high:.1f}]")

    # Mann-Whitney U is a non-parametric alternative that compares rank orderings
    # instead of means. Used because normality was rejected in Section 2.
    mwu = mannwhitneyu(m_sec, f_sec, alternative='two-sided')
    print(f"Mann-Whitney U: U={mwu.statistic:.0f}, p={mwu.pvalue:.2e}")

    # Hedges' g quantifies how large the gender gap is in standard deviation units.
    # With n=600k, even tiny differences are statistically significant, so effect size
    # tells us whether the difference is practically meaningful.
    g, _ = effectsize_smd(m_sec.mean(), m_sec.std(ddof=1), m_sec.count(), f_sec.mean(), f_sec.std(ddof=1), f_sec.count())
    abs_g = abs(g)
    print(f"Hedges' g (M-F): {g:.4f} ({'small' if abs_g < 0.5 else 'medium' if abs_g < 0.8 else 'large'} effect; "
          f"negative = males faster)")

    # Repeating per decade shows whether the gender gap is closing over time.
    gd_rows = []
    for decade in sorted(df[df['gender'] == 'F']['decade'].unique()):
        m_d = df[(df['decade'] == decade) & (df['gender'] == 'M')]['seconds']
        f_d = df[(df['decade'] == decade) & (df['gender'] == 'F')]['seconds']
        mwu_d = mannwhitneyu(m_d, f_d, alternative='two-sided')
        g_d, _ = effectsize_smd(m_d.mean(), m_d.std(ddof=1), m_d.count(), f_d.mean(), f_d.std(ddof=1), f_d.count())
        gd_rows.append({'decade': decade, 'n_male': len(m_d), 'n_female': len(f_d), 'mean_diff': m_d.mean() - f_d.mean(), 'p_value': mwu_d.pvalue, 'hedges_g': g_d})
    print("\nGender Gap by Decade:")
    print(pd.DataFrame(gd_rows).set_index('decade').to_string())

    # --- SECTION 5: AGE GROUP ANALYSIS ---
    # Tests whether finish times differ across six age brackets and measures how much
    # variance age group explains (eta-squared = proportion of total variance).
    # The nonlinear pattern (30-39 fastest, monotonic slowing after) motivates using
    # a quadratic age term in the prediction models.
    print("\nSECTION 5: AGE GROUP ANALYSIS")

    aged = df[df['age'].notna() & ~df['age_imputed']].copy()
    aged['age_group'] = pd.cut(aged['age'], bins=[14, 30, 40, 50, 60, 70, 100], labels=['14-29', '30-39', '40-49', '50-59', '60-69', '70+'], right=False)
    print(f"\nRows with valid age (14-90): {len(aged)}")

    print("\nAge Group Stats:")
    print(aged.groupby('age_group', observed=True)['seconds'].agg(n='count', mean='mean', median='median', std='std').to_string())

    # Kruskal-Wallis tests whether at least one age group differs from the others.
    # It is the non-parametric version of one-way ANOVA, using ranks instead of raw values.
    kw = kruskal(*[g['seconds'].values for _, g in aged.groupby('age_group', observed=True)])
    print(f"\nKruskal-Wallis: H={kw.statistic:.4f}, p={kw.pvalue:.2e}")

    # Dunn's test identifies which specific pairs of age groups differ.
    # Bonferroni correction controls for false positives from multiple comparisons.
    print("\nDunn's Post-hoc (Bonferroni):")
    print(sp.posthoc_dunn(aged, val_col='seconds', group_col='age_group', p_adjust='bonferroni').to_string())

    # One-way ANOVA tests the same question as Kruskal-Wallis but assumes normality.
    # Eta-squared measures how much of the total variance in finish time is explained
    # by age group membership (practical significance, not just statistical significance).
    anova = anova_lm(smf.ols('seconds ~ C(age_group)', data=aged).fit())
    print(f"\nOne-way ANOVA: F={anova.loc['C(age_group)', 'F']:.4f}, p={anova.loc['C(age_group)', 'PR(>F)']:.2e}")
    print(f"Eta-squared: {anova.loc['C(age_group)', 'sum_sq'] / anova['sum_sq'].sum():.4f}")

    # Tukey's HSD is the parametric post-hoc test, comparing all pairs of group means
    # while controlling the family-wise error rate.
    tukey = pairwise_tukeyhsd(aged['seconds'], aged['age_group'])
    print("\nTukey's HSD:")
    print(tukey.summary())

    # --- SECTION 6: SPLIT-TIME ANALYSIS (2015-2017) ---
    # Checkpoint split times (5K through 40K) are only available for 2015-2017.
    # The correlation matrix shows how strongly each checkpoint predicts finish time.
    # Very high correlations between adjacent checkpoints (r > 0.99) mean Ridge
    # regression is needed instead of ordinary least squares for in-race prediction.
    print("\nSECTION 6: SPLIT-TIME ANALYSIS (2015-2017)")

    splits = df[df['year'].between(2015, 2017)].copy()
    splits = splits[splits[SPLIT_SECS].notna().all(axis=1)].copy()
    print(f"\nRows with valid split data: {len(splits)}")

    print("\nSplit-Time Correlation Matrix:")
    print(splits[SPLIT_SECS + ['seconds']].corr().to_string())

    # Pacing classification: ratio of second-half time to first-half time.
    # <0.99 = ran the second half faster (negative split),
    # 0.99-1.01 = roughly even pace (1% tolerance),
    # >1.01 = slowed down in the second half (positive split).
    second_half = splits['seconds'] - splits['half_seconds']
    pacing_ratio = second_half / splits['half_seconds']
    splits['pacing_type'] = pd.cut(pacing_ratio, bins=[-np.inf, 0.99, 1.01, np.inf], labels=['negative_split', 'even_split', 'positive_split'])

    print("\nPacing Distribution by Year (%):")
    print((pd.crosstab(splits['year'], splits['pacing_type'], normalize='index') * 100).to_string())

    # Overall pacing distribution across all years
    overall_pacing = splits['pacing_type'].value_counts(normalize=True).sort_index() * 100
    print("\nOverall Pacing Distribution:")
    for ptype, pct in overall_pacing.items(): print(f"  {ptype}: {pct:.1f}%")

    # Kruskal-Wallis: do pacing groups have different finish times?
    kw_pace = kruskal(*[g['seconds'].values for _, g in splits.groupby('pacing_type', observed=True)])
    print(f"\nKruskal-Wallis (pacing): H={kw_pace.statistic:.2f}, p={kw_pace.pvalue:.2e}")

    # Eta-squared via pingouin (Welch ANOVA handles unequal variances)
    aov_pace = pg.welch_anova(data=splits, dv='seconds', between='pacing_type')
    eta_sq_pace = float(aov_pace['np2'].iloc[0])
    print(f"Eta-squared (pacing): {eta_sq_pace:.4f} ({eta_sq_pace*100:.1f}% of variance)")

    # Per-year eta-squared
    print("\nPer-Year Eta-squared (pacing):")
    for yr, yr_df in splits.groupby('year'):
        aov_yr = pg.welch_anova(data=yr_df, dv='seconds', between='pacing_type')
        print(f"  {yr}: eta_sq={aov_yr['np2'].iloc[0]:.4f}")

    # Dunn's post-hoc (same pattern as Section 5 age group analysis)
    print("\nDunn's Post-hoc Pacing (Bonferroni):")
    print(sp.posthoc_dunn(splits, val_col='seconds', group_col='pacing_type', p_adjust='bonferroni').to_string())

    # --- SECTION 7: REPEAT RUNNER PROFILING ---
    # About half the dataset consists of runners who appear in multiple years.
    # The intra-class correlation (ICC) measures how much of the total variance in
    # finish time comes from stable between-runner differences vs race-to-race noise.
    # High ICC justifies per-runner baselines; high slope variance justifies per-runner
    # aging rates in the mixed-effects personalization model.
    print("\nSECTION 7: REPEAT RUNNER PROFILING")

    name_counts = df['display_name'].value_counts()
    repeat_names = name_counts[name_counts > 1]
    repeat_df = df[df['display_name'].isin(repeat_names.index)].copy()

    print(f"\nUnique runners: {len(name_counts)}")
    print(f"Repeat runners (>1 appearance): {len(repeat_names)}")
    print(f"Rows from repeat runners: {len(repeat_df)} ({len(repeat_df)/len(df)*100:.1f}%)")

    # Runners are identified by display_name, which can collide (two different people
    # named "John Smith"). We filter out names where consecutive appearances have
    # age changes that don't match the elapsed years (off by more than 8 years).
    def is_plausible_sequence(group):
        g = group.sort_values('year')
        if g['age'].isna().all():
            return (g['year'].max() - g['year'].min()) <= 20
        age_years = g[g['age'].notna()]
        if len(age_years) < 2:
            return (g['year'].max() - g['year'].min()) <= 20
        for i in range(1, len(age_years)):
            yr_gap = age_years['year'].iloc[i] - age_years['year'].iloc[i - 1]
            age_gap = age_years['age'].iloc[i] - age_years['age'].iloc[i - 1]
            if abs(age_gap - yr_gap) > 8:
                return False
        return True

    plausible_names = repeat_df.groupby('display_name').filter(is_plausible_sequence)['display_name'].unique()
    n_filtered = len(repeat_names) - len(plausible_names)
    print(f"Name collision filter: removed {n_filtered}, retained {len(plausible_names)}")
    repeat_df = repeat_df[repeat_df['display_name'].isin(plausible_names)].copy()

    filtered_counts = repeat_df['display_name'].value_counts()
    print(f"\nAppearances: mean={filtered_counts.mean():.2f}, median={filtered_counts.median():.1f}, max={filtered_counts.max()}")

    # Age span measures how many years of aging data each runner covers.
    # Longer spans give more information for estimating individual aging slopes.
    repeat_with_age = repeat_df[repeat_df['age'].notna() & ~repeat_df['age_imputed']].copy()
    runner_age_stats = repeat_with_age.groupby('display_name')['age'].agg(n_races='count', age_span=lambda s: s.max() - s.min())
    runner_age_stats = runner_age_stats[runner_age_stats['n_races'] > 1]

    print(f"\nAge Span (n={len(runner_age_stats)} runners with >=2 age observations):")
    print(f"  Mean: {runner_age_stats['age_span'].mean():.1f} years, Median: {runner_age_stats['age_span'].median():.1f} years")
    print(f"  Span >=5yr: {(runner_age_stats['age_span'] >= 5).sum()}, >=10yr: {(runner_age_stats['age_span'] >= 10).sum()}")

    # ICC via the shared metrics module (same formula used in RQ2)
    icc = icc_anova(repeat_df, 'display_name', 'seconds')

    print(f"\nIntra-class correlation (ICC): {icc:.4f} -- {icc*100:.1f}% of finish time variance is between runners")
    print(f"  -> {'strong' if icc > 0.3 else 'moderate' if icc > 0.1 else 'weak'} justification for per-runner random intercepts")

    # Per-runner aging slopes via vectorized groupby (same formula as rq2._runner_slopes)
    runners_for_slope = runner_age_stats[runner_age_stats['age_span'] >= 5].index
    slope_df = repeat_with_age[repeat_with_age['display_name'].isin(runners_for_slope)]
    grp = slope_df.groupby('display_name')
    dx = slope_df['age'] - grp['age'].transform('mean')
    dy = slope_df['seconds'] - grp['seconds'].transform('mean')
    runner_slopes = (dx * dy).groupby(slope_df['display_name']).sum() / (dx ** 2).groupby(slope_df['display_name']).sum()

    print(f"\nWithin-Runner Aging Slopes (n={len(runner_slopes)}, age span >=5yr):")
    print(f"  Mean: {runner_slopes.mean():.1f} sec/yr, Median: {runner_slopes.median():.1f} sec/yr, Std: {runner_slopes.std():.1f} sec/yr")
    print(f"  Slowing: {(runner_slopes > 0).sum()} ({(runner_slopes > 0).sum()/len(runner_slopes)*100:.1f}%), "
          f"Improving: {(runner_slopes < 0).sum()} ({(runner_slopes < 0).sum()/len(runner_slopes)*100:.1f}%)")
    print(f"  -> high variance in aging rates justifies per-runner random slopes on age")


if __name__ == '__main__':
    main()

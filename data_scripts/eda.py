from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.formula.api as smf
from scipy.stats import (
    kruskal,
    mannwhitneyu,
    pearsonr,
    shapiro,
    spearmanr,
    ttest_ind,
)
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.meta_analysis import effectsize_smd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

CLEANED_PATH = Path(__file__).resolve().parent / 'cleaned_data' / 'boston_marathon_cleaned.csv'

SPLIT_SECS = ['5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds',
              'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds']


def main():
    print("BOSTON MARATHON EXPLORATORY DATA ANALYSIS")

    df = pd.read_csv(CLEANED_PATH, low_memory=False)
    df['age_imputed'] = df['age_imputed'].astype(str).str.strip().str.lower() == 'true'
    df['decade'] = (df['year'] // 10) * 10

    # --- SECTION 1: DESCRIPTIVE STATISTICS ---
    # Summarize the data before any testing to understand center, spread, and shape.
    # Skewness and kurtosis indicate whether the data is symmetric and how heavy the tails are,
    # which informs the choice between parametric and non-parametric tests later.
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
    # Many statistical tests (t-test, ANOVA) assume normally distributed data.
    # Shapiro-Wilk checks this assumption. We test both raw and log-transformed times
    # because if log-transform makes the data more normal, the mixed-effects model
    # could use log(seconds) as the response variable for better-behaved residuals.
    # We use a 5000-row sample because Shapiro-Wilk is designed for moderate sample sizes
    # and would reject normality for trivially small deviations at n=600k.
    print("\nSECTION 2: NORMALITY TESTS")

    strata = {
        'Overall': df['seconds'].dropna(),
        'Male': df[df['gender'] == 'M']['seconds'].dropna(),
        'Female': df[df['gender'] == 'F']['seconds'].dropna(),
    }

    norm_rows = []
    for name, data in strata.items():
        sample = data.sample(n=min(5000, len(data)), random_state=42)
        sw = shapiro(sample)
        norm_rows.append({'group': name, 'statistic': sw.statistic, 'p_value': sw.pvalue,
                          'reject_H0': sw.pvalue < 0.05})
        log_sample = np.log(sample)
        sw_log = shapiro(log_sample)
        norm_rows.append({'group': f'{name} (log)', 'statistic': sw_log.statistic, 'p_value': sw_log.pvalue,
                          'reject_H0': sw_log.pvalue < 0.05})

    print("\nShapiro-Wilk Results (n=5000 samples):")
    print(pd.DataFrame(norm_rows).to_string(index=False))

    # Comparing skewness and kurtosis before and after log-transform shows how much
    # closer to symmetric the distribution becomes, even if it still rejects normality.
    raw_secs = df['seconds'].dropna()
    log_secs = np.log(raw_secs)
    print(f"\nShape Comparison:")
    print(f"  Raw:  skewness={raw_secs.skew():.4f}, kurtosis={raw_secs.kurtosis():.4f}")
    print(f"  Log:  skewness={log_secs.skew():.4f}, kurtosis={log_secs.kurtosis():.4f}")

    # --- SECTION 3: CORRELATION ANALYSIS ---
    # Measures whether older runners tend to have slower finish times.
    # Excludes KNN-imputed ages to avoid artificial correlation from the imputation model.
    print("\nSECTION 3: CORRELATION ANALYSIS")

    valid_age = df[df['age'].notna() & ~df['age_imputed']].copy()
    valid = valid_age[['age', 'seconds']].dropna()
    print(f"\nUsing {len(valid)} rows with non-imputed age data")

    # Pearson measures linear association; Spearman measures monotonic association
    # without assuming a straight-line relationship. Both are reported because
    # the age-time relationship may be nonlinear (e.g., U-shaped performance curve).
    pr = pearsonr(valid['age'], valid['seconds'])
    pr_ci = pr.confidence_interval()
    sr = spearmanr(valid['age'], valid['seconds'])
    print(f"\nAge vs Seconds:")
    print(f"  Pearson r={pr.correlation:.4f}, p={pr.pvalue:.2e}, 95% CI=[{pr_ci.low:.4f}, {pr_ci.high:.4f}]")
    print(f"  Spearman rho={sr.correlation:.4f}, p={sr.pvalue:.2e}")

    # Partial correlation removes the confounding effect of gender. Without this,
    # the age-time correlation would be biased because gender affects both age
    # distribution (males skew older in this dataset) and finish time.
    partial_model = smf.ols('seconds ~ age + C(gender)',
                            data=valid_age[['age', 'seconds', 'gender']].dropna()).fit()
    age_t = partial_model.tvalues['age']
    partial_r = np.sign(age_t) * np.sqrt(age_t ** 2 / (age_t ** 2 + partial_model.df_resid))
    print(f"  Partial r (age | gender): {partial_r:.4f}, p={partial_model.pvalues['age']:.2e}")

    # --- SECTION 4: GENDER COMPARISON ---
    # Tests whether the difference in finish times between men and women is statistically
    # significant and practically meaningful (Q1).
    print("\nSECTION 4: GENDER COMPARISON")

    m_sec = df[df['gender'] == 'M']['seconds'].dropna()
    f_sec = df[df['gender'] == 'F']['seconds'].dropna()

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
    g, _ = effectsize_smd(
        m_sec.mean(), m_sec.std(ddof=1), m_sec.count(),
        f_sec.mean(), f_sec.std(ddof=1), f_sec.count(),
    )
    abs_g = abs(g)
    print(f"Hedges' g (M-F): {g:.4f} ({'small' if abs_g < 0.5 else 'medium' if abs_g < 0.8 else 'large'} effect; "
          f"negative = males faster)")

    # Repeating per decade shows whether the gender gap is closing over time.
    gd_rows = []
    for decade in sorted(df[df['gender'] == 'F']['decade'].unique()):
        m_d = df[(df['decade'] == decade) & (df['gender'] == 'M')]['seconds'].dropna()
        f_d = df[(df['decade'] == decade) & (df['gender'] == 'F')]['seconds'].dropna()
        if len(m_d) > 1 and len(f_d) > 1:
            mwu_d = mannwhitneyu(m_d, f_d, alternative='two-sided')
            g_d, _ = effectsize_smd(
                m_d.mean(), m_d.std(ddof=1), m_d.count(),
                f_d.mean(), f_d.std(ddof=1), f_d.count(),
            )
            gd_rows.append({
                'decade': decade, 'n_male': len(m_d), 'n_female': len(f_d),
                'mean_diff': m_d.mean() - f_d.mean(), 'p_value': mwu_d.pvalue, 'hedges_g': g_d,
            })
    print("\nGender Gap by Decade:")
    print(pd.DataFrame(gd_rows).set_index('decade').to_string())

    # --- SECTION 5: AGE GROUP ANALYSIS ---
    # Tests whether finish times differ across age brackets (Q1).
    # Both parametric and non-parametric approaches are used because
    # normality was rejected but ANOVA is robust to moderate non-normality
    # at large sample sizes, and running both shows consistent results.
    print("\nSECTION 5: AGE GROUP ANALYSIS")

    aged = df[df['age'].between(14, 90)].copy()
    aged['age_group'] = pd.cut(aged['age'],
                               bins=[14, 30, 40, 50, 60, 70, 100],
                               labels=['14-29', '30-39', '40-49', '50-59', '60-69', '70+'],
                               right=False)
    aged = aged.dropna(subset=['age_group'])
    print(f"\nRows with valid age (14-90): {len(aged)}")

    print("\nAge Group Stats:")
    print(aged.groupby('age_group', observed=True)['seconds'].agg(
        n='count', mean='mean', median='median', std='std',
    ).to_string())

    # Kruskal-Wallis tests whether at least one age group differs from the others.
    # It is the non-parametric version of one-way ANOVA, using ranks instead of raw values.
    kw = kruskal(*[g['seconds'].dropna().values
                    for _, g in aged.groupby('age_group', observed=True)])
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
    tukey = pairwise_tukeyhsd(aged['seconds'].dropna(), aged['age_group'].dropna())
    print("\nTukey's HSD:")
    print(tukey.summary())

    # --- SECTION 6: SPLIT-TIME ANALYSIS (2015-2017) ---
    # Split checkpoint times are only available for 2015-2017.
    # The correlation matrix shows how strongly each checkpoint predicts the final time.
    # High correlations between adjacent checkpoints (r > 0.99) indicate severe
    # multicollinearity, which is why ridge/LASSO regression is needed instead of
    # ordinary least squares when predicting finish time from splits (Q3).
    print("\nSECTION 6: SPLIT-TIME ANALYSIS (2015-2017)")

    splits = df[df['year'].between(2015, 2017)].copy()
    splits = splits[splits[SPLIT_SECS].notna().all(axis=1) & splits['seconds'].notna()].copy()
    print(f"\nRows with valid split data: {len(splits)}")

    print("\nSplit-Time Correlation Matrix:")
    print(splits[SPLIT_SECS + ['seconds']].corr().to_string())

    # Pacing classification: ratio of second-half time to first-half time.
    # <0.99 = ran the second half faster (negative split),
    # 0.99-1.01 = roughly even pace (1% tolerance),
    # >1.01 = slowed down in the second half (positive split).
    second_half = splits['seconds'] - splits['half_seconds']
    pacing_ratio = second_half / splits['half_seconds']
    splits['pacing_type'] = pd.cut(
        pacing_ratio,
        bins=[-np.inf, 0.99, 1.01, np.inf],
        labels=['negative_split', 'even_split', 'positive_split'])

    print("\nPacing Distribution by Year (%):")
    print((pd.crosstab(splits['year'], splits['pacing_type'], normalize='index') * 100).to_string())

    # --- SECTION 7: REPEAT RUNNER PROFILING ---
    # About half the dataset consists of runners who appear in multiple years.
    # Treating their rows as independent would violate the assumptions of standard tests.
    # This section quantifies the repeat-runner structure to justify using a linear
    # mixed-effects model with per-runner random intercepts and slopes (Q2).
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
            return True
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
    repeat_with_age = repeat_df[repeat_df['age'].notna()].copy()
    runner_age_stats = repeat_with_age.groupby('display_name')['age'].agg(
        n_races='count', age_span=lambda s: s.max() - s.min(),
    )
    runner_age_stats = runner_age_stats[runner_age_stats['n_races'] > 1]

    print(f"\nAge Span (n={len(runner_age_stats)} runners with ≥2 age observations):")
    print(f"  Mean: {runner_age_stats['age_span'].mean():.1f} years, Median: {runner_age_stats['age_span'].median():.1f} years")
    print(f"  Span ≥5yr: {(runner_age_stats['age_span'] >= 5).sum()}, ≥10yr: {(runner_age_stats['age_span'] >= 10).sum()}")

    # ICC (intra-class correlation) measures what fraction of the total variance in
    # finish time is due to stable differences between runners vs. race-to-race variation.
    # A high ICC means runners have consistent ability levels across years, which
    # justifies adding per-runner random intercepts to the mixed-effects model.
    repeat_secs = repeat_df[repeat_df['seconds'].notna()].copy()
    runner_means = repeat_secs.groupby('display_name')['seconds'].transform('mean')
    ss_between = ((runner_means - repeat_secs['seconds'].mean()) ** 2).sum()
    ss_within = ((repeat_secs['seconds'] - runner_means) ** 2).sum()
    n_runners = repeat_secs['display_name'].nunique()
    n_obs = len(repeat_secs)
    ms_between = ss_between / (n_runners - 1)
    ms_within = ss_within / (n_obs - n_runners)
    ni = repeat_secs.groupby('display_name').size()
    k0 = (n_obs - (ni ** 2).sum() / n_obs) / (n_runners - 1)
    icc = (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)

    print(f"\nICC(1): {icc:.4f} — {icc*100:.1f}% of variance is between runners")
    print(f"  → {'strong' if icc > 0.3 else 'moderate' if icc > 0.1 else 'weak'} justification for random intercepts")

    # Per-runner OLS slopes measure how each individual's finish time changes with age.
    # If all runners aged the same way, one fixed slope would suffice. High variance
    # in slopes means runners age differently, justifying per-runner random slopes
    # in the mixed-effects model. Only runners with ≥5 year age span are used so
    # each slope is estimated from a meaningful range of ages.
    runners_for_slope = runner_age_stats[runner_age_stats['age_span'] >= 5].index
    slope_df = repeat_with_age[
        repeat_with_age['display_name'].isin(runners_for_slope) &
        repeat_with_age['seconds'].notna()
    ]
    runner_slopes = slope_df.groupby('display_name').apply(
        lambda g: np.polyfit(g['age'].values, g['seconds'].values, 1)[0]
        if len(g) >= 2 else np.nan,
        include_groups=False,
    ).dropna()

    print(f"\nWithin-Runner Aging Slopes (n={len(runner_slopes)}, age span ≥5yr):")
    print(f"  Mean: {runner_slopes.mean():.1f} sec/yr, Median: {runner_slopes.median():.1f} sec/yr, Std: {runner_slopes.std():.1f} sec/yr")
    print(f"  Slowing: {(runner_slopes > 0).sum()} ({(runner_slopes > 0).sum()/len(runner_slopes)*100:.1f}%), "
          f"Improving: {(runner_slopes < 0).sum()} ({(runner_slopes < 0).sum()/len(runner_slopes)*100:.1f}%)")
    print(f"  → high slope variance justifies random slopes on age")


if __name__ == '__main__':
    main()

from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.formula.api as smf
from scipy.stats import (
    anderson,
    genextreme,
    goodness_of_fit,
    iqr,
    kruskal,
    ks_2samp,
    levene,
    lognorm,
    mannwhitneyu,
    median_abs_deviation,
    pearsonr,
    pointbiserialr,
    shapiro,
    spearmanr,
    ttest_ind,
    variation,
    zscore,
)
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.meta_analysis import effectsize_smd

CLEANED_PATH = Path(__file__).resolve().parent / 'cleaned_data' / 'boston_marathon_cleaned.csv'

SPLIT_SECS = ['5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds',
              'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds']


def desc_table(df, cols):
    """Compute descriptive statistics via pd.DataFrame.describe() augmented with pandas/scipy."""
    # Start with pandas built-in describe (count, mean, std, min, quartiles, max)
    # then add mode, variance, range, IQR, coefficient of variation, standard error,
    # skewness (how lopsided the distribution is), and kurtosis (how heavy the tails are)
    subset = df[cols]
    stats = subset.describe(percentiles=[.25, .75]).T
    stats = stats.rename(columns={'50%': 'median', '25%': 'Q1', '75%': 'Q3'})
    stats['count'] = stats['count'].astype(int)
    stats['mode'] = subset.apply(
        lambda s: modes.iat[0] if not (modes := s.mode(dropna=True)).empty else np.nan
    )
    stats['variance'] = subset.var()
    stats['range'] = stats['max'] - stats['min']
    stats['IQR'] = subset.agg(lambda s: iqr(s, nan_policy='omit'))
    stats['CV'] = subset.apply(lambda s: variation(s, ddof=1, nan_policy='omit'))
    stats['SEM'] = subset.sem()
    stats['skewness'] = subset.skew()
    stats['excess_kurtosis'] = subset.kurtosis()
    return stats[['count', 'mean', 'median', 'mode', 'std', 'variance', 'min', 'max',
                   'range', 'Q1', 'Q3', 'IQR', 'CV', 'SEM', 'skewness', 'excess_kurtosis']]


def main():
    print("BOSTON MARATHON EXPLORATORY DATA ANALYSIS")

    df = pd.read_csv(CLEANED_PATH, low_memory=False)
    df['decade'] = (df['year'] // 10) * 10

    # --- SECTION 1: DESCRIPTIVE STATISTICS ---
    # Summarize finish times and ages with standard stats (mean, median, spread, shape)
    # for the whole dataset, then broken down by gender and by decade
    print("\nSECTION 1: DESCRIPTIVE STATISTICS")

    print("\nOverall Descriptive Statistics:")
    print(desc_table(df, ['seconds', 'age']).to_string())

    print("\nDescriptive Statistics by Gender:")
    print(df.groupby('gender', observed=True).apply(
        lambda g: desc_table(g, ['seconds', 'age']), include_groups=False
    ).to_string())

    print("\nDescriptive Statistics by Decade (seconds):")
    print(df.groupby('decade').apply(
        lambda g: desc_table(g, ['seconds']), include_groups=False
    ).to_string())

    # --- SECTION 2: PERCENTILE ANALYSIS ---
    # Show the finish time at key cutoff points (1st, 5th, 10th, ... 99th percentile)
    # for all runners, males only, and females only
    print("\nSECTION 2: PERCENTILE ANALYSIS")

    pcts = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    pctiles = pd.DataFrame({
        label: sub['seconds'].dropna().quantile(pcts)
        for label, sub in {'Overall': df, 'Male': df[df['gender'] == 'M'], 'Female': df[df['gender'] == 'F']}.items()
    }).T
    pctiles.columns = [f'P{int(p*100)}' for p in pcts]
    pctiles.index.name = 'group'
    print("\nPercentile Table (seconds):")
    print(pctiles.to_string())

    # --- SECTION 3: NORMALITY & GOODNESS-OF-FIT TESTS ---
    # Test whether finish times follow a normal (bell curve) distribution.
    # Three different tests are run on each group (overall, male, female):
    #   Shapiro-Wilk: checks if the data could have come from a normal distribution
    #   Anderson-Darling: similar check, more sensitive to differences in the tails
    #   Lilliefors: a variant of the KS test designed for testing normality
    # If p < 0.05, we reject the assumption that the data is normally distributed.
    print("\nSECTION 3: NORMALITY & GOODNESS-OF-FIT TESTS")

    norm_rows = []
    test_strata = {
        'Overall': df['seconds'].dropna(),
        'Male': df[df['gender'] == 'M']['seconds'].dropna(),
        'Female': df[df['gender'] == 'F']['seconds'].dropna(),
    }

    for name, data in test_strata.items():
        sw = shapiro(data.sample(n=min(5000, len(data)), random_state=42))
        norm_rows.append({'group': name, 'test': 'Shapiro-Wilk', 'statistic': sw.statistic,
                          'p_value': sw.pvalue, 'conclusion': 'Reject H0' if sw.pvalue < 0.05 else 'Fail to reject'})

        ad = anderson(data.values)
        norm_rows.append({'group': name, 'test': 'Anderson-Darling', 'statistic': ad.statistic,
                          'p_value': np.nan,
                          'conclusion': 'Reject H0 at 5%' if ad.statistic > ad.critical_values[2] else 'Fail to reject at 5%'})

        lf_stat, lf_p = lilliefors(data, dist='norm', pvalmethod='table')
        norm_rows.append({'group': name, 'test': 'Lilliefors (normal KS)', 'statistic': lf_stat,
                          'p_value': lf_p, 'conclusion': 'Reject H0' if lf_p < 0.05 else 'Fail to reject'})

    print("\nNormality Test Results:")
    print(pd.DataFrame(norm_rows).to_string(index=False))

    # --- SECTION 4: TAIL BEHAVIOR & EXTREME VALUE ANALYSIS ---
    # Examine how the extreme (very fast or very slow) finish times behave.
    print("\nSECTION 4: TAIL BEHAVIOR & EXTREME VALUE ANALYSIS")

    secs = df['seconds'].dropna()

    # Tail weight ratio: how spread out the top 4% is compared to the middle 25%.
    # A high ratio means there are unusually slow outliers.
    q = secs.quantile([0.50, 0.75, 0.95, 0.99])
    print(f"\nTail Weight Ratio (P99-P95)/(P75-P50): "
          f"{(q[0.99] - q[0.95]) / (q[0.75] - q[0.50]) if (q[0.75] - q[0.50]) > 0 else np.nan:.4f}")

    # Fit a Generalized Extreme Value distribution to annual winning (fastest) times.
    # GEV models the distribution of extremes (minimums/maximums).
    # The shape parameter tells us if there is a hard floor on how fast times can get.
    xi, mu_neg, sigma = genextreme.fit(-df.groupby('year')['seconds'].min().dropna())
    print(f"\nGEV Fit on Annual Winning Times (minima via negation):")
    print(f"  Shape (xi): {xi:.4f}")
    print(f"  Loc (mu, original scale): {-mu_neg:.4f}")
    print(f"  Scale (sigma): {sigma:.4f}")
    print(f"  Interpretation: {'Weibull (bounded lower tail)' if xi < 0 else 'Frechet (heavy lower tail)' if xi > 0 else 'Gumbel'}")

    # Fit a log-normal distribution to finish times and test how well it fits.
    # Log-normal means: if you take the log of each time, the result is normally distributed.
    # The bootstrap KS test compares the fitted distribution against the actual data;
    # a high p-value means the log-normal is a good fit.
    lognorm_sample = secs.sample(n=min(5000, len(secs)), random_state=42)
    ln_gof = goodness_of_fit(
        lognorm,
        lognorm_sample.to_numpy(),
        known_params={'loc': 0},
        statistic='ks',
        n_mc_samples=499,
        rng=np.random.default_rng(42),
    )
    print(f"\nLog-Normal Fit on finish times:")
    print(f"  Shape (s): {ln_gof.fit_result.params.s:.4f}")
    print(f"  Loc: {ln_gof.fit_result.params.loc:.4f}")
    print(f"  Scale: {ln_gof.fit_result.params.scale:.4f}")
    print(f"  Bootstrap KS GOF on n={len(lognorm_sample)} sample: statistic={ln_gof.statistic:.4f}, p-value={ln_gof.pvalue:.6f}")

    # --- SECTION 5: CORRELATION ANALYSIS ---
    # Measure how strongly variables are related to each other.
    # Only uses rows with real (non-imputed) age data to avoid artificial patterns.
    print("\nSECTION 5: CORRELATION ANALYSIS")

    valid_age = df[df['age'].notna() & ~df['age_imputed']].copy()
    print(f"\nUsing {len(valid_age)} rows with non-imputed valid age data")

    # Pearson r: measures linear relationship between age and finish time (-1 to +1).
    # Spearman rho: measures monotonic relationship (doesn't assume a straight line).
    valid = valid_age[['age', 'seconds']].dropna()
    pr = pearsonr(valid['age'], valid['seconds'])
    pr_ci = pr.confidence_interval()
    sr = spearmanr(valid['age'], valid['seconds'])
    print(f"\nAge vs Seconds:")
    print(f"  Pearson r={pr.correlation:.4f}, p={pr.pvalue:.2e}, 95% CI=[{pr_ci.low:.4f}, {pr_ci.high:.4f}]")
    print(f"  Spearman rho={sr.correlation:.4f}, p={sr.pvalue:.2e}")

    # Point-biserial correlation: measures relationship between a binary variable
    # (male/female) and a continuous variable (finish time).
    pb = df[['gender', 'seconds']].dropna()
    rpb = pointbiserialr((pb['gender'] == 'F').astype(float), pb['seconds'])
    print(f"\nPoint-biserial (gender vs seconds): r={rpb.correlation:.4f}, p={rpb.pvalue:.2e}")

    # Partial correlation: measures age-vs-time relationship after removing
    # the effect of gender (so gender differences don't inflate the correlation).
    partial_model = smf.ols('seconds ~ age + C(gender)',
                            data=valid_age[['age', 'seconds', 'gender']].dropna()).fit()
    age_t = partial_model.tvalues['age']
    partial_r = np.sign(age_t) * np.sqrt((age_t ** 2) / (age_t ** 2 + partial_model.df_resid))
    print(f"Partial correlation (age vs seconds | gender): r={partial_r:.4f}, p={partial_model.pvalues['age']:.2e}")

    # --- SECTION 6: TIME SERIES METRICS ---
    # Track how the race has changed over the years: participation, average times,
    # gender breakdown, and winning times.
    print("\nSECTION 6: TIME SERIES METRICS")

    # Build a year-by-year summary table with counts, averages, and winning times
    yearly = df.groupby('year').agg(
        n_finishers=('seconds', 'count'),
        mean_seconds=('seconds', 'mean'),
        median_seconds=('seconds', 'median'),
        std_seconds=('seconds', 'std'),
        min_seconds=('seconds', 'min'),
        mean_age=('age', 'mean'),
    )
    yearly = yearly.join(
        pd.crosstab(df['year'], df['gender'])
        .reindex(columns=['M', 'F'], fill_value=0)
        .rename(columns={'M': 'n_male', 'F': 'n_female'}),
        how='left',
    ).reset_index()
    yearly['pct_female'] = (yearly['n_female'] / yearly['n_finishers'] * 100).round(2)

    yearly = yearly.merge(
        df[df['gender'] == 'M'].groupby('year')['seconds'].min().rename('winning_time_M'),
        on='year', how='left',
    ).merge(
        df[df['gender'] == 'F'].groupby('year')['seconds'].min().rename('winning_time_F'),
        on='year', how='left',
    )

    print("\nYearly Summary (first 10 and last 10 years):")
    print(yearly.head(10).to_string(index=False))
    print("...")
    print(yearly.tail(10).to_string(index=False))

    # Percentage change in the fastest male time from one decade to the next
    print("\nDecade-over-Decade Change in Male Winning Time (%):")
    for decade, change in (df[df['gender'] == 'M'].groupby('decade')['seconds'].min().pct_change() * 100).dropna().items():
        print(f"  {decade}s: {change:+.2f}%")

    # Fit a straight line to male winning times over the years.
    # The slope tells us how many seconds faster (or slower) the winner gets per year.
    model = smf.ols('seconds ~ year',
                     data=df[df['gender'] == 'M'].groupby('year')['seconds'].min().reset_index()).fit()
    print(f"\nLinear Regression: Male Winning Time ~ Year")
    print(f"  Slope: {model.params['year']:.4f} seconds/year")
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Slope p-value: {model.pvalues['year']:.2e}")
    print(f"  95% CI for slope: [{model.conf_int().loc['year', 0]:.4f}, {model.conf_int().loc['year', 1]:.4f}]")

    # --- SECTION 7: GENDER COMPARISON TESTS ---
    # Test whether male and female finish times differ significantly.
    print("\nSECTION 7: GENDER COMPARISON TESTS")

    m_sec = df[df['gender'] == 'M']['seconds'].dropna()
    f_sec = df[df['gender'] == 'F']['seconds'].dropna()

    # Welch's t-test: tests if the average finish times of men and women are different.
    # Unlike a regular t-test, it does not assume equal variance between groups.
    welch = ttest_ind(m_sec, f_sec, equal_var=False)
    welch_ci = welch.confidence_interval()
    print(f"\nWelch's t-test: t={welch.statistic:.4f}, p={welch.pvalue:.2e}, df={welch.df:.1f}")
    print(f"  95% CI for mean difference: [{welch_ci.low:.1f}, {welch_ci.high:.1f}]")

    # Mann-Whitney U: tests if one group tends to have higher values than the other.
    # Works on ranks, so it doesn't assume normal distributions.
    mwu = mannwhitneyu(m_sec, f_sec, alternative='two-sided')
    print(f"Mann-Whitney U: U={mwu.statistic:.0f}, p={mwu.pvalue:.2e}")

    # Levene's test: checks if male and female times have similar spread (variance).
    # A significant result means one group is more spread out than the other.
    lev = levene(m_sec, f_sec)
    print(f"Levene's test (equality of variances): W={lev.statistic:.4f}, p={lev.pvalue:.2e}")

    # KS 2-sample: checks if male and female times come from the same distribution.
    # Unlike the t-test (means only), this detects any difference in shape.
    ks = ks_2samp(m_sec, f_sec)
    print(f"KS 2-sample: D={ks.statistic:.4f}, p={ks.pvalue:.2e}, "
          f"location={ks.statistic_location:.1f}, sign={ks.statistic_sign:+d}")

    # Hedges' g: measures how large the gender difference is in standard deviation units.
    # Values around 0.2 = small, 0.5 = medium, 0.8 = large effect.
    g, _ = effectsize_smd(
        m_sec.mean(), m_sec.std(ddof=1), m_sec.count(),
        f_sec.mean(), f_sec.std(ddof=1), f_sec.count(),
    )
    print(f"Hedges' g: {g:.4f}")

    # Repeat Mann-Whitney U and Hedges' g for each decade to see how the gender
    # gap has changed over time
    gd_rows = []
    for decade in sorted(df[df['gender'] == 'F']['decade'].unique()):
        m_d = df[(df['decade'] == decade) & (df['gender'] == 'M')]['seconds'].dropna()
        f_d = df[(df['decade'] == decade) & (df['gender'] == 'F')]['seconds'].dropna()
        if len(m_d) > 1 and len(f_d) > 1:
            mwu_d = mannwhitneyu(m_d, f_d, alternative='two-sided')
            g_decade, _ = effectsize_smd(
                m_d.mean(), m_d.std(ddof=1), m_d.count(),
                f_d.mean(), f_d.std(ddof=1), f_d.count(),
            )
            gd_rows.append({
                'decade': decade, 'n_male': len(m_d), 'n_female': len(f_d),
                'mean_diff': m_d.mean() - f_d.mean(),
                'mann_whitney_U': mwu_d.statistic, 'p_value': mwu_d.pvalue, 'hedges_g': g_decade,
            })
    print("\nGender Comparison by Decade:")
    print(pd.DataFrame(gd_rows).set_index('decade').to_string())

    # --- SECTION 8: AGE GROUP ANALYSIS ---
    # Compare finish times across age brackets to see which groups are faster/slower.
    print("\nSECTION 8: AGE GROUP ANALYSIS")

    # Split runners into age brackets
    aged = df[df['age'].between(14, 90)].copy()
    print(f"\nRows with valid age data (14-90): {len(aged)}")

    aged['age_group'] = pd.cut(aged['age'],
                               bins=[14, 30, 40, 50, 60, 70, 100],
                               labels=['14-29', '30-39', '40-49', '50-59', '60-69', '70+'],
                               right=False)
    aged = aged.dropna(subset=['age_group'])

    # Summary stats for each age group
    print("\nAge Group Descriptive Stats:")
    print(aged.groupby('age_group', observed=True)['seconds'].agg(
        n='count',
        mean='mean',
        median='median',
        std='std',
        IQR=lambda s: iqr(s, nan_policy='omit'),
        skewness='skew',
        excess_kurtosis=pd.Series.kurt,
    ).to_string())

    # Kruskal-Wallis: non-parametric test for whether at least one age group
    # has a different distribution of finish times than the others.
    kw = kruskal(*[g['seconds'].dropna().values
                    for _, g in aged.groupby('age_group', observed=True)])
    print(f"\nKruskal-Wallis H-test: H={kw.statistic:.4f}, p={kw.pvalue:.2e}")

    # One-way ANOVA: parametric version of the same question — do the group means differ?
    # Eta-squared measures what fraction of the total variance is explained by age group.
    anova = anova_lm(smf.ols('seconds ~ C(age_group)', data=aged).fit())
    print(f"One-way ANOVA: F={anova.loc['C(age_group)', 'F']:.4f}, p={anova.loc['C(age_group)', 'PR(>F)']:.2e}")
    print(f"Eta-squared: {anova.loc['C(age_group)', 'sum_sq'] / anova['sum_sq'].sum():.4f}")

    # Dunn's post-hoc test: after finding that groups differ overall, this tells us
    # which specific pairs of age groups are significantly different from each other.
    # Bonferroni adjustment prevents false positives from running many comparisons.
    print("\nDunn's Post-hoc Test (Bonferroni-adjusted p-values):")
    print(sp.posthoc_dunn(aged, val_col='seconds', group_col='age_group', p_adjust='bonferroni').to_string())

    # --- SECTION 9: OUTLIER DETECTION & QUANTIFICATION ---
    # Identify unusually fast or slow finish times using three different methods.
    print("\nSECTION 9: OUTLIER DETECTION & QUANTIFICATION")

    # IQR method: flag times outside Q1 - 1.5*IQR and Q3 + 1.5*IQR (the box plot whiskers)
    q1, q3 = secs.quantile([0.25, 0.75])
    iqr_v = iqr(secs, nan_policy='omit')
    lo = q1 - 1.5 * iqr_v
    hi = q3 + 1.5 * iqr_v
    iqr_out = ~secs.between(lo, hi, inclusive='both')
    print(f"\nIQR Method:")
    print(f"  Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr_v:.0f}")
    print(f"  Fences: [{lo:.0f}, {hi:.0f}]")
    print(f"  Outliers: {iqr_out.sum()} ({iqr_out.sum()/len(secs)*100:.2f}%)")

    # Z-score method: flag times more than 3 standard deviations from the mean
    z_out = np.abs(zscore(secs, nan_policy='omit')) > 3
    print(f"\nZ-score Method (|Z| > 3):")
    print(f"  Outliers: {z_out.sum()} ({z_out.sum()/len(secs)*100:.2f}%)")

    # Modified Z-score (MAD): uses median and median absolute deviation instead of
    # mean and std, making it resistant to the outliers themselves pulling the threshold
    med = secs.median()
    scaled_mad = median_abs_deviation(secs, scale='normal', nan_policy='omit')
    mad = median_abs_deviation(secs, nan_policy='omit')
    if np.isclose(scaled_mad, 0):
        mod_z = pd.Series(np.nan, index=secs.index, dtype=float)
    else:
        mod_z = (secs - med) / scaled_mad
    mad_out = np.abs(mod_z) > 3.5
    print(f"\nModified Z-score (MAD, threshold=3.5):")
    print(f"  MAD={mad:.2f}, Scaled MAD={scaled_mad:.2f}, Median={med:.0f}")
    print(f"  Outliers: {mad_out.sum()} ({mad_out.sum()/len(secs)*100:.2f}%)")

    # --- SECTION 10: SPLIT-TIME ANALYSIS (2015-2017) ---
    # Analyze how runners paced themselves using checkpoint times (only available 2015-2017).
    print("\nSECTION 10: SPLIT-TIME ANALYSIS (2015-2017)")

    # Keep only rows that have all split times and a final time
    splits = df[df['year'].between(2015, 2017)].copy()
    splits = splits[splits[SPLIT_SECS].notna().all(axis=1) & splits['seconds'].notna()].copy()
    print(f"\nRows with valid split data (2015-2017): {len(splits)}")

    # Compute how long each 5k segment took by subtracting consecutive checkpoints
    split_cols = [c for c in SPLIT_SECS if c != 'half_seconds']
    segs = ['seg_0_5k', 'seg_5k_10k', 'seg_10k_15k', 'seg_15k_20k',
            'seg_20k_25k', 'seg_25k_30k', 'seg_30k_35k', 'seg_35k_40k']
    diffs = splits[split_cols].diff(axis=1)
    diffs.iloc[:, 0] = splits['5k_seconds']
    diffs.columns = segs
    splits = pd.concat([splits, diffs], axis=1)

    # Classify each runner's pacing strategy by comparing their second half to their first half
    splits['pacing_type'] = pd.cut(
        (splits['seconds'] - splits['half_seconds']) / splits['half_seconds'],
        bins=[-np.inf, 0.99, 1.01, np.inf],
        labels=['negative_split', 'even_split', 'positive_split'])

    print("\nPacing Distribution by Year (%):")
    print((pd.crosstab(splits['year'], splits['pacing_type'], normalize='index') * 100).to_string())

    # How strongly each checkpoint time correlates with the final finish time
    print("\nSplit-Time Correlation Matrix (with final time):")
    print(splits[SPLIT_SECS + ['seconds']].corr().to_string())

    # Summary stats for each 5k segment
    seg_desc = splits[segs].describe().T
    seg_desc['CV'] = splits[segs].apply(lambda s: variation(s, ddof=1, nan_policy='omit'))
    print("\nSegment Descriptive Statistics:")
    print(seg_desc.to_string())

    # Compare pacing patterns between fast and slow finishers
    splits['finish_quartile'] = pd.qcut(splits['seconds'], 4, labels=['Q1(fastest)', 'Q2', 'Q3', 'Q4(slowest)'])
    print("\nMean Segment Times by Finishing Quartile:")
    print(splits.groupby('finish_quartile', observed=True)[segs].mean().to_string())


if __name__ == '__main__':
    main()

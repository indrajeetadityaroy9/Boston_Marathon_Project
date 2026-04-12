from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy.stats import kruskal, ks_2samp, mannwhitneyu, pearsonr, shapiro, spearmanr, ttest_ind

PROCESSED_CSV = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'boston_marathon_cleaned.csv'
SPLIT_COLUMNS = ['5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds', 'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds']


def compute_icc_anova(df, group_col, value_col):
    """Shrout-Fleiss ICC(1) with unbalanced-group correction (pingouin cannot handle this data)."""
    k, n = df[group_col].nunique(), len(df)
    grp = df.groupby(group_col)[value_col]
    gm, grand = grp.transform('mean'), df[value_col].mean()
    msb = ((gm - grand) ** 2).sum() / (k - 1)
    msw = ((df[value_col] - gm) ** 2).sum() / (n - k)
    k0 = (n - (grp.size() ** 2).sum() / n) / (k - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def compute_runner_age_slopes(df):
    """Vectorized per-runner OLS slopes of finish time on age (seconds per year)."""
    grp = df.groupby('display_name')
    ca = df['age'] - grp['age'].transform('mean')
    cs = df['seconds'] - grp['seconds'].transform('mean')
    return (ca * cs).groupby(df['display_name']).sum() / (ca ** 2).groupby(df['display_name']).sum()


def run_exploratory_data_analysis():
    print("BOSTON MARATHON EXPLORATORY DATA ANALYSIS")

    df = pd.read_csv(PROCESSED_CSV, engine='pyarrow')
    df['gender'] = df['gender'].astype('category')
    df['decade'] = (df['year'] // 10) * 10

    # 1. DATA QUALITY REPORT
    print("\n1. DATA QUALITY REPORT")
    print(f"\n  Rows: {len(df):,}  Columns: {len(df.columns)}  Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  Unique display_names: {df['display_name'].nunique():,}")

    print("\n  Missing values per column:")
    missing = df.isna().mean() * 100
    for col, pct in missing[missing > 0].sort_values(ascending=False).items():
        print(f"    {col:<30} {pct:>6.2f}%  ({df[col].isna().sum():>7,})")

    print("\n  Row-count distribution across decades:")
    print(df.groupby('decade').size().to_string())

    nc = df['name_collides_within_year']
    print(f"\n  Within-year name collisions: {int(nc.sum()):,} rows in {df[nc].groupby(['year', 'display_name']).ngroups:,} (year, display_name) groups")

    n_out = df.groupby(['year', 'gender'], observed=True).apply(
        lambda g: ((s := g['seconds'], q := s.quantile([0.25, 0.75]), iqr := q.iloc[1] - q.iloc[0]) and False) or ((s < q.iloc[0] - 3 * iqr) | (s > q.iloc[1] + 3 * iqr)).sum(),
        include_groups=False,
    ).sum()
    print(f"\n  Extreme finish-time outliers (3*IQR per year*gender): {n_out:,} ({n_out / len(df) * 100:.3f}% of rows)")
    print(f"\n  Imputed ages: {int(df['age_imputed'].sum()):,} rows")

    # 2. DESCRIPTIVE STATISTICS
    print("\n2. DESCRIPTIVE STATISTICS")
    print("\nOverall:")
    print(df[['seconds', 'age']].describe().T.to_string())
    print(f"  seconds  skewness={df['seconds'].skew():.4f}  kurtosis={df['seconds'].kurtosis():.4f}")
    print(f"  age      skewness={df['age'].skew():.4f}  kurtosis={df['age'].kurtosis():.4f}")

    print("\nBy Gender:")
    for gender, gdf in df.groupby('gender', observed=True):
        print(f"\n  {gender} (n={len(gdf)}):")
        print(gdf[['seconds', 'age']].describe().T.to_string())

    print("\nBy Decade (seconds):")
    print(df.groupby('decade')['seconds'].describe().to_string())

    # 3. NORMALITY TESTS
    print("\n3. NORMALITY TESTS")
    strata = {'Overall': df['seconds'], 'Male': df[df['gender'] == 'M']['seconds'], 'Female': df[df['gender'] == 'F']['seconds']}
    norm_rows = []
    for name, s in strata.items():
        samp = s.sample(n=5000, random_state=42)
        r = shapiro(samp)
        norm_rows.append({'group': name, 'statistic': r.statistic, 'p_value': r.pvalue})
        rl = shapiro(np.log(samp))
        norm_rows.append({'group': f'{name} (log)', 'statistic': rl.statistic, 'p_value': rl.pvalue})
    print("\nShapiro-Wilk Results (n=5000 samples):")
    print(pd.DataFrame(norm_rows).to_string(index=False))

    log_sec = np.log(df['seconds'])
    print(f"\nLog-transform shape: skewness={log_sec.skew():.4f}, kurtosis={log_sec.kurtosis():.4f}")

    # 4. CORRELATION ANALYSIS
    print("\n4. CORRELATION ANALYSIS")
    va = df[df['age'].notna() & ~df['age_imputed']].copy()
    print(f"\nUsing {len(va):,} rows with non-imputed age data")

    pr = pearsonr(va['age'], va['seconds'])
    ci = pr.confidence_interval()
    sr = spearmanr(va['age'], va['seconds'])
    print(f"\nAge vs Seconds:")
    print(f"  Pearson r={pr.correlation:.4f}, p={pr.pvalue:.2e}, 95% CI=[{ci.low:.4f}, {ci.high:.4f}]")
    print(f"  Spearman rho={sr.correlation:.4f}, p={sr.pvalue:.2e}")

    va['female'] = (va['gender'] == 'F').astype(int)
    pc = pg.partial_corr(data=va, x='age', y='seconds', covar='female')
    print(f"  Partial r (age | gender): {pc['r'].iloc[0]:.4f}, p={pc['p_val'].iloc[0]:.2e}")

    # 5. GENDER COMPARISON
    print("\n5. GENDER COMPARISON")
    m_sec, f_sec = df[df['gender'] == 'M']['seconds'], df[df['gender'] == 'F']['seconds']

    w = ttest_ind(m_sec, f_sec, equal_var=False)
    wci = w.confidence_interval()
    print(f"\nWelch's t-test: t={w.statistic:.4f}, p={w.pvalue:.2e}, df={w.df:.1f}")
    print(f"  95% CI for mean difference: [{wci.low:.1f}, {wci.high:.1f}]")

    mw = mannwhitneyu(m_sec, f_sec, alternative='two-sided')
    print(f"Mann-Whitney U: U={mw.statistic:.0f}, p={mw.pvalue:.2e}")
    print(f"Hedges' g (M-F): {pg.compute_effsize(m_sec, f_sec, eftype='hedges'):.4f}")

    gap_rows = []
    for dec in sorted(df[df['gender'] == 'F']['decade'].unique()):
        md, fd = df[(df['decade'] == dec) & (df['gender'] == 'M')]['seconds'], df[(df['decade'] == dec) & (df['gender'] == 'F')]['seconds']
        mwd = mannwhitneyu(md, fd, alternative='two-sided')
        gap_rows.append({'decade': dec, 'n_male': len(md), 'n_female': len(fd), 'mean_diff': md.mean() - fd.mean(), 'p_value': mwd.pvalue, 'hedges_g': pg.compute_effsize(md, fd, eftype='hedges')})
    print("\nGender Gap by Decade:")
    print(pd.DataFrame(gap_rows).set_index('decade').to_string())

    # 6. AGE GROUP ANALYSIS
    print("\n6. AGE GROUP ANALYSIS")
    adf = df[df['age'].notna() & ~df['age_imputed']].copy()
    adf['age_group'] = pd.cut(adf['age'], bins=[14, 30, 40, 50, 60, 70, np.inf], labels=['14-29', '30-39', '40-49', '50-59', '60-69', '70+'], right=False)
    print(f"\nRows with valid age: {len(adf):,}")

    print("\nAge Group Stats:")
    print(adf.groupby('age_group', observed=True)['seconds'].agg(n='count', mean='mean', median='median', std='std').to_string())

    kw = kruskal(*[g['seconds'].values for _, g in adf.groupby('age_group', observed=True)])
    print(f"\nKruskal-Wallis: H={kw.statistic:.4f}, p={kw.pvalue:.2e}")

    print("\nDunn's Post-hoc (Bonferroni):")
    print(sp.posthoc_dunn(adf, val_col='seconds', group_col='age_group', p_adjust='bonferroni').to_string())
    print(f"\nEta-squared: {pg.welch_anova(data=adf, dv='seconds', between='age_group')['np2'].iloc[0]:.4f}")

    # 7. SPLIT-TIME ANALYSIS (2015-2017)
    print("\n7. SPLIT-TIME ANALYSIS (2015-2017)")
    sdf = df[df['year'].between(2015, 2017) & df[SPLIT_COLUMNS].notna().all(axis=1)].copy()
    print(f"\nRows with valid split data: {len(sdf):,}")

    print("\nSplit-Time Correlation Matrix:")
    print(sdf[SPLIT_COLUMNS + ['seconds']].corr().to_string())

    pr_ratio = (sdf['seconds'] - sdf['half_seconds']) / sdf['half_seconds']
    sdf['pacing_type'] = pd.cut(pr_ratio, bins=[-np.inf, 0.99, 1.01, np.inf], labels=['negative_split', 'even_split', 'positive_split'])

    print("\nPacing Distribution by Year (%):")
    print((pd.crosstab(sdf['year'], sdf['pacing_type'], normalize='index') * 100).to_string())

    pdist = sdf['pacing_type'].value_counts(normalize=True).sort_index() * 100
    print("\nOverall Pacing Distribution:")
    for pt, pct in pdist.items():
        print(f"  {pt}: {pct:.1f}%")

    kwp = kruskal(*[g['seconds'].values for _, g in sdf.groupby('pacing_type', observed=True)])
    print(f"\nKruskal-Wallis (pacing): H={kwp.statistic:.2f}, p={kwp.pvalue:.2e}")

    peta = float(pg.welch_anova(data=sdf, dv='seconds', between='pacing_type')['np2'].iloc[0])
    print(f"Eta-squared (pacing): {peta:.4f} ({peta * 100:.1f}% of variance)")

    print("\nPer-Year Eta-squared (pacing):")
    for yr, ydf in sdf.groupby('year'):
        print(f"  {yr}: eta_sq={pg.welch_anova(data=ydf, dv='seconds', between='pacing_type')['np2'].iloc[0]:.4f}")

    print("\nDunn's Post-hoc Pacing (Bonferroni):")
    print(sp.posthoc_dunn(sdf, val_col='seconds', group_col='pacing_type', p_adjust='bonferroni').to_string())

    # 8. REPEAT RUNNER PROFILING
    print("\n8. REPEAT RUNNER PROFILING")
    counts = df['display_name'].value_counts()
    rdf = df[df['display_name'].isin(counts[counts > 1].index)]
    print(f"\nUnique runners: {len(counts):,}")
    print(f"Repeat runners (>1 appearance): {(counts > 1).sum():,}")
    print(f"Rows from repeat runners: {len(rdf):,} ({len(rdf) / len(df) * 100:.1f}%)")

    radf = rdf[rdf['age'].notna() & ~rdf['age_imputed'] & ~rdf['name_collides_within_year']].copy()
    print(f"After within-year collision filter: {radf['display_name'].nunique():,} runners, {len(radf):,} rows")

    spans = radf.groupby('display_name')['age'].agg(n_races='count', age_span=lambda a: a.max() - a.min())
    spans = spans[spans['n_races'] > 1]
    print(f"\nAge Span (n={len(spans):,} runners with >=2 age observations):")
    print(spans['age_span'].describe().to_string())

    icc = compute_icc_anova(radf, 'display_name', 'seconds')
    print(f"\nIntra-class correlation (ICC): {icc:.4f} -- {icc * 100:.1f}% of finish time variance is between runners")

    slopes = compute_runner_age_slopes(radf)
    nv = slopes.notna().sum()
    print(f"\nWithin-Runner Aging Slopes (n={nv:,} runners with age variation):")
    print(f"  Mean: {slopes.mean():.1f} sec/yr, Median: {slopes.median():.1f} sec/yr, Std: {slopes.std():.1f} sec/yr")
    print(f"  Slowing: {(slopes > 0).sum():,} ({(slopes > 0).mean() * 100:.1f}%), Improving: {(slopes < 0).sum():,} ({(slopes < 0).mean() * 100:.1f}%)")

    # 9. TEMPORAL DRIFT OVER TIME
    print("\n9. TEMPORAL DRIFT OVER TIME")
    dec_sum = df.groupby('decade').agg(
        n=('seconds', 'size'), mean_seconds=('seconds', 'mean'), std_seconds=('seconds', 'std'),
        mean_age=('age', 'mean'), female_frac=('gender', lambda g: (g == 'F').mean()),
    ).reset_index()
    print("\nPer-decade summary (seconds numeric, age in years):")
    print(dec_sum.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    decades = sorted(df['decade'].unique())
    ks_rows = []
    for d1, d2 in zip(decades[:-1], decades[1:]):
        s1, s2 = df[df['decade'] == d1]['seconds'], df[df['decade'] == d2]['seconds']
        ks = ks_2samp(s1, s2)
        ks_rows.append({'earlier': f'{int(d1)}s', 'later': f'{int(d2)}s', 'n_earlier': len(s1), 'n_later': len(s2), 'mean_earlier': float(s1.mean()), 'mean_later': float(s2.mean()), 'delta_mean': float(s2.mean() - s1.mean()), 'ks_stat': float(ks.statistic), 'ks_p': float(ks.pvalue)})
    print("\nAdjacent-decade KS comparisons on finish time:")
    print(pd.DataFrame(ks_rows).to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print("\nPer-year mean finish time (2000-2019):")
    rec = df[df['year'] >= 2000].groupby('year').agg(
        n=('seconds', 'size'), mean_seconds=('seconds', 'mean'),
        median_seconds=('seconds', 'median'), female_frac=('gender', lambda g: (g == 'F').mean()),
    ).reset_index()
    rec['delta_vs_prev'] = rec['mean_seconds'].diff()
    print(rec.to_string(index=False, float_format=lambda v: f'{v:.2f}'))


if __name__ == '__main__':
    run_exploratory_data_analysis()

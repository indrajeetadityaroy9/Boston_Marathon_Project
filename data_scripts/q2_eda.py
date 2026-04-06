#!/usr/bin/env python3.11

from pathlib import Path
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy import stats
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

_FIT_KWARGS = dict(method='lbfgs', maxiter=200)

CLEANED_PATH = Path(__file__).resolve().parent.parent / 'cleaned_data' / 'boston_marathon_cleaned.csv'
SUMMARY_PATH = Path(__file__).resolve().parent.parent / 'q2_results_summary.txt'


def load_and_filter_data():
    print("\nSTEP 1: Q2 SAMPLE CONSTRUCTION")

    df = pd.read_csv(CLEANED_PATH, low_memory=False,
                     usecols=['year', 'display_name', 'age', 'gender', 'seconds', 'age_imputed'],
                     dtype={'gender': 'category', 'display_name': str})
    print(f"\n1. Full dataset: {len(df):,} rows")

    df = df[~df['age_imputed'] & df['age'].notna()].copy()
    print(f"2. Non-imputed age: {len(df):,} rows, {df['display_name'].nunique():,} unique names")

    df = df[df.groupby('display_name')['display_name'].transform('size') > 1].copy()
    print(f"3. Repeat runners (>1 appearance): {len(df):,} rows, "
          f"{df['display_name'].nunique():,} runners")

    df.sort_values(['display_name', 'year'], inplace=True)
    same_runner = df['display_name'].values[1:] == df['display_name'].values[:-1]
    age_diff = np.diff(df['age'].values)
    year_diff = np.diff(df['year'].values)
    bad_gap = same_runner & (np.abs(age_diff - year_diff) > 8)
    bad_runners_set = set(df['display_name'].values[1:][bad_gap])
    n_total_runners = df['display_name'].nunique()
    plausible_mask = ~df['display_name'].isin(bad_runners_set)
    n_removed = n_total_runners - (n_total_runners - len(bad_runners_set))
    df = df[plausible_mask].copy()
    print(f"4. Age-consistency filter (+/-8yr): removed {n_removed:,}, "
          f"retained {len(df):,} rows, {df['display_name'].nunique():,} runners")

    df = df[df['year'] >= 2000].copy()
    print(f"5. Year >= 2000: {len(df):,} rows, {df['display_name'].nunique():,} runners")

    df = df[df.groupby('display_name')['display_name'].transform('size') > 1].copy()
    print(f"6. Re-check repeat status: {len(df):,} rows, "
          f"{df['display_name'].nunique():,} runners")

    return df



def compute_icc(df, groups, y):
    print("\nSTEP 3: INTRA-CLASS CORRELATION (ICC)")

    n_runners = df['display_name'].nunique()
    n_obs = len(df)

    grp_seconds = df.groupby('display_name')['seconds']
    runner_means = grp_seconds.transform('mean')
    grand_mean = y.mean()
    ms_between = ((runner_means - grand_mean) ** 2).sum() / (n_runners - 1)
    ms_within = ((y - runner_means.values) ** 2).sum() / (n_obs - n_runners)
    ni = grp_seconds.size()
    k0 = (n_obs - (ni ** 2).sum() / n_obs) / (n_runners - 1)
    icc_anova = (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)
    print(f"\n1. ANOVA-based ICC(1): {icc_anova:.4f} ({icc_anova*100:.1f}%)")

    null_exog = pd.DataFrame({'const': np.ones(n_obs)})
    null_result = MixedLM(y, null_exog, groups=groups).fit(reml=True, **_FIT_KWARGS)
    tau2_u = null_result.cov_re.iloc[0, 0]
    sig2_u = null_result.scale
    icc_uncond = tau2_u / (tau2_u + sig2_u)
    print(f"2. Unconditional model ICC: {icc_uncond:.4f} ({icc_uncond*100:.1f}%)")
    print(f"   tau^2={tau2_u:,.0f}, sigma^2={sig2_u:,.0f}")

    exog = sm.add_constant(df[['age_c', 'female', 'year_c']])
    m1_reml = MixedLM(y, exog, groups=groups).fit(reml=True, **_FIT_KWARGS)
    tau2_c = m1_reml.cov_re.iloc[0, 0]
    sig2_c = m1_reml.scale
    icc_cond = tau2_c / (tau2_c + sig2_c)
    print(f"3. Conditional ICC (after age, gender, year): {icc_cond:.4f} ({icc_cond*100:.1f}%)")
    print(f"   tau^2={tau2_c:,.0f}, sigma^2={sig2_c:,.0f}")

    result = {
        'icc_anova': icc_anova,
        'icc_unconditional': icc_uncond,
        'icc_conditional': icc_cond,
        'tau2_uncond': tau2_u, 'sigma2_uncond': sig2_u,
        'tau2_cond': tau2_c, 'sigma2_cond': sig2_c,
    }
    return result, m1_reml


def compute_runner_slopes(df):
    print("\nSTEP 4: PER-RUNNER AGING SLOPES")

    age_grp = df.groupby('display_name')['age']
    age_min = age_grp.min()
    age_max = age_grp.max()
    age_span = age_max - age_min
    eligible = age_span[age_span >= 5].index
    slope_df = df[df['display_name'].isin(eligible)].copy()

    grp = slope_df.groupby('display_name')
    age_mean = grp['age'].transform('mean')
    sec_mean = grp['seconds'].transform('mean')
    slope_df['_dx'] = slope_df['age'] - age_mean
    slope_df['_dy'] = slope_df['seconds'] - sec_mean
    slope_df['_dxdy'] = slope_df['_dx'] * slope_df['_dy']
    slope_df['_dx2'] = slope_df['_dx'] ** 2
    runner_slopes = grp['_dxdy'].sum() / grp['_dx2'].sum()

    n = len(runner_slopes)
    result = {
        'n_slopes': n,
        'mean_slope': runner_slopes.mean(),
        'std_slope': runner_slopes.std(),
        'median_slope': runner_slopes.median(),
        'pct_slowing': (runner_slopes > 0).mean() * 100,
        'pct_improving': (runner_slopes < 0).mean() * 100,
    }

    print(f"\nQ2 sample slopes (age span >=5yr): n={n}")
    print(f"  Mean: {result['mean_slope']:.1f} sec/yr")
    print(f"  Std:  {result['std_slope']:.1f} sec/yr")
    print(f"  Median: {result['median_slope']:.1f} sec/yr")
    print(f"  Slowing (positive): {result['pct_slowing']:.1f}%")
    print(f"  Improving (negative): {result['pct_improving']:.1f}%")
    print(f"\n--- Comparison with eda.py full-sample slopes ---")
    print(f"  eda.py: n=22,453, mean=77.8, std=265.5, improving=34.6%")
    print(f"  Q2:     n={n:,}, mean={result['mean_slope']:.1f}, "
          f"std={result['std_slope']:.1f}, improving={result['pct_improving']:.1f}%")
    print(f"  (Difference: Q2 is restricted to 2000-2019 + age-consistency filter)")

    return result


def fit_mixed_models(df, y, groups, exog, exog_re, m1_reml):
    print("\nSTEP 5: MIXED-EFFECTS MODELS")

    print("\nFitting M0 (OLS)...")
    m0 = sm.OLS(y, exog).fit()
    print(f"  M0: R^2={m0.rsquared:.4f}")

    print("\nFitting M1 (ML for LRT)...")
    m1_ml = MixedLM(y, exog, groups=groups).fit(
        reml=False, start_params=m1_reml.params, **_FIT_KWARGS)
    print(f"  M1_ML: converged={m1_ml.converged}")

    print("\nFitting M2 (random intercept + slope)...")
    m2_reml = MixedLM(y, exog, groups=groups, exog_re=exog_re).fit(
        reml=True, **_FIT_KWARGS)
    print(f"  M2_REML: converged={m2_reml.converged}")
    m2_ml = MixedLM(y, exog, groups=groups, exog_re=exog_re).fit(
        reml=False, start_params=m2_reml.params, **_FIT_KWARGS)
    print(f"  M2_ML: converged={m2_ml.converged}")

    print("\n--- Fixed Effects (from REML) ---")
    print(f"\n{'':>12} {'M0':>14} {'M1':>14} {'M2':>14}")
    print("-" * 56)
    for var in exog.columns:
        print(f"  {var:>10} {m0.params[var]:>14.2f} "
              f"{m1_reml.fe_params[var]:>14.2f} {m2_reml.fe_params[var]:>14.2f}")

    print(f"\n--- M2 REML Detailed Fixed Effects ---")
    ci = m2_reml.conf_int()
    for var in exog.columns:
        print(f"  {var:>10}: b={m2_reml.fe_params[var]:>10.2f}, "
              f"SE={m2_reml.bse_fe[var]:>8.2f}, p={m2_reml.pvalues[var]:.2e}, "
              f"95% CI=[{ci.loc[var, 0]:.2f}, {ci.loc[var, 1]:.2f}]")

    print(f"\n--- M1 Random Effects (REML) ---")
    print(f"  tau^2 (intercept): {m1_reml.cov_re.iloc[0, 0]:,.0f}")
    print(f"  sigma^2 (residual):  {m1_reml.scale:,.0f}")

    cov_re = m2_reml.cov_re
    tau2_0 = cov_re.iloc[0, 0]
    tau2_1 = cov_re.iloc[1, 1]
    cov_01 = cov_re.iloc[0, 1]
    corr_01 = cov_01 / np.sqrt(tau2_0 * tau2_1)

    print(f"\n--- M2 Random Effects (REML) ---")
    print(f"  tau^2_0 (intercept var): {tau2_0:,.0f}")
    print(f"  tau^2_1 (slope var):     {tau2_1:.2f}")
    print(f"  cov(b0, b1):        {cov_01:.2f}")
    print(f"  corr(b0, b1):       {corr_01:.4f}")
    print(f"  sigma^2 (residual):       {m2_reml.scale:,.0f}")

    print(f"\n--- In-Sample RMSE ---")
    print(f"  {'Model':<6} {'Conditional':>14} {'Marginal':>14} {'Within-runner':>14}")
    print(f"  {'':>6} {'(known runner)':>14} {'(new runner)':>14} {'SD (sigma)':>14}")
    print(f"  {'-'*6} {'-'*14} {'-'*14} {'-'*14}")

    Xb = exog.values @ m0.params.values
    rmse_m0 = np.sqrt(np.mean((y - Xb) ** 2))
    print(f"  {'M0':<6} {rmse_m0:>14.1f} {rmse_m0:>14.1f} {rmse_m0:>14.1f}")

    for label, m in [('M1', m1_reml), ('M2', m2_reml)]:
        cond = np.sqrt(np.mean(m.resid ** 2))
        marg = np.sqrt(np.mean((y - exog.values @ m.fe_params.values) ** 2))
        within = np.sqrt(m.scale)
        print(f"  {label:<6} {cond:>14.1f} {marg:>14.1f} {within:>14.1f}")

    print(f"\n  Conditional: uses runner-specific BLUPs (best case for known runners)")
    print(f"  Marginal: fixed effects only (prediction for new/unseen runners)")
    print(f"  Within-runner SD: model residual sigma (race-to-race noise for same runner)")

    return m0, m1_reml, m1_ml, m2_reml, m2_ml


def model_selection(m0, m1_ml, m2_ml):
    print("\nSTEP 6: MODEL SELECTION")

    lr_01 = 2 * (m1_ml.llf - m0.llf)
    p_std_01 = stats.chi2.sf(lr_01, df=1)
    p_corr_01 = 0.5 * stats.chi2.sf(lr_01, df=1)
    print(f"\n--- LRT: M0 vs M1 (df=1) ---")
    print(f"  LR statistic: {lr_01:.2f}")
    print(f"  Standard p-value: {p_std_01:.2e}")
    print(f"  Boundary-corrected p: {p_corr_01:.2e}")

    lr_12 = 2 * (m2_ml.llf - m1_ml.llf)
    p_std_12 = stats.chi2.sf(lr_12, df=2)
    p_corr_12 = 0.5 * stats.chi2.sf(lr_12, df=1) + 0.5 * stats.chi2.sf(lr_12, df=2)
    print(f"\n--- LRT: M1 vs M2 (df=2) ---")
    print(f"  LR statistic: {lr_12:.2f}")
    print(f"  Standard p-value: {p_std_12:.2e}")
    print(f"  Boundary-corrected p: {p_corr_12:.2e}")

    print(f"\n--- Model Comparison (ML) ---")
    print(f"{'Model':>6} {'k':>4} {'log-lik':>14} {'AIC':>14} {'BIC':>14}")
    print("-" * 56)
    rows = [('M0', 5, m0.llf, m0.aic, m0.bic),
            ('M1', 6, m1_ml.llf, m1_ml.aic, m1_ml.bic),
            ('M2', 8, m2_ml.llf, m2_ml.aic, m2_ml.bic)]
    best_aic = min(r[3] for r in rows)
    best_bic = min(r[4] for r in rows)
    for name, k, llf, aic, bic in rows:
        print(f"  {name:>4} {k:>4} {llf:>14.1f} {aic:>14.1f} {bic:>14.1f}"
              f"  (dAIC={aic - best_aic:+.1f}, dBIC={bic - best_bic:+.1f})")

    return {
        'lr_01': lr_01, 'p_standard_01': p_std_01, 'p_corrected_01': p_corr_01,
        'lr_12': lr_12, 'p_standard_12': p_std_12, 'p_corrected_12': p_corr_12,
        'm0_llf': m0.llf, 'm0_aic': m0.aic, 'm0_bic': m0.bic,
        'm1_llf': m1_ml.llf, 'm1_aic': m1_ml.aic, 'm1_bic': m1_ml.bic,
        'm2_llf': m2_ml.llf, 'm2_aic': m2_ml.aic, 'm2_bic': m2_ml.bic,
    }


def variance_decomposition(m2_reml, df, exog):
    print("\nSTEP 7: VARIANCE DECOMPOSITION")

    var_fixed = np.var(exog.values @ m2_reml.fe_params.values)
    cov_re = m2_reml.cov_re
    tau2_0 = cov_re.iloc[0, 0]
    var_slope = cov_re.iloc[1, 1] * df['age_c'].var()
    sigma2 = m2_reml.scale
    total = var_fixed + tau2_0 + var_slope + sigma2

    r2_marginal = var_fixed / total
    r2_conditional = (var_fixed + tau2_0 + var_slope) / total

    print(f"\n{'Component':<25} {'Variance':>14} {'Proportion':>12}")
    print("-" * 53)
    print(f"  {'Fixed effects':<23} {var_fixed:>14,.0f} {var_fixed/total:>11.1%}")
    print(f"  {'Random intercepts (tau^2_0)':<23} {tau2_0:>14,.0f} {tau2_0/total:>11.1%}")
    print(f"  {'Random slopes (tau^2_1*Var)':<23} {var_slope:>14,.0f} {var_slope/total:>11.1%}")
    print(f"  {'Residual (sigma^2)':<23} {sigma2:>14,.0f} {sigma2/total:>11.1%}")
    print(f"  {'Total':<23} {total:>14,.0f} {'100.0%':>12}")
    print(f"\n  Marginal R^2 (fixed only):     {r2_marginal:.4f}")
    print(f"  Conditional R^2 (fixed+random): {r2_conditional:.4f}")

    return {
        'var_fixed': var_fixed, 'tau2_intercept': tau2_0,
        'var_slope': var_slope, 'sigma2': sigma2, 'total_var': total,
        'r2_marginal': r2_marginal, 'r2_conditional': r2_conditional,
    }


def residual_diagnostics(m2_reml, df):
    print("\nSTEP 8: RESIDUAL DIAGNOSTICS")

    resid = m2_reml.resid
    result = {}

    print(f"\n--- Conditional Residuals ---")
    skew_val = stats.skew(resid)
    kurt_val = stats.kurtosis(resid)
    print(f"  Mean:     {resid.mean():.2f}")
    print(f"  Std:      {resid.std():.2f}")
    print(f"  Skewness: {skew_val:.4f}")
    print(f"  Kurtosis: {kurt_val:.4f}")
    sw_stat, sw_p = stats.shapiro(
        pd.Series(resid).sample(n=5000, random_state=42).values)
    print(f"  Shapiro-Wilk (n=5000): W={sw_stat:.4f}, p={sw_p:.2e}")
    result.update({'resid_skew': skew_val, 'resid_kurt': kurt_val,
                   'shapiro_w': sw_stat, 'shapiro_p': sw_p})

    re_df = pd.DataFrame(m2_reml.random_effects).T
    for i, col in enumerate(re_df.columns):
        label = 'intercept' if i == 0 else 'slope'
        vals = re_df[col].values
        print(f"\n--- Random {label}s ---")
        print(f"  Mean:     {vals.mean():.2f}")
        print(f"  Std:      {vals.std():.2f}")
        print(f"  Skewness: {stats.skew(vals):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(vals):.4f}")
        threshold = 3 * vals.std()
        n_extreme = int((np.abs(vals) > threshold).sum())
        print(f"  |RE| > 3 SD: {n_extreme} runners ({n_extreme/len(vals)*100:.2f}%)")
        top10 = np.argsort(np.abs(vals))[::-1][:10]
        print(f"  Top 10 most extreme {label}s:")
        for rank, idx in enumerate(top10, 1):
            print(f"    {rank}. {re_df.index[idx]}: {vals[idx]:.1f}")
        result[f're_{label}_mean'] = vals.mean()
        result[f're_{label}_std'] = vals.std()
        result[f're_{label}_skew'] = stats.skew(vals)
        result[f're_{label}_kurt'] = stats.kurtosis(vals)
        result[f're_{label}_n_extreme'] = n_extreme

    df_r = df[['year', 'gender', 'age']].copy()
    df_r['resid'] = resid

    print(f"\n--- Mean Residual by Year ---")
    year_means = df_r.groupby('year')['resid'].mean()
    for yr, mr in year_means.items():
        flag = " *** FLAGGED" if abs(mr) > 100 else ""
        print(f"  {yr}: {mr:>8.1f}{flag}")
    flagged = int((year_means.abs() > 100).sum())
    print(f"\n  Flagged years (|mean resid| > 100s): {flagged}/{len(year_means)}")
    print(f"  Year-level residual std: {year_means.std():.1f}s")
    print(f"  NOTE: Linear year effect is insufficient. Consider year as factor.")
    result['year_flagged_count'] = flagged
    result['year_resid_std'] = year_means.std()

    print(f"\n--- Mean Residual by Gender ---")
    for g, mr in df_r.groupby('gender')['resid'].mean().items():
        print(f"  {g}: {mr:>8.1f}")

    print(f"\n--- Mean Residual by Age Decile ---")
    df_r['age_decile'] = pd.qcut(df_r['age'], 10, duplicates='drop')
    for decile, mr in df_r.groupby('age_decile', observed=True)['resid'].mean().items():
        print(f"  {decile}: {mr:>8.1f}")

    return result


def additional_analyses(m2_reml, m2_ml, df, y, groups, exog_re):
    print("\nSTEP 9: ADDITIONAL ANALYSES")
    result = {}

    cov_re = m2_reml.cov_re
    corr = cov_re.iloc[0, 1] / np.sqrt(cov_re.iloc[0, 0] * cov_re.iloc[1, 1])
    print(f"\n--- Random Effects Correlation ---")
    print(f"  corr(intercept, slope) = {corr:.4f}")
    if corr < 0:
        print(f"  Interpretation: faster runners (lower intercept) tend to have "
              f"steeper positive slopes (age more)")
    else:
        print(f"  Interpretation: faster runners tend to age more slowly")
    result['re_correlation'] = corr

    print(f"\n--- Gender x Age Interaction ---")
    exog_int = sm.add_constant(df[['age_c', 'female', 'year_c']].copy())
    exog_int['age_c:female'] = df['age_c'].values * df['female'].values

    print("  Fitting interaction model (ML)...")
    int_result = MixedLM(y, exog_int, groups=groups, exog_re=exog_re).fit(
        reml=False, **_FIT_KWARGS)
    print(f"  M2_interaction_ML: converged={int_result.converged}")

    coef = int_result.fe_params['age_c:female']
    se = int_result.bse_fe['age_c:female']
    pval = int_result.pvalues['age_c:female']
    d_aic = int_result.aic - m2_ml.aic
    male_rate = int_result.fe_params['age_c']
    female_rate = male_rate + coef

    print(f"  age_c:female coefficient: {coef:.2f}")
    print(f"  SE: {se:.2f}")
    print(f"  p-value: {pval:.2e}")
    print(f"  dAIC vs base M2: {d_aic:+.1f}")
    print(f"  Male aging rate: {male_rate:.1f} sec/yr")
    print(f"  Female aging rate: {female_rate:.1f} sec/yr")
    if pval < 0.05:
        print(f"  -> Men and women age at significantly different rates")
    else:
        print(f"  -> No significant gender difference in aging rate")

    result.update({
        'interaction_coef': coef, 'interaction_se': se,
        'interaction_p': pval, 'interaction_daic': d_aic,
        'male_aging_rate': male_rate, 'female_aging_rate': female_rate,
    })
    return result


def save_summary(all_results):
    lines = ["Q2 MIXED-EFFECTS MODELING -- RESULTS SUMMARY", "=" * 60]
    for section, data in all_results.items():
        lines.append(f"\n[{section}]")
        if isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"  {k} = {v:.6f}" if isinstance(v, float) else f"  {k} = {v}")
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n')
    print(f"\nSummary saved to {SUMMARY_PATH}")


def main():
    t0 = time.time()
    print("Q2 MIXED-EFFECTS MODELING ANALYSIS")
    print("Boston Marathon 2000-2019")
    print("=" * 70)

    all_results = {}

    df = load_and_filter_data()

    df.sort_values('display_name', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['age_c'] = df['age'] - df['age'].mean()
    df['year_c'] = df['year'] - 2010
    df['female'] = (df['gender'] == 'F').astype(int)
    y = df['seconds'].values
    groups = df['display_name'].values
    exog = sm.add_constant(df[['age_c', 'female', 'year_c']])
    exog_re = sm.add_constant(df[['age_c']])

    icc_result, m1_reml = compute_icc(df, groups, y)
    all_results['ICC'] = icc_result

    all_results['Runner Slopes'] = compute_runner_slopes(df)

    m0, m1_reml, m1_ml, m2_reml, m2_ml = fit_mixed_models(
        df, y, groups, exog, exog_re, m1_reml)

    fe_detail = {}
    for var in m2_reml.fe_params.index:
        fe_detail[f'{var}_coef'] = m2_reml.fe_params[var]
        fe_detail[f'{var}_se'] = m2_reml.bse_fe[var]
        fe_detail[f'{var}_pvalue'] = m2_reml.pvalues[var]
    all_results['Fixed Effects'] = fe_detail

    cov_re = m2_reml.cov_re
    all_results['Random Effects'] = {
        'tau2_intercept': cov_re.iloc[0, 0],
        'tau2_slope': cov_re.iloc[1, 1],
        'cov_intercept_slope': cov_re.iloc[0, 1],
        'corr_intercept_slope': cov_re.iloc[0, 1] / np.sqrt(cov_re.iloc[0, 0] * cov_re.iloc[1, 1]),
        'sigma2_residual': m2_reml.scale,
    }

    rmse_dict = {'rmse_m0': np.sqrt(np.mean(m0.resid ** 2))}
    for label, m in [('m1', m1_reml), ('m2', m2_reml)]:
        rmse_dict[f'rmse_{label}_conditional'] = np.sqrt(np.mean(m.resid ** 2))
        rmse_dict[f'rmse_{label}_marginal'] = np.sqrt(
            np.mean((y - exog.values @ m.fe_params.values) ** 2))
        rmse_dict[f'rmse_{label}_within_runner'] = np.sqrt(m.scale)
    all_results['Model Fit'] = rmse_dict

    all_results['Model Selection'] = model_selection(m0, m1_ml, m2_ml)
    all_results['Variance Decomposition'] = variance_decomposition(m2_reml, df, exog)
    all_results['Diagnostics'] = residual_diagnostics(m2_reml, df)
    all_results['Additional'] = additional_analyses(m2_reml, m2_ml, df, y, groups, exog_re)

    save_summary(all_results)
    print(f"\nTotal runtime: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f} min)")


if __name__ == '__main__':
    main()

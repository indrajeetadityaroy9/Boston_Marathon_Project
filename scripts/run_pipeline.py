"""Run all three research questions and print formatted results.

This is the only file with print statements. RQ1/RQ2/RQ3 modules return data;
this file formats and displays it. Execute: uv run scripts/run_pipeline.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import statsmodels.api as sm

from boston_marathon import config as cfg
from boston_marathon import data, metrics, rq1, rq2, rq3
from boston_marathon.lme4_backend import fit_lmer
from boston_marathon.rq2 import _SM_TO_LME4, _EXOG_COLS, _fe_array


def run_rq1():
    print("DEMOGRAPHIC BASELINE PREDICTION")
    print("Boston Marathon Finish Time -- Pre-Race Prediction from Demographics")
    print("=" * 70)

    print("STEP 1: DATA LOADING")
    df = data.load_cleaned(usecols=['year', 'display_name', 'age', 'gender', 'seconds', 'age_imputed'])
    df = data.filter_non_imputed(df)
    df = data.add_prior_history(df)
    train, test = data.temporal_split(df, cfg.RQ1_TRAIN_YEARS, cfg.RQ1_TEST_YEARS)
    age_mean = train['age'].mean()
    for d in [train, test]:
        data.add_centered_features(d, age_mean, cfg.YEAR_CENTER)
    print(f"  Non-imputed, age-valid, 2000+: {len(df):,} rows")
    print(f"  Age centered at {age_mean:.1f} (train mean)")
    print(f"  Train (2000-2017): {len(train):,} rows ({train['prior_mean_time'].notna().sum():,} with prior history)")
    print(f"  Test  (2018-2019): {len(test):,} rows ({test['prior_mean_time'].notna().sum():,} with prior history)")

    print("\nSTEP 2: MODEL FITTING AND EVALUATION")
    models, results = rq1.fit_models(train, test)
    hist_train = next(r for r in results if r['model'] == 'History OLS' and r['split'] == 'train')
    hist_test = next(r for r in results if r['model'] == 'History OLS' and r['split'] == 'test')
    print(f"\n  History model: Train subset: {hist_train['n']:,} rows, Test subset: {hist_test['n']:,} rows")

    print("\n--- Prediction Results ---")
    print(f"{'Model':<30} {'RMSE (s)':>10} {'RMSE (min)':>11} {'MAE (s)':>9} "
          f"{'MAE (min)':>10} {'R^2':>8} {'MAPE%':>7}")
    print("-" * 90)
    for r in results:
        label = f"{r['model']} [{r['split']}]"
        print(f"  {label:<28} {r['rmse_s']:>10.0f} {r['rmse_min']:>10.1f} "
              f"{r['mae_s']:>9.0f} {r['mae_min']:>9.1f} {r['r2']:>8.4f} {r['mape']:>6.1f}%")

    print("\n--- Model Coefficients ---")
    for label, spec in models.items():
        print(f"\n  {label}:")
        for feat, coef in zip(spec['features'], spec['model'].coef_):
            print(f"    {feat:>20}: {coef:>10.2f}")
        print(f"    {'intercept':>20}: {spec['model'].intercept_:>10.2f}")

    print("\nSTEP 3: CROSS-VALIDATION (5-fold grouped by year, quadratic model)")
    cv = rq1.cross_validate_quadratic(train)
    print(f"  Fold RMSEs: {', '.join(f'{r:.0f}' for r in cv['fold_rmses'])}")
    print(f"  Mean RMSE: {cv['mean_rmse']:.0f} +/- {cv['std_rmse']:.0f}")
    print(f"  Fold R^2s: {', '.join(f'{r:.4f}' for r in cv['fold_r2s'])}")
    print(f"  Mean R^2: {cv['mean_r2']:.4f} +/- {cv['std_r2']:.4f}")

    print("\nSTEP 4: FEATURE IMPORTANCE (standardized coefficients)")
    print(f"\n  {'Feature':>15} {'Std Coef':>10} {'Direction':>10}")
    print("  " + "-" * 38)
    for feat, coef, direction in rq1.standardized_importance(train):
        print(f"  {feat:>15} {coef:>10.1f} {direction:>10}")

    best = min((r for r in results if r['split'] == 'test'), key=lambda r: r['rmse_s'])
    print(f"\n{'='*70}")
    print(f"Best test RMSE: {best['rmse_s']:.0f}s ({best['rmse_min']:.1f} min) -- {best['model']} [test]")


def run_rq2():
    t0 = time.time()
    print("\nPERSONALIZED MIXED-EFFECTS MODELING ANALYSIS")
    print("Boston Marathon Repeat Runners 2000-2019")
    print("=" * 70)

    print("\nSTEP 1: REPEAT-RUNNER SAMPLE CONSTRUCTION")
    df = data.load_cleaned(usecols=['year', 'display_name', 'age', 'gender', 'seconds', 'age_imputed'])
    print(f"\n1. Full dataset: {len(df):,} rows")
    df = data.build_repeat_runner_sample(df)
    print(f"  Final modeling sample: {len(df):,} rows, {df['display_name'].nunique():,} runners")

    print("\nSTEP 2: FEATURE ENGINEERING")
    df.sort_values('display_name', inplace=True)
    df.reset_index(drop=True, inplace=True)
    age_mean = df['age'].mean()
    data.add_centered_features(df, age_mean, cfg.YEAR_CENTER)
    y = df['seconds'].values
    exog = sm.add_constant(df[_EXOG_COLS])
    print(f"  Age centered at {age_mean:.1f}, year centered at {cfg.YEAR_CENTER}")

    print("\nSTEP 3: FITTING MIXED-EFFECTS MODELS")
    df_w = df.copy()
    rq2._add_weather_cols(df_w)
    mdls = rq2.fit_mixed_models(df, weather_df=df_w)
    m0, m1r, m2r = mdls['m0'], mdls['m1_reml'], mdls['m2_reml']
    m3r = mdls['m3_reml']
    print(f"  OLS R-squared={m0.rsquared:.4f}")
    print(f"  Random Intercept converged={mdls['m1_ml']['converged']}, "
          f"Random Intercept+Slope converged={mdls['m2_ml']['converged']}, "
          f"Weather-Augmented converged={mdls['m3_ml']['converged']}")

    print("\nSTEP 4: INTRA-CLASS CORRELATION")
    icc_a = metrics.icc_anova(df, 'display_name', 'seconds')
    print(f"\n1. ANOVA-based ICC(1): {icc_a:.4f} ({icc_a*100:.1f}%)")
    icc_c = m1r['tau2_0'] / (m1r['tau2_0'] + m1r['sigma2'])
    print(f"2. Conditional ICC: {icc_c:.4f} ({icc_c*100:.1f}%)")
    print(f"   tau^2={m1r['tau2_0']:,.0f}, sigma^2={m1r['sigma2']:,.0f}")

    print("\nSTEP 5: PER-RUNNER AGING SLOPES")
    slopes = rq2.compute_runner_slopes(df)
    print(f"\nRepeat-runner sample slopes (age span >=5yr): n={len(slopes)}")
    print(f"  Mean: {slopes.mean():.1f} sec/yr, Std: {slopes.std():.1f}, Median: {slopes.median():.1f}")
    print(f"  Slowing: {(slopes > 0).mean()*100:.1f}%, Improving: {(slopes < 0).mean()*100:.1f}%")

    print("\n--- Fixed Effects (from REML) ---")
    print(f"\n{'':>12} {'OLS':>14} {'Rand Intcpt':>14} {'Intcpt+Slope':>14} {'Intcpt+Slp+Wx':>14}")
    print("-" * 72)
    for var in exog.columns:
        lk = _SM_TO_LME4.get(var, var)
        print(f"  {var:>10} {m0.params[var]:>14.2f} {m1r['fe_params'][lk]:>14.2f} "
              f"{m2r['fe_params'][lk]:>14.2f} {m3r['fe_params'][lk]:>14.2f}")
    for var in ['temp_c', 'humid_c', 'wind_c']:
        print(f"  {var:>10} {'--':>14} {'--':>14} {'--':>14} {m3r['fe_params'][var]:>14.2f}")

    tau2_0, tau2_1, corr_01 = m2r['tau2_0'], m2r['tau2_1'], m2r['corr_01']
    print(f"\n--- Variance Components (Random Intercept + Slope) ---")
    print(f"  tau^2_0={tau2_0:,.0f}, tau^2_1={tau2_1:.2f}, "
          f"cov={corr_01 * np.sqrt(tau2_0 * tau2_1):.2f}, corr={corr_01:.4f}, sigma^2={m2r['sigma2']:,.0f}")
    print(f"\n--- Variance Components (Random Intercept + Slope + Weather) ---")
    print(f"  tau^2_0={m3r['tau2_0']:,.0f}, tau^2_1={m3r['tau2_1']:.2f}, "
          f"corr={m3r['corr_01']:.4f}, sigma^2={m3r['sigma2']:,.0f}")

    print(f"\n--- In-Sample RMSE ---")
    rmse_ols = metrics.rmse(y, exog.values @ m0.params.values)
    print(f"  {'OLS':<30} {rmse_ols:>10.1f}")
    m2r_fe = None
    for label, r in [('Random Intercept', m1r), ('Random Intercept + Slope', m2r)]:
        fe_vals = _fe_array(r['fe_params'], exog.columns)
        if label == 'Random Intercept + Slope': m2r_fe = fe_vals
        print(f"  {label:<30} {r['cond_rmse']:>10.1f}")
    print(f"  {'Random Intcpt + Slope + Weather':<30} {m3r['cond_rmse']:>10.1f}")

    print("\nSTEP 6: MODEL COMPARISON (Likelihood Ratio Tests)")
    rows, lrt01, lrt12 = rq2.model_comparison_table(m0, mdls['m1_ml'], mdls['m2_ml'])
    print(f"  OLS vs Random Intercept:          LR={lrt01['lr_stat']:.2f}, p_corrected={lrt01['p_corrected']:.2e}")
    print(f"  Random Intercept vs Intcpt+Slope:  LR={lrt12['lr_stat']:.2f}, p_corrected={lrt12['p_corrected']:.2e}")
    m3_ml = mdls['m3_ml']
    rows.append(('Intcpt+Slp+Wx', 11, m3_ml['loglik'], m3_ml['aic'], m3_ml['bic']))
    best_aic = min(r[3] for r in rows)
    for name, k, llf, aic, bic in rows:
        print(f"  {name:>14} k={k:>2} AIC={aic:,.1f} BIC={bic:,.1f} (dAIC={aic-best_aic:+.1f})")

    print("\nSTEP 7: VARIANCE DECOMPOSITION")
    vd = metrics.variance_decomposition(m2r, df, m2r_fe)
    for comp, val in [('Fixed Effects', vd['var_fixed']), ('Random Intercepts', vd['tau2_0']),
                      ('Random Slopes', vd['var_slope']), ('Residual', vd['sigma2'])]:
        print(f"  {comp:<20} {val:>14,.0f} ({val/vd['total']:>5.1%})")
    print(f"  Marginal R^2={vd['r2_marginal']:.4f}, Conditional R^2={vd['r2_conditional']:.4f}")

    print("\nSTEP 8: RESIDUAL DIAGNOSTICS")
    diag = metrics.residual_diagnostics(m2r['resid'])
    print(f"  Skewness={diag['skewness']:.4f}, Kurtosis={diag['kurtosis']:.4f}, "
          f"Shapiro p={diag['shapiro_p']:.2e}")
    year_means = df[['year']].assign(resid=m2r['resid']).groupby('year')['resid'].mean()
    flagged = int((year_means.abs() > 100).sum())
    print(f"  Flagged years (|mean resid| > 100s): {flagged}/{len(year_means)}")
    print(f"  Year-level residual std: {year_means.std():.1f}s")

    print("\nSTEP 9: RANDOM EFFECTS SUMMARY")
    print(f"  corr(intercept, slope) = {corr_01:.4f}")

    print("\nSTEP 10: EXPORTING BLUPs (Best Linear Unbiased Predictions)")
    blup_df = rq2.export_blups(m2r, cfg.BLUP_CSV)
    print(f"  Exported {len(blup_df):,} runner BLUPs to {cfg.BLUP_CSV.name}")
    print(f"  Intercept: mean={blup_df['blup_intercept'].mean():.1f}, std={blup_df['blup_intercept'].std():.1f}")
    print(f"  Slope: mean={blup_df['blup_slope'].mean():.2f}, std={blup_df['blup_slope'].std():.2f}")

    print("\nSTEP 11: PREDICTION EVALUATION (Value of Personalization)")
    pe = rq2.evaluate_personalization(m2r, y, df)
    print(f"  Random Intercept + Slope:           Marginal RMSE={pe['marg_rmse']:.1f}, "
          f"Conditional RMSE={pe['cond_rmse']:.1f}, "
          f"Value of Personalization={pe['personalization']:.0f}s ({pe['personalization']/60:.1f} min)")
    pe_w = rq2.evaluate_personalization(m3r, y, df_w, exog_cols=rq2._WEATHER_EXOG_COLS)
    print(f"  Random Intcpt + Slope + Weather:     Marginal RMSE={pe_w['marg_rmse']:.1f}, "
          f"Conditional RMSE={pe_w['cond_rmse']:.1f}, "
          f"Value of Personalization={pe_w['personalization']:.0f}s ({pe_w['personalization']/60:.1f} min)")
    print(f"  90% Prediction Interval: Known runner={pe['pi_known']:.0f}s ({pe['pi_known']/60:.0f} min), "
          f"New runner={pe['pi_new']:.0f}s ({pe['pi_new']/60:.0f} min)")

    print("\nSTEP 12: TEMPORAL HOLD-OUT (Out-of-Sample Evaluation)")
    ho = rq2.temporal_holdout(df)
    print(f"  Train: {ho['n_train']:,} obs, {ho['n_train_runners']:,} runners")
    print(f"  Random Intercept + Slope Known:          cond RMSE={ho['cond_rmse_known']:.1f}, marg RMSE={ho['marg_rmse_known']:.1f}")
    print(f"  Random Intercept + Slope New:             marg RMSE={ho['marg_rmse_new']:.1f}")
    print(f"  Random Intercept + Slope Personalization: {ho['marg_rmse_known'] - ho['cond_rmse_known']:.1f}s (out-of-sample)")
    print(f"  Random Intcpt + Slope + Weather Known:    cond RMSE={ho['w_cond_rmse_known']:.1f}, marg RMSE={ho['w_marg_rmse_known']:.1f}")
    print(f"  Random Intcpt + Slope + Weather New:      marg RMSE={ho['w_marg_rmse_new']:.1f}")
    print(f"  Random Intcpt + Slope + Weather Personalization: {ho['w_marg_rmse_known'] - ho['w_cond_rmse_known']:.1f}s (out-of-sample)")
    print(f"  Exported {ho['n_blups_exported']:,} leak-free BLUPs")

    print("\nSTEP 13: WEATHER-AUGMENTED MODEL (temperature + humidity + wind)")
    w = rq2.sensitivity_weather(df)
    print(f"  temperature:  {w['temp_coef']:>8.1f} sec/deg C (SE={w['temp_se']:.1f}, p={w['temp_p']:.2e})")
    print(f"  humidity:     {w['humid_coef']:>8.2f} sec/%     (SE={w['humid_se']:.2f}, p={w['humid_p']:.2e})")
    print(f"  wind:         {w['wind_coef']:>8.1f} sec/mph   (SE={w['wind_se']:.1f}, p={w['wind_p']:.2e})")
    print(f"  Flagged years: {w['flagged']}/20, Year-level std: {w['year_std']:.1f}s, Conditional RMSE: {w['cond_rmse']:.1f}s")

    print("\nSTEP 14: LOG-TRANSFORM SENSITIVITY")
    lg = rq2.sensitivity_log(df)
    print(f"  Log residuals: skewness={lg['skewness']:.4f}, kurtosis={lg['kurtosis']:.4f}")
    print(f"  Back-transformed RMSE: {lg['rmse_bt']:.1f}s, Duan smearing factor: {lg['smearing_factor']:.4f}")

    print("\nSTEP 15: SPLINE AGE COMPARISON")
    sp = rq2.sensitivity_spline(df)
    print(f"  Quadratic AIC={sp['quad_aic']:.1f}, Spline AIC={sp['spline_aic']:.1f}, delta AIC={sp['daic']:+.1f}")
    print(f"  Quadratic RMSE={sp['quad_rmse']:.1f}, Spline RMSE={sp['spline_rmse']:.1f}")

    print("\nSTEP 16: SURVIVAL BIAS CHECK")
    sb = rq2.sensitivity_survival(df)
    print(f"  Active runners (n={sb['n_active']:,}): {sb['slope_active']:.1f} sec/yr")
    print(f"  Dropped runners (n={sb['n_dropped']:,}): {sb['slope_dropped']:.1f} sec/yr")
    print(f"  Difference: {sb['diff']:.1f} sec/yr ({sb['pct_diff']:.1f}%)")

    print(f"\nTotal runtime: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f} min)")


def run_rq3():
    print("\nPROGRESSIVE IN-RACE PREDICTION")
    print("Boston Marathon -- Finish Time Prediction from Checkpoint Splits")
    print("=" * 70)

    print("STEP 1: DATA LOADING")
    train, test = data.load_splits_with_blups()
    n_blup_total = train['blup_intercept'].notna().sum() + test['blup_intercept'].notna().sum()
    n_blup_test = test['blup_intercept'].notna().sum()
    total = len(train) + len(test)
    print(f"  Runners with complete splits: {total:,}")
    print(f"  Leak-free BLUPs: {n_blup_total:,} ({n_blup_total/total*100:.1f}%) total, "
          f"{n_blup_test:,} ({n_blup_test/len(test)*100:.1f}%) in test")
    print(f"  Train: {len(train):,}, Test: {len(test):,}")

    print("\nSTEP 2: PROGRESSIVE CHECKPOINT PREDICTION")
    results_df = rq3.run_progressive(train, test)

    print("\n--- Prediction Convergence (test set 2017) ---")
    pivot = results_df[results_df['model'].isin([cfg.NAIVE, cfg.SPLITS, cfg.DEMO])].pivot(
        index='checkpoint', columns='model', values='rmse').reindex(cfg.CP_ORDER)
    print(f"\n{'Checkpoint':>12} {'Naive':>8} {'Splits':>8} {'Demo':>8}")
    print("-" * 42)
    for cp in cfg.CP_ORDER:
        print(f"  {cp:>10} {pivot.loc[cp, cfg.NAIVE]:>6.0f} {pivot.loc[cp, cfg.SPLITS]:>6.0f} "
              f"{pivot.loc[cp, cfg.DEMO]:>6.0f}")

    blup_r = results_df[results_df['model'].isin([cfg.FULL, cfg.SPLITS_SUBSET])]
    print(f"\n--- History Augmentation (n={blup_r['n_test'].iloc[0]:,} runners with BLUPs) ---")
    for cp in cfg.CP_ORDER:
        sr = blup_r[(blup_r['checkpoint'] == cp) & (blup_r['model'] == cfg.SPLITS_SUBSET)]['rmse'].values[0]
        fr = blup_r[(blup_r['checkpoint'] == cp) & (blup_r['model'] == cfg.FULL)]['rmse'].values[0]
        print(f"  {cp:>10} splits={sr:>6.0f} full={fr:>6.0f} improvement={sr - fr:>+6.0f}")

    print("\nSTEP 3: FEATURE IMPORTANCE (standardized Ridge coefficients)")
    for cp_label, coefs in rq3.feature_importance(train).items():
        print(f"\n  At {cp_label}:")
        for feat, coef in coefs[:5]:
            print(f"    {feat:>15}: {coef:>10.1f}")

    print("\nSTEP 4: CROSSOVER ANALYSIS (splits vs personalized prediction)")
    xover = rq3.crossover_analysis(results_df)
    print(f"  Personalized RMSE benchmark (Random Intercept + Slope in-sample): {cfg.PERSONALIZED_RMSE:.0f}s")
    for _, r in xover.iterrows():
        marker = " <-- CROSSOVER" if r['is_crossover'] else ""
        print(f"  {r['checkpoint']:>10} {r['rmse']:>6.0f}s {'YES' if r['beats'] else 'no':>6}{marker}")

    print("\nSTEP 4b: CUMULATIVE SPLITS vs SINGLE CHECKPOINT")
    cvs, lo, hi = rq3.cumulative_vs_single(results_df)
    for _, r in cvs.iterrows():
        print(f"  {r['checkpoint']:>10} cumulative={r['cumul_rmse']:>6.0f} single={r['single_rmse']:>6.0f} advantage={r['advantage']:>+6.0f}")
    print(f"  Mid-race advantage: {lo:.0f} to {hi:.0f} seconds")

    print("\nSTEP 4c: YEAR FEATURE DEGRADATION (extrapolation to unseen test year)")
    yd, max_d = rq3.year_degradation(results_df)
    for _, r in yd.iterrows():
        print(f"  {r['checkpoint']:>10} without_year={r['no_year']:>6.0f} with_year={r['with_year']:>6.0f} degradation={r['degradation']:>+6.0f}")
    print(f"  Maximum degradation: {max_d:.0f} seconds")

    print("\n--- 90% Prediction Interval Width Convergence ---")
    for _, r in rq3.pi_convergence(results_df).iterrows():
        print(f"  {r['checkpoint']:>10} RMSE={r['rmse']:>6.0f} Prediction Interval={r['pi_width']:>6.0f}s ({r['pi_min']:.1f} min)")

    print("\nSTEP 5: QUANTILE REGRESSION PREDICTION INTERVALS (90% asymmetric)")
    qpi = rq3.quantile_pi(train, test)
    print(f"\n{'Checkpoint':>12} {'Pctl Width':>11} {'Pctl Coverage':>14} {'QR Width':>9} {'QR Coverage':>12}")
    print("-" * 62)
    for _, r in qpi.iterrows():
        print(f"  {r['checkpoint']:>10} {r['pctl_width']:>9.0f}s {r['pctl_coverage']:>13.1%} "
              f"{r['qr_width']:>7.0f}s {r['qr_coverage']:>11.1%}")

    splits_idx = results_df[results_df['model'] == cfg.SPLITS].set_index('checkpoint')['rmse']
    print(f"\n{'='*70}")
    print(f"Prediction improves from RMSE={splits_idx['5K']:.0f}s at 5K "
          f"to {splits_idx['HALF']:.0f}s at halfway to {splits_idx['40K']:.0f}s at 40K")


def main():
    run_rq1()
    run_rq2()
    run_rq3()


if __name__ == '__main__':
    main()

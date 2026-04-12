"""Run the full Boston Marathon finish-time analysis pipeline."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import root_mean_squared_error as rmse

from src import config, splits, ablation, regression, data, mixed_effects, calibration, figures, tables
from src.inference import bootstrap_rmse_comparison


def spawn_rngs():
    """Spawn independent RNGs so stage order does not change results."""
    labels = ('ridge', 'cqr', 'never_seen', 'blup_5k', 'ablation', 'calibration')
    return dict(zip(labels, np.random.default_rng(0).spawn(len(labels))))


def compute_scales():
    """Compute year-drift scale and data-derived diagnostics."""
    df = data.load_processed_results(columns=['year', 'seconds'])
    drift = df[df['year'].between(2000, 2019)].groupby('year')['seconds'].mean().diff()
    scale = float(drift.abs().median())

    sdf = data.load_processed_results(columns=['year', 'seconds'] + config.CUMULATIVE_SPLIT_TIME_COLUMNS).dropna(subset=config.CUMULATIVE_SPLIT_TIME_COLUMNS)
    sd = float(sdf['seconds'].std())
    r5k = float(sdf['5k_seconds'].corr(sdf['seconds']))
    resid_sd = sd * np.sqrt(1 - r5k ** 2)
    shift = float(sdf.loc[sdf['year'] == config.IN_RACE_SPLIT_PREDICTION_TEST_YEAR, 'seconds'].mean()) - float(sdf.loc[sdf['year'].isin(config.IN_RACE_SPLIT_PREDICTION_TRAIN_YEARS), 'seconds'].mean())

    print("DATA-DERIVED SCALES")
    print(f"  year_drift_scale = {scale:8.2f} s")
    print(f"  5k_residual_sd   = {resid_sd:8.2f} s")
    print(f"  train_test_shift = {shift:8.2f} s\n")
    return scale, resid_sd, shift


def fit_pre_race_models():
    """Fit pre-race regression models and return fitted artifacts."""
    print("PRE-RACE REGRESSION (Section 3)")
    print("=" * 80)
    raw = data.load_processed_results(columns=['year', 'display_name', 'age', 'gender', 'seconds', 'age_imputed'])
    pre = raw[~raw['age_imputed'] & raw['age'].notna() & (raw['year'] >= 2000)].copy()
    pre['log_seconds'] = np.log(pre['seconds'].to_numpy())
    pre = data.add_prior_boston_history_features(pre)
    train = pre[pre['year'].isin(config.PRE_RACE_REGRESSION_TRAIN_YEARS)].copy()
    test = pre[pre['year'].isin(config.PRE_RACE_REGRESSION_TEST_YEARS)].copy()
    age_mean = float(train['age'].mean())
    data.add_centered_pre_race_features(train, age_mean)
    data.add_centered_pre_race_features(test, age_mean)
    print(f"  train = {len(train):,} rows, test = {len(test):,} rows")

    demo_fit = regression.fit_hc3_robust_log_seconds_regression(train, regression.PRE_RACE_DEMOGRAPHIC_FEATURES)
    train_hist = train[train['prior_mean_time'].notna()].copy()
    test_hist = test[test['prior_mean_time'].notna()].copy()
    hist_fit = regression.fit_hc3_robust_log_seconds_regression(train_hist, regression.PRE_RACE_DEMOGRAPHIC_AND_HISTORY_FEATURES)
    print(f"  history-eligible: train = {len(train_hist):,}, test = {len(test_hist):,}")

    demo_eval = regression.evaluate_log_seconds_regression(*demo_fit, test_hist, regression.PRE_RACE_DEMOGRAPHIC_FEATURES)
    hist_eval = regression.evaluate_log_seconds_regression(*hist_fit, test_hist, regression.PRE_RACE_DEMOGRAPHIC_AND_HISTORY_FEATURES)
    print(f"  demographic model RMSE = {demo_eval['rmse_seconds']:7.1f} s  cal_slope = {demo_eval['calibration_slope']:.4f}")
    print(f"  history model RMSE     = {hist_eval['rmse_seconds']:7.1f} s  cal_slope = {hist_eval['calibration_slope']:.4f}\n")

    return {'demo_fit': demo_fit, 'hist_fit': hist_fit, 'age_mean': age_mean, 'train_mean': float(train['seconds'].mean()), 'pool': pre}


def fit_mixed_effects_model():
    """Fit the runner mixed-effects model and return fitted artifacts."""
    print("MIXED-EFFECTS PERSONALIZATION (Section 4)")
    print("=" * 80)
    raw = data.load_processed_results(columns=['year', 'display_name', 'age', 'gender', 'seconds', 'age_imputed'])
    repeat = data.build_repeat_runner_analysis_sample(raw)
    print(f"  repeat pool = {len(repeat):,} rows, {repeat['display_name'].nunique():,} runners")

    me = mixed_effects.fit_temporal_holdout_runner_mixed_effects(repeat)
    print(f"  test_known = {len(me['test_known']):,}  test_never_seen = {me['n_test_never_seen']:,}")
    print(f"  runner effects = {len(me['runner_random_effects_leakfree']):,}")
    print(f"  marginal  RMSE test_known  = {me['marginal_mixed_effects_rmse_on_test_known']:7.1f} s")
    print(f"  conditional RMSE test_known= {me['conditional_mixed_effects_rmse_on_test_known']:7.1f} s")
    print(f"  marginal  RMSE never-seen  = {me['marginal_mixed_effects_rmse_on_never_seen']:7.1f} s")
    print(f"  Var(fixed)     = {me['variance_explained_by_fixed_effects_only']:.4f}")
    print(f"  Var(fixed+rand)= {me['variance_explained_by_fixed_and_random_effects']:.4f}")
    slope_ci = me['random_slope_standard_deviation_confidence_interval']
    print(f"  tau_slope 95% CI = ({slope_ci[0]:.6f}, {slope_ci[1]:.6f})\n")
    return me


def fit_checkpoint_models(train_df, test_df, me_conditional_rmse):
    """Fit ridge models at each checkpoint and print RMSE curve."""
    print("IN-RACE PREDICTION FROM SPLITS (Section 5)")
    print("=" * 80)
    n_holders = int(test_df['runner_intercept'].notna().sum())
    print(f"  train = {len(train_df):,}  test = {len(test_df):,}  holders = {n_holders:,}  never-seen = {len(test_df) - n_holders:,}")

    cp_df, models = splits.fit_checkpoint_ridge_models(train_df, test_df)
    rmse_curve = cp_df[cp_df['variant'] == 'no_runner_history'].set_index('checkpoint')['rmse_seconds'].reindex(config.CHECKPOINT_LABELS)
    print(f"\n  Checkpoint RMSE (ridge, no runner history):")
    for lbl in config.CHECKPOINT_LABELS:
        print(f"    {lbl:>10}  {rmse_curve.loc[lbl]:9.1f}")

    cross = next((l for l in config.CHECKPOINT_LABELS if rmse_curve.loc[l] < me_conditional_rmse), None)
    print(f"\n  Crossover (< mixed-effects {me_conditional_rmse:.1f}): {cross}\n")
    return {'models': models, 'cp_df': cp_df}


def run_ablation(pre_state, me_state, irace_state, test_df, scale, rngs):
    """Run the M0-M5 nested model comparison and supplementary analyses."""
    print("NESTED MODEL COMPARISON (Section 6)")
    print("=" * 80)

    abl = ablation.evaluate_nested_model_comparison(
        pre_state['demo_fit'], pre_state['hist_fit'], pre_state['train_mean'], pre_state['age_mean'],
        me_state['fixed_effects_coefficients_leakfree'], me_state['age_mean_leakfree_from_training'],
        me_state['runner_random_effects_leakfree'], me_state['duan_smearing_factor'],
        irace_state['models'], rngs['ablation'],
    )
    print(f"  common test = {int(abl['n'].iloc[0]):,} rows\n")
    hdr = f"  {'stage':>5}  {'model':<54}  {'rmse':>8}  {'rmse 95% CI':>18}  {'delta':>8}  {'delta CI':>18}  {'x scale':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in abl.iterrows():
        if pd.isna(r['improvement_seconds']):
            print(f"  {r['stage']:>5}  {r['description']:<54}  {r['rmse_seconds']:8.1f}  [{r['rmse_ci_lower']:7.1f}, {r['rmse_ci_upper']:7.1f}]")
        else:
            xs = f"{r['improvement_seconds'] / scale:+.2f}x"
            print(f"  {r['stage']:>5}  {r['description']:<54}  {r['rmse_seconds']:8.1f}  [{r['rmse_ci_lower']:7.1f}, {r['rmse_ci_upper']:7.1f}]  {r['improvement_seconds']:+8.1f}  [{r['improvement_ci_lower']:+7.1f}, {r['improvement_ci_upper']:+7.1f}]  {xs:>7}")
    print("\n  Note: M0-M3 use log(seconds) + Duan smearing; M4-M5 use raw-seconds Ridge.")

    # Supplementary: BLUP contribution at 5K (do runner effects help on top of splits?)
    td = test_df.copy()
    models = irace_state['models']
    m5k = models[('5K', 'no_runner_history')]
    m5k_h = models[('5K', 'with_runner_history')]
    td['ridge_pred'] = m5k['ridge_regression_model'].predict(td[m5k['features']].to_numpy())
    holder_mask = td['runner_intercept'].notna()
    holders = td[holder_mask].copy()
    holders['ridge_hist_pred'] = m5k_h['ridge_regression_model'].predict(holders[m5k_h['features']].to_numpy())

    blup = bootstrap_rmse_comparison(
        holders['seconds'].to_numpy(float), holders['ridge_pred'].to_numpy(),
        holders['ridge_hist_pred'].to_numpy(), holders['display_name'].to_numpy(), rngs['blup_5k'],
    )
    print(f"\n  Supplementary: BLUP contribution at 5K (holders only, n={len(holders):,})")
    print(f"    ridge without runner history = {blup['rmse_a']:7.1f} s [{blup['rmse_a_ci'][0]:7.1f}, {blup['rmse_a_ci'][1]:7.1f}]")
    print(f"    ridge with runner history    = {blup['rmse_b']:7.1f} s [{blup['rmse_b_ci'][0]:7.1f}, {blup['rmse_b_ci'][1]:7.1f}]")
    print(f"    delta RMSE                   = {blup['delta']:+7.1f} s [{blup['delta_ci'][0]:+7.1f}, {blup['delta_ci'][1]:+7.1f}]")

    # Supplementary: never-seen runners (do splits suffice without any prior data?)
    never = td[~holder_mask].copy()
    never['age_centered'] = never['age'] - me_state['age_mean_leakfree_from_training']
    never['marginal_pred'] = np.exp(mixed_effects.compute_marginal_log_seconds_prediction(
        never, me_state['fixed_effects_coefficients_leakfree'],
    )) * me_state['duan_smearing_factor']

    ns_comp = bootstrap_rmse_comparison(
        never['seconds'].to_numpy(float), never['ridge_pred'].to_numpy(),
        never['marginal_pred'].to_numpy(), never['display_name'].to_numpy(), rngs['never_seen'],
    )
    print(f"\n  Supplementary: never-seen runners at 5K (n={len(never):,})")
    print(f"    ridge at 5K      = {ns_comp['rmse_a']:7.1f} s [{ns_comp['rmse_a_ci'][0]:7.1f}, {ns_comp['rmse_a_ci'][1]:7.1f}]")
    print(f"    marginal ME      = {ns_comp['rmse_b']:7.1f} s [{ns_comp['rmse_b_ci'][0]:7.1f}, {ns_comp['rmse_b_ci'][1]:7.1f}]")
    print(f"    delta RMSE       = {ns_comp['delta']:+7.1f} s [{ns_comp['delta_ci'][0]:+7.1f}, {ns_comp['delta_ci'][1]:+7.1f}]\n")

    return abl, blup, ns_comp


def run_calibration(pre_state, me_state, train_df, test_df, rngs):
    """Run uncertainty quantification diagnostics."""
    print("UNCERTAINTY QUANTIFICATION (Section 7)")
    print("=" * 80)
    print("  Testing whether interval estimates are honest under temporal shift.\n")

    # Pre-race coverage on 2017
    pool = pre_state['pool']
    tp = pool[pool['year'].isin(config.PRE_RACE_REGRESSION_TEST_YEARS) & pool['prior_mean_time'].notna()].copy()
    tp['log_seconds'] = np.log(tp['seconds'].to_numpy())
    data.add_centered_pre_race_features(tp, pre_state['age_mean'])
    obs_pre = tp['seconds'].to_numpy(float)

    cov_rows = []
    for label, fit_key, feat in [('demographic', 'demo_fit', regression.PRE_RACE_DEMOGRAPHIC_FEATURES), ('demographic + history', 'hist_fit', regression.PRE_RACE_DEMOGRAPHIC_AND_HISTORY_FEATURES)]:
        reg, duan = pre_state[fit_key]
        q05, q95 = np.percentile(np.asarray(reg.resid, float), [5, 95])
        X = np.column_stack([np.ones(len(tp)), tp[feat].to_numpy(float)])
        pred_log = reg.predict(X)
        lo, hi = np.exp(pred_log + q05) * duan, np.exp(pred_log + q95) * duan
        cov_rows.append({'method': f'pre-race {label}', 'coverage': float(((obs_pre >= lo) & (obs_pre <= hi)).mean())})

    # Mixed-effects coverage filtered to 2017
    me_resid, me_duan = me_state['mixed_model_residuals'], me_state['duan_smearing_factor']
    mask_2017 = (me_state['test_known']['year'] == 2017).to_numpy()
    me_pred = me_state['pred_seconds_conditional_test_known'][mask_2017]
    me_obs = me_state['test_known'].loc[mask_2017, 'seconds'].to_numpy(float)
    cond_log = np.log(me_pred / me_duan)
    q05_me, q95_me = np.percentile(np.asarray(me_resid, float), [5, 95])
    me_lo, me_hi = np.exp(cond_log + q05_me) * me_duan, np.exp(cond_log + q95_me) * me_duan
    cov_rows.append({'method': 'mixed-effects conditional', 'coverage': float(((me_obs >= me_lo) & (me_obs <= me_hi)).mean())})

    # CQR coverage (computed here, not in fit_checkpoint_models)
    cqr_df = splits.compute_split_conformal_quantile_intervals(train_df, test_df, rngs['cqr'])
    for _, row in cqr_df.iterrows():
        cov_rows.append({'method': f"CQR {row['checkpoint']}", 'coverage': row['split_conformal_coverage']})

    print("  Coverage diagnostic (nominal 90%):")
    for r in cov_rows:
        print(f"    {r['method']:<30} {r['coverage']:8.1%}")
    print(f"\n  All below 90%: {all(r['coverage'] < 0.90 for r in cov_rows)}")

    # Conformity score shift
    ss = calibration.compute_conformity_score_shift(train_df, test_df, rngs['calibration'])
    print(f"\n  Conformity score shift (5K): scale_ratio={ss['scale_ratio']:.3f}  KS={ss['ks_statistic']:.4f}  p={ss['ks_pvalue']:.2e}")

    # Honest recalibration
    recal = calibration.run_honest_recalibration_experiments(train_df, test_df, rngs['calibration'])
    print(f"\n  Honest recalibration:")
    print(f"    {'method':<20} {'checkpoint':>10} {'baseline':>9} {'corrected':>10} {'width':>8}")
    for _, r in recal.iterrows():
        print(f"    {r['method']:<20} {r['checkpoint']:>10} {r['baseline_coverage']:8.1%} {r['corrected_coverage']:9.1%} {r['interval_width']:7.0f}")
    print(f"\n  Best corrected: {recal['corrected_coverage'].max():.1%} (target 90%)\n")
    return {'coverage_rows': cov_rows, 'score_shift': ss, 'recalibration_df': recal, 'cqr_df': cqr_df}


def compute_data_summary_stats():
    """Compute ICC and log-skewness for the data summary table."""
    df = data.load_processed_results(columns=['year', 'display_name', 'age', 'gender', 'seconds', 'age_imputed'])
    clean = df[~df['age_imputed'] & df['age'].notna() & (df['year'] >= 2000)]
    log_skewness = float(skew(np.log(clean['seconds'].to_numpy())))

    repeat = clean[clean.groupby('display_name')['display_name'].transform('size') > 1].copy()
    k = repeat['display_name'].nunique()
    n = len(repeat)
    grp = repeat.groupby('display_name')['seconds']
    gm = grp.transform('mean')
    grand = repeat['seconds'].mean()
    msb = ((gm - grand) ** 2).sum() / (k - 1)
    msw = ((repeat['seconds'] - gm) ** 2).sum() / (n - k)
    k0 = (n - (grp.size() ** 2).sum() / n) / (k - 1)
    icc = float((msb - msw) / (msb + (k0 - 1) * msw))

    return {
        'n_rows': len(df), 'n_columns': 28, 'years': '1897-2019',
        'n_distinct_runners': df['display_name'].nunique(),
        'n_complete_splits': int(data.load_processed_results(columns=config.CUMULATIVE_SPLIT_TIME_COLUMNS).dropna().shape[0]),
        'icc': icc, 'log_skewness': log_skewness,
    }


def generate_outputs(scale, resid_sd, shift, abl, blup, ns_comp, irace_state, cal, stats):
    """Generate all publication artifacts."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    figures.plot_ablation_bar(abl, scale, config.FIGURES_DIR / 'fig1_ablation.pdf')
    figures.plot_checkpoint_rmse(irace_state['cp_df'], config.FIGURES_DIR / 'fig2_checkpoint_rmse.pdf')
    figures.plot_coverage_diagnostic(cal['coverage_rows'], config.FIGURES_DIR / 'fig3_coverage.pdf')
    figures.plot_conformity_shift(cal['score_shift']['cal_scores'], cal['score_shift']['test_scores'], config.FIGURES_DIR / 'fig4_conformity_shift.pdf')

    tables.generate_data_summary(stats, config.TABLES_DIR / 'tab_data_summary.tex')
    tables.generate_ablation_table(abl, scale, config.TABLES_DIR / 'tab_ablation.tex')
    tables.generate_checkpoint_table(irace_state['cp_df'], config.TABLES_DIR / 'tab_checkpoint_rmse.tex')
    tables.generate_coverage_table(cal['coverage_rows'], config.TABLES_DIR / 'tab_coverage.tex')
    tables.generate_recalibration_table(cal['recalibration_df'], config.TABLES_DIR / 'tab_recalibration.tex')

    metrics = {
        'ablation': abl.to_dict(orient='records'),
        'year_drift_scale_seconds': scale,
        'five_k_residual_sd_seconds': resid_sd,
        'train_test_shift_seconds': shift,
        'cqr_coverage': cal['cqr_df'].to_dict(orient='records'),
        'checkpoint_rmse': irace_state['cp_df'].to_dict(orient='records'),
        'blup_contribution': blup,
        'never_seen_comparison': ns_comp,
        'coverage_diagnostic': cal['coverage_rows'],
        'recalibration': cal['recalibration_df'].to_dict(orient='records'),
        'score_shift': {k: v for k, v in cal['score_shift'].items() if k not in ('cal_scores', 'test_scores')},
    }
    config.PIPELINE_METRICS_JSON.parent.mkdir(exist_ok=True)
    with open(config.PIPELINE_METRICS_JSON, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"Artifacts written to {config.PIPELINE_METRICS_JSON.parent.resolve()}/")


def run_pipeline():
    rngs = spawn_rngs()
    scale, resid_sd, shift = compute_scales()
    pre = fit_pre_race_models()
    me = fit_mixed_effects_model()

    re_df = me['runner_random_effects_leakfree']
    train_df, test_df = data.load_in_race_split_dataset(re_df)

    irace = fit_checkpoint_models(train_df, test_df, me['conditional_mixed_effects_rmse_on_test_known'])
    abl, blup, ns_comp = run_ablation(pre, me, irace, test_df, scale, rngs)
    cal = run_calibration(pre, me, train_df, test_df, rngs)

    stats = compute_data_summary_stats()
    generate_outputs(scale, resid_sd, shift, abl, blup, ns_comp, irace, cal, stats)


if __name__ == '__main__':
    run_pipeline()

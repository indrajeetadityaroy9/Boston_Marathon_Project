"""Nested model comparison on the common 2017 split-test subset."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error as rmse

from boston_marathon import config
from boston_marathon.inference import compute_bca_cluster_bootstrap_interval
from boston_marathon.regression import PRE_RACE_DEMOGRAPHIC_AND_HISTORY_FEATURES, PRE_RACE_DEMOGRAPHIC_FEATURES
from boston_marathon.data import add_centered_pre_race_features, add_prior_boston_history_features, load_processed_results
from boston_marathon.mixed_effects import compute_marginal_log_seconds_prediction

MODELS = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
MODEL_DESCRIPTIONS = {'M0': 'Mean baseline', 'M1': '+ demographics (quadratic regression, HC3-robust)', 'M2': '+ prior Boston history (history regression, HC3-robust)', 'M3': '+ runner random effects (mixed-effects conditional)', 'M4': '+ cumulative splits through 5K (ridge regression)', 'M5': '+ cumulative splits through 40K (ridge regression)'}


def evaluate_nested_model_comparison(demo_fit, hist_fit, train_mean, age_mean_train, fe_coefs, age_mean_lf, re_df, duan, fitted_models, rng):
    """Evaluate nested models M0-M5 on one common test set with a joint BCa bootstrap."""
    raw = load_processed_results(['year', 'display_name', 'age', 'gender', 'seconds', 'age_imputed'] + config.CUMULATIVE_SPLIT_TIME_COLUMNS)
    pre = add_prior_boston_history_features(raw[~raw['age_imputed'] & raw['age'].notna() & (raw['year'] >= 2000)].copy())

    lf_re = re_df.reset_index()
    ct = pre[(pre['year'] == config.IN_RACE_SPLIT_PREDICTION_TEST_YEAR) & pre[config.CUMULATIVE_SPLIT_TIME_COLUMNS].notna().all(axis=1)].merge(lf_re, on='display_name', how='left').copy()
    add_centered_pre_race_features(ct, age_mean_train)
    ct['age_centered_leakfree'] = ct['age'] - age_mean_lf
    obs = ct['seconds'].to_numpy(float)
    n = len(ct)

    demo_results, demo_duan = demo_fit
    hist_results, hist_duan = hist_fit

    lf_ct = ct.assign(age_centered=ct['age_centered_leakfree'])
    re_ct = re_df.reindex(ct['display_name'].values)
    ri = re_ct['runner_intercept'].fillna(0).to_numpy()
    rs = re_ct['runner_age_slope'].fillna(0).to_numpy()
    cond_log_m3 = compute_marginal_log_seconds_prediction(lf_ct, fe_coefs) + ri + rs * ct['age_centered_leakfree'].to_numpy()

    def pred_reg(fit, duan_f, feat_cols):
        X = sm.add_constant(ct[feat_cols].to_numpy(float))
        return np.exp(fit.predict(X)) * duan_f

    m1_pred = pred_reg(demo_results, demo_duan, PRE_RACE_DEMOGRAPHIC_FEATURES)
    m2_full = pred_reg(hist_results, hist_duan, PRE_RACE_DEMOGRAPHIC_AND_HISTORY_FEATURES)
    m2_pred = np.where(ct['prior_mean_time'].notna(), m2_full, m1_pred)

    preds = {
        'M0': np.full(n, train_mean),
        'M1': m1_pred,
        'M2': m2_pred,
        'M3': np.exp(cond_log_m3) * duan,
        'M4': fitted_models[('5K', 'no_runner_history')]['ridge_regression_model'].predict(ct[fitted_models[('5K', 'no_runner_history')]['features']].to_numpy()),
        'M5': fitted_models[('40K', 'no_runner_history')]['ridge_regression_model'].predict(ct[fitted_models[('40K', 'no_runner_history')]['features']].to_numpy()),
    }

    bdf = ct.reset_index(drop=True)
    for lbl, p in preds.items():
        bdf[f'_pred_{lbl}'] = p
    bdf['_y'] = obs

    def stat(sub):
        y = sub['_y'].to_numpy()
        rmses = np.array([float(rmse(y, sub[f'_pred_{lbl}'].to_numpy())) for lbl in MODELS])
        return np.concatenate([rmses, rmses[:-1] - rmses[1:]])

    pt, lo, hi = compute_bca_cluster_bootstrap_interval(stat, bdf, 'display_name', rng=rng)
    nm = len(MODELS)
    return pd.DataFrame([{
        'stage': lbl, 'description': MODEL_DESCRIPTIONS[lbl], 'n': n,
        'rmse_seconds': float(pt[i]), 'rmse_ci_lower': float(lo[i]), 'rmse_ci_upper': float(hi[i]),
        'improvement_seconds': np.nan if i == 0 else float(pt[nm + i - 1]),
        'improvement_ci_lower': np.nan if i == 0 else float(lo[nm + i - 1]),
        'improvement_ci_upper': np.nan if i == 0 else float(hi[nm + i - 1]),
    } for i, lbl in enumerate(MODELS)])

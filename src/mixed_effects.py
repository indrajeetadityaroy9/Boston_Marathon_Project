"""Runner mixed-effects analysis for repeat Boston Marathon participants."""
import os
import subprocess
os.environ['R_HOME'] = subprocess.check_output(['R', 'RHOME'], text=True).strip()

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import conversion, default_converter, numpy2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import root_mean_squared_error as rmse

from . import config
from .data import add_centered_pre_race_features

_conv = default_converter + numpy2ri.converter
importr('lme4')

_COEF_NAMES = ['(Intercept)', 'age_centered', 'I(age_centered^2)', 'female', 'year_centered', 'age_centered:female', 'I(age_centered^2):female']


def fit_lme4_mixed_effects_model(df, formula, reml=True):
    """Fit lme4::lmer and return components used by the pipeline."""
    num_cols = list(df.select_dtypes('number').columns)
    str_cols = [c for c in df.columns if c not in num_cols]
    with conversion.localconverter(_conv):
        for i, c in enumerate(num_cols):
            ro.globalenv[f'.c{i}'] = df[c].to_numpy()
    for i, c in enumerate(str_cols):
        ro.globalenv[f'.s{i}'] = ro.StrVector(df[c].astype(str).to_list())

    df_parts = [f'`{c}`=.c{i}' for i, c in enumerate(num_cols)] + [f'`{c}`=.s{i}' for i, c in enumerate(str_cols)]
    ro.r(f'.df <- data.frame({", ".join(df_parts)}, stringsAsFactors=TRUE)')
    ro.r(f'm <- lme4::lmer({formula}, data=.df, REML={"TRUE" if reml else "FALSE"})')

    fe_names = list(map(str, ro.r('names(fixef(m))')))
    fe_vals = list(map(float, ro.r('fixef(m)')))
    grp = str(ro.r('names(VarCorr(m))')[0])
    re_sds = list(map(float, ro.r(f'attr(VarCorr(m)[["{grp}"]], "stddev")')))
    resid = np.array(ro.r('residuals(m)'))

    ro.r(f'.re <- ranef(m, condVar=FALSE)[["{grp}"]]')
    re_cols = list(map(str, ro.r('colnames(.re)')))
    re_frame = pd.DataFrame({c: np.array(ro.r(f'.re[, {i+1}]')) for i, c in enumerate(re_cols)}, index=list(map(str, ro.r('rownames(.re)'))))

    ro.r('.ci <- confint(m, parm = "theta_", method = "profile", level = 0.95, signames = FALSE)')
    ci_names = list(map(str, ro.r('rownames(.ci)')))
    ci_lo, ci_hi = list(map(float, ro.r('.ci[,1]'))), list(map(float, ro.r('.ci[,2]')))

    return {
        'fixed_effects_coefficients': dict(zip(fe_names, fe_vals)),
        'random_intercept_variance': re_sds[0] ** 2,
        'random_slope_variance': re_sds[1] ** 2,
        'residual_variance': float(ro.r('sigma(m)')[0]) ** 2,
        'residuals': resid,
        'runner_random_effects_frame': re_frame,
        'random_effect_standard_deviation_profile_ci': {n: (l, h) for n, l, h in zip(ci_names, ci_lo, ci_hi)},
    }


def build_fixed_effects_matrix(df):
    """Construct the fixed-effects design matrix matching lme4's coefficient order."""
    ac, f, yc = df['age_centered'].to_numpy(float), df['female'].to_numpy(float), df['year_centered'].to_numpy(float)
    return np.column_stack([np.ones(len(df)), ac, ac**2, f, yc, ac*f, ac**2*f])


def compute_marginal_log_seconds_prediction(df, fe_coefs):
    """Fixed-effects-only log_seconds prediction."""
    return build_fixed_effects_matrix(df) @ np.array([fe_coefs[n] for n in _COEF_NAMES])


def fit_temporal_holdout_runner_mixed_effects(repeat_df):
    """Fit the runner mixed-effects model and export leak-free runner effects."""
    train = repeat_df[repeat_df['year'] <= config.RUNNER_MIXED_EFFECTS_TRAIN_END_YEAR].copy()
    test = repeat_df[repeat_df['year'] > config.RUNNER_MIXED_EFFECTS_TRAIN_END_YEAR].copy()
    age_mean = float(train['age'].mean())

    holders = set(train.groupby('display_name').size().loc[lambda c: c >= 2].index)
    test_known = test[test['display_name'].isin(holders)].copy()
    test_never = test[~test['display_name'].isin(holders)].copy()

    for d in (train, test_known, test_never):
        d['log_seconds'] = np.log(d['seconds'].to_numpy())
        add_centered_pre_race_features(d, age_mean)

    input_cols = ['log_seconds', 'display_name'] + config.PRE_RACE_FIXED_EFFECT_FEATURES
    formula = 'log_seconds ~ age_centered + I(age_centered^2) + female + year_centered + age_centered:female + I(age_centered^2):female + (1 + age_centered | display_name)'
    fit = fit_lme4_mixed_effects_model(train[input_cols].sort_values('display_name').reset_index(drop=True), formula, reml=True)

    fe_vec = np.array([fit['fixed_effects_coefficients'][n] for n in _COEF_NAMES])
    re_df = fit['runner_random_effects_frame']
    re_df.columns = ['runner_intercept', 'runner_age_slope']
    duan = float(np.exp(fit['residuals']).mean())

    marginal_log_known = build_fixed_effects_matrix(test_known) @ fe_vec
    re_known = re_df.reindex(test_known['display_name'].values)
    cond_log_known = marginal_log_known + re_known['runner_intercept'].to_numpy() + re_known['runner_age_slope'].to_numpy() * test_known['age_centered'].to_numpy()

    obs_known, obs_never = test_known['seconds'].to_numpy(float), test_never['seconds'].to_numpy(float)
    pred_marg_known = np.exp(marginal_log_known) * duan
    pred_cond_known = np.exp(cond_log_known) * duan
    pred_marg_never = np.exp(build_fixed_effects_matrix(test_never) @ fe_vec) * duan

    fe_var = float(np.var(build_fixed_effects_matrix(train) @ fe_vec))
    rs_var = fit['random_slope_variance'] * float(train['age_centered'].var())
    total_var = fe_var + fit['random_intercept_variance'] + rs_var + fit['residual_variance']

    leakfree_re = re_df.loc[re_df.index.isin(holders)].rename_axis('display_name')

    return {
        'mixed_model_residuals': fit['residuals'],
        'fixed_effects_coefficients_leakfree': fit['fixed_effects_coefficients'],
        'age_mean_leakfree_from_training': age_mean,
        'random_slope_standard_deviation_confidence_interval': fit['random_effect_standard_deviation_profile_ci']['sd_age_centered|display_name'],
        'duan_smearing_factor': duan,
        'runner_random_effects_leakfree': leakfree_re,
        'test_known': test_known,
        'n_test_never_seen': len(test_never),
        'marginal_mixed_effects_rmse_on_test_known': float(rmse(obs_known, pred_marg_known)),
        'conditional_mixed_effects_rmse_on_test_known': float(rmse(obs_known, pred_cond_known)),
        'marginal_mixed_effects_rmse_on_never_seen': float(rmse(obs_never, pred_marg_never)),
        'pred_seconds_marginal_test_known': pred_marg_known,
        'pred_seconds_conditional_test_known': pred_cond_known,
        'variance_explained_by_fixed_effects_only': fe_var / total_var,
        'variance_explained_by_fixed_and_random_effects': (fe_var + fit['random_intercept_variance'] + rs_var) / total_var,
    }

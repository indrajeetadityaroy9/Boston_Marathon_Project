"""Statistical metrics shared across RQ1, RQ2, and RQ3.

Regression metrics (RMSE, MAE, R^2, MAPE) delegate to sklearn.
Mixed-model metrics (ICC, Nakagawa-Schielzeth R^2, boundary LRT, prediction intervals)
are custom — no standard library provides these for the LME context.
Called by rq1.py, rq2.py, rq3.py, and run_pipeline.py.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (root_mean_squared_error as rmse,
                             mean_absolute_error, r2_score, mean_absolute_percentage_error)


def regression_metrics(y_true, y_pred):
    """Standard regression metrics in both seconds and minutes. Used by rq1 and rq3."""
    r = rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {'rmse_s': r, 'rmse_min': r / 60, 'mae_s': mae, 'mae_min': mae / 60,
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100}


def icc_anova(df, group_col, value_col):
    """Fraction of total variance attributable to between-group differences (ICC(1)).
    Used in RQ2 to show 70% of finish-time variance is stable runner ability."""
    n_r, n_o = df[group_col].nunique(), len(df)
    grp = df.groupby(group_col)[value_col]
    runner_means = grp.transform('mean')
    grand_mean = df[value_col].mean()
    ms_b = ((runner_means - grand_mean) ** 2).sum() / (n_r - 1)
    ms_w = ((df[value_col] - runner_means) ** 2).sum() / (n_o - n_r)
    ni = grp.size()
    k0 = (n_o - (ni ** 2).sum() / n_o) / (n_r - 1)
    return (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w)


def icc_conditional(tau2, sigma2):
    """ICC after accounting for fixed effects: tau^2 / (tau^2 + sigma^2)."""
    return tau2 / (tau2 + sigma2), tau2, sigma2


def variance_decomposition(result, df, fe_param_values):
    """Decompose total variance into fixed effects, random intercepts, random slopes, and residual.
    Marginal R^2 = fixed only (new runners); conditional R^2 = fixed + random (known runners)."""
    import statsmodels.api as sm
    exog = sm.add_constant(df[['age_c', 'female', 'year_c']])
    var_fixed = np.var(exog.values @ np.array(fe_param_values))
    tau2_0, sigma2 = result['tau2_0'], result['sigma2']
    var_slope = result['tau2_1'] * df['age_c'].var()
    total = var_fixed + tau2_0 + var_slope + sigma2
    return {'var_fixed': var_fixed, 'tau2_0': tau2_0, 'var_slope': var_slope,
            'sigma2': sigma2, 'total': total,
            'r2_marginal': var_fixed / total,
            'r2_conditional': (var_fixed + tau2_0 + var_slope) / total}


def boundary_lrt(llf_restricted, llf_full, df_diff):
    """Test whether adding random effects significantly improves the model (Stram & Lee 1994).
    Uses a chi-squared mixture because variance parameters are tested at their boundary (>= 0)."""
    lr = 2 * (llf_full - llf_restricted)
    p_std = stats.chi2.sf(lr, df=df_diff)
    if df_diff == 1:
        p_corr = 0.5 * stats.chi2.sf(lr, df=1)
    else:
        p_corr = 0.5 * stats.chi2.sf(lr, df=df_diff - 1) + 0.5 * stats.chi2.sf(lr, df=df_diff)
    return {'lr_stat': lr, 'p_standard': p_std, 'p_corrected': p_corr}


def residual_diagnostics(resid, seed=42):
    """Check whether residuals are approximately normal. Subsample avoids trivial rejection."""
    sw_stat, sw_p = stats.shapiro(pd.Series(resid).sample(n=5000, random_state=seed).values)
    return {'mean': np.mean(resid), 'std': np.std(resid),
            'skewness': stats.skew(resid), 'kurtosis': stats.kurtosis(resid),
            'shapiro_stat': sw_stat, 'shapiro_p': sw_p}


def prediction_interval_width(sigma2, tau2_0=None, level=0.90):
    """How wide a 90% prediction interval is for a known runner (sigma^2 only) or
    a new runner (sigma^2 + tau^2, accounting for unknown runner ability)."""
    z = stats.norm.ppf((1 + level) / 2)
    return 2 * z * np.sqrt(sigma2 if tau2_0 is None else tau2_0 + sigma2)

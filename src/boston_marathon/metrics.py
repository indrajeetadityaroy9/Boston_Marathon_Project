"""Scoring functions used across all three research questions.

rq1 and rq3 use regression_metrics to measure how close predictions are to actual
finish times. rq2 uses the mixed-model functions to answer questions that standard
libraries don't cover: how much of the variation is between runners vs within runners
(ICC), how much the model explains for known vs new runners (variance decomposition),
whether adding random effects is statistically justified (boundary likelihood ratio test), and how wide
a prediction interval should be (prediction_interval_width). rq3 uses empirical_coverage
to check whether prediction intervals actually contain 90% of finish times.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (root_mean_squared_error as rmse,
                             mean_absolute_error, r2_score, mean_absolute_percentage_error)


def regression_metrics(y_true, y_pred):
    """Measure how far predictions are from actual finish times.

    Returns RMSE (typical error size), MAE (average absolute error), R-squared
    (fraction of variance explained), and MAPE (percentage error), each in both
    seconds and minutes for readable reporting.
    """
    r, mae = rmse(y_true, y_pred), mean_absolute_error(y_true, y_pred)
    return {'rmse_s': r, 'rmse_min': r / 60, 'mae_s': mae, 'mae_min': mae / 60,
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100}


def icc_anova(df, group_col, value_col):
    """What fraction of finish-time variation comes from who the runner is?

    An ICC of 0.70 means 70% of the spread in times is due to stable differences
    between runners (some are fast, some are slow), and only 30% is race-to-race
    noise. High ICC justifies giving each runner their own parameters in the model.

    Uses one-way ANOVA with a correction (k0) for runners having different numbers
    of races. Standard pingouin ICC fails on this unbalanced data.
    """
    n_r, n_o = df[group_col].nunique(), len(df)
    grp = df.groupby(group_col)[value_col]
    runner_means, grand_mean = grp.transform('mean'), df[value_col].mean()
    ms_b = ((runner_means - grand_mean) ** 2).sum() / (n_r - 1)
    ms_w = ((df[value_col] - runner_means) ** 2).sum() / (n_o - n_r)
    k0 = (n_o - (grp.size() ** 2).sum() / n_o) / (n_r - 1)
    return (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w)


def variance_decomposition(result, df, fe_param_values):
    """Break total finish-time variance into four pieces to understand what drives prediction.

    The four pieces are:
      fixed effects  - what demographics (age, gender, year) explain
      intercepts     - how much runners differ in baseline ability
      slopes         - how much runners differ in aging rate
      residual       - leftover noise the model can't explain

    Marginal R-squared tells you how well the model predicts a brand-new runner
    (only demographics available). Conditional R-squared tells you how well it
    predicts a runner you've seen before (demographics + their personal history).
    """
    import statsmodels.api as sm
    var_fixed = np.var(sm.add_constant(df[['age_c', 'female', 'year_c']]).values @ np.array(fe_param_values))
    tau2_0, sigma2, var_slope = result['tau2_0'], result['sigma2'], result['tau2_1'] * df['age_c'].var()
    total = var_fixed + tau2_0 + var_slope + sigma2
    return {'var_fixed': var_fixed, 'tau2_0': tau2_0, 'var_slope': var_slope, 'sigma2': sigma2,
            'total': total, 'r2_marginal': var_fixed / total,
            'r2_conditional': (var_fixed + tau2_0 + var_slope) / total}


def boundary_lrt(llf_restricted, llf_full, df_diff):
    """Test whether adding per-runner parameters significantly improves the model.

    Compares a simpler model (e.g. no random effects) against a more complex one
    (e.g. with random intercepts). The twist is that variance can't be negative,
    so the standard chi-squared test gives wrong p-values when the true variance
    is near zero. The Stram & Lee (1994) correction uses a 50:50 mixture of
    chi-squared distributions to account for this boundary problem.
    """
    lr = 2 * (llf_full - llf_restricted)
    p_std = stats.chi2.sf(lr, df=df_diff)
    p_corr = (0.5 * stats.chi2.sf(lr, df=1) if df_diff == 1
              else 0.5 * stats.chi2.sf(lr, df=df_diff - 1) + 0.5 * stats.chi2.sf(lr, df=df_diff))
    return {'lr_stat': lr, 'p_standard': p_std, 'p_corrected': p_corr}


def residual_diagnostics(resid, seed=42):
    """Check whether the model's errors look roughly like a bell curve.

    Reports skewness (lopsidedness), kurtosis (heavy tails), and a Shapiro-Wilk
    normality test. Uses a random 5000-row sample for Shapiro-Wilk because the
    full dataset is so large that even tiny departures from normality get flagged
    as significant.
    """
    sample = np.random.default_rng(seed).choice(resid, 5000, replace=False)
    sw_stat, sw_p = stats.shapiro(sample)
    return {'mean': np.mean(resid), 'std': np.std(resid),
            'skewness': stats.skew(resid), 'kurtosis': stats.kurtosis(resid),
            'shapiro_stat': sw_stat, 'shapiro_p': sw_p}


def prediction_interval_width(sigma2, tau2_0=None, level=0.90):
    """How wide a 90% prediction interval needs to be.

    For a runner we've seen before, the interval only needs to cover race-to-race
    noise (sigma2). For a completely new runner, it also needs to cover the
    uncertainty about their ability level (tau2_0), making the interval wider.
    """
    z = stats.norm.ppf((1 + level) / 2)
    return 2 * z * np.sqrt(sigma2 if tau2_0 is None else tau2_0 + sigma2)


def empirical_coverage(y_true, lower, upper):
    """What percentage of actual finish times fell inside the predicted interval?

    If the model says "90% of runners should finish between X and Y seconds,"
    this checks whether that's actually true on the test set. A well-calibrated
    interval should hit close to 90%.
    """
    return np.mean((y_true >= lower) & (y_true <= upper))

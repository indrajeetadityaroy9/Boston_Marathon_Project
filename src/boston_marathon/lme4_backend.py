"""Runs R's lme4 mixed-effects models from Python and returns results as a dictionary.

rq2.py calls fit_lmer() every time it needs to fit a model with per-runner random effects.
This module sends data to R, fits the model there (lme4 is much faster than statsmodels
for large grouped data), and pulls back everything Python needs: coefficients, variances,
per-runner adjustments (Best Linear Unbiased Predictions), residuals, and fit statistics.
"""
import os
import subprocess

# R needs to know where it's installed; auto-detect if not already set
if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = subprocess.check_output(['R', 'RHOME'], text=True).strip()

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

# Set up automatic conversion between pandas/numpy and R data types
_conv = default_converter + pandas2ri.converter + numpy2ri.converter
importr('lme4')


def fit_lmer(df, formula, reml=True):
    """Fit a mixed-effects model using R's lme4 and return all results as a Python dict.

    Takes a pandas DataFrame and an R-style formula like:
        'seconds ~ age_c + female + year_c + (1 + age_c | display_name)'

    The part before | lists the fixed effects (population-level trends).
    The part in parentheses defines random effects (per-runner adjustments):
        (1 | runner)           = each runner gets their own baseline offset
        (1 + age_c | runner)   = each runner gets their own baseline AND aging rate

    Set reml=True for final reporting (better variance estimates).
    Set reml=False when comparing models with likelihood ratio tests.

    Returns a dict with:
        fe_params   - population-level coefficients (e.g. how much gender affects time)
        fe_se       - standard errors for those coefficients
        tau2_0      - how much runners vary in baseline ability (random intercept variance)
        tau2_1      - how much runners vary in aging rate (random slope variance, 0 if none)
        corr_01     - correlation between a runner's baseline and their aging rate
        sigma2      - leftover variance not explained by fixed or random effects
        resid       - difference between actual and predicted time for each observation
        blup_df     - per-runner adjustments: how much faster/slower each runner is than average
        loglik      - log-likelihood (higher = better fit, used for model comparison)
        aic, bic    - information criteria (lower = better, penalize model complexity)
        cond_rmse   - prediction error when using both population trends and runner adjustments
        converged   - whether the optimization finished without warnings
    """
    # Send the pandas DataFrame to R's workspace
    with conversion.localconverter(_conv):
        ro.globalenv['df'] = df
    ro.r(f'm <- lme4::lmer({formula}, data=df, REML={"TRUE" if reml else "FALSE"})')

    # Pull out fixed-effect coefficients and their standard errors
    fe_names = list(map(str, ro.r('names(fixef(m))')))
    fe_vals = list(map(float, ro.r('fixef(m)')))
    se_vals = list(map(float, ro.r('summary(m)$coefficients[,"Std. Error"]')))

    # Pull out random-effect variances (how much runners differ from each other)
    grp = str(ro.r('names(VarCorr(m))')[0])
    sd = list(map(float, ro.r(f'attr(VarCorr(m)[["{grp}"]], "stddev")')))
    has_slope = len(sd) > 1

    # Residuals: actual finish time minus what the model predicted
    resid = np.array(ro.r('residuals(m)'))

    # Best Linear Unbiased Predictions: each runner's personal offset from the population average
    # condVar=FALSE skips computing uncertainty around each BLUP (saves ~80 seconds)
    ro.r(f'.re <- ranef(m, condVar=FALSE)[["{grp}"]]')
    with conversion.localconverter(_conv):
        blup_df = pd.DataFrame(ro.r('.re'))
        blup_df.index = list(map(str, ro.r('rownames(.re)')))

    return {
        'converged': bool(ro.r('is.null(m@optinfo$conv$lme4$messages)')[0]),
        'fe_params': dict(zip(fe_names, fe_vals)),
        'fe_se': dict(zip(fe_names, se_vals)),
        'tau2_0': sd[0] ** 2,
        'tau2_1': sd[1] ** 2 if has_slope else 0.0,
        'corr_01': float(ro.r(f'attr(VarCorr(m)[["{grp}"]], "correlation")[1,2]')[0]) if has_slope else 0.0,
        'sigma2': float(ro.r('sigma(m)')[0]) ** 2,
        'resid': resid,
        'blup_df': blup_df,
        'loglik': float(ro.r('as.numeric(logLik(m))')[0]),
        'aic': float(ro.r('AIC(m)')[0]),
        'bic': float(ro.r('BIC(m)')[0]),
        'cond_rmse': float(np.sqrt(np.mean(resid ** 2))),
    }

"""Bridge to R's lme4 package via rpy2 for mixed-effects model fitting.

Called by rq2.py for all random-intercept and random-slope models. Returns a plain dict
(fixed effects, variance components, BLUPs, residuals, AIC/BIC) so that the rest of the
Python codebase never touches R objects directly. 21.8x faster than statsmodels MixedLM.
"""
import os
import subprocess

if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = subprocess.check_output(['R', 'RHOME'], text=True).strip()

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri

_conv = default_converter + pandas2ri.converter + numpy2ri.converter
_lme4 = importr('lme4')


def fit_lmer(df, formula, reml=True):
    """Fit a linear mixed-effects model in R and return all results as a Python dict.
    The formula uses R syntax (e.g., 'seconds ~ age_c + (1 + age_c | display_name)').
    BLUPs are extracted with condVar=FALSE to avoid an 82-second variance computation."""
    with conversion.localconverter(_conv):
        ro.globalenv['df'] = df
    ro.r(f'm <- lme4::lmer({formula}, data=df, REML={"TRUE" if reml else "FALSE"})')

    fe_names = [str(x) for x in ro.r('names(fixef(m))')]
    fe_vals = [float(x) for x in ro.r('fixef(m)')]
    se_vals = [float(x) for x in ro.r('summary(m)$coefficients[,"Std. Error"]')]

    grp = str(ro.r('names(VarCorr(m))')[0])
    sd = [float(x) for x in ro.r(f'attr(VarCorr(m)[["{grp}"]], "stddev")')]
    has_slope = len(sd) > 1

    resid = np.array(ro.r('residuals(m)'))

    # Extract BLUPs once (ranef with condVar=FALSE avoids 82s variance computation)
    ro.r(f'.re <- ranef(m, condVar=FALSE)[["{grp}"]]')
    with conversion.localconverter(_conv):
        blup_df = pd.DataFrame(ro.r('.re'))
        blup_df.index = [str(x) for x in ro.r('rownames(.re)')]

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

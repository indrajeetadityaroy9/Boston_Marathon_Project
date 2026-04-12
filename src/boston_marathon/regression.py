"""Pre-race regression models for demographics and prior Boston history."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error as rmse

from boston_marathon.config import PRE_RACE_FIXED_EFFECT_FEATURES as PRE_RACE_DEMOGRAPHIC_FEATURES
PRE_RACE_DEMOGRAPHIC_AND_HISTORY_FEATURES = PRE_RACE_DEMOGRAPHIC_FEATURES + ['log1p_prior_appearances', 'prior_mean_time']


def fit_hc3_robust_log_seconds_regression(train_df, feature_columns):
    """Fit HC3-robust OLS on log_seconds and return (fit, duan_smearing_factor)."""
    X = sm.add_constant(train_df[feature_columns].to_numpy(float))
    fit = sm.OLS(train_df['log_seconds'].to_numpy(), X).fit(cov_type='HC3')
    return fit, float(np.exp(fit.resid).mean())


def evaluate_log_seconds_regression(fit, duan_factor, eval_df, feature_columns):
    """Evaluate a fitted pre-race regression on an analysis subset."""
    X = sm.add_constant(eval_df[feature_columns].to_numpy(float))
    pred = np.exp(fit.predict(X)) * duan_factor
    obs = eval_df['seconds'].to_numpy(float)
    return {
        'observed_seconds': obs, 'predicted_seconds': pred,
        'rmse_seconds': float(rmse(obs, pred)),
        'calibration_slope': float(sm.OLS(obs, sm.add_constant(pred)).fit().params[1]),
        'mean_residual_per_year': pd.Series(obs - pred, index=eval_df['year'].to_numpy()).groupby(level=0).mean(),
        'test_frame': eval_df,
    }

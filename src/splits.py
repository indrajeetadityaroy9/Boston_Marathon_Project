"""In-race finish-time prediction from cumulative split checkpoints."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error as rmse

from . import config
from .inference import fit_split_cqr

IN_RACE_DEMOGRAPHIC_FEATURES = ['age', 'female']


def fit_checkpoint_ridge_models(train_df, test_df):
    """Fit RidgeCV at each checkpoint and return summary + fitted models."""
    train_holders = train_df[train_df['runner_intercept'].notna()]
    test_holders = test_df[test_df['runner_intercept'].notna()]

    rows, models = [], {}
    for ci, label in enumerate(config.CHECKPOINT_LABELS):
        feats = config.CUMULATIVE_SPLIT_TIME_COLUMNS[:ci + 1] + IN_RACE_DEMOGRAPHIC_FEATURES
        X_tr, y_tr = train_df[feats].to_numpy(float), train_df['seconds'].to_numpy(float)
        X_te, y_te = test_df[feats].to_numpy(float), test_df['seconds'].to_numpy(float)

        ridge = RidgeCV(cv=None).fit(X_tr, y_tr)
        rows.append({'checkpoint': label, 'variant': 'no_runner_history', 'alpha': ridge.alpha_, 'rmse_seconds': float(rmse(y_te, ridge.predict(X_te))), 'n': len(test_df)})
        models[(label, 'no_runner_history')] = {'ridge_regression_model': ridge, 'features': feats}

        if label == '5K':
            hist_feats = feats + ['runner_intercept', 'runner_age_slope']
            X_tr_h, y_tr_h = train_holders[hist_feats].to_numpy(float), train_holders['seconds'].to_numpy(float)
            X_te_h, y_te_h = test_holders[hist_feats].to_numpy(float), test_holders['seconds'].to_numpy(float)
            ridge_h = RidgeCV(cv=None).fit(X_tr_h, y_tr_h)
            rows.append({'checkpoint': '5K', 'variant': 'with_runner_history', 'alpha': ridge_h.alpha_, 'rmse_seconds': float(rmse(y_te_h, ridge_h.predict(X_te_h))), 'n': len(test_holders)})
            models[('5K', 'with_runner_history')] = {'ridge_regression_model': ridge_h, 'features': hist_feats}

    return pd.DataFrame(rows), models


def compute_split_conformal_quantile_intervals(train_df, test_df, rng):
    """Compute Split-CQR 90% prediction intervals at each checkpoint."""
    shuf = rng.permutation(len(train_df))
    fit_idx, cal_idx = shuf[:len(train_df) // 2], shuf[len(train_df) // 2:]
    y_tr, y_te = train_df['seconds'].to_numpy(float), test_df['seconds'].to_numpy(float)

    rows = []
    for ci, label in enumerate(config.CHECKPOINT_LABELS):
        feats = config.CUMULATIVE_SPLIT_TIME_COLUMNS[:ci + 1]
        X_tr = sm.add_constant(train_df[feats].to_numpy())
        X_te = sm.add_constant(test_df[feats].to_numpy())
        cqr = fit_split_cqr(X_tr, y_tr, X_te, fit_indices=fit_idx, cal_indices=cal_idx)
        cov = float(((y_te >= cqr['pi_lower']) & (y_te <= cqr['pi_upper'])).mean())
        rows.append({'checkpoint': label, 'split_conformal_coverage': cov})

    return pd.DataFrame(rows)

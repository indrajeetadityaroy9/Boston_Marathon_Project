"""Temporal calibration diagnosis: conformity score shift and honest recalibration."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ks_2samp
from sklearn.linear_model import RidgeCV
from statsmodels.regression.quantile_regression import QuantReg

from . import config
from .inference import fit_split_cqr


def compute_conformity_score_shift(train_df, test_df, rng):
    """Compare conformity score distributions between calibration and test sets at 5K."""
    col = config.CUMULATIVE_SPLIT_TIME_COLUMNS[0]
    y_tr, y_te = train_df['seconds'].to_numpy(float), test_df['seconds'].to_numpy(float)
    shuf = rng.permutation(len(train_df))
    fit_idx, cal_idx = shuf[:len(train_df) // 2], shuf[len(train_df) // 2:]

    X_tr = sm.add_constant(train_df[[col]].to_numpy(float))
    X_te = sm.add_constant(test_df[[col]].to_numpy(float))
    lo = QuantReg(y_tr[fit_idx], X_tr[fit_idx]).fit(q=0.05, max_iter=10_000)
    hi = QuantReg(y_tr[fit_idx], X_tr[fit_idx]).fit(q=0.95, max_iter=10_000)

    cal_scores = np.maximum(lo.predict(X_tr[cal_idx]) - y_tr[cal_idx], y_tr[cal_idx] - hi.predict(X_tr[cal_idx]))
    test_scores = np.maximum(lo.predict(X_te) - y_te, y_te - hi.predict(X_te))
    ks_stat, ks_p = ks_2samp(cal_scores, test_scores)
    return {'scale_ratio': float(test_scores.std() / cal_scores.std()), 'ks_statistic': float(ks_stat), 'ks_pvalue': float(ks_p), 'cal_scores': cal_scores, 'test_scores': test_scores}


def run_honest_recalibration_experiments(train_df, test_df, rng):
    """Test two leakage-free recalibration strategies across all checkpoints."""
    years = sorted(train_df['year'].unique())
    fit_df, val_df = train_df[train_df['year'] == years[0]], train_df[train_df['year'] == years[1]]
    y_fit, y_val, y_te = fit_df['seconds'].to_numpy(float), val_df['seconds'].to_numpy(float), test_df['seconds'].to_numpy(float)

    rows = []
    for ci, label in enumerate(config.CHECKPOINT_LABELS):
        feats = config.CUMULATIVE_SPLIT_TIME_COLUMNS[:ci + 1]
        X_fit, X_val, X_te = sm.add_constant(fit_df[feats].to_numpy(float)), sm.add_constant(val_df[feats].to_numpy(float)), sm.add_constant(test_df[feats].to_numpy(float))

        # Baseline CQR on pooled training
        X_all, y_all = sm.add_constant(train_df[feats].to_numpy(float)), train_df['seconds'].to_numpy(float)
        base = fit_split_cqr(X_all, y_all, X_te, rng=rng)
        base_cov = float(((y_te >= base['pi_lower']) & (y_te <= base['pi_upper'])).mean())

        # Temporal CQR: fit on fit_year, calibrate on val_year
        lo_t = QuantReg(y_fit, X_fit).fit(q=0.05, max_iter=10_000)
        hi_t = QuantReg(y_fit, X_fit).fit(q=0.95, max_iter=10_000)
        scores_t = np.maximum(lo_t.predict(X_val) - y_val, y_val - hi_t.predict(X_val))
        rank = int(np.ceil((len(scores_t) + 1) * 0.90))
        thr = np.partition(scores_t, rank - 1)[rank - 1]
        tcqr_lo, tcqr_hi = lo_t.predict(X_te) - thr, hi_t.predict(X_te) + thr
        rows.append({'method': 'temporal_cqr', 'checkpoint': label, 'baseline_coverage': base_cov, 'corrected_coverage': float(((y_te >= tcqr_lo) & (y_te <= tcqr_hi)).mean()), 'interval_width': float(np.median(tcqr_hi - tcqr_lo))})

        # Post-hoc residual inflation: fit Ridge on fit_year
        ridge = RidgeCV(cv=None).fit(fit_df[feats].to_numpy(), y_fit)
        resid_fit = y_fit - ridge.predict(fit_df[feats].to_numpy())
        resid_val = y_val - ridge.predict(val_df[feats].to_numpy())
        resid_te = y_te - ridge.predict(test_df[feats].to_numpy())
        q05, q95 = np.percentile(resid_fit, [5, 95])
        required_m = np.maximum(resid_val / q95, resid_val / q05)
        mult = float(np.percentile(required_m, 90))
        ph_cov = float(((resid_te >= q05 * mult) & (resid_te <= q95 * mult)).mean())
        rows.append({'method': 'posthoc_inflation', 'checkpoint': label, 'baseline_coverage': base_cov, 'corrected_coverage': ph_cov, 'interval_width': float(q95 * mult - q05 * mult)})

    return pd.DataFrame(rows)

"""BCa cluster bootstrap, split conformal quantile regression, and model comparison."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import root_mean_squared_error as rmse
from statsmodels.regression.quantile_regression import QuantReg


def compute_bca_cluster_bootstrap_interval(statistic_fn, df, cluster_col, rng):
    """BCa cluster-bootstrap 95% interval for a scalar or vector statistic."""
    raw_ids = df[cluster_col].to_numpy()
    _, cidx = np.unique(raw_ids, return_inverse=True)
    k = cidx.max() + 1
    rows_by_cluster = [np.where(cidx == i)[0] for i in range(k)]

    theta = np.asarray(statistic_fn(df), dtype=float)
    boot = np.empty((1000,) + theta.shape, dtype=float)
    for b in range(1000):
        ids = rng.integers(0, k, size=k)
        boot[b] = np.asarray(statistic_fn(df.iloc[np.concatenate([rows_by_cluster[i] for i in ids])]), dtype=float)

    all_idx = np.arange(len(df))
    jack = np.empty((k,) + theta.shape, dtype=float)
    for i in range(k):
        mask = np.ones(len(df), dtype=bool)
        mask[rows_by_cluster[i]] = False
        jack[i] = np.asarray(statistic_fn(df.iloc[all_idx[mask]]), dtype=float)

    d = jack.mean(axis=0) - jack
    a = np.where((den := 6.0 * (d ** 2).sum(axis=0) ** 1.5) > 0, (d ** 3).sum(axis=0) / den, 0.0)

    p_below = np.clip((boot < theta[np.newaxis, ...]).sum(axis=0) / 1000, 1 / 1001, 1000 / 1001)
    z0 = norm.ppf(p_below)
    za_lo, za_hi = norm.ppf(0.025), norm.ppf(0.975)
    p_lo = norm.cdf(z0 + (z0 + za_lo) / (1 - a * (z0 + za_lo)))
    p_hi = norm.cdf(z0 + (z0 + za_hi) / (1 - a * (z0 + za_hi)))

    sorted_boot = np.sort(boot, axis=0)

    def interp(p):
        pos = p * 999
        lo_i, hi_i = np.floor(pos).astype(int), np.minimum(np.floor(pos).astype(int) + 1, 999)
        frac = pos - lo_i
        if theta.ndim == 0:
            return sorted_boot[lo_i] * (1 - frac) + sorted_boot[hi_i] * frac
        return np.take_along_axis(sorted_boot, lo_i[np.newaxis], 0)[0] * (1 - frac) + np.take_along_axis(sorted_boot, hi_i[np.newaxis], 0)[0] * frac

    ci_lo, ci_hi = interp(p_lo), interp(p_hi)
    if theta.ndim == 0:
        return float(theta), float(ci_lo), float(ci_hi)
    return theta, ci_lo, ci_hi


def fit_split_cqr(X_train, y_train, X_test, fit_indices=None, cal_indices=None, rng=None):
    """Split-CQR 90% prediction intervals. X must include a constant column."""
    if fit_indices is None:
        shuf = rng.permutation(len(y_train))
        fit_indices, cal_indices = shuf[:len(y_train) // 2], shuf[len(y_train) // 2:]

    lo_model = QuantReg(y_train[fit_indices], X_train[fit_indices]).fit(q=0.05, max_iter=10_000)
    hi_model = QuantReg(y_train[fit_indices], X_train[fit_indices]).fit(q=0.95, max_iter=10_000)

    scores = np.maximum(lo_model.predict(X_train[cal_indices]) - y_train[cal_indices], y_train[cal_indices] - hi_model.predict(X_train[cal_indices]))
    rank = int(np.ceil((len(scores) + 1) * 0.90))
    offset = np.partition(scores, rank - 1)[rank - 1]

    return {'pi_lower': lo_model.predict(X_test) - offset, 'pi_upper': hi_model.predict(X_test) + offset, 'conformity_scores': scores, 'conformal_offset': offset}


def bootstrap_rmse_comparison(y, pred_a, pred_b, clusters, rng):
    """Paired cluster-bootstrap comparison of two models' RMSE."""
    paired = pd.DataFrame({'_y': y, '_a': pred_a, '_b': pred_b, 'c': clusters})
    def stat(sub):
        obs = sub['_y'].to_numpy()
        ra, rb = float(rmse(obs, sub['_a'].to_numpy())), float(rmse(obs, sub['_b'].to_numpy()))
        return np.array([ra, rb, ra - rb])
    pt, lo, hi = compute_bca_cluster_bootstrap_interval(stat, paired, 'c', rng=rng)
    return {
        'rmse_a': float(pt[0]), 'rmse_a_ci': (float(lo[0]), float(hi[0])),
        'rmse_b': float(pt[1]), 'rmse_b_ci': (float(lo[1]), float(hi[1])),
        'delta': float(pt[2]), 'delta_ci': (float(lo[2]), float(hi[2])),
    }

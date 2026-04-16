import numpy as np
from scipy.stats import norm


def compute_conformal_quantile(residuals, alpha):
    n = len(residuals)
    return np.quantile(residuals, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0))


def compute_bca_cluster_bootstrap_rmse(y, preds, clusters, rng, cfg):
    n = len(y)
    _, cidx = np.unique(clusters, return_inverse=True)
    k = cidx.max() + 1
    n_per_cluster = np.bincount(cidx, minlength=k).astype(float)
    sse = np.array([np.bincount(cidx, weights=(y - p) ** 2, minlength=k) for p in preds])

    rmses = np.sqrt(sse.sum(axis=1) / n)
    theta = np.concatenate([rmses, rmses[:-1] - rmses[1:]])

    counts = np.array([np.bincount(rng.integers(0, k, size=k), minlength=k)
                        for _ in range(cfg.n_bootstrap_resamples)]).astype(float)
    br = np.sqrt(sse @ counts.T / (n_per_cluster @ counts.T))
    boot = np.concatenate([br, br[:-1] - br[1:]]).T

    jr = np.sqrt((sse.sum(axis=1)[:, None] - sse) / (n - n_per_cluster))
    jack = np.concatenate([jr, jr[:-1] - jr[1:]]).T

    B = len(boot)
    d = jack.mean(axis=0) - jack
    d2 = (d ** 2).sum(axis=0)
    a = np.where(d2 > 1e-12, (d ** 3).sum(axis=0) / (6.0 * d2 ** 1.5), 0.0)

    z0 = norm.ppf(np.clip((boot < theta).sum(axis=0) / B, 1 / (B + 1), B / (B + 1)))
    za_lo, za_hi = norm.ppf((1 - cfg.bca_confidence_level) / 2), norm.ppf(1 - (1 - cfg.bca_confidence_level) / 2)
    p_lo = norm.cdf(z0 + (z0 + za_lo) / np.maximum(np.abs(1 - a * (z0 + za_lo)), 1e-12))
    p_hi = norm.cdf(z0 + (z0 + za_hi) / np.maximum(np.abs(1 - a * (z0 + za_hi)), 1e-12))

    sb, last, idx = np.sort(boot, axis=0), B - 1, np.arange(len(theta))
    def _interp(p):
        pos = p * last
        lo_i = np.floor(pos).astype(int)
        return sb[lo_i, idx] * (1 - (pos - lo_i)) + sb[np.minimum(lo_i + 1, last), idx] * (pos - lo_i)

    return theta, _interp(p_lo), _interp(p_hi)


def bootstrap_rmse_comparison(y, pred_a, pred_b, clusters, rng, cfg):
    pt, lo, hi = compute_bca_cluster_bootstrap_rmse(y, [pred_a, pred_b], clusters, rng, cfg)
    return {"rmse_a": pt[0], "rmse_a_ci": (lo[0], hi[0]),
            "rmse_b": pt[1], "rmse_b_ci": (lo[1], hi[1]),
            "delta": pt[2], "delta_ci": (lo[2], hi[2])}

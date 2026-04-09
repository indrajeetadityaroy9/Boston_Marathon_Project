"""RQ3: Once the race starts, how quickly does prediction improve?

At each of nine checkpoints (5K through 40K), fits Ridge regression on cumulative split times.
Tests seven model variants (naive pace, splits only, + demographics, + year, + BLUPs, single
checkpoint, BLUP subset). Identifies the crossover point where in-race data outperforms
RQ2's personalized pre-race prediction. BLUPs come from data.load_blups() (exported by RQ2).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src import config as cfg
from src.metrics import regression_metrics

_DEMO = ['age', 'female']
_BLUP = ['blup_intercept', 'blup_slope']


def _fit_ridge(X_tr, y_tr, X_te, y_te):
    """Fit, predict, score in one line. Used for the five non-PI model variants."""
    return regression_metrics(y_te, Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_te))


def run_progressive(train, test):
    """Produce the prediction convergence curve: RMSE at each checkpoint for all model variants."""
    y_tr, y_te = train['seconds'].values, test['seconds'].values
    tr_b = train[train['blup_intercept'].notna()]
    te_b = test[test['blup_intercept'].notna()]
    y_tr_b, y_te_b = tr_b['seconds'].values, te_b['seconds'].values

    results = []
    for i, cp in enumerate(cfg.SPLIT_COLS):
        cumul = cfg.SPLIT_COLS[:i + 1]
        km, label = cfg.CHECKPOINT_KM[i], cfg.CP_ORDER[i]

        def add(model_label, r, n, pi=None):
            results.append({'checkpoint': label, 'km': km, 'model': model_label,
                            'rmse': r['rmse_s'], 'mae': r['mae_s'], 'r2': r['r2'],
                            'n_test': n, 'pi_width': pi})

        # Naive: constant-pace extrapolation
        add(cfg.NAIVE, regression_metrics(y_te, test[cp].values * (cfg.MARATHON_KM / km)), len(test))

        # Splits-only Ridge + PI
        m = Ridge(alpha=1.0).fit(train[cumul].values, y_tr)
        rs = regression_metrics(y_te, m.predict(test[cumul].values))
        train_resid = y_tr - m.predict(train[cumul].values)
        add(cfg.SPLITS, rs, len(test), np.percentile(train_resid, 95) - np.percentile(train_resid, 5))

        # Splits + demographics
        add(cfg.DEMO, _fit_ridge(train[cumul + _DEMO].values, y_tr, test[cumul + _DEMO].values, y_te), len(test))

        # Single checkpoint only
        add(cfg.SINGLE, _fit_ridge(train[[cp]].values, y_tr, test[[cp]].values, y_te), len(test))

        # Splits + demographics + year
        dy = cumul + _DEMO + ['year_c']
        add(cfg.DEMO_YEAR, _fit_ridge(train[dy].values, y_tr, test[dy].values, y_te), len(test))

        # Full (splits + demo + BLUPs) on BLUP subset
        full = cumul + _DEMO + _BLUP
        add(cfg.FULL, _fit_ridge(tr_b[full].values, y_tr_b, te_b[full].values, y_te_b), len(te_b))

        # Splits-only on BLUP subset for fair comparison
        add(cfg.SPLITS_SUBSET, _fit_ridge(tr_b[cumul].values, y_tr_b, te_b[cumul].values, y_te_b), len(te_b))

    return pd.DataFrame(results)


def crossover_analysis(results_df):
    """At which checkpoint does knowing how fast they're running beat knowing who they are?"""
    splits = results_df[results_df['model'] == cfg.SPLITS].sort_values('km')
    rows, found = [], False
    for _, r in splits.iterrows():
        beats = r['rmse'] < cfg.PERSONALIZED_RMSE
        rows.append({'checkpoint': r['checkpoint'], 'rmse': r['rmse'],
                     'beats': beats, 'is_crossover': beats and not found})
        if beats: found = True
    return pd.DataFrame(rows)


def cumulative_vs_single(results_df):
    """How much does using all prior splits help vs just the latest checkpoint?"""
    cumul = results_df[results_df['model'] == cfg.SPLITS].set_index('checkpoint')
    single = results_df[results_df['model'] == cfg.SINGLE].set_index('checkpoint')
    df = pd.DataFrame([{'checkpoint': cp, 'cumul_rmse': cumul.loc[cp, 'rmse'],
                         'single_rmse': single.loc[cp, 'rmse'],
                         'advantage': single.loc[cp, 'rmse'] - cumul.loc[cp, 'rmse']}
                        for cp in cfg.CP_ORDER])
    mid = df[df['checkpoint'].isin(['10K', '15K', '20K', 'HALF', '25K', '30K'])]
    return df, mid['advantage'].min(), mid['advantage'].max()


def year_degradation(results_df):
    """Does including the year feature help or hurt? (It hurts — extrapolation to 2017 test year.)"""
    demo = results_df[results_df['model'] == cfg.DEMO].set_index('checkpoint')
    demo_yr = results_df[results_df['model'] == cfg.DEMO_YEAR].set_index('checkpoint')
    df = pd.DataFrame([{'checkpoint': cp, 'no_year': demo.loc[cp, 'rmse'],
                         'with_year': demo_yr.loc[cp, 'rmse'],
                         'degradation': demo_yr.loc[cp, 'rmse'] - demo.loc[cp, 'rmse']}
                        for cp in cfg.CP_ORDER])
    return df, df['degradation'].max()


def feature_importance(train):
    """Which features dominate at early vs late checkpoints? Standardized coefficient ranking."""
    y = train['seconds'].values
    result = {}
    for idx in (0, 4, 8):
        feats = cfg.SPLIT_COLS[:idx + 1] + _DEMO
        coefs = Ridge(alpha=1.0).fit(StandardScaler().fit_transform(train[feats].values), y).coef_
        result[cfg.CP_ORDER[idx]] = sorted(zip(feats, coefs), key=lambda x: abs(x[1]), reverse=True)
    return result


def pi_convergence(results_df):
    """How does prediction uncertainty narrow as the race progresses?"""
    s = results_df[results_df['model'] == cfg.SPLITS].set_index('checkpoint')
    return pd.DataFrame([{'checkpoint': cp, 'rmse': s.loc[cp, 'rmse'],
                          'pi_width': s.loc[cp, 'pi_width'], 'pi_min': s.loc[cp, 'pi_width'] / 60}
                         for cp in cfg.CP_ORDER])

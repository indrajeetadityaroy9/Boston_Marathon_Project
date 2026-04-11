"""RQ3: Once the race starts, how quickly can we predict the finish time?

Runners pass nine checkpoints during the marathon (5K, 10K, 15K, 20K, half, 25K,
30K, 35K, 40K). At each one, we have their cumulative split times and ask: how
close can we get to predicting their final time?

Seven model variants are compared at every checkpoint:

  Naive         assume the runner keeps their current pace for the rest of the race
  Splits        Ridge regression on all split times collected so far
  Demo          Splits + age and gender
  Single        Ridge on only the most recent checkpoint (shows value of earlier splits)
  Demo+Year     Demo + race year (tests whether year helps or hurts)
  Full          Demo + the runner's personal parameters from RQ2 (only for known runners)
  Splits-subset Splits-only on the same subset as Full (fair comparison baseline)

Ridge regression is used instead of ordinary least squares because adjacent split
times are almost perfectly correlated (r > 0.99), which makes OLS coefficients
wildly unstable. Ridge adds a small penalty that stabilizes them.

The key findings are:
  - Prediction error drops from ~1259 seconds at 5K to ~124 seconds at 40K
  - Early in the race, knowing who the runner is (RQ2) beats knowing their pace
  - Cumulative splits outperform the latest split alone by 45-86 seconds at mid-race
  - Including race year as a feature hurts by up to 108 seconds (can't extrapolate)

Quantile regression prediction intervals give each runner their own interval width
based on their split times, rather than applying the same fixed width to everyone.
This better handles the right-skewed distribution where slow finishers are harder
to predict than fast ones.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from boston_marathon import config as cfg
from boston_marathon.metrics import empirical_coverage, regression_metrics

_DEMO = ['age', 'female']
_BLUP = ['blup_intercept', 'blup_slope']


def run_progressive(train, test):
    """The central analysis: fit all seven models at each checkpoint and record their accuracy.

    Loops through the nine checkpoints from 5K to 40K. At each one, builds the feature
    set from all splits available so far (cumulative) and fits every model variant.
    The Splits model also computes a prediction interval based on how spread out the
    training residuals are (5th to 95th percentile range).
    """
    y_tr, y_te = train['seconds'].values, test['seconds'].values

    # Subset to runners who have personal parameters from RQ2 (about 30% of split data)
    tr_b = train[train['blup_intercept'].notna()]
    te_b = test[test['blup_intercept'].notna()]
    y_tr_b, y_te_b = tr_b['seconds'].values, te_b['seconds'].values

    results = []
    for i, (cp, km, label) in enumerate(zip(cfg.SPLIT_COLS, cfg.CHECKPOINT_KM, cfg.CP_ORDER)):
        cumul = cfg.SPLIT_COLS[:i + 1]

        def record(model, metrics, n_test, pi_width=None):
            results.append({'checkpoint': label, 'km': km, 'model': model, 'rmse': metrics['rmse_s'], 'mae': metrics['mae_s'], 'r2': metrics['r2'], 'n_test': n_test, 'pi_width': pi_width})

        record(cfg.NAIVE, regression_metrics(y_te, test[cp].values * (cfg.MARATHON_KM / km)), len(test))

        split_model = Ridge(alpha=1.0).fit(train[cumul].values, y_tr)
        split_preds = split_model.predict(test[cumul].values)
        lo, hi = np.percentile(y_tr - split_model.predict(train[cumul].values), [5, 95])
        record(cfg.SPLITS, regression_metrics(y_te, split_preds), len(test), hi - lo)

        for model, feats, train_df, test_df, y_train, y_test in (
            (cfg.DEMO, cumul + _DEMO, train, test, y_tr, y_te),
            (cfg.SINGLE, [cp], train, test, y_tr, y_te),
            (cfg.DEMO_YEAR, cumul + _DEMO + ['year_c'], train, test, y_tr, y_te),
            (cfg.FULL, cumul + _DEMO + _BLUP, tr_b, te_b, y_tr_b, y_te_b),
            (cfg.SPLITS_SUBSET, cumul, tr_b, te_b, y_tr_b, y_te_b),
        ):
            preds = Ridge(alpha=1.0).fit(train_df[feats].values, y_train).predict(test_df[feats].values)
            record(model, regression_metrics(y_test, preds), len(test_df))

    return pd.DataFrame(results)


def crossover_analysis(results_df):
    """At which checkpoint does knowing the pace beat knowing the person?

    Compares the Splits model RMSE at each checkpoint against RQ2's in-sample
    personalized RMSE (996 seconds). The first checkpoint where splits win is
    the crossover point — before that, identity matters more than current speed.
    """
    splits = results_df[results_df['model'] == cfg.SPLITS].sort_values('km')[['checkpoint', 'rmse']].copy()
    splits['beats'] = splits['rmse'] < cfg.PERSONALIZED_RMSE
    splits['is_crossover'] = splits['beats'] & ~splits['beats'].shift(1, fill_value=False)
    return splits.reset_index(drop=True)


def cumulative_vs_single(results_df):
    """How much do earlier checkpoints help beyond just the latest one?

    At mid-race, using all splits so far gives 45-86 seconds less error than using
    only the most recent checkpoint. This means earlier splits carry information
    about pacing strategy (fast start vs slow start) that the latest time alone misses.
    """
    cumul = results_df[results_df['model'] == cfg.SPLITS].set_index('checkpoint')
    single = results_df[results_df['model'] == cfg.SINGLE].set_index('checkpoint')
    df = pd.DataFrame([{'checkpoint': cp, 'cumul_rmse': cumul.loc[cp, 'rmse'], 'single_rmse': single.loc[cp, 'rmse'], 'advantage': single.loc[cp, 'rmse'] - cumul.loc[cp, 'rmse']} for cp in cfg.CP_ORDER])
    mid = df[df['checkpoint'].isin(['10K', '15K', '20K', 'HALF', '25K', '30K'])]
    return df, mid['advantage'].min(), mid['advantage'].max()


def year_degradation(results_df):
    """Does including the race year as a feature help or hurt?

    The model trains on 2015-2016 but predicts 2017. Since 2017 wasn't in training,
    the year coefficient has to extrapolate, which adds error. At 5K the damage is
    up to 108 seconds. This is why the main Demo model excludes year.
    """
    demo = results_df[results_df['model'] == cfg.DEMO].set_index('checkpoint')
    demo_yr = results_df[results_df['model'] == cfg.DEMO_YEAR].set_index('checkpoint')
    df = pd.DataFrame([{'checkpoint': cp, 'no_year': demo.loc[cp, 'rmse'], 'with_year': demo_yr.loc[cp, 'rmse'], 'degradation': demo_yr.loc[cp, 'rmse'] - demo.loc[cp, 'rmse']} for cp in cfg.CP_ORDER])
    return df, df['degradation'].max()


def feature_importance(train):
    """Which features drive the prediction at early, middle, and late race stages?

    Fits Ridge on standardized features at 5K, halfway, and 40K. Standardizing
    makes coefficient sizes comparable across features with different units (split
    times in thousands of seconds vs gender as 0/1). At every stage the latest split
    dominates, but demographics contribute most early when split data is sparse.
    """
    y = train['seconds'].values
    result = {}
    for idx in (0, 4, 8):
        feats = cfg.SPLIT_COLS[:idx + 1] + _DEMO
        coefs = Ridge(alpha=1.0).fit(StandardScaler().fit_transform(train[feats].values), y).coef_
        result[cfg.CP_ORDER[idx]] = sorted(zip(feats, coefs), key=lambda item: abs(item[1]), reverse=True)
    return result


def pi_convergence(results_df):
    """How does the prediction interval shrink as the race progresses?

    Pulls the percentile-based prediction interval width from the Splits model at
    each checkpoint. The interval narrows from about 56 minutes at 5K down to about
    3 minutes at 40K, matching the RMSE convergence.
    """
    splits = results_df[results_df['model'] == cfg.SPLITS].set_index('checkpoint')
    return pd.DataFrame([{'checkpoint': cp, 'rmse': splits.loc[cp, 'rmse'], 'pi_width': splits.loc[cp, 'pi_width'], 'pi_min': splits.loc[cp, 'pi_width'] / 60} for cp in cfg.CP_ORDER])


def quantile_pi(train, test):
    """Compare two ways of building 90% prediction intervals.

    The percentile method takes the Ridge model's training errors and uses the 5th
    and 95th percentile as a fixed-width band around each prediction. Every runner
    gets the same band width regardless of their split times.

    Quantile regression fits separate models for the 5th and 95th percentile of
    finish time, so the band width adapts to each runner's splits. A runner with
    erratic early splits gets a wider interval than a steady one. This better
    handles the right-skewed distribution where slow finishers are harder to predict.

    Both methods use splits-only features for a fair comparison.
    """
    y_tr, y_te = train['seconds'].values, test['seconds'].values
    rows = []
    for i, cp in enumerate(cfg.SPLIT_COLS):
        cumul = cfg.SPLIT_COLS[:i + 1]
        X_tr, X_te = train[cumul].values, test[cumul].values

        # Quantile regression: separate models for the low and high bounds
        qr_lo = QuantileRegressor(quantile=0.05, alpha=0.001, solver='highs').fit(X_tr, y_tr)
        qr_hi = QuantileRegressor(quantile=0.95, alpha=0.001, solver='highs').fit(X_tr, y_tr)
        lo, hi = qr_lo.predict(X_te), qr_hi.predict(X_te)

        # Percentile method: fixed band from training residuals shifted by prediction
        m = Ridge(alpha=1.0).fit(X_tr, y_tr)
        preds = m.predict(X_te)
        resid_lo, resid_hi = np.percentile(y_tr - m.predict(X_tr), [5, 95])
        pctl_lo, pctl_hi = preds + resid_lo, preds + resid_hi
        rows.append({'checkpoint': cfg.CP_ORDER[i], 'qr_width': np.mean(hi - lo), 'qr_coverage': empirical_coverage(y_te, lo, hi), 'pctl_width': np.mean(pctl_hi - pctl_lo), 'pctl_coverage': empirical_coverage(y_te, pctl_lo, pctl_hi)})
    return pd.DataFrame(rows)

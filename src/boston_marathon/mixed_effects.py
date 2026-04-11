"""RQ2: Does knowing who the runner is make predictions better?

About half the dataset consists of runners who appear in multiple years. This module
gives each repeat runner their own personal parameters (how fast they are, how quickly
they're aging) on top of the population-level trends from RQ1.

Four models are fit in sequence, each adding complexity:

  Plain OLS                          same as RQ1's quadratic model, no personal parameters
  Random Intercept                   each runner gets their own baseline speed offset
  Random Intercept + Slope           each runner also gets their own aging rate
  Random Intercept + Slope + Weather adds race-day temperature, humidity, and wind

All mixed-effects models are fit in R via lme4_backend.fit_lmer because R's lme4
is much faster than Python alternatives on 188K observations with 66K groups.

The per-runner adjustments (Best Linear Unbiased Predictions) from the Random
Intercept + Slope model are saved to CSV so RQ3 can use them as features in the
checkpoint prediction models.

Four sensitivity analyses check whether the results are fragile:
  - Weather: does race-day weather explain why some years are faster?
  - Log-transform: does log(seconds) fix the skewed residuals?
  - Spline age: is the quadratic age curve too simple?
  - Survival bias: do runners who quit age differently from those who stay?
"""
import numpy as np
import statsmodels.api as sm
from scipy import stats

from boston_marathon import config as cfg
from boston_marathon.lme4_backend import fit_lmer
from boston_marathon.metrics import prediction_interval_width, rmse

# R-syntax formulas for lme4. The part in parentheses defines per-runner parameters:
#   (1 | runner)           = each runner gets a personal speed offset
#   (1 + age_c | runner)   = each runner gets a personal speed offset AND aging rate
RI_FORMULA = 'seconds ~ age_c + female + year_c + (1 | display_name)'
RIRS_FORMULA = 'seconds ~ age_c + female + year_c + (1 + age_c | display_name)'
WEATHER_FORMULA = 'seconds ~ age_c + female + year_c + temp_c + humid_c + wind_c + (1 + age_c | display_name)'

# Column lists used when building the design matrix for predictions
_EXOG_COLS = ['age_c', 'female', 'year_c']
_WEATHER_EXOG_COLS = ['age_c', 'female', 'year_c', 'temp_c', 'humid_c', 'wind_c']
_WEATHER_COLS = ['temp_c', 'humid_c', 'wind_c']

# lme4 calls the intercept '(Intercept)' but statsmodels calls it 'const'
_SM_TO_LME4 = {'const': '(Intercept)'}


def _fe_array(fe_params, col_names):
    """Reorder lme4's coefficient dict into a numpy array matching the statsmodels column layout."""
    return np.array([fe_params[_SM_TO_LME4.get(c, c)] for c in col_names])


def _runner_slopes(df, names):
    """How many seconds per year is each runner slowing down (or speeding up)?

    Computes a simple linear slope of finish time vs age for each runner, all at
    once using vectorized pandas math instead of looping over runners one by one.
    """
    sub = df[df['display_name'].isin(names)]
    grp = sub.groupby('display_name')
    dx = sub['age'] - grp['age'].transform('mean')
    dy = sub['seconds'] - grp['seconds'].transform('mean')
    return (dx * dy).groupby(sub['display_name']).sum() / (dx ** 2).groupby(sub['display_name']).sum()


def compute_runner_slopes(df):
    """Get aging slopes for runners with at least 5 years of age data.

    Runners with only 2-3 years of data give noisy slope estimates, so we require
    a minimum age span. The large variation in these slopes (SD ~ 267 sec/yr, with
    34% of runners actually getting faster) is what justifies giving each runner
    their own aging rate in the mixed-effects model.
    """
    age_span = df.groupby('display_name')['age'].agg(lambda s: s.max() - s.min())
    return _runner_slopes(df, age_span[age_span >= 5].index)


def _add_weather_cols(df, center_means=None):
    """Add race-day temperature, humidity, and wind as centered columns.

    Converts temperature from Fahrenheit to Celsius and subtracts the mean from
    each weather variable so the intercept stays interpretable. When center_means
    is provided (from a training set), those same means are applied to test data
    to prevent information leakage.
    """
    from boston_marathon.weather import get_race_weather

    humidity, wind = get_race_weather()
    df['temp_c'] = (df['year'].map(cfg.RACE_TEMP_F) - 32) * 5 / 9
    df['humid_c'] = df['year'].map(humidity).astype(float)
    df['wind_c'] = df['year'].map(wind)
    if center_means is None:
        center_means = {col: df[col].mean() for col in _WEATHER_COLS}
    for col in _WEATHER_COLS:
        df[col] -= center_means[col]
    return center_means


def fit_mixed_models(df, weather_df=None):
    """Fit the full sequence of models: OLS, random intercept, random intercept+slope.

    Each lme4 model is fit twice: once with REML (gives better variance estimates
    for reporting) and once with ML (needed for fair likelihood ratio tests).
    If weather_df is provided, also fits the weather-augmented model.
    """
    result = {'m0': sm.OLS(df['seconds'].values, sm.add_constant(df[_EXOG_COLS])).fit()}
    for prefix, formula, data in (('m1', RI_FORMULA, df), ('m2', RIRS_FORMULA, df), ('m3', WEATHER_FORMULA, weather_df)):
        if data is None:
            continue
        result[f'{prefix}_reml'] = fit_lmer(data, formula, reml=True)
        result[f'{prefix}_ml'] = fit_lmer(data, formula, reml=False)
    return result


def export_blups(result, output_path):
    """Save each runner's Best Linear Unbiased Predictions (intercept and slope) to CSV.

    RQ3 loads these as features in the checkpoint prediction models, letting it
    incorporate "who this runner is" alongside "how fast they're currently running."
    """
    blup_df = result['blup_df'].copy().set_axis(['blup_intercept', 'blup_slope'], axis=1).rename_axis('display_name').reset_index()
    blup_df.to_csv(output_path, index=False)
    return blup_df


def evaluate_personalization(result, y, df, exog_cols=None):
    """How many seconds of prediction error does knowing the runner remove?

    Compares two prediction modes:
      Marginal  - predict using only population trends (as if the runner is new)
      Conditional - predict using population trends + this runner's personal adjustments

    The difference is the "value of personalization" — the payoff from having
    individual race history. Also computes prediction interval widths for both cases.
    """
    exog_cols = _EXOG_COLS if exog_cols is None else exog_cols
    exog = sm.add_constant(df[exog_cols])
    marg_pred = exog.values @ _fe_array(result['fe_params'], exog.columns)
    marg_rmse = rmse(y, marg_pred)
    sigma2, tau2_0 = result['sigma2'], result['tau2_0']
    return {
        'marg_rmse': marg_rmse, 'cond_rmse': result['cond_rmse'], 'personalization': marg_rmse - result['cond_rmse'],
        'marg_mae': np.abs(y - marg_pred).mean(), 'cond_mae': np.abs(result['resid']).mean(),
        'pi_known': prediction_interval_width(sigma2), 'pi_new': prediction_interval_width(sigma2, tau2_0),
    }


def temporal_holdout(df_full):
    """The honest out-of-sample test: train on past data, predict future years.

    Trains on 2000-2016 and tests on 2017-2019 to simulate real-world deployment
    where you can't see the future. Fits both the base model and weather-augmented model,
    evaluates each on runners seen during training vs brand-new runners, and exports
    leak-free per-runner predictions that RQ3 can safely use for the 2017 test year.

    Weather columns are centered using only the training set mean — using the full
    dataset mean would leak information about 2017-2019 weather into the training data.
    """
    from boston_marathon.data import add_centered_features

    train_df = df_full[df_full['year'] <= cfg.RQ2_HOLDOUT_MAX_YEAR].copy()
    train_df = train_df[train_df.groupby('display_name')['display_name'].transform('size') > 1].copy()
    test_df = df_full[df_full['year'] > cfg.RQ2_HOLDOUT_MAX_YEAR].copy()
    train_runners = set(train_df['display_name'].unique())
    test_known = test_df[test_df['display_name'].isin(train_runners)].copy()
    test_new = test_df[~test_df['display_name'].isin(train_runners)].copy()

    train_df = train_df.sort_values('display_name').reset_index(drop=True)
    age_mean = train_df['age'].mean()
    for frame in (train_df, test_known, test_new):
        add_centered_features(frame, age_mean, cfg.YEAR_CENTER)

    # Base model without weather
    result = fit_lmer(train_df, RIRS_FORMULA, reml=True)
    known_exog = sm.add_constant(test_known[_EXOG_COLS])
    fe_vals = _fe_array(result['fe_params'], known_exog.columns)
    known_blups = result['blup_df'].reindex(test_known['display_name'])
    marg_k = known_exog.values @ fe_vals
    cond_k = marg_k + known_blups.iloc[:, 0].to_numpy() + known_blups.iloc[:, 1].to_numpy() * test_known['age_c'].to_numpy()
    marg_n = sm.add_constant(test_new[_EXOG_COLS]).values @ fe_vals

    # Weather model — center using training means only to avoid leakage
    train_w = train_df.copy()
    w_means = _add_weather_cols(train_w)
    for frame in (test_known, test_new):
        _add_weather_cols(frame, center_means=w_means)
    result_w = fit_lmer(train_w, WEATHER_FORMULA, reml=True)
    known_w_exog = sm.add_constant(test_known[_WEATHER_EXOG_COLS])
    fe_vals_w = _fe_array(result_w['fe_params'], known_w_exog.columns)
    known_w_blups = result_w['blup_df'].reindex(test_known['display_name'])
    marg_k_w = known_w_exog.values @ fe_vals_w
    cond_k_w = marg_k_w + known_w_blups.iloc[:, 0].to_numpy() + known_w_blups.iloc[:, 1].to_numpy() * test_known['age_c'].to_numpy()
    marg_n_w = sm.add_constant(test_new[_WEATHER_EXOG_COLS]).values @ fe_vals_w

    export_blups(result, cfg.BLUP_LEAKFREE_CSV)
    return {
        'converged': result['converged'], 'n_train': len(train_df), 'n_train_runners': train_df['display_name'].nunique(),
        'n_test': len(test_df), 'n_known': len(test_known), 'n_known_runners': test_known['display_name'].nunique(),
        'n_new': len(test_new), 'n_new_runners': test_new['display_name'].nunique(),
        'marg_rmse_known': rmse(test_known['seconds'].values, marg_k), 'cond_rmse_known': rmse(test_known['seconds'].values, cond_k),
        'marg_rmse_new': rmse(test_new['seconds'].values, marg_n), 'n_blups_exported': len(result['blup_df']),
        'w_converged': result_w['converged'], 'w_marg_rmse_known': rmse(test_known['seconds'].values, marg_k_w),
        'w_cond_rmse_known': rmse(test_known['seconds'].values, cond_k_w), 'w_marg_rmse_new': rmse(test_new['seconds'].values, marg_n_w),
    }


# Sensitivity analyses

def sensitivity_weather(df):
    """How much does race-day weather explain about year-to-year variation?

    Adds temperature, humidity, and wind to the model and checks whether the years
    that used to have large residuals (model was consistently off) now have smaller
    ones. Temperature alone explains about 72% of the year-level residual pattern.
    """
    df = df.copy()
    _add_weather_cols(df)
    r = fit_lmer(df, WEATHER_FORMULA, reml=True)
    year_means = df.assign(resid=r['resid']).groupby('year')['resid'].mean()
    fe, se = r['fe_params'], r['fe_se']
    pvals = {name: 2 * stats.norm.sf(abs(fe[name] / se[name])) for name in _WEATHER_COLS}
    return {
        'converged': r['converged'], 'cond_rmse': r['cond_rmse'], 'flagged': int((year_means.abs() > 100).sum()), 'year_std': year_means.std(),
        'temp_coef': fe['temp_c'], 'temp_se': se['temp_c'], 'temp_p': pvals['temp_c'],
        'humid_coef': fe['humid_c'], 'humid_se': se['humid_c'], 'humid_p': pvals['humid_c'],
        'wind_coef': fe['wind_c'], 'wind_se': se['wind_c'], 'wind_p': pvals['wind_c'],
    }


def sensitivity_log(df):
    """Are the results driven by the skewed finish-time distribution?

    Refits the model on log(seconds) instead of raw seconds. If the conclusions
    are similar, the skew isn't distorting things. Uses Duan smearing to convert
    log-scale predictions back to seconds for a fair RMSE comparison.
    """
    df = df.copy()
    y_raw = df['seconds'].values
    df['log_seconds'] = np.log(y_raw)
    r = fit_lmer(df, 'log_seconds ~ age_c + female + year_c + (1 + age_c | display_name)', reml=True)
    smearing = np.mean(np.exp(r['resid']))
    return {'converged': r['converged'], 'skewness': stats.skew(r['resid']), 'kurtosis': stats.kurtosis(r['resid']), 'rmse_bt': rmse(y_raw, np.exp(df['log_seconds'].values - r['resid']) * smearing), 'smearing_factor': smearing, 'fe_params': r['fe_params']}


def sensitivity_spline(df):
    """Is the simple age-squared curve missing important aging patterns?

    Compares the quadratic age model against a flexible B-spline with 5 degrees of
    freedom. If the spline has much better AIC but similar RMSE, the quadratic
    captures the main trend but misses fine details that don't help prediction.
    """
    spline = fit_lmer(df, 'seconds ~ splines::bs(age_c, df=5) + female + year_c + (1 + age_c | display_name)', reml=False)
    quadratic = fit_lmer(df, 'seconds ~ age_c + I(age_c^2) + female + year_c + (1 + age_c | display_name)', reml=False)
    return {
        'quad_aic': quadratic['aic'], 'quad_bic': quadratic['bic'], 'quad_rmse': quadratic['cond_rmse'],
        'spline_aic': spline['aic'], 'spline_bic': spline['bic'], 'spline_rmse': spline['cond_rmse'],
        'daic': spline['aic'] - quadratic['aic'], 'dbic': spline['bic'] - quadratic['bic'],
    }


def sensitivity_survival(df):
    """Are the aging estimates biased because slow runners stop qualifying?

    Compares the aging rate of runners still active in 2017-2019 against those who
    dropped out before 2015. If slower runners quit at higher rates, the remaining
    sample would look artificially healthy (survival bias). A small difference means
    Boston's qualifying standard keeps the sample relatively unbiased.
    """
    runner_stats = df.groupby('display_name').agg(n_races=('year', 'count'), age_span=('age', lambda s: s.max() - s.min()), last_year=('year', 'max'))
    eligible = runner_stats[(runner_stats['n_races'] > 1) & (runner_stats['age_span'] >= 5)]
    sa = _runner_slopes(df, eligible[eligible['last_year'] >= 2017].index)
    sd = _runner_slopes(df, eligible[eligible['last_year'] < 2015].index)
    diff = sa.mean() - sd.mean()
    return {'n_active': len(sa), 'slope_active': sa.mean(), 'std_active': sa.std(), 'n_dropped': len(sd), 'slope_dropped': sd.mean(), 'std_dropped': sd.std(), 'diff': diff, 'pct_diff': abs(diff) / sa.mean() * 100}

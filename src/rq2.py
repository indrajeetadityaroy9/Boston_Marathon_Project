"""RQ2: How much does knowing the individual runner improve prediction?

Fits three nested mixed-effects models via lme4_backend.py:
  M2.0 OLS:   fixed effects only (same structure as RQ1's quadratic model)
  M2.1 RI:    + per-runner baseline ability (random intercept)
  M2.2 RI+RS: + per-runner aging rate (random slope on age)

Also runs four sensitivity analyses: weather covariates (weather.py), log-transform,
penalized spline age, and survival bias. Exports per-runner BLUPs used by RQ3.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src import config as cfg
from src.lme4_backend import fit_lmer
from src.metrics import rmse, boundary_lrt, prediction_interval_width
from sklearn.metrics import mean_absolute_error

RI_FORMULA = 'seconds ~ age_c + female + year_c + (1 | display_name)'
RIRS_FORMULA = 'seconds ~ age_c + female + year_c + (1 + age_c | display_name)'
_EXOG_COLS = ['age_c', 'female', 'year_c']
_SM_TO_LME4 = {'const': '(Intercept)'}


def _fe_array(fe_params, col_names):
    """Translate between statsmodels ('const') and lme4 ('(Intercept)') naming."""
    return np.array([fe_params[_SM_TO_LME4.get(c, c)] for c in col_names])


def _runner_slopes(df, names):
    """How fast does each runner slow down? Vectorized OLS slope of seconds vs age."""
    sub = df[df['display_name'].isin(names)]
    grp = sub.groupby('display_name')
    dx = sub['age'] - grp['age'].transform('mean')
    dy = sub['seconds'] - grp['seconds'].transform('mean')
    return (dx * dy).groupby(sub['display_name']).sum() / (dx ** 2).groupby(sub['display_name']).sum()


def compute_runner_slopes(df):
    """Aging slopes for runners observed over at least 5 years (meaningful span)."""
    age_grp = df.groupby('display_name')['age']
    eligible = (age_grp.max() - age_grp.min()) >= 5
    return _runner_slopes(df, eligible[eligible].index)


def fit_mixed_models(df):
    """Fit all five model variants (OLS + 2 RI + 2 RI+RS) for comparison."""
    m0 = sm.OLS(df['seconds'].values, sm.add_constant(df[_EXOG_COLS])).fit()
    return {'m0': m0,
            'm1_reml': fit_lmer(df, RI_FORMULA, reml=True),
            'm1_ml': fit_lmer(df, RI_FORMULA, reml=False),
            'm2_reml': fit_lmer(df, RIRS_FORMULA, reml=True),
            'm2_ml': fit_lmer(df, RIRS_FORMULA, reml=False)}


def model_comparison_table(m0, m1_ml, m2_ml):
    """Does adding random effects significantly improve fit? Compare via AIC/BIC/LRT."""
    rows = [('OLS', 5, m0.llf, m0.aic, m0.bic),
            ('Rand-Int', 6, m1_ml['loglik'], m1_ml['aic'], m1_ml['bic']),
            ('RI+Slope', 8, m2_ml['loglik'], m2_ml['aic'], m2_ml['bic'])]
    return rows, boundary_lrt(m0.llf, m1_ml['loglik'], 1), boundary_lrt(m1_ml['loglik'], m2_ml['loglik'], 2)


def export_blups(result, output_path):
    """Write per-runner random effects to CSV for use by RQ3's checkpoint models."""
    blup_df = result['blup_df'].copy()
    blup_df.columns = ['blup_intercept', 'blup_slope']
    blup_df.insert(0, 'display_name', blup_df.index)
    blup_df.reset_index(drop=True, inplace=True)
    blup_df.to_csv(output_path, index=False)
    return blup_df


def evaluate_personalization(result, y, df):
    """How much better is prediction when you know the runner? Marginal vs conditional RMSE."""
    exog = sm.add_constant(df[_EXOG_COLS])
    marg_pred = exog.values @ _fe_array(result['fe_params'], exog.columns)
    marg_rmse = rmse(y, marg_pred)
    sigma2, tau2_0 = result['sigma2'], result['tau2_0']
    return {'marg_rmse': marg_rmse, 'cond_rmse': result['cond_rmse'],
            'personalization': marg_rmse - result['cond_rmse'],
            'marg_mae': mean_absolute_error(y, marg_pred),
            'cond_mae': np.mean(np.abs(result['resid'])),
            'pi_known': prediction_interval_width(sigma2),
            'pi_new': prediction_interval_width(sigma2, tau2_0)}


def temporal_holdout(df_full):
    """Honest test: train on 2000-2016, predict 2017-2019, export leak-free BLUPs for RQ3."""
    from src.data import add_centered_features
    train_df = df_full[df_full['year'] <= cfg.RQ2_HOLDOUT_MAX_YEAR].copy()
    train_df = train_df[train_df.groupby('display_name')['display_name'].transform('size') > 1].copy()
    test_df = df_full[df_full['year'] > cfg.RQ2_HOLDOUT_MAX_YEAR].copy()
    train_runners = set(train_df['display_name'].unique())
    test_known = test_df[test_df['display_name'].isin(train_runners)].copy()
    test_new = test_df[~test_df['display_name'].isin(train_runners)].copy()

    train_df.sort_values('display_name', inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    age_mean_tr = train_df['age'].mean()
    for d in [train_df, test_known, test_new]:
        add_centered_features(d, age_mean_tr, cfg.YEAR_CENTER)

    result = fit_lmer(train_df, RIRS_FORMULA, reml=True)
    blup_df = result['blup_df']

    # Marginal + conditional predictions for known runners
    X_k = sm.add_constant(test_known[_EXOG_COLS])
    fe_vals = _fe_array(result['fe_params'], X_k.columns)
    marg_k = X_k.values @ fe_vals
    cond_k = marg_k + blup_df.iloc[:, 0][test_known['display_name']].values + \
             blup_df.iloc[:, 1][test_known['display_name']].values * test_known['age_c'].values
    # Marginal for new runners
    marg_n = sm.add_constant(test_new[_EXOG_COLS]).values @ fe_vals

    export_blups(result, cfg.BLUP_LEAKFREE_CSV)
    return {'converged': result['converged'],
            'n_train': len(train_df), 'n_train_runners': train_df['display_name'].nunique(),
            'n_test': len(test_df), 'n_known': len(test_known), 'n_known_runners': test_known['display_name'].nunique(),
            'n_new': len(test_new), 'n_new_runners': test_new['display_name'].nunique(),
            'marg_rmse_known': rmse(test_known['seconds'].values, marg_k),
            'cond_rmse_known': rmse(test_known['seconds'].values, cond_k),
            'marg_rmse_new': rmse(test_new['seconds'].values, marg_n),
            'n_blups_exported': len(blup_df)}


def sensitivity_weather(df):
    """Does race-day weather explain the year-to-year residual pattern? Adds temp + humidity + wind."""
    from src.weather import get_race_weather
    humidity, wind = get_race_weather()
    df = df.copy()
    df['temp_c'] = (df['year'].map(cfg.RACE_TEMP_F) - 32) * 5 / 9
    df['humid_c'] = df['year'].map(humidity).astype(float)
    df['wind_c'] = df['year'].map(wind)
    for col in ['temp_c', 'humid_c', 'wind_c']:
        df[col] -= df[col].mean()
    r = fit_lmer(df, 'seconds ~ age_c + female + year_c + temp_c + humid_c + wind_c + (1 + age_c | display_name)', reml=True)
    year_means = pd.DataFrame({'year': df['year'], 'resid': r['resid']}).groupby('year')['resid'].mean()
    fe, se = r['fe_params'], r['fe_se']
    def _p(name): return 2 * (1 - stats.norm.cdf(abs(fe[name] / se[name])))
    return {'converged': r['converged'], 'cond_rmse': r['cond_rmse'],
            'flagged': int((year_means.abs() > 100).sum()), 'year_std': year_means.std(),
            'temp_coef': fe['temp_c'], 'temp_se': se['temp_c'], 'temp_p': _p('temp_c'),
            'humid_coef': fe['humid_c'], 'humid_se': se['humid_c'], 'humid_p': _p('humid_c'),
            'wind_coef': fe['wind_c'], 'wind_se': se['wind_c'], 'wind_p': _p('wind_c')}


def sensitivity_log(df):
    """Are conclusions robust to the non-normal residuals? Refit on log scale, back-transform."""
    df = df.copy()
    y_raw = df['seconds'].values
    df['log_seconds'] = np.log(y_raw)
    r = fit_lmer(df, 'log_seconds ~ age_c + female + year_c + (1 + age_c | display_name)', reml=True)
    smearing = np.mean(np.exp(r['resid']))
    return {'converged': r['converged'], 'skewness': stats.skew(r['resid']),
            'kurtosis': stats.kurtosis(r['resid']),
            'rmse_bt': rmse(y_raw, np.exp(df['log_seconds'].values - r['resid']) * smearing),
            'smearing_factor': smearing, 'fe_params': r['fe_params']}


def sensitivity_spline(df):
    """Is the quadratic age assumption too rigid? Compare against a flexible B-spline."""
    r_s = fit_lmer(df, 'seconds ~ splines::bs(age_c, df=5) + female + year_c + (1 + age_c | display_name)', reml=False)
    r_q = fit_lmer(df, 'seconds ~ age_c + I(age_c^2) + female + year_c + (1 + age_c | display_name)', reml=False)
    return {'quad_aic': r_q['aic'], 'quad_bic': r_q['bic'], 'quad_rmse': r_q['cond_rmse'],
            'spline_aic': r_s['aic'], 'spline_bic': r_s['bic'], 'spline_rmse': r_s['cond_rmse'],
            'daic': r_s['aic'] - r_q['aic'], 'dbic': r_s['bic'] - r_q['bic']}


def sensitivity_survival(df):
    """Do runners who quit age differently? Compare slopes for active vs dropped-out runners."""
    runner_stats = df.groupby('display_name').agg(
        n_races=('year', 'count'), age_span=('age', lambda s: s.max() - s.min()),
        last_year=('year', 'max'))
    eligible = runner_stats[(runner_stats['n_races'] > 1) & (runner_stats['age_span'] >= 5)]
    sa = _runner_slopes(df, eligible[eligible['last_year'] >= 2017].index)
    sd = _runner_slopes(df, eligible[eligible['last_year'] < 2015].index)
    diff = sa.mean() - sd.mean()
    return {'n_active': len(sa), 'slope_active': sa.mean(), 'std_active': sa.std(),
            'n_dropped': len(sd), 'slope_dropped': sd.mean(), 'std_dropped': sd.std(),
            'diff': diff, 'pct_diff': abs(diff) / sa.mean() * 100}

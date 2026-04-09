"""RQ1: How well can demographics alone predict marathon finish time?

Fits three OLS models of increasing complexity:
  M1.0 Linear:    age, gender, year
  M1.1 Quadratic: + age^2 and age-gender interaction
  M1.2 History:   + prior race count and mean prior finish time (repeat runners only)

This establishes the prediction floor that RQ2 (personalization) and RQ3 (in-race splits) must beat.
Data comes from data.py; metrics from metrics.py; results printed by run_pipeline.py.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from src.metrics import regression_metrics


LINEAR_FEATS = ['age_c', 'female', 'year_c']
QUAD_FEATS = ['age_c', 'age_c2', 'female', 'year_c', 'age_c_female']
HIST_FEATS = QUAD_FEATS + ['prior_appearances', 'prior_mean_time']


def fit_models(train, test):
    """Fit all three demographic models on train, evaluate on both train and test."""
    y_train, y_test = train['seconds'].values, test['seconds'].values
    models = {}
    results = []

    for label, feats in [('Linear OLS', LINEAR_FEATS), ('Quadratic OLS', QUAD_FEATS)]:
        X_train, X_test = train[feats].values, test[feats].values
        m = LinearRegression().fit(X_train, y_train)
        models[label] = {'features': feats, 'model': m}
        for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            r = regression_metrics(y, m.predict(X))
            r.update(model=label, split=split)
            results.append(r)

    train_h = train[train['prior_mean_time'].notna()]
    test_h = test[test['prior_mean_time'].notna()]
    m = LinearRegression().fit(train_h[HIST_FEATS].values, train_h['seconds'].values)
    models['History OLS'] = {'features': HIST_FEATS, 'model': m}
    for split, sub in [('train', train_h), ('test', test_h)]:
        r = regression_metrics(sub['seconds'].values, m.predict(sub[HIST_FEATS].values))
        r.update(model='History OLS', split=split, n=len(sub))
        results.append(r)

    return models, results


def cross_validate_quadratic(train):
    """Cross-validate the quadratic model with year-grouped folds (no future leakage)."""
    cv = cross_validate(LinearRegression(), train[QUAD_FEATS].values, train['seconds'].values,
                        cv=GroupKFold(n_splits=5), groups=train['year'].values,
                        scoring=['neg_mean_squared_error', 'r2'])
    rmses = np.sqrt(-cv['test_neg_mean_squared_error'])
    r2s = cv['test_r2']
    return {'fold_rmses': rmses, 'mean_rmse': rmses.mean(), 'std_rmse': rmses.std(),
            'fold_r2s': r2s, 'mean_r2': r2s.mean(), 'std_r2': r2s.std()}


def standardized_importance(train):
    """Which features matter most? Standardize, fit, rank by absolute coefficient."""
    X = StandardScaler().fit_transform(train[QUAD_FEATS].values)
    coefs = LinearRegression().fit(X, train['seconds'].values).coef_
    return sorted([(f, c, 'slower' if c > 0 else 'faster') for f, c in zip(QUAD_FEATS, coefs)],
                  key=lambda x: abs(x[1]), reverse=True)


def get_coefficients(models):
    """Extract raw coefficients and intercepts for display."""
    return [(label, list(zip(spec['features'], spec['model'].coef_)), spec['model'].intercept_)
            for label, spec in models.items()]

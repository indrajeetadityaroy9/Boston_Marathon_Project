"""RQ1: Can we predict marathon finish time from demographics alone?

This is the starting point of the analysis. Before the race, all we know about a
runner is their age, gender, and what year it is. We fit three progressively richer
models to see how much these features can predict:

  Linear      just age, gender, year (explains ~15% of variance)
  Quadratic   adds age-squared and an age-gender interaction (explains ~17%)
  History     adds the runner's average time from past Boston Marathons (explains ~51%)

The big jump from 17% to 51% shows that knowing a runner's personal history is far
more valuable than demographics. This motivates RQ2 (mixed-effects models that formally
capture individual differences) and sets the baseline RMSE that RQ3 (in-race splits)
must beat.

Features come from data.add_centered_features and data.add_prior_history.
Evaluation uses metrics.regression_metrics. Output is printed by scripts/run_pipeline.py.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from boston_marathon.metrics import regression_metrics

LINEAR_FEATS = ['age_c', 'female', 'year_c']
QUAD_FEATS = ['age_c', 'age_c2', 'female', 'year_c', 'age_c_female']
HIST_FEATS = QUAD_FEATS + ['prior_appearances', 'prior_mean_time']


def fit_models(train, test):
    """Train all three demographic models and measure their accuracy on both sets.

    The Linear and Quadratic models run on all runners. The History model only runs
    on runners who have at least one prior Boston Marathon appearance, because it
    needs their past finish times as input. This means the History model is evaluated
    on a smaller subset (about 19K of the 52K test runners).
    """
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
    """Check how stable the Quadratic model is across different year groupings.

    Splits the training data into 5 folds grouped by year (so all of 2005 is either
    in train or validation, never split across both). This prevents the model from
    "cheating" by learning year-specific patterns during training and testing on the
    same year. Reports the RMSE and R-squared for each fold plus the average.
    """
    cv = cross_validate(LinearRegression(), train[QUAD_FEATS].values, train['seconds'].values, cv=GroupKFold(n_splits=5), groups=train['year'].values, scoring=['neg_mean_squared_error', 'r2'])
    rmses = np.sqrt(-cv['test_neg_mean_squared_error'])
    r2s = cv['test_r2']
    return {'fold_rmses': rmses, 'mean_rmse': rmses.mean(), 'std_rmse': rmses.std(), 'fold_r2s': r2s, 'mean_r2': r2s.mean(), 'std_r2': r2s.std()}


def standardized_importance(train):
    """Which demographic features matter most for predicting finish time?

    Puts all features on the same scale (mean=0, std=1) before fitting, so the
    coefficient sizes directly show each feature's importance. Without scaling,
    age (range ~20-80) would look less important than gender (0 or 1) just because
    of different units. Returns features sorted from most to least influential,
    with direction (slower = positive coefficient, faster = negative).
    """
    X = StandardScaler().fit_transform(train[QUAD_FEATS].values)
    coefs = LinearRegression().fit(X, train['seconds'].values).coef_
    return sorted(((feat, coef, 'slower' if coef > 0 else 'faster') for feat, coef in zip(QUAD_FEATS, coefs)), key=lambda item: abs(item[1]), reverse=True)

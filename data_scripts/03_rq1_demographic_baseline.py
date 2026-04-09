#!/usr/bin/env python3
"""Pre-Race Demographic Baseline Prediction (Research Question 1).

Predicts marathon finish time from pre-race demographics alone (age, gender, year).
Establishes the baseline prediction error that the personalized (mixed-effects)
and in-race (checkpoint) models must beat.

Models:
  Linear OLS:     finish_time ~ centered_age + female + centered_year
  Quadratic OLS:  finish_time ~ centered_age + centered_age^2 + female + centered_year
                  + centered_age * female interaction
  History OLS:    Above + prior_appearances + prior_mean_time (repeat runners only)

Evaluation: temporal hold-out (train on 2000-2017, test on 2018-2019).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

CLEANED_PATH = Path(__file__).resolve().parent.parent / 'cleaned_data' / 'boston_marathon_cleaned.csv'

TRAIN_YEARS = range(2000, 2018)
TEST_YEARS = (2018, 2019)


def load_data():
    """Load cleaned data, filter to non-imputed ages from 2000 onward."""
    print("STEP 1: DATA LOADING")

    df = pd.read_csv(CLEANED_PATH, low_memory=False,
                     usecols=['year', 'display_name', 'age', 'gender', 'seconds', 'age_imputed'],
                     dtype={'gender': 'category', 'display_name': str})
    df = df[~df['age_imputed'] & df['age'].notna() & (df['year'] >= 2000)].copy()
    df = df.dropna(subset=['seconds'])
    print(f"  Non-imputed, age-valid, 2000+: {len(df):,} rows")

    # Compute prior race history (leak-free: uses only prior years via shift)
    df.sort_values(['display_name', 'year'], inplace=True)
    df['prior_mean_time'] = df.groupby('display_name')['seconds'].transform(
        lambda x: x.expanding().mean().shift(1))
    df['prior_appearances'] = df.groupby('display_name').cumcount()

    train = df[df['year'].isin(TRAIN_YEARS)].copy()
    test = df[df['year'].isin(TEST_YEARS)].copy()

    # Center age using train mean only to avoid test-data leakage
    age_mean = train['age'].mean()
    for d in [train, test]:
        d['age_c'] = d['age'] - age_mean
        d['age_c2'] = d['age_c'] ** 2
        d['year_c'] = d['year'] - 2010
        d['female'] = (d['gender'] == 'F').astype(int)
        d['age_c_female'] = d['age_c'] * d['female']

    n_hist_train = train['prior_mean_time'].notna().sum()
    n_hist_test = test['prior_mean_time'].notna().sum()
    print(f"  Age centered at {age_mean:.1f} (train mean)")
    print(f"  Train (2000-2017): {len(train):,} rows ({n_hist_train:,} with prior history)")
    print(f"  Test  (2018-2019): {len(test):,} rows ({n_hist_test:,} with prior history)")

    return train, test


def evaluate(y_true, y_pred, label):
    """Compute prediction error metrics: root mean squared error, mean absolute error,
    R-squared (variance explained), and mean absolute percentage error."""
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = np.mean(np.abs(residuals) / y_true) * 100
    return {'model': label, 'rmse_s': rmse, 'rmse_min': rmse / 60,
            'mae_s': mae, 'mae_min': mae / 60, 'r2': r2, 'mape': mape}


def fit_and_evaluate(train, test):
    """Fit demographic baseline models and evaluate on the held-out test set."""
    print("\nSTEP 2: MODEL FITTING AND EVALUATION")

    y_train = train['seconds'].values
    y_test = test['seconds'].values

    models = {
        'Linear OLS': {
            'features': ['age_c', 'female', 'year_c'],
            'model': LinearRegression(),
        },
        'Quadratic OLS': {
            'features': ['age_c', 'age_c2', 'female', 'year_c', 'age_c_female'],
            'model': LinearRegression(),
        },
    }

    results = []
    for label, spec in models.items():
        feats = spec['features']
        model = spec['model']

        X_train = train[feats].values
        X_test = test[feats].values

        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        res_train = evaluate(y_train, pred_train, f'{label} [train]')
        res_test = evaluate(y_test, pred_test, f'{label} [test]')
        results.extend([res_train, res_test])

    # History-augmented model: demographics + prior race history (repeat runners only)
    # prior_mean_time = average of all previous Boston Marathon finish times for this runner
    # prior_appearances = number of times this runner has previously appeared
    hist_feats = ['age_c', 'age_c2', 'female', 'year_c', 'age_c_female',
                  'prior_appearances', 'prior_mean_time']
    train_hist = train[train['prior_mean_time'].notna()].copy()
    test_hist = test[test['prior_mean_time'].notna()].copy()

    if len(train_hist) > 0 and len(test_hist) > 0:
        m_hist = LinearRegression()
        m_hist.fit(train_hist[hist_feats].values, train_hist['seconds'].values)

        res_train_h = evaluate(train_hist['seconds'].values,
                               m_hist.predict(train_hist[hist_feats].values),
                               'History OLS [train]')
        res_test_h = evaluate(test_hist['seconds'].values,
                              m_hist.predict(test_hist[hist_feats].values),
                              'History OLS [test]')
        results.extend([res_train_h, res_test_h])
        models['History OLS'] = {'features': hist_feats, 'model': m_hist}

        print(f"\n  History model evaluated on runners with at least one prior appearance:")
        print(f"    Train subset: {len(train_hist):,} rows")
        print(f"    Test subset:  {len(test_hist):,} rows")

    return results, models


def print_results(results):
    """Print comparison table."""
    print("\n--- Prediction Results ---")
    print(f"{'Model':<30} {'RMSE (s)':>10} {'RMSE (min)':>11} {'MAE (s)':>9} "
          f"{'MAE (min)':>10} {'R^2':>8} {'MAPE%':>7}")
    print("-" * 90)
    for r in results:
        print(f"  {r['model']:<28} {r['rmse_s']:>10.0f} {r['rmse_min']:>10.1f} "
              f"{r['mae_s']:>9.0f} {r['mae_min']:>9.1f} {r['r2']:>8.4f} {r['mape']:>6.1f}%")


def print_coefficients(train, models):
    """Print coefficient tables for interpretability."""
    print("\n--- Model Coefficients ---")

    for label, spec in models.items():
        feats = spec['features']
        model = spec['model']
        print(f"\n  {label}:")
        if hasattr(model, 'coef_'):
            for feat, coef in zip(feats, model.coef_):
                print(f"    {feat:>15}: {coef:>10.2f}")
            print(f"    {'intercept':>15}: {model.intercept_:>10.2f}")


def cross_validate(train):
    """5-fold cross-validation grouped by year on the quadratic model."""
    print("\nSTEP 3: CROSS-VALIDATION (5-fold grouped by year, quadratic model)")

    feats = ['age_c', 'age_c2', 'female', 'year_c', 'age_c_female']
    X = train[feats].values
    y = train['seconds'].values
    groups = train['year'].values

    model = LinearRegression()
    gkf = GroupKFold(n_splits=5)

    neg_mse = cross_val_score(model, X, y, cv=gkf, groups=groups,
                              scoring='neg_mean_squared_error')
    rmse_folds = np.sqrt(-neg_mse)
    print(f"  Fold RMSEs: {', '.join(f'{r:.0f}' for r in rmse_folds)}")
    print(f"  Mean RMSE: {rmse_folds.mean():.0f} +/- {rmse_folds.std():.0f}")

    r2_folds = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring='r2')
    print(f"  Fold R^2s: {', '.join(f'{r:.4f}' for r in r2_folds)}")
    print(f"  Mean R^2: {r2_folds.mean():.4f} +/- {r2_folds.std():.4f}")


def feature_importance(train):
    """Report which features contribute most via standardized coefficients."""
    print("\nSTEP 4: FEATURE IMPORTANCE (standardized coefficients)")

    feats = ['age_c', 'age_c2', 'female', 'year_c', 'age_c_female']
    X = train[feats].values
    y = train['seconds'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)

    importance = sorted(zip(feats, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {'Feature':>15} {'Std Coef':>10} {'Direction':>10}")
    print("  " + "-" * 38)
    for feat, coef in importance:
        direction = 'slower' if coef > 0 else 'faster'
        print(f"  {feat:>15} {coef:>10.1f} {direction:>10}")


def main():
    print("DEMOGRAPHIC BASELINE PREDICTION")
    print("Boston Marathon Finish Time -- Pre-Race Prediction from Demographics")
    print("=" * 70)

    train, test = load_data()
    results, models = fit_and_evaluate(train, test)
    print_results(results)
    print_coefficients(train, models)
    cross_validate(train)
    feature_importance(train)

    print("\n" + "=" * 70)
    test_results = [r for r in results if '[test]' in r['model']]
    best = min(test_results, key=lambda r: r['rmse_s'])
    print(f"Best test RMSE: {best['rmse_s']:.0f}s ({best['rmse_min']:.1f} min) -- {best['model']}")
    print(f"This is the baseline that the personalized and in-race models must beat.")


if __name__ == '__main__':
    main()

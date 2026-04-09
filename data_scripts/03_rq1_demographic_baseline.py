#!/usr/bin/env python3
"""RQ1: Demographic Baseline Prediction.

Predicts marathon finish time from pre-race demographics alone (age, gender, year).
Establishes the baseline RMSE that personalized (RQ2) and in-race (RQ3) models must beat.

Models:
  M1.0  OLS: seconds ~ age_c + female + year_c
  M1.1  OLS: seconds ~ age_c + age_c^2 + female + year_c + age_c:female

Evaluation: temporal hold-out (train 2000-2017, test 2018-2019).
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
                     usecols=['year', 'age', 'gender', 'seconds', 'age_imputed'],
                     dtype={'gender': 'category'})
    df['age_imputed'] = df['age_imputed'].astype(str).str.strip().str.lower() == 'true'
    df = df[~df['age_imputed'] & df['age'].notna() & (df['year'] >= 2000)].copy()
    df = df.dropna(subset=['seconds'])
    print(f"  Non-imputed, age-valid, 2000+: {len(df):,} rows")

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

    print(f"  Age centered at {age_mean:.1f} (train mean)")
    print(f"  Train (2000-2017): {len(train):,} rows")
    print(f"  Test  (2018-2019): {len(test):,} rows")

    return train, test


def evaluate(y_true, y_pred, label):
    """Compute RMSE, MAE, R^2, MAPE."""
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
    """Fit M1.0 and M1.1 and evaluate on test set."""
    print("\nSTEP 2: MODEL FITTING AND EVALUATION")

    y_train = train['seconds'].values
    y_test = test['seconds'].values

    models = {
        'M1.0 OLS (linear)': {
            'features': ['age_c', 'female', 'year_c'],
            'model': LinearRegression(),
        },
        'M1.1 OLS (quad+int)': {
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
    """5-fold GroupKFold CV (grouped by year) on training set for M1.1."""
    print("\nSTEP 3: CROSS-VALIDATION (5-fold GroupKFold by year, M1.1)")

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
    print("RQ1: DEMOGRAPHIC BASELINE PREDICTION")
    print("Boston Marathon Finish Time -- Pre-Race Prediction")
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
    print(f"This is the baseline that RQ2 (personalization) and RQ3 (in-race) must beat.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""RQ3: Progressive In-Race Prediction.

At each checkpoint (5K through 40K), predicts marathon finish time using
cumulative split times, optionally augmented with demographics and BLUPs.

Models at each checkpoint:
  M3.naive  -- Constant-pace extrapolation (no model)
  M3.splits -- Ridge regression on cumulative splits
  M3.demo   -- Ridge on splits + age, gender, year
  M3.full   -- Ridge on splits + age, gender, year + BLUP intercept/slope

Evaluation: train on 2015-2016, test on 2017.
Inspired by Aguilera Moreno (2026) progressive checkpoint prediction framework.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CLEANED_PATH = Path(__file__).resolve().parent.parent / 'cleaned_data' / 'boston_marathon_cleaned.csv'
BLUP_PATH = Path(__file__).resolve().parent.parent / 'cleaned_data' / 'runner_blups.csv'

# M2.2 in-sample conditional RMSE from 04_rq2_personalized_mixed_effects.py
M2_COND_RMSE = 996.2

SPLIT_COLS = ['5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds',
              'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds']

NEEDED_COLS = ['year', 'display_name', 'age', 'gender', 'seconds'] + SPLIT_COLS

CHECKPOINT_KM = {
    '5k_seconds': 5.0, '10k_seconds': 10.0, '15k_seconds': 15.0,
    '20k_seconds': 20.0, 'half_seconds': 21.0975, '25k_seconds': 25.0,
    '30k_seconds': 30.0, '35k_seconds': 35.0, '40k_seconds': 40.0,
}

MARATHON_KM = 42.195


def load_data():
    """Load 2015-2017 split data and join BLUPs."""
    print("STEP 1: DATA LOADING")

    df = pd.read_csv(CLEANED_PATH, low_memory=False, usecols=NEEDED_COLS)

    splits = df[df['year'].between(2015, 2017)].copy()
    splits = splits[splits[SPLIT_COLS].notna().all(axis=1) &
                    splits['seconds'].notna() &
                    splits['age'].notna()].copy()

    splits['female'] = (splits['gender'] == 'F').astype(int)
    splits['year_c'] = splits['year'] - 2010

    print(f"  Runners with complete splits: {len(splits):,}")

    # Load and join BLUPs from RQ2
    has_blups = False
    if BLUP_PATH.exists():
        blups = pd.read_csv(BLUP_PATH)
        splits = splits.merge(blups, on='display_name', how='left')
        n_with_blup = splits['blup_intercept'].notna().sum()
        print(f"  Runners with BLUPs from RQ2: {n_with_blup:,} ({n_with_blup/len(splits)*100:.1f}%)")
        has_blups = True
    else:
        print("  WARNING: runner_blups.csv not found. Run 04_rq2_personalized_mixed_effects.py first for M3.full.")
        splits['blup_intercept'] = np.nan
        splits['blup_slope'] = np.nan

    train = splits[splits['year'].isin([2015, 2016])].copy()
    test = splits[splits['year'] == 2017].copy()
    print(f"  Train (2015-2016): {len(train):,}")
    print(f"  Test  (2017):      {len(test):,}")

    return train, test, has_blups


def evaluate(y_true, y_pred):
    """Compute RMSE, MAE, R^2."""
    res = y_true - y_pred
    rmse = np.sqrt(np.mean(res ** 2))
    mae = np.mean(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return rmse, mae, r2


def run_progressive_prediction(train, test, has_blups):
    """Run all 4 model variants at each of 9 checkpoints."""
    print("\nSTEP 2: PROGRESSIVE CHECKPOINT PREDICTION")

    y_train = train['seconds'].values
    y_test = test['seconds'].values
    demo_cols = ['age', 'female', 'year_c']
    blup_cols = ['blup_intercept', 'blup_slope']

    # Prepare BLUP subsets (only runners with BLUPs for M3.full)
    train_blup = train[train['blup_intercept'].notna()].copy()
    test_blup = test[test['blup_intercept'].notna()].copy()

    results = []
    for i, cp in enumerate(SPLIT_COLS):
        cumul_splits = SPLIT_COLS[:i + 1]
        km = CHECKPOINT_KM[cp]
        cp_label = cp.replace('_seconds', '').upper()

        # M3.naive: constant-pace extrapolation using latest checkpoint
        naive_pred = test[cp].values * (MARATHON_KM / km)
        rmse_n, mae_n, r2_n = evaluate(y_test, naive_pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': 'M3.naive',
                        'rmse': rmse_n, 'mae': mae_n, 'r2': r2_n, 'n_test': len(test)})

        # M3.splits: Ridge on cumulative splits
        m = Ridge(alpha=1.0).fit(train[cumul_splits].values, y_train)
        pred = m.predict(test[cumul_splits].values)
        rmse_s, mae_s, r2_s = evaluate(y_test, pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': 'M3.splits',
                        'rmse': rmse_s, 'mae': mae_s, 'r2': r2_s, 'n_test': len(test)})

        # M3.demo: Ridge on splits + demographics
        demo_feats = cumul_splits + demo_cols
        m = Ridge(alpha=1.0).fit(train[demo_feats].values, y_train)
        pred = m.predict(test[demo_feats].values)
        rmse_d, mae_d, r2_d = evaluate(y_test, pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': 'M3.demo',
                        'rmse': rmse_d, 'mae': mae_d, 'r2': r2_d, 'n_test': len(test)})

        # M3.full: Ridge on splits + demographics + BLUPs (subset with BLUPs only)
        if has_blups and len(test_blup) > 0:
            full_feats = cumul_splits + demo_cols + blup_cols
            y_train_b = train_blup['seconds'].values
            y_test_b = test_blup['seconds'].values
            m = Ridge(alpha=1.0).fit(train_blup[full_feats].values, y_train_b)
            pred = m.predict(test_blup[full_feats].values)
            rmse_f, mae_f, r2_f = evaluate(y_test_b, pred)
            results.append({'checkpoint': cp_label, 'km': km, 'model': 'M3.full',
                            'rmse': rmse_f, 'mae': mae_f, 'r2': r2_f,
                            'n_test': len(test_blup)})

            # Also run M3.splits on same BLUP subset for fair comparison
            m = Ridge(alpha=1.0).fit(train_blup[cumul_splits].values, y_train_b)
            pred = m.predict(test_blup[cumul_splits].values)
            rmse_sb, mae_sb, r2_sb = evaluate(y_test_b, pred)
            results.append({'checkpoint': cp_label, 'km': km, 'model': 'M3.splits (BLUP subset)',
                            'rmse': rmse_sb, 'mae': mae_sb, 'r2': r2_sb,
                            'n_test': len(test_blup)})

    return pd.DataFrame(results)


def print_convergence_table(results_df):
    """Print the prediction convergence curve as a table."""
    print("\n--- Prediction Convergence (test set 2017) ---")

    # Main models on full test set
    main = results_df[results_df['model'].isin(['M3.naive', 'M3.splits', 'M3.demo'])]
    pivot = main.pivot(index='checkpoint', columns='model', values='rmse')
    pivot = pivot.reindex(
        [cp.replace('_seconds', '').upper() for cp in SPLIT_COLS]
    )

    print(f"\n{'Checkpoint':>12} {'M3.naive':>12} {'M3.splits':>12} {'M3.demo':>12}")
    print(f"{'':>12} {'RMSE (s)':>12} {'RMSE (s)':>12} {'RMSE (s)':>12}")
    print("-" * 52)
    for cp in [c.replace('_seconds', '').upper() for c in SPLIT_COLS]:
        row = pivot.loc[cp]
        print(f"  {cp:>10} {row.get('M3.naive', 0):>10.0f} "
              f"{row.get('M3.splits', 0):>10.0f} {row.get('M3.demo', 0):>10.0f}")

    # BLUP comparison at selected checkpoints
    blup_results = results_df[results_df['model'].isin(['M3.full', 'M3.splits (BLUP subset)'])]
    if not blup_results.empty:
        print(f"\n--- BLUP Augmentation (n={blup_results['n_test'].iloc[0]:,} runners with history) ---")
        print(f"{'Checkpoint':>12} {'M3.splits':>12} {'M3.full':>12} {'Improvement':>12}")
        print(f"{'':>12} {'RMSE (s)':>12} {'RMSE (s)':>12} {'(seconds)':>12}")
        print("-" * 52)
        for cp in [c.replace('_seconds', '').upper() for c in SPLIT_COLS]:
            s_row = blup_results[(blup_results['checkpoint'] == cp) &
                                 (blup_results['model'] == 'M3.splits (BLUP subset)')]
            f_row = blup_results[(blup_results['checkpoint'] == cp) &
                                 (blup_results['model'] == 'M3.full')]
            if not s_row.empty and not f_row.empty:
                s_rmse = s_row['rmse'].values[0]
                f_rmse = f_row['rmse'].values[0]
                improvement = s_rmse - f_rmse
                print(f"  {cp:>10} {s_rmse:>10.0f} {f_rmse:>10.0f} {improvement:>+10.0f}")


def print_r2_convergence(results_df):
    """Print R^2 convergence for all models."""
    print("\n--- R^2 Convergence ---")

    main = results_df[results_df['model'].isin(['M3.naive', 'M3.splits', 'M3.demo'])]
    pivot = main.pivot(index='checkpoint', columns='model', values='r2')
    pivot = pivot.reindex(
        [cp.replace('_seconds', '').upper() for cp in SPLIT_COLS]
    )

    print(f"\n{'Checkpoint':>12} {'M3.naive':>10} {'M3.splits':>10} {'M3.demo':>10}")
    print("-" * 46)
    for cp in [c.replace('_seconds', '').upper() for c in SPLIT_COLS]:
        row = pivot.loc[cp]
        print(f"  {cp:>10} {row.get('M3.naive', 0):>9.4f} "
              f"{row.get('M3.splits', 0):>9.4f} {row.get('M3.demo', 0):>9.4f}")


def feature_importance_at_checkpoints(train, test):
    """Show which features dominate at early vs late checkpoints via standardized coefficients."""
    print("\nSTEP 3: FEATURE IMPORTANCE AT SELECTED CHECKPOINTS (standardized)")

    demo_cols = ['age', 'female', 'year_c']
    y_train = train['seconds'].values

    for cp_idx, cp_label in [(0, '5K'), (4, 'HALF'), (8, '40K')]:
        cumul = SPLIT_COLS[:cp_idx + 1]
        feats = cumul + demo_cols
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train[feats].values)
        m = Ridge(alpha=1.0).fit(X_scaled, y_train)

        coefs = sorted(zip(feats, m.coef_), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  At {cp_label} (features: {len(feats)}):")
        for feat, coef in coefs[:5]:
            print(f"    {feat:>15}: {coef:>10.1f}")


def crossover_analysis(results_df):
    """Identify where splits-only prediction beats personalized history."""
    print("\nSTEP 4: CROSSOVER ANALYSIS")

    m2_cond_rmse = M2_COND_RMSE

    splits_results = results_df[results_df['model'] == 'M3.splits'].sort_values('km')
    print(f"\n  RQ2 M2.2 conditional RMSE (known runner, pre-race): {m2_cond_rmse:.0f}s")
    print(f"\n  {'Checkpoint':>12} {'M3.splits RMSE':>15} {'Beats M2.2?':>12}")
    print("  " + "-" * 42)
    crossover_found = False
    for _, row in splits_results.iterrows():
        beats = row['rmse'] < m2_cond_rmse
        marker = " <-- CROSSOVER" if beats and not crossover_found else ""
        if beats:
            crossover_found = True
        print(f"  {row['checkpoint']:>12} {row['rmse']:>13.0f}s {'YES' if beats else 'no':>12}{marker}")

    if crossover_found:
        print(f"\n  Insight: At early checkpoints, knowing WHO the runner is (RQ2 BLUP)")
        print(f"  is more informative than knowing HOW FAST they started (RQ3 splits).")
        print(f"  The crossover occurs when cumulative split data becomes rich enough")
        print(f"  to outperform the runner's personalized baseline.")


def main():
    print("RQ3: PROGRESSIVE IN-RACE PREDICTION")
    print("Boston Marathon -- Checkpoint-Based Finish Time Prediction")
    print("=" * 70)

    train, test, has_blups = load_data()
    results_df = run_progressive_prediction(train, test, has_blups)
    print_convergence_table(results_df)
    print_r2_convergence(results_df)
    feature_importance_at_checkpoints(train, test)
    crossover_analysis(results_df)

    print("\n" + "=" * 70)
    best_5k = results_df[(results_df['checkpoint'] == '5K') &
                          (results_df['model'] == 'M3.splits')]['rmse'].values[0]
    best_half = results_df[(results_df['checkpoint'] == 'HALF') &
                            (results_df['model'] == 'M3.splits')]['rmse'].values[0]
    best_40k = results_df[(results_df['checkpoint'] == '40K') &
                           (results_df['model'] == 'M3.splits')]['rmse'].values[0]
    print(f"Prediction improves from RMSE={best_5k:.0f}s at 5K "
          f"to {best_half:.0f}s at halfway to {best_40k:.0f}s at 40K")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Progressive In-Race Checkpoint Prediction (Research Question 3).

At each checkpoint (5K through 40K), predicts marathon finish time using
cumulative split times, optionally augmented with demographics and individual
runner history (BLUPs = best linear unbiased predictors from the mixed-effects model).

Models at each checkpoint:
  Naive baseline     -- Constant-pace extrapolation (assumes even pacing)
  Splits-only Ridge  -- Ridge regression on all checkpoint times up to current point
  Splits+demographics Ridge -- Above + age and gender
  Full Ridge         -- Above + per-runner predicted offsets from the personalization model

Evaluation: train on 2015-2016, test on 2017.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CLEANED_PATH = Path(__file__).resolve().parent.parent / 'cleaned_data' / 'boston_marathon_cleaned.csv'
BLUP_PATH = Path(__file__).resolve().parent.parent / 'cleaned_data' / 'runner_blups_leakfree.csv'

# In-sample conditional RMSE from the mixed-effects personalization model
# (from 04_rq2_personalized_mixed_effects.py, random intercept + slope model)
PERSONALIZED_RMSE = 996.2

SPLIT_COLS = ['5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds',
              'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds']

NEEDED_COLS = ['year', 'display_name', 'age', 'gender', 'seconds'] + SPLIT_COLS

CHECKPOINT_KM = [5.0, 10.0, 15.0, 20.0, 21.0975, 25.0, 30.0, 35.0, 40.0]

MARATHON_KM = 42.195
CP_ORDER = [c.replace('_seconds', '').upper() for c in SPLIT_COLS]

# Model label constants
NAIVE = 'Naive (constant pace)'
SPLITS = 'Ridge (splits only)'
DEMO = 'Ridge (splits + demographics)'
FULL = 'Ridge (splits + demo + history)'
SPLITS_SUBSET = 'Ridge splits (history subset)'
SINGLE = 'Ridge (single checkpoint)'
DEMO_YEAR = 'Ridge (splits + demo + year)'


def load_data():
    """Load 2015-2017 split data and join per-runner predicted offsets."""
    print("STEP 1: DATA LOADING")

    df = pd.read_csv(CLEANED_PATH, low_memory=False, usecols=NEEDED_COLS)

    splits = df[df['year'].between(2015, 2017)].copy()
    splits = splits[splits[SPLIT_COLS].notna().all(axis=1) &
                    splits['age'].notna()].copy()

    splits['female'] = (splits['gender'] == 'F').astype(int)
    splits['year_c'] = splits['year'] - 2016

    print(f"  Runners with complete splits: {len(splits):,}")

    blups = pd.read_csv(BLUP_PATH)
    splits = splits.merge(blups, on='display_name', how='left')
    n_with_blup = splits['blup_intercept'].notna().sum()
    print(f"  Runners with leak-free history offsets: {n_with_blup:,} ({n_with_blup/len(splits)*100:.1f}%)")

    train = splits[splits['year'].isin([2015, 2016])].copy()
    test = splits[splits['year'] == 2017].copy()
    print(f"  Train (2015-2016): {len(train):,}")
    print(f"  Test  (2017):      {len(test):,}")

    return train, test


def evaluate(y_true, y_pred):
    """Compute root mean squared error, mean absolute error, and R-squared."""
    res = y_true - y_pred
    rmse = np.sqrt(np.mean(res ** 2))
    mae = np.mean(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return rmse, mae, r2


def run_progressive_prediction(train, test):
    """Run all model variants at each of 9 checkpoints."""
    print("\nSTEP 2: PROGRESSIVE CHECKPOINT PREDICTION")

    y_train = train['seconds'].values
    y_test = test['seconds'].values
    demo_cols = ['age', 'female']
    blup_cols = ['blup_intercept', 'blup_slope']

    # Subset of runners with individual history offsets
    train_blup = train[train['blup_intercept'].notna()].copy()
    test_blup = test[test['blup_intercept'].notna()].copy()

    results = []
    for i, cp in enumerate(SPLIT_COLS):
        cumul_splits = SPLIT_COLS[:i + 1]
        km = CHECKPOINT_KM[i]
        cp_label = cp.replace('_seconds', '').upper()

        # Naive baseline: assume runner maintains current average pace to the finish
        naive_pred = test[cp].values * (MARATHON_KM / km)
        rmse_n, mae_n, r2_n = evaluate(y_test, naive_pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': NAIVE,
                        'rmse': rmse_n, 'mae': mae_n, 'r2': r2_n, 'n_test': len(test)})

        # Splits-only Ridge: predict from all checkpoint times up to this point
        m = Ridge(alpha=1.0).fit(train[cumul_splits].values, y_train)
        pred = m.predict(test[cumul_splits].values)
        rmse_s, mae_s, r2_s = evaluate(y_test, pred)
        # 90% prediction interval width from training residuals (percentile-based)
        train_pred = m.predict(train[cumul_splits].values)
        train_resid = y_train - train_pred
        pi_width = np.percentile(train_resid, 95) - np.percentile(train_resid, 5)
        results.append({'checkpoint': cp_label, 'km': km, 'model': SPLITS,
                        'rmse': rmse_s, 'mae': mae_s, 'r2': r2_s, 'n_test': len(test),
                        'pi_width': pi_width})

        # Splits + demographics: add age and gender to the split features
        # (year is excluded because test year extrapolation degrades predictions)
        demo_feats = cumul_splits + demo_cols
        m = Ridge(alpha=1.0).fit(train[demo_feats].values, y_train)
        pred = m.predict(test[demo_feats].values)
        rmse_d, mae_d, r2_d = evaluate(y_test, pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': DEMO,
                        'rmse': rmse_d, 'mae': mae_d, 'r2': r2_d, 'n_test': len(test)})

        # Single-checkpoint Ridge: only the current checkpoint time (no prior splits)
        m = Ridge(alpha=1.0).fit(train[[cp]].values, y_train)
        pred = m.predict(test[[cp]].values)
        rmse_single, mae_single, r2_single = evaluate(y_test, pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': SINGLE,
                        'rmse': rmse_single, 'mae': mae_single, 'r2': r2_single,
                        'n_test': len(test)})

        # Splits + demographics + year: measures year-coefficient degradation
        demo_year_feats = cumul_splits + demo_cols + ['year_c']
        m = Ridge(alpha=1.0).fit(train[demo_year_feats].values, y_train)
        pred = m.predict(test[demo_year_feats].values)
        rmse_dy, mae_dy, r2_dy = evaluate(y_test, pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': DEMO_YEAR,
                        'rmse': rmse_dy, 'mae': mae_dy, 'r2': r2_dy,
                        'n_test': len(test)})

        # History-augmented models (only for runners with individual offsets)
        full_feats = cumul_splits + demo_cols + blup_cols
        y_train_b = train_blup['seconds'].values
        y_test_b = test_blup['seconds'].values

        # Full model: splits + demographics + per-runner history offsets
        m = Ridge(alpha=1.0).fit(train_blup[full_feats].values, y_train_b)
        pred = m.predict(test_blup[full_feats].values)
        rmse_f, mae_f, r2_f = evaluate(y_test_b, pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': FULL,
                        'rmse': rmse_f, 'mae': mae_f, 'r2': r2_f,
                        'n_test': len(test_blup)})

        # Splits-only on same subset for fair comparison
        m = Ridge(alpha=1.0).fit(train_blup[cumul_splits].values, y_train_b)
        pred = m.predict(test_blup[cumul_splits].values)
        rmse_sb, mae_sb, r2_sb = evaluate(y_test_b, pred)
        results.append({'checkpoint': cp_label, 'km': km, 'model': SPLITS_SUBSET,
                        'rmse': rmse_sb, 'mae': mae_sb, 'r2': r2_sb,
                        'n_test': len(test_blup)})

    return pd.DataFrame(results)


def print_convergence_table(results_df):
    """Print the prediction convergence curve as a table."""
    cp_order = CP_ORDER

    # Core models on full test set
    print("\n--- Prediction Convergence (test set 2017, all runners) ---")
    main = results_df[results_df['model'].isin([NAIVE, SPLITS, DEMO])]
    pivot = main.pivot(index='checkpoint', columns='model', values='rmse').reindex(cp_order)

    hdr = ['Naive', 'Splits', 'Splits+Demo']
    print(f"\n{'Checkpoint':>12} {hdr[0]:>12} {hdr[1]:>12} {hdr[2]:>12}")
    print(f"{'':>12} {'RMSE (s)':>12} {'RMSE (s)':>12} {'RMSE (s)':>12}")
    print("-" * 52)
    for cp in cp_order:
        row = pivot.loc[cp]
        print(f"  {cp:>10} {row.get(NAIVE, 0):>10.0f} "
              f"{row.get(SPLITS, 0):>10.0f} {row.get(DEMO, 0):>10.0f}")

    # History-augmented comparison
    blup_results = results_df[results_df['model'].isin([FULL, SPLITS_SUBSET])]
    print(f"\n--- History Augmentation (n={blup_results['n_test'].iloc[0]:,} known runners) ---")
    print(f"{'Checkpoint':>12} {'Splits':>12} {'+ History':>12} {'Gain':>12}")
    print(f"{'':>12} {'RMSE (s)':>12} {'RMSE (s)':>12} {'(seconds)':>12}")
    print("-" * 52)
    for cp in cp_order:
        s_row = blup_results[(blup_results['checkpoint'] == cp) &
                             (blup_results['model'] == SPLITS_SUBSET)]
        f_row = blup_results[(blup_results['checkpoint'] == cp) &
                             (blup_results['model'] == FULL)]
        s_rmse = s_row['rmse'].values[0]
        f_rmse = f_row['rmse'].values[0]
        print(f"  {cp:>10} {s_rmse:>10.0f} {f_rmse:>10.0f} {s_rmse - f_rmse:>+10.0f}")


def print_r2_convergence(results_df):
    """Print R-squared convergence for all models."""
    print("\n--- R-squared Convergence ---")
    cp_order = CP_ORDER

    main = results_df[results_df['model'].isin([NAIVE, SPLITS, DEMO])]
    pivot = main.pivot(index='checkpoint', columns='model', values='r2').reindex(cp_order)

    print(f"\n{'Checkpoint':>12} {'Naive':>10} {'Splits':>10} {'Splits+Demo':>10}")
    print("-" * 46)
    for cp in cp_order:
        row = pivot.loc[cp]
        print(f"  {cp:>10} {row.get(NAIVE, 0):>9.4f} "
              f"{row.get(SPLITS, 0):>9.4f} {row.get(DEMO, 0):>9.4f}")


def feature_importance_at_checkpoints(train):
    """Show which features dominate at early vs late checkpoints via standardized coefficients."""
    print("\nSTEP 3: FEATURE IMPORTANCE AT SELECTED CHECKPOINTS (standardized)")

    demo_cols = ['age', 'female']
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


def print_pi_convergence(results_df):
    """Print 90% prediction interval width convergence for splits-only Ridge."""
    print("\n--- 90% Prediction Interval Width Convergence (splits-only Ridge) ---")
    cp_order = CP_ORDER
    splits_df = results_df[results_df['model'] == SPLITS].set_index('checkpoint')

    print(f"\n{'Checkpoint':>12} {'RMSE (s)':>10} {'90% PI (s)':>12} {'90% PI (min)':>13}")
    print("-" * 50)
    for cp in cp_order:
        rmse = splits_df.loc[cp, 'rmse']
        pi = splits_df.loc[cp, 'pi_width']
        print(f"  {cp:>10} {rmse:>8.0f} {pi:>10.0f} {pi/60:>11.1f}")


def cumulative_vs_single_analysis(results_df):
    """Compare cumulative-splits Ridge vs single-checkpoint Ridge at each checkpoint."""
    print("\nSTEP 4b: CUMULATIVE vs SINGLE-CHECKPOINT COMPARISON")
    cp_order = CP_ORDER

    cumul = results_df[results_df['model'] == SPLITS].set_index('checkpoint')
    single = results_df[results_df['model'] == SINGLE].set_index('checkpoint')

    print(f"\n  {'Checkpoint':>12} {'Cumulative':>12} {'Single':>12} {'Advantage':>12}")
    print(f"  {'':>12} {'RMSE (s)':>12} {'RMSE (s)':>12} {'(seconds)':>12}")
    print("  " + "-" * 52)
    for cp in cp_order:
        c_rmse = cumul.loc[cp, 'rmse']
        s_rmse = single.loc[cp, 'rmse']
        diff = s_rmse - c_rmse
        print(f"  {cp:>12} {c_rmse:>10.0f} {s_rmse:>10.0f} {diff:>+10.0f}")

    # Report the mid-race range (10K-30K) for the proposal claim
    mid_cps = ['10K', '15K', '20K', 'HALF', '25K', '30K']
    mid_diffs = [single.loc[cp, 'rmse'] - cumul.loc[cp, 'rmse'] for cp in mid_cps]
    print(f"\n  Mid-race (10K-30K) advantage range: {min(mid_diffs):.0f} to {max(mid_diffs):.0f} seconds")


def year_degradation_analysis(results_df):
    """Compare demo (no year) vs demo+year to quantify year-coefficient degradation."""
    print("\nSTEP 4c: YEAR-COEFFICIENT DEGRADATION")
    cp_order = CP_ORDER

    demo = results_df[results_df['model'] == DEMO].set_index('checkpoint')
    demo_year = results_df[results_df['model'] == DEMO_YEAR].set_index('checkpoint')

    print(f"\n  {'Checkpoint':>12} {'No Year':>12} {'With Year':>12} {'Degradation':>12}")
    print(f"  {'':>12} {'RMSE (s)':>12} {'RMSE (s)':>12} {'(seconds)':>12}")
    print("  " + "-" * 52)
    max_degrad = 0
    for cp in cp_order:
        d_rmse = demo.loc[cp, 'rmse']
        dy_rmse = demo_year.loc[cp, 'rmse']
        degrad = dy_rmse - d_rmse
        max_degrad = max(max_degrad, degrad)
        print(f"  {cp:>12} {d_rmse:>10.0f} {dy_rmse:>10.0f} {degrad:>+10.0f}")
    print(f"\n  Maximum degradation from including year: {max_degrad:.0f} seconds")


def crossover_analysis(results_df):
    """Identify where splits-only prediction beats the pre-race personalized prediction."""
    print("\nSTEP 4: CROSSOVER ANALYSIS")

    personalized_rmse = PERSONALIZED_RMSE
    print(f"\n  Pre-race personalized RMSE (known runner, in-sample): {personalized_rmse:.0f}s")

    splits_results = results_df[results_df['model'] == SPLITS].sort_values('km')
    print(f"\n  {'Checkpoint':>12} {'Splits RMSE':>15} {'Beats personalized?':>20}")
    print("  " + "-" * 50)
    crossover_found = False
    for _, row in splits_results.iterrows():
        beats = row['rmse'] < personalized_rmse
        marker = " <-- CROSSOVER" if beats and not crossover_found else ""
        if beats:
            crossover_found = True
        print(f"  {row['checkpoint']:>12} {row['rmse']:>13.0f}s {'YES' if beats else 'no':>20}{marker}")

    print(f"\n  Insight: At early checkpoints, knowing WHO the runner is (from their")
    print(f"  race history) is more informative than knowing HOW FAST they started.")
    print(f"  The crossover occurs when cumulative split data becomes rich enough")
    print(f"  to outperform the runner's personalized baseline.")


def main():
    print("PROGRESSIVE IN-RACE PREDICTION")
    print("Boston Marathon -- Finish Time Prediction from Checkpoint Splits")
    print("=" * 70)

    train, test = load_data()
    results_df = run_progressive_prediction(train, test)
    print_convergence_table(results_df)
    print_r2_convergence(results_df)
    feature_importance_at_checkpoints(train)
    crossover_analysis(results_df)
    cumulative_vs_single_analysis(results_df)
    year_degradation_analysis(results_df)
    print_pi_convergence(results_df)

    print("\n" + "=" * 70)
    splits_idx = results_df[results_df['model'] == SPLITS].set_index('checkpoint')['rmse']
    print(f"Prediction improves from RMSE={splits_idx['5K']:.0f}s at 5K "
          f"to {splits_idx['HALF']:.0f}s at halfway to {splits_idx['40K']:.0f}s at 40K")


if __name__ == '__main__':
    main()

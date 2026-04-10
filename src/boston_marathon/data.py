"""Data loading and sample construction for all three research questions.

Sits between scripts/clean_data.py (raw CSV) and rq1/rq2/rq3 (analysis). Handles
Parquet caching, age filtering, feature centering, prior history, repeat-runner
sample construction, and BLUP merging for RQ3. Paths from config.py.
"""
import numpy as np
import pandas as pd
from boston_marathon import config as cfg

_PARQUET = cfg.CLEANED_CSV.with_suffix('.parquet')


def load_cleaned(usecols=None):
    """Load the 615K-row cleaned dataset. Caches as Parquet on first call for fast reloads."""
    if _PARQUET.exists() and _PARQUET.stat().st_mtime >= cfg.CLEANED_CSV.stat().st_mtime:
        return pd.read_parquet(_PARQUET, engine='pyarrow', columns=usecols)
    df = pd.read_csv(cfg.CLEANED_CSV, dtype={'gender': 'category', 'display_name': str})
    df.to_parquet(_PARQUET, engine='pyarrow', compression='snappy')
    return df[usecols] if usecols is not None else df


def filter_non_imputed(df):
    """Keep non-imputed, age-known, year >= 2000 rows (used by RQ1 and RQ2)."""
    return df[~df['age_imputed'] & df['age'].notna() & (df['year'] >= 2000)].copy()


def add_centered_features(df, age_mean, year_center=2010):
    """Add age_c, age_c2, year_c, female, age_c_female in place. age_mean from train set."""
    df['age_c'] = df['age'] - age_mean
    df['age_c2'] = df['age_c'] ** 2
    df['year_c'] = df['year'] - year_center
    df['female'] = (df['gender'] == 'F').astype(int)
    df['age_c_female'] = df['age_c'] * df['female']


def add_prior_history(df):
    """Expanding mean of prior finish times + appearance count (one-year shift, no leakage)."""
    df = df.sort_values(['display_name', 'year'])
    df['prior_mean_time'] = df.groupby('display_name')['seconds'].transform(
        lambda x: x.expanding().mean().shift(1))
    df['prior_appearances'] = df.groupby('display_name').cumcount()
    return df


def build_repeat_runner_sample(df):
    """RQ2 sample: non-imputed, repeat, age-consistent, 2000+, 2+ obs (~188K rows, ~66K runners)."""
    df = df[~df['age_imputed'] & df['age'].notna()].copy()
    df = df[df.groupby('display_name')['display_name'].transform('size') > 1].copy()
    df.sort_values(['display_name', 'year'], inplace=True)
    same = df['display_name'].values[1:] == df['display_name'].values[:-1]
    bad = set(df['display_name'].values[1:][same & (np.abs(np.diff(df['age'].values) - np.diff(df['year'].values)) > 8)])
    df = df[~df['display_name'].isin(bad)].copy()
    df = df[df['year'] >= 2000].copy()
    df = df[df.groupby('display_name')['display_name'].transform('size') > 1].copy()
    return df


def temporal_split(df, train_years, test_years):
    """Split a dataframe by year into (train, test). Boundaries are defined in config.py."""
    return df[df['year'].isin(train_years)].copy(), df[df['year'].isin(test_years)].copy()


def load_blups():
    """Load leak-free per-runner predictions (trained pre-2017) for safe use in RQ3's 2017 test set."""
    return pd.read_csv(cfg.BLUP_LEAKFREE_CSV, engine='pyarrow')


def load_splits_with_blups():
    """RQ3 dataset: 2015-2017 complete splits merged with leak-free per-runner predictions. Returns (train, test)."""
    needed = ['year', 'display_name', 'age', 'gender', 'seconds'] + cfg.SPLIT_COLS
    df = load_cleaned(usecols=needed)
    splits = df[df['year'].between(2015, 2017)].copy()
    splits = splits[splits[cfg.SPLIT_COLS].notna().all(axis=1) & splits['age'].notna()].copy()
    splits['female'] = (splits['gender'] == 'F').astype(int)
    splits['year_c'] = splits['year'] - 2016
    splits = splits.merge(load_blups(), on='display_name', how='left')
    return (splits[splits['year'].isin(cfg.RQ3_TRAIN_YEARS)].copy(),
            splits[splits['year'] == cfg.RQ3_TEST_YEAR].copy())

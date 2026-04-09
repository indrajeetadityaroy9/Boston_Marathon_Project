"""Load and prepare the cleaned Boston Marathon dataset for modeling.

Reads from cleaned_data/boston_marathon_cleaned.csv (produced by data_scripts/01).
Creates a Parquet cache on first load for faster repeat access. Provides building-block
functions used by run_pipeline.py to construct the RQ1, RQ2, and RQ3 modeling samples.
"""
import numpy as np
import pandas as pd
from src import config as cfg

_PARQUET = cfg.CLEANED_CSV.with_suffix('.parquet')


def load_cleaned(usecols=None):
    """Load the cleaned dataset. First call caches as Parquet; subsequent calls read the cache."""
    if _PARQUET.exists() and _PARQUET.stat().st_mtime >= cfg.CLEANED_CSV.stat().st_mtime:
        return pd.read_parquet(_PARQUET, engine='pyarrow', columns=usecols)
    df = pd.read_csv(cfg.CLEANED_CSV, engine='pyarrow', dtype={'gender': 'category', 'display_name': str})
    df.to_parquet(_PARQUET, engine='pyarrow', compression='snappy')
    return df[usecols] if usecols is not None else df


def filter_non_imputed(df):
    """Keep only non-imputed, age-known, year-2000+ rows. Used by RQ1 and as input to RQ2."""
    return df[~df['age_imputed'] & df['age'].notna() & (df['year'] >= 2000)].copy()


def add_centered_features(df, age_mean, year_center=2010):
    """Add centered demographic features in-place. age_mean must come from the training set."""
    df['age_c'] = df['age'] - age_mean
    df['age_c2'] = df['age_c'] ** 2
    df['year_c'] = df['year'] - year_center
    df['female'] = (df['gender'] == 'F').astype(int)
    df['age_c_female'] = df['age_c'] * df['female']


def add_prior_history(df):
    """Add each runner's historical mean finish time and race count, leak-free via one-year shift.
    Used by RQ1's history-augmented model (M1.2)."""
    df = df.sort_values(['display_name', 'year'])
    df['prior_mean_time'] = df.groupby('display_name')['seconds'].transform(
        lambda x: x.expanding().mean().shift(1))
    df['prior_appearances'] = df.groupby('display_name').cumcount()
    return df


def build_repeat_runner_sample(df):
    """Build the RQ2 modeling sample: repeat runners with consistent age progressions.
    Applies a six-step filter chain (non-imputed, repeat, age-gap check, year cutoff)."""
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
    """Split by year into (train, test). Year boundaries are in config.py per RQ."""
    return df[df['year'].isin(train_years)].copy(), df[df['year'].isin(test_years)].copy()


def load_blups():
    """Load per-runner random effects exported by RQ2's temporal hold-out (trained pre-2017)."""
    return pd.read_csv(cfg.BLUP_LEAKFREE_CSV, engine='pyarrow')


def load_splits_with_blups():
    """Prepare the RQ3 dataset: 2015-2017 checkpoint splits merged with RQ2's leak-free BLUPs."""
    needed = ['year', 'display_name', 'age', 'gender', 'seconds'] + cfg.SPLIT_COLS
    df = load_cleaned(usecols=needed)
    splits = df[df['year'].between(2015, 2017)].copy()
    splits = splits[splits[cfg.SPLIT_COLS].notna().all(axis=1) & splits['age'].notna()].copy()
    splits['female'] = (splits['gender'] == 'F').astype(int)
    splits['year_c'] = splits['year'] - 2016
    splits = splits.merge(load_blups(), on='display_name', how='left')
    return (splits[splits['year'].isin(cfg.RQ3_TRAIN_YEARS)].copy(),
            splits[splits['year'] == cfg.RQ3_TEST_YEAR].copy())

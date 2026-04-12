"""Load the cleaned CSV and build the derived samples used by the pipeline."""
import numpy as np
import pandas as pd

from . import config


def load_processed_results(columns=None):
    """Read `boston_marathon_cleaned.csv` with pyarrow."""
    return pd.read_csv(config.PROCESSED_RESULTS_CSV, engine='pyarrow', usecols=columns)


def add_centered_pre_race_features(df, age_mean):
    """Add the fixed-effect columns used by the regression models in place."""
    df['age_centered'] = df['age'] - age_mean
    df['age_centered_squared'] = df['age_centered'] ** 2
    df['year_centered'] = df['year'] - config.YEAR_CENTER
    df['female'] = (df['gender'] == 'F').astype(int)
    df['age_centered_female_interaction'] = df['age_centered'] * df['female']
    df['age_centered_squared_female_interaction'] = df['age_centered_squared'] * df['female']


def add_prior_boston_history_features(df):
    """Add leak-free prior-history features and return the sorted frame."""
    df = df.sort_values(['display_name', 'year'])
    grp = df.groupby('display_name')
    cum_seconds = grp['seconds'].cumsum()
    cum_count = grp.cumcount() + 1
    df['prior_mean_time'] = cum_seconds.groupby(df['display_name']).shift() / (cum_count - 1)
    df['prior_appearances'] = cum_count - 1
    df['log1p_prior_appearances'] = np.log1p(df['prior_appearances'].to_numpy())
    return df


def build_repeat_runner_analysis_sample(df):
    """Build the repeat-runner sample used by the mixed-effects stage."""
    df = df[~df['age_imputed'] & df['age'].notna() & (df['year'] >= 2000)]
    df = df[df.groupby('display_name')['display_name'].transform('size') > 1]
    sorted_df = df.sort_values(['display_name', 'year'])
    names = sorted_df['display_name'].to_numpy()
    mismatch = np.abs(np.diff(sorted_df['age'].to_numpy()) - np.diff(sorted_df['year'].to_numpy()))
    colliding = set(names[1:][(names[1:] == names[:-1]) & (mismatch > 8)])
    df = df[~df['display_name'].isin(colliding)]
    return df[df.groupby('display_name')['display_name'].transform('size') > 1].copy()


def load_in_race_split_dataset(re_df):
    """Load the split-time dataset, merge runner effects, and return (train, test)."""
    df = load_processed_results(['year', 'display_name', 'age', 'gender', 'seconds'] + config.CUMULATIVE_SPLIT_TIME_COLUMNS)
    df = (
        df[df[config.CUMULATIVE_SPLIT_TIME_COLUMNS].notna().all(axis=1)]
        .assign(female=lambda d: (d['gender'] == 'F').astype(int), year_centered=lambda d: d['year'] - config.YEAR_CENTER)
        .merge(re_df.reset_index(), on='display_name', how='left')
    )
    return (
        df[df['year'].isin(config.IN_RACE_SPLIT_PREDICTION_TRAIN_YEARS)].copy(),
        df[df['year'] == config.IN_RACE_SPLIT_PREDICTION_TEST_YEAR].copy(),
    )

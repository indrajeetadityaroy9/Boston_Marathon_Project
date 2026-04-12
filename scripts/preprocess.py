import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required side-effect)
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_RESULTS_DIR = PROJECT_ROOT / 'data' / 'processed'

RAW_SPLIT_TIME_COLUMNS = ['5k', '10k', '15k', '20k', 'half', '25k', '30k', '35k', '40k']
STANDARDIZED_RESULT_COLUMNS = [
    'year', 'display_name', 'first_name', 'last_name', 'age', 'gender',
    'residence', 'city', 'state', 'country_residence', 'country_citizenship',
    'bib', 'pace', 'official_time', 'seconds', 'overall', 'gender_result',
    'division_result', *RAW_SPLIT_TIME_COLUMNS, 'pace_seconds_per_mile',
]
REDUNDANT_TIME_STRING_COLUMNS = ['official_time', 'pace'] + RAW_SPLIT_TIME_COLUMNS
RAW_MISSING_VALUE_MARKERS = ['NULL', '', ' ', 'null', 'N/A', 'NA', 'n/a', 'na', '-']
MARATHON_DISTANCE_MILES = 26.2188


def parse_time_strings_to_seconds(time_series):
    """Parse `H:MM:SS` and `MM:SS` strings into total seconds."""
    vals = pd.Series(time_series, copy=False).astype('string').str.strip()
    vals = vals.mask(vals.eq(''), pd.NA)
    mm_ss = vals.str.fullmatch(r'\d{1,2}:\d{2}')
    return pd.to_timedelta(vals.mask(mm_ss, '00:' + vals), errors='coerce').dt.total_seconds()


def load_and_standardize_yearly_results(file_path, year):
    """Read one yearly CSV and map it to the standard cleaned-data schema."""
    df = pd.read_csv(file_path, na_values=RAW_MISSING_VALUE_MARKERS, dtype='string', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.strip('"')
    df['year'] = year

    if year >= 2015:
        df = df.rename(columns={'contry_citizenship': 'country_citizenship'})
        parts = df[['city', 'state', 'country_residence']].astype('string').apply(lambda c: c.str.strip()).replace('', pd.NA)
        df['residence'] = parts.apply(lambda row: ', '.join(row.dropna()), axis=1).replace('', pd.NA)

    df = df.reindex(columns=STANDARDIZED_RESULT_COLUMNS, fill_value=pd.NA)
    text_cols = [c for c in STANDARDIZED_RESULT_COLUMNS if c != 'year']
    df[text_cols] = df[text_cols].astype('string')
    return df


def convert_results_to_analysis_types(df):
    """Strip text fields, parse numerics, and convert time strings to seconds."""
    str_cols = df.select_dtypes(include='string').columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip()).replace(r'^\s*$', pd.NA, regex=True)

    for col in ('overall', 'gender_result', 'division_result'):
        df[col] = pd.to_numeric(df[col])
    df['seconds'] = pd.to_numeric(df['seconds']).astype('Float64')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    df['_official_time_seconds'] = parse_time_strings_to_seconds(df['official_time'])
    df['pace_seconds_per_mile'] = parse_time_strings_to_seconds(df['pace'])
    df = df.assign(**{f'{s}_seconds': parse_time_strings_to_seconds(df[s]) for s in RAW_SPLIT_TIME_COLUMNS})
    return df


def impute_age_mice(df, *, random_state=42):
    """Impute missing ages with per-year MICE (deterministic, single pass)."""
    df['age_imputed'] = False
    age_by_year = df.groupby('year')['age']
    years = [y for y, g in age_by_year if g.isna().any() and g.notna().any()]
    if not years:
        print("  No missing ages to impute.")
        return df

    print(f"  Imputing {sum(age_by_year.get_group(y).isna().sum() for y in years):,} missing age values via per-year MICE...")

    for year in years:
        mask = df['year'] == year
        sub = df.loc[mask & df[['seconds', 'gender']].notna().all(axis=1), ['age', 'seconds', 'gender']].copy()
        missing_idx = sub.index[sub['age'].isna()]

        pre = ColumnTransformer([
            ('age', 'passthrough', ['age']),
            ('num', StandardScaler(), ['seconds']),
            ('gender', OrdinalEncoder(categories=[['M', 'F']], dtype=float), ['gender']),
        ], verbose_feature_names_out=False).set_output(transform='pandas')
        pipe = Pipeline([('pre', pre), ('imp', IterativeImputer(sample_posterior=False, random_state=random_state))]).set_output(transform='pandas')
        imputed = pipe.fit_transform(sub)['age'].clip(lower=14, upper=90)

        df.loc[missing_idx, 'age'] = imputed.loc[missing_idx].round().astype(int)
        df.loc[missing_idx, 'age_imputed'] = True
        print(f"    {year}: n_imputed={len(missing_idx):,}")
    return df


def run_data_cleaning():
    print("Boston Marathon Data Cleaning")

    manifest = [
        (fp, int(m.group(1)))
        for fp in sorted(RAW_RESULTS_DIR.glob('results*.csv'))
        if '_includes-wheelchair' not in fp.name and '_without-diverted' not in fp.name
        and (m := re.search(r'results(\d{4})', fp.stem))
    ]
    print(f"  Loading {len(manifest)} raw CSVs ({manifest[0][1]}-{manifest[-1][1]})...")
    df = pd.concat([load_and_standardize_yearly_results(fp, yr) for fp, yr in manifest], ignore_index=True)
    print(f"  Loaded: {len(df):,} rows")

    df = convert_results_to_analysis_types(df)

    df['seconds'] = df['seconds'].astype('Float64').combine_first(df['_official_time_seconds'].astype('Float64'))
    df = df.drop(columns=['_official_time_seconds'])
    df['pace_seconds_per_mile'] = df['pace_seconds_per_mile'].fillna(df['seconds'] / MARATHON_DISTANCE_MILES)

    n_before = len(df)
    df['gender'] = df['gender'].astype('string').str.strip().str.upper()
    df = df[df['gender'].isin(['M', 'F'])].copy()
    print(f"  Gender filter (keep M/F): {n_before:,} -> {len(df):,} ({n_before - len(df):,} dropped)")

    df['age'] = df['age'].where(df['age'].between(14, 90))
    df = impute_age_mice(df)

    n_before = len(df)
    df = df[~(df['seconds'] <= 0)].drop_duplicates(subset=['year', 'display_name', 'seconds'])
    print(f"  Non-positive seconds + duplicate filter: {n_before:,} -> {len(df):,} ({n_before - len(df):,} dropped)")

    df['gender'] = df['gender'].astype('category')
    df['name_collides_within_year'] = df.duplicated(subset=['year', 'display_name'], keep=False)
    n_collisions = int(df['name_collides_within_year'].sum())
    n_groups = df[df['name_collides_within_year']].groupby(['year', 'display_name']).ngroups
    print(f"  Within-year name collision flag: {n_collisions:,} rows in {n_groups:,} distinct (year, display_name) groups")

    df = df.drop(columns=list(REDUNDANT_TIME_STRING_COLUMNS))
    print(f"  Dropped redundant string columns: {REDUNDANT_TIME_STRING_COLUMNS}")

    PROCESSED_RESULTS_DIR.mkdir(exist_ok=True)
    out = PROCESSED_RESULTS_DIR / 'boston_marathon_cleaned.csv'
    df.to_csv(out, index=False)
    print(f"  Saved cleaned data to {out} ({len(df):,} rows, {len(df.columns)} columns, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    run_data_cleaning()

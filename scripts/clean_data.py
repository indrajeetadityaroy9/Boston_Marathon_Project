import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed'

SPLIT_COLS = ['5k', '10k', '15k', '20k', 'half', '25k', '30k', '35k', '40k']

UNIFIED_COLS = [
    'year', 'display_name', 'first_name', 'last_name', 'age', 'gender',
    'residence', 'city', 'state', 'country_residence', 'country_citizenship',
    'bib', 'pace', 'official_time', 'seconds',
    'overall', 'gender_result', 'division_result',
    '5k', '10k', '15k', '20k', 'half', '25k', '30k', '35k', '40k',
    'projected_time', 'pace_seconds_per_mile',
]

NA_VALUES = ['NULL', '', ' ', 'null', 'N/A', 'NA', 'n/a', 'na', '-']

MARATHON_MILES = 26.2188


def parse_time_to_seconds(series):
    """Convert MM:SS and H:MM:SS strings to total seconds with vectorized pandas parsing."""
    values = pd.Series(series, copy=False).astype('string').str.strip()
    values = values.mask(values.eq(''), pd.NA)
    mm_ss = values.str.fullmatch(r'\d{1,2}:\d{2}')
    normalized = values.mask(mm_ss, '00:' + values)
    td = pd.to_timedelta(normalized, errors='coerce')
    return td.dt.total_seconds()


def load_and_unify(filepath, year):
    """Load a single CSV and unify to the standard schema."""
    filepath = Path(filepath)
    with open(filepath, 'r', errors='replace') as f:
        total_lines = sum(1 for _ in f) - 1
    df = pd.read_csv(filepath, na_values=NA_VALUES, dtype='string', on_bad_lines='skip')
    skipped = total_lines - len(df)
    if skipped > 0:
        print(f"  WARNING: {filepath.name}: skipped {skipped} malformed rows")
    df.columns = df.columns.str.strip().str.strip('"')
    df['year'] = year

    if 'place_overall' in df.columns:
        if 'contry_citizenship' in df.columns:
            df = df.rename(columns={'contry_citizenship': 'country_citizenship'})
        address_parts = (
            df[['city', 'state', 'country_residence']]
            .astype('string')
            .apply(lambda col: col.str.strip())
            .replace('', pd.NA)
        )
        df['residence'] = address_parts.stack(future_stack=True).dropna().groupby(level=0).agg(', '.join).reindex(df.index)
        if 'name' in df.columns and 'display_name' not in df.columns:
            df = df.rename(columns={'name': 'display_name'})

    df = df.reindex(columns=UNIFIED_COLS, fill_value=pd.NA)
    text_like_cols = [col for col in UNIFIED_COLS if col != 'year']
    df[text_like_cols] = df[text_like_cols].astype('string')

    return df


def clean_types(df):
    """Cast numeric columns and parse time strings."""
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    df[str_cols] = df[str_cols].astype('string').apply(lambda c: c.str.strip())
    df[str_cols] = df[str_cols].replace(r'^\s*$', pd.NA, regex=True)

    numeric_cast_cols = ['age', 'overall', 'gender_result', 'division_result', 'bib']
    df[numeric_cast_cols] = df[numeric_cast_cols].apply(pd.to_numeric, errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce').astype('Float64')

    df['_official_time_seconds'] = parse_time_to_seconds(df['official_time'])
    df['pace_seconds_per_mile'] = parse_time_to_seconds(df['pace'])
    df = df.assign(**{f'{c}_seconds': parse_time_to_seconds(df[c]) for c in SPLIT_COLS})

    return df


def impute_age(df, *, n_neighbors=5, min_validity=0.50):
    """Impute missing age via KNNImputer for years with sufficient valid age data."""
    df['age_imputed'] = False

    age_grp = df.groupby('year')['age']
    age_validity = age_grp.count() / age_grp.size()
    imputable_years = age_validity[age_validity >= min_validity].index.tolist()

    mask_imputable = df['year'].isin(imputable_years)
    sub = df.loc[mask_imputable].copy()

    n_to_impute = sub['age'].isna().sum()
    print(f"  Imputing {n_to_impute} missing age values via KNNImputer...")

    feature_cols = ['age', 'year', 'seconds', 'gender']
    context_features = ['year', 'seconds', 'gender']

    valid_for_impute = sub[context_features].notna().all(axis=1)
    sub_valid = sub.loc[valid_for_impute, feature_cols].copy()
    was_null = sub_valid['age'].isna()

    preprocess = ColumnTransformer(transformers=[
        ('age', 'passthrough', ['age']),
        ('num', StandardScaler(), ['year', 'seconds']),
        ('gender', OrdinalEncoder(categories=[['M', 'F']], dtype=float,
                                  handle_unknown='use_encoded_value', unknown_value=np.nan), ['gender']),
    ], verbose_feature_names_out=False).set_output(transform='pandas')
    pipe = Pipeline([('pre', preprocess), ('imp', KNNImputer(n_neighbors=n_neighbors))
                     ]).set_output(transform='pandas')
    imputed_series = pipe.fit_transform(sub_valid)['age']

    imputed_indices = was_null[was_null].index
    df.loc[imputed_indices, 'age'] = imputed_series.loc[imputed_indices].round().astype(int)
    df.loc[imputed_indices, 'age_imputed'] = True

    return df


def main():
    print("Boston Marathon Data Cleaning")

    manifest = [
        (path, int(match.group(1)))
        for path in sorted(DATA_DIR.glob('results*.csv'))
        if '_includes-wheelchair' not in path.name
        and '_without-diverted' not in path.name
        and (match := re.search(r'results(\d{4})', path.stem))
    ]
    df = pd.concat(
        [load_and_unify(fp, yr) for fp, yr in manifest],
        ignore_index=True,
    )

    df = clean_types(df)

    df['seconds'] = df['seconds'].astype('Float64').combine_first(df['_official_time_seconds'].astype('Float64'))
    df = df.drop(columns=['_official_time_seconds'])
    df['pace_seconds_per_mile'] = df['pace_seconds_per_mile'].fillna(df['seconds'] / MARATHON_MILES)

    # Keep only rows with valid gender (M or F)
    df['gender'] = df['gender'].astype('string').str.strip().str.upper()
    valid_gender = df['gender'].isin(['M', 'F'])
    n_removed = (~valid_gender).sum()
    print(f"  Removed {n_removed} rows with invalid/missing gender values")
    df = df[valid_gender].copy()

    # Treat age=0 as missing, reject ages outside [14, 90]
    df['age'] = df['age'].replace(0, np.nan)
    invalid_age = df['age'].notna() & ~df['age'].between(14, 90)
    print(f"  Replaced {invalid_age.sum()} implausible ages (< 14 or > 90) with NaN")
    df['age'] = df['age'].where(~invalid_age)

    df = impute_age(df)

    # Final validation
    bad_seconds = df['seconds'].notna() & (df['seconds'] <= 0)
    print(f"  Removed {bad_seconds.sum()} rows with seconds <= 0")
    df = df[~bad_seconds]

    n_before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {n_before_dedup - len(df)} exact duplicate rows")

    df['gender'] = df['gender'].astype('category')

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / 'boston_marathon_cleaned.csv'
    df.to_csv(output_path, index=False)
    print(f"\n  Saved cleaned data to {output_path}")
    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")


if __name__ == '__main__':
    main()

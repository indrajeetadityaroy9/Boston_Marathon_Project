"""Build one clean CSV from the raw per-year Boston Marathon result files.

The raw files in data/raw/ use different column names and formats across years.
This script does the following:

1. Read each results*.csv file and reshape it to one standard layout
2. Convert time strings into total seconds
3. Drop rows with missing gender or invalid age
4. Impute missing ages for years with enough age data
5. Drop non-positive times and exact duplicates
6. Flag rows where different runners share the same name in the same race
7. Drop text columns that already have numeric versions
8. Run sanity checks and stop if any fail
9. Save one clean file at data/processed/boston_marathon_cleaned.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required side-effect)
from sklearn.impute import IterativeImputer
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

# Text columns whose values already exist in numeric form.
# These are dropped before saving the output.
REDUNDANT_STRING_COLS = ['official_time', 'pace', 'projected_time'] + SPLIT_COLS

NA_VALUES = ['NULL', '', ' ', 'null', 'N/A', 'NA', 'n/a', 'na', '-']

MARATHON_MILES = 26.2188


def parse_time_to_seconds(series):
    """Convert time strings into total seconds.

    Handles both `H:MM:SS` and `MM:SS` formats.
    Empty strings and invalid values become NaN.
    """
    values = pd.Series(series, copy=False).astype('string').str.strip()
    values = values.mask(values.eq(''), pd.NA)
    # Add "00:" to `MM:SS` values so pandas can parse them as HH:MM:SS.
    mm_ss = values.str.fullmatch(r'\d{1,2}:\d{2}')
    normalized = values.mask(mm_ss, '00:' + values)
    # `errors='coerce'` converts malformed legacy time strings to NaN.
    # Those rows are removed later if finish time is missing or invalid.
    td = pd.to_timedelta(normalized, errors='coerce')
    return td.dt.total_seconds()


def load_and_unify(filepath, year):
    """Read one year's raw CSV and reshape it to the standard column layout.

    Different years use different column names and formats. This function:
    - reads the file with standard missing-value tokens
    - adds the `year` column
    - fixes a known typo in newer files
    - builds `residence` from city/state/country when needed
    - keeps only `UNIFIED_COLS` and fills missing columns with NaN

    `on_bad_lines='skip'` drops rows with broken CSV formatting.
    """
    df = pd.read_csv(filepath, na_values=NA_VALUES, dtype='string', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.strip('"')
    df['year'] = year

    # Modern files have separate city/state/country columns and a known typo.
    # Older files already store residence in one column.
    if year >= 2015:
        df = df.rename(columns={'contry_citizenship': 'country_citizenship'})
        address_parts = (df[['city', 'state', 'country_residence']]
                         .astype('string')
                         .apply(lambda col: col.str.strip())
                         .replace('', pd.NA))
        df['residence'] = (address_parts.stack(future_stack=True)
                           .dropna()
                           .groupby(level=0)
                           .agg(', '.join)
                           .reindex(df.index))

    df = df.reindex(columns=UNIFIED_COLS, fill_value=pd.NA)
    text_like_cols = [col for col in UNIFIED_COLS if col != 'year']
    df[text_like_cols] = df[text_like_cols].astype('string')

    return df


def clean_types(df):
    """Convert text columns to numeric form and parse times into seconds.

    This function:
    - strips whitespace and converts blank strings to NaN
    - parses `overall`, `gender_result`, `division_result`, and `seconds`
    - parses `age` with bounded tolerance for a few invalid rows
    - keeps `bib` as a string column
    - parses `official_time` into `_official_time_seconds`
    - parses `pace` into `pace_seconds_per_mile`
    - parses each split into a `<split>_seconds` column
    """
    str_cols = df.select_dtypes(include='string').columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip()).replace(r'^\s*$', pd.NA, regex=True)

    # Parse numeric result columns. Unexpected non-numeric values should fail.
    for col in ['overall', 'gender_result', 'division_result']:
        df[col] = pd.to_numeric(df[col])
    df['seconds'] = pd.to_numeric(df['seconds']).astype('Float64')

    # Parse age and allow only a very small number of invalid age values.
    pre_age_na = int(df['age'].isna().sum())
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    unexpected_age_na = int(df['age'].isna().sum()) - pre_age_na
    assert unexpected_age_na <= 5, (
        f"Expected at most 5 non-numeric age values across all raw files, "
        f"found {unexpected_age_na}. Investigate raw data."
    )

    # Keep `bib` as string because values like 'F1' and 'F11' are valid.
    df['_official_time_seconds'] = parse_time_to_seconds(df['official_time'])
    df['pace_seconds_per_mile'] = parse_time_to_seconds(df['pace'])
    df = df.assign(**{f'{c}_seconds': parse_time_to_seconds(df[c]) for c in SPLIT_COLS})

    return df


def impute_age(df, *, random_state=42, min_validity=0.50):
    """Impute missing ages for years with enough observed age data.

    For each eligible year, fit a per-year MICE model using finish time and
    gender to predict age. Posterior sampling helps preserve the age
    distribution better than a single-point prediction.

    A per-year model avoids mixing year-to-year drift into the imputation.

    Parameters:
      random_state   seed for reproducible posterior draws
      min_validity   minimum fraction of non-missing ages required for a year
    """
    df['age_imputed'] = False

    # Fraction of rows in each year that already have age values.
    age_validity = df.groupby('year')['age'].count() / df.groupby('year')['age'].size()
    eligible_years = age_validity[age_validity >= min_validity].index

    # Keep only years that are eligible and still have missing ages.
    years_needing_imputation = [
        year for year in eligible_years
        if df.loc[df['year'] == year, 'age'].isna().any()
    ]
    if not years_needing_imputation:
        print("  No missing ages to impute in eligible years.")
        return df

    n_to_impute_total = sum(
        df.loc[df['year'] == year, 'age'].isna().sum()
        for year in years_needing_imputation
    )
    print(f"  Imputing {n_to_impute_total:,} missing age values via per-year MICE "
          f"(IterativeImputer sample_posterior=True)...")
    print(f"  Imputation variance diagnostic (imputed_std / real_std, closer to 1 = better):")

    feature_cols = ['age', 'seconds', 'gender']
    context_features = ['seconds', 'gender']

    for year in years_needing_imputation:
        year_mask = df['year'] == year
        # Use rows from this year with valid predictors.
        sub = df.loc[year_mask & df[context_features].notna().all(axis=1), feature_cols].copy()
        missing_idx = sub.index[sub['age'].isna()]

        # Standardize `seconds`, encode gender, and impute `age`.
        preprocess = ColumnTransformer(transformers=[
            ('age', 'passthrough', ['age']),
            ('num', StandardScaler(), ['seconds']),
            ('gender', OrdinalEncoder(categories=[['M', 'F']], dtype=float), ['gender']),
        ], verbose_feature_names_out=False).set_output(transform='pandas')
        imputer = IterativeImputer(sample_posterior=True, random_state=random_state, max_iter=10)
        pipe = Pipeline([('pre', preprocess), ('imp', imputer)]).set_output(transform='pandas')
        # Clamp imputed ages to a plausible range before rounding.
        imputed_col = pipe.fit_transform(sub)['age'].clip(lower=14, upper=90)

        df.loc[missing_idx, 'age'] = imputed_col.loc[missing_idx].round().astype(int)
        df.loc[missing_idx, 'age_imputed'] = True

        # Per-year variance diagnostic.
        real = df.loc[year_mask & ~df['age_imputed'] & df['age'].notna(), 'age']
        imp = df.loc[missing_idx, 'age']
        ratio = imp.std(ddof=1) / real.std(ddof=1) if len(real) > 1 and len(imp) > 1 else float('nan')
        print(f"    {year}: n_imputed={len(imp):,}, real_std={real.std(ddof=1):.2f}, "
              f"imputed_std={imp.std(ddof=1):.2f}, ratio={ratio:.3f}")
    return df


def mark_within_year_collisions(df):
    """Flag rows where different runners share the same name in the same race.

    Adds a boolean column `name_collides_within_year` that is True when a
    `(year, display_name)` pair appears more than once.
    """
    df['name_collides_within_year'] = df.duplicated(subset=['year', 'display_name'], keep=False)
    return df


def assert_invariants(df):
    """Run sanity checks. Any failure stops the script.

    These checks validate the assumptions used by downstream analysis.
    """
    # Convert `seconds` to numpy so NaN values fail the comparison.
    seconds_np = df['seconds'].to_numpy()
    assert (seconds_np > 0).all(), "seconds must be strictly positive (and not NA)"
    assert (seconds_np < 50_000).all(), f"seconds exceeds 50,000 (>13h, absurd): {(seconds_np >= 50_000).sum()} rows"
    assert df['gender'].isin(['M', 'F']).all(), "gender must be M or F"
    age_ok = df['age'].isna() | df['age'].between(14, 90)
    assert age_ok.all(), f"age out of [14,90]: {(~age_ok).sum()} rows"

    # Finishing places must nest correctly.
    place_ok = (df['gender_result'].isna() | df['overall'].isna() |
                (df['gender_result'] <= df['overall']))
    assert place_ok.all(), f"gender_result > overall: {(~place_ok).sum()} rows"
    div_ok = (df['division_result'].isna() | df['gender_result'].isna() |
              (df['division_result'] <= df['gender_result']))
    assert div_ok.all(), f"division_result > gender_result: {(~div_ok).sum()} rows"

    # Split times must be strictly increasing, and 40k must be less than finish time.
    split_seconds_cols = [f'{c}_seconds' for c in SPLIT_COLS]
    sp = df[df[split_seconds_cols].notna().all(axis=1)]
    full_sequence = sp[split_seconds_cols + ['seconds']]
    viols_per_transition = (full_sequence.diff(axis=1).iloc[:, 1:] <= 0).sum()
    total_viols = int(viols_per_transition.sum())
    assert total_viols == 0, (
        f"split/finish monotonicity violated: {viols_per_transition[viols_per_transition > 0].to_dict()}"
    )

    print("  All data-quality invariants satisfied:")
    print("    0 < seconds < 50_000, gender in {M,F}, age in [14,90], place nesting, split monotonicity")


def main():
    print("Boston Marathon Data Cleaning")

    # Collect each `results*.csv` file, excluding wheelchair and diverted-course variants.
    manifest = [
        (path, int(match.group(1)))
        for path in sorted(DATA_DIR.glob('results*.csv'))
        if '_includes-wheelchair' not in path.name
        and '_without-diverted' not in path.name
        and (match := re.search(r'results(\d{4})', path.stem))
    ]
    print(f"  Loading {len(manifest)} raw CSVs ({manifest[0][1]}-{manifest[-1][1]})...")
    df = pd.concat([load_and_unify(fp, yr) for fp, yr in manifest], ignore_index=True)
    n_loaded = len(df)
    print(f"  Loaded: {n_loaded:,} rows")

    df = clean_types(df)

    # Fill missing `seconds` from parsed `official_time`.
    # Then fill missing pace from finish time and marathon distance.
    df['seconds'] = df['seconds'].astype('Float64').combine_first(df['_official_time_seconds'].astype('Float64'))
    df = df.drop(columns=['_official_time_seconds'])
    df['pace_seconds_per_mile'] = df['pace_seconds_per_mile'].fillna(df['seconds'] / MARATHON_MILES)

    # Keep only M/F gender values after cleaning spacing and case.
    df['gender'] = df['gender'].astype('string').str.strip().str.upper()
    n_before_gender = len(df)
    df = df[df['gender'].isin(['M', 'F'])].copy()
    print(f"  Gender filter (keep M/F): {n_before_gender:,} -> {len(df):,} ({n_before_gender-len(df):,} dropped)")

    # Treat age 0 and ages outside 14-90 as missing.
    df['age'] = df['age'].replace(0, np.nan).where(df['age'].between(14, 90))

    df = impute_age(df)

    # Drop rows with non-positive or missing finish time, then remove exact duplicates.
    n_before_valid = len(df)
    df = df[~(df['seconds'] <= 0)].drop_duplicates()
    print(f"  Non-positive seconds + duplicate filter: {n_before_valid:,} -> {len(df):,} ({n_before_valid-len(df):,} dropped)")

    df['gender'] = df['gender'].astype('category')
    df = mark_within_year_collisions(df)
    n_collide = df['name_collides_within_year'].sum()
    collide_groups = df[df['name_collides_within_year']].groupby(['year', 'display_name']).ngroups
    print(f"  Within-year name collision flag: {n_collide:,} rows in {collide_groups:,} distinct (year, display_name) groups")

    # Drop text time columns after numeric versions are available.
    df = df.drop(columns=list(REDUNDANT_STRING_COLS))
    print(f"  Dropped redundant string columns: {REDUNDANT_STRING_COLS}")

    assert_invariants(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / 'boston_marathon_cleaned.csv'
    df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved cleaned data to {output_path} ({len(df):,} rows, {len(df.columns)} columns, {size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
"""Build one clean CSV from the raw per-year Boston Marathon result files.

The raw files in data/raw/ have different column names and formats across years.
This script does the following, in order:

  1. Read every results*.csv file and reshape it to the same 29-column layout
  2. Turn time strings like "2:45:30" into plain numbers of seconds
  3. Drop rows with missing gender or out-of-range age
  4. Fill in missing ages for years that have enough age data to learn from
  5. Drop non-positive times and exact duplicate rows
  6. Flag rows where two different runners share a name in the same race
  7. Drop text columns whose numeric versions already exist (saves ~14 MB)
  8. Run a handful of sanity checks and crash if any of them fail
  9. Save one clean file at data/processed/boston_marathon_cleaned.csv

The final output has 28 columns and about 615K rows.
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

# Text columns whose information is already stored numerically elsewhere. These
# are dropped before writing the output to keep the file small. `official_time`
# becomes `seconds`, `pace` becomes `pace_seconds_per_mile`, and each split like
# `5k` becomes `5k_seconds`.
REDUNDANT_STRING_COLS = ['official_time', 'pace', 'projected_time'] + SPLIT_COLS

NA_VALUES = ['NULL', '', ' ', 'null', 'N/A', 'NA', 'n/a', 'na', '-']

MARATHON_MILES = 26.2188


def parse_time_to_seconds(series):
    """Turn a column of time strings into a column of total seconds.

    Handles both `H:MM:SS` format (like `"2:45:30"` for 2 hours 45 minutes
    30 seconds) and `MM:SS` format (like `"45:30"` for 45 minutes 30 seconds).
    Empty strings and unparseable values become NaN. Runs on the whole column
    at once using pandas time parsing, so it's fast even on 600K+ rows.
    """
    values = pd.Series(series, copy=False).astype('string').str.strip()
    values = values.mask(values.eq(''), pd.NA)
    # `MM:SS` rows get a `"00:"` prefix so pandas can parse them as HH:MM:SS.
    mm_ss = values.str.fullmatch(r'\d{1,2}:\d{2}')
    normalized = values.mask(mm_ss, '00:' + values)
    td = pd.to_timedelta(normalized, errors='coerce')
    return td.dt.total_seconds()


def load_and_unify(filepath, year):
    """Read one year's raw CSV and reshape it to the standard column layout.

    Different years use different column names and orderings. This function:
      - Reads the file, treating a few standard missing-value tokens as NaN.
      - Stamps the given `year` on every row.
      - Fixes a known typo in newer files (`contry_citizenship` -> `country_citizenship`).
      - Builds a combined `residence` string from city/state/country where those
        columns exist separately (newer files have them split out).
      - Drops every column not in `UNIFIED_COLS` and adds any missing ones as NaN,
        so every year's DataFrame comes out with the same 29-column shape.

    `on_bad_lines='skip'` tells pandas to drop rows whose quoting is broken.
    One raw file (results2019.csv) has a stray backslash-quote on one line that
    the CSV parser can't handle. This is the only place in the whole script
    that silently skips bad input.
    """
    df = pd.read_csv(filepath, na_values=NA_VALUES, dtype='string', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.strip('"')
    df['year'] = year

    # Newer files (post-2001) have a `place_overall` column, split city/state/
    # country fields, and sometimes a misspelled citizenship column. Detecting
    # `place_overall` is how newer files are identified.
    if 'place_overall' in df.columns:
        if 'contry_citizenship' in df.columns:
            df = df.rename(columns={'contry_citizenship': 'country_citizenship'})
        address_parts = df[['city', 'state', 'country_residence']].astype('string').apply(lambda col: col.str.strip()).replace('', pd.NA)
        df['residence'] = address_parts.stack(future_stack=True).dropna().groupby(level=0).agg(', '.join).reindex(df.index)
        if 'name' in df.columns and 'display_name' not in df.columns:
            df = df.rename(columns={'name': 'display_name'})

    df = df.reindex(columns=UNIFIED_COLS, fill_value=pd.NA)
    text_like_cols = [col for col in UNIFIED_COLS if col != 'year']
    df[text_like_cols] = df[text_like_cols].astype('string')

    return df


def clean_types(df):
    """Convert text columns into numbers and parse time strings into seconds.

    Every column is still a string at this point. This function:
      - Strips whitespace from text columns and turns blank strings into NaN.
      - Casts age, finishing places, and bib numbers to numeric dtypes.
      - Casts the main finish time (`seconds`) to a nullable float.
      - Adds a temporary `_official_time_seconds` column by parsing the
        `official_time` string (used later as a backup for `seconds`).
      - Adds a `pace_seconds_per_mile` column by parsing the `pace` string.
      - Adds a `<split>_seconds` column for each of the nine checkpoint splits.

    The raw text time columns are still present after this step; they get
    dropped at the end of `main()` once they are no longer needed.
    """
    str_cols = df.select_dtypes(include='string').columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip()).replace(r'^\s*$', pd.NA, regex=True)

    numeric_cast_cols = ['age', 'overall', 'gender_result', 'division_result', 'bib']
    df[numeric_cast_cols] = df[numeric_cast_cols].apply(pd.to_numeric, errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce').astype('Float64')

    df['_official_time_seconds'] = parse_time_to_seconds(df['official_time'])
    df['pace_seconds_per_mile'] = parse_time_to_seconds(df['pace'])
    df = df.assign(**{f'{c}_seconds': parse_time_to_seconds(df[c]) for c in SPLIT_COLS})

    return df


def impute_age(df, *, random_state=42, min_validity=0.50):
    """Fill in missing ages for years that have enough real ages to learn from.

    For each race year where at least half the rows already have an age AND at
    least one row is missing one, a small Bayesian linear regression is fit
    that predicts age from finish time and gender on that year's data, then
    the missing ages are filled in by *drawing* from the regression's
    posterior distribution. Drawing — instead of always taking the single
    best-guess prediction — keeps the spread of imputed ages close to the
    spread of real ages. A simpler nearest-neighbour imputer would pull
    every imputed age toward the local mean and collapse the variance.

    A separate model is fit per year on purpose. Finish times drift year to
    year (course changes, participation policies, weather), and a single
    global model would confuse that year-to-year drift for within-year spread
    and produce imputed ages with inflated variance.

    After filling in values, the function prints a per-year diagnostic: the
    ratio of (std of imputed ages) to (std of real ages). A ratio near 1.0
    means the natural spread was preserved well.

    Parameters:
      random_state   seed for reproducible posterior draws
      min_validity   minimum fraction of non-missing ages a year must have
                     before it is trusted as training data for itself
    """
    df['age_imputed'] = False

    # Fraction of rows in each year that already have an age.
    age_validity = df.groupby('year')['age'].count() / df.groupby('year')['age'].size()
    eligible_years = age_validity[age_validity >= min_validity].index

    # Keep only the years that (a) are eligible AND (b) actually contain at least
    # one missing age. Years that are fully populated or fully empty need nothing.
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
        # Build the training subset: rows from this year that have both a
        # finish time and a gender (so the regression has inputs to work with).
        sub = df.loc[year_mask & df[context_features].notna().all(axis=1), feature_cols].copy()
        missing_idx = sub.index[sub['age'].isna()]

        # Standardise `seconds`, encode gender as 0/1, leave `age` as-is. The
        # imputer will fit a Bayesian regression and draw ages for NaN rows.
        preprocess = ColumnTransformer(transformers=[
            ('age', 'passthrough', ['age']),
            ('num', StandardScaler(), ['seconds']),
            ('gender', OrdinalEncoder(categories=[['M', 'F']], dtype=float), ['gender']),
        ], verbose_feature_names_out=False).set_output(transform='pandas')
        imputer = IterativeImputer(sample_posterior=True, random_state=random_state, max_iter=10)
        pipe = Pipeline([('pre', preprocess), ('imp', imputer)]).set_output(transform='pandas')
        # Clamp imputed ages to a plausible range before rounding to integers.
        imputed_col = pipe.fit_transform(sub)['age'].clip(lower=14, upper=90)

        df.loc[missing_idx, 'age'] = imputed_col.loc[missing_idx].round().astype(int)
        df.loc[missing_idx, 'age_imputed'] = True

        # Per-year variance check. If the ratio is much below 1, the imputer
        # is regressing toward the mean and the imputed values look too uniform.
        real = df.loc[year_mask & ~df['age_imputed'] & df['age'].notna(), 'age']
        imp = df.loc[missing_idx, 'age']
        ratio = imp.std(ddof=1) / real.std(ddof=1) if len(real) > 1 and len(imp) > 1 else float('nan')
        print(f"    {year}: n_imputed={len(imp):,}, real_std={real.std(ddof=1):.2f}, "
              f"imputed_std={imp.std(ddof=1):.2f}, ratio={ratio:.3f}")
    return df


def mark_within_year_collisions(df):
    """Flag rows where two different runners share the same name in the same race.

    Common names like "Aaron Smith" sometimes belong to more than one person in
    the same Boston Marathon — about 3,419 such (year, name) groups affecting
    7,374 rows in this dataset. Without a flag, per-runner analyses would treat
    these rows as one person and compute nonsense year-over-year slopes across
    different people.

    This adds a boolean column `name_collides_within_year` that is True for any
    row whose (year, display_name) pair appears more than once, using pandas'
    canonical `DataFrame.duplicated(keep=False)` idiom. `keep=False` marks every
    member of a duplicate group, not just the second and later ones.
    """
    df['name_collides_within_year'] = df.duplicated(subset=['year', 'display_name'], keep=False)
    return df


def assert_invariants(df):
    """Run a handful of sanity checks. Any failure crashes the script.

    These are the properties every downstream analysis assumes to hold. If an
    assertion fires, either the raw data has a new problem not seen before
    or the cleaning logic regressed. Either way, execution should stop here
    rather than silently pass broken data on to the modelling step.
    """
    # Convert `seconds` to a numpy array before comparing. The column has the
    # pandas nullable `Float64` dtype, and on that dtype `Series.gt(0).all()`
    # quietly skips any NaN rows — a missing finish time would pass the check.
    # numpy treats `np.nan > 0` as False, so any NaN correctly fails the check.
    seconds_np = df['seconds'].to_numpy()
    assert (seconds_np > 0).all(), "seconds must be strictly positive (and not NA)"
    assert (seconds_np < 50_000).all(), f"seconds exceeds 50,000 (>13h, absurd): {(seconds_np >= 50_000).sum()} rows"
    assert df['gender'].isin(['M', 'F']).all(), "gender must be M or F"
    age_ok = df['age'].isna() | df['age'].between(14, 90)
    assert age_ok.all(), f"age out of [14,90]: {(~age_ok).sum()} rows"

    # Finishing places have to nest: a runner's division place can't be higher
    # than their gender place, and their gender place can't be higher than
    # their overall place. Rows where either column is NaN are skipped.
    place_ok = (df['gender_result'].isna() | df['overall'].isna() |
                (df['gender_result'] <= df['overall']))
    assert place_ok.all(), f"gender_result > overall: {(~place_ok).sum()} rows"
    div_ok = (df['division_result'].isna() | df['gender_result'].isna() |
              (df['division_result'] <= df['gender_result']))
    assert div_ok.all(), f"division_result > gender_result: {(~div_ok).sum()} rows"

    # Runners can only move forward on the course, so each split time must be
    # strictly larger than the previous one (5k < 10k < 15k < ... < 40k) and
    # the 40k split must be smaller than the final finish time. Only rows
    # that have every checkpoint filled in are checked, which in practice
    # means the 2015-2017 finishers.
    #
    # The check uses pandas' canonical `DataFrame.diff(axis=1)`, which
    # computes each cell's value minus the cell to its left. For a strictly
    # increasing row every post-diff cell must be positive; any cell ≤ 0 is
    # a violation of the split/finish ordering. This one vectorised call
    # covers all 10 transitions (5k→10k→...→40k→seconds) at once.
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

    # Collect every `results*.csv` in data/raw/, skipping the wheelchair and
    # diverted-course variants (which are not included in the final dataset).
    # Each entry of `manifest` is a (file path, year) pair.
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

    # Some rows have no `seconds` column but do have an `official_time` string.
    # Use the parsed version of `official_time` as a backup when `seconds` is
    # missing. Then compute `pace_seconds_per_mile` for any row where the pace
    # string was absent or unparseable.
    df['seconds'] = df['seconds'].astype('Float64').combine_first(df['_official_time_seconds'].astype('Float64'))
    df = df.drop(columns=['_official_time_seconds'])
    df['pace_seconds_per_mile'] = df['pace_seconds_per_mile'].fillna(df['seconds'] / MARATHON_MILES)

    # Drop rows whose gender is not "M" or "F". Raw files sometimes contain
    # blanks, "U", "X", or stray whitespace. Upper-case and strip whitespace
    # first so spelling differences don't survive the filter.
    df['gender'] = df['gender'].astype('string').str.strip().str.upper()
    n_before_gender = len(df)
    df = df[df['gender'].isin(['M', 'F'])].copy()
    print(f"  Gender filter (keep M/F): {n_before_gender:,} -> {len(df):,} ({n_before_gender-len(df):,} dropped)")

    # Treat age = 0 and ages outside 14-90 as missing, so the age imputer
    # can fill in plausible values for those rows later.
    df['age'] = df['age'].replace(0, np.nan).where(df['age'].between(14, 90))

    df = impute_age(df)

    # Drop rows with a non-positive or missing finish time, and any exact-
    # duplicate rows (identical across every column).
    n_before_valid = len(df)
    df = df[~(df['seconds'] <= 0)].drop_duplicates()
    print(f"  Non-positive seconds + duplicate filter: {n_before_valid:,} -> {len(df):,} ({n_before_valid-len(df):,} dropped)")

    df['gender'] = df['gender'].astype('category')
    df = mark_within_year_collisions(df)
    n_collide = df['name_collides_within_year'].sum()
    collide_groups = df[df['name_collides_within_year']].groupby(['year', 'display_name']).ngroups
    print(f"  Within-year name collision flag: {n_collide:,} rows in {collide_groups:,} distinct (year, display_name) groups")

    # Drop the text copies of times and splits. Their numeric versions are all
    # present (`5k_seconds` replaces `5k`, `seconds` replaces `official_time`,
    # and `pace_seconds_per_mile` replaces `pace`). Saves about 14 MB on disk.
    df = df.drop(columns=[c for c in REDUNDANT_STRING_COLS if c in df.columns])
    print(f"  Dropped redundant string columns: {REDUNDANT_STRING_COLS}")

    assert_invariants(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / 'boston_marathon_cleaned.csv'
    df.to_csv(output_path, index=False)

    # Delete the parquet cache if one exists. The next downstream script that
    # wants the cleaned data will read this fresh CSV and rebuild the cache
    # on its own the first time.
    parquet_cache = OUTPUT_DIR / 'boston_marathon_cleaned.parquet'
    if parquet_cache.exists():
        parquet_cache.unlink()
        print(f"  Invalidated stale parquet cache: {parquet_cache.name}")

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved cleaned data to {output_path} ({len(df):,} rows, {len(df.columns)} columns, {size_mb:.1f} MB)")


if __name__ == '__main__':
    main()

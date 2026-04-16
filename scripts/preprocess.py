import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

NA_VALUES = ["NULL", "", " ", "null", "N/A", "NA", "n/a", "na", "-"]


def run_data_cleaning():
    RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

    frames = []
    for fp in sorted(RAW_DIR.glob("results*.csv")):
        if "_includes-wheelchair" in fp.name or "_without-diverted" in fp.name:
            continue
        m = re.search(r"results(\d{4})", fp.stem)
        if m:
            frames.append(pd.read_csv(fp, na_values=NA_VALUES, dtype="string", on_bad_lines="skip")
                          .rename(columns=str.strip).rename(columns=lambda c: c.strip('"')).assign(year=int(m.group(1))))
    df = pd.concat(frames, ignore_index=True)

    # Residence field for 2015+
    post = df["year"] >= 2015
    parts = df.loc[post, ["city", "state", "country_residence"]].astype("string").apply(lambda c: c.str.strip()).replace("", pd.NA)
    df.loc[post, "residence"] = parts.apply(lambda row: ", ".join(row.dropna()), axis=1).replace("", pd.NA)

    df = df.rename(columns={"contry_citizenship": "country_citizenship"}).reindex(
        columns=["year", "display_name", "first_name", "last_name", "age", "gender", "residence", "city", "state",
                  "country_residence", "country_citizenship", "bib", "pace", "official_time", "seconds", "overall",
                  "gender_result", "division_result", "5k", "10k", "15k", "20k", "half", "25k", "30k", "35k", "40k",
                  "pace_seconds_per_mile"], fill_value=pd.NA)

    string_cols = [c for c in df.columns if c != "year"]
    df[string_cols] = df[string_cols].astype("string").apply(lambda c: c.str.strip()).replace(r"^\s*$", pd.NA, regex=True)
    for c in ("overall", "gender_result", "division_result"):
        df[c] = pd.to_numeric(df[c])
    df["seconds"] = pd.to_numeric(df["seconds"]).astype("Float64")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Parse time splits
    split_cols = ["5k", "10k", "15k", "20k", "half", "25k", "30k", "35k", "40k"]
    for s in ["official_time", "pace"] + split_cols:
        vals = df[s].mask(df[s].eq(""), pd.NA)
        corrected = vals.mask(vals.str.fullmatch(r"\d{1,2}:\d{2}"), "00:" + vals)
        parsed = pd.to_timedelta(corrected, errors="coerce").dt.total_seconds()
        df[f"{s}_seconds" if s in split_cols else f"_{s}_seconds"] = parsed

    df["seconds"] = df["seconds"].combine_first(df["_official_time_seconds"].astype("Float64"))
    df["pace_seconds_per_mile"] = df.pop("_pace_seconds").fillna(df["seconds"] / 26.2188)
    df = df.drop(columns=["_official_time_seconds", "official_time", "pace"] + split_cols)
    df = df.loc[df["gender"].str.upper().isin(["M", "F"])]
    df["gender"] = df["gender"].str.upper().astype("category")
    df["age"] = df["age"].where(df["age"].between(14, 90))
    df["age_imputed"] = False

    # MICE imputation per year
    for yr in [y for y, g in df.groupby("year")["age"] if g.isna().any() and g.notna().any()]:
        mask = (df["year"] == yr) & df[["seconds", "gender"]].notna().all(axis=1)
        imp = Pipeline([
            ("pre", ColumnTransformer([("age", "passthrough", ["age"]), ("num", StandardScaler(), ["seconds"]),
                                        ("gender", OrdinalEncoder(categories=[["M", "F"]], dtype=float), ["gender"])],
                                       verbose_feature_names_out=False).set_output(transform="pandas")),
            ("imp", IterativeImputer(sample_posterior=False, random_state=42)),
        ]).set_output(transform="pandas").fit_transform(df.loc[mask, ["age", "seconds", "gender"]])
        missing = df.index[(df["year"] == yr) & df["age"].isna()]
        df.loc[missing, "age"] = imp.loc[missing, "age"].clip(lower=14, upper=90).round().astype(int)
        df.loc[missing, "age_imputed"] = True

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (df[~(df["seconds"] <= 0)].drop_duplicates(subset=["year", "display_name", "seconds"])
     .assign(name_collides_within_year=lambda d: d.duplicated(subset=["year", "display_name"], keep=False))
     .to_csv(PROC_DIR / "boston_marathon_cleaned.csv", index=False))


if __name__ == "__main__":
    run_data_cleaning()

import numpy as np
import pandas as pd
import meteostat
from datetime import datetime

RACE_DATES = [
    "2000-04-17", "2001-04-16", "2002-04-15", "2003-04-21", "2004-04-19",
    "2005-04-18", "2006-04-17", "2007-04-16", "2008-04-21", "2009-04-20",
    "2010-04-19", "2011-04-18", "2012-04-16", "2013-04-15", "2014-04-21",
    "2015-04-20", "2016-04-18", "2017-04-17", "2018-04-16", "2019-04-15",
]


def load_processed_results(cfg):
    return pd.read_csv(cfg.data_csv, engine="pyarrow", usecols=cfg.pipeline_columns)


def add_centered_pre_race_features(df, age_mean, cfg):
    df["age_centered"] = df["age"] - age_mean
    df["age_centered_squared"] = df["age_centered"] ** 2
    df["year_centered"] = df["year"] - cfg.year_center
    df["female"] = (df["gender"] == "F").astype(int)
    df["age_centered_female_interaction"] = df["age_centered"] * df["female"]
    df["age_centered_squared_female_interaction"] = df["age_centered_squared"] * df["female"]


def add_prior_boston_history_features(df):
    df = df.sort_values(["display_name", "year"]).drop_duplicates(subset=["display_name", "year"])
    cum_count = df.groupby("display_name").cumcount() + 1
    df["prior_mean_time"] = df.groupby("display_name")["seconds"].cumsum().groupby(df["display_name"]).shift() / (cum_count - 1)
    df["prior_appearances"] = cum_count - 1
    df["log1p_prior_appearances"] = np.log1p(cum_count - 1)
    return df


def build_repeat_runner_analysis_sample(df, cfg):
    df = df[~df["age_imputed"] & df["age"].notna() & (df["year"] >= cfg.analysis_start_year)
            & (df.groupby("display_name")["year"].transform("size") > 1)].sort_values(["display_name", "year"])
    names, age_diff, yr_diff = df["display_name"].to_numpy(), np.diff(df["age"].to_numpy()), np.diff(df["year"].to_numpy())
    same_name = names[1:] == names[:-1]
    deviation = np.abs(age_diff - yr_diff)
    collision_threshold = int(np.ceil(np.percentile(deviation[same_name], 95))) + 1
    collisions = set(names[1:][(same_name) & (deviation > collision_threshold)])
    df = df[~df["display_name"].isin(collisions)]
    return df[df.groupby("display_name")["year"].transform("size") > 1].copy()


def load_in_race_split_dataset(re_df, df, weather_df, cfg):
    df = df[df[cfg.cumulative_split_time_columns].notna().all(axis=1)].copy()
    df["female"] = (df["gender"] == "F").astype(int)
    df["year_centered"] = df["year"] - cfg.year_center
    df["bib_feature"] = df["bib"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df["age_group"] = pd.cut(df["age"], bins=cfg.age_group_bins, labels=cfg.age_group_labels, right=False)
    df = df.merge(re_df.reset_index(), on="display_name", how="left").merge(weather_df[["year", "max_temp_f"]], on="year", how="left")
    return df[df["year"].isin(cfg.in_race_train_years)].copy(), df[df["year"] == cfg.in_race_test_year].copy()


def compute_segment_pacing_features(df, cfg):
    out = df.copy()
    cols, dists = cfg.cumulative_split_time_columns, cfg.segment_distances_m
    cumulative = np.column_stack([out[c].to_numpy(float) for c in cols])
    seg_paces = np.column_stack([cumulative[:, :1], np.diff(cumulative, axis=1)]) / dists
    for i in range(len(cols)):
        out[f"seg_pace_{i}"] = seg_paces[:, i]
    for i in range(1, len(cols)):
        out[f"seg_pace_change_{i}"] = seg_paces[:, i] - seg_paces[:, i - 1]
    return out


def compute_latent_information_features(train, test, cfg):
    train, test = train.copy(), test.copy()
    bib_median = train["bib_feature"].median()
    train["bib_feature"], test["bib_feature"] = train["bib_feature"].fillna(bib_median), test["bib_feature"].fillna(bib_median)
    for d in (train, test):
        d["heat_exposure"] = (d["5k_seconds"] / 5000.0 * 42195.0 / 3600.0) * d["max_temp_f"]

    cum_dists = np.cumsum(cfg.segment_distances_m)
    for ci, col in enumerate(cfg.cumulative_split_time_columns):
        train["_cpace"], test["_cpace"] = train[col] / cum_dists[ci], test[col] / cum_dists[ci]
        cohort = (train.groupby(["gender", "age_group"], observed=True)["_cpace"]
                  .agg(["mean", "std"]).reset_index().rename(columns={"mean": "_cmean", "std": "_cstd"}))
        cohort["_cstd"] = cohort["_cstd"].replace(0, 1)
        train = train.merge(cohort, on=["gender", "age_group"], how="left")
        test = test.merge(cohort, on=["gender", "age_group"], how="left")
        train[f"z_pace_{ci}"] = ((train["_cpace"] - train["_cmean"]) / train["_cstd"]).fillna(0.0)
        test[f"z_pace_{ci}"] = ((test["_cpace"] - test["_cmean"]) / test["_cstd"]).fillna(0.0)
        train.drop(columns=["_cmean", "_cstd"], inplace=True)
        test.drop(columns=["_cmean", "_cstd"], inplace=True)
        if ci > 0:
            pace_cols = [f"seg_pace_{j}" for j in range(ci + 1)]
            train[f"pace_variance_{ci}"] = train[pace_cols].std(axis=1).fillna(0.0)
            test[f"pace_variance_{ci}"] = test[pace_cols].std(axis=1).fillna(0.0)
        else:
            train[f"pace_variance_{ci}"] = test[f"pace_variance_{ci}"] = 0.0

    return train.drop(columns=["_cpace"]), test.drop(columns=["_cpace"])


def fetch_race_day_weather(cfg):
    if cfg.weather_csv.exists():
        return pd.read_csv(cfg.weather_csv)
    rows = []
    for d in RACE_DATES:
        h = meteostat.hourly("72509", datetime.strptime(d + " 10:00", "%Y-%m-%d %H:%M"),
                             datetime.strptime(d + " 18:00", "%Y-%m-%d %H:%M")).fetch()
        rows.append({"year": int(d[:4]), "date": d, "mean_temp_f": h["temp"].mean() * 9/5 + 32,
                      "max_temp_f": h["temp"].max() * 9/5 + 32, "mean_humidity": h["rhum"].mean(),
                      "mean_wind_mph": h["wspd"].mean() * 0.621371})
    weather = pd.DataFrame(rows)
    weather.to_csv(cfg.weather_csv, index=False)
    print(f"  cached weather data to {cfg.weather_csv}")
    return weather

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy.stats import kruskal, linregress, mannwhitneyu, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from .data import add_centered_pre_race_features, add_prior_boston_history_features, compute_latent_information_features, compute_segment_pacing_features
from .inference import bootstrap_rmse_comparison, compute_bca_cluster_bootstrap_rmse

NESTED_MODELS = ["Sample_Mean", "OLS_Log_Quadratic_Demographics_HC3", "OLS_Log_Quadratic_Demographics_History_HC3",
                 "Ridge_GCV_5K_Splits_Demographics", "Ridge_GCV_40K_Splits_Demographics"]
MODEL_DESCRIPTIONS = {
    "Sample_Mean": "Sample mean constant prediction",
    "OLS_Log_Quadratic_Demographics_HC3": "+ demographics (OLS log-quadratic, HC3 robust SE)",
    "OLS_Log_Quadratic_Demographics_History_HC3": "+ prior history (OLS log-quadratic, HC3 robust SE)",
    "Ridge_GCV_5K_Splits_Demographics": "+ 5K cumulative splits (ridge, GCV-regularized)",
    "Ridge_GCV_40K_Splits_Demographics": "+ 40K cumulative splits (ridge, GCV-regularized)",
}
EXPANDED_STAGES = [
    ("Sample_Mean", "linear", "Sample mean constant prediction"),
    ("OLS_Log_Quadratic_Demographics_HC3", "linear", "+ demographics (OLS log-quadratic, HC3 robust SE)"),
    ("OLS_Log_Quadratic_Demographics_History_HC3", "linear", "+ prior history (OLS log-quadratic, HC3 robust SE)"),
    ("Ridge_GCV_5K_Splits_Demographics", "linear", "+ 5K cumulative splits (ridge, GCV-regularized)"),
    ("LGBM_5K_Splits_Pacing_Latent", "nonlinear", "+ 5K splits + pacing + latent (LightGBM gradient boosting)"),
    ("Ridge_GCV_40K_Splits_Demographics", "linear", "+ 40K cumulative splits (ridge, GCV-regularized)"),
    ("LGBM_40K_Splits_Pacing_Latent", "nonlinear", "+ 40K splits + pacing + latent (LightGBM gradient boosting)"),
]


def _conformal_coverage(y, pred, q):
    return np.mean((y >= pred - q) & (y <= pred + q))


def _prepare_common_evaluation_data(df, age_mean_train, weather_df, cfg):
    splits = add_prior_boston_history_features(
        df[~df["age_imputed"] & df["age"].notna() & (df["year"] >= cfg.analysis_start_year)].copy()
    ).dropna(subset=cfg.cumulative_split_time_columns)
    splits["female"] = (splits["gender"] == "F").astype(int)
    splits["bib_feature"] = splits["bib"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    splits["age_group"] = pd.cut(splits["age"], bins=cfg.age_group_bins, labels=cfg.age_group_labels, right=False)
    merged = splits.merge(weather_df[["year", "max_temp_f"]], on="year", how="left")
    tr = compute_segment_pacing_features(merged.loc[merged["year"].isin(cfg.in_race_train_years)].copy(), cfg)
    te = compute_segment_pacing_features(merged.loc[merged["year"] == cfg.in_race_test_year].copy(), cfg)
    _, te = compute_latent_information_features(tr, te, cfg)
    add_centered_pre_race_features(te, age_mean_train, cfg)
    return te


def evaluate_nested_model_comparison(demo_fit, hist_fit, train_mean, fitted_models, rng, test_df, cfg):
    obs, n = test_df["seconds"].to_numpy(float), len(test_df)
    ols_demo_pred = np.exp(demo_fit[0].predict(test_df).to_numpy()) * demo_fit[1]
    ols_hist_pred = np.exp(hist_fit[0].predict(test_df).to_numpy()) * hist_fit[1]
    ridge_5k, ridge_40k = fitted_models[("5K", "no_runner_history")], fitted_models[("40K", "no_runner_history")]
    pred_list = [np.full(n, train_mean), ols_demo_pred,
                  np.where(test_df["prior_mean_time"].notna(), ols_hist_pred, ols_demo_pred),
                  ridge_5k["ridge_regression_model"].predict(test_df[ridge_5k["features"]].to_numpy()),
                  ridge_40k["ridge_regression_model"].predict(test_df[ridge_40k["features"]].to_numpy())]

    pt, lo, hi = compute_bca_cluster_bootstrap_rmse(obs, pred_list, test_df["display_name"].to_numpy(), rng, cfg)
    has_hist = test_df["prior_mean_time"].notna()
    coverages = [np.nan, _conformal_coverage(obs, ols_demo_pred, demo_fit[2]),
                  _conformal_coverage(obs[has_hist], ols_hist_pred[has_hist], hist_fit[2]),
                  _conformal_coverage(obs, pred_list[3], ridge_5k["q_hat"]), _conformal_coverage(obs, pred_list[4], ridge_40k["q_hat"])]
    q_hats = [np.nan, demo_fit[2], hist_fit[2], ridge_5k["q_hat"], ridge_40k["q_hat"]]

    rows = []
    for i, lbl in enumerate(NESTED_MODELS):
        row = {"stage": lbl, "description": MODEL_DESCRIPTIONS[lbl], "n": n,
               "rmse_seconds": pt[i], "rmse_ci_lower": lo[i], "rmse_ci_upper": hi[i],
               "empirical_coverage": coverages[i], "q_hat": q_hats[i]}
        if i == 0:
            row.update(improvement_seconds=np.nan, improvement_ci_lower=np.nan, improvement_ci_upper=np.nan)
        else:
            j = len(NESTED_MODELS) + i - 1
            row.update(improvement_seconds=pt[j], improvement_ci_lower=lo[j], improvement_ci_upper=hi[j])
        rows.append(row)
    return pd.DataFrame(rows)


def compare_personalization(pre_state, lme_state, holders, rng, cfg):
    pre_test = pre_state["pool"][pre_state["pool"]["year"].isin(cfg.pre_race_test_years)].copy()
    add_centered_pre_race_features(pre_test, pre_state["age_mean"], cfg)
    pre_feats = ["display_name", "prior_mean_time", "log1p_prior_appearances"] + cfg.pre_race_fixed_effect_features
    common = holders.drop(columns=["female", "year_centered"]).merge(
        pre_test[pre_feats].drop_duplicates("display_name"), on="display_name", how="inner")

    ols_hist_pred = np.exp(pre_state["hist_fit"][0].predict(common).to_numpy()) * pre_state["hist_fit"][1]
    common["age_centered"] = common["age"] - lme_state["age_mean_leakfree_from_training"]
    common["female"] = (common["gender"] == "F").astype(int)
    common["year_centered"] = common["year"] - cfg.year_center
    re_m = lme_state["runner_random_effects_leakfree"].reindex(common["display_name"].values)
    lme_pred = np.exp(lme_state["lme_result_object"].predict(common).to_numpy()
                      + re_m["runner_intercept"].to_numpy()
                      + re_m["runner_age_slope"].to_numpy() * common["age_centered"].to_numpy(float)) * lme_state["duan_smearing_factor"]

    obs = common["seconds"].to_numpy(float)
    result = bootstrap_rmse_comparison(obs, ols_hist_pred, lme_pred, common["display_name"].to_numpy(), rng, cfg)
    result.update(n=len(common), ols_history_coverage=_conformal_coverage(obs, ols_hist_pred, pre_state["hist_fit"][2]),
                  lme_coverage=_conformal_coverage(obs, lme_pred, lme_state["q_hat_lme"]))
    return result


def evaluate_information_decay(checkpoint_rmses, rng, full_df, cfg):
    splits = add_prior_boston_history_features(
        full_df[~full_df["age_imputed"] & full_df["age"].notna() & (full_df["year"] >= cfg.analysis_start_year)].copy()
    ).dropna(subset=cfg.cumulative_split_time_columns)
    splits["female"] = (splits["gender"] == "F").astype(int)
    train, test = splits[splits["year"].isin(cfg.in_race_train_years)].copy(), splits[splits["year"] == cfg.in_race_test_year].copy()
    y_tr, y_te, clusters = train["seconds"].to_numpy(float), test["seconds"].to_numpy(float), test["display_name"].to_numpy()
    rng_a, rng_b = [r.spawn(len(cfg.checkpoint_labels)) for r in rng.spawn(2)]

    tr_hist, te_hist = train[train["prior_mean_time"].notna()].copy(), test[test["prior_mean_time"].notna()].copy()
    y_tr_h, y_te_h, clusters_h = tr_hist["seconds"].to_numpy(float), te_hist["seconds"].to_numpy(float), te_hist["display_name"].to_numpy()

    demo_rows = []
    for ci, label in enumerate(cfg.checkpoint_labels):
        sf = cfg.cumulative_split_time_columns[:ci + 1]
        df_ = sf + ["age", "female"]
        alpha = RidgeCV(cv=None).fit(train[sf].to_numpy(float), y_tr).alpha_
        p_a = Ridge(alpha=alpha).fit(train[sf].to_numpy(float), y_tr).predict(test[sf].to_numpy(float))
        p_b = Ridge(alpha=alpha).fit(train[df_].to_numpy(float), y_tr).predict(test[df_].to_numpy(float))
        c = bootstrap_rmse_comparison(y_te, p_a, p_b, clusters, rng_a[ci], cfg)
        demo_rows.append({"checkpoint": label, "splits_only_rmse": c["rmse_a"], "splits_demo_rmse": c["rmse_b"],
                           "demo_contribution": c["delta"], "ci_lo": c["delta_ci"][0], "ci_hi": c["delta_ci"][1]})

    history_rows = []
    for ci, label in enumerate(cfg.checkpoint_labels):
        bf = cfg.cumulative_split_time_columns[:ci + 1] + ["age", "female"]
        hf = bf + ["prior_mean_time", "prior_appearances"]
        alpha = RidgeCV(cv=None).fit(tr_hist[bf].to_numpy(float), y_tr_h).alpha_
        p_a = Ridge(alpha=alpha).fit(tr_hist[bf].to_numpy(float), y_tr_h).predict(te_hist[bf].to_numpy(float))
        p_b = Ridge(alpha=alpha).fit(tr_hist[hf].to_numpy(float), y_tr_h).predict(te_hist[hf].to_numpy(float))
        c = bootstrap_rmse_comparison(y_te_h, p_a, p_b, clusters_h, rng_b[ci], cfg)
        history_rows.append({"checkpoint": label, "without_history_rmse": c["rmse_a"], "with_history_rmse": c["rmse_b"],
                              "delta": c["delta"], "ci_lo": c["delta_ci"][0], "ci_hi": c["delta_ci"][1]})

    rdf = checkpoint_rmses[checkpoint_rmses["variant"] == "no_runner_history"].set_index("checkpoint")
    marginal_rows = [{"checkpoint": label, "rmse": rdf.loc[label, "rmse_seconds"],
                       "marginal_gain": rdf.loc[cfg.checkpoint_labels[i-1], "rmse_seconds"] - rdf.loc[label, "rmse_seconds"] if i > 0 else np.nan,
                       "q_hat": rdf.loc[label, "q_hat"], "interval_width": 2 * rdf.loc[label, "q_hat"]}
                      for i, label in enumerate(cfg.checkpoint_labels)]

    return {"demographic_decay": pd.DataFrame(demo_rows), "history_crossover": pd.DataFrame(history_rows),
            "marginal_value": pd.DataFrame(marginal_rows), "n_test": len(test), "n_test_with_history": len(te_hist)}


def run_pacing_analysis(split_df, cfg):
    pdf = split_df[split_df["half_seconds"].notna() & split_df["seconds"].notna()].copy().reset_index(drop=True)
    pdf["pace_ratio"] = (pdf["seconds"] - pdf["half_seconds"]) / pdf["half_seconds"]
    tol = cfg.pacing_even_tolerance
    pdf["pace_group"] = pd.cut(pdf["pace_ratio"], bins=[-np.inf, 1 - tol, 1 + tol, np.inf],
                                labels=["negative", "even", "positive"])
    groups = ["negative", "even", "positive"]
    group_data = [pdf[pdf["pace_group"] == g]["seconds"].to_numpy() for g in groups]
    h_stat, p_val = kruskal(*group_data)
    dunn = sp.posthoc_dunn(pdf, val_col="seconds", group_col="pace_group", p_adjust="bonferroni")
    pct = pd.crosstab(pdf["gender"], pdf["pace_group"], normalize="index")

    return {"n": len(pdf),
            "distribution": {g: pdf["pace_group"].value_counts().get(g, 0) / len(pdf) for g in groups},
            "median_seconds": {g: pdf[pdf["pace_group"] == g]["seconds"].median() for g in groups},
            "kruskal_wallis_H": h_stat, "kruskal_wallis_p": p_val,
            "eta_squared": pg.anova(data=pdf, dv="seconds", between="pace_group")["np2"][0],
            "dunn_posthoc": dunn.to_dict(),
            "all_pairwise_significant": bool((dunn.to_numpy()[np.triu_indices(3, k=1)] < 0.05).all()),
            "gender_pacing_pct": {gender: {g: pct.loc[gender, g] for g in groups if g in pct.columns} for gender in pct.index}}


def run_weather_analysis(full_df, weather_df, cfg):
    df = full_df[full_df["age"].notna() & (full_df["year"] >= cfg.analysis_start_year)].copy()
    df["age_group"] = pd.cut(df["age"], bins=cfg.weather_age_group_bins, labels=cfg.weather_age_group_labels, right=False)
    yearly = (df.groupby(["year", "age_group", "gender"], observed=True)["seconds"].mean().reset_index()
              .merge(weather_df[["year", "max_temp_f", "mean_humidity", "mean_wind_mph"]], on="year", how="inner"))

    def _mwu(cond_col, thresh_col, alt, lbl_t, lbl_f):
        yearly[cond_col] = yearly[thresh_col] >= weather_df[thresh_col].median()
        results = []
        for g in ["M", "F"]:
            for ag in cfg.weather_age_group_labels:
                sub = yearly[(yearly["gender"] == g) & (yearly["age_group"] == ag)]
                tv, fv = sub[sub[cond_col]]["seconds"], sub[~sub[cond_col]]["seconds"]
                if len(tv) < 3 or len(fv) < 3:
                    continue
                u, p = mannwhitneyu(tv, fv, alternative=alt)
                row = {"gender": g, "age_group": ag, f"n_{lbl_t}": len(tv), f"n_{lbl_f}": len(fv),
                       f"mean_{lbl_t}": tv.mean(), f"mean_{lbl_f}": fv.mean(),
                       "diff_seconds": tv.mean() - fv.mean(), "U": u, "p": p}
                if cond_col == "warm":
                    row.update(diff_median_seconds=tv.median() - fv.median(),
                               diff_pct=(tv.mean() - fv.mean()) / fv.mean() * 100)
                results.append(row)
        return results

    mwu = _mwu("warm", "max_temp_f", "greater", "warm", "cool")
    mwu_hum = _mwu("humid", "mean_humidity", "two-sided", "humid", "dry")
    mwu_wind = _mwu("windy", "mean_wind_mph", "two-sided", "windy", "calm")

    reg = [{"gender": g, "age_group": ag, "n": len(sub), "slope_sec_per_f": r.slope,
             "slope_min_per_f": r.slope / 60, "r_value": r.rvalue, "p_value": r.pvalue, "std_err": r.stderr}
            for g in ["M", "F"] for ag in cfg.weather_age_group_labels
            if len(sub := yearly[(yearly["gender"] == g) & (yearly["age_group"] == ag)]) >= 5
            and (r := linregress(sub["max_temp_f"], sub["seconds"]))]

    oy = df.groupby("year")["seconds"].mean().reset_index().merge(weather_df[["year", "max_temp_f"]], on="year", how="inner")
    sp_rho, sp_p = spearmanr(oy["max_temp_f"], oy["seconds"])
    tr_mean = weather_df[weather_df["year"].isin(cfg.in_race_train_years)]["max_temp_f"].mean()
    te_w = weather_df[weather_df["year"] == cfg.in_race_test_year]
    te_max = te_w["max_temp_f"].iloc[0] if len(te_w) else np.nan

    return {"temp_threshold_f": weather_df["max_temp_f"].median(),
            "mwu_by_age_gender": mwu, "n_significant_p05": sum(1 for r in mwu if r["p"] < 0.05), "n_tests": len(mwu),
            "mwu_humidity_by_age_gender": mwu_hum, "n_significant_humidity_p05": sum(1 for r in mwu_hum if r["p"] < 0.05), "n_humidity_tests": len(mwu_hum),
            "mwu_wind_by_age_gender": mwu_wind, "n_significant_wind_p05": sum(1 for r in mwu_wind if r["p"] < 0.05), "n_wind_tests": len(mwu_wind),
            "linear_regression_by_age_gender": reg, "spearman_rho": sp_rho, "spearman_p": sp_p,
            "train_test_temp_gap": {"train_mean_max_f": tr_mean, "test_max_f": te_max, "gap_f": te_max - tr_mean}}


def evaluate_expanded_model_comparison(demo_fit, hist_fit, train_mean, fitted_ridge, lgbm_cp, rng, test_df, cfg):
    obs, n = test_df["seconds"].to_numpy(float), len(test_df)
    _, lgbm_fitted = lgbm_cp
    ols_demo_pred = np.exp(demo_fit[0].predict(test_df).to_numpy()) * demo_fit[1]
    ols_hist_pred = np.exp(hist_fit[0].predict(test_df).to_numpy()) * hist_fit[1]
    ridge_5k, ridge_40k = fitted_ridge[("5K", "no_runner_history")], fitted_ridge[("40K", "no_runner_history")]
    lgbm_5k, lgbm_40k = lgbm_fitted[("5K", "lgbm_pacing")], lgbm_fitted[("40K", "lgbm_pacing")]

    preds = [np.full(n, train_mean), ols_demo_pred, np.where(test_df["prior_mean_time"].notna(), ols_hist_pred, ols_demo_pred),
              ridge_5k["ridge_regression_model"].predict(test_df[ridge_5k["features"]].to_numpy()),
              lgbm_5k["model"].predict(test_df[lgbm_5k["features"]].to_numpy(float)),
              ridge_40k["ridge_regression_model"].predict(test_df[ridge_40k["features"]].to_numpy()),
              lgbm_40k["model"].predict(test_df[lgbm_40k["features"]].to_numpy(float))]
    pt, lo, hi = compute_bca_cluster_bootstrap_rmse(obs, preds, test_df["display_name"].to_numpy(), rng, cfg)
    covs = [np.nan, np.nan, np.nan, _conformal_coverage(obs, preds[3], ridge_5k["q_hat"]),
             _conformal_coverage(obs, preds[4], lgbm_5k["q_hat"]), _conformal_coverage(obs, preds[5], ridge_40k["q_hat"]),
             _conformal_coverage(obs, preds[6], lgbm_40k["q_hat"])]
    qs = [np.nan, np.nan, np.nan, ridge_5k["q_hat"], lgbm_5k["q_hat"], ridge_40k["q_hat"], lgbm_40k["q_hat"]]
    bi = {"LGBM_5K_Splits_Pacing_Latent": 3, "LGBM_40K_Splits_Pacing_Latent": 5}

    rows = []
    for i, (stage, family, desc) in enumerate(EXPANDED_STAGES):
        row = {"stage": stage, "model_family": family, "description": desc, "n": n,
               "rmse_seconds": pt[i], "rmse_ci_lower": lo[i], "rmse_ci_upper": hi[i],
               "empirical_coverage": covs[i], "q_hat": qs[i]}
        if stage in bi:
            j = bi[stage]
            row.update(delta_vs_baseline=pt[j] - pt[i], delta_ci_lower=lo[j] - hi[i], delta_ci_upper=hi[j] - lo[i])
        else:
            row.update(delta_vs_baseline=np.nan, delta_ci_lower=np.nan, delta_ci_upper=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def verify_objectives(ablation_df, expanded_abl_df, personalization, ridge_5k_re_comp, ns_comp, decay, lme_state, irace_state):
    abl = ablation_df.set_index("stage")
    lgbm = expanded_abl_df[expanded_abl_df["model_family"] == "nonlinear"].set_index("stage")
    ci_nz = lambda lo, hi: (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    dd = decay["demographic_decay"].set_index("checkpoint")
    hc = decay["history_crossover"].set_index("checkpoint")
    mv = decay["marginal_value"].set_index("checkpoint")
    q5k = irace_state["models"][("5K", "no_runner_history")]["q_hat"]

    return {
        "phase1": {
            "demographics_reduces_error": bool(
                (abl.loc["Sample_Mean", "rmse_seconds"] - abl.loc["OLS_Log_Quadratic_Demographics_HC3", "rmse_seconds"]) > (abl.loc["OLS_Log_Quadratic_Demographics_HC3", "rmse_seconds"] - abl.loc["OLS_Log_Quadratic_Demographics_History_HC3", "rmse_seconds"])
                and ci_nz(abl.loc["OLS_Log_Quadratic_Demographics_HC3", "improvement_ci_lower"], abl.loc["OLS_Log_Quadratic_Demographics_HC3", "improvement_ci_upper"])),
            "history_adds_gain": bool(
                (abl.loc["OLS_Log_Quadratic_Demographics_HC3", "rmse_seconds"] - abl.loc["OLS_Log_Quadratic_Demographics_History_HC3", "rmse_seconds"]) > 0
                and ci_nz(abl.loc["OLS_Log_Quadratic_Demographics_History_HC3", "improvement_ci_lower"], abl.loc["OLS_Log_Quadratic_Demographics_History_HC3", "improvement_ci_upper"])),
            "personalization_helps_repeat": bool(personalization["delta"] > 0 and ci_nz(*personalization["delta_ci"])),
        },
        "phase2": {
            "5k_beats_all_prerace": bool(
                abl.loc["Ridge_GCV_5K_Splits_Demographics", "rmse_seconds"] < abl.loc["OLS_Log_Quadratic_Demographics_History_HC3", "rmse_seconds"] < abl.loc["OLS_Log_Quadratic_Demographics_HC3", "rmse_seconds"]
                and ci_nz(abl.loc["Ridge_GCV_5K_Splits_Demographics", "improvement_ci_lower"], abl.loc["Ridge_GCV_5K_Splits_Demographics", "improvement_ci_upper"])),
            "conformal_crossover_5k": bool(q5k < lme_state["q_hat_lme"]),
            "splits_work_for_never_seen": bool(ns_comp["delta"] < 0 and ci_nz(*ns_comp["delta_ci"])),
            "lgbm_gains_significant_5k": bool(lgbm.loc["LGBM_5K_Splits_Pacing_Latent", "delta_vs_baseline"] > 0 and lgbm.loc["LGBM_5K_Splits_Pacing_Latent", "delta_ci_lower"] > 0) if "LGBM_5K_Splits_Pacing_Latent" in lgbm.index else None,
            "lgbm_gains_significant_40k": bool(lgbm.loc["LGBM_40K_Splits_Pacing_Latent", "delta_vs_baseline"] > 0 and lgbm.loc["LGBM_40K_Splits_Pacing_Latent", "delta_ci_lower"] > 0) if "LGBM_40K_Splits_Pacing_Latent" in lgbm.index else None,
            "lgbm_tighter_conformal_interval_5k": bool(lgbm.loc["LGBM_5K_Splits_Pacing_Latent", "q_hat"] < q5k) if "LGBM_5K_Splits_Pacing_Latent" in lgbm.index else False,
            "lgbm_tighter_conformal_interval_40k": bool(lgbm.loc["LGBM_40K_Splits_Pacing_Latent", "q_hat"] < abl.loc["Ridge_GCV_40K_Splits_Demographics", "q_hat"]) if "LGBM_40K_Splits_Pacing_Latent" in lgbm.index else False,
        },
        "phase3": {
            "demographics_decay": bool(dd.loc["40K", "demo_contribution"] < dd.loc["5K", "demo_contribution"]),
            "history_decay": bool(hc.loc["5K", "delta"] > 0 and hc.loc["HALF", "ci_hi"] < hc.loc["5K", "ci_lo"]),
            "conformal_interval_monotonic_decay": bool(mv.loc[["25K", "30K", "35K", "40K"], "interval_width"].is_monotonic_decreasing),
        },
    }

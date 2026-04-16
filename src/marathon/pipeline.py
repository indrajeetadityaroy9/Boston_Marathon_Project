import numpy as np
import pandas as pd
from . import data, evaluation, models, reporting, visualization
from .inference import bootstrap_rmse_comparison
from .config import MarathonConfig


def run_pipeline():
    cfg = MarathonConfig()
    fdf, wdf = data.load_processed_results(cfg), data.fetch_race_day_weather(cfg)

    raw = fdf[["year", "display_name", "age", "gender", "seconds", "age_imputed"]].copy()
    filtered = raw[~raw["age_imputed"] & raw["age"].notna() & (raw["year"] >= cfg.analysis_start_year)].copy()
    pre = data.add_prior_boston_history_features(filtered.assign(log_seconds=np.log(filtered["seconds"].to_numpy())))
    train_pre = pre[pre["year"].isin(cfg.pre_race_train_years)].copy()
    age_mean = train_pre["age"].mean()
    data.add_centered_pre_race_features(train_pre, age_mean, cfg)
    repeat_df = data.build_repeat_runner_analysis_sample(raw, cfg)

    streams = np.random.default_rng(cfg.seed).spawn(9)
    rngs = {"never_seen": streams[1], "ridge_5k_re": streams[2], "ablation": streams[3],
            "personalization": streams[4], "lgbm_checkpoint": streams[6],
            "expanded_ablation": streams[7], "decay": streams[8]}

    # Phase 1: Pre-race models
    demo_fit = models.fit_hc3_robust_log_seconds_regression(train_pre, cfg.pre_race_fixed_effect_features, cfg)
    hist_fit = models.fit_hc3_robust_log_seconds_regression(
        train_pre[train_pre["prior_mean_time"].notna()].copy(),
        cfg.pre_race_fixed_effect_features + ["log1p_prior_appearances", "prior_mean_time"], cfg)
    lme_state = models.fit_temporal_holdout_runner_mixed_effects(repeat_df, cfg)

    # Phase 2: In-race checkpoint models
    t_raw, te_raw = data.load_in_race_split_dataset(lme_state["runner_random_effects_leakfree"], fdf, wdf, cfg)
    train_df, test_df = data.compute_latent_information_features(
        data.compute_segment_pacing_features(t_raw, cfg), data.compute_segment_pacing_features(te_raw, cfg), cfg)
    cp_df, ridge_fitted = models.fit_checkpoint_ridge_models(train_df, test_df, cfg)
    lgbm_cp_df, lgbm_fitted = models.fit_checkpoint_lgbm_models(train_df, test_df, rngs["lgbm_checkpoint"], cfg)

    # Nested model comparison
    common_eval = evaluation._prepare_common_evaluation_data(fdf, age_mean, wdf, cfg)
    abl = evaluation.evaluate_nested_model_comparison(demo_fit, hist_fit, train_pre["seconds"].mean(),
                                                       ridge_fitted, rngs["ablation"], common_eval, cfg)
    personalization = evaluation.compare_personalization(
        {"pool": pre, "age_mean": age_mean, "hist_fit": hist_fit}, lme_state,
        test_df[test_df["runner_intercept"].notna()].copy(), rngs["personalization"], cfg)

    # Ridge 5K runner random effects contribution
    td = test_df.copy()
    m5k, m5k_h = ridge_fitted[("5K", "no_runner_history")], ridge_fitted[("5K", "with_runner_history")]
    td["ridge_pred"] = m5k["ridge_regression_model"].predict(td[m5k["features"]].to_numpy())
    holders = td[td["runner_intercept"].notna()].copy()
    holders["ridge_hist_pred"] = m5k_h["ridge_regression_model"].predict(holders[m5k_h["features"]].to_numpy())
    ridge_5k_re_comp = bootstrap_rmse_comparison(holders["seconds"].to_numpy(float), holders["ridge_pred"].to_numpy(),
                                      holders["ridge_hist_pred"].to_numpy(), holders["display_name"].to_numpy(), rngs["ridge_5k_re"], cfg)

    # Never-seen runner comparison
    never = td[td["runner_intercept"].isna()].copy()
    never["age_centered"] = never["age"] - lme_state["age_mean_leakfree_from_training"]
    never["marginal_pred"] = np.exp(lme_state["lme_result_object"].predict(never).to_numpy()) * lme_state["duan_smearing_factor"]
    ns_comp = bootstrap_rmse_comparison(never["seconds"].to_numpy(float), never["ridge_pred"].to_numpy(),
                                         never["marginal_pred"].to_numpy(), never["display_name"].to_numpy(), rngs["never_seen"], cfg)

    # Phase 3: Information decay & characterization
    obs = evaluation.evaluate_information_decay(cp_df, rngs["decay"], fdf, cfg)
    split_df = pd.concat([train_df, test_df])
    pacing = evaluation.run_pacing_analysis(split_df, cfg)
    weather = evaluation.run_weather_analysis(fdf, wdf, cfg)
    expanded_abl = evaluation.evaluate_expanded_model_comparison(
        demo_fit, hist_fit, train_pre["seconds"].mean(), ridge_fitted,
        (lgbm_cp_df, lgbm_fitted), rngs["expanded_ablation"], common_eval, cfg)
    verification = evaluation.verify_objectives(abl, expanded_abl, personalization, ridge_5k_re_comp, ns_comp, obs, lme_state, {"models": ridge_fitted})

    # Dataset summary for reporting
    data_summary = {"n_train_pre": len(train_pre), "n_train_inrace": len(train_df),
                    "n_test": len(test_df), "n_distinct_runners": filtered["display_name"].nunique(),
                    "n_repeat_runners": repeat_df["display_name"].nunique()}

    # Output
    reporting.generate_outputs(abl, personalization, ridge_5k_re_comp, ns_comp, {"cp_df": cp_df}, obs, verification, pacing, weather,
                                expanded_abl, {"cp_df": lgbm_cp_df, "fitted": lgbm_fitted}, lme_state, data_summary, cfg)
    visualization.generate_all_figures(pacing, weather, wdf, split_df, fdf, {"cp_df": cp_df}, obs,
                                        {"cp_df": lgbm_cp_df, "fitted": lgbm_fitted}, abl, cfg.figures_dir, cfg)

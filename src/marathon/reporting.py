import json
import pandas as pd

_latex_stage = lambda s: s.replace("_", " ")


def generate_outputs(abl, personalization, ridge_5k_re_comp, ns_comp, irace_state, decay, verification,
                     pacing, weather, expanded_abl, lgbm_checkpoint, lme_state, data_summary, cfg):
    tables_dir = cfg.project_root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    fmt_delta = lambda r: "--" if pd.isna(r["improvement_seconds"]) else f"{r['improvement_seconds']:.1f} [{r['improvement_ci_lower']:.1f}, {r['improvement_ci_upper']:.1f}]"
    fmt_cov = lambda r: "--" if pd.isna(r["empirical_coverage"]) else f"{r['empirical_coverage'] * 100:.1f}"
    fmt_q = lambda r: "--" if pd.isna(r["q_hat"]) else f"{2 * r['q_hat'] / 60:.1f}"

    # Table 1: Nested information hierarchy with BCa CIs and conformal calibration
    with open(tables_dir / "tab_ablation_hierarchy.tex", "w") as f:
        f.write("\\begin{tabular}{llrrrrr}\n\\hline\n"
                "Stage & Description & $N$ & RMSE & $\\Delta$RMSE [95\\% CI] & Coverage (\\%) & $2\\hat{q}$ (min) \\\\\n\\hline\n")
        for _, r in abl.iterrows():
            f.write(f"{_latex_stage(r['stage'])} & {r['description']} & {int(r['n'])} & {r['rmse_seconds']:.1f} & {fmt_delta(r)} & {fmt_cov(r)} & {fmt_q(r)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # Table 2: Conformal crossover — pre-race personalized vs live 5K
    ols_hist_row = abl[abl["stage"] == "OLS_Log_Quadratic_Demographics_History_HC3"].iloc[0]
    ridge_5k_row = abl[abl["stage"] == "Ridge_GCV_5K_Splits_Demographics"].iloc[0]
    with open(tables_dir / "tab_conformal_crossover.tex", "w") as f:
        f.write("\\begin{tabular}{lrrr}\n\\hline\nModel & RMSE & Coverage (\\%) & $2\\hat{q}$ (min) \\\\\n\\hline\n")
        f.write(f"OLS Log-Quadratic Demographics+History (HC3) & {personalization['rmse_a']:.1f} & {personalization['ols_history_coverage']*100:.1f} & {2*ols_hist_row['q_hat']/60:.1f} \\\\\n")
        f.write(f"LME REML Random Intercept+Age Slope by Runner & {personalization['rmse_b']:.1f} & {personalization['lme_coverage']*100:.1f} & {2*lme_state['q_hat_lme']/60:.1f} \\\\\n")
        f.write(f"Ridge GCV 5K Splits+Demographics & {ridge_5k_row['rmse_seconds']:.1f} & {ridge_5k_row['empirical_coverage']*100:.1f} & {2*ridge_5k_row['q_hat']/60:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # Table 3: Model complexity benchmark — Ridge vs LightGBM at 5K and 40K
    with open(tables_dir / "tab_complexity_benchmark.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\hline\n"
                "Checkpoint & Ridge RMSE & LGBM RMSE & $\\Delta$RMSE [95\\% CI] & Ridge $2\\hat{q}$ (min) & LGBM $2\\hat{q}$ (min) \\\\\n\\hline\n")
        for cp, ridge_stage, lgbm_stage in [("5K", "Ridge_GCV_5K_Splits_Demographics", "LGBM_5K_Splits_Pacing_Latent"),
                                             ("40K", "Ridge_GCV_40K_Splits_Demographics", "LGBM_40K_Splits_Pacing_Latent")]:
            ridge_row = expanded_abl[expanded_abl["stage"] == ridge_stage].iloc[0]
            lgbm_row = expanded_abl[expanded_abl["stage"] == lgbm_stage].iloc[0]
            f.write(f"{cp} & {ridge_row['rmse_seconds']:.1f} & {lgbm_row['rmse_seconds']:.1f} "
                    f"& {lgbm_row['delta_vs_baseline']:.1f} [{lgbm_row['delta_ci_lower']:.1f}, {lgbm_row['delta_ci_upper']:.1f}] "
                    f"& {2*ridge_row['q_hat']/60:.1f} & {2*lgbm_row['q_hat']/60:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # Table 4: Personalization comparison — OLS log-quadratic vs LME REML conditional on repeat runners
    p = personalization
    with open(tables_dir / "tab_personalization.tex", "w") as f:
        f.write("\\begin{tabular}{lrr}\n\\hline\nModel & RMSE (s) & 95\\% CI \\\\\n\\hline\n")
        f.write(f"OLS Log-Quadratic Demographics+History (HC3) & {p['rmse_a']:.1f} & [{p['rmse_a_ci'][0]:.1f}, {p['rmse_a_ci'][1]:.1f}] \\\\\n")
        f.write(f"LME REML Conditional (Random Intercept+Age Slope) & {p['rmse_b']:.1f} & [{p['rmse_b_ci'][0]:.1f}, {p['rmse_b_ci'][1]:.1f}] \\\\\n")
        f.write("\\hline\n")
        f.write(f"$\\Delta$ RMSE & +{p['delta']:.1f} & [+{p['delta_ci'][0]:.1f}, +{p['delta_ci'][1]:.1f}] \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # Table 5: Information takeover — hard switch at 5K checkpoint
    cp_5k = irace_state["cp_df"]
    ridge_5k_re_rmse = cp_5k[(cp_5k["checkpoint"] == "5K") & (cp_5k["variant"] == "with_runner_history")].iloc[0]["rmse_seconds"]
    n_holders = int(cp_5k[(cp_5k["checkpoint"] == "5K") & (cp_5k["variant"] == "with_runner_history")].iloc[0]["n"])
    n_checkpoint_test = int(cp_5k[(cp_5k["checkpoint"] == "5K") & (cp_5k["variant"] == "no_runner_history")].iloc[0]["n"])
    n_full = int(abl.iloc[0]["n"])
    n_never = n_checkpoint_test - n_holders
    ols_hist_rmse = abl[abl["stage"] == "OLS_Log_Quadratic_Demographics_History_HC3"].iloc[0]["rmse_seconds"]
    ridge_5k_full_rmse = abl[abl["stage"] == "Ridge_GCV_5K_Splits_Demographics"].iloc[0]["rmse_seconds"]
    with open(tables_dir / "tab_hard_switch.tex", "w") as f:
        f.write("\\begin{tabular}{llrr}\n\\hline\nModel & Population & $n$ & RMSE (s) \\\\\n\\hline\n")
        f.write(f"Best pre-race (OLS, HC3) & All & {n_full:,} & {ols_hist_rmse:.1f} \\\\\n")
        f.write(f"Ridge 5K+Demographics & All & {n_full:,} & {ridge_5k_full_rmse:.1f} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Ridge 5K+Demographics & Repeat runners & {n_holders:,} & {ridge_5k_re_comp['rmse_a']:.1f} \\\\\n")
        f.write(f"Ridge 5K + Runner RE & Repeat runners & {n_holders:,} & {ridge_5k_re_rmse:.1f} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Ridge 5K+Demographics & Never-seen & {n_never:,} & {ns_comp['rmse_a']:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # Table 6: Dataset summary
    ds, wg = data_summary, weather["train_test_temp_gap"]
    with open(tables_dir / "tab_data_summary.tex", "w") as f:
        f.write("\\begin{tabular}{lr}\n\\hline\nProperty & Value \\\\\n\\hline\n")
        f.write(f"Pre-race training rows & {ds['n_train_pre']:,} \\\\\n")
        f.write(f"In-race training rows & {ds['n_train_inrace']:,} \\\\\n")
        f.write(f"Test rows & {ds['n_test']:,} \\\\\n")
        f.write(f"Test year & {cfg.in_race_test_year} \\\\\n")
        f.write(f"Distinct runners & {ds['n_distinct_runners']:,} \\\\\n")
        f.write(f"Repeat runners & {ds['n_repeat_runners']:,} \\\\\n")
        f.write(f"Train mean max temp & {wg['train_mean_max_f']:.1f}°F \\\\\n")
        f.write(f"Test max temp & {wg['test_max_f']:.1f}°F \\\\\n")
        f.write(f"Temperature gap & +{wg['gap_f']:.1f}°F \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # Table 7: Information decay — demographic and history ΔRMSE across checkpoints
    d_df, h_df = decay["demographic_decay"], decay["history_crossover"]
    with open(tables_dir / "tab_information_decay.tex", "w") as f:
        f.write("\\begin{tabular}{lrr}\n\\hline\n"
                "Checkpoint & Demo $\\Delta$RMSE [95\\% CI] & History $\\Delta$RMSE [95\\% CI] \\\\\n\\hline\n")
        for (_, dr), (_, hr) in zip(d_df.iterrows(), h_df.iterrows()):
            f.write(f"{dr['checkpoint']} & {dr['demo_contribution']:.1f} [{dr['ci_lo']:.1f}, {dr['ci_hi']:.1f}] "
                    f"& {hr['delta']:.1f} [{hr['ci_lo']:.1f}, {hr['ci_hi']:.1f}] \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # Table 8: LME REML variance decomposition and shrinkage
    ni = lme_state["shrinkage_ni_summary"]
    with open(tables_dir / "tab_mixed_effects.tex", "w") as f:
        f.write("\\begin{tabular}{lr}\n\\hline\nMetric & Value \\\\\n\\hline\n")
        f.write(f"Fixed-effects $R^2$ & {lme_state['variance_explained_by_fixed_effects_only']:.3f} \\\\\n")
        f.write(f"Fixed + Random $R^2$ & {lme_state['variance_explained_by_fixed_and_random_effects']:.3f} \\\\\n")
        f.write(f"Shrinkage (RE predictor / raw variance ratio) & {lme_state['shrinkage_re_predictor_vs_raw_var_ratio']:.3f} \\\\\n")
        f.write(f"Appearances per runner (median [IQR]) & {ni['median']:.0f} [{ni['q1']:.0f}, {ni['q3']:.0f}] \\\\\n")
        f.write(f"Marginal RMSE (known) & {lme_state['marginal_mixed_effects_rmse_on_test_known']:.1f} \\\\\n")
        f.write(f"Conditional RMSE (known) & {lme_state['conditional_mixed_effects_rmse_on_test_known']:.1f} \\\\\n")
        f.write(f"Marginal RMSE (never-seen) & {lme_state['marginal_mixed_effects_rmse_on_never_seen']:.1f} \\\\\n")
        f.write(f"$2\\hat{{q}}_{{\\text{{LME}}}}$ (min) & {2 * lme_state['q_hat_lme'] / 60:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # Pipeline metrics JSON — single source of truth for all numerical results
    metrics = {"verification": verification, "ablation": abl.to_dict("records"),
               "checkpoint_rmse": irace_state["cp_df"].to_dict("records"),
               "personalization": personalization, "ridge_5k_runner_random_effects_contribution": ridge_5k_re_comp,
               "never_seen_comparison": ns_comp,
               "phase3_demographic_decay": decay["demographic_decay"].to_dict("records"),
               "phase3_history_crossover": decay["history_crossover"].to_dict("records"),
               "phase3_marginal_value": decay["marginal_value"].to_dict("records"),
               "pacing_analysis": pacing, "weather_analysis": weather,
               "expanded_ablation": expanded_abl.to_dict("records"),
               "lgbm_checkpoint": lgbm_checkpoint["cp_df"].to_dict("records"),
               "data_summary": data_summary,
               "lme_variance_decomposition": {
                   "variance_explained_by_fixed_effects_only": lme_state["variance_explained_by_fixed_effects_only"],
                   "variance_explained_by_fixed_and_random_effects": lme_state["variance_explained_by_fixed_and_random_effects"],
                   "shrinkage_re_predictor_vs_raw_var_ratio": lme_state["shrinkage_re_predictor_vs_raw_var_ratio"],
                   "shrinkage_ni_summary": lme_state["shrinkage_ni_summary"],
                   "marginal_mixed_effects_rmse_on_test_known": lme_state["marginal_mixed_effects_rmse_on_test_known"],
                   "conditional_mixed_effects_rmse_on_test_known": lme_state["conditional_mixed_effects_rmse_on_test_known"],
                   "marginal_mixed_effects_rmse_on_never_seen": lme_state["marginal_mixed_effects_rmse_on_never_seen"],
                   "q_hat_lme": lme_state["q_hat_lme"],
               },
               "config": cfg.to_dict()}
    with open(cfg.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2, default=float)

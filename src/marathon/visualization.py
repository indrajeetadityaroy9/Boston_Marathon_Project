import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Global AISTATS publication standard
# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011, Nature Methods) + consistent sizing
_PALETTE = {
    "blue":     "#0072B2",
    "orange":   "#E69F00",
    "green":    "#009E73",
    "red":      "#D55E00",
    "purple":   "#CC79A7",
    "skyblue":  "#56B4E9",
    "yellow":   "#F0E442",
    "teal":     "#1B9E77",
}

# Pacing group colors (consistent across distribution, boxplot, gender)
_PACE_COLORS = [_PALETTE["green"], _PALETTE["blue"], _PALETTE["red"]]

# Model comparison colors
_PRE_RACE_COLOR = _PALETTE["blue"]       # pre-race models (Sample Mean, OLS)
_IN_RACE_COLOR  = _PALETTE["red"]        # in-race models (Ridge)
_RIDGE_COLOR    = _PALETTE["blue"]       # ridge series in line plots
_LGBM_COLOR     = _PALETTE["purple"]     # LightGBM series

# Significance encoding
_SIG_COLOR      = _PALETTE["red"]        # p < 0.05
_NONSIG_COLOR   = _PALETTE["skyblue"]    # p >= 0.05

# Weather panel colors
_TEMP_MAX_COLOR  = _PALETTE["red"]
_TEMP_MEAN_COLOR = _PALETTE["orange"]
_HUMIDITY_COLOR  = _PALETTE["blue"]
_WIND_COLOR      = _PALETTE["teal"]

# Regression grid
_SCATTER_COLOR   = _PALETTE["blue"]
_REGLINE_COLOR   = _PALETTE["red"]

_FIG_SINGLE = (6.0, 4.0)    # single-panel figure
_FIG_WIDE   = (7.5, 3.2)    # wide multi-panel (weather trends)
_PAD        = 0.15           # savefig padding

_RC = {
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": _PAD,
}


def _save(fig, ax, path, title, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core contribution figures
# ---------------------------------------------------------------------------

def plot_conformal_interval_width_by_checkpoint(irace_state, outpath, cfg):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        cp = pd.DataFrame(irace_state["cp_df"]).query("variant == 'no_runner_history'")
        ax.plot(np.arange(len(cp)), cp["q_hat"].to_numpy() * 2 / 60.0,
                "o-", color=_RIDGE_COLOR, linewidth=2.5, markersize=6)
        ax.set_xticks(np.arange(len(cp)))
        ax.set_xticklabels(cp["checkpoint"].tolist(), rotation=45, ha="right")
        ax.grid(True, linestyle="--", alpha=0.4)
        _save(fig, ax, outpath, "Conformal Prediction Interval Width by Checkpoint",
              "90% Prediction Interval Width (min)")


def plot_information_decay(obs_decay, outpath, cfg):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        d_df = pd.DataFrame(obs_decay["demographic_decay"])
        h_df = pd.DataFrame(obs_decay["history_crossover"])
        x = np.arange(len(d_df))
        ax.plot(x, d_df["demo_contribution"].to_numpy() / 60.0,
                "o-", color=_PALETTE["blue"], linewidth=2.5, markersize=6, label="Demographics")
        ax.plot(x, h_df["delta"].to_numpy() / 60.0,
                "s-", color=_PALETTE["red"], linewidth=2.5, markersize=6, label="Prior History")
        ax.axhline(0, color=_PALETTE["skyblue"], linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(d_df["checkpoint"].tolist(), rotation=45, ha="right")
        ax.legend(loc="upper right", frameon=True, edgecolor="white", fancybox=False)
        ax.grid(True, linestyle="--", alpha=0.4)
        _save(fig, ax, outpath, "Marginal Information Decay",
              "Marginal Value (min RMSE reduction)")


def plot_function_class_convergence(irace_state, lgbm_checkpoint, outpath, cfg):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        r_df = pd.DataFrame(irace_state["cp_df"]).query("variant == 'no_runner_history'").set_index("checkpoint")
        l_df = pd.DataFrame(lgbm_checkpoint["cp_df"]).set_index("checkpoint")
        x, labels = np.arange(len(cfg.checkpoint_labels)), cfg.checkpoint_labels
        ridge_vals = [r_df.loc[l, "q_hat"] * 2 / 60.0 if l in r_df.index else np.nan for l in labels]
        lgbm_vals  = [l_df.loc[l, "q_hat"] * 2 / 60.0 if l in l_df.index else np.nan for l in labels]
        ax.plot(x, ridge_vals, "o-", color=_RIDGE_COLOR, linewidth=2.5, markersize=6, label="Ridge (Linear)")
        ax.plot(x, lgbm_vals, "x--", color=_LGBM_COLOR, linewidth=2.5, markersize=7, label="LightGBM (Non-linear)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(loc="upper right", frameon=True, edgecolor="white", fancybox=False)
        ax.grid(True, linestyle="--", alpha=0.4)
        _save(fig, ax, outpath, "Function Class Convergence",
              "90% Prediction Interval Width (min)")


def plot_ablation_waterfall(abl, outpath, cfg):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        stages = abl["stage"].tolist()
        rmses  = abl["rmse_seconds"].to_numpy()
        ci_lo  = abl["rmse_ci_lower"].to_numpy()
        ci_hi  = abl["rmse_ci_upper"].to_numpy()
        y = np.arange(len(stages))[::-1]
        colors = [_PRE_RACE_COLOR if s in (
            "Sample_Mean", "OLS_Log_Quadratic_Demographics_HC3",
            "OLS_Log_Quadratic_Demographics_History_HC3"
        ) else _IN_RACE_COLOR for s in stages]
        for yi, xi, lo, hi, c in zip(y, rmses, ci_lo, ci_hi, colors):
            ax.errorbar(xi, yi, xerr=[[xi - lo], [hi - xi]], fmt="o", color=c, ecolor=c,
                        elinewidth=2, capsize=5, capthick=2, markersize=8, zorder=3)
        display_labels = [s.replace("_", " ") for s in stages]
        ax.set_yticks(y)
        ax.set_yticklabels(display_labels, fontsize=9)
        ols_hist_rmse  = rmses[stages.index("OLS_Log_Quadratic_Demographics_History_HC3")]
        ridge_5k_rmse  = rmses[stages.index("Ridge_GCV_5K_Splits_Demographics")]
        ols_hist_y     = y[stages.index("OLS_Log_Quadratic_Demographics_History_HC3")]
        ridge_5k_y     = y[stages.index("Ridge_GCV_5K_Splits_Demographics")]
        mid_y = (ols_hist_y + ridge_5k_y) / 2
        ax.annotate("", xy=(ridge_5k_rmse, ridge_5k_y - 0.15),
                    xytext=(ols_hist_rmse, ols_hist_y + 0.15),
                    arrowprops=dict(arrowstyle="<->", color=_PALETTE["orange"], lw=1.5))
        ax.text((ols_hist_rmse + ridge_5k_rmse) / 2, mid_y,
                f"$\\Delta$={ols_hist_rmse - ridge_5k_rmse:.0f}s",
                ha="center", va="center", fontsize=9, color=_PALETTE["orange"],
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        ax.set_xlabel("RMSE (seconds)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        _save(fig, ax, outpath, "Nested Information Hierarchy", "")


def plot_checkpoint_rmse(irace_state, lgbm_checkpoint, outpath, cfg):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        r_df = pd.DataFrame(irace_state["cp_df"]).query("variant == 'no_runner_history'").set_index("checkpoint")
        l_df = pd.DataFrame(lgbm_checkpoint["cp_df"]).set_index("checkpoint")
        x, labels = np.arange(len(cfg.checkpoint_labels)), cfg.checkpoint_labels
        ridge_vals = [r_df.loc[l, "rmse_seconds"] if l in r_df.index else np.nan for l in labels]
        lgbm_vals  = [l_df.loc[l, "rmse_seconds"] if l in l_df.index else np.nan for l in labels]
        ax.plot(x, ridge_vals, "o-", color=_RIDGE_COLOR, linewidth=2.5, markersize=6, label="Ridge (Linear)")
        ax.plot(x, lgbm_vals, "x--", color=_LGBM_COLOR, linewidth=2.5, markersize=7, label="LightGBM (Non-linear)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(loc="upper right", frameon=True, edgecolor="white", fancybox=False)
        ax.grid(True, linestyle="--", alpha=0.4)
        _save(fig, ax, outpath, "Checkpoint RMSE: Ridge vs LightGBM", "RMSE (seconds)")


# ---------------------------------------------------------------------------
# Pacing characterization figures
# ---------------------------------------------------------------------------

def plot_pacing_distribution(pacing, outpath):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        groups = ["negative", "even", "positive"]
        vals = [pacing["distribution"][g] * 100 for g in groups]
        bars = ax.bar([g.capitalize() for g in groups], vals,
                      color=_PACE_COLORS, edgecolor="white", width=0.5)
        for b, g in zip(bars, groups):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{pacing['distribution'][g]*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Pacing Group")
        _save(fig, ax, outpath, "Pacing Strategy Distribution", "% of Runners")


def plot_pacing_boxplot(split_df, outpath, cfg):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        pdf = split_df[split_df["half_seconds"].notna() & split_df["seconds"].notna()].copy()
        tol = cfg.pacing_even_tolerance
        pdf["pace_group"] = pd.cut(
            (pdf["seconds"] - pdf["half_seconds"]) / pdf["half_seconds"],
            bins=[-np.inf, 1 - tol, 1 + tol, np.inf],
            labels=["Negative", "Even", "Positive"])
        data = [pdf[pdf["pace_group"] == g]["seconds"].to_numpy() / 60
                for g in ["Negative", "Even", "Positive"]]
        bp = ax.boxplot(
            data, labels=["Negative", "Even", "Positive"], patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=_PALETTE["blue"], linewidth=1.2),
            capprops=dict(color=_PALETTE["blue"], linewidth=1.2),
            flierprops=dict(marker=".", markersize=2, alpha=0.3,
                            markerfacecolor=_PALETTE["skyblue"], markeredgecolor="none"),
            boxprops=dict(linewidth=1.2))
        for patch, c in zip(bp["boxes"], _PACE_COLORS):
            patch.set_facecolor(c)
            patch.set_edgecolor(c)
            patch.set_alpha(0.8)
        ax.set_xlabel("Pacing Group")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x // 60)}h{int(x % 60):02d}m"))
        _save(fig, ax, outpath, "Finish Time by Pacing Strategy", "Finish Time")


def plot_pacing_gender(pacing, outpath):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        genders = sorted(pacing["gender_pacing_pct"].keys())
        labels_map = {"F": "Female", "M": "Male"}
        colors_map = {"F": _PALETTE["purple"], "M": _PALETTE["teal"]}
        groups = ["negative", "even", "positive"]
        for i, g in enumerate(genders):
            ax.bar(np.arange(3) + (i - 0.5) * 0.35,
                   [pacing["gender_pacing_pct"][g].get(k, 0) * 100 for k in groups],
                   0.35, label=labels_map.get(g, g),
                   color=colors_map.get(g, _PALETTE["skyblue"]), alpha=0.85, edgecolor="white")
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["Negative", "Even", "Positive"])
        ax.set_xlabel("Pacing Group")
        ax.legend(frameon=True, edgecolor="white", fancybox=False)
        _save(fig, ax, outpath, "Pacing Strategy by Gender", "% of Runners")


# ---------------------------------------------------------------------------
# Weather characterization figures
# ---------------------------------------------------------------------------

def plot_weather_trends(weather_df, outpath):
    with plt.rc_context({**_RC, "figure.figsize": _FIG_WIDE}):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
        yr = weather_df["year"]
        ax1.plot(yr, weather_df["max_temp_f"], "o-", color=_TEMP_MAX_COLOR,
                 markersize=4, linewidth=1.2, label="Max")
        ax1.plot(yr, weather_df["mean_temp_f"], "o-", color=_TEMP_MEAN_COLOR,
                 markersize=4, linewidth=1.2, label="Mean")
        ax1.set_ylabel("Temperature (\u00b0F)")
        ax1.set_title("Temperature")
        ax1.legend(fontsize=9, frameon=True, edgecolor="white", fancybox=False)
        ax2.plot(yr, weather_df["mean_humidity"], "o-", color=_HUMIDITY_COLOR,
                 markersize=4, linewidth=1.2)
        ax2.set_ylabel("Mean Humidity (%)")
        ax2.set_title("Humidity")
        ax3.plot(yr, weather_df["mean_wind_mph"], "o-", color=_WIND_COLOR,
                 markersize=4, linewidth=1.2)
        ax3.set_ylabel("Mean Wind (mph)")
        ax3.set_title("Wind Speed")
        for ax in (ax1, ax2, ax3):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle("Race-Day Weather (10 AM \u2013 6 PM)", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)


def plot_temp_significance_bars(mwu, title, outpath):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        colors = [_SIG_COLOR if r["p"] < 0.05 else _NONSIG_COLOR for r in mwu]
        ax.bar(range(len(mwu)), [r["p"] for r in mwu],
               color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0.05, linestyle="--", color=_PALETTE["orange"], linewidth=1.0,
                   label="$\\alpha = 0.05$")
        ax.set_xticks(range(len(mwu)))
        ax.set_xticklabels([r["age_group"] for r in mwu], rotation=45, ha="right")
        ax.set_xlabel("Age Group")
        ax.set_yscale("log")
        ax.legend(frameon=True, edgecolor="white", fancybox=False)
        _save(fig, ax, outpath, title, "p-value")


def plot_temp_effect_magnitude(mwu, gender_label, outpath):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SINGLE)
        colors = [_SIG_COLOR if r["p"] < 0.05 else _NONSIG_COLOR for r in mwu]
        ax.bar(range(len(mwu)), [r["diff_seconds"] / 60 for r in mwu],
               color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(mwu)))
        ax.set_xticklabels([r["age_group"] for r in mwu], rotation=45, ha="right")
        ax.set_xlabel("Age Group")
        ax.axhline(0, color=_PALETTE["skyblue"], linewidth=0.8, linestyle="--")
        _save(fig, ax, outpath, f"{gender_label}: Temperature Slowdown",
              "Warm \u2212 Cool Diff (min)")


def plot_temp_regression_grid(full_df, weather_df, gender, reg, outpath, cfg):
    yearly = (full_df[full_df["age"].notna() & (full_df["year"] >= cfg.analysis_start_year)]
              .assign(age_group=pd.cut(full_df["age"], bins=cfg.weather_age_group_bins,
                                       labels=cfg.weather_age_group_labels, right=False))
              .groupby(["year", "age_group", "gender"], observed=True)["seconds"]
              .mean().reset_index()
              .merge(weather_df[["year", "max_temp_f"]], on="year", how="inner"))
    yearly = yearly[yearly["gender"] == gender]
    ncols = min(5, len(reg))
    nrows = -(-len(reg) // ncols)

    with plt.rc_context({**_RC, "figure.figsize": (7.5, 3.0 * nrows)}):
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
        flat = axes.flatten()
        for ax, r in zip(flat, reg):
            sub = yearly[yearly["age_group"] == r["age_group"]]
            if sub.empty:
                continue
            ax.scatter(sub["max_temp_f"], sub["seconds"] / 3600, s=18,
                       color=_SCATTER_COLOR, alpha=0.7, edgecolors="none", zorder=2)
            xl = np.linspace(sub["max_temp_f"].min(), sub["max_temp_f"].max(), 50)
            intercept = ((sub["seconds"] / 3600).mean()
                         - r["slope_sec_per_f"] / 3600 * sub["max_temp_f"].mean())
            ax.plot(xl, intercept + r["slope_sec_per_f"] / 3600 * xl,
                    color=_REGLINE_COLOR, linewidth=1.2, zorder=3)
            sig_marker = "*" if r["p_value"] < 0.05 else ""
            ax.set_title(f"{r['age_group']}\n{r['slope_min_per_f']:.2f} min/\u00b0F"
                         f" (p={r['p_value']:.3f}){sig_marker}", fontsize=8)
            ax.grid(True, linewidth=0.3, alpha=0.4)
        for ax in flat[len(reg):]:
            ax.set_visible(False)
        # shared axis labels
        for ax in axes[-1]:
            ax.set_xlabel("Max Temp (\u00b0F)", fontsize=9)
        for ax in axes[:, 0]:
            ax.set_ylabel("Finish Time (hr)", fontsize=9)
        gender_label = "Men" if gender == "M" else "Women"
        fig.suptitle(f"{gender_label}: Finish Time vs Temperature by Age Group",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_figures(pacing, weather, weather_df, split_df, full_df,
                         irace_state, obs_decay, lgbm_checkpoint, abl,
                         output_dir, cfg):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_conformal_interval_width_by_checkpoint(
        irace_state, output_dir / "fig_conformal_interval_width.pdf", cfg)
    plot_information_decay(obs_decay, output_dir / "fig_information_decay.pdf", cfg)
    plot_function_class_convergence(
        irace_state, lgbm_checkpoint,
        output_dir / "fig_function_class_convergence.pdf", cfg)
    plot_ablation_waterfall(abl, output_dir / "fig_ablation_waterfall.pdf", cfg)
    plot_checkpoint_rmse(
        irace_state, lgbm_checkpoint, output_dir / "fig_checkpoint_rmse.pdf", cfg)
    plot_pacing_distribution(pacing, output_dir / "fig_pacing_distribution.pdf")
    plot_pacing_boxplot(split_df, output_dir / "fig_pacing_boxplot.pdf", cfg)
    plot_pacing_gender(pacing, output_dir / "fig_pacing_gender.pdf")
    plot_weather_trends(weather_df, output_dir / "fig_weather_trends.pdf")
    for gc, label in [("M", "men"), ("F", "women")]:
        g_mwu = [r for r in weather["mwu_by_age_gender"] if r["gender"] == gc]
        g_reg = [r for r in weather["linear_regression_by_age_gender"]
                 if r["gender"] == gc]
        plot_temp_significance_bars(
            g_mwu, f"Warm vs Cool ({label.title()})",
            output_dir / f"fig_temp_pvalues_{label}.pdf")
        plot_temp_effect_magnitude(
            g_mwu, label.title(), output_dir / f"fig_temp_magnitude_{label}.pdf")
        plot_temp_regression_grid(
            full_df, weather_df, gc, g_reg,
            output_dir / f"fig_temp_regression_{label}.pdf", cfg)

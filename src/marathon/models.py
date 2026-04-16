import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error as rmse
from .data import add_centered_pre_race_features
from .inference import compute_conformal_quantile

_LME_FIXED_EFFECTS_FORMULA = ("log_seconds ~ age_centered + I(age_centered**2) + female + year_centered"
                              " + age_centered:female + I(age_centered**2):female")
IN_RACE_DEMOGRAPHIC_FEATURES = ["age", "female"]


def fit_hc3_robust_log_seconds_regression(train_df, feature_columns, cfg):
    fit = smf.ols("log_seconds ~ " + " + ".join(feature_columns), data=train_df).fit(cov_type="HC3")
    duan = np.exp(fit.resid).mean()
    hat = fit.get_influence().hat_matrix_diag
    loo_pred = np.exp(train_df["log_seconds"] - fit.resid / (1 - hat)) * duan
    return fit, duan, compute_conformal_quantile(np.abs(train_df["seconds"] - loo_pred), cfg.conformal_alpha)


def fit_temporal_holdout_runner_mixed_effects(repeat_df, cfg):
    train = repeat_df[repeat_df["year"] <= cfg.mixed_effects_train_end_year].copy()
    test = repeat_df[repeat_df["year"] > cfg.mixed_effects_train_end_year].copy()
    age_mean = train["age"].mean()

    holders = set(train.groupby("display_name").size().loc[lambda c: c >= 2].index)
    test_known = test[test["display_name"].isin(holders)].copy()
    test_never = test[~test["display_name"].isin(holders)].copy()

    for d in (train, test_known, test_never):
        d["log_seconds"] = np.log(d["seconds"].to_numpy())
        add_centered_pre_race_features(d, age_mean, cfg)

    fit_df = train.sort_values("display_name").reset_index(drop=True)
    lme_fit = smf.mixedlm(_LME_FIXED_EFFECTS_FORMULA, fit_df, groups="display_name", re_formula="~age_centered").fit(reml=True, method="lbfgs")
    re_df = pd.DataFrame(lme_fit.random_effects).T
    re_df.columns = ["runner_intercept", "runner_age_slope"]
    duan = np.exp(fit_df["log_seconds"].to_numpy(float) - lme_fit.fittedvalues).mean()

    marg_known = lme_fit.predict(test_known).to_numpy()
    re_known = re_df.reindex(test_known["display_name"].values)
    cond_known = marg_known + re_known["runner_intercept"].to_numpy() + re_known["runner_age_slope"].to_numpy() * test_known["age_centered"].to_numpy()

    obs_known, obs_never = test_known["seconds"].to_numpy(float), test_never["seconds"].to_numpy(float)
    pred_marg_known = np.exp(marg_known) * duan
    pred_cond_known = np.exp(cond_known) * duan
    pred_marg_never = np.exp(lme_fit.predict(test_never).to_numpy()) * duan

    fe_var = np.var(lme_fit.predict(train).to_numpy())
    rs_var = lme_fit.cov_re.iloc[1, 1] * train["age_centered"].var()
    total_var = fe_var + lme_fit.cov_re.iloc[0, 0] + rs_var + lme_fit.scale
    leakfree_re = re_df.loc[re_df.index.isin(holders)].rename_axis("display_name")
    ni_holders = train.groupby("display_name").size().loc[lambda s: s.index.isin(holders)]
    raw_intercepts = train.groupby("display_name")["log_seconds"].mean() - lme_fit.predict(train).to_numpy().mean()

    return {
        "age_mean_leakfree_from_training": age_mean, "duan_smearing_factor": duan,
        "runner_random_effects_leakfree": leakfree_re, "lme_result_object": lme_fit,
        "test_known": test_known, "n_test_never_seen": len(test_never),
        "marginal_mixed_effects_rmse_on_test_known": rmse(obs_known, pred_marg_known),
        "conditional_mixed_effects_rmse_on_test_known": rmse(obs_known, pred_cond_known),
        "marginal_mixed_effects_rmse_on_never_seen": rmse(obs_never, pred_marg_never),
        "q_hat_lme": compute_conformal_quantile(np.abs(obs_known - pred_cond_known), cfg.conformal_alpha),
        "variance_explained_by_fixed_effects_only": fe_var / total_var,
        "variance_explained_by_fixed_and_random_effects": (fe_var + lme_fit.cov_re.iloc[0, 0] + rs_var) / total_var,
        "shrinkage_ni_summary": {"mean": ni_holders.mean(), "median": ni_holders.median(),
                                  "q1": ni_holders.quantile(0.25), "q3": ni_holders.quantile(0.75), "max": ni_holders.max()},
        "shrinkage_re_predictor_vs_raw_var_ratio": leakfree_re["runner_intercept"].var() / raw_intercepts.loc[raw_intercepts.index.isin(holders)].var(),
    }


def _ridge_conformal(X_tr, y_tr, cfg):
    ridge = RidgeCV(cv=None).fit(X_tr, y_tr)
    hat = np.sum((X_tr @ np.linalg.inv(X_tr.T @ X_tr + ridge.alpha_ * np.eye(X_tr.shape[1]))) * X_tr, axis=1)
    q_hat = compute_conformal_quantile(np.abs((y_tr - ridge.predict(X_tr)) / (1 - hat)), cfg.conformal_alpha)
    return ridge, q_hat


def fit_checkpoint_ridge_models(train_df, test_df, cfg):
    train_holders, test_holders = train_df[train_df["runner_intercept"].notna()], test_df[test_df["runner_intercept"].notna()]
    rows, fitted = [], {}
    y_tr, y_te = train_df["seconds"].to_numpy(float), test_df["seconds"].to_numpy(float)

    for ci, label in enumerate(cfg.checkpoint_labels):
        feats = cfg.cumulative_split_time_columns[:ci + 1] + IN_RACE_DEMOGRAPHIC_FEATURES
        ridge, q_hat = _ridge_conformal(train_df[feats].to_numpy(float), y_tr, cfg)
        rows.append({"checkpoint": label, "variant": "no_runner_history", "alpha": ridge.alpha_,
                      "rmse_seconds": rmse(y_te, ridge.predict(test_df[feats].to_numpy(float))), "q_hat": q_hat, "n": len(test_df)})
        fitted[(label, "no_runner_history")] = {"ridge_regression_model": ridge, "features": feats, "q_hat": q_hat}

        if label == "5K":
            hf = feats + ["runner_intercept", "runner_age_slope"]
            ridge_h, q_hat_h = _ridge_conformal(train_holders[hf].to_numpy(float), train_holders["seconds"].to_numpy(float), cfg)
            rows.append({"checkpoint": "5K", "variant": "with_runner_history", "alpha": ridge_h.alpha_,
                          "rmse_seconds": rmse(test_holders["seconds"].to_numpy(float), ridge_h.predict(test_holders[hf].to_numpy(float))),
                          "q_hat": q_hat_h, "n": len(test_holders)})
            fitted[("5K", "with_runner_history")] = {"ridge_regression_model": ridge_h, "features": hf, "q_hat": q_hat_h}

    return pd.DataFrame(rows), fitted


def fit_checkpoint_lgbm_models(train_df, test_df, rng, cfg):
    indices = rng.permutation(len(train_df))
    n_fit = int(len(train_df) * (1 - cfg.conformal_split_ratio))
    train_fit, train_calib = train_df.iloc[indices[:n_fit]], train_df.iloc[indices[n_fit:]]
    y_fit, y_calib, y_te = train_fit["seconds"].to_numpy(float), train_calib["seconds"].to_numpy(float), test_df["seconds"].to_numpy(float)
    rows, fitted, base_seed = [], {}, int(rng.integers(2**31))

    for ci, label in enumerate(cfg.checkpoint_labels):
        feats = (cfg.cumulative_split_time_columns[:ci + 1] + IN_RACE_DEMOGRAPHIC_FEATURES
                 + [f"seg_pace_{j}" for j in range(ci + 1)] + [f"seg_pace_change_{j}" for j in range(1, ci + 1)]
                 + ["bib_feature", f"z_pace_{ci}", f"pace_variance_{ci}", "heat_exposure"])
        model = LGBMRegressor(n_estimators=500, learning_rate=0.01, max_depth=4,
                               random_state=base_seed + ci, **cfg.lgbm_deterministic_params).fit(train_fit[feats].to_numpy(float), y_fit)
        q_hat = compute_conformal_quantile(np.abs(y_calib - model.predict(train_calib[feats].to_numpy(float))), cfg.conformal_alpha)
        pred = model.predict(test_df[feats].to_numpy(float))
        rows.append({"checkpoint": label, "variant": "lgbm_pacing", "alpha": np.nan,
                      "rmse_seconds": rmse(y_te, pred), "q_hat": q_hat, "n": len(test_df)})
        fitted[(label, "lgbm_pacing")] = {"model": model, "features": feats, "predictions": pred, "q_hat": q_hat}

    return pd.DataFrame(rows), fitted

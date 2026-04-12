# Information Decomposition of Marathon Finish-Time Prediction

## Abstract

This project measures how much four information sources each improve marathon finish-time prediction: pre-race demographics, prior Boston Marathon history, runner-specific random effects, and in-race split times. Six nested models (M0-M5) are compared on a common held-out test set with BCa cluster-bootstrap confidence intervals. Main findings: (1) prior race history gives the largest pre-race gain; (2) runner random effects help, but only modestly; (3) a single early checkpoint (5 km) already beats all pre-race models; (4) all prediction intervals undercover on held-out years due to temporal distribution shift, and honest recalibration does not fully fix this.

A single deterministic pipeline fits four stages on data up to 2016 and evaluates everything on a common 2017 test set.

**Stage 1** fits HC3-robust OLS on log(seconds) with Duan (1983) smearing to convert predictions back to seconds. Two feature sets: demographics alone (age, gender, year, interactions) and demographics plus leak-free prior history (expanding-window mean time, log appearances).

**Stage 2** fits a linear mixed-effects model via `lme4::lmer` (Bates et al., 2015) on log(seconds) with REML, using a random intercept and random age slope per runner. This captures how individual runners differ from the population trend. Only pre-2017 data is used; the resulting best linear unbiased predictors of the random effects are exported for later stages. The model assumes normal random effects following standard practice; Vu et al. (2024) analyze the consequences of this assumption for prediction.

**Stage 3** fits ridge regression (Hoerl and Kennard, 1970) at each of nine cumulative checkpoints (5K through 40K), predicting raw seconds from cumulative splits plus demographics. The regularization parameter is chosen by generalized cross-validation.

**Evaluation** compares all six models (M0-M5) on one common 2017 test subset with paired BCa cluster-bootstrap confidence intervals (Efron, 1987), clustering by runner to handle repeat participants. Improvements are reported in seconds and scaled by the year-drift reference (median absolute year-over-year shift in mean finish time). Two supplementary analyses address subgroup claims the population-averaged table cannot capture: the runner-effects contribution (do predicted random effects help on top of 5K splits?) and the never-seen subgroup (do splits work for runners with no prior Boston data?).

**Calibration** checks whether prediction intervals hold up under temporal distribution shift, using split conformal quantile regression (Romano et al., 2019). All methods undercover on held-out years. A K-S test on conformity scores pinpoints the cause, and honest temporal recalibration experiments measure the remaining gap.

## Nested Model Comparison

| Stage | Model | Information Added |
|-------|-------|-------------------|
| M0 | Mean baseline | None (training grand mean) |
| M1 | + demographics | Age, gender, year, age-gender interactions |
| M2 | + prior history | Expanding-window mean time, log prior appearances |
| M3 | + runner effects | Best linear unbiased predictors (random intercept + random age slope) |
| M4 | + 5K splits | Cumulative split seconds through the 5 km checkpoint |
| M5 | + 40K splits | Cumulative split seconds through the 40 km checkpoint |

All models train on data up to 2016 and are tested on a common 2017 subset (runners with complete 9-checkpoint splits). M0-M3 predict log(seconds) with Duan smearing; M4-M5 predict raw seconds via ridge regression.

## References

- **Duan (1983).** *Smearing estimate: a nonparametric retransformation method.* JASA 78(383). Corrects bias when converting log-scale predictions back to the original scale. Used in Stages 1-3.

- **MacKinnon & White (1985).** *Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties.* J. Econometrics 29(3). HC3 robust standard errors for pre-race regression, avoiding homoskedasticity assumptions.

- **Efron (1987).** *Better bootstrap confidence intervals.* JASA 82(397). BCa bootstrap for second-order accurate confidence intervals. Extended here to cluster resampling to handle within-runner correlation.

- **Hoerl & Kennard (1970).** *Ridge regression: biased estimation for nonorthogonal problems.* Technometrics 12(1). Ridge regression with GCV regularization for checkpoint-based split prediction.

- **Bates, Machler, Bolker, & Walker (2015).** *Fitting linear mixed-effects models using lme4.* J. Stat. Softw. 67(1). The runner mixed-effects model is fit via `lme4::lmer` through the `rpy2` bridge. Profile likelihood CIs on variance components follow the methods in this package.

- **Romano, Patterson, & Candes (2019).** *Conformalized quantile regression.* NeurIPS 32. Split-CQR gives distribution-free prediction intervals with finite-sample coverage guarantees under exchangeability. Used here to build 90% intervals at each checkpoint and to diagnose coverage loss under temporal shift.

- **Vu, Hui, Muller, & Welsh (2024).** *Random effects misspecification and its consequences for prediction in GLMMs.* arXiv:2411.19384. Shows that assuming normal random effects affects best linear unbiased predictors and mean squared prediction errors even when fixed-effect estimates stay consistent. The M2-vs-M3 comparison here empirically measures the practical value of this modeling choice.

## Requirements

- Python >= 3.14
- R with the `lme4` package

```bash
uv sync
```

## Reproducing Results

```bash
uv run scripts/run.py
```

This runs every stage, the nested comparison, calibration diagnostics, and writes all figures, tables, and metrics. Takes roughly 15-20 minutes (mostly the `lme4` fit and BCa bootstrap).

To rebuild the processed dataset from raw CSVs (only needed if the raw data changes):

```bash
uv run scripts/preprocess.py
```

To run the supplementary exploratory data analysis:

```bash
uv pip install -e ".[eda]"
uv run scripts/supplementary/eda.py
```

## Reproducibility

The pipeline is fully deterministic:

- **Seeds:** `np.random.default_rng(0).spawn(N)` gives each stage its own RNG stream. Execution order does not matter.
- **Splits:** All train/test boundaries are year-based, defined in `config.py`. No random splitting.
- **Ridge:** `RidgeCV(cv=None)` uses generalized cross-validation (deterministic).
- **Identical output:** Running the pipeline twice produces the same `pipeline_metrics.json`, figures, and tables.

## Claim-to-Code Mapping

| Claim | Evidence | Module | Output |
|-------|----------|--------|--------|
| Demographics explain coarse variation | M0 -> M1 | `regression.py` | `tab_ablation.tex` |
| Prior history is the largest pre-race gain | M1 -> M2 | `regression.py` | `tab_ablation.tex`, `fig1_ablation.pdf` |
| Runner effects add modest further value | M2 -> M3 | `mixed_effects.py` | `tab_ablation.tex` |
| 5K splits beat all pre-race models | M3 -> M4 | `splits.py` | `tab_ablation.tex`, `fig2_checkpoint_rmse.pdf` |
| Runner effects are subsumed by splits | Runner-effects supplement | `inference.py` | `pipeline_metrics.json` |
| Splits work for runners with no prior data | Never-seen supplement | `inference.py` | `pipeline_metrics.json` |
| Intervals undercover under temporal shift | Coverage diagnostic | `calibration.py` | `tab_coverage.tex`, `fig3_coverage.pdf` |
| Recalibration does not fully restore coverage | Recalibration experiments | `calibration.py` | `tab_recalibration.tex`, `fig4_conformity_shift.pdf` |

## License

Research code for academic use.

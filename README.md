# Information Decomposition in Marathon Finish-Time Prediction

A statistical framework for predicting Boston Marathon finish times that breaks down predictive accuracy along a runner's information timeline -- from pre-race registration through in-race checkpoints.

The key finding: at every stage, accuracy is limited by the information available, not the model's complexity. A single 5K split time makes all pre-race personalization obsolete, and upgrading from ridge regression to gradient boosting produces negligible improvement at late checkpoints. The information frontier -- not the function class -- determines accuracy.

The framework uses conformal prediction interval widths (split-conformal and LOO-conformal calibration) that shrink monotonically as race data accumulates, giving a model-agnostic measure of when each information source stops mattering.

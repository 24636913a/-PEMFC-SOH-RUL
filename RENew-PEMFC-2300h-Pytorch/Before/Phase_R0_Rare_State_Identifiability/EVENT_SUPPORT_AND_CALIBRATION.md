# Event support and probability calibration

Train OOF support is **SUFFICIENT** under the preregistered threshold: 66 E1 plus 9 E2 global independent events, spanning 7 continuous segments. This is an exploratory Development diagnosis, not an independent generalization claim.

OOF Brier scores are fold_1=0.01848, fold_2=0.01618, fold_3=0.01564. Mean predicted probabilities (0.241, 0.238, 0.236) exceed observed mixed frequencies (0.210, 0.210, 0.210), so existing probabilities show aggregate overprediction and are not declared calibrated. No calibration model or threshold was fitted.

See `outputs/calibration/probability_calibration.csv` and `outputs/figures/train_oof_probability_calibration.png`.

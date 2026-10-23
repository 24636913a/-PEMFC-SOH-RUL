# Fold and horizon diagnostics

Horizon RMSE/MAE are in `outputs/HORIZON_DIAGNOSTICS.csv`; bias and output-dispersion diagnostics are in `outputs/T0_HORIZON_BEHAVIOR_DIAGNOSTICS.csv`. These values retain overlapping forecast-query weighting and should not be interpreted as independent-sample evidence. Error tail and target multiplicity diagnostics are in `outputs/ERROR_DIAGNOSTICS.csv`.

Across folds, the equal-fold RMSE rises from 20.201 mV at 0.5 h to 28.102 mV at 12 h (39.1% higher). It is not monotone at every step: the maximum equal-fold horizon RMSE is 48.871 mV at step 19. The signed 12 h biases are fold-specific rather than sharing a common direction (Fold 1 -0.790 mV; Fold 2 2.727 mV; Fold 3 -0.485 mV). Thus the observed degradation is primarily error dispersion, not a uniformly signed late-horizon drift.

Fold 3 has 11,184 forecast queries and RMSE 33.453 mV. The frozen matched-origin context values are Raw F2 TimesNet 26.680 mV and the legacy TFT-like implementation 19.395 mV. Canonical T0 therefore does not reproduce a Fold-3-only advantage. There is no repeated, cross-fold periodic error minimum supporting a cycle-specific advantage; Figure 4 is descriptive and does not select a horizon-specific model.

# Forecast-error augmentation audit

FEA samples a complete 24×9 residual trajectory only from causal Ridge OOF sources whose target labels ended before the destination training origin. It never samples validation residuals. Synthetic paths are projected with a fold-train-only current/density area estimate; projection frequency is reported below.

| model | fold | fair_train_windows | historical_residual_source_rows | active_area_cm2_train_only | mean_epoch_augmentation_fraction | mean_epoch_projection_trajectory_fraction | same_fair_sample_set_as_M0 |
|---|---|---|---|---|---|---|---|
| M1_SA_TimeXer_FEA | fold_1 | 874 | 961 | 280.001262 | 0.501751 | 1.000000 | True |
| M1_SA_TimeXer_FEA | fold_2 | 1107 | 1194 | 280.001660 | 0.502294 | 1.000000 | True |
| M1_SA_TimeXer_FEA | fold_3 | 1352 | 1439 | 280.001723 | 0.501849 | 1.000000 | True |

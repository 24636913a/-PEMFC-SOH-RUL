# Decomposition contract

Frozen `StepAlignedTimeXer` computes `base = base.head(history encoding)`, then applies horizon-specific FiLM to `history state + horizon state`; `prediction = base + correction(modified)`. Thus B/C are a decomposition of one jointly trained forward pass, not separately trained models. The frozen contract is 48 x 0.5 h = 24 h historical input, 24 x 0.5 h = 12 h forecast, 14 history features, nine future-condition features, DeltaV target, and final voltage recovery `anchor + scaled DeltaV`.

| decomposition_max_residual_mV | p1_p2_base_max_difference_mV | squared_error_max_residual_mV2 | test_accessed |
|---|---|---|---|
| 0.000119209 | 0 | 0.0276605 | False |

Baseline cache reproduction:

| model | fold_average_definition | mean_r2_delta | mean_rmse_delta_mV | mean_mae_delta_mV | mean_p95_mV | mean_p99_mV |
|---|---|---|---|---|---|---|
| P0 | unweighted_mean_three_folds | 0.983989 | 24.5382 | 12.3925 | 34.201 | 127.226 |
| P1 | unweighted_mean_three_folds | 0.984111 | 24.4429 | 12.3181 | 33.401 | 127.226 |
| P2 | unweighted_mean_three_folds | 0.986918 | 21.3466 | 10.3817 | 26.1468 | 69.1861 |
| RS | unweighted_mean_three_folds | 0.983944 | 24.2018 | 11.3836 | 28.4834 | 100.878 |
| B_only | unweighted_mean_three_folds | 0.0258056 | 192.383 | 118.672 | 483.577 | 512.014 |

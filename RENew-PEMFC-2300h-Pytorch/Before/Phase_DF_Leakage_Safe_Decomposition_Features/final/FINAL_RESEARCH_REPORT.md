# Phase DF final research report

## Scope and protocol

This frozen-backbone feature study retained the 48-step / 24 h information set, 24-step / 12 h horizon, frozen Phase BS TimesNet architecture and Development-only boundary. All decomposition is origin-conditioned causal-block processing of `V[t-47:t]`, not global full-series decomposition. Phase BS protocol SHA256: `fd710c4f2c1036c9a4b49068bd38d16aa39ad9f4bbc94a41b30f59f37afd6788`. Test was not used.

## Stage 0

SimpleTrend: `PASS`. VMD: `PASS` with Train-only unsupervised resolved `alpha=2000`, `K=2`, `DC=0`. EMD: `PASS` with valid-IMF rate `1.0`. Synthetic future perturbation and prefix invariance passed. No leakage failure occurred.

## Seed 42 summaries (mV)

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_inner_plus_refit_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMDFeature_TimesNet | 42 | 13.3350 | 4.1384 | 0.9942 | 26.2695 | 9.5847 | 37.4921 | 527.5075 | 24.2894 | 22.5213 | 35.2726 | 149304 | 83.9961 | 413.9226 |
| RawF2_TimesNet | 42 | 14.2066 | 3.9504 | 0.9926 | 27.6394 | 9.0272 | 48.4224 | 542.1563 | 25.7175 | 26.6799 | 37.3129 | 149046 | nan | 0.0000 |
| SimpleTrend_TimesNet | 42 | 13.5667 | 3.6487 | 0.9933 | 26.8950 | 7.9110 | 43.3933 | 513.9196 | 24.5607 | 25.4942 | 37.3104 | 149304 | 83.9961 | 626.0318 |
| VMDFeature_TimesNet | 42 | 12.9652 | 4.0651 | 0.9945 | 26.8090 | 9.6555 | 38.7631 | 521.8515 | 23.8814 | 22.0820 | 34.9433 | 149304 | 83.9961 | 445.5444 |

## Multi-seed summaries (mV)

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_inner_plus_refit_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMDFeature_TimesNet | 42 | 13.3350 | 4.1384 | 0.9942 | 26.2695 | 9.5847 | 37.4921 | 527.5075 | 24.2894 | 22.5213 | 35.2726 | 149304 | 83.9961 | 413.9226 |
| EMDFeature_TimesNet | 2024 | 13.1296 | 3.9626 | 0.9938 | 26.9963 | 9.5272 | 43.9523 | 532.0061 | 23.8762 | 24.3517 | 36.8159 | 149304 | 83.2930 | 346.6516 |
| EMDFeature_TimesNet | 3407 | 12.4015 | 3.7549 | 0.9944 | 27.5193 | 8.5924 | 35.5041 | 551.5418 | 23.8775 | 23.4223 | 36.5892 | 149304 | 85.7305 | 390.6027 |
| RawF2_TimesNet | 42 | 14.2066 | 3.9504 | 0.9926 | 27.6394 | 9.0272 | 48.4224 | 542.1563 | 25.7175 | 26.6799 | 37.3129 | 149046 | nan | 0.0000 |
| RawF2_TimesNet | 2024 | 13.6233 | 3.8247 | 0.9929 | 29.2654 | 8.6573 | 45.8780 | 570.9616 | 25.0332 | 26.6678 | 38.6684 | 149046 | nan | 0.0000 |
| RawF2_TimesNet | 3407 | 13.7505 | 4.4992 | 0.9935 | 26.6769 | 11.0268 | 43.8434 | 512.0992 | 24.8388 | 24.2217 | 35.8443 | 149046 | nan | 0.0000 |
| SimpleTrend_TimesNet | 42 | 13.5667 | 3.6487 | 0.9933 | 26.8950 | 7.9110 | 43.3933 | 513.9196 | 24.5607 | 25.4942 | 37.3104 | 149304 | 83.9961 | 626.0318 |
| SimpleTrend_TimesNet | 2024 | 12.9126 | 3.8659 | 0.9942 | 25.9670 | 9.1864 | 43.3738 | 541.7669 | 23.2847 | 23.0497 | 34.5039 | 149304 | 83.2930 | 368.6881 |
| SimpleTrend_TimesNet | 3407 | 14.4576 | 4.2903 | 0.9919 | 25.6681 | 9.7730 | 65.3623 | 551.7627 | 24.7604 | 28.3741 | 36.2970 | 149304 | 83.2930 | 276.0729 |
| VMDFeature_TimesNet | 42 | 12.9652 | 4.0651 | 0.9945 | 26.8090 | 9.6555 | 38.7631 | 521.8515 | 23.8814 | 22.0820 | 34.9433 | 149304 | 83.9961 | 445.5444 |
| VMDFeature_TimesNet | 2024 | 13.1795 | 3.8346 | 0.9938 | 29.6349 | 8.8554 | 45.8788 | 534.0299 | 26.2700 | 24.3457 | 36.8984 | 149304 | 83.9961 | 353.3720 |
| VMDFeature_TimesNet | 3407 | 14.0133 | 4.0277 | 0.9926 | 26.9537 | 9.1584 | 58.0316 | 538.2820 | 24.2674 | 27.0991 | 37.3189 | 149304 | 83.2930 | 449.8906 |

## Frozen-gate interpretation

Seed-42 authorization is recorded without retroactive changes in `outputs/SEED42_GATES.csv`:

| method | seed | raw_predictive_gain_pass | decomposition_specific_seed42_pass | additional_seeds_authorized | formal_95pct | fold3_103pct | p99_105pct | target_equal_105pct | fold3_target_equal_110pct | endpoint_110pct | formal_strictly_better_than_simpletrend | fold3_105pct_of_simpletrend | p99_105pct_of_simpletrend | endpoint_105pct_of_simpletrend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleTrend_TimesNet | 42 | False | nan | True | False | True | True | True | True | True | nan | nan | nan | nan |
| VMDFeature_TimesNet | 42 | True | True | True | True | True | True | True | True | True | True | True | True | True |
| EMDFeature_TimesNet | 42 | True | True | True | True | True | True | True | True | True | True | True | True | True |

The formal three-seed gates are:

| method | multiseed_performed | robust_raw_gain | formal_95pct | paired_seeds_at_least_2 | no_seed_over_110pct | fold3_103pct | p99_105pct | target_equal_105pct | fold3_target_equal_110pct | endpoint_110pct | decomposition_specific_advance | formal_mean_strictly_better_than_simpletrend | fold3_105pct_of_simpletrend | p99_105pct_of_simpletrend | endpoint_105pct_of_simpletrend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleTrend_TimesNet | True | False | False | True | True | True | False | True | True | True | nan | nan | nan | nan | nan |
| VMDFeature_TimesNet | True | False | False | True | True | True | True | True | True | True | nan | nan | nan | nan | nan |
| EMDFeature_TimesNet | True | True | True | True | True | True | True | True | True | True | nan | nan | nan | nan | nan |
| VMDFeature_TimesNet | nan | nan | nan | True | nan | nan | nan | nan | nan | nan | False | True | True | True | False |
| EMDFeature_TimesNet | nan | nan | nan | True | nan | nan | nan | nan | nan | nan | True | True | True | True | True |

SimpleTrend did not meet the registered robust Raw-gain gate. VMD did not meet the registered decomposition-specific advancement gate because its mean endpoint-risk condition versus SimpleTrend failed. EMD met the robust Raw-gain gate and the registered decomposition-specific gate versus SimpleTrend. This supports an **EMD local slow-varying representation result under this frozen 48-step forecasting protocol only**; it is not a general claim about EMD, SOH, RUL, irreversible degradation, or longer information sets.

## Decision

Final status: **`EMD_SPECIFIC_ADVANCE`**. The Raw vs SimpleTrend and decomposition-vs-SimpleTrend gates are recorded in `outputs/SEED42_GATES.csv` and `outputs/MULTISEED_GATES.csv`. Extra seeds were run only for `['EMDFeature_TimesNet', 'SimpleTrend_TimesNet', 'VMDFeature_TimesNet']`. Any generic trend effect is not interpreted as decomposition-specific value unless the registered paired SimpleTrend comparisons pass.

`TIMESNET_FREEZE` remains in force and backbone search remains `CLOSED`. This phase makes no SOH, RUL, irreversible-degradation-mode, or extended-context claim.

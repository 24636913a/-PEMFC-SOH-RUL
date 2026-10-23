# Final backbone screening report

## Decision

**TIMESNET_FREEZE**

The formal outcome is the arithmetic mean of the three fold-level `forecast_query_weighted_RMSE` values. The target-time-equal quantity is a compulsory sensitivity and was not substituted as the formal score. No Test row, target, prediction, or metric was read or constructed.

## Historical context only — excluded from this decision

The prior N2.6R TimesNet three-seed Development value (about 9.908 mV) and the N2.6R ModernTCN seed-42 value (about 15.373 mV) use historical protocols. They are recorded only as context: this phase retrained the fixed TimesNet control and did not use either number for model selection.

## Seed 42 gate

| model | seed | formal_95pct | fold3_103pct | p99_105pct | target_equal_105pct | fold3_target_equal_110pct | implementation_clean | seed42_gate_pass | additional_seeds_authorized |
|---|---|---|---|---|---|---|---|---|---|
| TimesNet | 42 | False | True | True | True | True | True | False | True |
| SimpleTM | 42 | False | False | False | False | False | True | False | False |
| DUET | 42 | False | False | False | False | False | True | False | False |
| ModernTCN | 42 | False | True | False | True | True | True | False | False |

## Per-fold Development metrics

| run_id | status | model | fold | seed | inner_fit_origins | inner_stopper_origins | inner_target_overlap_count | selected_epoch | inner_stopper_selection_metric | inner_stopper_best_RMSE_mV | inner_time_s | outer_refit_time_s | total_inner_plus_refit_time_s | parameter_count | peak_gpu_memory_MiB | max_target_end_age_h | test_accessed | external_feature_decomposition | future_covariates_used | all_legal_validation_origins_retained | official_auxiliary_loss_rule | rmse_mV | mae_mV | r2 | endpoint_rmse_mV | p95_abs_error_mV | p99_abs_error_mV | max_abs_error_mV | target_time_equal_rmse_mV | query_count | unique_target_row_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bs_duet_fold_1_seed42 | complete | DUET | fold_1 | 42 | 867 | 158 | 0 | 231 | DeltaV_RMSE_mV | 65.3147 | 48.1630 | 50.0460 | 98.2090 | 54761 | 52.2188 | 1130.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 37.0515 | 18.0649 | 0.9638 | 51.1806 | 61.7715 | 219.9421 | 299.4597 | 41.1116 | 5592 | 256 |
| bs_duet_fold_2_seed42 | complete | DUET | fold_2 | 42 | 1065 | 193 | 0 | 8 | DeltaV_RMSE_mV | 40.9048 | 8.5010 | 2.1572 | 10.6582 | 54761 | 38.0718 | 1383.0000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 58.5497 | 25.2931 | 0.9100 | 71.3978 | 65.0035 | 280.9009 | 500.2833 | 59.7032 | 5880 | 268 |
| bs_duet_fold_3_seed42 | complete | DUET | fold_3 | 42 | 1281 | 229 | 0 | 177 | DeltaV_RMSE_mV | 48.3455 | 56.4511 | 54.1707 | 110.6218 | 54761 | 39.5293 | 1819.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 45.3864 | 19.2157 | 0.9458 | 54.1907 | 62.4585 | 252.0783 | 450.3087 | 46.8427 | 11184 | 558 |
| bs_moderntcn_fold_1_seed42 | complete | ModernTCN | fold_1 | 42 | 867 | 158 | 0 | 201 | DeltaV_RMSE_mV | 18.1268 | 63.5288 | 64.9170 | 128.4458 | 421264 | 84.0205 | 1130.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 13.6862 | 9.7363 | 0.9951 | 17.8155 | 35.3068 | 43.8021 | 131.2201 | 16.4726 | 5592 | 256 |
| bs_moderntcn_fold_2_seed42 | complete | ModernTCN | fold_2 | 42 | 1065 | 193 | 0 | 121 | DeltaV_RMSE_mV | 13.6579 | 50.2658 | 48.1789 | 98.4447 | 421264 | 85.7065 | 1383.0000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 16.6578 | 11.1485 | 0.9927 | 29.5573 | 35.1202 | 50.2059 | 290.9568 | 28.8643 | 5880 | 268 |
| bs_moderntcn_fold_3_seed42 | complete | ModernTCN | fold_3 | 42 | 1281 | 229 | 0 | 133 | DeltaV_RMSE_mV | 15.5754 | 66.0105 | 61.3572 | 127.3677 | 421264 | 84.0205 | 1819.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 22.7875 | 11.3725 | 0.9863 | 32.1944 | 38.6008 | 93.9662 | 461.7292 | 33.0527 | 11184 | 558 |
| bs_simpletm_fold_1_seed42 | complete | SimpleTM | fold_1 | 42 | 867 | 158 | 0 | 293 | DeltaV_RMSE_mV | 63.5940 | 46.2566 | 53.0976 | 99.3542 | 9000 | 82.8696 | 1130.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 29.4435 | 13.8694 | 0.9771 | 34.7363 | 37.8613 | 118.5607 | 277.0019 | 29.5370 | 5592 | 256 |
| bs_simpletm_fold_2_seed42 | complete | SimpleTM | fold_2 | 42 | 1065 | 193 | 0 | 289 | DeltaV_RMSE_mV | 31.1709 | 55.7691 | 64.5722 | 120.3414 | 9000 | 50.0957 | 1383.0000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 38.5647 | 14.5686 | 0.9609 | 49.4937 | 37.0617 | 203.5615 | 499.0033 | 42.4443 | 5880 | 268 |
| bs_simpletm_fold_3_seed42 | complete | SimpleTM | fold_3 | 42 | 1281 | 229 | 0 | 297 | DeltaV_RMSE_mV | 37.8788 | 69.7376 | 77.7473 | 147.4849 | 9000 | 52.2188 | 1819.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 35.6550 | 15.5347 | 0.9665 | 49.6979 | 42.5979 | 214.8356 | 448.5813 | 41.5081 | 11184 | 558 |
| bs_timesnet_fold_1_seed42 | complete | TimesNet | fold_1 | 42 | 867 | 158 | 0 | 67 | DeltaV_RMSE_mV | 24.6148 | 10.8881 | 8.2904 | 19.1785 | 149046 | 82.8696 | 1130.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 8.7533 | 4.5154 | 0.9980 | 16.8592 | 12.5474 | 33.4974 | 152.0310 | 13.5455 | 5592 | 256 |
| bs_timesnet_fold_2_seed42 | complete | TimesNet | fold_2 | 42 | 1065 | 193 | 0 | 295 | DeltaV_RMSE_mV | 2.4666 | 38.6196 | 44.9056 | 83.5252 | 149046 | 82.8696 | 1383.0000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 7.1865 | 1.7472 | 0.9986 | 27.7997 | 4.3612 | 6.9877 | 308.9166 | 26.2941 | 5880 | 268 |
| bs_timesnet_fold_3_seed42 | complete | TimesNet | fold_3 | 42 | 1281 | 229 | 0 | 140 | DeltaV_RMSE_mV | 7.4157 | 26.5164 | 24.3336 | 50.8500 | 149046 | 82.8696 | 1819.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 26.6799 | 5.5885 | 0.9813 | 38.2593 | 10.1729 | 104.7820 | 542.1563 | 37.3129 | 11184 | 558 |
| bs_timesnet_fold_1_seed2024 | complete | TimesNet | fold_1 | 2024 | 867 | 158 | 0 | 111 | DeltaV_RMSE_mV | 23.1424 | 15.2310 | 13.7110 | 28.9420 | 149046 | 84.6509 | 1130.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 6.8325 | 3.4004 | 0.9988 | 11.8573 | 9.4435 | 25.2753 | 106.1466 | 9.6882 | 5592 | 256 |
| bs_timesnet_fold_2_seed2024 | complete | TimesNet | fold_2 | 2024 | 1065 | 193 | 0 | 299 | DeltaV_RMSE_mV | 2.1660 | 38.0196 | 45.0049 | 83.0245 | 149046 | 84.6509 | 1383.0000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 7.3697 | 1.8867 | 0.9986 | 28.1542 | 4.6225 | 8.5995 | 309.5986 | 26.7431 | 5880 | 268 |
| bs_timesnet_fold_3_seed2024 | complete | TimesNet | fold_3 | 2024 | 1281 | 229 | 0 | 65 | DeltaV_RMSE_mV | 7.9845 | 14.7064 | 11.1609 | 25.8672 | 149046 | 82.8696 | 1819.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 26.6678 | 6.1870 | 0.9813 | 47.7847 | 11.9061 | 103.7592 | 570.9616 | 38.6684 | 11184 | 558 |
| bs_timesnet_fold_1_seed3407 | complete | TimesNet | fold_1 | 3407 | 867 | 158 | 0 | 51 | DeltaV_RMSE_mV | 31.3345 | 8.8123 | 6.3411 | 15.1533 | 149046 | 82.8696 | 1130.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 10.0533 | 5.6194 | 0.9973 | 16.1496 | 15.8667 | 45.0727 | 165.8959 | 14.3446 | 5592 | 256 |
| bs_timesnet_fold_2_seed3407 | complete | TimesNet | fold_2 | 3407 | 1065 | 193 | 0 | 290 | DeltaV_RMSE_mV | 2.2359 | 38.2774 | 43.9040 | 82.1814 | 149046 | 82.8696 | 1383.0000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 6.9765 | 2.2198 | 0.9987 | 25.8689 | 5.5772 | 8.7706 | 296.3897 | 24.3275 | 5880 | 268 |
| bs_timesnet_fold_3_seed3407 | complete | TimesNet | fold_3 | 3407 | 1281 | 229 | 0 | 102 | DeltaV_RMSE_mV | 6.4217 | 22.4953 | 17.9998 | 40.4951 | 149046 | 82.8696 | 1819.5000 | False | False | False | True | DUET L_importance loss_coef=1 added; zero for other official models | 24.2217 | 5.6584 | 0.9846 | 38.0122 | 11.6365 | 77.6869 | 512.0992 | 35.8443 | 11184 | 558 |

## Per-seed three-fold equal summaries

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_inner_plus_refit_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DUET | 42 | 46.9959 | 20.8579 | 0.9398 | 58.9230 | 63.0778 | 250.9738 | 500.2833 | 49.2191 | 45.3864 | 46.8427 | 54761 | 52.2188 | 219.4890 |
| ModernTCN | 42 | 17.7105 | 10.7524 | 0.9914 | 26.5224 | 36.3426 | 62.6581 | 461.7292 | 26.1299 | 22.7875 | 33.0527 | 421264 | 85.7065 | 354.2581 |
| SimpleTM | 42 | 34.5544 | 14.6576 | 0.9682 | 44.6426 | 39.1736 | 178.9859 | 499.0033 | 37.8298 | 35.6550 | 41.5081 | 9000 | 82.8696 | 367.1804 |
| TimesNet | 42 | 14.2066 | 3.9504 | 0.9926 | 27.6394 | 9.0272 | 48.4224 | 542.1563 | 25.7175 | 26.6799 | 37.3129 | 149046 | 82.8696 | 153.5536 |
| TimesNet | 2024 | 13.6233 | 3.8247 | 0.9929 | 29.2654 | 8.6573 | 45.8780 | 570.9616 | 25.0332 | 26.6678 | 38.6684 | 149046 | 84.6509 | 137.8337 |
| TimesNet | 3407 | 13.7505 | 4.4992 | 0.9935 | 26.6769 | 11.0268 | 43.8434 | 512.0992 | 24.8388 | 24.2217 | 35.8443 | 149046 | 82.8696 | 137.8298 |

## Multi-seed final gate

| model | seed_count | formal_RMSE_mean_mV | formal_RMSE_std_mV | formal_RMSE_min_mV | formal_RMSE_max_mV | fold3_RMSE_mean_mV | P99_mean_mV | formal_improvement_fraction | paired_seeds_better_than_timesnet | formal_improvement_5pct | paired_seeds_at_least_2 | fold3_103pct | p99_105pct | target_equal_105pct | fold3_target_equal_110pct | final_eligible | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TimesNet | 3 | 13.8601 | 0.2504 | 13.6233 | 14.2066 | 25.8565 | 46.0479 | 0.0000 | 0.0000 | False | False | True | True | True | True | False | final_gate_failed_or_incumbent |
| SimpleTM | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | False | seed42_gate_failed_no_extra_seed |
| DUET | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | False | seed42_gate_failed_no_extra_seed |
| ModernTCN | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | False | seed42_gate_failed_no_extra_seed |

## Terminal stop

All three challengers failed the same seed-42 gate and received no extra seeds. The terminal backbone-search rule therefore applies: this phase does not start a fifth model, an adaptation, a future-condition predictor, a gate, or any Test evaluation.

## Next-stage design only

After the backbone is fixed, a separate future protocol may compare Raw F2 with EMD-derived, VMD-derived and selected degradation-related representations. It must first define causal rolling decomposition, condition-dominant versus degradation-dominant modes, stability, leakage controls and Train-only fitting. No decomposition was executed here; SimpleTM's internal SWT remains an architecture operation, not an external feature representation.

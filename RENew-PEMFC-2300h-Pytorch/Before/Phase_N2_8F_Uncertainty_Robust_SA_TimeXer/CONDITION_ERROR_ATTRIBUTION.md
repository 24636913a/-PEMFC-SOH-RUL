# Condition-error attribution

All analyses are development-only. Correlations and same-checkpoint group substitutions are diagnostic associations, not causal claims. Current and current-density are retained as internally consistent groups.

## Error by variable

| variable | standardized_future_exo_mae | standardized_future_exo_rmse | raw_future_exo_mae |
|---|---|---|---|
| actual_current_A_mean | 0.071812 | 0.115034 | 4.886696 |
| actual_current_A_peak_to_peak | 0.132754 | 0.184811 | 14.286609 |
| actual_current_A_std | 0.089522 | 0.132008 | 2.786785 |
| actual_current_density_A_cm2_mean | 0.071811 | 0.115025 | 0.017452 |
| actual_current_density_A_cm2_peak_to_peak | 0.132763 | 0.184810 | 0.051027 |
| actual_current_density_A_cm2_std | 0.089570 | 0.132068 | 0.009958 |
| load_transition_count | 0.101542 | 0.137804 | 1.070663 |
| operation_point_transitions | 0.076228 | 0.097045 | 0.490745 |
| zero_current_fraction | 0.120927 | 0.202857 | 0.024810 |

## Group association

| condition_group | pearson_r | spearman_rho | mean_standardized_exo_error | mean_voltage_mae_mV |
|---|---|---|---|---|
| current_amplitude | 0.775320 | 0.636364 | 0.071812 | 12.392522 |
| current_variability | 0.575334 | 0.588697 | 0.111152 | 12.392522 |
| load_transitions | 0.208872 | 0.352022 | 0.088885 | 12.392522 |
| zero_current | 0.800690 | 0.665447 | 0.120927 | 12.392522 |

## Same-checkpoint substitutions

| condition_group | baseline_rmse_mV | substitution_rmse_mV | rmse_difference_mV |
|---|---|---|---|
| current_amplitude | 24.538245 | 21.890446 | -2.647800 |
| current_variability | 24.538245 | 26.142968 | 1.604723 |
| load_transitions | 24.538245 | 24.697866 | 0.159620 |
| zero_current | 24.538245 | 15.416839 | -9.121406 |

# SOH Anchor Selection

Status: **RESOLVED**

Primary j*: **1 A/cm²**  
Secondary j* for sensitivity analysis: **1.6 A/cm²**

The decision uses only non-after-Air-Air Train + Validation checkpoints. Test checkpoints at 2000 h and 2300 h do not contribute to coverage, ranking, monotonicity, correlation, noise, Air-Air sensitivity or the normalization formula.

Candidates first pass hard gates for coverage, at least three Train and two Validation checkpoints, and at least 0.050 V distance above the 500 mV lower limit. Passing candidates are then ranked with equal fixed contributions from coverage, repeated-measurement std, monotonicity, |Spearman rho|, 1500 h Air-Air sensitivity, degradation-to-repeat-noise ratio and lower-limit margin. These weights were fixed before viewing Test anchors.

Primary metrics: coverage `1.000`, Spearman rho `-0.964`, monotonicity `0.960`, non-increasing fraction `0.833`, repeated std `0.000554 V`, margin to 500 mV `0.1792 V`.

## Candidate comparison

| target_jstar_A_cm2 | eligible_checkpoint_count | eligible_checkpoint_total | coverage | train_checkpoint_count | validation_checkpoint_count | voltage_mean_across_checkpoints_V | median_repeated_measurement_std_V | ageing_pearson_r | ageing_spearman_rho | monotonicity_score | non_increasing_fraction | relative_voltage_loss | absolute_voltage_loss_V | air_air_start_shift_1500h_V | air_air_start_sensitivity_abs_V | minimum_distance_to_500mV_limit_V | degradation_to_repeat_noise_ratio | hard_gate_pass | selection_scope | selection_score | selection_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 7 | 7 | 1.0 | 5 | 2 | 0.685007 | 0.000554 | -0.880987 | -0.964286 | 0.959605 | 0.833333 | 0.028215 | 0.019721 | 0.003916 | 0.003916 | 0.179232 | 35.619732 | True | train+validation_only | 0.644424 | 1.0 |
| 1.6 | 7 | 7 | 1.0 | 5 | 2 | 0.616577 | 0.001366 | -0.898463 | -1.0 | 1.0 | 1.0 | 0.046722 | 0.02978 | 0.006455 | 0.006455 | 0.107599 | 21.798247 | True | train+validation_only | 0.612485 | 2.0 |
| 1.4 | 7 | 7 | 1.0 | 5 | 2 | 0.639343 | 0.002031 | -0.89835 | -0.964286 | 0.997439 | 0.833333 | 0.039755 | 0.026142 | 0.005634 | 0.005634 | 0.131446 | 12.873336 | True | train+validation_only | 0.456037 | 3.0 |
| 0.8 | 7 | 7 | 1.0 | 5 | 2 | 0.708372 | 0.003505 | -0.908736 | -0.964286 | 0.956276 | 0.833333 | 0.019596 | 0.014072 | 0.002787 | 0.002787 | 0.204056 | 4.014738 | True | train+validation_only | 0.428571 | 4.0 |
| 1.2 | 7 | 7 | 1.0 | 5 | 2 | 0.661391 | 0.002567 | -0.895598 | -0.964286 | 0.97931 | 0.833333 | 0.032995 | 0.022336 | 0.004681 | 0.004681 | 0.154622 | 8.702104 | True | train+validation_only | 0.423421 | 5.0 |

## SOH definition

`SOH_V(t_i) = V(j*, t_i) / V(j*, t_ref)` with `t_ref = BoL` from the same 10-cell nominal polarization configuration.

The requested Candidate B, `1 - [V_ref - V(t_i)] / V_ref`, simplifies exactly to Candidate A. It is retained as an algebraic cross-check, not treated as an independent label.

Only sparse characterization anchors are created. No hourly or 30 min SOH interpolation is generated. No EOL-normalized SOH and no `RUL = 2300 - t` label are used.

# Sampling Resolution Decision

Recommended primary resolution: **0.5 h**

Robustness/sensitivity resolution: **1 h**

Decision scope: Train + Validation only. Test values, trends and forecasting errors were not used.

The comparison uses the same physical task at both resolutions. The reference legal-window comparison is past 24 h to future 24 h: 30 min uses L=48/H=48 and 1 h uses L=24/H=24.

Empirical duty-cycle period from Train-only ACF/spectrum agreement: **9.500 h**.

This is the period on the `OpHrs2` operating-age axis. The paper reports about 12 h in wall-clock time and explicitly includes three short and one longer stop; removing stop time explains why the operating-age period is shorter.

## Frozen-rule checks

| criterion | passed_for_30min |
| --- | --- |
| legal_windows_significantly_more | True |
| coverage_not_materially_worse | True |
| trendability_not_materially_worse | True |
| noise_ratio_not_materially_worse | True |
| dynamic_resolution_better | True |

## Evidence

| sampling_resolution_hours | selection_scope | total_bins | valid_bins | valid_bin_fraction | low_coverage_bin_fraction | gap_affected_bin_fraction | startup_affected_bin_fraction | shutdown_affected_bin_fraction | legal_windows_24h_to_24h | median_within_bin_voltage_temporal_std_V | median_within_bin_current_std_A | matched_condition_residual_std_V | matched_condition_local_noise_V | matched_voltage_age_spearman_rho | matched_voltage_long_term_slope_V_per_1000h | trendability_abs_spearman | matched_degradation_amplitude_V | noise_to_degradation_ratio | load_transition_count | operation_point_transition_count | median_operation_points_per_bin | mean_operation_points_per_bin | current_profile_preservation_score | average_time_points_per_duty_cycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | train+validation_only | 3003 | 2986 | 0.994339 | 0.003996 | 0.006327 | 0.210123 | 0.210123 | 1680 | 0.040839 | 81.700532 | 0.003914 | 0.001876 | -0.776949 | -0.005879 | 0.776949 | 0.007778 | 0.241138 | 57687 | 40378 | 5.0 | 4.394938 | 0.45184 | 19.0 |
| 1.0 | train+validation_only | 1505 | 1490 | 0.990033 | 0.006645 | 0.01196 | 0.419269 | 0.419269 | 844 | 0.041474 | 94.588422 | 0.003722 | 0.001814 | -0.805096 | -0.005787 | 0.805096 | 0.007533 | 0.240829 | 57681 | 40364 | 6.0 | 5.72691 | 0.341725 | 9.5 |

No deep or classical forecasting model was trained. If the recommendation is unresolved, both resolutions must continue only to a small baseline diagnostic in Phase N2.

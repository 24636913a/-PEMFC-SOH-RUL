# Pure-bin contract

The primary pure-bin indicator is computed only from frozen N1 fields and before any voltage residual, age contrast or anchor result is inspected. A 0.5 h bin is pure when all conditions hold:

1. `number_of_operation_points == 1`;
2. `operation_point_transitions == 0`;
3. `bin_valid_for_windows == True`;
4. `coverage_ratio >= 0.80`.

For a selected positive-current operation-point observation, the bin must additionally satisfy `zero_current_fraction == 0` and `actual_current_density_A_cm2_mean > 0`. No voltage, age, SOH, EMD feature, forecast error, or future information enters this definition.

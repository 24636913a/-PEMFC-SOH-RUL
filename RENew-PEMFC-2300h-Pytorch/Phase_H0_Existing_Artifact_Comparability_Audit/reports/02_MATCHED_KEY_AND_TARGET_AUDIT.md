# Matched key and target audit

Common keys were formed before any prediction error was accessed. For each NPZ, keys were reconstructed only through the exact OPR mapping `target_row=origin_row+horizon_step`. NPZ `target_end_age_h` is verified as the OPR final target bin-center age plus 0.25 h (the 30-minute bin-end convention). No nearest-time matching was used.

## Coverage

| fold | model | complete_key_count | common_key_count | coverage_fraction | missing_from_model_vs_common | extra_vs_common | origin_count | duplicate_keys | boundary_crossing_keys |
|---|---|---|---|---|---|---|---|---|---|
| fold_1 | H0_original | 5592 | 5592 | 1.000000 | 0 | 0 | 233 | 0 | 0 |
| fold_1 | H0_PEMFC_A | 5592 | 5592 | 1.000000 | 0 | 0 | 233 | 0 | 0 |
| fold_1 | H0_PEMFC_Life | 5592 | 5592 | 1.000000 | 0 | 0 | 233 | 0 | 0 |
| fold_1 | SA_P0 | 5592 | 5592 | 1.000000 | 0 | 0 | 233 | 0 | 0 |
| fold_1 | SA_P2 | 5592 | 5592 | 1.000000 | 0 | 0 | 233 | 0 | 0 |
| fold_1 | SA_M0_reference_only | 5592 | 5592 | 1.000000 | 0 | 0 | 233 | 0 | 0 |
| fold_2 | H0_original | 5880 | 5880 | 1.000000 | 0 | 0 | 245 | 0 | 0 |
| fold_2 | H0_PEMFC_A | 5880 | 5880 | 1.000000 | 0 | 0 | 245 | 0 | 0 |
| fold_2 | H0_PEMFC_Life | 5880 | 5880 | 1.000000 | 0 | 0 | 245 | 0 | 0 |
| fold_2 | SA_P0 | 5880 | 5880 | 1.000000 | 0 | 0 | 245 | 0 | 0 |
| fold_2 | SA_P2 | 5880 | 5880 | 1.000000 | 0 | 0 | 245 | 0 | 0 |
| fold_2 | SA_M0_reference_only | 5880 | 5880 | 1.000000 | 0 | 0 | 245 | 0 | 0 |
| fold_3 | H0_original | 11184 | 11184 | 1.000000 | 0 | 0 | 466 | 0 | 0 |
| fold_3 | H0_PEMFC_A | 11184 | 11184 | 1.000000 | 0 | 0 | 466 | 0 | 0 |
| fold_3 | H0_PEMFC_Life | 11184 | 11184 | 1.000000 | 0 | 0 | 466 | 0 | 0 |
| fold_3 | SA_P0 | 11184 | 11184 | 1.000000 | 0 | 0 | 466 | 0 | 0 |
| fold_3 | SA_P2 | 11184 | 11184 | 1.000000 | 0 | 0 | 466 | 0 | 0 |
| fold_3 | SA_M0_reference_only | 11184 | 11184 | 1.000000 | 0 | 0 | 466 | 0 | 0 |

## Truth, anchor and target age alignment

| reference_model | model | key_count | max_y_true_difference_V | max_anchor_difference_V | max_target_age_difference_h | pass |
|---|---|---|---|---|---|---|
| H0_original | H0_original | 22656 | 0.000000 | 0.000000 | 0.000000 | True |
| H0_original | H0_PEMFC_A | 22656 | 0.000000 | 0.000000 | 0.000000 | True |
| H0_original | H0_PEMFC_Life | 22656 | 0.000000 | 0.000000 | 0.000000 | True |
| H0_original | SA_P0 | 22656 | 0.000000 | 0.000000 | 0.000000 | True |
| H0_original | SA_P2 | 22656 | 0.000000 | 0.000000 | 0.000000 | True |
| H0_original | SA_M0_reference_only | 22656 | 0.000000 | 0.000000 | 0.000000 | True |

## Pointwise prediction equivalence checks

| left_model | right_model | key_count | max_prediction_difference_V | mean_absolute_prediction_difference_V | numerically_identical_at_voltage_tolerance | interpretation |
|---|---|---|---|---|---|---|
| H0_original | H0_PEMFC_A | 22656 | 0.000000 | 0.000000 | True | same saved prediction values only; not evidence of independent replication |
| H0_original | H0_PEMFC_Life | 22656 | 0.234558 | 0.011799 | False | same saved prediction values only; not evidence of independent replication |
| SA_P0 | SA_P2 | 22656 | 0.429312 | 0.006652 | False | same saved prediction values only; not evidence of independent replication |
| SA_P0 | SA_M0_reference_only | 22656 | 0.000000 | 0.000000 | True | same saved prediction values only; not evidence of independent replication |

H0_original and H0_PEMFC_A are evaluated pointwise here; numerical equality of their saved arrays is not labelled an independent statistical replication.

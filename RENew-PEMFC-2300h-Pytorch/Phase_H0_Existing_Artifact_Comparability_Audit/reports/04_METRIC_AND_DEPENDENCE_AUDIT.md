# Metric and dependence audit

The macro average is the unweighted mean of the three fold RMSE values. Pooled RMSE is calculated over all overlapping prediction points; it is not an independent-trajectory estimate. Unique-target aggregation is published separately as descriptive sensitivity, not a significance test.

## Matched descriptive metrics

| scope | aggregation | model | fold | origin_count | prediction_point_count | unique_target_row_count | rmse_deltaV_mV | mae_deltaV_mV | r2_deltaV | p95_absolute_error_mV | p99_absolute_error_mV | endpoint_12h_rmse_mV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matched_common_keys | fold | H0_original | fold_1 | 233 | 5592 | 256 | 31.668770 | 18.213408 | 0.973541 | 52.605471 | 114.066875 | 36.709917 |
| matched_common_keys | fold | H0_original | fold_2 | 245 | 5880 | 268 | 39.881034 | 19.791047 | 0.958230 | 55.432072 | 187.172438 | 45.373180 |
| matched_common_keys | fold | H0_original | fold_3 | 466 | 11184 | 558 | 36.961407 | 18.899152 | 0.964042 | 53.015658 | 188.156778 | 49.893953 |
| matched_common_keys | unweighted_three_fold_macro | H0_original | macro | 314 | 22656 | 1082 | 36.170404 | 18.967869 | 0.965271 | 53.684400 | 163.132030 | 43.992350 |
| matched_common_keys | pooled_prediction_points | H0_original | pooled | 944 | 22656 | 1082 | 36.533793 | 18.961372 | 0.964870 | 53.676426 | 178.973666 | 45.780035 |
| matched_common_keys | fold | H0_PEMFC_A | fold_1 | 233 | 5592 | 256 | 31.668770 | 18.213408 | 0.973541 | 52.605471 | 114.066875 | 36.709917 |
| matched_common_keys | fold | H0_PEMFC_A | fold_2 | 245 | 5880 | 268 | 39.881034 | 19.791047 | 0.958230 | 55.432072 | 187.172438 | 45.373180 |
| matched_common_keys | fold | H0_PEMFC_A | fold_3 | 466 | 11184 | 558 | 36.961407 | 18.899152 | 0.964042 | 53.015658 | 188.156778 | 49.893953 |
| matched_common_keys | unweighted_three_fold_macro | H0_PEMFC_A | macro | 314 | 22656 | 1082 | 36.170404 | 18.967869 | 0.965271 | 53.684400 | 163.132030 | 43.992350 |
| matched_common_keys | pooled_prediction_points | H0_PEMFC_A | pooled | 944 | 22656 | 1082 | 36.533793 | 18.961372 | 0.964870 | 53.676426 | 178.973666 | 45.780035 |
| matched_common_keys | fold | H0_PEMFC_Life | fold_1 | 233 | 5592 | 256 | 32.528422 | 18.865264 | 0.972085 | 56.171014 | 124.782449 | 38.346600 |
| matched_common_keys | fold | H0_PEMFC_Life | fold_2 | 245 | 5880 | 268 | 40.894888 | 20.646852 | 0.956080 | 57.255377 | 201.549228 | 45.955438 |
| matched_common_keys | fold | H0_PEMFC_Life | fold_3 | 466 | 11184 | 558 | 40.596628 | 21.377584 | 0.956621 | 63.631006 | 211.085715 | 53.043379 |
| matched_common_keys | unweighted_three_fold_macro | H0_PEMFC_Life | macro | 314 | 22656 | 1082 | 38.006646 | 20.296567 | 0.961595 | 59.019132 | 179.139131 | 45.781806 |
| matched_common_keys | pooled_prediction_points | H0_PEMFC_Life | pooled | 944 | 22656 | 1082 | 38.842918 | 20.567839 | 0.960288 | 60.952123 | 192.036563 | 47.957998 |
| matched_common_keys | fold | SA_P0 | fold_1 | 233 | 5592 | 256 | 21.883613 | 12.765853 | 0.987366 | 37.246452 | 97.964598 | 31.762232 |
| matched_common_keys | fold | SA_P0 | fold_2 | 245 | 5880 | 268 | 23.845808 | 12.305037 | 0.985067 | 33.912148 | 114.137746 | 30.678480 |
| matched_common_keys | fold | SA_P0 | fold_3 | 466 | 11184 | 558 | 27.885315 | 12.106674 | 0.979533 | 31.444526 | 169.574456 | 38.143228 |
| matched_common_keys | unweighted_three_fold_macro | SA_P0 | macro | 314 | 22656 | 1082 | 24.538245 | 12.392522 | 0.983989 | 34.201042 | 127.225600 | 33.527980 |
| matched_common_keys | pooled_prediction_points | SA_P0 | pooled | 944 | 22656 | 1082 | 25.487862 | 12.320856 | 0.982901 | 33.425689 | 129.651590 | 34.806246 |
| matched_common_keys | fold | SA_P2 | fold_1 | 233 | 5592 | 256 | 17.670893 | 10.345669 | 0.991762 | 26.552893 | 45.073219 | 41.296984 |
| matched_common_keys | fold | SA_P2 | fold_2 | 245 | 5880 | 268 | 15.987233 | 9.758537 | 0.993288 | 25.863140 | 55.544779 | 29.601087 |
| matched_common_keys | fold | SA_P2 | fold_3 | 466 | 11184 | 558 | 30.381570 | 11.040766 | 0.975705 | 26.024427 | 106.940372 | 35.528185 |
| matched_common_keys | unweighted_three_fold_macro | SA_P2 | macro | 314 | 22656 | 1082 | 21.346566 | 10.381657 | 0.986918 | 26.146820 | 69.186123 | 35.475419 |
| matched_common_keys | pooled_prediction_points | SA_P2 | pooled | 944 | 22656 | 1082 | 24.475714 | 10.536419 | 0.984232 | 26.165508 | 68.630090 | 35.657446 |

## Historical P0/P2 reproduction against OPR

| model | OPR_label | reconstructed_full_artifact_macro_rmse_mV | OPR_reported_macro_rmse_mV | absolute_difference_mV | pass |
|---|---|---|---|---|---|
| SA_P0 | P0 | 24.538245 | 24.538245 | 0.000000 | True |
| SA_P2 | P2 | 21.346566 | 21.346566 | 0.000000 | True |

SA P0 and N2.8F M0 pointwise maximum prediction difference: 0 V. This check establishes saved-artifact relationship only; it does not erase the H0/SA training-protocol confounds.

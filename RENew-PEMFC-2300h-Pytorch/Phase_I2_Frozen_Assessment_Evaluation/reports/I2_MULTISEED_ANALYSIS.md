# Equal-seed and paired analysis

Reused Development Assessment; n=3 frozen seeds. Not independent Test. All 9072 queries per run retained. No new training, no new model selection.

| model | n_seeds | RMSE_mV_mean | RMSE_mV_sample_SD | MAE_mV_mean | MAE_mV_sample_SD | R2_mean | R2_sample_SD | endpoint_RMSE_mV_mean | endpoint_RMSE_mV_sample_SD | P95_mV_mean | P95_mV_sample_SD | P99_mV_mean | P99_mV_sample_SD | max_abs_error_mV_mean | max_abs_error_mV_sample_SD | target_time_equal_RMSE_mV_mean | target_time_equal_RMSE_mV_sample_SD | RMSE_min_mV | RMSE_max_mV | pooled_prediction_point_RMSE_mV_descriptive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | 3 | 21.7454933 | 4.6130226 | 7.44077964 | 1.44823663 | 0.987267916 | 0.00556384062 | 18.4514521 | 10.536078 | 16.4076229 | 5.77815271 | 29.7064112 | 9.18587959 | 378.375985 | 110.871876 | 21.5772929 | 4.06480527 | 18.9127603 | 27.0684818 | 22.0692802 |
| F1 | 3 | 22.8928783 | 13.6030134 | 7.29925004 | 1.56096634 | 0.983075097 | 0.0184819954 | 22.4955573 | 20.048861 | 16.3905464 | 1.45247686 | 58.2923715 | 56.4160504 | 376.145486 | 139.275454 | 21.9932678 | 12.4114803 | 12.0520698 | 38.1570434 | 25.4449444 |
| F2 | 3 | 12.1970523 | 6.39381923 | 5.14700375 | 1.79721374 | 0.995398598 | 0.00374073373 | 7.06919513 | 2.48652867 | 13.0991693 | 4.32452423 | 21.7480151 | 6.77805321 | 289.281067 | 45.2088717 | 12.2981681 | 6.41698645 | 5.16532586 | 17.6615155 | 13.2673296 |

Equal-seed arithmetic mean RMSE is primary aggregation. Pooled prediction-point RMSE is descriptive and distinct; no seed predictions averaged into an ensemble. Sample SD uses ddof=1, not a confidence interval.

| seed | D10_F1_minus_F0_mV | D10_F1_minus_F0_relative_improvement_percent | D21_F2_minus_F1_mV | D21_F2_minus_F1_relative_improvement_percent | D20_F2_minus_F0_mV | D20_F2_minus_F0_relative_improvement_percent |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | -0.443238521 | 2.34359508 | -0.808006329 | 4.37480914 | -1.25124485 | 6.61587641 |
| 52 | 18.9018056 | -98.164488 | -24.3927278 | 63.9271958 | -5.49092225 | 28.5165123 |
| 62 | -15.016412 | 55.4756343 | -6.88674396 | 57.141587 | -21.9031559 | 80.9175635 |

| comparison | mean_difference_mV | sample_SD_mV | improved_seed_count | maximum_degradation_mV | mean_relative_improvement_percent |
| --- | --- | --- | --- | --- | --- |
| D10_F1_minus_F0 | 1.14738503 | 17.014962 | 2 | 18.9018056 | -13.4484195 |
| D21_F2_minus_F1 | -10.695826 | 12.2450644 | 3 | 0 | 41.8145307 |
| D20_F2_minus_F0 | -9.54844101 | 10.9074723 | 3 | 0 | 38.6833174 |

Negative new-minus-old favours new (the first/minuend model). This corrects the contradictory prose in the user instruction without changing its explicit subtraction formulas. No significance inference from 3 seeds.

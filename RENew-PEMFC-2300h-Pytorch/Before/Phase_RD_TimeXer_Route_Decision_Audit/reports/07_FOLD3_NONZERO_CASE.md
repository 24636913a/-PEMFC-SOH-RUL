# Fold 3 nonzero-current case

This formal TTD object means individual 30-minute target bins with nonzero current; it is not a claim that every full 24-step trajectory is continuously nonzero. The harm occurs in four independent temporal segments. In the cached object, state-correct `both_nonmixed` rows favor P2, while 54 predicted-mixed / true-nonmixed rows covering five unique target times dominate loss. This shows state errors co-occur with harmful DeltaC, but cannot isolate probability error, current magnitude error, or FiLM response.

State groups:

| state_group | state_correct | prediction_point_count | unique_target_timestamp_count | P1_RMSE_mV | P2_RMSE_mV | mean_D_mV2 | net_D_sum_mV2 | p95_abs_deltaC_mV |
|---|---|---|---|---|---|---|---|---|
| both_nonmixed | True | 6448 | 325 | 18.9271 | 12.3127 | -206.634 | -1.33237e+06 | 7.94341 |
| pred_mixed_true_nonmixed | False | 54 | 5 | 207.277 | 333.965 | 68569 | 3.70273e+06 | 215.329 |

Segments:

| temporal_segment_id | start_time | end_time | duration_h | unique_target_count | prediction_point_count | mean_D_mV2 | positive_D_sum_mV2 | negative_D_sum_mV2 | net_D_sum_mV2 | P1_RMSE_mV | P2_RMSE_mV | current_MAE_A | q_MAE | C_delta_P95_mV | C_delta_P99_mV | C_delta_max_mV | affected_horizons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 2025-06-03 13:06:10.005000 | 2025-06-10 13:12:24.005000 | 135.5 | 160 | 3512 | 373.839 | 1.43053e+06 | -117603 | 1.31292e+06 | 18.2182 | 26.5658 | 2.46255 | 0.00299232 | 6.99593 | 68.5954 | 158.323 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 |
| 14 | 2025-05-29 08:50:21.005000 | 2025-05-31 21:59:00.005000 | 49.5 | 60 | 1091 | -80.3546 | 1.06117e+06 | -1.14884e+06 | -87666.8 | 43.4635 | 42.5291 | 2.99171 | 0.00496461 | 26.3721 | 338.039 | 403.737 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 |
| 16 | 2025-06-27 18:12:08.005000 | 2025-06-29 13:04:55.005000 | 34.5 | 44 | 677 | 1325.53 | 942784 | -45403 | 897380 | 36.3292 | 51.4328 | 4.37535 | 0.0122956 | 9.5769 | 99.5249 | 137.201 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 |
| 13 | 2025-05-24 00:24:22.005000 | 2025-05-26 19:29:26.005000 | 54.5 | 64 | 1222 | 202.717 | 405250 | -157530 | 247720 | 20.4252 | 24.8979 | 2.10963 | 0.00120293 | 12.8293 | 100.12 | 225.393 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 |

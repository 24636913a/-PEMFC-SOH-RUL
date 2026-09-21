# Fair model comparison

| model | n_origins | mean_curve_MAE_mV | mean_curve_RMSE_mV | pooled_grid_RMSE_mV | j1_MAE_mV |
| --- | --- | --- | --- | --- | --- |
| M0_Persistence | 3 | 1.8342 | 1.8885 | 2.3644 | 1.8615 |
| M1_LinearTrend | 3 | 5.8845 | 6.0519 | 6.2277 | 5.6366 |
| M2_LSTM | 2 | 5.0159 | 5.0880 | 5.1564 | 4.7094 |
| M3_TimeXer | 2 | 3.0315 | 3.0968 | 3.2058 | 2.6480 |
| M4_FixedAverage | 2 | 2.9508 | 2.9829 | 3.1811 | 3.1080 |
| M4_Fusion | 1 | 5.3117 | 5.4348 | 5.4348 | 4.6162 |

All-model common-origin comparison is only 1500→1750 (n=1); it is a case comparison, not a generalization ranking. M2/M3 share two origins, whereas M0/M1 cover three. Never compare their unequal-coverage aggregate values as proof of superiority.

# Training report

Dynamic encoder resources and fit sizes:

| origin_stage | model | dynamic_train_windows | dynamic_validation_windows | parameter_count | best_epoch | device |
| --- | --- | --- | --- | --- | --- | --- |
| 850 | M1_LSTM | 0 | 0 | 0 |  | NOT_RUN_sparse_health_head |
| 850 | M2_TimeXer | 0 | 0 | 0 |  | NOT_RUN_sparse_health_head |
| 1150 | M1_LSTM | 1024 | 257 | 2065 | 18.000000 | cuda |
| 1150 | M2_TimeXer | 1024 | 257 | 15010 | 16.000000 | cuda |
| 1500 | M1_LSTM | 1291 | 323 | 2065 | 17.000000 | cuda |
| 1500 | M2_TimeXer | 1291 | 323 | 15010 | 18.000000 | cuda |

Prediction status counts:

| model | NOT_ESTIMABLE | RUN |
| --- | --- | --- |
| M0_linear_trend | 0 | 3 |
| M0_persistence | 0 | 3 |
| M1_LSTM | 1 | 2 |
| M2_TimeXer | 1 | 2 |
| M3_Fusion | 2 | 1 |
| M3_equal_weight_diagnostic | 2 | 1 |

The fitted health head is intentionally low capacity because each origin offers only 2–4 transitions with an available historical load-cycle window.

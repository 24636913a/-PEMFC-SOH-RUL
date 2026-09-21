# Training report

| origin_stage | model | status | reason | dynamic_train_windows | dynamic_validation_windows | purged_windows | parameter_count | best_epoch | device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 850 | M2_LSTM | NOT_RUN | sparse curve supervision |  |  |  |  |  |  |
| 850 | M3_TimeXer | NOT_RUN | sparse curve supervision |  |  |  |  |  |  |
| 1150 | M2_LSTM | RUN |  | 1024.00000 | 233.00000 | 24.00000 | 2065.00000 | 18.00000 | cuda |
| 1150 | M3_TimeXer | RUN |  | 1024.00000 | 233.00000 | 24.00000 | 15010.00000 | 18.00000 | cuda |
| 1500 | M2_LSTM | RUN |  | 1291.00000 | 252.00000 | 71.00000 | 2065.00000 | 18.00000 | cuda |
| 1500 | M3_TimeXer | RUN |  | 1291.00000 | 252.00000 | 71.00000 | 15010.00000 | 16.00000 | cuda |

Seed is fixed at 42; LSTM and TimeXer share the same dynamic task, data scope, early-stopping budget and purge rule.

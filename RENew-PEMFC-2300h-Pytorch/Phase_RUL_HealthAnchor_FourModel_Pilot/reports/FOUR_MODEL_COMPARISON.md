# Four-model comparison

| model | origins | MAE_mV | RMSE_mV |
| --- | --- | --- | --- |
| M0_linear_trend | 3 | 5.2908 | 5.5329 |
| M0_persistence | 3 | 1.9652 | 2.2842 |
| M1_LSTM | 2 | 8.1853 | 9.8379 |
| M2_TimeXer | 2 | 0.5265 | 0.5671 |
| M3_Fusion | 1 | 13.2036 | 13.2036 |
| M3_equal_weight_diagnostic | 1 | 6.9793 | 6.9793 |

Comparisons are descriptive only: models have unequal feasible-origin coverage because the health labels are sparse. No architecture is selected from these numbers.

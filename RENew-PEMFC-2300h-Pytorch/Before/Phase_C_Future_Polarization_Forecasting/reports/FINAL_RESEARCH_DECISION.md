# Final research decision

## 已证明的事实

The five-point common current-density-grid curve task is constructible for seven comparable normal nominal Development curves. Inputs, target-completion purge, fold-local scaling/PCA and Test firewall were enforced. However, M2/M3 lack estimable supervision at the first origin and M4 only has one all-model common origin.

| model | n_origins | mean_curve_MAE_mV | mean_curve_RMSE_mV | pooled_grid_RMSE_mV | j1_MAE_mV |
| --- | --- | --- | --- | --- | --- |
| M0_Persistence | 3 | 1.8342 | 1.8885 | 2.3644 | 1.8615 |
| M1_LinearTrend | 3 | 5.8845 | 6.0519 | 6.2277 | 5.6366 |
| M2_LSTM | 2 | 5.0159 | 5.0880 | 5.1564 | 4.7094 |
| M3_TimeXer | 2 | 3.0315 | 3.0968 | 3.2058 | 2.6480 |
| M4_FixedAverage | 2 | 2.9508 | 2.9829 | 3.1811 | 3.1080 |
| M4_Fusion | 1 | 5.3117 | 5.4348 | 5.4348 | 4.6162 |

All-model common origin (1500→1750, n=1):

| model | Curve_MAE_mV | Curve_RMSE_mV | j1_abs_error_mV |
| --- | --- | --- | --- |
| M0_Persistence | 1.1109 | 1.1717 | 1.3730 |
| M1_LinearTrend | 5.6542 | 5.7786 | 4.9526 |
| M2_LSTM | 5.7900 | 5.9249 | 4.9855 |
| M3_TimeXer | 2.1458 | 2.2680 | 1.1414 |
| M4_FixedAverage | 1.8221 | 1.8774 | 1.9221 |
| M4_Fusion | 5.3117 | 5.4348 | 4.6162 |

## 探索性观察

TimeXer has lower error than LSTM in both of their shared origins and at all five current-density points; its observed effect is therefore not restricted to j=1.0. It nevertheless does not beat curve persistence in the only all-model common 1500→1750 case (2.268 vs 1.172 mV Curve-RMSE). Fixed averaging is 1.877 mV in that same case, while learned fusion is 5.435 mV. These are case observations, not architecture-selection evidence.

## 尚未验证的假设

Whether dynamic encoders add stable curve-prediction value, whether LSTM/TimeXer are complementary, and whether performance transfers to independently held-out hardware remain unverified. This is neither true RUL nor EOL prediction.

## 唯一建议

**C — 寻找独立数据验证。** The chief limitation is the number of independent curve transitions, not curve-grid density or time alignment. Do not choose a paper core neural architecture from this pilot or add more fusion modules before independent, configuration-comparable curve sequences exist.

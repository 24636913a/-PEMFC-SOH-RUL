# 稀疏事件回放

| horizon | model | evaluation_class | n | MAE_SOH_pp | RMSE_SOH_pp | P95_abs_SOH_pp | max_abs_SOH_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | M0_REFERENCE_PERSISTENCE | Exploratory calendar replay | 3 | 0.2811724894230168 | 0.3268026154702232 | 0.4813093703819759 | 0.5116961246122042 |
| H1 | M1_RIDGE96_M2_NC | Exploratory calendar replay | 3 | 0.0797299804370414 | 0.0887214022892251 | 0.1128768492402754 | 0.114368499870654 |
| H3 | M0_REFERENCE_PERSISTENCE | Exploratory calendar replay | 3 | 0.2811724894230168 | 0.3268026154702232 | 0.4813093703819759 | 0.5116961246122042 |
| H3 | M1_RIDGE96_M2_NC | Exploratory calendar replay | 3 | 0.0323853745939906 | 0.0429048773460137 | 0.0658932752912144 | 0.0707141263833754 |
| M0_ALL_EVENTS | M0_REFERENCE_PERSISTENCE | Exploratory calendar replay | 6 | 0.5115824219424747 | 0.7028055500486265 | 1.2800014087760796 | 1.5361031701640382 |

逐点真值、预测、误差、真实起点见SOH_EVENT_PREDICTIONS.csv；身份与模型标签截止见P1S_TO_SOH_EXACT_FORECAST_LINKS.csv。M2_NC以最近完整P1S为校准算子，过去3/4/5锚点拟合，与AHC中位数算子区别明确。仅三事件、不假定在线发布时间，无统计显著性或泛化论证。

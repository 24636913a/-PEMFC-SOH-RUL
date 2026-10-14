# EOL与条件场景

EOL_NOT_ESTABLISHED。历史资料中确有条件特定500mV测量下限，但它不等于j=1参考SOH的终寿准则；约200周期后研究性完整表征支持EoT，而非已知failure。不能采用2300−t，也未认证右删失。

下表是以最后Development参考SOH为出发点，使用过去参考日历日期的固定OLS斜率、并设置零斜率恢复敏感性所得条件阈值时间。所有值为日历时间，不是运行小时RUL。全量post-BoL拟合、last3拟合和leave-one-out斜率包络是外推假设敏感性，不是置信区间。稀疏七参考不足以验证数年外推，超过365日的交点还标记out_of_plot_horizon；未达阈值不是无限真实寿命。

| model | threshold | threshold_is_assumption | conditional_crossing_calendar_h | conditional_crossing_calendar_days | reaches_within365_calendar_days | status | actual_failure_time | actual_RUL_accuracy | operating_hour_conversion_available |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| post_BoL_all_past_OLS | 0.97 | True | 471.2378592569261 | 19.63491080237192 | True | CONDITIONAL_EXTRAPOLATION_NOT_RUL_VALIDATION | nan | nan | False |
| post_BoL_all_past_OLS | 0.95 | True | 5751.480369131843 | 239.64501538049345 | True | CONDITIONAL_EXTRAPOLATION_NOT_RUL_VALIDATION | nan | nan | False |
| post_BoL_all_past_OLS | 0.9 | True | 18952.086643819104 | 789.670276825796 | False | CONDITIONAL_EXTRAPOLATION_NOT_RUL_VALIDATION | nan | nan | False |
| last3_past_OLS | 0.97 | True | 7136.114244428209 | 297.338093517842 | True | CONDITIONAL_EXTRAPOLATION_NOT_RUL_VALIDATION | nan | nan | False |
| last3_past_OLS | 0.95 | True | 87096.61200275838 | 3629.0255001149326 | False | CONDITIONAL_EXTRAPOLATION_NOT_RUL_VALIDATION | nan | nan | False |
| last3_past_OLS | 0.9 | True | 286997.85639858333 | 11958.24401660764 | False | CONDITIONAL_EXTRAPOLATION_NOT_RUL_VALIDATION | nan | nan | False |
| zero_slope_recovery_sensitivity | 0.97 | True | nan | nan | False | NO_CROSSING_UNDER_FLAT_ASSUMPTION | nan | nan | False |
| zero_slope_recovery_sensitivity | 0.95 | True | nan | nan | False | NO_CROSSING_UNDER_FLAT_ASSUMPTION | nan | nan | False |
| zero_slope_recovery_sensitivity | 0.9 | True | nan | nan | False | NO_CROSSING_UNDER_FLAT_ASSUMPTION | nan | nan | False |

固定过去日历斜率的三事件回测MAE为0.279603 SOH百分点，M0为0.281172，平均差极小；1500恢复事件的斜率误差0.754331明显大于M0的0.123993。它未建立稳定长期趋势优势。

同一95%假设阈值，全post-BoL斜率为239.65日、last3为3629.03日，零斜率不穿越。巨大的条件差异说明寿命不可识别，而不是已得到可信数年寿命估计。绘图区仅365日，更远解析交点只做敏感性证据，不能作为推荐寿命。

未使用Test验证轨迹或阈值，真实RUL准确率NA。条件轨迹仅展示不可辨识寿命的假设依赖，不能视为长期模型已合格；也未将M1短周期预测直接外推到这些长期情景。

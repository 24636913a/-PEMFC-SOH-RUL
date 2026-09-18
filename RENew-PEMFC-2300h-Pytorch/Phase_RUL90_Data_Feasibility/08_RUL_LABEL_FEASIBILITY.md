# RUL 标签可行性

本数据属于 **情况 B：有可追溯的稀疏参考 SOH，但完整观测未提供可确认的 90% EOL**。因此它在 90% 规则下是右删失（right-censored），没有真实点值 RUL 标签，也不能计算真实 EOL RUL 的 MAE/RMSE。

对于每个仍高于阈值的 reference checkpoint，最多可声明下界 `RUL(t) > t_last_valid - t`；其中最终正常 reference point 是 `2300 h`。该下界不是 RUL ground truth。若后续选择 90% 外推，应明确为 `extrapolated RUL`，需要额外 run-to-failure 或独立 EOL 数据验证。

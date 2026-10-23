# 冻结评价指标

Authority：C_CODE `metric`；I_METRIC `independent_metrics`/`diagnostics`。本轮只静态审查，合成测试用于公式，不读取或生成真实 Test metrics。

一个模型/Seed的键 `(origin_row,horizon_step,target_row)`，h=1..24，target=origin+h。跨数据源/region必须加入唯一source/region标识，不能把不同源同row合并。每模型固定相同全集Q，无重复，无按误差删query。真值 `V_target - V_origin`，预测/真值单位V；error=prediction−truth，展示×1000mV。

## 主指标

`forecast_query_weighted_RMSE_mV = 1000 sqrt(sum_Q(error²)/|Q|)`。

24步全部保留；重叠查询仍按原单位等权。不新定义cycle/rare-state权重。

## 强制敏感性

`target_time_equal_weight_RMSE_mV = 1000 sqrt(mean_target(mean_queries_for_target(error²)))`。

均值发生在**平方误差**空间，不先平均同一target多个预测值。ΔV因origin anchor不同，同target的ΔV真值也可能不同，必须各query先对自己的truth计算error，不能替换为单个target ΔV truth。

## 同时固定

- MAE = mean(|error|)×1000mV。
- R² = 1−SSE/sum_Q((truth−mean_Q(truth))²)；constant truth时undefined，不假造0/1。
- h1..h24各步RMSE、h24（12h endpoint）RMSE。
- P95/P99 = 所有query absolute error的linear quantile，index=(n−1)q。
- max absolute error、origin数、query数、唯一target时刻数。
- target SSE = sum对应query(error_mV²)；max-target share和top5-target share；总SSE=0时share undefined。贡献加和只在SSE/MSE空间，不相加RMSE差。

指标按每checkpoint独立计算。异常/重叠/窗口极端误差必须保留；诊断不代表独立物理失效机制。敏感性、endpoint、P99、concentration均不替代主指标；不执行新阈值搜索。No significance/bootstrap by overlap queries。

新阶段 `synthetic_metrics(...,synthetic=True)` 仅能接收合成fixture；默认拒绝真实指标入口。它不是Test runner。

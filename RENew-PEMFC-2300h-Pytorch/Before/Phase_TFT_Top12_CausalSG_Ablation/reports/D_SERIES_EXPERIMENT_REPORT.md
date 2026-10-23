# Phase D — Top12 特征扩展与因果 SG 消融实验

## 1. 实验目的

本轮在相同 RealHyFC Development 单次时间顺序 7:1:2 划分上，分离考察 Top10→Top12 特征掩码与单侧因果 SG(7,2) 的影响。它是已暴露 Assessment 上的 Seed42 探索性控制变量实验，不是独立盲测或新的 backbone 排名。

## 2. 数据与安全协议

仅通过 `pd.read_csv(nrows=3007)` 读取混合表，过滤后保留 3005 个 Development rows。Train/Validation/Assessment 分别为 [0,2103)、[2103,2404)、[2404,3005)，其合法窗口数为 {'train': 1380, 'validation': 205, 'assessment': 378}。所有目标集合隔离；未访问 Test、未来协变量或 Test 标签。

## 3. 模型、公共 scaler 与筛选

CanonicalStrictHistoryTFT 未改变，参数来自历史 T1-R Fold2。四组参数量均为 323483。公共 scaler 仅拟合未滤波 Train histories，公共 ΔV scale 为 0.13680319 V；SG 只变换历史平均电压通道，之后仍使用该 scaler。

Top10：cell_voltage_mean, cell_voltage_min, zero_current_fraction, load_transition_count, cell_voltage_max, actual_current_density_A_cm2_std, actual_current_A_std, actual_current_A_peak_to_peak, actual_current_density_A_cm2_peak_to_peak, actual_current_density_A_cm2_mean

Top12：cell_voltage_mean, cell_voltage_min, zero_current_fraction, load_transition_count, cell_voltage_max, actual_current_density_A_cm2_std, actual_current_A_std, actual_current_A_peak_to_peak, actual_current_density_A_cm2_peak_to_peak, actual_current_density_A_cm2_mean, actual_current_A_mean, cell_voltage_std

新增：actual_current_A_mean, cell_voltage_std。这些 Spearman 仅描述定义为 ΔV 的训练关联，不能解释为退化机理重要性。

## 4. 结果

|组别|RMSE mV|MAE mV|R²|12h endpoint RMSE mV|P95 mV|P99 mV|Target-equal RMSE mV|
|---|---:|---:|---:|---:|---:|---:|---:|
|D0|13.1692|3.6136|0.9955|4.2123|8.4650|14.1913|12.9765|
|D1|17.9818|7.0542|0.9915|17.7381|17.0383|27.7403|17.7848|
|D2|11.0729|6.0091|0.9968|9.9160|16.4795|26.4799|11.2192|
|D3|27.1713|8.5733|0.9807|9.1614|18.8279|68.1797|26.6058|

## 5. 对照解释

- **D1 vs D0**：Top12 的 RMSE 变化为 +4.8126 mV；应同时结合 MAE、endpoint 与 P99 判断，而非仅凭 RMSE。
- **D2 vs D0**：在 Top10 下，SG 的 RMSE 变化为 -2.0964 mV。
- **D3 vs D1**：在 Top12 下，SG 的 RMSE 变化为 +9.1894 mV。
- **D3 vs D2**：在 SG 下，Top12 的 RMSE 变化为 +16.0984 mV。

## 6. 历史 C1 复现性

D0 RMSE 为 13.169241 mV，历史 C1 为 11.445854 mV；差值为 +1.723387 mV。D0 使用同一排序、窗口、原始 Train scaler、初始化与训练算法；任何差异仅作运行环境/数值确定性审计，不用于再训练。

## 7. 失败案例与局限

逐 target-row、horizon 与 Top10 极端误差均已保存。重叠窗口不被当作独立样本，未做显著性声称，也未删除 row 2952 或其他高误差事件。本轮只含一次时间切分和 Seed42；Assessment 已被先前查看，故不能视为独立泛化证明。短期 ΔV 精度也不等价于 SOH 或 RUL 精度。

## 8. 最终证据状态

`TOP12_SG_COMPLEMENTARITY_NOT_ESTABLISHED`。D0–D3 均已完成真实训练、独立缓存复算与因果 SG 审计。到此停止，不启动 Top13/Top14、SG 参数搜索、Optuna、Test 或其他模型。

## 9. 逐项控制变量判断

- **D1 vs D0（Top12 是否优于 Top10）**：D1 相比 D0 的 RMSE 增加 4.8126 mV、MAE 增加 3.4407 mV、12 h endpoint RMSE 增加 13.5258 mV、P99 增加 13.5490 mV。新增 `actual_current_A_mean` 与 `cell_voltage_std` 在本固定切分中没有带来收益。
- **D2 vs D0（SG 是否改善 Top10）**：D2 的 RMSE 降低 2.0964 mV，但 MAE 增加 2.3955 mV、endpoint RMSE 增加 5.7038 mV、P99 增加 12.2886 mV。因此 SG 的 RMSE 改善没有转化为跨主要指标的一致优势，不能称为稳健附加收益。
- **D3 vs D1（Top12 下 SG 是否有效）**：D3 的 RMSE、MAE 与 P99 均恶化，分别增加 9.1894、1.5191 与 40.4393 mV。SG 未能挽回 Top12 的表现。
- **D3 vs D2（SG 下增加两个特征是否有效）**：D3 比 D2 的 RMSE 增加 16.0984 mV、P99 增加 41.6997 mV，未观察到特征扩展与 SG 的互补收益。

## 10. 历史 C1 复现性审计

尽管 D0 使用与历史 C1 匹配的任务、窗口数、Train-only 原始 scaler 规则、Top10 排名、Seed42、Canonical TFT 与冻结超参数，D0 RMSE 为 13.1692 mV，历史 C1 为 11.4459 mV，差异为 +1.7234 mV。`D_C1_REPRODUCIBILITY_AUDIT.json` 记录了这些一致性检查。严格逐数值复现未建立；本轮没有为解释该差异而额外重训、换种子、改超参数或删样本。因此，这一跨阶段比较只能作为复现性审计，不应把 D0–D3 的差异与 C 系列偶然训练差异混为一谈。

## 11. 误差集中与重叠窗口

`D_ERROR_ANALYSIS.csv` 保留四组 447 个 target row 的平均绝对误差、24 个 horizon RMSE，以及各组前 10 大绝对误差。所有组均完整保留高误差样本；该文件也明确同一 target time 的多 origin 预测是重叠、相关观测，未被作为独立样本做显著性判断。D3 的最大绝对误差达到 499.88 mV，说明其组合方案在尾部事件上尤其不稳。

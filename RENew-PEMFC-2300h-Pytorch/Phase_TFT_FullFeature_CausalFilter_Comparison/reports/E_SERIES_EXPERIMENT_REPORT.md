# Phase E — TFT 全特征与因果滤波联合对照实验

## 1. 研究目的

本研究在固定的 RealHyFC Development 7:1:2 时间顺序划分上，系统比较 Top10 与全部 Raw14 输入特征，以及无滤波、单侧因果 SG(7,2) 和逐 48-step 窗口因果 EWMA(α=0.3)。目的是在公共 scaler 下检查特征数量与因果滤波是否存在探索性互补。

## 2. 数据与实验协议

混合 causal 表只以 `pd.read_csv(nrows=3007)` 打开，再按 `bin_end_ophrs2_h < 1819.997738` 过滤为 3005 个 Development rows。行区间为 Train [0,2103)、Validation [2103,2404)、Assessment [2404,3005)，合法窗口数为 {'train': 1380, 'validation': 205, 'assessment': 378}。Assessment 已在历史 C/D 阶段暴露，故状态为 `DEVELOPMENT_EXPLORATORY_REUSED_ASSESSMENT`，并非新的独立盲测。未访问 Test。

## 3. TFT 模型及公共标准化

六组均为独立初始化的 Seed42 CanonicalStrictHistoryTFT，固定 T1-R Fold2 超参数；模型参数量均为 323483。公共 scaler 仅由未滤波合法 Train 历史拟合，目标 ΔV scale 为 0.136803187141 V；滤波只发生在输入、公共 scaler 之前。SG 和 EWMA 的未来扰动因果性测试均通过。

## 4. 特征配置与滤波

Top10：cell_voltage_mean, cell_voltage_min, zero_current_fraction, load_transition_count, cell_voltage_max, actual_current_density_A_cm2_std, actual_current_A_std, actual_current_A_peak_to_peak, actual_current_density_A_cm2_peak_to_peak, actual_current_density_A_cm2_mean。Top10 使用标准化空间零掩码，Raw14 保留所有 14 个输入通道；两者模型输入维度均保持 14，因而参数量不变。

## 5. 六组结果

|模型|特征|滤波|RMSE(mV)|MAE(mV)|R²|
|---|---|---|---:|---:|---:|
|E0|Top10|Raw|19.3188|5.0763|0.9902|
|E1|Top10|SG(7,2)|20.7352|5.2133|0.9888|
|E2|Top10|EWMA(0.3)|7.7858|5.4073|0.9984|
|E3|Raw14|Raw|20.9004|5.5641|0.9886|
|E4|Raw14|SG(7,2)|27.1091|7.7076|0.9808|
|E5|Raw14|EWMA(0.3)|14.1650|5.4024|0.9948|

## 6. 对照分析

- E0 vs E3 (Raw): Raw14−Top10 = +1.5816 mV
- E1 vs E4 (SG): Raw14−Top10 = +6.3739 mV
- E2 vs E5 (EWMA): Raw14−Top10 = +6.3792 mV
- Top10 SG−Raw = +1.4164 mV; EWMA−Raw = -11.5329 mV
- Raw14 SG−Raw = +6.2088 mV; EWMA−Raw = -6.7354 mV

RMSE 是唯一预注册主排序指标；本轮最低 RMSE 为 **E2**，对应状态 `TOP10_EWMA_LOWEST_EXPLORATORY_RMSE`。MAE 和 R²仅作辅助报告，未被用于改写 RMSE 排名。单次 Seed42 的细小差异不能作为稳定性能优势或物理机理证据。

## 7. 历史数值复现性

历史 C1 与 D0 的记录差异已单独审计，结论为 `C1_D0_STRICT_REPRODUCIBILITY_UNRESOLVED`。C 的条件各自 scaler 与 D/E 的公共 scaler 口径不同，且历史元数据不足以严格锁定所有执行级数值差异。Phase E 仅以 E0–E5 的同阶段、同 scaler 新训练结果排序，不跨阶段拼接改善幅度。

## 8. 实验局限性

本阶段只有单次时间顺序切分和 Seed42；Assessment 被历史阶段复用；滑动窗口令同一 target time 可被多个 origin 预测；没有独立 Test 泛化结论。任务是短期 48→24 ΔV 预测，而非 SOH 或 RUL 预测。

## 9. 研究结论

`PHASE_E_DEVELOPMENT_EXPLORATORY_ONLY`；`TOP10_EWMA_LOWEST_EXPLORATORY_RMSE`。本阶段至此收口，不自动搜索其他特征数、滤波参数、backbone、Optuna、额外随机种子或 Test。

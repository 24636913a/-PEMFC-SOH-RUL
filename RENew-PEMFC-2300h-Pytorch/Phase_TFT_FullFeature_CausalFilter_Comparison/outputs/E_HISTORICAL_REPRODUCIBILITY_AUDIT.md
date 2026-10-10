# Phase E 历史复现性只读审计

C1（Top10、Raw）的记录 RMSE 为 11.445854 mV，D0（Top10、Raw、公共未滤波 scaler）的记录 RMSE 为 13.169241 mV，差异为 +1.723387 mV。

- Top10 排名与 D 阶段一致：`True`。
- 两阶段均记录了相同的 3005-row Development 前缀、7:1:2 行边界、48→24 任务、Seed42 和 CanonicalStrictHistoryTFT。
- C 采用条件各自拟合 scaler；D 采用公共未滤波 Train scaler，故 scaler 口径本身已不同。
- 现存元数据不足以从历史 checkpoint 精确归因全部数值差异；状态为 `C1_D0_STRICT_REPRODUCIBILITY_UNRESOLVED`。

Phase E 不用历史 C/D 数值选优，六组均从独立初始化重新训练，并用单一公共 scaler 比较。

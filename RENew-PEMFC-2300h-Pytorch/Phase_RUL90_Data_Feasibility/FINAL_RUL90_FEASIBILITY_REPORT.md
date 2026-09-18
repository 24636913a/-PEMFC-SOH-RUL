# FINAL RUL90 数据可行性报告

## 最终判断

RealHyFC 2300 h 数据可以支撑以固定名义工况极化曲线构造的稀疏 SOH/健康电压研究，**不能直接支撑“以 90% 初始参考电压为已观测真实 EOL”的监督 RUL 论文**。原因不是 90% 定义本身不合法，而是完整观测没有提供/确认该阈值的 EOL event；在本冻结定义下，它是右删失而非 run-to-failure 标签数据。

冻结方法先在 Development 完成，随后完整 2300 h 数据只作为 `FULL_DATA_EXPLORATORY_DESCRIPTIVE_ANALYSIS` 使用。故 `full_data_target_accessed=true`、`test_seal_for_future_model_selection=compromised`，但 `test_prediction_metrics_computed=false`、`model_training_performed=false`。

## 核心数值

| 项目 | 数值 |
| --- | --- |
| 冻结 j* | nominal polarization, active 10-cell median, frozen j*=1 A/cm2 |
| 初始 V_ref | 0.698953307 V |
| 末端 V_ref | 0.677983895 V |
| 最小有效 SOH | 0.96949682 |
| 最大确认退化 | 3.05032% |
| 90% 阈值 | 0.629057976 V |
| 90% 是否观测到 | False |
| 95% 是否观测到 | False |
| 真实 RUL 标签 | False |

## 必须回答的 15 个研究问题

1. **实际运行小时**：OpHrs2 覆盖 72.7887–2256.8 h，跨度 2184.01 h。
2. **官方 EOL**：NOT_FOUND；仅检出 lower-voltage-limit 文字证据。
3. **实验结束是否等于失效**：不能确认，不能把约 2300 h 自动等同 failure。
4. **性能诊断点**：正常名义 polarization reference checkpoints 共 9 个（含 BoL 至 2300 h）。
5. **周期性固定工况**：有：名义极化曲线，适合固定 j* 的稀疏参考测量。
6. **V_ref**：可构建，条件是仅使用冻结 j*、非 Air-Air-start、同构名义 polarization。
7. **初始参考**：0.698953307 V，source-labelled BoL，需保留非零 OpHrs 的限定。
8. **末端参考**：0.677983895 V，最后正常 checkpoint。
9. **最大确认退化**：3.05032%（以冻结 reference 定义）。
10. **90%**：False。
11. **95%**：False。
12. **真实 RUL**：不可计算；无已观测 90% EOL point label。
13. **可验证性**：可验证 reference voltage/SOH observation 与短时动态电压；不可验证真实 EOL RUL 误差。
14. **12h TimeXer 缺口**：其输出不是 reference voltage；缺少慢 health-state、EOL observation 与未来场景定义。
15. **唯一研究方向**：reference-condition SOH / long-horizon health-voltage forecasting，而非 supervised true-RUL。

## 关键图表

- `time_coverage.png`
- `time_segments.png`
- `current_voltage_vs_age.png`
- `voltage_distribution_by_stage.png`
- `cell_voltage_dispersion_vs_age.png`
- `reference_voltage_vs_age.png`
- `soh_vs_age.png`
- `polarization_curves_reference_points.png`
- `timexer_12h_deltaV_development.png`

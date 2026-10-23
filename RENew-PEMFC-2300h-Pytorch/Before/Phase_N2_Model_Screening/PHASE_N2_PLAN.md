# Phase N2 — 强基线、历史长度诊断与现代时序模型筛选

英文名称：Phase N2 — Strong Baselines, Temporal-Diagnostic Analysis and Modern Time-Series Model Screening

本文件在任何 Phase N2 模型训练前冻结。后续不得根据 Validation（验证集）结果临时改变比较协议。

## 1. 数据源与只读边界

- 主数据：`Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv`。
- 1 h 稳健性数据：`Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_1h.csv`。
- 主老化坐标：`OpHrs2 [h]`。
- 主电压：由 U1–U5、U7–U11 得到的 `cell_voltage_mean`。
- `U6` 永久排除；`exported_average_cell_voltage` 仅用于一致性检查。
- Phase N1、旧项目和原始 2300 h 数据在本阶段均为只读。

## 2. Test firewall（测试集防火墙）

- 总体 Train：`[72.788690, 1383.195476) h`。
- 总体 Validation：`[1383.195476, 1819.997738) h`。
- 最终 Test：`[1819.997738, 2256.800000] h`。
- Phase N2 仅允许读取 Test schema、行数、时间范围、缺失结构和 Phase N1 已生成的合法窗口数量。
- 所有建模脚本在构造 sequence target（序列目标）前强制截断到 `OpHrs2 < 1819.997738 h`。
- 禁止构造或计算 Test 的预测、MAE、RMSE、R²、Skill Score、误差分布、per-horizon 结果或轨迹图。
- 所有图仅使用 development folds（开发折）的 Validation 数据。

## 3. Expanding-window validation（扩展窗口验证）

边界均相对完整 `OpHrs2` span 冻结：

| Fold | Train | Validation |
| --- | --- | --- |
| Fold 1 | `[72.788690, 946.393214)` | `[946.393214, 1164.794345)` |
| Fold 2 | `[72.788690, 1164.794345)` | `[1164.794345, 1383.195476)` |
| Fold 3 | `[72.788690, 1383.195476)` | `[1383.195476, 1819.997738)` |

每个 fold 的 scaler（缩放器）和缺失值填补统计量仅由该 fold Train 拟合。窗口必须完全处于对应 Train 或 Validation 区间，并完全处于同一个 Phase N1 `temporal_segment_id`，且所有 bin `bin_valid_for_windows=True`。

## 4. 分辨率与预测任务

- 主分辨率：0.5 h。
- 1 h 仅在最终候选确定后进行相同物理尺度的稳健性检查，不用于重新选模型。
- 目标：Direct Multi-Horizon Residual Forecasting（直接多步残差预测）。
- 标签：`ΔV(t+h)=V(t+h)−V(t)`。
- 重建：`V_hat(t+h)=V(t)+ΔV_hat(t+h)`。
- 严格禁止 future covariates（未来协变量）；模型只能使用 prediction origin（预测起点）及以前的变量。

## 5. 物理时间尺度

- 默认 Lookback（历史回看长度）：24 h；0.5 h 下 `L=48`。
- Lookback 诊断：12/24/48 h，对应 `L=24/48/96`。
- 主 Forecast Horizon（预测时距）：12 h；0.5 h 下 `H=24`。
- Horizon sensitivity（时距敏感性）：4/8/9.5/12/24 h。
- 9.5 h 对应 19 steps，用于“预测下一个完整经验负载循环”的诊断。

## 6. Matched-Origin（匹配预测起点）

12/24/48 h Lookback 比较以 48 h 历史确定共同合法 origin。短历史只截取同一序列的末 12/24 h。三种配置必须具有完全相同的 Train/Validation origins、target timestamps、anchor `V_t` 和 future ΔV labels；任一不一致均 FAIL。

## 7. Feature Groups（特征组）

- F1 — Voltage State（电压状态）：5 个 active-cell voltage features。
- F2 — Voltage + Load（电压+负载）：F1 加 actual current/current-density 波动和运行点转换特征。
- F3 — Condition-Aware（运行条件感知）：F2 加物理意义明确的 actual temperature、pressure、RH、flow、stoichiometry 核心均值。
- 不使用 214 列全集；不加入 set point、metadata、diagnostic-only、future-known-only、EIS、polarization 或 spatial EIS。
- 精确列名冻结在 `PHASE_N2_FEATURE_GROUPS.md` 和配置文件中。

Feature-group GO 规则：F2 相对 F1、F3 相对 F2 均要求平均 RMSE 改善至少 2%、至少 2/3 folds 改善、且无 fold 恶化超过 3%。Ridge 和 LSTM 结论冲突时保留更简单组，并报告 `feature contribution unresolved（特征贡献尚未完全解决）`。

## 8. 强基线

- B0 Persistence（持续性）。
- B1 Local Linear Trend（局部线性趋势）。
- B2 Cycle Repeat / Seasonal Naive（周期重复/季节朴素），固定周期 19 steps。
- B3 Ridge Lag Regression（岭回归滞后），候选 `alpha=[0.01,0.1,1,10]`，仅在 development folds 选择。
- B4 state-space 不强制；仅在实现稳定且不影响主要流程时加入。

所有深度模型必须在强基线完成后才运行。

## 9. Lookback 诊断

使用冻结特征组，以 Ridge 和 LSTM 比较 12/24/48 h，并使用 Matched-Origin。24 h 保持默认；48 h 只有在平均 RMSE 改善至少 2%、至少 2/3 folds 改善、无 fold 恶化超过 3%，且 moving-block bootstrap（移动块自助法）支持时才可替换。12 h 与 24 h 差异小于 2%且不稳定时保留 24 h。

## 10. 现代模型筛选

- M0 LSTM（长短期记忆网络）。
- M1 TCN（时间卷积网络）。
- M2 iTransformer（倒置 Transformer）。
- M3 TimeMixer（时间多尺度混合模型）。
- M4 PatchTST（分块时间序列 Transformer）。

采用本地、清晰、可审计的 PyTorch 实现，统一 sequence builder、scaler、loss、early stopping 和评估管线。禁止增加其他模型或 ensemble（集成）。

## 11. 统一训练设置

- 第一轮随机种子：42。
- Top-2 和 LSTM 参考种子：`[42,2024,3407]`。
- optimizer：AdamW。
- learning rate：`1e-3`。
- weight decay：`1e-5`。
- batch size：64。
- max epochs：300。
- early-stopping patience：30。
- loss：MSE on multi-horizon voltage increments。
- 只允许配置文件中预先写明的小规模架构候选；不运行 Optuna。

## 12. 评价与统计

报告 reconstructed voltage（重建电压）的 RMSE（V/mV）、MAE、Bias、R²、Skill_Persistence、Skill_Cycle 和 Skill_Best。必须输出每 0.5 h step 的 per-horizon metrics（逐时距指标）。

关键比较使用 paired moving-block bootstrap（配对移动块自助法），block length=19 steps，重复 2000 次，报告 mean/median ΔRMSE 和 95% CI。

## 13. 筛选与停止规则

第一轮按三个 folds 的 Mean RMSE、Mean Skill_Best、Worst-fold Skill_Best、per-horizon stability 和参数量选 Top-2。Top-2 与 LSTM 运行至少三个种子。现代模型只有在 Mean Skill_Best>0、至少 2/3 folds Skill_Best>0、且多种子平均 Skill_Best>0 时，才可称为增加预测价值。

若所有 modern models（现代模型）`Skill_Best<=0`，立即给出 NO-GO，不增加模型。性能差异小于 1%时优先更简单、参数更少、更稳定和更易解释的模型。

完成 baselines、特征组、Matched-Origin Lookback、现代筛选、Top-2 多种子、horizon sensitivity、1 h robustness、bootstrap 和 frozen config 后停止。禁止进入 Final Test、RUL、uncertainty、PINN、ensemble、EIS/spatial EIS 融合或 Phase N3。

## 14. GO 条件

- baseline、fold、causal 和 matched-origin audits 全部通过；
- Test 防火墙无违规；
- 至少一个 learned model 满足现代模型 GO 条件；
- 最终 backbone 的配置、特征、Lookback、Horizon、scaler 和 seeds 可冻结。

不满足任一条件则 Phase N2 为 FAIL/NO-GO。


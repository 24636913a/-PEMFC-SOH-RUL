# RENew-PEMFC-2300h-Pytorch

2300 h PEMFC 耐久性数据研究工作区。

## 已完成阶段

### Phase N1 — 数据基础、采样分辨率与 SOH 目标构建

- 主 ageing coordinate（老化坐标）：`OpHrs2 [h]`。
- 有效单体：U1–U5、U7–U11；排除 U6。
- 主采样率：0.5 h；1 h 用于稳健性比较。
- 总体 Train/Validation/Test：chronological 60%/20%/20%。
- 主 SOH anchor：`j*=1.0 A/cm²`；次 anchor：`1.6 A/cm²`。
- 结果：`Phase_N1_Data_Foundation/PHASE_N1_RESULTS.md`。

### Phase N2 — 强基线、历史长度诊断与现代时序模型筛选

- Development validation：三个 expanding-window folds（扩展窗口折）。
- Feature Group：F2（电压+负载），14列。
- Lookback：24 h；Primary Horizon：12 h。
- 最强简单基线：Ridge Lag Regression（岭回归滞后模型）。
- Top-2 modern candidates：PatchTST、iTransformer。
- 最终 deterministic backbone（确定性预测骨干）：LSTM。
- LSTM 多 seed Mean Skill_Best：约0.692。
- 结果：`Phase_N2_Model_Screening/PHASE_N2_RESULTS.md`。
- 冻结配置：`Phase_N2_Model_Screening/outputs/frozen_config/PHASE_N2_FROZEN_CONFIG.yaml`。

## 约束

- 原始数据和旧 `New-PEMFC-Pytorch` 项目均为只读。
- 最终20% Test 在 Phase N1/N2 中只允许结构审计；Phase N2 未生成任何 Test prediction 或 Test metric。
- 尚未执行 Phase N3、RUL、uncertainty、ensemble、PINN、EIS/spatial EIS 融合或 dense SOH interpolation。

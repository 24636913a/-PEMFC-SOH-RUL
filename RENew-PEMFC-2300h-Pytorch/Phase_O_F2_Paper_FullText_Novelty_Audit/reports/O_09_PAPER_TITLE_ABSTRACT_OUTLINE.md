# O-09 — 条件性论文标题、摘要与大纲

## 候选标题

中文：**基于历史工况感知双路径时间融合Transformer的质子交换膜燃料电池动态多步电压预测**。

1. **History-Regime-Aware Dual-Path Temporal Fusion Transformers for Dynamic Multi-Step PEMFC Voltage Forecasting**
2. **Dynamic PEMFC Voltage Forecasting with a Dual-Path TFT under Unknown Future Operating Conditions**
3. **A Protocol-Aware Empirical Study of Dual-Path Temporal Fusion Transformers for PEMFC Voltage Forecasting**

第三个最符合当前“增量设计+协议实证”证据。没有first、novel physical mechanism、generalizable SOH、RUL或PINN。

## 中文摘要草案（非完整论文，须随最终证据组织校核）

动态工况下的质子交换膜燃料电池电压同时受到运行状态与长期变化影响，未来真实工况在部署时通常不可得。本研究在RealHyFC记录上定义仅使用过去24h信息、直接预测未来12h电压增量的任务，研究保留TFT主体的历史双路径编码。LSTM与因果多尺度深度卷积共享历史变量选择输入，历史工况统计控制encoder特征融合；未来decoder仅使用相对预测步长。共同冻结预处理及三个随机种子的既有Development结果中，F2 Validation RMSE均值为7.2454mV，F0/F1为9.0676/9.1556mV。整体结果支持该配置的应用价值，但卷积分支独立收益不稳定，学习门控相对固定融合尚未证明独立增益，尾部/端点及目标时间等权评价显示区域敏感性。Assessment已复用，且没有合格的独立确认性Test，因此结论限于Development动态电压预测，不涵盖物理机理、SOH或RUL验证。

## English abstract draft

Dynamic PEMFC voltage reflects both operating-state variation and longer-term changes, while actual future operating conditions are unavailable at deployment. This study examines a history-regime-aware dual-path TFT on RealHyFC, using 24 hours of historical information to directly forecast voltage increments over the next 12 hours. The LSTM and causal multi-scale depthwise-convolution paths share the historical variable-selection representation and are fused within the encoder; the future decoder receives only a relative horizon index. Under the existing frozen preprocessing and three-seed Development protocol, the mean Validation RMSE of F2 is 7.2454 mV, compared with 9.0676 and 9.1556 mV for F0 and F1. These results support the combined configuration within the studied Development setting, but the independent contribution of learned fusion is not established. Tail, endpoint and target-time-weighted diagnostics reveal sensitivity that aggregate error alone does not capture. Reused Assessment data and the absence of an eligible independent confirmatory Test limit the conclusions to dynamic voltage forecasting, without validated physical-mechanism, SOH or RUL claims.

## 七章大纲及真实证据文件

所有历史路径相对研究根目录；O报告相对本阶段。本稿不读取或重算预测cache。

|章节|拟写内容|已经存在的证据|
|---|---|---|
|1 引言与相关研究|动态电压任务、现有TFT/混合模型、有限贡献|O账本、O-01/02/03/05及各本地原PDF DOI/页码；不以旧总结为事实|
|2 RealHyFC数据与任务|10-cell；0.5h；48→24 ΔV；数据范围和split|`Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml`；`Phase_L_Test_Eligibility_and_Reproducibility_Freeze/reports/L_FINAL_EVALUATION_READINESS_DECISION.md`；不重开N1|
|3 预处理及信息边界|14slots/Top10、因果EWMA、Train-only scaler、decoder索引|H `scripts/h_common.py`；`Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py`及`outputs/E_COMMON_SCALER.json`；F `models/frozen_feature_schema.py`；O-04|
|4 双路径TFT|VSN共享、LSTM/causal CNN、post-LSTM融合、gate公式、参数量|`Phase_TFT_Original_Baseline/models/canonical_tft.py`；F四models源码；J `outputs/architecture/J_PARAMETER_COUNTS.csv`；O身份报告|
|5 实验及公平性|同split/预处理、seeds42/52/62、budget与early stop、主/敏感指标|H协议；L frozen manifest；I2 checkpoint manifest；H repeatability报告；O-07（不写成三fold）|
|6 结果与消融|九Validation、Assessment、horizon/endpoint/tails；F1不稳；gate未证实|H `outputs/metrics/H_VALIDATION_METRICS.csv`/`H_MULTISEED_STATISTICS.csv`；I2 `outputs/metrics/I2_ASSESSMENT_METRICS.csv`/`I2_MULTISEED_SUMMARY.csv`及`outputs/diagnostics/I2_ALL_HORIZON_METRICS.csv`；J fixed_gate与gate_statistics表|
|7 讨论/局限/结论|已有思想、弱化gate、重叠/极端误差、Test资格、非SOH/RUL、物理暂停|J最终决策；K `reports/K_FINAL_RESEARCH_RECOMMENDATION.md`；L最终决策；M/M1/M2最终决策；N `N_FINAL_TRAINING_READINESS_DECISION.md`；O-08/10|

## 图表使用和待办

架构/信息边界图可在写稿时根据自有源码重绘；本轮只提供张量/公式，不生成新指标图。原论文图不用于投稿插图。结果图若引用历史文件，须先核对H九checkpoint身份；目前九模型Validation完整horizon/尾部图不齐，B类需求另行授权。没有这部分时不能让摘要声称全horizon稳定。

先写条件性初稿，不写强确认结论。缺失证据用明确待办，不用想象数字填空；模型、超参、seed、评分口径不再为论文美观而改动。

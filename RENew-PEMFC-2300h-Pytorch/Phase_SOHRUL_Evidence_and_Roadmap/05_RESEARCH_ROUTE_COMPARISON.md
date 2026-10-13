# 三条研究路线比较与明确排序

本文件给出项目内证据驱动的研究判断，不是新模型结果或“全球首创”文献结论。既有 Development 已用于多次方案制定，后续研究均须披露探索性背景；本轮不利用 ELTE 两点误差挑方案。

## 1. 三路线的九项比较

|维度|A：P1S-B2/因果 P1S 健康代理|B：稀疏极化 reference-condition SOH|C：条件退化与寿命情景|
|---|---|---|---|
|唯一问题能否明确|可以：健康代理及其因果可获得性/预测区间，不等同真实 SOH|可以：合法历史能否估计独立参考条件性能|需先固定失效/假设阈值、未来工况与外推域|
|现有数据是否够|够描述 141 周期；131 post-reference 事件可作有限单堆探索；不足跨堆确认|7 normal Dev 点、6 post-BoL 校准点不足复杂模型与独立确认|无确认 EOL；当前不够真实 RUL|
|已有正面证据|PAA 条件调整方向、B2 优于 CHM、原始保持率可因果获得、Persistence 强|极化指标物理定义/条件范围清楚，但 sparse 与 recoveries 并存|目前只有定义/可行性边界；没有已验证寿命成绩|
|是否重复失败|不应再试全序列平滑作在线标签、神经均值或 PSDM rate 调参；新草案只检验固定 Persistence 的 uncertainty|重复以同六点扩大网络会重复 CHM/health pilots；必须先补独立参考|重新硬单调/设 2300 EOL 会重复无标签的伪 RUL|
|新增工作量|先独立审核来源资格；若批准，一个小 CPU variance experiment；不需 raw/NN|新增独立可比观测、重复测量/多轨迹，再另行预注册|较大：长期有效模型、情景资料、可执行 EOL/多寿命结局|
|最大风险|共享年龄与工况、测量 proxy、参考 availability、重复 Development 选择|少锚点/同源验证、BoL 配置、条件混杂|外推错设、阈值任意、未来工况未知、删失误解释|
|创新是否站得住|可能是可证伪的 availability 与 calibration 问题；不能仅“B2+某网络”叫创新|需新独立证据或有可验证映射的真实问题；当前无新架构创新资格|仅明确条件的研究设计；当前无真实 RUL 创新成果|
|硕士下一步适合度|最高：问题小、基线强、正负结果都可交接；不保证新性能贡献|备选：取决于能否得到新增参考资料/实验条件|暂缓：证据链和时间成本超出当前最小问题|
|GPU 是否值得投入|**NO**：拟议问题不需要 GPU|**NO**：先补数据，不扩大容量|**NO**：先解决 EOL/长期预测资格|

证据：[S07][S08][S09][S10][S11][S13][S15][S31][S34][S35]。

## 2. 推荐排序

### 优先 A：保留 P1S-B2 的回顾价值，在线主目标用 raw causal P1S retention

选择理由不是 B2 数值最好，而是这条路已有明确稳定测量定义、可获得时间勘误、强简单对照和可复用的物理周期身份。P1S-B2 可连接稀疏参考工况性能，但全六锚点校准不用于新在线 target。[S10][S12][S13]

**不重复旧实验的最小问题：** 固定点预测为 R_origin，起点可知的工况偏离能否解释 H6 误差尺度，从而改善概率校准？PSDM 试过潜在均值/rate dynamics，不是同一固定均值的条件方差问题。CNP1S 试过电压/校准归一化，也不能作为该概率假设的已完成证据。[S14][S28]

此题尚无正面实验结果；不能预设工况一定提高预测区间。若来源时间资格无法证明，或不优于 Gaussian Persistence，就停止该候选。详见草案，不执行。

### 备选 B：独立参考工况测量优先于模型容量

保留 reference-condition performance SOH proxy 定义；增加当时可知、可比较、重复并有独立验证资格的观测。现有六校准点不足支撑更复杂模型的稳定泛化。[S04][S09][S34]

EIS 是可考虑的额外测量，不是现成提升模块。先核对 8/10-cell 及具体物理堆/条件，必要时需要同堆同条件的新测量。[S02] 本轮只列数据条件，不为 B 制定第二套训练草案。

### 暂缓 C：不进入真实 RUL，也不靠任意阈值“转正”

缺失真实 EOL、长期预测和情景资料；90%/95% 未观察到。可以在今后独立决策中设计假设阈值情景，但必须保留 conditional/extrapolative 口径，不称 validated RUL。[S03][S35]

本轮不为 C 选择阈值、模型或 failure probability；不将它列作下一轮可训练路线。

## 3. 与动态电压论文的关系

F2/P2/P3 可按已有证据单独推进动态预测成果，不需要一个未验证健康模块“补齐题目”。P3 G2 Best 比自身 P2 好，不代表胜过同预算 G0/G1，更不代表健康信息增益。现有 F2 不修改、不重新门控、不追加 Physics-F2。[S23][S25][S26]

健康路线可以形成“有条件的健康代理与因果预测边界”研究。论文价值来自可检验的问题、同信息强基线与诚实局限，不来自强行声称 SOH/RUL 已完成。

## 4. 当前准入结论

- 资料/路线收口：完成。
- 下一阶段小型 CPU 假设检验：**仅有草案；需研究负责人审批并冻结 availability 与实现，当前未授权执行。**
- 新 GPU/神经网络训练、fine-tuning/multitask、backbone search：**NO-GO**。
- 真实 RUL：**NO-GO**。
- 若不能证明 priority A 的现有 Development artifact 字段时间资格，则不执行拟合；保留当前证据整理，不自动切换 B/C，也不回到神经网络。

## 引用来源

[S02]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/polarization_configuration_compatibility.md>
[S03]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/FINAL_RUL90_FEASIBILITY_REPORT.md>
[S04]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL_HealthAnchor_FourModel_Pilot/reports/FINAL_RESEARCH_DECISION.md>
[S07]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/reports/FINAL_P1S_PREANCHOR_AUDIT.md>
[S08]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/reports/FINAL_PAA_VALIDITY_QUALIFICATION.md>
[S09]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model/reports/FINAL_CHM_D_DEVELOPMENT_REPORT.md>
[S10]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/reports/FINAL_AFFINE_CALIBRATION_CHARACTERIZATION.md>
[S11]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PHI_P1S_B2_Health_Index_Consolidation/reports/P1S_B2_HEALTH_INDICATOR_CONSOLIDATION_REPORT.md>
[S12]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTA_P1S_B2_Causal_Target_Audit/reports/P1S_B2_CAUSAL_TARGET_AUDIT_REPORT.md>
[S13]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTAE_P1S_Reference_Availability_Erratum/reports/CTA_REFERENCE_AVAILABILITY_ERRATUM.md>
[S14]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/reports/PSDM_FEASIBILITY_REPORT.md>
[S15]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/outputs/PSDM_FEASIBILITY_SUMMARY.json>
[S23]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_FINAL_RESEARCH_RECOMMENDATION.md>
[S25]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_P2_Final_Gate_Ablation_Training/reports/P2_FINAL_RESEARCH_DECISION.md>
[S26]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_P3_Full450_Epoch_Sensitivity/reports/P3_SUPERVISOR_ONE_PAGE_SUMMARY.md>
[S28]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CNP1S_Condition_Normalized_P1S_Development_Study/reports/CONDITION_NORMALIZED_P1S_DEVELOPMENT_STUDY.md>
[S31]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/reports/H3_H6_ORIGIN_CONDITIONED_FORECASTING_REPORT.md>
[S34]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md>
[S35]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md>

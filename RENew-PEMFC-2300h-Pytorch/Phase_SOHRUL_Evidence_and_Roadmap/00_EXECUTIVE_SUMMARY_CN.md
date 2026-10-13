# RealHyFC SOH/RUL 研究路线收口：执行摘要

审计基准：main `053760cb317d077059ac02acba6ec5abb78e2236`。本阶段只有证据整理和方案设计，没有新实验、模型推理、参数拟合或新增 Test 目标访问。

## 一句话结论

**先把“可因果获得的健康代理及其预测不确定性”研究清楚，不再强行把动态电压模型改造成真实 SOH/RUL 模型。** 优先路线 A；需要新增独立参考观测的路线 B 为备选；路线 C 的寿命情景外推暂缓。三条路线目前都不建议投入新的 GPU 训练。

## 已经做到什么

1. 数据基础和动态电压预测研究已经形成可复核成果。F2 的任务是过去 24 h → 未来 12 h 动态 ΔV，不是参考工况 SOH。P2/P3 未证明动态门控具有稳定独立收益，不能继续靠增加轮数或换门控追成绩。[S01][S23][S25][S26]
2. 极化测试提供 7 个 Development 标称参考点（含 BoL），主参考 j*=1.0 A/cm²、辅助 1.6。它们能够描述参考工况性能保持率，但不是逐半小时真实 SOH 标签，也不是 7 条独立寿命。[S01][S34]
3. P1S 的冻结稳定工况测量及 B2 校准，是目前最有证据的健康代理桥梁。PHI 保存了 141 个完整周期、86.11–1712.79 h 的密集观测；B2 在六锚点 LOAO 中优于 CHM，但只有六个锚点，不能证明跨堆真实 SOH。[S07][S09][S10][S11]
4. **在线目标必须再收紧。** 全六锚点 B2、全序列单调/平滑曲线只能回顾解释；P1S 参考值在 171.82091 h 后才可获得。正式因果观测从 COMPLETE_011 开始，共 131 个 post-reference P1S 保持率事件。[S12][S13][S14]

## 最大困难不是缺一个更新的网络

健康参考少、共享年龄趋势、工况匹配不足、参考值的可获得时间，以及没有真实失效结局，才是当前证据瓶颈。

- 六个 PAA 锚点的中位数来源只有 3/6 位于联合共同支持域；方向一致不等于排除了所有工况混杂。[S08][S40]
- CHM 未通过主要资格；TimesNet scratch、迁移和所谓 same-cycle 研究均未建立稳定健康估计。CSHSE 实际 0/105 个窗口使用目标周期信息，不能把它当成已经真正检验了 current-state estimation。[S09][S16][S17][S18]
- H6 Persistence 为 0.604104 mV-equivalent；TimesNet 为约 0.716930，PSDM 为 0.707559。PSDM 的主要 CRPS 也未胜过 Gaussian Persistence，不能因为区间覆盖率过关就宣称潜在物理退化状态成立。[S15][S31]
- 官方可执行 EOL 和 failure timestamp 未建立；历史审计未观察到 90% 或 95% 参考保持率阈值。2300 h 是记录终点，不能用 2300−t 制造 RUL。[S03][S35][S47]

## 最值得继续什么

**路线 A（优先，但须纠正在线口径）：** 保留 P1S-B2 作为回顾性、退化相关 HI；在线研究使用参考可获得后的原始 P1S 保持率。下一阶段草案只问一个简单问题：保持 Persistence 点预测不变，起点已知工况偏离能否改善其 H6 预测区间校准？这不是重新训练 PSDM，也不是再造神经网络；目前只是待审核假设，尚无成功证据。

**路线 B（备选）：** 增加独立、可比较的参考极化观测和重复测量，再研究 reference-condition SOH proxy 的估计。EIS 可作为额外证据候选，但必须先解决 8-cell/10-cell、物理堆身份、工况及时间可比性，不能凭文件存在就承诺提高精度。[S02][S34]

**路线 C（暂缓）：** 假设阈值下的寿命情景只能是条件外推，不能称为真实失效验证的 RUL。没有可信长期预测和失效准则前，不进入 RUL 拟合。[S35]

## 科学边界与下一步

历史后期目标已经在部分阶段暴露；本阶段是**零新增 Test 目标访问**，不是宣称“整个项目 Test 从未打开”。ELTE 的两点好成绩仍属探索性，不能升级为独立确认性验证。[S19][S21][S37]

建议负责人先审核 [下一阶段最小草案](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_SOHRUL_Evidence_and_Roadmap/06_NEXT_PHASE_EXPERIMENT_PROPOSAL_DRAFT.md>)，尤其是起点工况字段的实际可获得性和固定评价单位。未经批准不得执行任何拟合。若字段来源或时间资格不能证明，优先路线也停止在资料整理；不补 raw、不放宽支持、不增加模型。

完整证据见 [研究沿革](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_SOHRUL_Evidence_and_Roadmap/01_EXISTING_HEALTH_RESEARCH_EVIDENCE.md>)、[健康定义](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_SOHRUL_Evidence_and_Roadmap/03_SOH_AND_HI_FEASIBILITY.md>)、[RUL 边界](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_SOHRUL_Evidence_and_Roadmap/04_RUL_AND_EOL_FEASIBILITY.md>) 与 [执行来源审计](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_SOHRUL_Evidence_and_Roadmap/08_EXECUTION_AND_SOURCE_AUDIT.md>)。

## 引用来源

[S01]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/PHASE_N1_RESULTS.md>
[S02]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/polarization_configuration_compatibility.md>
[S03]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/FINAL_RUL90_FEASIBILITY_REPORT.md>
[S07]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/reports/FINAL_P1S_PREANCHOR_AUDIT.md>
[S08]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/reports/FINAL_PAA_VALIDITY_QUALIFICATION.md>
[S09]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model/reports/FINAL_CHM_D_DEVELOPMENT_REPORT.md>
[S10]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/reports/FINAL_AFFINE_CALIBRATION_CHARACTERIZATION.md>
[S11]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PHI_P1S_B2_Health_Index_Consolidation/reports/P1S_B2_HEALTH_INDICATOR_CONSOLIDATION_REPORT.md>
[S12]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTA_P1S_B2_Causal_Target_Audit/reports/P1S_B2_CAUSAL_TARGET_AUDIT_REPORT.md>
[S13]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTAE_P1S_Reference_Availability_Erratum/reports/CTA_REFERENCE_AVAILABILITY_ERRATUM.md>
[S14]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/reports/PSDM_FEASIBILITY_REPORT.md>
[S15]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/outputs/PSDM_FEASIBILITY_SUMMARY.json>
[S16]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TSHE_TimesNet_P1S_B2_Health_State_Estimation/reports/TIMESNET_HEALTH_STATE_ESTIMATION_REPORT.md>
[S17]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CSTL_Chronology_Safe_Pretrained_TimesNet_Health_Transfer/reports/PRETRAINED_TIMESNET_HEALTH_TRANSFER_REPORT.md>
[S18]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CSHSE_Causal_SameCycle_Health_State_Estimation/reports/CAUSAL_SAME_CYCLE_HEALTH_STATE_ESTIMATION_REPORT.md>
[S19]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_ELTE_Exploratory_Late_Life_Target_Evaluation/reports/EXPLORATORY_LATE_LIFE_TARGET_EVALUATION.md>
[S21]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/docs/HISTORICAL_TEST_EXPOSURE_DISCLOSURE.md>
[S23]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_FINAL_RESEARCH_RECOMMENDATION.md>
[S25]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_P2_Final_Gate_Ablation_Training/reports/P2_FINAL_RESEARCH_DECISION.md>
[S26]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_P3_Full450_Epoch_Sensitivity/reports/P3_SUPERVISOR_ONE_PAGE_SUMMARY.md>
[S31]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/reports/H3_H6_ORIGIN_CONDITIONED_FORECASTING_REPORT.md>
[S34]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md>
[S35]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md>
[S37]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_L_Test_Eligibility_and_Reproducibility_Freeze/README.md>
[S40]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/docs/PAA_CLAIM_BOUNDARY.md>
[S47]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/04_OFFICIAL_EOL_AUDIT.md>

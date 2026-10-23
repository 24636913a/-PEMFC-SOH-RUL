# 面向导师的简明汇报提纲

## 一、现在的研究不是推倒重来

动态电压预测已经有成果。我们用过去 24 h 预测未来 12 h 的电压变化，但它里面同时包含负载和工况影响，不能直接当作 SOH。最近的门控消融与 450 轮敏感性研究没有证明动态调融合比例有稳定独立收益，所以这条结构创新先收口，不继续加模型。[S23][S25][S26]

## 二、健康路线已有基础，不是没有任何结果

数据中有重复出现的 P1S 稳定工况。按冻结规则测量，可以得到 141 个完整周期的运行电压保持率，再用简单 B2 线性标定与六个极化参考点连接。

B2 在六点留一检验中误差约 0.000477，复杂 CHM 约 0.001517。这说明目前**简单健康代理比复杂模型更站得住**，但不代表已经知道真实 SOH。六个锚点都来自一个堆，其中只有三点的中位数来源位于联合工况支持域。[S07][S08][S09][S10][S11]

## 三、需要向导师解释清楚的两次修正

第一，全六锚点标定和全序列平滑是回顾性整理，不能装作早期在线已经知道。参考电压也要在前十个周期完成后才能确定。因此在线研究应从参考可得后的 131 个原始 P1S 保持率事件开始。[S12][S13][S14]

第二，过去 TimesNet 健康实验未稳定成功，不只是 seed 问题。任务时序、标签范围和工况支持都有影响。CSHSE 想看当前周期信息，但严格排除所有标签 dwell 后，实际 105 个窗口没有一个使用目标周期信息，所以它不能证明“真正当前健康估计已经检验并失败”。[S16][S17][S18]

## 四、为什么暂时不做真实 RUL

没有确认的真实 EOL，历史数据也没到 90%/95% 参考保持率阈值。2300 h 是试验记录结束，不是失效。用 2300−t 当寿命标签会把预定结束时间当成答案。[S03][S35]

可以将来研究某个假设健康阈值下还剩多少时间，但必须写清未来工况、阈值假设和外推限制。那不叫经过真实失效验证的 RUL。本轮不设阈值，也不计算剩余小时。

## 五、建议下一步做一个小而能证伪的问题

优先研究：**现在的工况是否能告诉我们，未来六个物理周期的 P1S 预测应该有多大的不确定性？**

点预测先不变，仍然用很强的 Persistence。H6 它的误差约 0.604 mV-equivalent，TimesNet 约 0.717、PSDM 约 0.708；继续堆网络没有现成依据。我们只提出一个固定、简单的条件误差尺度模型，与普通 Gaussian Persistence 和同容量年龄对照比较。[S15][S31]

研究价值不是“又组合两个模块”，而是回答：工况可知信息有没有超出年龄与简单标尺的概率预测价值。结果如果没有，就正式停止这项候选，不再调几十组设置。

## 六、备选与暂缓

备选：增加独立可比的极化参考/重复测量，再研究 reference-condition SOH。EIS 可能补证据，但先核对 8/10-cell 和物理堆/工况，不凭文件多就承诺效果。[S02][S34]

暂缓：真实 RUL、长远寿命情景、神经多任务、重新搜索 backbone。现有 F2 电压成果可独立成稿，健康研究不必强行接上 F2。

## 七、需要导师作出的决定

1. 是否认可把“真实 SOH/RUL”收紧为“因果健康代理与概率预测边界”的近期目标？
2. 是否批准 [最小下一阶段草案](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_SOHRUL_Evidence_and_Roadmap/06_NEXT_PHASE_EXPERIMENT_PROPOSAL_DRAFT.md>) 的唯一条件方差问题？批准前先核实起点字段 availability。
3. 能否获得独立参考测量或更多寿命轨迹，作为备选路线和将来真实 RUL 的必要资料？

本次只是规划，尚未开始任何新训练/拟合。历史晚期目标已暴露，不能再次称独立 Test；本轮没有新增目标访问。[S19][S21][S37]

## 引用来源

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

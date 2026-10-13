# EOL、删失、条件阈值与真实 RUL 可行性

## 1. 已冻结事实：不要重算或制造失效

RUL90 的历史 full-data exploratory 审计未发现官方 90% reference-voltage EOL、明确 failure time 或“试验结束即失效”的标签。其参考性能最低约 0.96950，90% 和 95% 阈值未观察到；这些数字来自已暴露完整资料的历史结果，本轮没有重读/重算。[S03][S47]

原资料确有 500 mV lower-voltage-limit 文字，不能说从未出现任何限值。但它处于 60/50 kPaG、1.8 A/cm² 的条件特定测量语境，不自动给出 j*=1.0 reference SOH 的正式 EOL，也未证明 durability 的 failure timestamp。原资料目标为超过 2000 operating hours、约 200 cycles 后完整 characterization，支持研究性 EoT，不是自动 run-to-failure。[S47][S35]

结论：**EOL_NOT_ESTABLISHED；真实完整 RUL 监督不成立。** 禁止将 2300 h 假定为 EOL，禁止 RUL=2300−t。[S35]

## 2. 四类结果必须分开

|概念|所需记录|能评价什么|本项目当前资格|
|---|---|---|---|
|已观测真实失效 RUL|预先定义且可追溯的失效准则、实际/区间 failure time T_f|T_f−t 的预测误差，前提是独立评价|没有|
|右删失寿命|定义 failure criterion；证明到观察截止 T_c 未发生；停止原因及删失机制|T_f>T_c 的有限信息，不是精确 RUL target|只有待按准则审定的潜在删失属性|
|假设阈值剩余时间|清楚标为假设的指标阈值和 crossing 规则|给定假设下的 crossing forecast|可以规划，但不是本轮执行，也不代表真实失效|
|模型外推寿命情景|明确未来 load/环境/停机/恢复情景、有效模型与不确定性|情景条件下的外推范围/敏感性|当前长期预测资格未建立，暂缓|

单堆、10 个 cell 通道、数千重叠窗口及七个极化点不能变成多条独立寿命。没有失效准则就不能一边“右删失”一边为每行填写精确剩余寿命。[S35]

## 3. 假设阈值可以研究什么

后续若另行批准，可以定义一个**不预设数值**的假设 θ：

    tau_theta(t | scenario)
      = inf{u >= 0 : predicted_health(t+u | scenario) <= theta}

它是健康代理阈值 crossing 的条件剩余时间，不是已验证真实 RUL。还需冻结持久跨越/瞬时跨越、恢复后是否重置、时间单位、可外推范围与无跨越时的表达；本阶段不选择 θ、不计算 hours/cycles、不输出 failure probability。

P1S-B2 线性校准、平滑单调与阈值相交不能证明 θ 对应真实失效。若 extrapolation 改变健康定义或未来工况，crossing 时间也改变；小样本下窄数值范围更不能保证准确外推到未观测退化水平。[S11][S12][S35]

## 4. 概率区间不等于寿命可信度

PSDM 的 Gaussian EKF intervals 属于已假定模型下的预测分布，含 observation noise；filtered-state interval 与 future predictive interval 不同。该模型 H6 主 CRPS/MAE 未过门槛，不能把覆盖率达标包装成寿命推断可用。[S14][S15]

长期 crossing uncertainty 还包括标定、未来运行工况、状态方程错设、恢复效应、外推和阈值定义。任何短期 interval coverage 都不能验证长远 failure-time coverage；单一 trajectory 内 coverage 也不是 across-stack 可靠性。不得将 PHI 描述性 ±1.96 带当成概率寿命区间。[S11][S35]

## 5. 未来工况未知必须写情景

至少明确负载/duty schedule、电流/温湿压力、停机频率、恢复/维护、reference measurement cadence、raw cycle 与 operating-hour/wall-clock 的转换。不知道这些，不能默认重复历史平均周期而称实际寿命。

H3/H6 是 3/6 raw physical cycles，不是 3/6 观测事件、也不是固定小时。H1E 已指出实际未来 target lead 不得当作起点已知输入；事后周期均值只支持描述，不提供真实未来运行安排。[S30][S31]

未知情景不能靠未来真值补入模型。情景预测应报条件与适用域，不同时宣称 unconditional failure forecast。

## 6. 怎样才能正式声称 RUL 准确

需要的不是更深网络，而是至少：

- 可执行且参考条件一致的 failure/EOL 定义，以及 failure mode/功能损失对应证据。
- 多条独立、可追踪的 run-to-failure 或已明确准则与停止原因的 censored trajectories；失效/删失时间和环境历史可靠。
- 当时可获得的健康参考，避免 future calibration、full-series smoothing 和 actual future lead。
- 独立的时间/个体评价和足够的失效事件；仅增加同堆窗口不能满足。
- 区分起点到未来的真正预测，与回顾性拟合/阈值外推；评价点误差、概率校准及未跨越/删失处理。
- 跨工况支持与单位/物理堆一致性；事先冻结方案，不以已经曝光的 2000/2300 调整阈值后再称独立验证。[S34][S35][S37]

新增观察类型和数量应由独立统计设计决定，本轮不编造所需精确样本量或声称几个新点足够。

## 7. 当前结论

真实 RUL：**NO-GO**。假设阈值情景：仅允许未来设计，不是当前模型执行资格。优先研究 P1S 因果代理的可预测性与误差区间；先证明有限 horizon 的概率效用，再讨论是否值得外推。

这不否定动态电压论文，也不把没有 failure label 的资料说成“没有任何健康价值”。恰当的贡献是健康代理、预测边界、条件不确定性和删失限制，而不是虚构真实寿命答案。[S23][S31][S35]

## 引用来源

[S03]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/FINAL_RUL90_FEASIBILITY_REPORT.md>
[S11]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PHI_P1S_B2_Health_Index_Consolidation/reports/P1S_B2_HEALTH_INDICATOR_CONSOLIDATION_REPORT.md>
[S12]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTA_P1S_B2_Causal_Target_Audit/reports/P1S_B2_CAUSAL_TARGET_AUDIT_REPORT.md>
[S14]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/reports/PSDM_FEASIBILITY_REPORT.md>
[S15]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/outputs/PSDM_FEASIBILITY_SUMMARY.json>
[S23]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_FINAL_RESEARCH_RECOMMENDATION.md>
[S30]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_H1E_Lead_Availability_And_Information_Set_Erratum/reports/H1_LEAD_AND_INFORMATION_SET_ERRATUM.md>
[S31]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/reports/H3_H6_ORIGIN_CONDITIONED_FORECASTING_REPORT.md>
[S34]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md>
[S35]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md>
[S37]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_L_Test_Eligibility_and_Reproducibility_Freeze/README.md>
[S47]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/04_OFFICIAL_EOL_AUDIT.md>

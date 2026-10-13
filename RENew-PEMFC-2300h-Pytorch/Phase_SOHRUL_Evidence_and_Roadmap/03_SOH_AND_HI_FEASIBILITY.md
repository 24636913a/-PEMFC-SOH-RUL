# SOH、运行健康代理与真实健康状态：资格审查

## 1. 先把三个层次分开

|层次|本项目的量|目前资格|
|---|---|---|
|参考工况性能保持率|SOH_V(t_i)=V(j*,t_i)/V(j*,BoL)|7 个稀疏 Development normal 参考点；条件可比时可信的性能 proxy|
|循环级健康代理 HI|R_P1S=V_P1S/V_ref；HI_P1S=aR+b|有退化关联与 affine 兼容证据；global B2/全序列趋势属回顾性，在线要按 CTAE|
|真实 SOH / 不可逆损伤状态|经独立测量验证、适用范围明确的健康量|尚未建立；不能靠改名、平滑或单调约束获得|

SOH 一词可用于**明确定义的 reference-condition SOH proxy**，不能省略“proxy”并暗示已独立验证真实 SOH。以下结论是资料审查，不是新实验。[S34][S11][S12]

## 2. 极化参考：有用，但稀疏且有条件

主 j*=1.0 A/cm²，辅助 j*=1.6；Development normal 点为 BoL、300、600、850、1150、1500、1750。辅助 j 来自相同曲线，不增加独立样本。N1 报告主参考年龄 rho≈−0.964，non-increasing fraction≈0.833，1500 recovery shift≈3.916 mV：长期下降与恢复波动并存，不能强迫每一点单调下降。[S01][S34]

冻结公式：

    SOH_V(t_i) = V(j*,t_i) / V(j*,BoL)

1−[V_ref−V]/V_ref 只是相同公式的另一写法，不是独立标签。BoL denominator 不确定性由全部比值共享，单曲线里的大量 raw samples 是局部相关观测，不是重复电堆寿命。[S34]

可比性结论 **CONDITIONAL**：nominal 10-cell per-cell voltage/current density 支持比较；入口温度约 75°C、冷却约 70°C、RH 50/35%、压力 160/150 kPaG、stoichiometry 1.5/1.7，以及 dwell/ramp/conditioning 必须对应具体参考。独立 8-cell BoL EIS/sensitivity 不可无条件拼接。运行恢复、稳态不足和 protocol 差异可能改变电压，不等于不可逆衰退。[S02][S34]

可以用于有限锚点的监督/校准假设检验，但每次拟合只能使用当时已发生的训练锚点；held-out target 不可用于拟合。七点不能提供逐 0.5 h 的 ground truth。插值只是一项额外模型假设，不得制造连续 SOH 标签。[S34]

历史报告 BoL voltage 约 0.698953 V；RUL90 与 K 的保留值末位略有差异（0.698953307 与 0.6989533305168152），属于冻结来源口径差异。本轮不重算、平均或覆盖；使用者必须指定采用哪份 authority。[S03][S34]

## 3. P1S 的真实物理定义

冻结 token 是 operation_point=221、set current=221.200 A。选择依据为冻结 RCR 模板中 eligible P1 token 的出现频率（该 token 为 93，其他主要候选 156.8 A 为 48、313.6 A 为 25），不是寻找最单调/最相关的电压区间。这些是模板内 token 频次，不是全部 P1S 测量或独立寿命数。[S39]

P1S 定义不是任意 30 min 平均电压。连续 state dwell 至少 120 s；primary 为 dwell 的 middle_50，last_25 作已冻结敏感性；取 active 10-cell mean voltage 的中位数。一个 complete-cycle 的 P1S 再对合法 P1S dwells 取中位数，所以实际标签可能由一个周期内多个分离 dwell 组成，而不是一个瞬时点。[S39][S18]

七个条件变量为 current density、anode inlet temperature/pressure/RH、hydrogen flow、anode/cathode stoichiometry。稳定电流不意味着其他条件都完全固定。PAA 的匹配分析提供条件调整后方向稳健性，但无距离 caliper，不等于充分消除全部混杂。[S39][S40]

146 个 raw complete cycles 最终提供 141 个合法 P1S measurements，覆盖 86.11–1712.79 h；现有冻结 complete 表没有 post-1819 cycle-level 标签。不能因为数据集叫 2300 h，就把这条健康轨迹画成完整 2300 h。[S05][S11]

## 4. B2 从哪里来、强到什么程度

冻结 Raw P1S B2 是非负 affine constrained least squares：

    Y = a R_P1S + b,   a >= 0
    a = 1.1988114938324186
    b = -0.2090362463118213
    R_P1S = V_P1S / 0.71383023635

Y 是稀疏参考极化性能 proxy，不是独立损伤真值。P1S 的 0.71383023635 V 是 earliest-ten complete-cycle 参照，与极化 BoL 的约 0.698953 V **不可互换**。[S10][S11][S39]

校准锚点为 300/600/850/1150/1500/1750。六个 LOAO held-out targets 上 B2 MAE≈0.00047687，优于 CHM≈0.00151714；严格 chronological prefix 的三个目标 MAE≈0.00076469。这支持“简单 affine 有限校准”，不支持对连续时间或新堆的准确率承诺。[S09][S10]

PAV 核对只有 300/850/1750 的 median-source 在 joint support。600/1150/1500 虽符合原 2/3 interval support gate，其中位数来源不在域。保留原结果，不能事后删三点重报更好成绩。[S08][S40]

共同年龄趋势仍是替代解释：P1S 与极化都随老化下降，六个点的小误差和高 Spearman 不证明 P1S 捕获了独立于年龄/工况的不可逆损伤。LOAO 是有限的插值式关联检验，且锚点/工况定义来自同一堆的探索历史。[S10][S40]

## 5. 在线资格必须以 CTA/CTAE 为准

全六锚点 B2 用到未来健康参考，不能追溯赋给早期事件并声称当时可知。CSTL 改为 fold1 300/600/850、fold2/3 加 1150，不用 1500/1750 定义早期 Validation；这修正一类泄漏，但仍是 proxy 标定，不是真 SOH。[S12][S17]

P1S earliest-ten reference 仅在 **171.82091 h** 后可获得。前十条为 REFERENCE_WARMUP_ONLY；第一个 causal target 是 COMPLETE_011（事件年龄 181.34366 h），之后共 131 观测。首个合格过去窗口 slope availability 为约 200.39166 h；计算因果性不是预测能力。[S13][S14]

因此：

- 回顾性：可以保留 141 点 raw P1S-B2 HI，并明确 global calibration 和参照的回看性质。
- 在线：仅使用 post-reference **raw causal P1S retention**；校准如使用，必须另有训练期已可获得的 B2 authority。
- 不得将未来锚点、full-sequence smoothing、未来实际 target lead 或 posthoc support 标签作为在线输入。[S12][S13][S30]

## 6. 原始 HI、平滑趋势与单调趋势并非同一证据

PHI 保存 raw，并生成 SG、decreasing isotonic、monotonic PCHIP。全序列算法让过去值依赖之后的数据，因此 CTA 将它们标为 RETROSPECTIVE_TREND_ONLY；只能用于回顾解释，不作为在线训练 target/feature。[S11][S12]

raw HI 有 35 次局部回升；isotonic 的 0 次回升由约束直接保证，不是“不可逆健康状态已被验证”。恢复/运行可逆波动、噪声或工况变化不能被算法强制单调掩盖。未来研究如需 causal trend，要独立注册 prefix-only 规则，不借用已曝光曲线选择平滑强度。[S12]

## 7. 不确定性到底是什么

|来源|现有解释|不能解释为|
|---|---|---|
|PHI sigma_measurement|middle_50 与 last_25 两个相关窗口估计的散布；窗口敏感性 proxy|独立重复测量方差、已辨认传感器噪声|
|PHI sigma_calibration|六个 leave-one-anchor-out B2 函数之间的差异|完整标定 posterior 或模型外推误差|
|PHI 合成 ±1.96 区间|描述性 quadrature 展示|实测 95% prediction coverage|
|极化局部散布|同一次曲线/局部窗口波动|跨日独立复测可靠性|
|PSDM predictive interval|latent state+observation noise 的 Gaussian EKF approximation|exact posterior、failure-time 可信区间|

来源：[S11][S34][S14]。不确定性至少应分测量、标定、模型、未来工况、外推/定义风险；不能只画一条很窄的“置信带”代表全部。

## 8. 真 SOH 还缺什么；EIS 只能作为待核验旁证

目前缺少：独立且可比较的参考条件重复测量；可逆恢复与持久损伤的区分；超出共同年龄趋势的独立退化证据；跨堆/新轨迹验证；以及健康指标与明确功能/失效准则的联系。[S34][S35][S40]

EIS 可能提供不同于电压的旁证，但“文件存在”不能证明有可识别的退化参数，更不能承诺精度提升。已有审计指出独立 BoL EIS/full sensitivity 可能为 8-cell，durability/nominal polarization 为 10-cell；export Number cells 与论文身份冲突仍需按文件确认。必须先核对物理堆/单体、有效面积、温压湿/current、频率协议、扰动和恢复状态、时序、参数辨识与重复性。不得直接将不同配置阻抗参数视作同一堆 SOH。[S02][S34][S46]

**本轮未读取任何新 EIS 值；逐文件配置身份未核实，不能给出“哪些文件已完全可比”的肯定清单。**

## 9. F2 与健康路线：不强行拼接

F2 提供合法历史到动态 ΔV 的建模成果、chronological 数据边界、train-only scaler、窗口身份与公平对照方法。它不能直接提供参考工况退化状态：未来负载、热湿压力与恢复都影响动态电压，ΔV 没有标准化到固定 j*。[S23][S34]

P1S-B2 提供的是较接近重复运行工况的代理桥梁；该桥梁没有证明冻结 F2 latent/output 能稳定估计健康。TSHE 使用 global B2 与 pre-cycle 输入；CSTL 修复标定后 Fold3 仍不稳；CSHSE 在实际 dwell 定义下 0/105 same-cycle history。问题同时涉及标签范围、时序可用性、工况/时间分布和表示，而不应自动归因 seed。[S16][S17][S18]

如果将来结合，需额外证明：模型输出到固定参考条件的可辨识映射、同键同信息对照、目标 measurement 严格隔离、校准时间可得、且超越 Age/LastVoltage/WindowVoltage/Persistence。当前缺这些证据，不通过模块拼接宣布创新。

**两条路线可以独立成稿：** 一条是限定 Development 的动态电压预测与结构消融；另一条是 P1S 健康代理的因果可获得性、简单预测边界与不确定性。后者尚待按草案授权检验，不保证形成新性能贡献。[S23][S25][S26][S31]

## 引用来源

[S01]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/PHASE_N1_RESULTS.md>
[S02]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/polarization_configuration_compatibility.md>
[S03]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/FINAL_RUL90_FEASIBILITY_REPORT.md>
[S05]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/reports/FINAL_RAW_CYCLE_AUDIT.md>
[S08]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/reports/FINAL_PAA_VALIDITY_QUALIFICATION.md>
[S09]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model/reports/FINAL_CHM_D_DEVELOPMENT_REPORT.md>
[S10]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/reports/FINAL_AFFINE_CALIBRATION_CHARACTERIZATION.md>
[S11]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PHI_P1S_B2_Health_Index_Consolidation/reports/P1S_B2_HEALTH_INDICATOR_CONSOLIDATION_REPORT.md>
[S12]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTA_P1S_B2_Causal_Target_Audit/reports/P1S_B2_CAUSAL_TARGET_AUDIT_REPORT.md>
[S13]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTAE_P1S_Reference_Availability_Erratum/reports/CTA_REFERENCE_AVAILABILITY_ERRATUM.md>
[S14]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/reports/PSDM_FEASIBILITY_REPORT.md>
[S16]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TSHE_TimesNet_P1S_B2_Health_State_Estimation/reports/TIMESNET_HEALTH_STATE_ESTIMATION_REPORT.md>
[S17]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CSTL_Chronology_Safe_Pretrained_TimesNet_Health_Transfer/reports/PRETRAINED_TIMESNET_HEALTH_TRANSFER_REPORT.md>
[S18]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CSHSE_Causal_SameCycle_Health_State_Estimation/reports/CAUSAL_SAME_CYCLE_HEALTH_STATE_ESTIMATION_REPORT.md>
[S23]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_FINAL_RESEARCH_RECOMMENDATION.md>
[S25]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_P2_Final_Gate_Ablation_Training/reports/P2_FINAL_RESEARCH_DECISION.md>
[S26]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_P3_Full450_Epoch_Sensitivity/reports/P3_SUPERVISOR_ONE_PAGE_SUMMARY.md>
[S30]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_H1E_Lead_Availability_And_Information_Set_Erratum/reports/H1_LEAD_AND_INFORMATION_SET_ERRATUM.md>
[S31]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/reports/H3_H6_ORIGIN_CONDITIONED_FORECASTING_REPORT.md>
[S34]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md>
[S35]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md>
[S39]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/configs/PREREGISTERED_PAA_PROTOCOL.yaml>
[S40]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/docs/PAA_CLAIM_BOUNDARY.md>
[S46]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/01_DATA_INVENTORY.md>

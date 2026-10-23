# 数据资产、合法访问范围与历史 Test 暴露

本阶段仅核对既有报告/小型配置及两份已证明 Development-only CSV 的表头，不重新扫描原始数据目录，也没有打开混合数据表、原始极化/EIS 或模型 checkpoint。

## 1. 已有数据资产，不等于本轮读取

以下数量来自 RUL90 的**历史 inventory**，不是本轮文件枚举、校验或数值重读。[S46]

|资产|历史数量/规模|可支持什么；限制|
|---|---|---|
|Loadcycle durability CSV|17 文件，历史 inventory 约 5527 MB；N1 约 8,604,654 行、约 1 s|原始运行动态。仅 OpHrs2 作为主老化坐标，DateTime 用连续性/gap；本轮禁止 raw 内容访问|
|标称/恢复极化曲线|11 文件|参考工况电压；正常/after-Air-Air 要分开；不能把所有曲线都视作独立寿命|
|EIS/electrochemical spectrum|68 文件|候选阻抗/电化学旁证；尚未完成本轮文件级 stack identity 与条件核对|
|Spatial EIS|419 文件|空间电化学表征；通道/位置并非独立堆寿命，不能直接拼成标签|
|Partial sensitivity tests|8 文件|工况影响，不自动是健康参考；8/10-cell 可比性需分别核实|
|Paper/protocol、text/metadata|1 PDF、1 文本类别|本轮通过既有审计摘录复核，不打开原 PDF 或 Test 来源|

原始约 2300 h 数据覆盖 OpHrs2 72.788690–2256.8 h。U6 非物理候选；active 10-cell 为 U1–U5、U7–U11。不存在因本次路线规划而改 mask、重建聚合或重新识别 raw cycle 的授权。[S01]

## 2. 冻结时间边界与不同样本单位

N1 原始单次 chronological 切分：[S01]

|角色|OpHrs2 范围（h）|
|---|---|
|Train|[72.788690, 1383.195476)|
|Validation|[1383.195476, 1819.997738)|
|Test|[1819.997738, 2256.8]|

后续 expanding folds：[S17][S31]

|Fold|Train（h）|Validation（h）|
|---|---|---|
|1|[72.788690,946.393214)|[946.393214,1164.794345)|
|2|[72.788690,1164.794345)|[1164.794345,1383.195476)|
|3|[72.788690,1383.195476)|[1383.195476,1819.997738)|

nominal checkpoint 不是实际 OpHrs2，也不是独立可获得时间。PAV 用原内部 DateTime 核实极化开始与测量先后；后续 event role 必须按实际 measurement interval，不能只按文件名“1750 h”。[S08][S39]

|计数|准确含义|不可替换为|
|---|---|---|
|146|RCR 合法 complete raw physical cycles|146 个可观测 P1S 标签或独立寿命|
|141|PHI 合法 complete P1S measurements，86.11–1712.79 h|完整 2300 h 在线健康序列|
|前 10 / 后 131|参考暖启动 / 参考可得之后 causal target observations|141 个全程在线目标|
|105|TSHE/CSTL/CSHSE 满足 48-bin 历史约束的 health events|131 个可用窗口或晚期 Test labels|
|7 / 6|含 BoL 的 normal Development 极化锚点 / post-BoL B2 校准锚点|7 或 6 条独立电堆寿命|
|H6 18/9/18|CMRF 每折合法 validation origin-target pairs|45 个独立失效时间|
|0|冻结 PAA complete 表提供的 post-1819 cycle-level 健康事件|原始试验没有晚期数据|

来源：[S05][S11][S13][S16][S18][S31]。这些是不同筛选层级，不能因数字不同宣布数据丢失。

N1 原分割表采用 boundary exclusion；后期 Development prefix 采用 nrows=3007 后 age 过滤得到 3005 行。N1 Train+Validation 计数与 3005 不同，不在本轮重新计算或合并。将来若某个任务授权读取 N1，必须遵守该任务的逻辑 prefix 契约，不执行完整文件 SHA 或将 metadata 当完整性替代。本次完全未打开 N1 CSV。[S01][S18][S31]

## 3. 三种来源访问分类

|分类|例子|本轮实际方式|完整数据/哈希许可|
|---|---|---|---|
|历史报告、冻结小配置|N1/RUL90/PAA/PHI/CTA/PSDM/K/P3 等|读取 Markdown/YAML；两份 Development-only summary JSON|不需要新 hash；不得沿报告路径自动打开其 raw/预测来源|
|已证明 Development-only 表|PAA complete measurements、CTAE reference audit|只读取首行 header，核对字段，不读取数据行|本轮未计算 hash、未加载全表|
|混合/可能混合数据、Test 源|N1 causal CSV、极化汇总、PHI/late-life CSV、raw Loadcycle、EIS、after-Air-Air|不读取文件本体；信息来自已授权历史报告|禁止 full-file hash、批量加载或探索 suffix|
|历史探索性晚期报告|ELTE、AHC-TV exposure/postmortem；PHI 的历史描述|读取已写成报告的结果与曝光事实|不打开 target 原始文件或重新生成预测|

两个 CSV header 来源为 [S50][S51]；PAA 的 Development 截止资格由 [S36][S39] 证明，CTAE 的范围由 [S13] 证明。字段存在仅支持方案资料可行性，**不证明未来条件协变量逐事件的时间资格与完整键映射已经通过**。

## 4. 历史 Test 访问链：不得抹除

|阶段/来源|记录的历史事实|证据效力|
|---|---|---|
|N1|计划写 Test 只作结构/数量；后续 disclosure 确认实际已解析并保存晚期极化数值，未用于 j*/模型选择|意图与实际分开；不能称整个项目从未看过晚期目标。[S43][S21]|
|RUL90|FULL_DATA_EXPLORATORY_DESCRIPTIVE_ANALYSIS；full-data target access 为 true；无模型训练/Test 预测评价|阈值可行性历史结果属探索性，不变成独立 Test。[S03]|
|RCR|事后 firewall qualification：post-boundary rows accessed 1,896,614，used 0；Test polarization/EIS 未读|周期结构结论保留，full protocol compliance=false；不写“raw 从未访问”。[S48]|
|RHA/RHA_P1|boundary-crossing chunk 中 accessed 31,316、used 0；Test polarization/EIS 未读|访问偏差与实际分析使用分开；不推断 Test-label leakage。[S06][S49]|
|TSHE|报告保存过完整 causal CSV provenance hash|记录该历史入口；未独立重跑验证实际读取轨迹，不能据此伪造 Test label 使用证据。[S16]|
|CSHSE|旧运行 full causal SHA 越过 Development 边界，正式 INVALID_PROTOCOL_BREACH；无 Test-label access 证据；clean rerun 重做|不能引用作废成绩；修复不让旧运行“从未出错”。[S18]|
|PSDM|早期 mixed N1 访问事件，无模型结果/Git 变化；后续正式 Development-only 使用 Stage-A snapshot|保留事故；正式结果与历史访问分别报告。[S14]|
|AHC-TV|正式 one-shot 在 target-blind Stage P/support 失败；target 尚未配对解盲|历史 READY README 不代表成功；禁止重启。[S20][S27]|
|ELTE|另行授权的探索性晚期评价实际打开 2000/2300 targets，先冻结预测；支持 0/3 与 2/3|探索性、同一堆、两点且支持不对称，不是 pristine independent Test。[S19]|
|PHI|引用已公开 ELTE 输出，不重新读晚期 raw target|描述性一致性不是新的独立验证。[S11]|
|L/F2|任务级 Test 历史资格审查给出 CONFIRMATORY_TEST_NOT_ELIGIBLE|不等于已证实 Test target 被用于 F2 拟合；但不能声称历史无曝光。[S37]|
|P2/P3|本阶段用的报告记录无新 Test/Assessment；Development 已复用|不制造新确认性 holdout。[S25][S26]|

本轮：**new_test_target_access=false；new_raw/EIS/polarization_access=false；historical_exploratory_report_access=true。** 未读取 ELTE target 数据表，未重算晚期成绩。整项“historical exposure”不宜用一个 test_accessed=false 去掩盖。

## 5. 8-cell / 10-cell 与 EIS 准入

Nominal 极化、durability 主堆为 10-cell；已有配置审计将独立 BoL EIS/full-sensitivity 试验关联到 8-cell 短堆，虽部分 CSV export 的 Number cells 写 10。该冲突不能用 MEA 批次/相同 280 cm² 面积消除。部分晚期配置缺面积元数据，也不能自行补全。[S02][S34]

本轮没有打开 EIS 文件，因而不能给出已认证的每个 EIS 文件→具体物理堆映射。任何 EIS 新阶段至少要先冻结：物理堆/单体身份、BoL 参照归属、有效面积、温压湿/电流、频率与扰动协议、测量时间、恢复状态、同设备校准、重复性及参数可辨识性。全部核对先于阻抗特征或健康标签拟合；Test EIS 不参与此路线规划。文件级未认证项记为缺口，不猜测。

## 6. 将来可消费的资产与当前缺口

可优先复用：Development-only P1S measurements、CTAE causal eligibility、冻结 raw-cycle identity、原 CMRF H6 sample authority、训练期可见的极化锚点配置。**当前只确认它们在历史报告中的来源与范围，没有对全部行做新键审计。**

不能自动消费：global B2 作为在线 target、全序列 trend、后期 ELTE target 表、任何 late raw、全 N1、未经认证 EIS。未来如果逐事件 availability 或精确 identity 不够，fail closed；不得为执行草案自动重读 raw 或扩增标签。

## 引用来源

[S01]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/PHASE_N1_RESULTS.md>
[S02]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/polarization_configuration_compatibility.md>
[S03]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/FINAL_RUL90_FEASIBILITY_REPORT.md>
[S05]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/reports/FINAL_RAW_CYCLE_AUDIT.md>
[S06]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RHA_Raw_Operation_Point_Health_Alignment/reports/FINAL_RAW_HEALTH_ALIGNMENT_AUDIT.md>
[S08]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/reports/FINAL_PAA_VALIDITY_QUALIFICATION.md>
[S11]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PHI_P1S_B2_Health_Index_Consolidation/reports/P1S_B2_HEALTH_INDICATOR_CONSOLIDATION_REPORT.md>
[S13]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTAE_P1S_Reference_Availability_Erratum/reports/CTA_REFERENCE_AVAILABILITY_ERRATUM.md>
[S14]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/reports/PSDM_FEASIBILITY_REPORT.md>
[S16]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TSHE_TimesNet_P1S_B2_Health_State_Estimation/reports/TIMESNET_HEALTH_STATE_ESTIMATION_REPORT.md>
[S17]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CSTL_Chronology_Safe_Pretrained_TimesNet_Health_Transfer/reports/PRETRAINED_TIMESNET_HEALTH_TRANSFER_REPORT.md>
[S18]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CSHSE_Causal_SameCycle_Health_State_Estimation/reports/CAUSAL_SAME_CYCLE_HEALTH_STATE_ESTIMATION_REPORT.md>
[S19]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_ELTE_Exploratory_Late_Life_Target_Evaluation/reports/EXPLORATORY_LATE_LIFE_TARGET_EVALUATION.md>
[S20]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/README.md>
[S21]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/docs/HISTORICAL_TEST_EXPOSURE_DISCLOSURE.md>
[S25]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_P2_Final_Gate_Ablation_Training/reports/P2_FINAL_RESEARCH_DECISION.md>
[S26]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_P3_Full450_Epoch_Sensitivity/reports/P3_SUPERVISOR_ONE_PAGE_SUMMARY.md>
[S27]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_TV_Target_Blind_Alignment_Postmortem/reports/AHC_TV_TARGET_BLIND_ALIGNMENT_FAILURE_POSTMORTEM.md>
[S31]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/reports/H3_H6_ORIGIN_CONDITIONED_FORECASTING_REPORT.md>
[S34]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md>
[S36]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/README.md>
[S37]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_L_Test_Eligibility_and_Reproducibility_Freeze/README.md>
[S39]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/configs/PREREGISTERED_PAA_PROTOCOL.yaml>
[S43]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/PHASE_N1_PLAN.md>
[S46]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/01_DATA_INVENTORY.md>
[S48]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/final/POSTHOC_FIREWALL_QUALIFICATION.yaml>
[S49]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RHA_P1_Posthoc_Single_Point_Identifiability/reports/FINAL_P1_IDENTIFIABILITY_AUDIT.md>
[S50]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv>
[S51]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTAE_P1S_Reference_Availability_Erratum/outputs/P1S_REFERENCE_AVAILABILITY_AUDIT.csv>

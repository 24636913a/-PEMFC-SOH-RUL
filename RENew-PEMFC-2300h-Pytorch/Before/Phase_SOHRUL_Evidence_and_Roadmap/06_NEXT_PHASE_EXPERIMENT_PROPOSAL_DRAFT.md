# 优先路线 A：最小下一阶段实验草案

**DRAFT_ONLY — 未经研究负责人批准，不得执行。**

本轮未创建新阶段 runner、训练/优化代码、实验成绩或模型产物。以下参数与成功标准都是待审核的 PROJECT_ANALYSIS_DESIGN，不是历史协议、通用科学阈值或已证明的收益。Approval 后须再冻结唯一协议与输入清单；不能拿本草案自动开跑。

## 1. 唯一核心研究问题

**保持 H6 Persistence 的预测均值不变，只利用起点已知的运行工况偏离，能否比 train-only Gaussian Persistence 更好地校准 causal P1S retention 的预测不确定性？**

检验的是 forecast error scale，不是 true SOH、不识别不可逆物理 rate、不输出 RUL。该问题受已知 PSDM/CMRF 负面结果启发，必须标为 post-history exploratory，不是“未看过 Development 的确认性实验”。[S14][S31]

## 2. 明确目标与冻结参照

观测：

    R_k = V_P1S(k) / 0.71383023635
    target = R_(raw_cycle_origin+6)
    forecast mean = R_origin

仅 CTAE causal_target_eligible=true，COMPLETE_011 起的 131 post-reference observations；不改 P1S、V_ref、B2 或参考暖启动规则。不使用 global B2/PHI trend 为 target。[S13][S14]

H6 是六个 raw physical cycles；不是六个“下一个可观测 P1S”或固定 6 h。必须复用 CMRF H6 原 validation query 键与完整 target measurement interval，不按误差/out-of-support 重新删选。[S31]

## 3. 输入与真正合法可获得时间

唯一新增候选输入 u_origin：起点 P1S measurement **已完成且当时可知**的七维工况，与该 fold training 条件分布的标准化距离。七变量沿 PAA：current density、anode inlet temperature/pressure/RH、hydrogen flow、anode/cathode stoichiometry。[S39][S50]

可用：R_origin、起点实际已完成的 condition measurement、历史 physical cycle identity。禁止：target 条件、未来实际 target lead、future reference、全 Dev support box 标签、global B2、full trend、Raw F2、新极化/EIS及晚期资料。

**准入缺口：** 本轮只读 header，尚未认证每个 origin 的 condition 统计用到的完整 dwell/availability 和 key 映射。下一阶段必须用已有 Development-only authority 证明值不含 origin 后数据、measurement 已在 origin 前完成、参考可得。证明不了则 DATA_AVAILABILITY_NOT_VERIFIED，停止，不自动重读 raw 或重建 P1S。

## 4. 数据覆盖及评价单位

已有 P1S 141 事件、86.11–1712.79 h；post-reference 131。固定 H6 outer-train pair 数 33/52/61，validation 18/9/18。原冻结样本 authority 共 191 fold-expanded rows；不可将它们称为 191 independent lifetimes。[S11][S31]

初步 authority：
- CTAE Development causal eligibility/retention 表。
- PAA Development complete-cycle conditions 表及既有测量 interval 证据。
- Development raw-cycle identity/completeness artifact。
- CMRF `inputs/H6_FROZEN_SAMPLE_AUTHORITY.csv`。
- 冻结 PAA/CTAE/CMRF 小型协议。

以上是未来许可清单草案，本轮没有加载这些 CSV 数据行。正式输入须先证明 Development-only，再登记路径/哈希；混合物理文件禁止完整读取/hash。缺少小型冻结表则停止，不为增加样本重开 Test/raw。

每个原 query 等权；主聚合为三 fold 等权，pooled 仅描述。固定评价单位，不额外尝试 target/cycle/condition 加权挑更好数值。

## 5. 与旧研究的关系

沿用 Persistence 强基线、原 R/physical horizon/queries；不重做 mean forecasting。PSDM 是 constrained EKF mean/rate，原 H6 MAE 与 CRPS 未获支持；此处不改其状态方程、参数或结果。CNP1S 的归一化无清晰 overall gain；本草案不复制其回归公式/新校准标签。[S14][S28][S31]

H1E 未认证的未来 actual lead 不得再出现。PHI ±1.96 描述带不是本草案的验证带。[S30][S11]

## 6. 强而简单的对照与唯一候选

所有均值固定 R_origin，信息和 evaluation keys 相同：

|模型|预测标准差|作用|
|---|---|---|
|Gaussian Persistence|sigma_H=std(outer-train DeltaR_H,ddof=1)，floor 1e−8|复用 PSDM 中的概率强基线|
|Scale-only Control|sigma_H exp(a)|剔除仅重新缩放 GP 的增益；固定 candidate b=0|
|Age-scale Control|sigma_H exp(a+b v_origin)|同两参数/同拟合预算；v 为起点 age 的 train-only standardized 值|
|Condition-scale Candidate|sigma_H exp(a+b u_origin)|唯一候选，不变 forecast mean|

u 为七变量按 outer-train median 与 IQR 标准化后的 Euclidean distance / sqrt(active variable count)。IQR=0 则仅以同训练分布 SD fallback；仍为常数则剔除并记录；不使用 validation scale/outcome。若无有效变量停止。不同条件单位不可直接混加。

候选与 Age Control 的 a∈[−3,3]、b∈[−1,1]，单固定 start=(0,0)，同 L-BFGS-B、maximum 200 iterations，outer-train Gaussian NLL；Scale-only 同 a bounds。**这些是待批准设计选择**；一旦冻结，validation 不用于改边界、加 start、试另一种距离/variance formula。训练失败需技术记录，不写成科学无收益。

输出 sigma>0；Gaussian variance 单位 retention²。没有物理方程/不可逆机制创新宣称，不将该候选叫 PINN/物理退化模型。缺 SciPy 不安装依赖擅自换优化方法，停止技术准入。

## 7. 可证伪的创新假设

已有正面证据只证明 P1S 代理与强 persistence；没有证明 origin condition distance 与 future error scale 相关。假设为：condition-scale 比 GP 和同容量 age-scale 提供更好的 chronological 概率预测，而非改善点预测或利用年龄拟合 target。

若 b≈0、或仅 scale-only 改善、或 candidate 不胜 age control，则不能声称独立工况不确定性信息。即使通过，也只支持本单堆探索性条件误差建模，不宣称全球首次、不可逆退化已辨识。[S07][S10][S15]

## 8. 泄漏防范

- 每个训练 pair 的 target measurement 全部结束于 outer-train boundary 之前，不能因 origin 在 train 就允许未来 target。
- V_ref 已可获得；条件只截至 origin，未来 reference / target condition / actual lead 不进入模型。
- 全部 transformation/variance 参数仅 training 拟合；禁止 whole-series normalization、future B2、smooth target。
- Query 精确键匹配、重复/缺失/target mismatch fail closed，不求 intersection 静默减样本。
- 训练拟合优度只是诊断，不能冒充 out-of-time evidence；validation 一次按固定设计评价。
- 无 raw/F2/late-life、无需 checkpoint；无新 source full scan/hash。禁止 source helper 绕过边界。[S12][S13][S30][S31]

## 9. 时间顺序划分

沿用 original expanding folds：

|Fold|Train|Validation|
|---|---|---|
|1|[72.788690,946.393214)|[946.393214,1164.794345)|
|2|[72.788690,1164.794345)|[1164.794345,1383.195476)|
|3|[72.788690,1383.195476)|[1383.195476,1819.997738)|

以上根据完整 measurement interval/原 authority role，不按 nominal age 重新分配。无随机 split；无 inner alpha/grid。模型/bounds/候选数事先固定，不用 validation 调 variance formula。进入 validation 后参数固定，但更晚 origin 的 R 与 conditions 在观测完成后正常可用；不会因为“此前是 validation”就变成未来数据。[S31]

## 10. 稀疏锚点重复使用防线

六极化锚点不用于 variance fitting/selection/gate，不把已拟合的 B2 对它们重验叫外部验证。后续若展示已发生的 sparse reference，只能明确 posthoc description；当前草案无需读取原极化文件。

不得引用 ELTE 两点为该候选成功标准；两点已经打开且支持不对称。不得挑更有利 j*、更新 global B2、复用 PHI 全序列 trend 来“改善”目标。[S08][S11][S19][S34]

## 11. 不确定性评价

Primary：H6 equal-fold Gaussian CRPS（retention 单位），并报每 fold。Secondary：NLL、80/95% predictive coverage、interval width；coverage 给分子/分母，pooled 与 equal-fold 分开。

    CRPS = sigma * [x*(2*Phi(x)-1)+2*phi(x)-1/sqrt(pi)]
    x = (R_target-mu)/sigma

只用 conditional predictive interval，不称 latent-state CI、calibration posterior、真实寿命区间。mV-equivalent=retention error×0.71383023635×1000，不能与动态电压 RMSE 混称。[S14][S15]

点 MAE 必须严格等于原 Persistence（浮点允许范围另行锁定），因为所有 mean 一样；若不同先当实现故障。

不做 IID row bootstrap、跨窗口独立性显著性或新增种子；origin-target 重叠、同一堆相关性保留。样本小，coverage 只是该轨迹描述，不能从 45 queries 推到 population coverage。无需估计 failure probability。

## 12. 必要消融，不扩大搜索

唯一消融 b=0 与 age-only scale；共同 reference/queries/训练预算。报告 condition u 的训练支持与 extrapolation，无 outcome-driven exclusion。固定 80/95 区间、不试其他权重口径、不再增 variance 模块。没有 B2/CN/SG/EMD/PSDM/multitask ablation。

## 13. 最小成功标准（全部为未批准的项目设计）

同时满足才允许称“origin-conditioned uncertainty 有探索性效用”：

1. H6 equal-fold CRPS 至少比 GP 低 5%，并严格优于 Scale-only 与 Age-scale。
2. 每 fold CRPS 不超过该 fold GP 的 110%；不能用一折好覆盖另一折崩溃。
3. 95% pooled coverage 在 [0.85,1.00]、80% 在 [0.65,0.95]；同时每折单列 coverage/宽度，不把 pooled pass 写成每折均校准。
4. mean widths 有限、无 NaN/Inf；候选 equal-fold 95% mean width 不高于 GP，以防单靠扩大区间通过。该宽度规则也是项目选择，不是理论定理。
5. 点预测/误差与原 Persistence 精确一致；输入和目标时间资格、query 完整性全部通过。

阈值借鉴原 PSDM 宽松 calibration 设计，但 **不是重新定义旧 PSDM gate**。负责人若修改草案，应在新结果前完成；见结果后不能换主指标、bounds 或宽度规则。[S14][S15]

通过不表示真实 SOH/RUL成立；不通过也不证明所有条件信息都无用，只证伪该固定候选在当前数据下的资格。

## 14. 停止条件

来源/availability/identity/row universe 不可证明 → 技术/数据准入停止；不重 raw、不补目标。
实现 bug → 保留现场，只按冻结协议修复；不改科学规则。
数值问题/optimizer失败 → TECHNICAL_FAILURE，不能写 Insufficient Gain。
指标 gate 失败 → 科学负面结果，停止该候选；不追加 trial、模型、条件子集、校准器、gate、seed 或 horizon。
全部通过 → 只报告并交接；不自动长期外推、RUL或下一模型。

## 15. 新数据是否需要

这项有限探索原则上可消费现有 Development-only artifacts，无需新 raw。但前提是小型表中 availability 和映射可证明；若不能则需要额外明确授权的资料整理，而不是本草案隐含授权。

真正独立确认、真实 SOH 或 RUL 仍需要新可比参考观测/多轨迹/明确失效结局。一个 CPU experiment 不解决这些根本缺口。[S34][S35]

## 16. Test 与历史探索限制

NO Test/2000/2300/after-Air-Air/EIS target。禁止查看 late CSV 或 full N1 hash，不用已有 late 点调 variance。

原 Development 多次复用；本草案本身在已知 CMRF/PSDM 结果后提出，只能叫 exploratory。无 pristine independent holdout，不能承诺确认性论文主结论。[S19][S21][S37]

**执行开关：OFF。需要负责人明确批准新问题、输入资格、固定统计规则和唯一 protocol 后才能创建代码/拟合。当前只交付此草案。**

## 引用来源

[S07]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/reports/FINAL_P1S_PREANCHOR_AUDIT.md>
[S08]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAV_PAA_Anchor_Validity_Qualification/reports/FINAL_PAA_VALIDITY_QUALIFICATION.md>
[S10]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/reports/FINAL_AFFINE_CALIBRATION_CHARACTERIZATION.md>
[S11]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PHI_P1S_B2_Health_Index_Consolidation/reports/P1S_B2_HEALTH_INDICATOR_CONSOLIDATION_REPORT.md>
[S12]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTA_P1S_B2_Causal_Target_Audit/reports/P1S_B2_CAUSAL_TARGET_AUDIT_REPORT.md>
[S13]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CTAE_P1S_Reference_Availability_Erratum/reports/CTA_REFERENCE_AVAILABILITY_ERRATUM.md>
[S14]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/reports/PSDM_FEASIBILITY_REPORT.md>
[S15]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/outputs/PSDM_FEASIBILITY_SUMMARY.json>
[S19]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_ELTE_Exploratory_Late_Life_Target_Evaluation/reports/EXPLORATORY_LATE_LIFE_TARGET_EVALUATION.md>
[S21]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/docs/HISTORICAL_TEST_EXPOSURE_DISCLOSURE.md>
[S28]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CNP1S_Condition_Normalized_P1S_Development_Study/reports/CONDITION_NORMALIZED_P1S_DEVELOPMENT_STUDY.md>
[S30]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_H1E_Lead_Availability_And_Information_Set_Erratum/reports/H1_LEAD_AND_INFORMATION_SET_ERRATUM.md>
[S31]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/reports/H3_H6_ORIGIN_CONDITIONED_FORECASTING_REPORT.md>
[S34]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md>
[S35]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md>
[S37]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_L_Test_Eligibility_and_Reproducibility_Freeze/README.md>
[S39]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/configs/PREREGISTERED_PAA_PROTOCOL.yaml>
[S50]: <E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv>

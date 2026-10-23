# N — 竞争论文核对与创新边界

核查日期：2026-10-12。以出版商摘要/引言/可见方法片段为一手证据，完整方法无法取得时写UNKNOWN。ScienceDirect直接open若403，使用同出版页被索引的片段；不以二手题录补造公式。不是完整系统综述，没有读取RealHyFC原数据/含Test曲线的源论文。

## 最接近的五类工作

|工作及正式来源|网络/物理方程|训练、推理信息|数据与消融|出版状态、尚未核实|
|---|---|---|---|---|
|Chou、Wang，[Future prediction…TFT](https://www.sciencedirect.com/science/article/pii/S0960148125006810)，DOI10.1016/j.renene.2025.123019|BiLSTM改造TFT；没有可见电化学残差|RF特征筛选，Bacon-Watts起点，TPE，quantile；未来hour index已知，未来测量未知|IEEE PHM2014 FC1/FC2；原/改TFT、预测步数对照|正式期刊 Renewable Energy247，Jul2025，123019；逐项消融配置、预处理全血缘UNKNOWN|
|Ko等，[PINN long-term prognostics](https://www.sciencedirect.com/science/article/pii/S0306261925000480)，DOI10.1016/j.apenergy.2025.125318|operation time/current density→voltage PINN；膜与催化剂退化控制方程|物理与data loss；训练current density复制到预测阶段|IEEE PHM challenge；加入/移除先验对照在引言说明|正式期刊 Applied Energy382，15Mar2025，125318；完整方程/边界/辨识和消融数值UNKNOWN|
|Wei等，[Crossformer/physics](https://www.sciencedirect.com/science/article/pii/S0378775326006919)，DOI10.1016/j.jpowsour.2026.239941|conv、DSW/TSA Crossformer及电化学约束|三阶段渐进训练，历史voltage与工况输入；未来工况来源UNKNOWN|steady/quasi-dynamic为PHM2014，另dynamic精确来源UNKNOWN；卷积/physics开关及少数据对照|正式期刊 J.Power Sources677，15Jun2026，239941；逐式残差和参数约束未取得|
|He等，[Future-guided KD](https://www.sciencedirect.com/science/article/pii/S0306261926010779)，DOI10.1016/j.apenergy.2026.128423|offline Transformer教师→compact BiLSTM；可见内容未给电化学教师|teacher训练时前移n步，inference移除；精确loss/teacher target isolation UNKNOWN|PHM2014 FC1/FC2、steady/dynamic/cross-condition，采样间隔与训练比例实验；完整KD消融UNKNOWN|期刊在线记录，Applied Energy424 Dec2026为**未来编期**；online-first具体日期/版本标签未核实，不称已出December纸刊，不归为仅预印本|
|Liu等，[PINNCP](https://www.sciencedirect.com/science/article/abs/pii/S0951832026007556)，DOI10.1016/j.ress.2026.112945|graybox NN学习generalized power-law元参数，半经验电化学约束；j0、ohmic resistance、jlim|loss嵌方程，独立calibration及physics-aware conformal scores；未来工况假设UNKNOWN|static/quasi/dynamic、同类型stack迁移；精确数据名与消融表UNKNOWN|期刊在线记录，RESS276 Dec2026未来编期；online-first日期/完整方法未核实；不是仅预印本|

## 与本项目的实质比较

改进TFT已有明确先例；加入BiLSTM、TPE、quantile不能当F2首创。本项目冻结结构是历史LSTM/causal multiscale conv的encoder融合，不替换backbone；原TFT已有VSN、attention、gating。与2023 [TFT/covariates工作](https://www.sciencedirect.com/science/article/abs/pii/S0360319923014611)相比，本项目明确decoder不读取实际未来工况。该2023论文可见片段划分17个past-known/future-known特征，完整分类内容仍未核实，不能断言它泄漏或与F2同信息。

FGL 已经研究训练期未来知识→历史学生。N若只把学生换成F2并加KD，没有充分新颖性。本项目潜在差异只能是**独立认证的电化学结构**、完整目标/dwell时间块隔离、同privileged信息非物理teacher对照、future-unknown动态ΔV任务。这些是待证设计，不是已实现贡献；更不能据摘要不见crossfit就声称原FGL没有crossfit或存在泄漏。

Ko的复制负载长期预测与当前48→24未知future load不同；Crossformer已经组合时序网络与物理约束，TFT+PINN两个名字不构成首次。PINNCP的内状态识别和统计coverage是**作者主张**，无参数标签不能自行证明结构可辨识，conformal保证也不能无条件迁移到重叠非平稳RealHyFC queries。N不执行UQ或借其身份命名SOH/RUL。

## 检索误认和证据限制

2024 [Li-ion ITFT](https://www.sciencedirect.com/science/article/abs/pii/S0360544224008867)，DOI10.1016/j.energy.2024.131114 是电池，不是PEMFC。2026 [High-accuracy…](https://www.sciencedirect.com/science/article/pii/S0360319926027229)，DOI10.1016/j.ijhydene.2026.156084 是QMC-TPE PatchTST，不是改进TFT。均只做文献纠错，不引入新数据集或模型实验。

额外检索到IEEE TTE DOI10.1109/TTE.2026.3661158的semiempirical/ECM UQ题录，但本轮未取得可靠一手方法，列为DISCOVERY_ONLY_NOT_METHOD_EVIDENCE。二手页面的预配置电流/温度、PF/ESN描述不用于授权本项目。2026半经验物理信息与不确定性重点对照已由上表PINNCP一手片段覆盖，但其逐式/消融核验仍不完整。

上述主要工作均有期刊出版页，不将它们改称预印本。对于December2026两个记录，仅确认在线可检索与未来卷期，不猜首次online日期。全文未核实不阻止保守NO-GO，但会阻止“全面超越/首次/严格复现”宣称。

检索范围：PEMFC improved TFT、Ko125318、Crossformer239941、FGL128423、PINNCP112945、semiempiricalECM UQ及DOE/LUPI；实际查询与可核实/未知字段见 `evidence/N_LITERATURE_SEARCH_LOG.md`。

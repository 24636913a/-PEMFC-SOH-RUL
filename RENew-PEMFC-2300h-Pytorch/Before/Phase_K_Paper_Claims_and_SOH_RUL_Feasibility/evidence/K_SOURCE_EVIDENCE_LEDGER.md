# Phase K 来源证据台账

审查日期：2026-10-11（Asia/Shanghai）。基准 `469c06531a1d1c7fbd0a071f75204a983026cc8a`。来源快照见 [K_SOURCE_MANIFEST.json](../outputs/K_SOURCE_MANIFEST.json)；该文件记录完整相对路径、工作区SHA256、基准Git blob SHA256、Git blob ID、最后来源commit、访问语义及九checkpoint独立哈希。

## 证据分级

- `PROJECT_VERIFIED_EVIDENCE`：项目已冻结报告/协议与已审计摘要；K验证其身份/历史未变化，**不**重跑科研结果。
- `SOURCE_DOCUMENTED_FACT`：既有原资料审计摘录/公开metadata；标注是项目内二次摘录，未冒充本轮重新阅读完整原PDF。
- `METHOD_LITERATURE_SUPPORT`：下列官方文档/原论文支持成熟方法属性，不证明本项目的实验性能或全球首创。
- `RESEARCH_HYPOTHESIS`：未来映射、独立评价、long-range health任务；没有在K执行。
- `INSUFFICIENT_EVIDENCE`：独立门控收益、物理机理、连续真实SOH、EOL/failuretime、跨堆寿命。

不将研究假设写成已验证成果。未重新计算MAE/RMSE，未读取模型预测数组。

## 项目来源

所有下面路径相对repository root（不是相对K）；链接可直接打开原证据。SHA为**本地字节**SHA，不把Windows换行转换隐藏成字节相等。仅文本身份允许CRLF→LF规范化与基准blob比对，二者原SHA分别保存在JSON；checkpoint绝不作文本转换。当前来源版本即冻结基准中该文件版本，最后修改commit见JSON。

|ID|源文件|本轮用途/证据范围|工作区SHA256|
|---|---|---|---|
|E01|[RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/outputs/T1R_FINAL_DECISION.json](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/outputs/T1R_FINAL_DECISION.json>)|T1-R frozen trial status、所选配置、INSUFFICIENT_GAIN；不导入runner|`b8e180df44248dfeadf205e93f8fbea8ce34876a81c7b698d09ab1ced229ce3c`|
|E02|[RENew-PEMFC-2300h-Pytorch/Phase_TFT_FullFeature_CausalFilter_Comparison/reports/E_SERIES_EXPERIMENT_REPORT.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TFT_FullFeature_CausalFilter_Comparison/reports/E_SERIES_EXPERIMENT_REPORT.md>)|E0–E5探索性输入处理比较、公共scaler及历史复现限制|`ce1184039796a0e683ab85f82cfa501ab70c7efca39a56b0cb280c85e23b11c1`|
|E03|[RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/reports/F_FINAL_EXPERIMENT_SUMMARY.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/reports/F_FINAL_EXPERIMENT_SUMMARY.md>)|F0/F1/F2单Seed初始结构比较，保留负面Assessment证据|`f31027fc3fb19db06b1cc39449ea25f9694d8d5742345f16958cb398fb9e4641`|
|E04|[RENew-PEMFC-2300h-Pytorch/Phase_H_Determinism_MultiSeed_Stability/reports/H_FINAL_CONCLUSION.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_H_Determinism_MultiSeed_Stability/reports/H_FINAL_CONCLUSION.md>)|H正常CUDA多SeedValidation、非bitwise重复性与三Seed表|`4bdf976c68e978babec2cc73ddecff1234ed0047038bda0073b84db10c226505`|
|E05|[RENew-PEMFC-2300h-Pytorch/Phase_I2_Frozen_Assessment_Evaluation/reports/I2_FINAL_SCIENTIFIC_CONCLUSION.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_I2_Frozen_Assessment_Evaluation/reports/I2_FINAL_SCIENTIFIC_CONCLUSION.md>)|I2复用Development Assessment多Seed汇总、target集中误差限制|`a40eba46f50bdefbb5ec1342aa182144ed44a07c0b2981d1a113a344f452adf3`|
|E06|[RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_FINAL_ARCHITECTURE_DECISION.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_FINAL_ARCHITECTURE_DECISION.md>)|J最终结构冻结/门控未决；原异常与cache-only完成史|`35b6dc5ae56f22d92ec40cd09b6c47ab56579878072f05a22ef8f7d69b82f8fd`|
|E07|[RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/J_SUMMARY.json](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/J_SUMMARY.json>)|J已审计数字汇总；仅读取汇总，不读取预测cache|`de9e02adabd61816d690172217999550dbe03abff02b79179f16f9fea187af48`|
|E08|[RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/integrity_audit/J_FINAL_READONLY_AUDIT.json](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/integrity_audit/J_FINAL_READONLY_AUDIT.json>)|J身份、31测试/48复算和只读完整性结论|`2996fdd716a5b7674166afea9aee9d050a55276abd1ce222edc128efa806b0da`|
|E09|[RENew-PEMFC-2300h-Pytorch/Phase_TFT_FullFeature_CausalFilter_Comparison/configs/PREREGISTERED_E_COMPARISON.yaml](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TFT_FullFeature_CausalFilter_Comparison/configs/PREREGISTERED_E_COMPARISON.yaml>)|E冻结特征顺序、预处理任务协议（静态）|`b17794132351454d93debfc67943dc14db73a5287fef1966ba748967e9852332`|
|E10|[RENew-PEMFC-2300h-Pytorch/Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py>)|E共享scaler、mask/滤波调用（静态，不执行）|`f479d4cd64a6a9351dd2d863f38dd4828017dc48ffa8b4e986a3c240a5574099`|
|E11|[RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/configs/F0_F2_EXECUTION_PROTOCOL.yaml](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/configs/F0_F2_EXECUTION_PROTOCOL.yaml>)|F正式48→24、Top10/EWMA、结构、指标协议|`c8ee4be440b9828cad04b615ddd19c15781033fdb22bbadeb1df18a59cf74cbf`|
|E12|[RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/models/causal_multiscale_branch.py](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/models/causal_multiscale_branch.py>)|因果多尺度depthwise Conv的真实实现（静态）|`d70bbb0093a0e06b41325a09de48ac18a97a223bb6a4259941ae17a2ea1b220c`|
|E13|[RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py>)|5→8→1门控、历史上下文及融合公式（静态）|`bbc219e5d22a664f1a45dbe7ac5d8a561a0bfc6d63097a7f847612e8af352c57`|
|E14|[RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py>)|双路径插入位置、native decoder保留（静态）|`a8e979c00a753c7a65de3e200ff0e7cb9562888d9555c63b94cef97f28fececc`|
|E15|[RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/SOH_ANCHOR_SELECTION.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/SOH_ANCHOR_SELECTION.md>)|Development-only j*选择与sparse归一化，未读Test锚点|`91da7cc136188c0e4eabfe726705b170214e5d969a03bff6f931ca99725fa745`|
|E16|[RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/soh_selection_summary.json](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/soh_selection_summary.json>)|已冻结j*、reference、configuration结论|`0ca2655b520e5675f49c2a1d0d564291e5672683edb4f0364d0f1941e9ef122e`|
|E17|[RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/polarization_configuration_compatibility.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/polarization_configuration_compatibility.md>)|标称10cell与8cell敏感性/EIS可比性、原资料协议摘录；可能出现Test来源元数据，无Test电压值用于K|`b1f6a5ee9b3c0010ce5a8918793569fcdd4dcf4f4c3f6ba058bf9b5f957524be`|
|E19|[RENew-PEMFC-2300h-Pytorch/Phase_HA_EMD_Health_Representation_Audit/reports/FINAL_HEALTH_REPRESENTATION_AUDIT.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_HA_EMD_Health_Representation_Audit/reports/FINAL_HEALTH_REPRESENTATION_AUDIT.md>)|HA正式健康证据不可辨识；DF预测结论不变|`442a8c9bedb8a8551e1d8aae7d0ee818b7464eb6f0cb3853a334feab65e21366`|
|E20|[RENew-PEMFC-2300h-Pytorch/Phase_HA_EMD_Health_Representation_Audit/reports/POST_ANALYSIS_REPORTING_CLARIFICATION.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_HA_EMD_Health_Representation_Audit/reports/POST_ANALYSIS_REPORTING_CLARIFICATION.md>)|HA pointestimate不能当age证据的澄清|`e40616664848ab6fd3f557edf5e0260eb0dca45cb7df84c8becb02f9d8252ed9`|
|E21|[RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/04_OFFICIAL_EOL_AUDIT.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/04_OFFICIAL_EOL_AUDIT.md>)|官方EOL既有原资料摘录；不读取RUL90全数据/阈值结果|`f0ddc2f357fe691ad40a6e5ec80b8cee941a76ee2ddee7270c3c4b83833419a9`|
|E22|[RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/PHASE_N1_RESULTS.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/PHASE_N1_RESULTS.md>)|N1数据/采样/坐标/窗口协议及任务资格；仅文档，不开causal表|`e4954c93beac90ccf3bff63a84a54eb3b63383b88a56bcaf44e8cf2bd7040015`|
|E23|[RENew-PEMFC-2300h-Pytorch/Phase_CSTL_Chronology_Safe_Pretrained_TimesNet_Health_Transfer/reports/PRETRAINED_TIMESNET_HEALTH_TRANSFER_REPORT.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_CSTL_Chronology_Safe_Pretrained_TimesNet_Health_Transfer/reports/PRETRAINED_TIMESNET_HEALTH_TRANSFER_REPORT.md>)|Development-only chronology-safe P1S-B2迁移及其未晋级结果；非极化dense真值|`0001f0b58e53e08591c9da2c34e358beb2c01d72f578883b6ff71e93a50b932c`|
|E24|[RENew-PEMFC-2300h-Pytorch/Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts/run_c_ablation.py](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts/run_c_ablation.py>)|Spearman评分与EWMA初始化真实实现（静态）|`e0053c63d6b942166362a09b2fce0166d9d29130fc0d5624eeaa0dcf32a65ce1`|
|E25|[RENew-PEMFC-2300h-Pytorch/Phase_TFT_Original_Baseline/models/canonical_tft.py](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_TFT_Original_Baseline/models/canonical_tft.py>)|Canonical TFT原body/strict futureunknown输入接口（静态）|`b6540cdb109565eb40fa9d631984699640a74233d3e869bca163e37f02db3618`|
|E26|[RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/models/frozen_feature_schema.py](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/models/frozen_feature_schema.py>)|Raw F2特征schema依赖来源（静态）|`f5b7e215d06099dada361c915b2260ac8d79da3661a8932ebba09ab550b9772e`|
|E27|[RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/checkpoint_identity/J_NINE_CHECKPOINT_IDENTITY.json](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/checkpoint_identity/J_NINE_CHECKPOINT_IDENTITY.json>)|J九个H checkpoint身份元数据；只对文件做字节identity校验，不load模型|`480a0ab2857a463c0094f02ee3748b9871faec30be6d8f2da140fe8cf207dd65`|
|E28|[RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_INNOVATION_EVIDENCE_MATRIX.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_INNOVATION_EVIDENCE_MATRIX.md>)|J既有创新证据与安全措辞|`198f6fa4e4b0a32760f2269b4ee3c071779ee2e831b49ab42fcf1920397a7932`|
|E29|[RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_SCIENTIFIC_LIMITATIONS.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_SCIENTIFIC_LIMITATIONS.md>)|J限制：选择偏差、门控机理、SOH/RUL缺口|`6999f5698d020f49a7f80702af1d50fa8ee01b2c178c425f16d97fb9f3154819`|
|E30|[RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_CACHE_ONLY_INTERFACE_CORRECTION.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_CACHE_ONLY_INTERFACE_CORRECTION.md>)|J原报告接口BLOCKED保留与独立cache-only修复解释|`2306170f63ebc42209f848be66eca6710df589f98d88847dd651744b3c502a63`|
|E31|[RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/reports/F_IMPLEMENTATION_AUDIT.md](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_F_TFT_DualPath_RegimeFusion/reports/F_IMPLEMENTATION_AUDIT.md>)|F启用路径engineering测试与参数量|`bf8f7c78c44aa6b77488df9ef5f71efe3930ec4b2da50681f938e13d7baacbc3`|

## E18：混合锚点CSV的严格前缀，不是全文来源

源：[soh_reference_anchors.csv](<E:/沈农信电/ PEMFC-SOH+RUL/RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/soh_reference_anchors.csv>)。

只读取header+8个Development记录，均先验在许可前缀内；7nominal（BoL/300/600/850/1150/1500/1750）+1500afterAirAir diagnostic。实现采用无缓冲逐字节读取至9个LF，停止前不会预读suffix；禁止读取整个Git blob或整文件hash。原始split均为train/validation且age<1819.997738。

Fingerprint类型：`ALLOWED_DEVELOPMENT_ANCHOR_LOGICAL_PREFIX`。
Fingerprint：`9a5ac8f48a8e780a45bad6a29a01ecfb9c82744e08d5f5fed88209ebfc022d39`。
Serialization：ordered compact JSON / UTF-8，固定原列顺序与record顺序。它只证明**许可逻辑前缀的本轮快照不变**，不称完整源文件SHA；K没有改写历史N1full-hash记录。

来源commit只通过Git元数据核验，不读取mixed CSV的完整blob；K prefix不能声称独立核验全文件来源。未读CSV第9条data record及后缀、未打开causal feature表、未读Raw Loadcycle/EIS/polarization文件。

## 公开方法与原资料元数据（定点核查）

|ID|主来源及可引用范围|
|---|---|
|L1|[SciPy官方spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)：Spearman衡量秩/单调关联；不授予物理因果解释。在线版本不是本地环境版本证明。|
|L2|[NIST EWMA](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm)：指数递推平滑是成熟统计方法；项目窗口初始化/因果访问规则另见E24，不将control-chart假设搬为本模型证据。|
|L3|Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)，DOI10.48550/arXiv.1907.10902；成熟软件框架来源，不证明T1-R晋级。|
|L4|[Optuna官方TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)：TPE采样器算法来源；本地冻结搜索实际设置以项目协议为准。|
|L5|Lim, B., Arik, S. O., Loeff, N., Pfister, T. (2019/2020). [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)，DOI10.48550/arXiv.1912.09363；原TFT已含recurrent/local处理、attention、variable selection和gating。|
|L6|[NIST Censoring](https://www.itl.nist.gov/div898/handbook/apr/section1/apr131.htm)：failure未观测时可存在right-censoring；本数据具体failure criterion及终止机制仍需资格核验。|
|L7|[RealHyFC数据集公开元数据](https://zenodo.org/records/20715542)：题名与相关Fensterle等IJHE论文DOI10.1016/j.ijhydene.2026.156060、HyFaB/RealHyFC说明。**仅metadata**，没有下载dataset/原PDF、没有外部请求N1内容/Test曲线。|

本轮检索范围：TFT原文/Optuna原文、官方EWMA/Spearman/TPE/censoring文档及RealHyFCbibliographic metadata。检索用于来源归属与概念核验；不是系统性novelty检索，不声称找不到全球同类架构。论文DOI与不同Fuel Cells文章不可混用；本轮未查阅含Test数值的原全文。以上原资料操作条件/500mV语境仅通过E17/E21审计摘录，归为SOURCE_DOCUMENTED_FACT（secondary extraction）。

## 冻结身份与执行历史

J original protocol `0244f05fd777ccd615fda7eb9533d129323dc27fbb58212e091130011d0d9b24`；
J original code identity `67b0d3588f75db56d435ddc5277d38324708f660ac10cc1850a5908d771612c8`；
cache completion manifest `ac2652eff04eb5ff72360446a5a629932fcd21101000741e55576b54b22023ff`；
complete execution identity `8d461f63bdcd4212634b962f2b875ed3f49a954a5515fafec51df5d997bad43c`。
这是E06/E08保留的历史身份，不是K重新运行J的授权。J原wrapperBLOCKED与后续cache-only完成分开记录。

K初次identity audit发现Windows历史文本CRLF与Git LF的**字节哈希不同**，停止后修正K的文档身份比较为明确CRLF→LF normalization，同时仍独立保存raw SHA和base SHA。未改任何历史文本或checkpoint；这不是科研结果变化。

提交前移除了K新文件末尾的额外空行，并仅更新本阶段访问清单的字节SHA记录；源文档、checkpoint和Development锚点前缀快照均未改变。本快照是已知结果后的审计记录，不是模型实验预注册。

## 反驳检查与限制

1. F2比分支对照好不等于门控独立收益：J固定gate方向混合；主张已限制为整体架构经验效果。
2. 稀疏proxy趋势好不等于连续SOH：相同检查点用于选择、测量内部相关、共享denominator、条件恢复均限制独立性。
3. 有500mV文字不等于已验证EOL：保留原限值语境，既不抹去它也不自造failure标签。
4. 一个stack/三seed/重叠window不能作为多独立lifetimes或统计显著性证据。
5. 此次没有OS级全进程I/O monitor；只读结论基于明确执行清单、production白名单/限定读取、防线测试、源身份与Git差异证据。

本台账由deep-research来源等级与academic-paper主张约束辅助编制。没有执行模拟审稿人/完整多agent流水线；未来论文需研究者复核资料和投稿表述。

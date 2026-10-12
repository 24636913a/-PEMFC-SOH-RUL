# O1 — 冻结Validation缓存指标与身份协议

Stage 0冻结；基准 `7553fcf3ad5f2b10bcf1f590ce90d19d46468bb4`。本文件必须先于正式重算结果生成，SHA单独保存；任何排名产生后不得修改公式、筛样本或阈值。本阶段仅新增O1文件。

## 输入与身份

只允许H `outputs/validation_predictions/H-F{0,1,2}-Seed{42,52,62}.csv`共九文件。预期哈希来自H `outputs/metrics/H_VALIDATION_METRICS.csv`的`prediction_file_sha256`；role必须MULTISEED_PRIMARY。Repeat-A/B不读取、不计算。小型authority只读H frozen YAML/SHA/指标表/静态窗口审核源码，L frozen model manifest，J learned指标/原H一致性表/身份manifest，O最终判断和缺口报告。

运行之前按白名单逐源记录/验证SHA；不打开manifest引用的权重。禁止raw/N1/Assessment逐点/Test访问、forward、训练、新预测。H protocol SHA必须为 `88aba09c7f9a08d43f658f9c66e482713bbb60607bc88bb8059f3638d5aad9dd`。

Validation `[2103,2404)`，历史48、预测24、0.5h；每run205origins/4920queries，h1…24各205。数量必须同时与H frozen protocol及H正式metrics的records核对，不仅依赖本指令。keys `(origin_row,target_row,horizon_step)`，target=origin+h，所有target在2103…2403；现有origin间断继承保存键宇宙，不填补。按历史H `audit_h_predictions.py`静态代码验证构造规则，不导入旧脚本或loader。

完整解析、所有identity列唯一且正确、整数键/无重复、24步完整、finite、九run共同keys与真值**逐项精确相等**。明确使用键对齐用于检查和配对，不改写原CSV、不删除/自动去重/补齐。保存行序纳入哈希；比较时排序key仅用于显式连接，绝不以排序掩盖缺失或不匹配。任一失败：`VALIDATION_CACHE_IDENTITY_BLOCKED`，不发布比较数值/正式图。

## 数值定义（V→mV）

CSV使用pandas `float_precision='round_trip'`，float64；`e=(y_pred_delta_V-y_true_delta_V)*1000`。主指标 `FORMAL_QUERY_WEIGHTED_RMSE=sqrt(fsum(e²)/N)`；MAE=`fsum(abs(e))/N`。绝对误差P95/P99使用numpy quantile(method='linear')；max保留全部样本。与H/J的mV数值容差统一绝对 `1e-10 mV`，不设容差改变原结果；真值/整数/identity/hash必须exact，不用误差容差放行。

Horizon h1…24各自计算全部五项query指标及n，h24为12h endpoint同一计算。目标等权 `sqrt(mean_target(mean_queries(e²)))`；不能先平均预测做ensemble。每目标存multiplicity、SSE和meanSE。

SSE集中度基于正式query SSE：最大目标、top5、top10 SSE/全部SSE；并列最大target_row/multiplicity。排序按SSE降序、target_row升序决定tie。零总SSE情况下share=0，仍记录数量。唯一target只是索引去重，不等于统计独立；queries也不是独立样本。无p值/bootstrap/CI/显著性宣称。

三Seed只做各run指标等Seed算术均值与sampleSD(ddof=1)，不pool预测、不做ensemble。所有指标按原主地位并列，不事后换指标。paired方向固定新模型减基线：F2−F0、F2−F1、F1−F0。每seed及horizon保留；不剔除极端query/target、不挑seed。

## 历史一致性与状态

Stage1身份全部通过后才能计算。Formal RMSE/MAE逐run对H；F2 learned的RMSE/MAE/P95/P99/endpoint/max/target等权对J。同权重身份通过L与J manifest联接；J已有`all_4920_keys_truth_and_prediction_exact`报告另列历史证据，不将其冒充本次逐点J缓存重新读取。若指标超容差/身份未能闭合停止，保存差异、状态BLOCKED，不擅自合并。J fixed .5仅沿用历史诊断，不重跑。

研究判断预定：F2 formal等Seed均值不优于两基线之一则`F2_ADVANTAGE_NOT_SUPPORTED`；优于两者且同seed formal均优、所有报告的error/集中share与两基线各seed均无劣化才`F2_ADVANTAGE_ROBUST_ACROSS_REPORTED_METRICS`；formal整体优势成立但任何其它报告指标/seed有反转则`F2_ADVANTAGE_METRIC_DEPENDENT`。身份/复算未通过为`INCONCLUSIVE`。这是描述性状态，集中share小本身不代表整体精度更好。

全部输出/交叉核验/图表完成才`VALIDATION_EVIDENCE_COMPLETED`；身份失败为BLOCKED；部分合法交付为PARTIALLY_COMPLETED。任何状态不升级独立Test、跨stack、gate机理、SOH/RUL/Physics资格。

## 图与停止

只从九合格缓存生成PNG300dpi+PDF矢量候选图；同组seed panel统一尺度，标签Validation/mV/seed，非独立Test；不靠截断轴掩盖最大误差。全部horizon、三seed及基线均展示。合成测试先通过再正式运行。旧O“447独立target rows”仅在O1勘误附录更正为“447个唯一目标时刻”，不修改O或重读Assessment。

生成完整表/图/报告后仅commit/push O1，停止。不新增实验或打开Test。

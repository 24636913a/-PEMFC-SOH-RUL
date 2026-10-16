# 独立文献一致性核验

日期：2026-10-17。状态：`PASS_WITH_MINOR_WORDING_NOTE`。

核验对象为最终01/02/03/REFERENCES、两份原始提取JSON/说明、官方书目与153条字段。此次没有独立重读全部7篇；重点另查Yi p2及Ko首页，并核对7份原PDF SHA。没有读取实验数据、加载模型或训练。

|检查|结论|证据|
|---|---|---|
|153_unique_fields|PASS|9 papers x 17 unique fields|
|9_unique_papers|PASS|P1-P6,A1-A3|
|source_extractions_vs_official_metadata|PASS|All 9 DOIs, titles, full author lists and bibliographic years agree|
|all153_values_grades_locators_exact_source_match|PASS|Each exported cell equals independently loaded extraction JSON|
|long_wide_tables_exact_match|PASS|Wide and long evidence tables preserve every value|
|all11_bibliographies_include_correct_authors_title_year_doi|PASS|Includes original TFT and RealHyFC data paper|
|seven_fulltext_two_abstract_only|PASS|Local availability independently inspected; full-text reading scope inherited from separate direct reviewers|
|seven_local_pdf_sha_match|PASS|Raw PDFs hashed, no experimental data file read|
|Tu_Yakut_unknown_not_negative|PASS|Both absences of full text remain UNKNOWN, never no ablation|
|Yi_dataset_correction_retained|PASS|Independently checked Yi PDF p2 section 2: 90kW, 400 serial cells,1374h, polarization j=.7A/cm2|
|Ko_four_authors|PASS|Independently checked Ko PDF first page; four authors|
|physical_horizon_conversion_guard|PASS|P1 steps not silently mapped to hours; P2 unequal platforms; P3 trial duration not horizon; P5 unknown|
|future_known_not_accused_leakage|PASS|Information-set mismatch is documented; author intent/unpublished code not inferred|
|existing_ablations_not_denied|PASS|P4 native/feature, P6 module, Jiang variants and Ko failed ablations retained|
|no_cross_paper_error_ranking|PASS|Manual review of final 01/02/03 and refs found no cross-dataset error leaderboard|
|no_false_first_or_independent_gate_gain|PASS|Novelty is specific interface/implementation, not first TFT/CNN/unknown-future forecasting|

## 科研边界判断

9篇×17字段完整，全部长表值/等级/定位与原提取逐条一致；宽表与长表一致。作者、DOI、题名及年份与官方元数据一致。7篇本地全文与2篇摘要/有限片段区别保留；Tu、Yakut的未知不被变成否定。Yi的400cell自建台架更正、Ko四作者、物理跨度不能随意换算、未来known不自动指控泄漏、已有消融与失败不被抹除、禁止跨论文误差排名均保留。

未发现阻断本轮文献对标的实质性错误。该结果只确认文献材料的一致性，不证明本项目模型独立增益、无偏稳定性或投稿资格。

## 已执行的微小修正

经明确授权，core_P1_P5_notes.md访问表中的“Yurdagül Benteşen Yakut等”已改为单作者措辞。未修改任何历史资料或其他人的科研字段。

## 两项非阻断建议

1. Crossref online字段为空是数据库缺少该字段，不是本地原PDF没有online日期。已核对7份首页：P1 2022-03-08；P2 2023-04-07；P4 2025-04-01；P6 2026-03-25；A1 2026-01-14；A2 2025-02-25；A3 2025-01-08。建议单设local_pdf_online_date_observed并保持Crossref原值。当前正式引用年份正确。
2. 02开头“仅摘要可得的两篇不作为图号证据”可改成“不作为未经原文片段支持的完整图号/章节结构证据”。P3的Fig1/Fig2a/Table1来自明确标记的出版社片段，正文实际没有升级为全文证据。

审查时文件SHA及全部机器检查见同名JSON。后续稿件形成后仍需单独检查正文主张与引用，不能以本次审查代替稿件终审。

## Online日期建议的收口

已按新增授权写出 `outputs/evidence_index/LOCAL_PDF_ONLINE_DATES.csv`：7篇首页实际印刷日期由独立PDF提取程序逐条解析，含原行、p1定位、DOI、原PDF SHA。Crossref的null字段保留原含义，不修改官方元数据。该项已解决；剩余仅02开头的非阻断措辞建议。

# Phase O — 全部新增成果路径

仓库绝对根：`E:\沈农信电\ PEMFC-SOH+RUL`。

本阶段绝对目录：`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_O_F2_Paper_FullText_Novelty_Audit`。

下面全部路径均以该阶段目录为基准，可点击进入。只新增本阶段，不覆盖任何历史文件。

## 研究报告

- [README.md](README.md)
- [reports/O_01_TFT_FULLTEXT_COMPARISON.md](reports/O_01_TFT_FULLTEXT_COMPARISON.md)
- [reports/O_02_DUAL_PATH_AND_DILATED_CNN_COMPARISON.md](reports/O_02_DUAL_PATH_AND_DILATED_CNN_COMPARISON.md)
- [reports/O_03_PHYSICS_INFORMED_PRIOR_ART_COMPARISON.md](reports/O_03_PHYSICS_INFORMED_PRIOR_ART_COMPARISON.md)
- [reports/O_04_INFERENCE_INFORMATION_SET_AUDIT.md](reports/O_04_INFERENCE_INFORMATION_SET_AUDIT.md)
- [reports/O_05_ARCHITECTURE_OVERLAP_MATRIX.md](reports/O_05_ARCHITECTURE_OVERLAP_MATRIX.md)
- [reports/O_06_F2_NOVELTY_ASSESSMENT.md](reports/O_06_F2_NOVELTY_ASSESSMENT.md)
- [reports/O_07_F2_EXPERIMENT_EVIDENCE_AUDIT.md](reports/O_07_F2_EXPERIMENT_EVIDENCE_AUDIT.md)
- [reports/O_08_REVIEWER_CRITICISM_AND_RESPONSES.md](reports/O_08_REVIEWER_CRITICISM_AND_RESPONSES.md)
- [reports/O_09_PAPER_TITLE_ABSTRACT_OUTLINE.md](reports/O_09_PAPER_TITLE_ABSTRACT_OUTLINE.md)
- [reports/O_10_FINAL_PUBLICATION_DECISION.md](reports/O_10_FINAL_PUBLICATION_DECISION.md)

## 来源与身份

- [evidence/O_PDF_IDENTITY_AND_SOURCE_LEDGER.csv](evidence/O_PDF_IDENTITY_AND_SOURCE_LEDGER.csv)：21份实际首页身份、SHA、DOI、阅读范围。
- [evidence/O_FULL_TEXT_EVIDENCE_LEDGER.md](evidence/O_FULL_TEXT_EVIDENCE_LEDGER.md)：7篇全文、5篇补充的逐主张页码/图/式/表及UNKNOWN。
- [evidence/O_F2_CODE_AND_CHECKPOINT_IDENTITY.md](evidence/O_F2_CODE_AND_CHECKPOINT_IDENTITY.md)：模块张量/公式、源码位置、历史九权重身份。
- [evidence/O_HISTORICAL_FIGURE_INDEX.csv](evidence/O_HISTORICAL_FIGURE_INDEX.csv)：44个H/J/I2历史图文件名；未读像素重算，格式重复不算44项独立实验。
- [outputs/O_STATIC_SOURCE_MANIFEST.json](outputs/O_STATIC_SOURCE_MANIFEST.json)：35个批准静态来源的当前文件SHA、基准Git blob SHA与来源提交。

## 审计和测试

- [outputs/O_READONLY_EXECUTION_AUDIT.json](outputs/O_READONLY_EXECUTION_AUDIT.json)
- [outputs/O_DOCUMENT_GUARD_TEST_LOG.txt](outputs/O_DOCUMENT_GUARD_TEST_LOG.txt)
- [outputs/O_DOCUMENT_GUARD_TEST_RESULTS.json](outputs/O_DOCUMENT_GUARD_TEST_RESULTS.json)
- [scripts/o_pdf_tools.py](scripts/o_pdf_tools.py)：仅论文提取/渲染。
- [scripts/o_catalog.py](scripts/o_catalog.py)：人工核对的身份清单。
- [scripts/run_o_document_audit.py](scripts/run_o_document_audit.py)：白名单静态来源/文档/历史身份核验。
- [tests/test_o_document_guards.py](tests/test_o_document_guards.py)：8项生产访问/哈希/manifest防线测试。
- [.gitignore](.gitignore)：内部论文全文、渲染图和缓存不入Git。
- [.gitattributes](.gitattributes)：文本行尾/空白规则。
- [O_DELIVERABLE_INDEX.md](O_DELIVERABLE_INDEX.md)：本索引。

## 内部且不提交

`runtime/`存放原论文文本提取与内部渲染图片。它们不是本项目原创研究图，也不是新的性能结果；不上传，不拿原论文图作为投稿插图。正式交付不依赖Git包含原文全文。

没有新增checkpoint、预测缓存、数据表、科研性能或训练结果。审计PASS是文件/文献整理完成，不是模型独立Test通过。

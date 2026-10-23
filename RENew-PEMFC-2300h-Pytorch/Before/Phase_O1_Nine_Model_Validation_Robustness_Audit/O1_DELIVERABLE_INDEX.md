# O1成果路径索引

绝对阶段目录：`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_O1_Nine_Model_Validation_Robustness_Audit`。

## 协议与源码

- [README.md](README.md)
- [protocol/O1_FROZEN_METRIC_PROTOCOL.md](protocol/O1_FROZEN_METRIC_PROTOCOL.md)
- [protocol/O1_FROZEN_METRIC_PROTOCOL.sha256](protocol/O1_FROZEN_METRIC_PROTOCOL.sha256)
- [protocol/O1_ALLOWED_CACHE_MANIFEST.json](protocol/O1_ALLOWED_CACHE_MANIFEST.json)
- [scripts/run_o1_cache_audit.py](scripts/run_o1_cache_audit.py)
- [scripts/o1_figures_reports.py](scripts/o1_figures_reports.py)
- [tests/test_o1_synthetic_metrics.py](tests/test_o1_synthetic_metrics.py)

## 全部数值与审计表

- [O1_NINE_MODEL_VALIDATION_METRICS.csv](outputs/O1_NINE_MODEL_VALIDATION_METRICS.csv)
- [O1_HORIZON_METRICS.csv](outputs/O1_HORIZON_METRICS.csv)
- [O1_HORIZON_SEED_SUMMARY.csv](outputs/O1_HORIZON_SEED_SUMMARY.csv)
- [O1_EQUAL_SEED_SUMMARY.csv](outputs/O1_EQUAL_SEED_SUMMARY.csv)
- [O1_TARGET_SSE_CONCENTRATION.csv](outputs/O1_TARGET_SSE_CONCENTRATION.csv)
- [O1_PAIRED_MODEL_COMPARISON.csv](outputs/O1_PAIRED_MODEL_COMPARISON.csv)
- [O1_PAIRED_HORIZON_COMPARISON.csv](outputs/O1_PAIRED_HORIZON_COMPARISON.csv)
- [O1_PAIRED_QUERY_ERROR_DIFFERENCES.csv](outputs/O1_PAIRED_QUERY_ERROR_DIFFERENCES.csv)
- [O1_SEED62_EXTREME_QUERIES.csv](outputs/O1_SEED62_EXTREME_QUERIES.csv)
- [O1_HISTORICAL_METRIC_DIFFERENCES.csv](outputs/O1_HISTORICAL_METRIC_DIFFERENCES.csv)
- [O1_IDENTITY_AND_RECOMPUTATION_AUDIT.json](outputs/O1_IDENTITY_AND_RECOMPUTATION_AUDIT.json)
- [O1_SYNTHETIC_TEST_RESULTS.json](outputs/O1_SYNTHETIC_TEST_RESULTS.json)
- [O1_SYNTHETIC_TEST_LOG.txt](outputs/O1_SYNTHETIC_TEST_LOG.txt)

## 六份正式报告

- [身份](reports/O1_VALIDATION_IDENTITY_REPORT.md)
- [完整稳健性](reports/O1_NINE_MODEL_ROBUSTNESS_COMPARISON.md)
- [Seed62](reports/O1_SEED62_EXTREME_ERROR_ANALYSIS.md)
- [O文字勘误](reports/O1_PHASE_O_TERMINOLOGY_ERRATA.md)
- [论文候选图表](reports/O1_PAPER_FIGURE_AND_TABLE_INDEX.md)
- [最终判断](reports/O1_FINAL_RESEARCH_DECISION.md)

## 原创图（每项另有同名PDF矢量版）

- [全部Seed/24步RMSE](figures/Fig1_Validation_Horizon_RMSE_Three_Seeds.png)
- [12h endpoint](figures/Fig2_Validation_12h_Endpoint.png)
- [Formal/target等权](figures/Fig3_Formal_vs_Target_Time_Equal.png)
- [P95/P99/max](figures/Fig4_Validation_Tail_and_Max_Error.png)
- [Top1/Top5 SSE](figures/Fig5_Validation_Target_SSE_Concentration.png)
- [等Seed mean/sampleSD](figures/Fig6_Validation_Horizon_Equal_Seed_Mean_SD.png)

图不可改称独立Test/跨stack/RUL结果。不删除异常数据，不以诊断改模型或正式指标。旧阶段保持不变。

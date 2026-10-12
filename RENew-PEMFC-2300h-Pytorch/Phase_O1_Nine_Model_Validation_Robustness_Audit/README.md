# Phase O1 — 九模型冻结Validation稳健性补证

基准 `7553fcf3ad5f2b10bcf1f590ce90d19d46468bb4`；只新增本阶段。

身份：`VALIDATION_CACHE_IDENTITY_PASS`。

稳健性：`F2_ADVANTAGE_METRIC_DEPENDENT`。

证据：`VALIDATION_EVIDENCE_COMPLETED`（仅本次Validation缓存补证范围）。

九run各205origins、4920queries、251个唯一目标时刻；三seed为42/52/62，Repeat-A/B排除。所有来源SHA、身份、共同keys/真值、H/J数值一致性通过。正式等Seed平均RMSE：F0=9.067563、F1=9.155598、F2=7.245400mV。F2整体正式优势不等于其它seed/尾部/端点/目标等权均优势，尤其Seed62仍有2247/2248低重数目标的极端残差。

## 入口

- [最终决策](reports/O1_FINAL_RESEARCH_DECISION.md)
- [九模型完整比较](reports/O1_NINE_MODEL_ROBUSTNESS_COMPARISON.md)
- [Seed62精确目标诊断](reports/O1_SEED62_EXTREME_ERROR_ANALYSIS.md)
- [全文成果索引](O1_DELIVERABLE_INDEX.md)

## 可复现命令

从仓库根目录执行：

```powershell
python -B RENew-PEMFC-2300h-Pytorch/Phase_O1_Nine_Model_Validation_Robustness_Audit/tests/test_o1_synthetic_metrics.py
python -B RENew-PEMFC-2300h-Pytorch/Phase_O1_Nine_Model_Validation_Robustness_Audit/scripts/run_o1_cache_audit.py
```

最初的 `--freeze` 在正式数值重算前执行，已冻结后禁止覆盖。协议SHA见protocol目录；来源清单区分metadata全文件SHA与九缓存预期SHA。只导入numpy/pandas/yaml/matplotlib及O1工具，不导入旧model/loader、torch或训练runner。

脚本只读取明确白名单，所有旧文件只读；不触碰raw/N1、Assessment逐点、Test数值、checkpoint内容。不生成新预测。图均Validation候选，PNG300dpi+PDF。唯一target不等于独立样本；不做p值/CI/bootstrap，sampleSD仅描述三Seed离散。

此前O-07“447独立target rows”已用新增O1附录勘误为“447个唯一目标时刻”，原O文件和数值保持不变。本阶段不解决gate独立增益、独立Test资格或SOH/RUL/物理问题，完成后停止。

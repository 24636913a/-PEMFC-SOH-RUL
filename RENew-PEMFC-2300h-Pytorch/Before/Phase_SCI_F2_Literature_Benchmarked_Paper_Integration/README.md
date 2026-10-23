# RealHyFC F2：文献对标、实验整合与中文工作稿

完成日期：2026-10-17。状态：`LITERATURE_BENCHMARKED_PAPER_INTEGRATION_COMPLETED`。

这是可追溯中文工作稿与实验材料，不是已完成独立验证的投稿终版。本阶段没有训练、模型前向、超参数搜索或Test读取；I2只复用已发布指标摘要。历史2896个受保护文件经SHA256核查保持不变。

## 从哪里开始

- [完整中文工作稿](manuscript/SCI_完整中文工作稿.md)：六章、声明及参考文献。
- [文献横向矩阵](literature/01_已发表PEMFC论文横向比较矩阵.md)：9篇×17字段，7篇全文、2篇摘要；未知细节不作否定证据。
- [修订六章大纲](outline/04_正式SCI论文大纲_文献对标修订版.md)：先于计算冻结，原“待复算”是设计时状态，最终完成情况见reports/01及02。
- [最终交接](reports/07_最终交接.md)、[证据完整性](reports/02_现有实验证据完整性报告.md)、[图表取舍](reports/03_论文图表审查报告.md)、[投稿风险](reports/04_论文创新性与审稿风险.md)、[最小后续方案](reports/05_尚缺实验及最小补充计划.md)。
- `outputs/evidence_index/CLAIM_EVIDENCE_MAP.csv`：逐项主张—证据—章节—图表映射；`FIGURE_INDEX.csv`含12组PNG/PDF及源数据与代码。
- `tables/`：10张核心CSV表；`outputs/experiment_tables/`、`condition_analysis/`、`baseline/`：全部逐点计算与统计；`outputs/audits/`：冻结、独立复算、测试、审查、SHA清单。

## 核心结果与边界

正式H三Seed F0/F1/F2的平均RMSE分别9.067563/9.155598/7.245400 mV，F2样本标准差1.100853 mV。新增零增量Persistence为195.078555 mV。同一205起点、4920查询、251个不同目标时间；重叠目标不能作为独立电池样本。Train历史三分位产生的九组中F2均值较低，但逐Seed并非普遍占优。

六个精选实例与正式三Seed分开。门控确实运行，P2独立增益不成立；六实例反事实只显示微小混合差异。保留Seed62终点和极端风险。现有Assessment已被复用，不是独立Test。本文不声称SOH、RUL、物理因果或跨堆泛化。

## 复现顺序

环境：Python3.11、numpy、pandas、matplotlib、Pillow、PyMuPDF；不需要torch或GPU。所有命令在本目录运行，建议`python -B`并设`PYTHONUTF8=1`、`PYTHONDONTWRITEBYTECODE=1`。

1. `scripts/verify_metadata.py`默认离线核对已保存官方元数据；**不使用--fetch覆盖本次冻结文献**。
2. `scripts/build_literature_matrix.py`使用本阶段保存证据提取JSON构建文献材料。已冻结版本应只核验，不必重新创建。
3. `scripts/prepare_analysis.py`是已执行的一次性冻结入口，**不要再次冻结**；`common.verified_config()`验证原协议与53项源身份。
4. `scripts/integrate_results.py`只复算既有合法缓存；`scripts/verify_results.py`独立逐指标复核；`scripts/build_tables.py`整理表。
5. `scripts/plot_paper_figures.py`绘制12组；`scripts/write_integration_reports.py`记录正文/补充取舍；`scripts/assemble_manuscript.py`拼接六章。
6. `python -B -m unittest discover -s tests -v`运行19项；独立文献/稿件审查由各自审查记录说明，不冒称真人同行评议。
7. `scripts/finalize_stage.py --verify-seal`验证交付SHA。无参数终检是提交前入口，不应用它覆盖已提交的原审计；本阶段不会修改历史目录。

最终Git SHA由Git提交和`outputs/GIT_RECEIPT.local.json`提供；回执故意不纳入自身提交，避免循环引用。安全推送失败时保留本地提交和错误，不改代理、不force push。参考PDF及文本只读并存于原位置/忽略runtime，不上传第三方全文。

# Phase O — F2论文全文对照与创新性审查

基准：`9723ed75910594155b4c0ae4888d9bbe8519ec11`；仓库 `24636913a/-PEMFC-SOH-RUL`；本阶段只新增文件，不修改历史研究。

## 结论与阅读入口

- 创新性：`INCREMENTAL_BUT_DEFENSIBLE`。
- 实验证据：`EVIDENCE_INCOMPLETE`。
- 写作：`READY_FOR_CONDITIONAL_DRAFT`，不等于已经达到正式投稿或独立确认性验证要求。

建议保留F2为主模型，定位为**明确未来信息边界的RealHyFC动态多步电压预测实证研究**。弱化gate为整体设计的一部分，不宣称其独立增益、机理解释或首创。不将电压预测改称已验证SOH/RUL或PINN。

先读 [最终决策](reports/O_10_FINAL_PUBLICATION_DECISION.md)，再读 [实验审计](reports/O_07_F2_EXPERIMENT_EVIDENCE_AUDIT.md) 与 [逐项重叠矩阵](reports/O_05_ARCHITECTURE_OVERLAP_MATRIX.md)。其余报告分别覆盖TFT、CNN/RNN、物理方法、信息集合、创新主张、审稿风险及论文大纲。全部新增文件见 [成果路径索引](O_DELIVERABLE_INDEX.md)。

## 证据范围

本地21份PDF核对实际首页；P1–P7全文阅读并视觉核对关键图表/公式；五份方法补充论文精读相关章节。中文“源代码”PDF不是Python源代码，也不作为另一篇独立论文。论文页码均为PDF从1开始的页序，而非默认期刊页码。未知实现必须标为UNKNOWN。

只整理历史已经保存的指标表、配置、源码和模型身份manifest。没有读取预测缓存、真实时序或checkpoint字节，没有重新计算任何性能。九个权重SHA来自历史manifest，不代表本轮重新认证权重内容。内部文本提取和页面图片存于gitignored `runtime/`，不作为原创投稿插图，也不上传原论文全文。

## 文件完整性检查

从仓库根目录运行：

```powershell
python -B RENew-PEMFC-2300h-Pytorch/Phase_O_F2_Paper_FullText_Novelty_Audit/scripts/run_o_document_audit.py --freeze
python -B RENew-PEMFC-2300h-Pytorch/Phase_O_F2_Paper_FullText_Novelty_Audit/scripts/run_o_document_audit.py
```

`--freeze`只用于首次建立静态来源快照；若快照存在拒绝覆盖。后续运行检查文档完整性、批准来源哈希和历史身份，不导入模型或执行旧runner。`outputs/O_READONLY_EXECUTION_AUDIT.json`记录本轮访问边界；不是OS级全进程监控，也不证明历史从未发生违规。

采用deep-research的一手证据分级、academic-paper-reviewer的内部批判框架及academic-paper的提纲模式；PDF技能要求实际渲染关键页面。提取工作并行完成，但不声称五位独立真人审稿人或完成正式同行评议。报告使用用户的中文协议式表达。证据CSV使用明确字段/顺序，不新增Excel工作簿。

完成后停止：无训练、无新预测、无Physics-F2开发、无论文投稿。

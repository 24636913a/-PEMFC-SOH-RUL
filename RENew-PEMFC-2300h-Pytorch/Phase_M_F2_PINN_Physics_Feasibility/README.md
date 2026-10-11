# Phase M — PEMFC 物理约束可行性与 PINN–F2 融合设计

基准：`82ee5ad35ada7dfe1e7a7c0add63de30fee3af43`。范围：仅 RealHyFC；文献、数学、字段语义与静态接口审查。

**科学结论：`NO_ADMISSIBLE_PINN_CONSTRAINT_YET`。唯一优先路线：P3，历史条件匹配的极化辅助监督；仅为有条件的补证方向，不授权训练。**

本阶段没有读取真实时序表、原始数据或 Test 数值，没有加载 checkpoint、运行模型、训练、搜索、部署物理损失或生成 SOH/RUL 标签。字段字典是 schema-only 元数据，不是实际运行观测；checkpoint 仅校验文件字节身份。

入口：[最终决策](reports/M_FINAL_PHYSICS_RESEARCH_DECISION.md)、[候选矩阵](outputs/M_PHYSICS_CANDIDATE_MATRIX.csv)、[只读审计](outputs/M_READONLY_AUDIT.json)、[文献台账](evidence/M_PHYSICS_LITERATURE_LEDGER.md)。历史阶段及冻结 F2 均不改动。

## 复核方法

从本目录运行：

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/audit_m_readonly.py --verify
```

`--freeze-sources` 只用于本次建立元数据/源码/既有报告白名单的基准身份；重复复核必须使用 `--verify`，不能重新冻结变化后的旧来源。审计不 import 历史训练脚本或模型；只采用 Python 标准库。它不是运行实验的入口，也不是全系统 I/O 监视器。

未来数据补证需要另行明确授权；本阶段提交后停止。既有晚期区域不具独立确认性 Test 资格，Development Assessment 也已复用。不能以重训、换命名或重新划分恢复独立性。

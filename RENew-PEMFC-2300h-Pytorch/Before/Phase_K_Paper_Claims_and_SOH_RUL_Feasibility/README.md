# Phase K — 论文主张冻结与 SOH/RUL 可行性

基准：`469c06531a1d1c7fbd0a071f75204a983026cc8a`。本阶段仅做证据审查与研究规划，不训练、不推理、不读取 Test 数值、不改变任何历史阶段。

结论：冻结 F2 作为**动态多步电压预测**论文核心，采用**统一的工况感知双路径时序融合 TFT**表述；动态门控独立增益仍未证实。稀疏极化电压比值是 reference-condition performance-retention proxy，不是连续 SOH ground truth。完整 RUL 不成立：`EOL_NOT_ESTABLISHED`。

入口：[最终推荐](reports/K_FINAL_RESEARCH_RECOMMENDATION.md)、[证据台账](evidence/K_SOURCE_EVIDENCE_LEDGER.md)、[未来计划](reports/K_FUTURE_EXPERIMENT_PLAN.md)。

## 审计与复现

`python -B scripts/audit_k_readonly.py --freeze` 建立本阶段只读证据快照，第一次执行已完成；已存在时拒绝覆盖。

`python -B scripts/run_k_firewall_tests.py` 实际运行 unittest 并持久化测试证据；仅测试只读访问防线，不调用模型。

`python -B scripts/audit_k_readonly.py --verify` 验证白名单文档与九个 checkpoint、受限 Development 锚点前缀和 Git 变更范围。输出完整性 JSON 与来源清单；不重新计算科研指标。

N1 的混合锚点 CSV **不是全文来源**：只读 header + 最前 8 个 Development 记录（7 个 nominal + 1 个 recovery），禁止全文件哈希。混合 causal feature CSV 完全不打开。证据快照是在已知历史结果后建立的文献/审计记录，不是新的实验预注册。

方法：显式白名单、限定行读取、与基准 Git blob 比对、checkpoint 字节哈希及 Git diff。审计不冒充操作系统级全进程 I/O 监控，也不宣称之前所有历史阶段从未开放过 Test。本轮未访问 Test 内容；将来是否独立须按具体任务核验暴露历史。

仅新建本目录。完成提交与推送后停止；没有训练入口或 Test 开启授权。

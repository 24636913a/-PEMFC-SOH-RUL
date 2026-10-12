# Phase M2 — Train-only Raw Dwell Qualification

仅RealHyFC/P3。冻结F2主任务不变，不加载checkpoint或模型。先metadata freeze与synthetic tests，再对已证明全Train的9个源验证登记byte extent/SHA。跨1310及其后源一律0byte读取。

运行：`python scripts/freeze_m2.py`（首次冻结，不读raw）；`python scripts/run_m2.py`（identity先于parse）；`python -m unittest discover -s tests -v`。

协议.json语法为有效YAML，配套.sha256先于任何raw内容读取。M2_INTEGRITY_AUDIT保存最终文件hash。身份验证hash覆盖9个全Train文件；明确没有mixed-file whole hash，不把它称为完整Train覆盖。

先读reports/M2_FINAL_P3_RESEARCH_DECISION.md。保持片段和统计quiet不是物理稳态认证；主科学状态未通过，不启动Physics-informed F2。

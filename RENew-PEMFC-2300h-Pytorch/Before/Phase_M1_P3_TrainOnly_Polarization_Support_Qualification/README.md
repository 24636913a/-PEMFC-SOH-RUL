# Phase M1 — Train-only历史极化物理支持资格审查

范围：仅P3，F2 Train [0,2103)。冻结主F2与历史M不变。不训练/forward，不读Validation/Assessment/Test。

运行：`python scripts/freeze_m1_protocol.py`（首次、真实数据前，合成测试+协议hash）；`python scripts/run_m1_qualification.py`（独立新进程）；`python -m unittest discover -s tests -v`。

两个.yaml为JSON语法的有效YAML，内容由hash绑定。再运行审查仅核验基准祖先和冻结code/source哈希，不重写协议。任何访问/身份失败即fail closed。

结果先读reports/M1_FINAL_P3_RESEARCH_DECISION.md。必要bin数量不是dwell/pair数量，未识别的量保留null。当前科学状态为P3_QUASI_STEADY_OR_STATE_COMPARABILITY_UNRESOLVED，不允许physics-F2训练。

# 尚未执行的一次性评价状态机

设计状态：NOT_AUTHORIZED → SOURCE_ELIGIBILITY_REVIEWED → SCIENTIFIC_PROTOCOL_FROZEN → RUNTIME_SYNTHETIC_VERIFIED → READY_FOR_EXPLICIT_AUTHORIZATION → AUTHORIZED_ONE_SHOT → PREDICTIONS_FROZEN_BEFORE_TARGET_SCORING → TARGET_SCORING_COMPLETED → CLOSED；任何失败→BLOCKED。

**本轮实际状态 BLOCKED**：existing late-life区域不具确认性资格；未识别替代独立区域；mixed-file真实输入/目标隔离未验收。L合成 guard tests PASS只表示本轮禁区被拒绝，不等于真正九模型Test runner合成E2E完成。本轮production `transition` 禁止进入AUTHORIZED或scoring，连模拟正式解盲也拒绝。

## 未来单独授权才可实现

Stage P：绑定全9checkpoint、代码/协议/scaler/source身份、事前window key集合，读取合法past snapshots；原H Path A推理；保存predictions、checkpoint/环境、所有keys/时间/anchor、预测文件hash与commit。不得读取每origin未来目标；broker必须明确混合source实际物理访问。不能用质量/未来条件加入预测端。

Stage T：仅StageP完整身份/预测冻结后，读取对应合法targets；exact-key对齐/复算既有全部指标；missing/duplicate/mismatch/异常终止，不改model/filter/seed/metrics/window。所有9先独立评分、后等Seed汇总，不挑winner。

one-shot ledger须在第一次授权范围内数据I/O前持久化；失败保留stage/exception/target_unblinded/files/identity，无重试、删除或reset。恢复是否安全由另行审查授权，不默认重跑。原AHC-TV ledger独立、已消费，永不重用。

未来如果只做post-hoc holdout，必须单独协议/明确命名探索性，不等于本状态机确认性资格自动变PASS。本轮只计划、不授权任何stage，不新增Test结果。

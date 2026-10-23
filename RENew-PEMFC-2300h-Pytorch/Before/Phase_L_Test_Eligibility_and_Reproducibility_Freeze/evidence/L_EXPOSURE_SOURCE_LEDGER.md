# 暴露史来源台账

本台账 ID 对应 `manifests/L_SOURCE_MANIFEST.json` 中相对路径、当前字节 SHA256、base Git blob SHA、最后修改提交与读取范围。所有结论可回查源，不读取源中引用的 Test 数值文件。

| 来源 ID | 证据与强度 | 本轮允许读取 / 排除 |
| --- | --- | --- |
| N1_CODE / N1_META | 完成的 foundation 元数据 + aggregation 静态实现；强于文件名 | 时间范围、数量、Development mask 统计；排除所有 N1 mixed CSV |
| RUL_CODE / RUL_LEDGER | SUCCESS 台账明确 full_data_target_accessed、compromised；强 | metadata 台账及全数据处理/报告 writer 静态代码；排除 full-data tables、图、rich 最终报告/决策 |
| AHC_LEDGER / AHC_RUN / AHC_STATE | 实际已消费、Stage P 中止、target_unblinded=false；强 | metadata-only ledger + 原状态机代码；不执行工作流 |
| AHC_P / AHC_RAW | reader 在 alignment 前解析全 raw 文件，含十个 cell 电压；结合执行记录 | 静态列清单/调用顺序；不读 raw/极化内容 |
| PM_CODE / PM_REPORT | 已保存的支持失败诊断 + 原 reader 调用；强 | 只读不含晚期电压数值的 count/flag 报告；排除 interval CSV 和 rich summary |
| DRIFT_CODE / DRIFT_README | scope + table schema 显示包含缓存 P1S 电压；中 | 不含目标数值的 README、静态 source；不独立声称恢复完整 process I/O |
| ELTE_CODE / ELTE_LEDGER / ELTE_EXPOSURE | consumed/completed、两个 TARGET_READ event、unblinded；强 | metadata-only ledger/flags、静态 writer；排除 ELTE 实际预测、目标和误差报告 |
| CSHSE_INCIDENT | 保留的 invalidation 文档；中 | checksum 边界事故文字，不读取旧数值结果 |
| H_CONFIG / H_REPORT / IR_REPORT / I2_REPORT / J_REPORT / K_REPORT | 已保存 Development 配置、结果、限制；强但不证明其他历史进程 | 只读已明确为 Development 的文档；不重复任何模型 forward |
| J_ID / K_ID | 九 checkpoint 的两套历史身份记录；强 | hash/metadata；本轮 checkpoint 仅字节 SHA、不 torch.load |
| TFT / F_SCHEMA / F_CONV / F1 / F2 / E_CODE / E_SCALER / C_CODE / H_COMMON / H_TRAIN / IR_COMMON / I_METRIC / I2_COMMON / I2_RUN / F_CONFIG | 复现静态契约 | 静态 source/config、已冻结 Train scaler；禁止 import 或调用历史 runner |

跨阶段检索还生成 `L_CROSS_PROJECT_PATH_INVENTORY.csv`：仅 tracked 文件名，涉及 test、holdout、late-life、2000/2300、ledger、prefix/firewall/exposure 的路径均入清单。未审查 rich content 记 metadata-only，不解释为“从未访问”。目录名含 2300h 是项目名称，不能据此认定所有文件包含 Test。

## 证据纪律与反证

历史台账优先于一般 README；代码可读取 ≠ 一定执行。AHC 正式 runner 在首个 checkpoint 失败，不能把代码后面的 full raw checksum 或 2300 正式预测宣称为已执行。Postmortem 则有保存报告与两个诊断结果，支持独立后续 raw reader 访问。

N1 自动聚合可能是正常预处理，不单独证明选择泄漏；CSHSE byte hash 事故也不等于确认读取 Test 标签。另一方面，RUL90 的明确全数据目标描述性访问与路线推荐、AHC 后验诊断及 ELTE 后续目标反馈，不能由“没算动态 RMSE”消除。没有证明 direct F2 Test-label training；也无法证明同一 late-life 区域从未影响后续设计。

本轮无外部 web/API、无 mixed Git blob、无 Test 数值摘要；来源 IDs 是本地审计引用，不是文献引用。deep-research 的来源分级/反驳检查用于事实核验；未虚构多 agent 审查。中文结构参考用户提供的审计指令。AI 辅助编制，最终论文叙述由研究者负责核对。

# 历史 Test 暴露取证

结论：现有晚期区域存在明确历史数值暴露，不是 pristine。这里区分**动态电压预测目标相关观测**、**极化健康目标**与**工况/metadata**，不笼统写所有任务都发生 supervised label leakage。

## 时间线与变量

1. **N1**：全数据 durability CSV 被自动扫描、因果聚合；active mask 选择明确限定 Train+Validation。Test 结构/count 的使用与研究选择分开。准备阶段全数据处理不自动证明后验模型选择，但证明不能仅用文件未评估来推断从未读过（N1_CODE/N1_META）。
2. **RUL90**：2026-09-18 的 finalize-existing SUCCESS 台账明确：`full_data_target_accessed=true`、`test_target_information_accessed=true`、`test_seal_for_future_model_selection=compromised`、`test_prediction_metrics_computed=false`。此前 full-descriptive pass 已访问全部 durability 电压/电流与 full polarization；恢复阶段没有重新读 raw。静态代码生成完整 signal/health 描述及后续研究方向推荐。该事实跨越 1819.997738 h 边界，早于目前 F2 冻结。没有必要读取历史 Test 数值再证实（RUL_LEDGER/RUL_CODE）。
3. **AHC-TV**：base `50d0964ed7c8b288b6440529ca71b269eeec4bc4`；2026-10-06 16:11:19Z 启动、16:12:59Z 中止。ledger=`ABORTED_AFTER_TEST_EXECUTION_STARTED`，failure=`STAGE_P / TEST_ALIGNMENT_INCONCLUSIVE`；`target_unblinded=false`、`stage_p_frozen=false`。一次性权限已消费。Reader M 极化只读 DateTime；但 Stage P 先通过 `_read_raw_operating` 无 ageing cutoff 解析17个 raw 源，包含 U1–5/U7–11、实际电流/电流密度和工况，再做对齐。不得推断 Stage P abort 意味电压未读，也不宣称后置 full raw hash 或两个正式预测已完成（AHC_LEDGER/AHC_RUN/AHC_P/AHC_RAW）。
4. **目标盲 postmortem**：再次使用相同 full raw reader，恢复电压/retention/conditions；2000 支持0/3、2300支持2/3；后续 support-drift 推荐来自已见晚期诊断。这里 target-blind 指 nominal polarization Y，不指 raw voltage。报告当时“极化仍盲”是当时事实，不是现在仍盲（PM_CODE/PM_REPORT）。
5. **support drift**：代码 `_require_target_blind_table` `pd.read_csv(path)` 包含 P1S_voltage_V/retention，主要诊断条件支持。本轮仅静态 trace + README，未重读 cached interval 数值，也没有恢复独立 OS I/O 日志。因此此源对执行范围的证据低于 actual ledger；不能把 schema 当作每个分支均执行的证明（DRIFT_CODE/DRIFT_README）。
6. **ELTE**：base `184433d2f392c164920e39c61cc9ed106e04f4e6`；metadata ledger 明确 exploratory-only、预测先冻结、2000/2300 `TARGET_READ`、`targets_unblinded=true`、`completed=true`。后续项目 exposure flag 明确目标已打开。静态 writer 显示健康预测/误差输出，L 未读任何这些数值。原 AHC 阻断记录没有被删除或变成 completed（ELTE_LEDGER/ELTE_EXPOSURE/ELTE_CODE）。
7. **CSHSE checksum事故**：全 mixed feature file SHA 导致 DATA_ACCESS_BOUNDARY_BREACH；旧run INVALID_PROTOCOL_BREACH；无 Test-label access evidence。不能改称确认 Test-label leakage（CSHSE_INCIDENT）。

## 对 F2 的具体影响

F2 ΔV 用未来 average active-cell voltage 减 origin voltage。AHC/RUL90 raw 电压与 N1 aggregate 电压有同一底层测量的信息交集，但 P1S、极化 Y 与 ΔV 并非同一标签。**未计算 F2 RMSE 不等于动态目标相关观测保持盲态；极化揭盲也不等于 F2 曾用极化标签训练。**

现有记录显示项目晚期诊断影响健康路线/后续探索性协议；没有直接证据证明某个 F2 层、学习率或 Seed 按 late-life F2 性能选择。本阶段不指控该因果链。确认性资格要求可追溯且能排除后验信息影响；同一已描述/已诊断晚期区域不能满足这一证明要求。保留已冻结 Development 成绩，不新算 Test。

## 覆盖边界

完整路径 inventory 不等于所有历史数值文件审阅。rich Test 报告/数组排除；exact F2 query 与历史 raw 使用的逐键交集不生成；未经记录的人工查看、外部下载与进程级行为无法复原。未知不是无访问。主要资格判断不依赖猜测这些未知项，而依赖上面的明确 ledger/诊断反馈。

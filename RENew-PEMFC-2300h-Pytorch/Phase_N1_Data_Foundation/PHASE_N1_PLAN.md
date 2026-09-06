# Phase N1 — Data Foundation, Sampling Resolution and SOH Target Construction

中文名称：阶段 N1——数据基础、采样分辨率与 SOH 目标构建

## 1. Scope and stopping point

本阶段只完成数据基础、采样分辨率、稀疏 SOH anchor、时间尺度和无泄漏窗口可行性分析。不训练任何神经网络、统计预测模型、RUL 模型，不执行超参数搜索，不进入 Phase N2。

所有新脚本、配置、日志、表格、图片和报告只写入：

`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch`

旧项目只读：

`E:\沈农信电\ PEMFC-SOH+RUL\New-PEMFC-Pytorch`

原始 2300 h 数据只读：

`E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据`

已有数据审计只读：

`E:\沈农信电\ PEMFC-SOH+RUL\数据\《耐久性测试期间燃料电池堆的空间解析电化学表征》的基础数据评估`

## 2. Frozen time axes

- 主 ageing coordinate：`OpHrs2 [h]`。
- `DateTime` 只用于真实时间间隔、停机/启动、characterization interruption、文件边界和时间连续性审计。
- 异常的 `OpHrs [h]` 不参与主寿命轴、split、聚合索引、SOH 或窗口构建。
- 所有排序先按 `OpHrs2 [h]`，同 ageing coordinate 内按 `DateTime`；遇到倒序、重复时间戳或非递增 sequence 时 fail-closed。

## 3. Active-cell audit protocol

候选通道为 `U1 [V]` 至 `U11 [V]`。对每个通道计算 median、mean、std、p1、p99、zero ratio、negative ratio、flatline ratio，并与 `U.S.AveCell [V]` 比较。

- active-cell 决策只使用 Train + Validation 范围的数据；Test 只检查字段存在性和可读性，不用其分布改变通道选择。
- flatline ratio 定义为连续样本绝对变化不超过 `1e-6 V` 的比例，同时报告唯一值数量和整体 std，避免把稳定工况误判为坏通道。
- 与导出平均单体电压的一致性同时比较 11 通道均值和排除每个候选通道后的均值；报告 MAE、RMSE、median absolute error 和相关性。
- 现有审计候选 mask 为 `U1,U2,U3,U4,U5,U7,U8,U9,U10,U11`。
- 若重新审计支持该结果，冻结为 10-cell active mask；若 U6 结果与现有审计矛盾，停止 active-mask 子任务，不继续生成依赖该 mask 的聚合表，并报告 `ACTIVE CELL MASK UNRESOLVED`。

输出：`active_cell_channel_audit.csv`、`ACTIVE_CELL_MASK.md`。

## 4. Causal aggregation protocol

并行构建两套数据：30 min 和 1 h。bin 使用 `OpHrs2 [h]` 的固定物理网格：

- 30 min：`bin_start = floor(OpHrs2 / 0.5) * 0.5`，区间为 `[bin_start, bin_start + 0.5)`。
- 1 h：`bin_start = floor(OpHrs2 / 1.0) * 1.0`，区间为 `[bin_start, bin_start + 1.0)`。
- 任何 bin 只使用该区间内真实观测，不 centered smoothing、不 centered rolling、不 bfill、不 future fill、不用后一 bin 补前一 bin、不跨 characterization gap 插值。
- 每个 bin 保存 ageing start/end/center、首末 `DateTime`、source file count 和 source file names，便于追溯。

预期样本数由各原始 Loadcycle 文件的正 `DateTime` 间隔中位数估计；对约 1 s 文件，30 min 和 1 h 的 nominal expected counts 分别约为 1800 和 3600。若一个 bin 涉及多个文件，按各文件对该 bin 的贡献时长和各自 nominal interval 求和。`coverage_ratio = raw_sample_count / expected_sample_count`，不截断大于 1 的值；重复采样单独标记。

预先冻结的覆盖定义：

- `low_coverage = coverage_ratio < 0.80`。
- 窗口构建要求每个输入和目标 bin 的 `coverage_ratio >= 0.80`。
- 该 0.80 阈值是 Phase N1 的协议常数，不从 Test 调整；报告同时给出 0.70/0.90 的敏感性计数，但不据此访问 Test 性能。

原始 gap 判据为相邻 `DateTime` 差大于 `max(5 s, 5 × source-file median positive interval)`；另将文件间和 characterization 级长缺口单独记录。`gap-before`、`gap-after` 均基于实际时间戳，不以 `OpHrs2` 连续性替代。

零电流判据为 `abs(I.S.act [A]) <= 0.1 A`。startup 表示 bin 内或 bin 边界从零电流转为正负载；shutdown 表示从正负载转为零电流。事件只由按时间排序的真实观测构造。

load-transition count 优先使用实际电流变化，并用 Train 内稳态增量噪声确定阈值后冻结到 Validation/Test；operation-point transition count 直接统计相邻真实 `operation_point` 值变化。阈值和推导结果写入配置与报告。

## 5. Feature families

### Voltage state

只用冻结后的 active cells 计算：`cell_voltage_mean`、`cell_voltage_median`、`cell_voltage_std`、`cell_voltage_range`、`cell_voltage_min`、`cell_voltage_max`。保留 `exported_average_cell_voltage`，并报告其与 10-cell mean 的一致性。

### Load

保存实际电流 mean/std/min/max/peak-to-peak、实际电流密度 mean/std/peak-to-peak、load-transition count、operation-point transition count。set 与 actual 始终分列。

### Operating conditions

按真实字段保存温度、压力、RH 的 mean/std/range，flow 和 stoichiometry 的 mean/std。优先 actual；set 可以保留但明确 `_set_`/`_actual_`。

### Coverage and events

保存 raw/expected sample count、coverage ratio、zero-current fraction、startup/shutdown、gap-before/gap-after、number of operation points、operation-point transitions、source file provenance。

输出：`causal_features_30min.csv`、`causal_features_1h.csv`、`causal_feature_dictionary.csv`、`causal_feature_availability.csv`。

## 6. Frozen chronological split

设 `a_min = min(OpHrs2)`，`a_max = max(OpHrs2)`：

- `cut_60 = a_min + 0.60 × (a_max - a_min)`。
- `cut_80 = a_min + 0.80 × (a_max - a_min)`。
- Train：`[a_min, cut_60)`。
- Validation：`[cut_60, cut_80)`。
- Test：`[cut_80, a_max]`。

bin 只有在完整区间位于一个 split 时才分配；跨 split cut point 的 bin 从建模候选中排除并计入 boundary loss。所有 sequence 必须完全位于同一 split。

Test 从本阶段冻结。只允许确认文件可读、时间范围、样本数量、结构完整和 manifest/window count；不使用 Test 决定采样率、active mask、j*、SOH 公式、特征、lookback、horizon、模型、loss、超参数或 preprocessing。

输出：`split_manifest.csv`、`FROZEN_DATA_SPLIT.md`。

## 7. Polarization configuration compatibility

逐项核查 BoL 与 durability checkpoints 的 cell count、MEA、active area、温度、压力、RH、stoichiometry、polarization protocol、per-cell voltage 和 current-density 定义。证据优先级：原始 CSV metadata/列名 > 原始 ReadMe/论文 PDF > 文件名推断。

- Case A：若 per-cell polarization 电压和 current density 在配置及协议上可复核为等价，允许直接比较并写明证据。
- Case B：若 cell count、MEA、active area、工况或协议不能完全确认，不把 BoL 强制设为 SOH=1；改用最早同 durability 10-cell 配置且位于 Train 的合法 checkpoint，或只提出 configuration-aware normalization。
- 缺少证据时结论为 `CONDITIONAL`，不把“未发现差异”当作“已证明可比”。

输出：`polarization_configuration_compatibility.md`。

如需读取唯一原始论文 PDF，将使用 PDF 技能，并保持原文件只读。

## 8. j* candidate selection and sparse SOH anchors

候选点初始为 `0.8, 1.0, 1.2, 1.4, 1.6 A/cm²`；只在真实 coverage 支持时增减。固定 j* 的电压使用候选点附近真实观测；允许在同一 polarization sweep 内采用预先冻结的小邻域稳健汇总，不跨 checkpoint 插值，不用未来 checkpoint 补值。

每个 j* 计算 voltage median/mean/std、valid sample count、coverage、ageing Pearson/Spearman、monotonicity、non-increasing fraction、relative voltage loss、Air-Air-start sensitivity、距 500 mV 下限的裕度。Primary/secondary 选择只使用 Train + Validation checkpoints；落入 Test 的 2000/2300 h 等 checkpoint 只保留为未来验证 anchor，不进入选择分数、权重或公式。

综合规则按以下顺序审查：coverage、非极端低负载、500 mV 裕度、重复测量 std、monotonicity、Spearman、Air-Air-start sensitivity、Train/Validation 可观测性。若指标明显冲突，写 `SOH anchor unresolved` 并停止 SOH normalization。

SOH 只建立稀疏 diagnostic anchors，不线性插值成 30 min/1 h ground truth。

候选定义：

`SOH_V(t_i) = V(j*, t_i) / V(j*, t_ref)`

`HI(t_i) = 1 - [V(j*, t_ref) - V(j*, t_i)] / V(j*, t_ref)`

两式代数上完全相同；Phase N1 将明确记录这一事实，不伪装为两个独立 health definitions。未确认 EOL 前不构造 threshold-normalized SOH，不使用 `SOH_EOS = 96%`，不使用 `RUL = 2300 - t`。

输出：`polarization_soh_anchor_candidates.csv`、`SOH_ANCHOR_SELECTION.md`、`soh_reference_anchors.csv`。

## 9. Sampling-resolution decision

采样率比较只使用 Train + Validation，并按相同物理 lookback/horizon 比较。比较：总/有效 bins、legal windows、低覆盖/gap/startup/shutdown 比例、within-bin voltage/current noise、matched-condition residual noise、相似工况下 voltage-age Spearman/长期斜率/trendability/degradation amplitude/noise-to-degradation ratio、平均每 duty cycle 时间点、transition retention、operation-point resolution、current-profile preservation。

matched-condition 定义、分箱边界和任何标准化参数只由 Train 拟合并冻结后应用于 Validation；不读取 Test 趋势或性能做决定。

若 30 min legal windows 明显更多、coverage 和 matched degradation 不明显变差、noise/degradation 不显著恶化且动态信息明显更完整，主采样率选 30 min；若噪声明显压过 degradation，主采样率选 1 h；若证据冲突则两者保留到小规模 baseline diagnostic，但 Phase N1 不训练模型。

输出：`sampling_resolution_comparison.csv`、`SAMPLING_RESOLUTION_DECISION.md`。

## 10. Load-cycle periodicity and temporal dependency

只用 Train 数据。依据 operation point、current、current density、gap-aware ACF、谱峰/periodogram 和事件间隔确认 duty-cycle 周期，不由 `2300/200` 直接假定 11.5 h。

ACF 按连续 segment 计算，禁止跨 gap 拼接；分析 average-cell voltage、matched-condition voltage、current density、current、cell-voltage dispersion 和 operation point。operation point 同时报告数值编码 ACF 的局限和 categorical state-match periodicity。

ACF 和谱峰只作为物理时间尺度证据，不单独决定 lookback。

输出：`load_cycle_periodicity_analysis.csv`、`temporal_dependency_analysis.csv` 及对应图片。

## 11. Window feasibility and fail-closed audit

物理 lookback：12/24/48 h。horizon：4/8/12/24 h；48 h 只作附加计数。30 min 与 1 h 使用相同物理任务转换为各自 L/H。

合法 sequence `X[t-L+1:t] -> Y[t+1:t+H]` 必须：

1. 全部 bins 位于同一 split；
2. ageing timestamps 严格按采样网格递增；
3. 输入和目标 timestamps 严格递增；
4. 不跨 characterization/file/date-time gap；
5. 每个输入和目标 bin coverage 达标；
6. 不使用未来观测构造历史；
7. scaler/normalizer 如后续使用，只能在 Train 拟合；
8. 任何字段缺失、边界不明、排序异常或 segment id 不一致都拒绝窗口。

对 Train/Validation/Test 分别统计 legal windows、gap loss、split-boundary loss、coverage loss 和 effective sequence count。Test 只统计数量和结构，不查看 target 结果。

输出：`window_feasibility_summary.csv`、`phase_n1_window_audit.py` 和窗口审计日志。

## 12. Future forecasting task design

主任务：degradation increment forecasting。

`DeltaV(t+h) = V(t+h) - V(t)`，并以 `V_hat(t+h) = V(t) + DeltaV_hat(t+h)` 重建。

absolute voltage 只作辅助诊断；continuous SOH forecasting 在存在可靠连续/半监督 SOH 后才启用。Phase N1 不为训练线性插值稀疏 SOH anchors。

未来变量分类：past-known、origin-known、future-unknown、future-known-by-protocol、diagnostic-only、offline-only。严格 Scenario A 中未来 actual current、temperature、pressure、RH、flow、stoichiometry 全部 future-unknown。Scenario B 只允许由协议明确预定义的未来 set point，且必须与 Scenario A 分开报告。本阶段只定义，不执行模型实验。

## 13. Planned figures

Sampling：

- `sampling_30min_vs_1h_signal_comparison.png`
- `sampling_30min_vs_1h_noise_comparison.png`
- `sampling_30min_vs_1h_coverage.png`

Durability：

- `average_cell_voltage_vs_operating_hours.png`
- `matched_condition_voltage_vs_operating_hours.png`
- `cell_voltage_dispersion_vs_operating_hours.png`
- `load_profile_example_cycle.png`

Polarization/SOH：

- `polarization_curves_development_checkpoints.png`
- `candidate_jstar_voltage_vs_ageing.png`
- `candidate_jstar_monotonicity.png`
- `soh_reference_anchors.png`

Temporal：

- `load_cycle_periodicity.png`
- `voltage_acf.png`
- `current_density_acf.png`
- `window_counts_by_lookback_horizon.png`

用于选择的主图只显示 Train + Validation。若补充图显示 Test，图面必须标注 `TEST — visualization only, not used for selection`。

## 14. Reproducibility and verification

- 所有路径、阈值、字段映射和随机无关分析参数写入 `configs/phase_n1_config.yaml`。
- 脚本分阶段记录 source inventory、行数、列名、文件大小、修改时间和 SHA-256；不改写原始文件。
- CSV 使用 UTF-8 with BOM，保持一行一个 bin/anchor/指标，单位写入字段名或数据字典。
- 核心 CSV 完成后检查列、dtype、唯一键、时间递增、缺失率、split containment、Test 非选择性使用和可重跑一致性。
- 图片检查尺寸、轴单位、图例、Test 标识和空图；报告引用真实输出，不复制成聊天日志。

## 15. Phase N1 completion gate

仅在 active-cell mask、两套 causal tables、split freeze、BoL/configuration compatibility、j*、稀疏 SOH anchors、采样比较、ACF/duty-cycle、lookback/horizon feasibility、fail-closed window audit 和 `PHASE_N1_RESULTS.md` 全部完成且无关键矛盾时给 `PASS` / `GO`。

若 active-cell 或 SOH anchor unresolved、配置不可审计、窗口审计失败、Test 参与选择或核心产物缺失，则 Phase N1 为 `NO-GO`。无论结果如何，本阶段完成后立即停止，不进入 Phase N2。

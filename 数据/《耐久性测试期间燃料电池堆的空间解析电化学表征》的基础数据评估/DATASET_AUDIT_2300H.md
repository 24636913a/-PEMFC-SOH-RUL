# DATASET_AUDIT_2300H

审计对象：`E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据`  
输出目录：`E:\沈农信电\ PEMFC-SOH+RUL\数据\《耐久性测试期间燃料电池堆的空间解析电化学表征》的基础数据评估`  
审计日期：2026-09-05T20:52:53.590672  
审计脚本：`audit_2300h.py`（所有原始读取均为只读；CSV 使用 chunked read）

## 1. Executive conclusion

### 事实（来自原始文件）

- 递归发现 `525` 个文件：CSV `36`、TXT `488`、PDF `1`。
- Loadcycle CSV 共 `17` 个，合计观测行 `8,604,654`。
    - 按 Loadcycle 内部 `DateTime` 的实际最小/最大值，观测时间范围为 `2025-01-27 12:09:18.005000` 至 `2025-07-26 12:06:23.005000`，历时约 `4319.95 h`；其中包含停机/维护/characterization 之间的长时间缺口，因此不是连续运行小时数。
- Loadcycle 中存在 `OpHrs [h]` 和 `OpHrs2 [h]`，全局范围分别约为 `107.1-8.572e+04 h` 和 `72.79-2257 h`；`OpHrs2` 观测跨度约 `2184.01 h`，不是从 0 h 开始。`OpHrs2` 全文件无 reset 且终点约 2256.8 h，和论文约 2300 operating hours 相符，故推荐把 `OpHrs2 [h]` 作为 ageing coordinate；`OpHrs [h]` 在 `Loadcycle_101-112.csv` 出现原始文本 `85,710.13`-`85,715.13` 后回到约 `1,547.50`，应标记为局部导出/记录异常，不能作为主寿命时间轴。
- 原始字段包含 10-cell 配置字段 `Number of cells`，含 `U1`-`U11` 共 `11` 个 cell voltage 通道；`U.S.AveCell [V]`、`MAX(U.S.MaxCell [V])`、`MIN(U.S.MinCell [V])` 和 `MIN(U.S [V])` 等 stack/cell-level voltage fields 存在。`MIN(U.S [V])` 是 stack-voltage candidate，但其 `MIN` 统计语义需以设备导出定义核实，报告不把它无条件当作专用 stack voltage。
- 论文 PDF 文本提到 durability test 结束于 200 cycles、约 2300 operating hours，并把 500 mV 作为 durability test 的 lower voltage limit；这说明存在一个运行限制/判据，但不等于原始数据中已经证明发生了 run-to-failure EOL。

### 审计判断

**建议将该数据集作为论文主数据集用于“运行条件感知的 SOH / degradation forecasting + 机理诊断”方向，但暂不作为严格 RUL 的主标签数据集。** 原因是它具有长时、重载、单体/空间/EIS/极化/敏感性等多模态诊断优势，同时当前原始文件审计未把 2300 h 证明为明确 failure endpoint；严格 RUL 需要独立、可复核的 EOL 定义。

**PHM2014 FC2 建议保留为外部验证集；FC1 建议作为敏感性/补充验证而不是主数据集。** 该结论针对任务设计与外部验证价值，不读取或修改旧实验的 final-test 结果，也不使用其结果做模型选择。

## 2. Dataset structure

`dataset_file_inventory.csv` 是逐文件清单，包含真实编码、分隔符、chunked row count、列数/列名和可读性。

关键目录：

- `Loadcycle_data`：17 个真实连续负载记录 CSV。
- `EIS`：cell-level EIS、BoL/1150 h/2300 h 以及压力、温度、化学计量比变体的 TXT 谱文件。
- `spatiallyResolvedEIS`：BoL/1150 h/2300 h 的空间 EIS 和温度、湿度、压力、化学计量比变体。
- 根目录：10 个 `Polcurve*.csv`（含 BoL、300/600/850/1150/1500/1750/2000/2300 h 和 after-Air-Air-start）以及 8 个 `PartialSensitivityTest*.csv`。
- 本地论文 PDF 和 `ReadMe_EIS_measurements.txt` 用于协议/命名解释；原始数值仍以 CSV/TXT 为优先证据。

## 3. Loadcycle timeline

`loadcycle_file_summary.csv` 逐文件记录了实际时间戳、OpHrs 范围、采样间隔、重复/倒序/缺口、reset 与 operation point。`loadcycle_timeline_summary.csv` 同时包含 file、global、gap、operation_point 级 metadata。

- 文件名如 `Loadcycle_0-14.csv`、`Loadcycle_15-25.csv` 被拆成 `filename_range_start/end`，但报告不会把这个范围直接称为真实 cycle number，因为文件内部没有名为 `cycle`/`cycle_number` 的字段。文件名区间只是文件元数据；真实的 load-cycle 计数需依据协议或内部控制日志确认。
- 结论是“可以重建有明确缺口标记的 operating-hours timeline”，不是“无缝连续时间序列”：17 个文件之间有 `16` 个正向时间间隔，文件内部还有 `2` 个按 1.5×文件中位采样间隔识别的 gap；没有文件级 timestamp overlap，也没有 timestamp reversal。
- `operation_point` 是重复出现的整数状态码，不等于 durability cycle number。它适合做运行状态/事件特征候选；每个状态的实际时长和切换应以时间戳聚合。
- 若文件边界存在时间重叠或时间间隔，`loadcycle_timeline_summary.csv` 保留这些事实，并没有简单 concat 或去重覆盖原始记录。
- startup/shutdown 的原始可见代理是 `I.S.act [A] == 0`、operation point 变化以及时间缺口；“维护”只有在缺口/协议注释支持时才能确认，不能由单一零电流段猜测。

## 4. Variable dictionary

`variable_dictionary.csv` 对所有 CSV 列名做了规则化语义分组。无法从原始列名确认的字段保留为 `unknown`；`MIN(U.S [V])` 作为 stack voltage candidate 单独标注为“exact MIN semantics unresolved”。

重点变量：

- stack/cell voltage fields：`MIN(`U.S [V]`)`；单体列：`U1 [V], U2 [V], U3 [V], U4 [V], U5 [V], U6 [V], U7 [V], U8 [V], U9 [V], U10 [V], U11 [V]`；平均/最大/最小单体统计存在。
- current/current density：`I.S.set [A]`、`I.S.act [A]`、`j.S.act [A/cm2]` 等。
- operating conditions：温度、压力、RH、flow、stoichiometry 及 set/actual 版本均存在。
- `Number of cells` 的原始值应作为配置元数据，不应在不同 BoL short-stack 与 durability 10-cell stack 之间直接混用作同一健康标签。

## 5. Sampling and continuity

`sampling_rate_analysis.csv` 记录每个 Loadcycle 文件、全局 pooled statistics 和每小时样本量。采样周期以实际 `DateTime` 差分计算；没有从论文描述倒推采样率。`sampling_interval_histogram.png` 展示其分布。

推荐原则：

- 保留原始秒级数据用于事件/故障/启动研究，但不直接用于长时 degradation forecasting。
- 主建模表建议以 1 h causal aggregation 为默认粒度；同时对 0.5 h 与 2 h 做**仅在后续任务设计阶段**的稳定性敏感性分析。1 h 是在 2300 h 量级、每小时大量原始观测、又需保留 load-regime coverage 之间的折中。
- 聚合只使用当前及过去窗口内观测，不 centered smoothing、不 bfill；遇到 characterization gap 需保留 coverage flag，而不是插值填平。
- raw voltage 被 load 强烈调制，因此应优先做 operating-condition-aware features，而不是全量 `voltage ~ ageing` 单变量回归。

## 6. Signal quality

`signal_quality_summary.csv` 对核心连续字段给出 min/max/mean/std/p1/p5/p50/p95/p99、NaN、zero、constant、物理范围外比例和 abrupt jump 统计。物理范围外比例是审计标记，不是删除规则；没有任何异常点被删除。

特别关注：

- `U6 [V]` 被识别为低电压/非物理水平或 flatline 候选。当前原始统计中 `U6 [V]` 的中位数约为负小值，与其他单体约 0.70 V 的共同水平明显不一致，因此 active-cell dispersion 应排除它；剩余 10 个 cell-voltage 通道与 `Number of cells=10` 一致。不要把该通道强行当作健康退化信号。
- abrupt jump 与 load transition 不等价于传感器故障；报告把两者分开，建议后续使用 operation_point/current change 作为 event covariate。

## 7. Operating regime characteristics

`loadcycle_hourly_features.csv` 是仅由原始 Loadcycle 观测做出的每小时 metadata-level/causal aggregation，不是训练集。包含小时均值、std、min、max、观测行数、operation point 集合和状态切换计数。

有/无 future-known schedule 的区别必须严格处理：实际未来 current、temperature、pressure、RH、flow、stoichiometry 如果来自运行时传感器，在 prediction origin 不可默认知道。只有实验协议明确预定义且任务允许使用的 schedule 才能标记为 future-known by protocol。

## 8. Degradation evidence

`degradation_signal_summary.csv` 汇总了：全量 hourly trend、按相似 current-density bins 的 matched-load view、cell voltage dispersion、possible recovery/reset and structural-change flags。所有趋势图均以原始时间顺序和 causal aggregation 为基础，不做未来填充。

应这样解释：

- raw stack/cell voltage 的下降若伴随 current/load regime 改变，只能称为“运行条件混合信号”；matched-load 或标准化 polarization point 才能支持较强的 degradation 论断。
- 若存在 load relaxation 后的 voltage recovery，它更像 reversible hydration/mass-transport component，不能直接作为 irreversible SOH loss。
- 1500 h 与 2300 h 附近的 Air-Air-start 结果通过 `afterAirAirStart` 分层保留；after 文件与同龄正常文件的差异不能被当作纯 ageing effect。

## 9. Polarization degradation

`polarization_ageing_summary.csv` 对每个 polarization file 在固定 `j* = 0.2...2.0 A/cm2` 候选点附近提取了实际 j 分布、平均/中位电压、样本量和 after-Air-Air-start 标记。没有对原始 curve 做插值填充。

固定 j 点 `V(j*, t)` 是最干净的 health anchor 候选之一。建议优先选择：跨 checkpoint 可获得、重复测量方差低、没有越过 500 mV lower-limit 的区间；具体稳定点以该 CSV 的 `voltage_std`、coverage 和 after 标志复核。BoL short-stack 与 durability stack 的 cell count/configuration 差异必须在比较中分层。

## 10. EIS degradation

`eis_ageing_summary.csv` 对真实 TXT 表格解析了频率、real/imag 或 magnitude/phase、measurement current、potential、points、HFR proxy、low-frequency real-minus-HFR、arc-height、phase/magnitude 等 robust raw-spectrum features，并保留 `external_load_A / testbench_load_A / nominal_load_A`。没有做 equivalent-circuit fitting。

- 该数据具备跨 BoL/1150 h/2300 h 的 cell EIS 条件；真实格式在 CSV 中保留为 `real_imag` 或 `magnitude_phase`。
- HFR proxy 采用最高频端的 real impedance 均值；这不是拟合后的 ohmic resistance，应以 proxy 命名。
- 跨 ageing 的 Nyquist/feature 图只使用同一约 504.2 A nominal-load group，避免把 BoL/2300 h 中的约 10 A 与约 50 A cell-EIS 负载差异误当成 degradation。
- `spatial_eis_summary.csv` 保留原始 `location_index_code`，并且没有把代码强行映射成 inlet/outlet 顺序。只有论文/协议明确说明的 cell/zone orientation 才能在后续建模前使用；当前 spatial heatmap 的横/纵轴是原始代码/观测顺序，不是假定的物理流向。

## 11. Spatial degradation

空间 EIS 的每个 stage/condition 组在原始目录中有 `tst00...tst16` 共 17 个采集文件；论文协议说明 PAD4 为 16 个空间阻抗区并带 reference，因此 17 个文件应按“空间区 + reference”整体处理，但每个代码与物理 zone 的精确映射仍需按论文图 1/设备记录核对。审计已回答“有多少原始位置代码”和“每个 stage/condition 是否存在”，但不会从 `00...16` 单独推断 inlet→outlet。

空间 HI 候选是 HFR/arc/low-frequency proxy 的跨位置 dispersion、分位数、max-min 和边界对比；必须保留位置 availability 和 stage/configuration metadata。

## 12. Sensitivity tests

`sensitivity_ageing_summary.csv` 比较 BoL 与 2300 h 的 Temperature/Humidity/Pressure/Stoichiometry 部分敏感性。它用于说明 raw voltage 的 condition dependence，而不是直接做长期预测标签。Humidity 的 condition-value 状态：**unresolved: raw humidity sensitivity CSV has no RH-named column**；脚本保留了文件级电压统计，但没有伪造 RH 响应曲线。

优先作为 future covariates 保留的变量：实际 current/current density、temperature、pressure、RH、flow、stoichiometry 及 set/actual discrepancy。未来实际传感器值默认 `future-unknown`；作为 diagnostic-only/offline-only 的 EIS/spatial/polarization 不能伪装成在线 causal input。

## 13. Candidate SOH definitions

`health_indicator_candidates.csv` 给出三套候选，并以 monotonicity/trendability/prognosability、load robustness、availability、causal availability、online SOH/RUL suitability 做结构化评估：

- A. Voltage-only：固定条件下 `V(j*, t)` 或 matched-load average-cell voltage；在线可用性最高，但对 condition confounding 最敏感。
- B. Electrochemical：固定 j 的 polarization voltage + HFR proxy + arc/low-frequency proxy；退化解释力更强，但 EIS 低频、诊断频率和在线可用性有限。
- C. Spatial-electrochemical：B + active-cell dispersion + spatial EIS heterogeneity/gradient；最适合机制分析和论文 novelty，但不是高频在线 SOH，不能默认用于实时 RUL。

审计结论：推荐把 A 作为主在线 health target，把 B/C 作为低频 diagnostic/auxiliary HI；不要把 2300 h 自动编码成 SOH=0。

## 14. Degradation forecasting feasibility

这套数据适合定义 `history -> future degradation`，但预测目标必须拆分：

1. absolute voltage：容易受 future load/temperature 等调制，不适合作为严格 future-unknown 主任务，除非 future schedule 是 protocol-known。
2. residual voltage `U(t+h)-U(t)`：更贴近 degradation increment，建议作为主预测 target；仍需 condition-aware origin features 和 coverage flag。
3. origin-relative/normalized target：适合跨段比较、减弱 stack/configuration scale，但必须只使用 origin 及历史信息计算 normalization。

建议候选 horizon：H=4/8/12/24 h，必要时 48 h；候选输入 N=15/24 h 仅作为起点，不可直接沿用 PHM2014。先依据 hourly feature ACF/有效 degradation time-scale 和 characterisation gaps 选定；本轮未训练任何模型。

## 15. RUL feasibility

严格 RUL 结论：**当前数据不能直接标注严格 RUL。**

- 原始 Loadcycle 终点是测试结束/characterization end；没有在 CSV 中找到名为 EOL/failure/failure_time 的字段。
- 论文明确提到 500 mV lower voltage limit；这是可以调查的 threshold candidate，但仍需确认它是 durability test 的运行下限、单体/平均单体/哪一个 voltage 定义，以及是否真正触发了“failure”。
- 因此当前状态应写为 `NOT RUN-TO-FAILURE`，不能写 `RUL = 2300 - t`。
- 可行的后续路线是 threshold-based RUL：先用论文/DOE/heavy-duty durability criterion 复核阈值，再在每个运行条件和 stack configuration 下定义 event time；未确认之前只能报告 threshold candidates 与 uncertainty，不生成 RUL labels。

## 16. Comparison with PHM2014

| Dimension | New 2300 h dataset | IEEE PHM2014 FC1/FC2 | Audit implication |
|---|---|---|---|
| duration | 约 2300 operating h by paper; raw DateTime/OpHrs require explicit mapping | shorter benchmark durability records with established PHM task convention | new set stronger for long-horizon degradation context |
| load realism | heavy-duty relevant RealHyFC/PEMTASTIC-derived load cycle | benchmark-oriented load profile | new set more application-realistic for paper main story |
| sampling richness | raw Loadcycle high-rate signals plus characterization modalities | mainly time-series voltage/current/conditions convention | new set has richer causal audit surface |
| variables | current, cell voltages, temp, pressure, RH, flow, stoich, states | fewer variables depending on released channel | new set stronger for condition-aware HI |
| cell/spatial data | cell-level voltage and spatial EIS; 10-cell durability stack | benchmark FC1/FC2 primarily system signals | new set stronger for mechanistic novelty |
| polarization/EIS | regular polarization + EIS + spatial EIS + sensitivity | not comparable in the usual PHM2014 files | new set stronger for auxiliary HI / explanation |
| SOH | feasible with matched-condition or standardized voltage; configuration caveats | easy benchmark SOH convention | new set needs stricter target definition |
| forecasting | feasible for 4-48 h degradation increments after gap-aware aggregation | convenient benchmark history/future windows | new set supports more defensible condition-aware task |
| strict RUL | not proven run-to-failure; threshold unresolved | benchmark RUL labels depend on chosen convention and should be treated separately | do not claim new-set strict RUL yet |
| novelty | high | lower for main contribution, useful as external benchmark | new set is better main dataset if target is reformulated |
| leakage risk | high if future sensors/after tests/characterization leaks into origin | high if final-test result used for model selection | freeze causal feature policy and chronological split |
| preprocessing | high: multi-format, gaps, events, configurations | lower | publish audit and preprocessing manifest |

FC2 is recommended as external validation only after the main task is defined without using FC2 final-test outcomes for model or hyperparameter selection. FC1 is still useful for a robustness/sensitivity appendix or cross-dataset generalization check; it need not remain a co-equal main dataset.

## 17. Recommended main-task formulation

推荐主任务：

`causal history of hourly operating-condition-aware features -> future origin-relative / residual degradation at H=(4, 8, 12, 24) h`

主 target：固定标准化条件下的 average-cell voltage residual，或经历史-only condition normalization 的 `ΔV_h = V(t+h)-V(t)`；原始 absolute voltage 作为辅助 target/diagnostic，不作为唯一严格 future-unknown target。

主输入：过去 N 小时的 average/min/max cell voltage、active-cell dispersion、actual/set current/current density、temperature/pressure/RH/flow/stoich、operation point transition count、coverage/gap flags、calendar/OpHrs coordinate。EIS/polarization/spatial features 作为低频 diagnostic branch 或 checkpoint covariates，只有在预测起点真实可获得时才可进入在线任务。

推荐粒度：1 h causal aggregation，保留 0.5 h/2 h 作为后续敏感性候选。

推荐 N/H：先用 N=24 h、H=4/8/12/24 h 做任务定义；最终 N 由 hourly ACF/趋势尺度和 gap coverage 确认。N=15/H=24 只能作为与 PHM2014 的对照，不应作为默认。

## 18. Recommended train/val/test protocol

1. 固定 source manifest、列映射、active-cell mask、aggregation rules、threshold candidates。
2. 按时间顺序做 blocked split，例如早期 60% train、中段 20% validation、后期 20% test；characterization checkpoints 和 Air-Air-start 作为事件标注，不能跨 split 泄露。
3. 所有 normalization、matched-condition bin definitions 和 outlier thresholds 只在 train 或 protocol-defined physical constants 中确定；禁止用全程未来分布反推历史。
4. 若要模拟在线预测，未来 current/temperature/pressure/RH/flow/stoich 一律按 `future-unknown` 处理；另做一个明确命名的 `future-known-by-protocol` ablation，不能混写。
5. 以时间 block 为单位做 validation；不要随机打散相邻小时。
6. 测试只在最终冻结一次使用；不要访问 PHM2014 final-test 结果进行新模型选择。

推荐 baseline（仅作为后续计划，不在本轮训练）： persistence / linear trend / seasonal-naive by operating point / ridge with lag features / Gaussian process or state-space degradation baseline。LSTM、Transformer、M-LSTM、TFT、TCN、multi-scale、uncertainty、RUL model 与超参数搜索均按本轮停止条件明确禁止。

推荐指标：MAE/RMSE for residual voltage, bias/mean error, calibration only if later uncertainty is authorized, monotonicity/trendability/prognosability for HI, condition-stratified error, event/gap-stratified error, and cross-dataset FC2 external generalization. Strict RUL metrics are deferred until EOL is independently confirmed.

## 19. Leakage risks

`causal_feature_availability.csv` 对每个变量标记 `past-known`, `origin-known`, `future-known by protocol`, `future-unknown`, `diagnostic-only`, `offline-only`。

高风险点：

- 用未来真实 current/temperature/pressure/RH/flow/stoich 预测未来 absolute voltage；
- 用 after-Air-Air-start polarization/EIS 解释 Air-Air-start 之前的历史；
- 用 2300 h 终点或全局 quantiles 定义历史 SOH/EOL；
- 用 centered rolling median、bfill、随机行拆分；
- 把 `OpHrs`/`OpHrs2` 不加语义核验地当作同一年龄时间轴；
- 把 `operation_point` 或文件名 `0-14` 当作 cycle number；
- 把 `U6` 等 flatline/非物理通道纳入 cell dispersion；
- 以 8-cell BoL EIS/敏感性 stack 的绝对值与 10-cell durability stack 直接拼成单一 SOH。

## 20. Final Go / No-Go decision

- 作为长时 SOH / condition-aware degradation forecasting 主数据集：**GO with scope restrictions**。
- 作为 strict run-to-failure RUL 主数据集：**NO-GO until EOL/threshold is independently resolved**。
- 替代 PHM2014 成为论文主数据集：**建议替代，前提是论文主问题改为可复现的 SOH/未来退化预测，而不是直接声称 strict RUL**。
- PHM2014 FC2 external validation：**GO**，但只在主任务、预处理、模型选择全部冻结后使用。
- FC1：**保留为可选补充/稳健性验证，不必作为主数据集**。

### Recommended pipeline

`Raw Loadcycle -> timestamp/cycle audit -> causal aggregation -> operating-condition-aware representation -> SOH/degradation target -> chronological split -> forecasting -> uncertainty (future work) -> threshold-based RUL (only after EOL confirmation)`

本轮在 `forecasting` 之后停止，没有训练、相对历史实验、多尺度、uncertainty、RUL 模型、多 seed 或超参数搜索。

## Generated figures

- `full_durability_voltage_trend.png`
- `current_load_profile.png`
- `missing_gap_timeline.png`
- `sampling_interval_histogram.png`
- `voltage_vs_ageing_colored_load.png`
- `polarization_curves_by_ageing.png`
- `voltage_degradation_fixed_current_density.png`
- `voltage_vs_current_density_by_ageing.png`
- `cell_voltage_dispersion_vs_ageing.png`
- `cell_voltage_range_vs_ageing.png`
- `eis_nyquist_comparison_across_ageing.png`
- `eis_feature_comparison_across_ageing.png`
- `selected_eis_features_vs_ageing.png`
- `spatial_eis_heatmap.png`
- `spatial_eis_position_comparison.png`
- `sensitivity_bol_vs_2300h.png`

## Paper protocol notes

page 1: spatially resolved degradation. A heavy-duty relevant load cycle was applied for more than 2000 h to a PEMFC
page 2: project-specific MEA served as the stack hardware using: a 10-cell stack / attributed to the applied load cycle. To enable a comparison between the / as the 10-cell configuration did not allow the highest stoichiometry to be / measurement segments for the local current density. In the 10-cell stack,
page 3: into 16 different zones based on the current density distribution map / applying a load cycle defined within the RealHyFC project and based on / the PEMTASTIC load cycle [33]. The cycle has a duration of approxi - / course of the load cycle (left) and the percentage distribution of the load
page 4: After every 25 load cycles, a partial characterization is performed, / diately following the load cycle is shown in Fig. 4. It is clear that the cell / irreversible losses. If the nominal polarization curves after air-air start / Fig. 1.Division of the CDDM measuring segments into a total of 16 different zones across the active area of the fuel cell. / Fig. 2.Current density profile (left) and percentage contribution of load points (right) of the load cycle from the RealHyFC project, which is based on the load cycle
page 5: BoL and after completing all 200 cycles are shown in Fig. 5.
page 6: The 2D images at BoL and after 2300 operating hours show a typical / the current density maximum reaches higher values after 2300 h. This is / Fig. 5.Partial sensitivity test at BoL and after 200 cycles (2300 h), a) temperature variation, b) humidity variation, c) pressure variation, d) stoichiometry variation.E. Fensterle et al.                                                                                                                                                                                                                                International  Journal  of Hydrogen  Energy  250 (2026)  156060
page 7: Representation of the current density distribution at BoL and after 2300 h as a 2D graph, as well as the evolution of the current density over several aging time points / 2300 h after air/air start (on the right). At each fixed voltage value, the local current density is averaged over all the CDDM segments of the zone. Arrows represent
page 8: maximum spread occurs both at BoL and after 2300 h between zones 4 / is 0.94 A cm2 at BoL and increases to 1.03 A cm2 after 2300 h. An / and after 2300 h at 1.8 A cm2. The variations characterize the indi- / because the cell voltage fell below 500 mV, which was defined as the / lower voltage limit for the durability test. / creases both at BoL and after 2300 h when increasing humidity, pres-
page 9: 2300 h are shown in Fig. 10.
page 10: sensitivity test at BoL and after 2300 h. The BoL data comes from the 8- / resolved measurements. The results after 2300 h are from the 10-cell / gram of the pressure variation is shown in Fig. 11 a). After 2300 h the / increasing pressure, both at BoL and after 2300 h. This is confirmed by / mass transport resistance, top: BoL, middle: after 1150 h and bottom: after 2300 h. / 2300 h 14.91 3.96 20.40 5.78 35.76 33.79E. Fensterle et al.                                                                                                                                                                                                                                International  Journal  of Hydrogen  Energy  250 (2026)  156060
page 11: further reducing the dehydration of the membrane. After 2300 h, the / show minimal resistances at 70 •C for both BoL and after 2300 h (Fig. 11 / resistance from 70 •C to 75 •C are observed. After 2300 h of operation, / decreases both at BoL and after 2300 h with increasing stoichiometry / On the overall, between BoL and after 2300 h, the shapes of the plots / respective reference cell (cell 5 of 8-cell short stack for BoL and cell 6 of 10-cell short stack for 2300 h) for the pressure (b), temperature (c) and stoichiometry (d)
page 12: After 2300 h, the membrane resistance at the cathode inlet decreases / the greatest homogeneity across the cathode flow at 2300 h. The ORR

## Reproducibility notes

- 原始数据目录只读；脚本只在输出目录生成文件。
- Loadcycle CSV 通过 pandas chunksize=100,000 读取；没有生成 GB 级 concat 副本。
- 聚合使用实际 timestamp/observed values；未做 centered smoothing、bfill 或未来观测填充。
- 所有 `unknown`/`unresolved` 都保留在结果，不以文件名或论文叙述替代原始字段证据。

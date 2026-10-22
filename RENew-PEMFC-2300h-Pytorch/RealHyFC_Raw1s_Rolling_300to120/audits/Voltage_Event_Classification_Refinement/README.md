# 电压突变观测关联细化

**状态：代码已编写，Codex仅做静态检查，尚未运行分类。没有本轮新比例、新原因结论或实际运行通过声明。**

正式入口：[`Refine_Voltage_Event_Classification_Interactive.py`](Refine_Voltage_Event_Classification_Interactive.py)。

本程序读取Startup_Precursor_Audit的已完成run：`20261022T071957_960946Z_b1590612`。沿用全部15,578个事件、原始分区、行号和原锚点；不再次检测事件、不加载F2、不训练、不读原始Loadcycle CSV、不改旧文件。旧全量审计不需要重跑。

## 已定位的旧判据问题

旧`startup_audit_core/events.py::classify`把事件前后180秒放在一起统计：D只要求其中出现一次加载状态下的≥5 A电流阶跃；E只要求冷却液温度极差>0.1℃或入口压力极差>1 kPag。它们登记的是宽窗口内“发生过变化”，缺少普通时刻背景，也无法判断变化发生在电压突变前还是后。

原已保存汇总（直接引用，不是本轮重算）：

| 分区 | 事件数 | 旧D电流标签命中 | 旧E温压标签命中 |
|---|---:|---:|---:|
| TRAIN | 10,897 | 10,893（99.963%） | 10,897（100%） |
| VALIDATION | 1,568 | 1,566（99.872%） | 1,568（100%） |
| EXPLORATORY_HOLDOUT | 3,113 | 3,111（99.936%） | 3,113（100%） |

这些比例不能解释为99%事件已确定由电流造成，或100%事件由温压造成。旧标签保持原样，新定义另存。

## VS Code手动运行

1. 打开本目录，选择原项目Python 3.11环境，需要NumPy、pandas、IPython（不需要torch/CUDA）。
2. 打开正式入口，点击 **Run All Cells**，或者按顺序运行`# %%`单元。会显示实际处理进度和完成后的真实统计表。
3. `SOURCE_RUN = None`默认使用原冻结run。如果缓存只存在于ZIP中，先把原完整归档恢复到另一个空目录，再把`SOURCE_RUN`设置为该run的完整路径。不要覆盖原run；缺失缓存时程序会停止，不会重建原始数据。
4. 如果交互窗口找不到目录，将入口顶部`STAGE_DIR`填成本目录绝对路径。
5. 每次运行生成新的`runs/<UTC时间_随机后缀>/`，不续写旧run。中断后保留partial与`ERROR.txt`，再次手动运行会创建另一个目录。

也可在本目录终端运行（这会真正执行分析，应由用户启动）：

```powershell
python -B .\Refine_Voltage_Event_Classification_Interactive.py
```

完成后独立核验（把路径换成本次实际输出目录）：

```powershell
python -B .\Verify_Refinement_Outputs.py '.\runs\本次实际目录'
python -B -m unittest discover -s tests -v
```

以上分析、核验和语义测试**均未由Codex执行**。只有用户执行后才会出现新CSV、真实分类比例和`INDEPENDENT_VERIFICATION.json`。

## 新判据：变化分三个层次、三个阶段

相对原冻结锚点的阶段：

| 阶段 | 时间范围 | 差分范围 |
|---|---|---|
| before | [-120,0)秒 | 两端都严格早于锚点 |
| instant | 上一条→锚点 | 同segment、同分区，实际采样间隔0.95～1.05秒 |
| after | (0,120]秒 | 两端都严格晚于锚点 |

不能跨segment、分区或时间缺口拼接。前/后至少90%时间及样本覆盖，变量至少95%有限数值且首尾有限，否则NA。时间使用原始日历纳秒时钟，不把行偏移当固定秒数。

九变量：实际电流、设定电流、内部设定电流、冷却液入口/出口温度、阳极/阴极入口压力、氢气流量、空气流量。按缓存列名定位；单位分别为A、℃、kPag和nl/min。缺列不补零，也不回读原始CSV。

- **数值变化**：有效相邻采样差分是否非零，只是数值记录发生变化。
- **Train幅度代理变化**：相邻变化是否超过Train普通对照非零瞬时增量P10的3倍。至少20个正增量才定标。该下限只是数值变化幅度代理，**不是仪器精度、物理阈值或真实动作判据**。
- **相对背景突出变化**：瞬间检查绝对阶跃；前/后分别检查最大绝对阶跃、首尾绝对变化、极差。逐变量逐阶段与Train普通对照P99比较（大于，非大于等于），下限兼顾上述幅度代理。至少100个有效对照；常量或零背景不强行设零阈值，记NA。

普通对照：固定60秒日历网格，匹配±0.1秒内最近实测点，排除本分区全部冻结事件段及两侧121秒，要求完整且不跨分区的上下文。Train对照不依据后两个分区的事件身份筛选。按时间均匀秩限额Train20,000、Validation/后留各5,000；不根据电压、其他变量幅度或结果筛选。事件与对照采用完全相同的特征提取器。固定网格可能与周期性工况存在相位偏差，结果只能描述这一注册背景，不能视为随机干预。

**先拟合并保存Train规则和SHA256，再分析事件与其他分区；不根据Validation/后留调阈值。** P99不是显著性阈值；多个指标与变量取OR后，实际普通时刻命中率可能高于1%，因此同时输出普通对照命中率和有效分母。不得看到比例后改阈值以追求更好看的分类。

具体公式、预算和停止条件见[`configs/PROTOCOL.json`](configs/PROTOCOL.json)。这不是优化搜索，只运行这一套预先声明的规则。

## 多标签与未知

按CURRENT、COMMAND、COOLANT_TEMP、INLET_PRESSURE、GAS_FLOW五家族分阶段聚合。任一已支持子项阳性即可记阳性；只有所有子项已支持且阴性才记阴性；否则NA。

- `BEFORE_*`、`INSTANT_*`、`AFTER_*`只表示观测时序关联。
- `AFTER_ONLY_*_NOT_A_PRECURSOR`只有前/瞬间均已支持且阴性、后为阳性才成立。
- `no_specific_association_in_observed_fields`表示没有超背景变化，不等于没有物理原因。
- `physical_cause_status`对所有事件都保持`UNDETERMINED`。不把观测关联升级成已识别的物理因果。

正向/混合事件锚点沿用第一正向触发；纯负向事件沿用第一触发。锚点前可能已包含本事件更早的负向触发，或另一个事件。输出保留这些污染标志、邻近事件数与锚点偏移；**before不自动等于预测前兆**。局部重叠事件也不当作独立重复试验。

## 65秒与7秒保留解释

旧长停留组的命令重加载中位间隔约65秒，短停留组约7秒；实际持续加载定义下长组可为66秒。程序直接引用原`restart_lag_distribution.csv`，不重新计算或改写。这些量从**电压恢复之后**计时到再次加载，不能反过来当作恢复前的预测信号。

## 本轮用户运行后输出

新目录`runs/<本次run>/`（默认Git忽略，避免未运行结果和大文件混入代码提交）：

- `event_observational_classification.csv`：全部15,578行，多标签、原事件身份、NA及污染标志。
- `event_phase_features.csv`、`variable_associations.csv`：420,606行变量×阶段证据；幅度、单位、数值变化时刻、支持状态及判据。
- `association_rates_by_phase.csv`、`ordinary_control_rates.csv`：事件及背景分阶段比例，均保留总数/有效数/NA数。
- `legacy_comparison.csv`：旧D/E与新家族分阶段定义对照，不是旧物理原因的重新认证。
- `train_control_index.csv`、其他分区对照索引、对应批处理特征表、`control_selection.json`。
- `frozen_train_rules.json`、`threshold_freeze.json`、协议、来源SHA、代码快照、输出SHA清单。
- `source_protection_before.json`、`source_protection_after.json`、`REPORT.md`、`run.log`和状态清单。

不复制2.2GB事件上下文CSV或整个原始审计。复用只读mmap，单次仅取局部片段，按128锚点批量写出；全缓存仅检查既有manifest身份及大小，不再次全文件散列。缺失/损坏或身份不符时停止，不重建、不删事件、不换锚点。运行时仍依赖原缓存未被外部修改；完整缓存SHA校验并未在本阶段重新执行。

## 静态验收边界

见[`docs/STATIC_REVIEW.md`](docs/STATIC_REVIEW.md)及`STATIC_SYNTAX.json`。静态语法和人工路径检查不等于真实运行验收。旧F2目录13个文件与冻结审计8个小文件的SHA保护记录另存于本目录；不读取/干预任何训练进程。当前没有新分类比例，不能把实现完成写成数据分析完成。

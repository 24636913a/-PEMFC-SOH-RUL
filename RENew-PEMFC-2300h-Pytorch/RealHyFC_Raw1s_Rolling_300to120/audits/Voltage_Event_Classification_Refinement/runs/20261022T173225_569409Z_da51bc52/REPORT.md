# 电压突变观测关联细化（用户本次运行结果）

这是一项事后观测关联分析。全部物理原因仍为UNDETERMINED；没有模型训练或因果识别。

冻结事件保留 15578 / 15578；来源：`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\RealHyFC_Raw1s_Rolling_300to120\audits\Startup_Precursor_Audit\runs\20261022T071957_960946Z_b1590612`。

分区事件数：TRAIN=10897, VALIDATION=1568, EXPLORATORY_HOLDOUT=3113

没有突出关联（全部已支持检验均阴性）：0；无法完整判定：1。二者不能混为一类。

## 旧比例为何接近100%

旧D在整个±180秒中只要出现一次≥5 A加载电流跳变就成立；旧E用整个窗口温压极差与0.1℃/1 kPag比较。混合事件前后和缺少普通时刻对照，使常见变化也被登记为关联。

新版将[-120,0)秒、上一采样→锚点、(0,120]秒分开。数值变化、Train数值变化幅度代理及超出Train普通背景分位数分别统计。普通对照排除冻结事件段及两侧121秒。阈值文件先冻结，再应用Validation/后留与事件。

## 如何看输出

- `event_observational_classification.csv`：每个冻结事件恰好一行，多标签、NA与未知均保留。
- `variable_associations.csv`：9变量×3阶段逐项证据、幅度、时间与缺失原因。
- `association_rates_by_phase.csv`与`ordinary_control_rates.csv`：对照事件与普通时刻的命中率和有效分母。
- `legacy_comparison.csv`：旧D/E及新分阶段家族标签；两套定义不同，不能声称类别直接等价。
- `frozen_train_rules.json`：实际Train控制样本数、q99、数值幅度代理及NA理由。

即使新比例仍很高，也不调阈值追求更低的比例。单变量多指标OR、变量家族OR和重叠上下文均会影响比例；q99不是事件检验的显著性，也不是家族1%假阳性保证。

## 时间解释

所有阶段相对原冻结anchor_row：正向/混合事件沿用第一正触发，纯负向沿用第一触发。因此“before”只表示锚点以前，可能已包含同段更早触发或邻近事件，相关污染字段已保存。它不能自动升级成独立启动前兆。AFTER_ONLY标签更不能用于恢复前预测。

原长停留段约65秒、短停留段约7秒为电压恢复后到再次加载的间隔；以下直接引用原汇总，不重算、不用作当前前兆阈值。actual_load定义下长停留中位数可为66秒，应保留原差异。

partition,stop_group,lag_type,comparable_spells,observed_lags,missing_lags,minimum,median,p05,p95,maximum,near65_fraction
EXPLORATORY_HOLDOUT,long_ge_3600s,command_lag_seconds,39,39,0,65.0,65.0,65.0,65.10090000000001,66.0,1.0
EXPLORATORY_HOLDOUT,long_ge_3600s,actual_load_lag_seconds,39,39,0,65.0,66.0,65.0,66.0,66.0,1.0
EXPLORATORY_HOLDOUT,long_ge_3600s,actual_nonzero_lag_seconds,39,39,0,48.0,65.0,65.0,66.0,66.0,0.9743589743589745
EXPLORATORY_HOLDOUT,short_lt_3600s,command_lag_seconds,118,118,0,5.999,7.0,6.0,7.001,8.0,0.0
EXPLORATORY_HOLDOUT,short_lt_3600s,actual_load_lag_seconds,118,118,0,6.999,7.000500000000001,7.0,8.0,8.001,0.0
EXPLORATORY_HOLDOUT,short_lt_3600s,actual_nonzero_lag_seconds,118,118,0,6.0,7.0,6.99985,8.0,8.0,0.0
TRAIN,long_ge_3600s,command_lag_seconds,138,138,0,64.999,65.0,65.0,65.99915,66.0,1.0
TRAIN,long_ge_3600s,actual_load_lag_seconds,138,138,0,64.999,66.0,65.0,66.0,66.001,1.0
TRAIN,long_ge_3600s,actual_nonzero_lag_seconds,138,138,0,16.0,65.0,64.99985,66.0,66.001,0.9855072463768116
TRAIN,short_lt_3600s,command_lag_seconds,422,422,0,5.999,7.0,6.0,7.0,8.0,0.0
TRAIN,short_lt_3600s,actual_load_lag_seconds,422,422,0,6.999,7.0,7.0,8.0,8.001,0.0
TRAIN,short_lt_3600s,actual_nonzero_lag_seconds,422,422,0,1.0,7.0,6.999,8.0,8.001,0.0
VALIDATION,long_ge_3600s,command_lag_seconds,20,20,0,65.0,65.0,65.0,66.0,66.0,1.0
VALIDATION,long_ge_3600s,actual_load_lag_seconds,20,20,0,65.0,66.0,65.0,66.00005,66.001,1.0
VALIDATION,long_ge_3600s,actual_nonzero_lag_seconds,20,20,0,47.0,65.0,64.1,66.0,66.0,0.95
VALIDATION,short_lt_3600s,command_lag_seconds,62,62,0,6.0,7.0,6.999,7.0,8.0,0.0
VALIDATION,short_lt_3600s,actual_load_lag_seconds,62,62,0,6.999,8.0,7.0,8.0,8.001,0.0
VALIDATION,short_lt_3600s,actual_nonzero_lag_seconds,62,62,0,6.999,7.0,7.0,8.0,8.0,0.0


## 限制

没有阀门/控制器硬件状态，因此温度、压力和气流变化时间不等于硬件动作时间；负载电流与电压瞬时反向仅为符号相容。共同控制、恢复和传感器噪声均可能产生关联。普通对照不是与事件同工况的随机干预，也不能用于推断因果风险比。

缓存只读、按局部片段取值；只校验既有manifest和文件大小，没有再次扫描/散列整个原始数据或缓存。独立核验和语义测试须由用户运行。本报告仅在当前运行的完整性核验完成后生成。

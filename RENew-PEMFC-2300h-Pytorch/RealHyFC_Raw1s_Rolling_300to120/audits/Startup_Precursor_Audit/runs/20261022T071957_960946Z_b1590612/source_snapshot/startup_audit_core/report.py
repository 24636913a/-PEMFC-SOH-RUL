"""Generate a Chinese report from completed actual CSVs only, never prior guesses."""
import numpy as np
import pandas as pd
from .data_io import dump


def read_table(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def markdown(frame):
    if not len(frame):
        return '无符合条件的记录；NA，不生成虚构成绩。'
    values = frame.fillna('NA').astype(str)
    return '\n'.join(['| '+' | '.join(values.columns)+' |', '| '+' | '.join(['---']*len(values.columns))+' |',
                      *['| '+' | '.join(row)+' |' for row in values.to_numpy()]])


def generate(run, source, events, classification, snapshots, timeline, metrics, cfg):
    thermal = []
    rows = snapshots.loc[(snapshots.seconds_before == 1) & snapshots.low_zero_recovery_proxy.fillna(False).astype(bool)]
    if len(rows):
        for (part, signal), group in rows.groupby(['partition', 'signal']):
            delta = group.past120_delta.dropna().to_numpy()
            thermal.append(dict(partition=part, signal=signal, supported_recovery_events=len(delta),
                                positive_120s_delta_events=int(np.sum(delta > 0)),
                                positive_fraction=float(np.mean(delta > 0)) if len(delta) else None,
                                median_delta=float(np.median(delta)) if len(delta) else None))
    pd.DataFrame(thermal).to_csv(run/'recovery_direction_summary.csv', index=False)
    lags = read_table(run/'restart_lag_distribution.csv')
    ordinary = read_table(run/'ordinary_signal_controls.csv')
    known = timeline.loc[timeline.recovery_DateTime.astype(str).str.startswith(('2025-07-02', '2025-07-16'))] if 'recovery_DateTime' in timeline else pd.DataFrame()
    threshold = __import__('json').loads((run/'voltage_threshold.json').read_text(encoding='utf-8'))
    cycle = __import__('json').loads((run/'cycle_mapping_identity.json').read_text(encoding='utf-8'))
    summary = dict(status='COMPLETED_FIXED_RULE_DESCRIPTIVE_AUDIT', raw_files=len(source['source_files']), raw_rows=source['rows'],
                   operational_voltage_episodes=len(events), positive_trigger_seconds=int(events.positive_triggers.sum()),
                   negative_trigger_seconds=int(events.negative_triggers.sum()),
                   low_zero_recovery_proxy_episodes=int(classification.C_low_zero_voltage_recovery_proxy.sum()),
                   voltage_threshold=threshold, recovery_direction_summary=thermal,
                   warning_metrics=metrics.to_dict('records'), ordinary_controls=ordinary.to_dict('records'),
                   restart_lag_summary=lags.to_dict('records'), known_dates_descriptive_checks=known.to_dict('records'),
                   cycle_mapping=cycle, holdout_used_for_thresholds=False, neural_network_training=False,
                   physical_cause_proven=False, future_plan_publication_time_known=False,
                   independent_events='operational merged episodes/spells; not independent cells/lifetimes',
                   hypotheses_from_two_known_events='do not promote to confirmed general rule')
    dump(run/'audit_summary.json', summary)
    val = metrics[(metrics.partition == 'VALIDATION') & (metrics.target_scope == 'LOW_ZERO_RECOVERY')]
    main = val[['rule', 'horizon_seconds', 'eligible_origins', 'precision', 'event_recall', 'exact_H_event_recall',
                'false_positive_rate', 'event_base_rate', 'effective_event_trials']]
    text = f'''# RealHyFC全量电压突变及启动前兆审计

这是用户实际执行后生成的CPU描述性审计，不是模型训练或F2改进实验。

## 通俗结果

读取17份原始文件，共{source['rows']:,}条记录。按冻结规则检出{len(events):,}个操作性电压事件段，其中低/零实流且零设定时的正向恢复代理事件{summary['low_zero_recovery_proxy_episodes']:,}个。正向触发秒{summary['positive_trigger_seconds']:,}、负向触发秒{summary['negative_trigger_seconds']:,}。触发秒、合并事件段、零命令停留段是三种不同统计单位，均不自动等于独立物理启动。

电压阈值{threshold['threshold_V']:.12g} V，模式{threshold['mode']}。只有Train用于本轮阈值拟合；三个时间分区分别报告。波动、电压回升和零电流都不能直接证明故障、退化恢复或设备完全关机。

### 温压是否提前上升：逐事件方向统计

下表检查事件前1秒记录相对其过去120秒的变化。它只能说明观测方向，不能单凭上升比例证明可靠预警；缺测、断点和不完整上下文保留为NA。

{markdown(pd.DataFrame(thermal))}

更早何时已经能观测到信号，见precursor_onsets.csv；窗口左端已报警者标为左删失，不能宣称找到了真正启动时刻。1/5/10/30/60/120秒前与事件时刻快照见precursor_features.csv。事件时刻的信息没有进入提前预警输入。

### Validation固定规则预警

{markdown(main)}

Precision=TP/(TP+FP)，误报率=FP/(FP+TN)。query_recall与event_recall分别针对重叠起点查询和操作性事件，不能互换。future-within-H命中可以只提前1秒；exact_H_event_recall单独检验约H秒前的真实历史，因此两列可能明显不同。三个提前量使用同一批事件，不能当作三倍独立实验。有效事件数小或NA时，不推断普遍性或统计显著性。

### 普通时刻是否也经常出现类似信号

普通对照排除全部正、负电压事件段前后120秒，逐秒统计同样的信号和Train阈值；ordinary_signal_controls.csv提供全部分区的普通/事件上下文数量、阈值超出比例、中位数与P95。precursor_evaluation.csv同时保留全部非事件起点的FP、TN与误报事件段数。若普通时刻也频繁越阈值，或Precision接近基础发生率，即不能仅靠该信号可靠识别恢复。组合规则是事先固定的OR与AND，没有在Validation/后留挑选最佳组合。

### 65秒是否重复

{markdown(lags)}

仅在起止命令均可观测、同一连续段、最大正向跃迁时实流接近零的停留段统计。65±{cfg['near_65_tolerance_seconds']}秒只是事先声明的描述性参考。长停留≥{cfg['long_stop_seconds']}秒与短停留分开；不符合者逐项保存restart_65_counterexamples.csv。零命令段中的最强跃迁锚点是回顾性选择，仅用于本节；预警标签采用每个事件段的首次正向触发，二者身份不能混用。首次非零实流与至少5A持续5秒的带载代理分别记录。孤立小脉冲不直接算硬件重启。缺失带载恢复记录可能是搜索上限/段落删失，不记为0秒。

### 周期与计划

历史RCR表中完整周期数为{cycle['historical_complete_cycles']}，只在其原有覆盖内对齐，不外推到后段，也没有用电压重建新周期。计划30/15/30/93分钟是用户提供的模板参考；模板相位仅由最早完整RCR周期的实际零命令段建立，不能当作控制器发布计划。时长匹配失败者仍保留候选差异，不硬配名义停机时长。event_cycle_mapping.csv与planned_vs_observed_timeline.csv区分模板、已记录命令、实际电流和电压恢复。周期中断、额外实验及相位漂移可能影响对应关系，不能从相似性单独判定。

**尚不能证明任意起点已知未来120秒的准确控制计划。**

## 科学边界及证据索引

- 第一层观测关联：event_classification.csv、event_transition_lags.csv、startup_timeline.csv、event_contexts.csv。温度/压力/气流值的变化时间不等于阀门或控制器硬件状态时间。
- 第二层提前价值：Train阈值frozen_precursor_rules.json，Validation固定规则检查与后留探索性结果precursor_evaluation.csv，逐事件命中身份precursor_event_detection.csv，普通对照ordinary_signal_controls.csv。没有模型选择、无新增神经网络、无F2输入改动。
- 第三层因果机制：**未被本审计证明**。记录到的气流/温压只能提供旁证；独立阀门状态、命令发布时间、控制器及测量回路状态等仍需补齐。不把未知事件强行归为噪声。
- 原始身份：source_manifest.json及cache内原始时间、文件ID、行号、全部十个有效单体、独立U6。断点不插值，不因误差或电压幅度删样本。图像对应的绘图CSV均保留。
- 后20%已经受历史研究接触，本次仅固定规则描述，不是独立盲测。重叠逐秒查询不独立；不做把秒数当事件数的统计检验。

所有结果仅依据本次CSV与原始数据，不保证温压一定有用，也没有将65秒写进F2预测逻辑。
'''
    (run/'AUDIT_REPORT.md').write_text(text, encoding='utf-8')
    return summary

"""P3 real-CSV figures and P4 bounded hypotheses. Never starts optimization."""
import math
import numpy as np
import pandas as pd
import core as c
from io_utils import csv,js,text,metrics,enrich


def diagnostics(run,frame):
    targets=[];tail=[];quarters=[];conditions=[];maxima=[];local=[]
    repeats=[]
    for kind in ['LSTM','F2']:
        for cohort in ['pilot','formal']:
            folder=run/'models'/cohort/f'{kind}_seed202'
            ledger=c.read_json(folder/'TRAINING_LEDGER.json')
            values=metrics(c.read_csv(folder/'predictions_VALIDATION.csv'))
            repeats.append(dict(model=kind,seed=202,cohort=cohort,best_epoch=ledger['best_epoch'],stop_epoch=ledger['stop_epoch'],
                initialization_sha256=ledger['initialization_sha256'],**values))
    csv(run/'formal_diagnostics/PILOT_FORMAL_202_VALIDATION_REPEATS.csv',repeats)
    repeat_differences=[]
    for kind in ['LSTM','F2']:
        folders=[run/'models'/cohort/f'{kind}_seed202' for cohort in ['pilot','formal']]
        histories=[c.read_csv(folder/'training_history.csv') for folder in folders]
        ledgers=[c.read_json(folder/'TRAINING_LEDGER.json') for folder in folders]
        n=min(map(len,histories));record=dict(model=kind,seed=202,common_epochs=n,
            initialization_equal=ledgers[0]['initialization_sha256']==ledgers[1]['initialization_sha256'],
            first_optimizer_update_equal=ledgers[0]['first_update_sha256']==ledgers[1]['first_update_sha256'])
        for column in ['train_standardized_MSE','validation_standardized_MSE','gradient_norm_min','gradient_norm_max','sampler_order_sha256','state_sha256','rng']:
            different=histories[0][column].iloc[:n].to_numpy()!=histories[1][column].iloc[:n].to_numpy()
            record['first_different_epoch_'+column]=int(np.flatnonzero(different)[0]+1) if different.any() else None
        repeat_differences.append(record)
    csv(run/'formal_diagnostics/PILOT_FORMAL_202_TRAJECTORY_CHECK.csv',repeat_differences)
    protocol=c.read_json(run/'FROZEN_PROTOCOL.json');bounds=protocol['diagnostic_time_quarter_boundaries']
    thresholds=protocol['diagnostic_thresholds_train_only']
    centres=c.read_csv(run/'P1_existing_F2/diagnostic_events.csv').target_row.astype(int).tolist()
    event_rows=[v for centre in centres for v in [centre-1,centre]]
    context=[]
    for centre in centres:
        block=frame.iloc[max(3074,centre-24):min(len(frame),centre+25)].copy()
        columns=['bin_end_ophrs2_h','first_datetime','last_datetime','cell_voltage_mean','cell_voltage_median',
            'cell_voltage_min','cell_voltage_max','actual_current_density_A_cm2_mean','zero_current_fraction',
            'startup_flag','shutdown_flag','bin_valid_for_windows','temporal_segment_id','raw_sample_count','coverage_ratio']
        values=block[columns].copy();values['target_row']=block.index;values['event_centre_row']=centre;context.append(values)
    csv(run/'formal_diagnostics/AGGREGATED_EVENT_CONTEXT.csv',pd.concat(context,ignore_index=True))
    for kind in ['LSTM','F2']:
        for seed in c.SUPPORTED_SEEDS:
            p=enrich(c.read_csv(run/f'models/formal/{kind}_seed{seed}/predictions_TEST.csv'),frame)
            sse=float(p.squared_error_mV2.sum());n=len(p);rank=p.sort_values('squared_error_mV2',ascending=False,kind='stable')
            groups=p.groupby(['target_row','target_age_h']).agg(query_count=('error_mV','size'),
                mean_squared_error_mV2=('squared_error_mV2','mean'),SSE_mV2=('squared_error_mV2','sum'),MAE_mV=('absolute_error_mV','mean')).reset_index()
            groups['model']=kind;groups['seed']=seed;groups['SSE_fraction']=groups.SSE_mV2/sse;targets.append(groups)
            tail.append(dict(model=kind,seed=seed,P95_mV=float(p.absolute_error_mV.quantile(.95)),P99_mV=float(p.absolute_error_mV.quantile(.99)),
                max_abs_mV=float(p.absolute_error_mV.max()),top1pct_SSE_fraction=float(rank.squared_error_mV2.iloc[:math.ceil(.01*n)].sum()/sse),
                top5pct_SSE_fraction=float(rank.squared_error_mV2.iloc[:math.ceil(.05*n)].sum()/sse),
                P1_two_event_four_targets_SSE_fraction=float(p.loc[p.target_row.isin(event_rows),'squared_error_mV2'].sum()/sse),
                target_time_equal_RMSE_mV=float(np.sqrt(groups.mean_squared_error_mV2.mean())),fixed_step24_RMSE_mV=metrics(p.loc[p.horizon_step==24])['RMSE_mV']))
            maxima.append(p.loc[[p.absolute_error_mV.idxmax()]].assign(model=kind))
            for i,(start,end) in enumerate(zip(bounds[:-1],bounds[1:]),1):
                quarters.append(dict(model=kind,seed=seed,quarter=i,start_row=start,end_row_exclusive=end,**metrics(p.loc[p.target_row.between(start,end-1)])))
            p['time_bin']=np.floor((p.target_age_h-frame.bin_end_ophrs2_h.iloc[3074])/10).astype(int)
            for index,g in p.groupby('time_bin'):local.append(dict(model=kind,seed=seed,time_bin=int(index),**metrics(g)))
            masks=dict(any_zero_current=p.zero_current_fraction>0,low_voltage=p.true_voltage_V<=thresholds['low_voltage_train_P05_V'],
                high_transition=p.load_transition_count>=thresholds['high_transition_train_P90'],
                large_voltage_jump=p.target_voltage_jump_V.abs()>=thresholds['voltage_jump_train_P95_V'])
            for name,mask in masks.items():
                for present in [False,True]:
                    g=p.loc[mask==present];conditions.append(dict(model=kind,seed=seed,condition=name,present=present,
                        **metrics(g),SSE_fraction=float(g.squared_error_mV2.sum()/sse)))
    csv(run/'formal_diagnostics/TARGET_ERRORS.csv',pd.concat(targets,ignore_index=True))
    csv(run/'formal_diagnostics/TAIL_METRICS.csv',tail);csv(run/'formal_diagnostics/MAX_ERROR_CASES.csv',pd.concat(maxima,ignore_index=True))
    csv(run/'formal_diagnostics/TIME_QUARTERS.csv',quarters);csv(run/'formal_diagnostics/CONDITIONS.csv',conditions)
    csv(run/'formal_diagnostics/LOCAL_10H.csv',local)
    js(run/'formal_diagnostics/DIAGNOSTIC_RULES.json',dict(time_quarters=bounds,Train_only_thresholds=thresholds,
        P1_adverse_centres=centres,P1_four_target_rows=event_rows,all_queries_retained=True,
        target_time_equal_metric_auxiliary=True,no_causal_claim=True))


def reports(run,table,means,paired):
    m=means.set_index('model');l=m.loc['LSTM'];f=m.loc['F2']
    gap=float(f.RMSE_mV_mean-l.RMSE_mV_mean);pct=gap/l.RMSE_mV_mean*100
    winner='完整双路径F2' if gap<0 else '普通LSTM' if gap>0 else '两者相同'
    win_rmse=int((paired.RMSE_mV_F2_minus_LSTM<0).sum());win_mae=int((paired.MAE_mV_F2_minus_LSTM<0).sum())
    rows=''.join(f'| {int(r.seed)} | {r.LSTM_RMSE_mV:.6f} | {r.F2_RMSE_mV:.6f} | {r.LSTM_MAE_mV:.6f} | {r.F2_MAE_mV:.6f} | {r.LSTM_R2:.9f} | {r.F2_R2:.9f} |\n' for r in paired.itertuples())
    mean_rows=''.join(f'| {r.model} | {r.RMSE_mV_mean:.6f} ± {r.RMSE_mV_sample_sd:.6f} | {r.MAE_mV_mean:.6f} ± {r.MAE_mV_sample_sd:.6f} | {r.R2_mean:.9f} ± {r.R2_sample_sd:.9f} |\n' for r in means.itertuples())
    tail=c.read_csv(run/'formal_diagnostics/TAIL_METRICS.csv').groupby('model').mean(numeric_only=True)
    part=table.groupby(['model','partition'])[['RMSE_mV','MAE_mV','R2']].mean()
    steps=c.read_csv(run/'tables/STEP_METRICS.csv').groupby(['model','step'])[['RMSE_mV','MAE_mV']].mean()
    raw_distribution=c.read_csv(run/'P1_existing_F2/partition_distributions.csv')
    raw_means=raw_distribution.pivot(index='variable',columns='partition',values='mean')
    baseline_tail_pct=c.read_json(run/'P1_existing_F2/DIAGNOSIS_SUMMARY.json')['mean_top1pct_SSE_fraction']*100
    event_context=c.read_csv(run/'formal_diagnostics/AGGREGATED_EVENT_CONTEXT.csv')
    event_centres=c.read_csv(run/'P1_existing_F2/diagnostic_events.csv').target_row.astype(int).tolist()
    centre_context=event_context.loc[event_context.target_row.isin(event_centres)].drop_duplicates('target_row')
    aggregation_evidence=centre_context[['target_row','bin_end_ophrs2_h','cell_voltage_mean','cell_voltage_median','cell_voltage_min','cell_voltage_max','zero_current_fraction','startup_flag','shutdown_flag','bin_valid_for_windows']].to_string(index=False)
    repeats=c.read_csv(run/'formal_diagnostics/PILOT_FORMAL_202_VALIDATION_REPEATS.csv')
    repeat_text=repeats[['model','cohort','best_epoch','stop_epoch','RMSE_mV','MAE_mV','R2']].to_string(index=False)
    repeat_trajectories=c.read_csv(run/'formal_diagnostics/PILOT_FORMAL_202_TRAJECTORY_CHECK.csv')
    repeat_trajectory_text=repeat_trajectories.to_string(index=False)
    f2_repeat=repeats.loc[repeats.model=='F2'].set_index('cohort')
    f2_trajectory=repeat_trajectories.loc[repeat_trajectories.model=='F2'].iloc[0]
    divergence=f2_trajectory.first_different_epoch_state_sha256
    divergence_text='共同Epoch内未观察到权重分歧' if pd.isna(divergence) else f'从第{int(divergence)}个Epoch末首次观察到权重分歧'
    repeat_summary=f"F2 Seed202的两次新训练初始化相同，{divergence_text}；pilot/formal最佳Epoch为{int(f2_repeat.loc['pilot','best_epoch'])}/{int(f2_repeat.loc['formal','best_epoch'])}，Validation RMSE为{f2_repeat.loc['pilot','RMSE_mV']:.6f}/{f2_repeat.loc['formal','RMSE_mV']:.6f} mV。采样顺序和RNG状态是否相同详见轨迹表。不能把一次更好结果替换进正式均值，也不能仅凭此锁定唯一CUDA算子。"
    maxima=c.read_csv(run/'formal_diagnostics/MAX_ERROR_CASES.csv');timing=[]
    for kind in ['LSTM','F2']:
        case=maxima.loc[(maxima.model==kind)&(maxima.seed==202)].iloc[0]
        points=c.read_csv(run/f'models/formal/{kind}_seed202/predictions_TEST.csv')
        local=points.loc[(points.origin_row==case.origin_row)&points.horizon_step.between(case.horizon_step-1,case.horizon_step+1)]
        real=local.loc[local.true_voltage_V.idxmin()];pred=local.loc[local.predicted_voltage_V.idxmin()]
        timing.append(dict(model=kind,seed=202,origin_row=int(case.origin_row),worst_target_row=int(case.target_row),
            true_local_min_step=int(real.horizon_step),predicted_local_min_step=int(pred.horizon_step),
            predicted_minus_true_min_lead_h=float((pred.horizon_step-real.horizon_step)*.5)))
    csv(run/'formal_diagnostics/WORST_QUERY_LOCAL_MINIMA.csv',timing)
    timing_text=pd.DataFrame(timing).to_string(index=False)
    timing_summary='；'.join(f"{r['model']}局部低谷位置差{r['predicted_minus_true_min_lead_h']:+.2f}h" for r in timing)
    pilot=c.read_json(run/'PILOT_GATE.json');formal=c.read_json(run/'ALL_FORMAL_WEIGHTS_FROZEN.json')
    text(run/'02_LSTM_vs_DualPath_Results.md',f'''# 真实LSTM与完整F2对照

以六种子等权平均RMSE衡量，本轮{winner}更低。F2减LSTM的平均RMSE差为{gap:.6f} mV（相对LSTM为{pct:+.2f}%）。F2的RMSE在{win_rmse}/6种子更低，MAE在{win_mae}/6种子更低。MAE和R²按各自结果解释，不能只用一个有利指标概括全部性能。

| Seed | LSTM RMSE(mV) | F2 RMSE(mV) | LSTM MAE(mV) | F2 MAE(mV) | LSTM R²(ΔV) | F2 R²(ΔV) |
|---|---:|---:|---:|---:|---:|---:|
{rows}
| 模型 | RMSE均值±样本SD(mV) | MAE均值±样本SD(mV) | R²均值±样本SD |
|---|---:|---:|---:|
{mean_rows}
均值是6个逐种子指标的等权平均，不是预测集成或拼接预测后的RMSE；样本SD使用ddof=1。配对差值全部列于tables/PAIRED_DIFFERENCES.csv，符号为F2减LSTM。未做12648次独立实验假设下的统计检验。

## 执行与工程验收

先完整训练Seed202两模型，pilot仅用Validation检查真实更新、样本身份和新checkpoint重放；随后12个正式实例全部重新初始化，没有复用pilot权重。共14次新训练：2次pilot+12次正式，pilot不混入正式均值。全部正式权重冻结后才计算本轮新后留成绩；P1历史缓存只用于之前的误差诊断。

本轮自然产生了Seed202的pilot与formal两次独立训练，可用其Validation结果检查重复波动（没有给pilot计算后留成绩，也不从两次中择优）：

```text
{repeat_text}
```

逐项首次分歧轮次（NA表示共同Epoch内未观察到差异）：

```text
{repeat_trajectory_text}
```

{repeat_summary}

该两次记录仅是工程诊断，不是预先设计的多次重复总体性能估计。相同模型/种子的初始化SHA及采样顺序另行严格核对；普通CUDA训练仍不承诺逐位一致。

所有模型共用同一组1750/218/527合法窗口、原始ΔV标签、Train-only scaler及固定Top10掩码。相同种子的共同Epoch采样顺序逐项一致，48步历史和24个目标全部严格对齐。原14通道中未选的4通道真实置零。

完整分区均值（mV；R²目标为原始ΔV）：

```text
{part.to_string()}
```

尾部与辅助口径（逐种子后再平均）：

```text
{tail[['P95_mV','P99_mV','max_abs_mV','top1pct_SSE_fraction','P1_two_event_four_targets_SSE_fraction','target_time_equal_RMSE_mV','fixed_step24_RMSE_mV']].to_string()}
```

## 最差查询所在完整窗口的时序诊断

图12同时展示Seed202两模型各自最差查询所在的24步窗口，并在每个相同起点叠加两个模型。图09的固定Step24曲线用于工况背景，不能代替全部步长的最大误差诊断。以下是各模型最差查询附近±1步内，预测局部最小值相对真实局部最小值的位置差；负号表示提前，正号表示延后：

```text
{timing_text}
```

这些例子可见半小时状态低谷误判。输入、标签与时间身份已严格核验，因此不能把它们直接说成CSV时序错位；仅两个事后案例也不足以宣称所有预测都有系统性偏移。没有按真实曲线移动预测、修正输出或重新计算较好指标。完整逐点绘图数据见figures/P3/12_paired_worst_query_windows_data.csv。

## 科学主张边界

LSTM是单层hidden64、最后隐藏状态接linear24，没有Attention/Transformer/TCN/外部Dropout；单层内部dropout=0。F2保留完整TFT、LSTM/CNN双路径与历史工况门控，原Dropout=.1。两者参数22040与332436，容量约差15.08倍。相同训练政策不等于控制了容量、正则化、输入归纳偏置或各模型专属调参预算。

本轮只比较完整F2与普通LSTM的总体表现，不能把差值全部归因于双路径或门控。F2沿用历史基础TFT选出的配置，LSTM使用同一配置，不宣称已完成LSTM专属最优超参搜索。六个种子来自历史F2筛选名单，当前平均不代表无偏随机初始化总体。

普通CUDA配置保留，同种子逐位重复性不保证；不能因一次复跑更好就替换正式实例。最后20%已受历史研究反馈影响，所有后留成绩均为探索性时间后留结果，不能写成独立盲测、外部泛化验证或健康/RUL验证。

每次训练均保留BEFORE_TRAINING、逐Epoch训练日志、最佳checkpoint、初始/首步/末步权重SHA、更新数与随机状态摘要。训练曲线是本次真实日志；Train显示批次平均MSE，Validation显示全查询MSE，正式Train/Validation分区指标另按同一全量口径计算。
''')
    # Two hypotheses only; NOT executed and never selected using new holdout feedback.
    text(run/'03_Prediction_Improvement_Plan.md',f'''# 最多两项后续假设：本轮不执行

本轮P1～P3已经完成，以下仅为后续独立登记的探索性假设。不能把建议写成已有改善，不重新搜索backbone，不改7:1:2、Top10、历史长度或预测长度，不输入未来真实工况。

## 假设1：Train事件敏感损失能否减少稀有状态大误差

P1旧F2最差1%查询贡献{baseline_tail_pct:.2f}%平方误差；本轮新F2对应平均为{tail.loc['F2','top1pct_SSE_fraction']*100:.2f}%，新LSTM为{tail.loc['LSTM','top1pct_SSE_fraction']*100:.2f}%。当前证据说明尾部值得研究，不能说明“转换次数越多误差越大”：P1高转换次数组的RMSE反而低于其他组。应使用Train真实电压跳变或零电流混合事件定义训练监督权重，不能直接把高转换次数当成困难样本。

最小方案：先仅Seed202、同一结构，普通MSE对比一个预先固定的事件加权MSE。事件阈值只从Train计算，权重例如固定2倍并归一化，不搜索多组系数。未来Train标签/工况可作为训练监督权重，但不能成为推理输入，也不能使用Validation/后留拟合阈值。新方案沿用同一Validation作工程检查，明确其重复使用的探索性身份。

预先停止条件：如果Validation整体RMSE比无权重基线恶化超过2%，或Validation事件组与P99均没有改善，就停止该假设，不靠后留挑权重。全部模型和成功标准冻结后才允许一次描述性后留回放；不重新声称独立Test。加权不能补足未知未来负载计划，可能只能改变误差权衡。

## 假设2：固定训练政策的重复稳定性是否影响最佳Epoch选择

先不增加网络组件。当前主程序已证明同种子普通CUDA轨迹可能分叉，严格模式又被原生插值反向阻断。完整训练日志、全量Train/Validation误差和早停记录应共同检查，不能仅凭Train batch-mean与Validation full-MSE曲线差距认定过拟合。

{repeat_summary}

本轮已存在的Seed202两次记录如下，优先使用这些证据，而不是直接追加训练：

```text
{repeat_text}
```

如果两次结果相同，则本轮没有新增早停不稳定的证据，该假设优先级降低，暂不追加重复训练；只有同条件下再次出现可追溯的分叉时，才考虑下面的固定预算验证。即使本次两次一致，也不升级为跨硬件/跨运行逐位确定性的证明。

最小方案：若以后获准，预先固定Seed202的3次独立重复，只观察Train/Validation最佳Epoch与误差范围，不挑最好的一次。使用独立repeat_id，保留全部初始化、RNG、梯度/权重摘要和相同采样顺序。严格确定性需先解决算子资格，不能擅自改算子或局部关闭严格模式来宣称问题已修复。

停止条件：完成预先登记的3次即停止，不因波动大继续追加直到得到好成绩。若Validation最佳成绩的波动已与模型差距同量级，论文应降低单次差值的主张强度，优先报告固定预算的重复范围。只有Train/Validation证据明确支持后，才另行设计完整六种子改进对照。

本轮没有执行上述实验，没有新增改进成绩。虽然Step1→24误差上升，盲目给长步长加权也可能牺牲短步长或放大不可知切换风险；为限制自由度，本轮不再叠加第三项时间权重假设。
''')
    text(run/'04_最终研究总结.md',f'''# 最终研究总结：通俗版

1. **7:1:2划分可以用于这次时间顺序实验。** 它保留过去训练、随后验证、最后评价的关系；没有随机划分或交叉验证。原始3843行保留，合法窗口1750/218/527。所有24个未来目标都在所属区间，scaler只看Train；检查没有发现行号错位或未来输入泄漏。但最后20%以前看过，不能叫独立盲测。
2. **以平均RMSE衡量，{winner}表现更低。** LSTM为{l.RMSE_mV_mean:.6f}±{l.RMSE_mV_sample_sd:.6f} mV，F2为{f.RMSE_mV_mean:.6f}±{f.RMSE_mV_sample_sd:.6f} mV；F2减LSTM为{gap:+.6f} mV（{pct:+.2f}%）。F2在{win_rmse}/6个种子的RMSE更低。MAE、R²及全部配对结果均保留，不能只挑最好种子。
3. **后期RMSE不是所有时刻都变差。** P1已有F2最差1%查询就贡献平均{baseline_tail_pct:.2f}%平方误差；约1874h和2150h存在突出的状态事件。零电流混合、低电压区及较大的真实电压跳变与大误差共同出现，误差随提前量总体增加。当前新模型的尾部和相同事件贡献另有真实CSV。高转换次数本身不总是更难，不能将相关性写成原因，也不能认定低电压就是不可逆衰退。
4. **没有发现原正式数据或预测代码的明显身份错误。** 本轮独立复算了旧CSV指标、时间、输入和标签；日历时钟、运行时钟、segment和边界检查通过。发现过本轮新诊断脚本的MAE单位转换遗漏，已在任何训练前留痕、修正并重新通过P0/P1，未改历史预测。原始F2的CUDA非逐位重复限制仍存在，没有伪装成已经解决。
5. **当前最值得研究的是稀有事件尾部与训练选择稳定性。** 先在Train/Validation限定预算检验事件加权是否值得做，再考虑固定重复次数的稳定性验证；本轮不执行额外优化。未知的未来负载状态不能靠改损失凭空获得。
6. **可进入论文的是本次真实协议、全部12个正式实例的同样本对照、各步长/尾部/时间段分析及真实图表。** 但它们仍应标作单RealHyFC、历史已暴露的探索性时间后留评价。F2与LSTM容量和正则化不同，差异不能直接归因为双路径；六种子也不是无偏随机总体。改进建议没有实验成绩，不能当作已验证贡献。

补充关键数据证据：Train/Validation/后留的电压均值分别为{raw_means.loc['cell_voltage_mean','TRAIN']:.6f}/{raw_means.loc['cell_voltage_mean','VALIDATION']:.6f}/{raw_means.loc['cell_voltage_mean','TEST']:.6f} V，电流密度均值和ΔV标准差也接近；没有证据把后期误差简单归为大幅全局边际分布漂移。两个不利事件的半小时均值约0.208 V，但中位数约0.003 V、零电流占比约73.51%，同时标记startup/shutdown；这些是包含启停的混合分箱，不能当成连续有载电压下降或不可逆损坏。记录本身满足当前有效窗口标记，所以没有按误差大小删除。

```text
{aggregation_evidence}
```

图12还保留两模型各自最差查询所在的完整同起点对照窗口；Seed202最差查询附近的{timing_summary}（负号提前，正号延后）。它们说明切换时序误判的具体表现，不证明全局固定延迟或数据错位。

这些证据更支持研究稀有状态事件、切换时序和模型外推稳定性。未来实际工况只用于事后核对，未知未来切换信息并未输入模型；这是一种较可信解释，尚不是唯一原因的因果证明。

重复性补充：{repeat_summary}

运行数量：2个Seed202完整pilot，另12个从头正式实例，共14次训练；pilot不混入论文正式均值。真实损失图、checkpoint、输入身份、逐点预测和重放审计全部保留。未经授权，没有删除、覆盖或移动任何旧代码和权重，没有重新选特征、扩展模型或启动其他研究阶段。
''')
    js(run/'RESULT_SUMMARY.json',dict(status='RESULTS_GENERATED',formal_models=12,pilot_models=2,RMSE_winner=winner,
        means=means.to_dict('records'),F2_minus_LSTM_RMSE_mV=gap,F2_relative_RMSE_change_percent=pct,
        F2_RMSE_wins=win_rmse,F2_MAE_wins=win_mae,optimization_experiments_executed=0))


def finalize(run,frame,windows,scaler,prepared,table,means,paired):
    import importlib.metadata as metadata
    packages={}
    for package in ['torch','pytorch-forecasting','lightning','pytorch-lightning','numpy','pandas','scipy','scikit-learn','matplotlib','ipykernel','nbformat']:
        try:packages[package]=metadata.version(package)
        except metadata.PackageNotFoundError:packages[package]=None
    js(run/'ENVIRONMENT_PACKAGES.json',dict(packages=packages,execution_environment=c.read_json(run/'FROZEN_PROTOCOL.json')['environment'],
        note='Recorded in the same execution kernel after training; no packages or device settings changed during this task'))
    print('P3: real comparison figures and complete error tables',flush=True)
    diagnostics(run,frame)
    from figures import comparison_figures,p1_report
    p1_report(run,c.read_json(run/'P1_existing_F2/DIAGNOSIS_SUMMARY.json'))
    comparison_figures(run,frame,windows)
    js(run/'P3_ACCEPTANCE.json',dict(status='PASS',formal_figures=12,all_figures_have_CSV=True,dpi=300,all_predictions_retained=True))
    print('P4: two bounded hypotheses, no extra training',flush=True)
    reports(run,table,means,paired)
    js(run/'P4_ACCEPTANCE.json',dict(status='PASS',proposed_hypotheses=2,optimization_trainings_executed=0))
    from verify_run import verify
    verify(run)
    text(run/'README.md','''# LSTM与完整F2：本次研究导航

正式成绩仅使用models/formal中的12个新实例；models/pilot的Seed202两个实例用于验收，不混入正式均值。P1_existing_F2为旧缓存只读诊断，不冒充新模型成绩。

报告顺序：00_实验协议与划分核验.md → 01_F2_Error_Diagnosis.md → 02_LSTM_vs_DualPath_Results.md → 03_Prediction_Improvement_Plan.md → 04_最终研究总结.md。

图像索引（全部300dpi PNG，并附PDF和绘图CSV）：

| 图 | 用途 |
|---|---|
| P3/01_six_seed_RMSE | 六种子RMSE公平比较 |
| P3/02_six_seed_R2 | 六种子原始ΔV R² |
| P3/03_mean_RMSE_SD | 等权均值和样本SD，非置信区间 |
| P3/04_horizon_RMSE | Step1～24误差变化 |
| P3/05_absolute_voltage_Step24 | 固定Step24绝对电压曲线，不拼接重叠窗口 |
| P3/06_scatter_all_queries | 全部12648查询/模型，保留极端点 |
| P3/07_scatter_actual_current | 目标实际电流密度着色，仅事后分析 |
| P3/08_error_over_time | 619唯一目标时间上的误差，未平滑 |
| P3/09_adverse_events_and_load | P1预先选定的不利事件与原始工况 |
| P3/10_training_loss | 本次Seed202两模型真实训练日志 |
| P3/11_fixed_origin_24step | 固定中位合法起点的单次24步预测 |
| P3/12_paired_worst_query_windows | Seed202两模型各自最差查询所在的完整同起点窗口，全部24步 |

P1图提供旧F2误差、固定10h局部RMSE、分区分布及不利案例。每个新模型实例另有本次实时training_history.png及CSV。

tables/存逐种子、均值±SD、配对差、全分区和逐步指标；formal_diagnostics/存尾部、事件贡献、条件组及原始最大误差记录。指标为ΔV；曲线和散点展示用原始起点电压恢复的绝对电压。

P5_AUDIT.json为独立输入/标签、早停、真实更新、checkpoint重放、指标和图像审计。全部后留结果属于探索性评价，不能称独立盲测；没有执行额外性能优化。
''')

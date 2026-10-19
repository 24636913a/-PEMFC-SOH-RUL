"""P0 protocol qualification and P1 cached F2 diagnosis; no training operations."""
from datetime import datetime,timezone
from pathlib import Path
import math
import os
import subprocess
import numpy as np
import pandas as pd
import torch
import core as c
from io_utils import js,csv,text,metrics,enrich


def prepare_run():
    print('P0: reading complete data and verifying frozen chronological windows',flush=True)
    run=c.HERE/'outputs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_Top10_LSTM_F2')
    run.mkdir(parents=True,exist_ok=False)
    frame,windows,scaler,prepared,excluded=c.load_data()
    identity=c.model_identity();environment=c.configure_environment()
    environment.update(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    c.require(not environment['deterministic_algorithms'],'UNEXPECTED_STRICT_MODE')
    old=c.read_csv(c.OLD_RUN/'MASK_APPLICATION_AUDIT.csv')
    partitions=[]
    for part,(start,end) in c.partitions(len(frame)).items():
        w=windows[part]
        expected=old.loc[(old.feature_mode=='top10')&(old.partition==part)].prepared_sha256.iloc[0]
        c.require(c.array_sha(prepared[part])==expected,'INPUT_IDENTITY_CHANGED:'+part)
        reference=c.read_csv(c.OLD_RUN/f'windows/{part}.csv')
        c.require(np.array_equal(reference.origin_row,np.repeat(w.origins,24)) and np.array_equal(reference.target_row,w.targets.ravel()),'WINDOW_IDENTITY_CHANGED')
        rows=np.unique(w.origins[:,None]+np.arange(-47,25)[None,:])
        # Check calendar continuity independently of the operating clock/segments.
        starts=pd.to_datetime(frame.first_datetime);ends=pd.to_datetime(frame.last_datetime)
        gaps=(starts-ends.shift()).dt.total_seconds().to_numpy()
        pairs=np.unique((w.origins[:,None]+np.arange(-46,25)[None,:]).ravel())
        c.require(np.isfinite(gaps[pairs]).all() and np.all((gaps[pairs]>=0)&(gaps[pairs]<=1.01)),'CALENDAR_DISCONTINUITY')
        partitions.append(dict(partition=part,start_row=start,end_row_exclusive=end,records=end-start,
            first_operating_h=float(frame.bin_end_ophrs2_h.iloc[start]),last_operating_h=float(frame.bin_end_ophrs2_h.iloc[end-1]),
            first_datetime=str(frame.first_datetime.iloc[start]),last_datetime=str(frame.last_datetime.iloc[end-1]),
            legal_origins=len(w.origins),queries=w.y.size,unique_targets=len(np.unique(w.targets)),
            origin_sha256=c.array_sha(w.origins),target_sha256=c.array_sha(w.targets),labels_sha256=c.array_sha(w.y),
            inputs_sha256=c.array_sha(prepared[part]),maximum_adjacent_calendar_gap_s=float(gaps[pairs].max())))
        csv(run/f'windows/{part}.csv',dict(origin_row=np.repeat(w.origins,24),target_row=w.targets.ravel(),horizon_step=np.tile(np.arange(1,25),len(w.origins))))
    train=frame.iloc[:2690];adjacent=train.temporal_segment_id.eq(train.temporal_segment_id.shift())&train.bin_end_ophrs2_h.diff().sub(.5).abs().lt(1e-6)
    thresholds=dict(low_voltage_train_P05_V=float(train.cell_voltage_mean.quantile(.05)),
        high_transition_train_P90=float(train.load_transition_count.quantile(.9)),
        voltage_jump_train_P95_V=float(train.cell_voltage_mean.diff().abs().loc[adjacent].quantile(.95)))
    protocol=dict(stage='P0',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        task_reference_remote_main='66b843ab1c53577d9fdb1071f27c5d8510b9d5c2; verified via GitHub connector at this task start, not a live check for later reruns',
        created_utc=datetime.now(timezone.utc).isoformat(),data_rows=3843,data_sha256=c.sha(c.SOURCE),
        official_source_sha256=c.sha(c.OFFICIAL),core_sha256=c.sha(c.HERE/'core.py'),model_identity=identity,
        partitions=partitions,features='fixed Top10',top10=c.MASKS['top10'],history=48,horizon=24,step_h=.5,
        ewma_alpha=.3,scaler_fit='Train legal raw histories only',target='raw DeltaV',R2_target='raw DeltaV',
        hyperparameters=c.HYPERPARAMETERS,max_epochs=450,patience=40,early_stop_min_delta=1e-12,
        seeds=list(c.SUPPORTED_SEEDS),models=['LSTM','F2'],pilot_seed=202,pilot_models=2,formal_models=12,
        no_top11=True,holdout_exploratory=True,no_new_loss_or_hyperparameter_search=True,
        LSTM=dict(layers=1,hidden_size=64,internal_dropout=0.,external_dropout=0.,head='linear 64 to24',parameters=22040),
        F2=dict(parameters=332436,dropout=.1,original_classes_exact=True),
        input_capacity_and_regularization_not_matched=True,seed_list_previously_F2_selected=True,
        diagnostic_time_quarter_boundaries=[3074,3266,3458,3650,3843],local_rmse_bin_operating_h=10,
        diagnostic_thresholds_train_only=thresholds,tail_top_query_fractions=[.01,.05],
        fixed_step_for_voltage_curves=24,example_seed=202,example_origin_rule='median legal holdout origin, not error selected',
        environment=environment,best_weights_frozen_before_new_formal_holdout_scores=True)
    js(run/'FROZEN_PROTOCOL.json',protocol);csv(run/'PARTITIONS.csv',partitions);csv(run/'EXCLUDED_WINDOWS.csv',excluded)
    js(run/'PREPROCESSOR.json',dict(mean=scaler.mean.tolist(),scale=scaler.scale.tolist(),target_scale_V=scaler.target_scale))
    text(run/'00_实验协议与划分核验.md',f'''# 单次7:1:2协议核验

3843条记录，时间顺序划分，不交叉验证。合法窗口1750/218/527，与原正式Top10输入、起点和目标身份严格一致。
Train[0,2690)、Validation[2690,3074)、后留[3074,3843)。边界依据记录数，不是实际运行小时或日历时间的固定百分比。

只用Train拟合scaler；固定Top10，不依据Validation或后留重新选特征；每窗口历史因果EWMA alpha=0.3。全部24个目标留在所属分区，历史可使用已发生的更早分区观测。本研究按滚动预测起点评价：到达每个起点时可以使用当时已经测得的后留历史观测；不是从分区边界一次性闭环预测整个后段。训练标签结束早于首Validation起点；Validation标签结束早于首后留起点。有效标记、segment、运行时钟、日历时钟、非有限值均已检查，剔除明细保留。

未来decoder只用确定相对时间坐标，未来真实工况不输入模型。工况数据用于P1事后诊断及散点着色，不能算在线可用信息。

划分适合时间顺序预测；同一电池单寿命且最后20%已受历史研究反馈，因此是探索性时间后留评价，不是独立盲测。重叠历史窗口可以合法共享已知观测；重叠目标查询不是独立实验。

普通LSTM为单层hidden64+linear24，无Attention，无外部Dropout，内部dropout=0；参数22040。完整F2参数332436，约15.08倍，Dropout=.1。训练政策和输入相同，但容量、归纳偏置、正则化和配置来源不同，不能声称控制全部架构因素或将差异归因为双路径模块。

先完整训练Seed202两模型用于工程验收，再从头正式训练六种子两模型12次，pilot不混入正式均值。全部权重冻结后再计算本轮正式后留结果。保持普通CUDA设置，不强开已知被插值反向阻断的严格模式；同种子逐位重复不保证。六种子是历史F2筛选名单，平均值不是随机初始化总体的无偏性能。

固定诊断：后留原始行按四个连续等行数区间[3074,3266,3458,3650,3843]分段；局部RMSE按10h实际运行小时分箱；保留全部查询及固定Step24指标。所有阈值在Train确定，不根据后留误差调阈值。

P0证据：FROZEN_PROTOCOL.json、PARTITIONS.csv、PREPROCESSOR.json、windows/、EXCLUDED_WINDOWS.csv。此后不启动任何额外优化实验。
''')
    js(run/'P0_ACCEPTANCE.json',dict(status='PASS',sample_and_preprocessing_identity_exact=True,calendar_and_operating_time_checks=True,no_training_started=True))
    print('P0 PASS: same1750/218/527 windows; LSTM22040 parameters, F2332436 parameters',flush=True)
    return run,frame,windows,scaler,prepared,protocol


def diagnose_existing(run,frame,windows,protocol):
    c.require(c.read_json(run/'P0_ACCEPTANCE.json')['status']=='PASS','P0_REQUIRED')
    print('P1: read-only F2 six-seed error diagnosis, no training',flush=True)
    out=run/'P1_existing_F2';out.mkdir(exist_ok=False)
    all_points=[];whole=[];step_rows=[];quarter=[];local=[];tails=[];targets=[];conditions=[];extreme=[];overlap=[]
    boundaries=protocol['diagnostic_time_quarter_boundaries'];threshold=protocol['diagnostic_thresholds_train_only']
    reference=c.read_csv(c.OLD_RUN/'HOLDOUT_PER_SEED_METRICS.csv')
    for seed in c.SUPPORTED_SEEDS:
        raw=c.read_csv(c.OLD_RUN/f'seed{seed}/top10/predictions_TEST.csv');w=windows['TEST']
        c.require(np.array_equal(raw.origin_row,np.repeat(w.origins,24)) and np.array_equal(raw.target_row,w.targets.ravel()) and
            np.array_equal(raw.y_true_delta_V,w.y.ravel().astype('float64')),'OLD_PREDICTION_IDENTITY')
        computed=c.calculate_metrics(raw);old=reference.loc[(reference.seed==seed)&(reference.feature_mode=='top10')].iloc[0]
        c.require(all(computed[k]==old[k] for k in ['RMSE_V','MAE_V','R2']),'OLD_METRIC_MISMATCH')
        p=enrich(raw,frame);all_points.append(p);whole.append(dict(seed=seed,**metrics(p)))
        for h,g in p.groupby('horizon_step'):step_rows.append(dict(seed=seed,step=int(h),**metrics(g)))
        for index,(start,end) in enumerate(zip(boundaries[:-1],boundaries[1:]),1):
            g=p.loc[(p.target_row>=start)&(p.target_row<end)];quarter.append(dict(seed=seed,quarter=index,start_row=start,end_row_exclusive=end,**metrics(g)))
        base=float(frame.bin_end_ophrs2_h.iloc[3074]);p['time_bin']=np.floor((p.target_age_h-base)/10).astype(int)
        for index,g in p.groupby('time_bin'):local.append(dict(seed=seed,time_bin=int(index),start_h=base+10*int(index),end_h=base+10*(int(index)+1),**metrics(g)))
        sse=float(p.squared_error_mV2.sum());sorted_points=p.sort_values('squared_error_mV2',ascending=False,kind='stable')
        tails.append(dict(seed=seed,P95_abs_error_mV=float(p.absolute_error_mV.quantile(.95)),P99_abs_error_mV=float(p.absolute_error_mV.quantile(.99)),
            max_abs_error_mV=float(p.absolute_error_mV.max()),top1pct_queries=math.ceil(.01*len(p)),
            top1pct_SSE_fraction=float(sorted_points.squared_error_mV2.iloc[:math.ceil(.01*len(p))].sum()/sse),
            top5pct_SSE_fraction=float(sorted_points.squared_error_mV2.iloc[:math.ceil(.05*len(p))].sum()/sse)))
        extreme.append(sorted_points.head(20))
        groups=p.groupby(['target_row','target_age_h']).agg(query_count=('error_mV','size'),mean_squared_error_mV2=('squared_error_mV2','mean'),
            sum_squared_error_mV2=('squared_error_mV2','sum'),mean_absolute_error_mV=('absolute_error_mV','mean')).reset_index()
        groups['seed']=seed;groups['SSE_fraction']=groups.sum_squared_error_mV2/sse;targets.append(groups)
        overlap.append(dict(seed=seed,queries=len(p),origins=p.origin_row.nunique(),unique_targets=p.target_row.nunique(),
            target_query_multiplicity_min=int(groups.query_count.min()),target_query_multiplicity_max=int(groups.query_count.max()),
            query_weighted_RMSE_mV=metrics(p)['RMSE_mV'],target_time_equal_RMSE_mV=float(np.sqrt(groups.mean_squared_error_mV2.mean())),
            fixed_step24_RMSE_mV=metrics(p.loc[p.horizon_step==24])['RMSE_mV']))
        masks={'low_voltage':p.true_voltage_V<=threshold['low_voltage_train_P05_V'],
            'any_zero_current':p.zero_current_fraction>0,'high_transition':p.load_transition_count>=threshold['high_transition_train_P90'],
            'large_voltage_jump':p.target_voltage_jump_V.abs()>=threshold['voltage_jump_train_P95_V']}
        for name,mask in masks.items():
            for flag in [False,True]:
                g=p.loc[mask==flag];conditions.append(dict(seed=seed,condition=name,present=flag,**metrics(g),SSE_fraction=float(g.squared_error_mV2.sum()/sse)))
    tables={'whole':whole,'steps':step_rows,'quarters':quarter,'local_10h':local,'tails':tails,'overlap':overlap,'conditions':conditions}
    for name,rows in tables.items():csv(out/(name+'.csv'),rows)
    csv(out/'target_SSE_contribution.csv',pd.concat(targets,ignore_index=True));csv(out/'largest_20_queries_per_seed.csv',pd.concat(extreme,ignore_index=True))
    steps=pd.DataFrame(step_rows);csv(out/'step_mean.csv',steps.groupby('step')[['RMSE_mV','MAE_mV','R2']].mean().reset_index())
    distribution=[];distribution_plot=[]
    for part,(start,end) in c.partitions(len(frame)).items():
        block=frame.iloc[start:end]
        for name in ['cell_voltage_mean','actual_current_density_A_cm2_mean','zero_current_fraction']:
            values=block[name].dropna().to_numpy();distribution.append(dict(partition=part,variable=name,unit='V' if name=='cell_voltage_mean' else 'A/cm2' if 'density' in name else 'fraction',n=len(values),
                mean=float(np.mean(values)),std=float(np.std(values)),min=float(np.min(values)),P05=float(np.quantile(values,.05)),median=float(np.median(values)),P95=float(np.quantile(values,.95)),max=float(np.max(values))))
            distribution_plot.extend(dict(partition=part,variable=name,value=float(v)) for v in values)
        values=windows[part].y.ravel().astype(float)
        distribution.append(dict(partition=part,variable='target_DeltaV_queries',unit='V',n=len(values),mean=float(np.mean(values)),std=float(np.std(values)),min=float(values.min()),P05=float(np.quantile(values,.05)),median=float(np.median(values)),P95=float(np.quantile(values,.95)),max=float(values.max())))
        distribution_plot.extend(dict(partition=part,variable='target_DeltaV_queries',value=float(v)) for v in values)
    csv(out/'partition_distributions.csv',distribution);csv(out/'distribution_plot_data.csv',distribution_plot)
    # Compare all three official partition metrics, not only the late errors.
    partition_metrics=[]
    for seed in c.SUPPORTED_SEEDS:
        for part in windows:
            points=c.read_csv(c.OLD_RUN/f'seed{seed}/top10/predictions_{part}.csv')
            c.require(np.array_equal(points.y_true_delta_V,windows[part].y.ravel().astype(float)),'PARTITION_LABEL_IDENTITY')
            partition_metrics.append(dict(seed=seed,partition=part,**metrics(points)))
    csv(out/'partition_metrics.csv',partition_metrics)
    summary=dict(status='PASS',no_training_executed=True,all_old_predictions_and_metrics_recomputed=True,
        six_seed_mean=pd.DataFrame(whole)[['RMSE_mV','MAE_mV','R2']].mean().to_dict(),
        selected_steps=steps.loc[steps.step.isin([1,6,12,24])].groupby('step')[['RMSE_mV','MAE_mV']].mean().reset_index().to_dict('records'),
        mean_top1pct_SSE_fraction=float(pd.DataFrame(tails).top1pct_SSE_fraction.mean()),
        mean_top5pct_SSE_fraction=float(pd.DataFrame(tails).top5pct_SSE_fraction.mean()),
        overlap_queries=12648,unique_targets=619,all_extremes_retained=True,
        interpretation='post-hoc association; no causal attribution or independent-event count')
    js(out/'DIAGNOSIS_SUMMARY.json',summary)
    # Report generation uses these actual tables; figures are produced without further selection.
    from figures import p1_figures,p1_report
    p1_figures(run,frame,all_points);p1_report(run,summary)
    js(run/'P1_ACCEPTANCE.json',summary)
    print('P1 PASS:',summary['six_seed_mean'],'Top1% SSE share',summary['mean_top1pct_SSE_fraction'],flush=True)
    return summary

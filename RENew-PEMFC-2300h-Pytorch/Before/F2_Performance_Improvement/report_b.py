"""Actual Stage-B CSV figures; presentation does not change selection."""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import frozen_f2 as f
import settings as s
from io_utils import csv,text,metrics,enrich
from report_a import save,segments
from repeat_diagnosis import compare

NAMES=['B_control','B_v2']
COLORS={'B_control':'#0F4D92','B_v2':'#42949E'}
LABELS={'B_control':'Native F2','B_v2':'Bounded residual F2-v2'}


def write(run,table,means,result):
    histories=[];paths=[];counter=[];steps=[];points={};extremes=[]
    frame,windows,scaler,prepared,excluded=f.load_selection_data()
    for seed in s.INITIAL_SEEDS:
        for name in NAMES:
            folder=run/f'models/B/{name}_seed{seed}'
            h=f.read_csv(folder/'training_history.csv').assign(seed=seed,candidate=name);histories.append(h)
            paths.append(f.read_csv(folder/'paths_VALIDATION.csv').assign(seed=seed,candidate=name))
            counter.append(f.read_csv(folder/'COUNTERFACTUAL_VALIDATION.csv').assign(seed=seed,candidate=name))
            q=f.read_csv(folder/'predictions_VALIDATION.csv')
            for step,g in q.groupby('horizon_step'):steps.append(dict(candidate=name,seed=seed,step=int(step),**metrics(g)))
            p=enrich(q,frame)
            extremes.append(p.sort_values(['absolute_error_mV','target_row','origin_row','horizon_step'],
                ascending=[False,True,True,True]).head(5).assign(candidate=name))
            if seed==202:points[name]=p
    hist=pd.concat(histories,ignore_index=True);path=pd.concat(paths,ignore_index=True);cf=pd.concat(counter,ignore_index=True)
    csv(run/'tables/B_TRAINING_DIAGNOSTICS.csv',hist);csv(run/'tables/B_GATE_AND_PATHS_VALIDATION.csv',path)
    csv(run/'tables/B_COUNTERFACTUAL_VALIDATION.csv',cf);csv(run/'tables/B_STEP_METRICS.csv',steps)
    csv(run/'tables/B_LARGEST_ERRORS_PER_MODEL.csv',pd.concat(extremes,ignore_index=True))
    x=np.arange(3);width=.35
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for ax,key in zip(axes,['RMSE_mV','MAE_mV','P99_mV']):
        for i,name in enumerate(NAMES):
            q=table.loc[table.candidate==name].set_index('seed').loc[list(s.INITIAL_SEEDS)]
            ax.bar(x+(i-.5)*width,q[key],width,label=LABELS[name],color=COLORS[name],edgecolor='black',lw=.6)
        ax.set_xticks(x,s.INITIAL_SEEDS);ax.set(xlabel='Predeclared seed',ylabel=key.replace('_mV',' (mV)'),title='Validation: 5232 queries per model')
    axes[0].legend(fontsize=8);save(run,'B/01_paired_Validation',fig,table)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,key in zip(axes,['RMSE_mV','MAE_mV']):
        q=means.set_index('candidate').loc[NAMES]
        ax.bar(np.arange(2),q[key+'_mean'],yerr=q[key+'_SD'],capsize=4,
            color=[COLORS[n] for n in NAMES],edgecolor='black',lw=.6)
        ax.set_xticks(np.arange(2),['Native','Residual v2']);ax.set(ylabel=key.replace('_mV',' (mV)'),title='Three fresh seeds: mean ± sample SD')
    save(run,'B/02_Validation_mean_SD',fig,means)
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True)
    for row,name in enumerate(NAMES):
        for column,seed in enumerate(s.INITIAL_SEEDS):
            h=hist.loc[(hist.candidate==name)&(hist.seed==seed)];ax=axes[row,column]
            ax.plot(h.epoch,h.train_unweighted_batch_MSE,color=COLORS[name],label='Train batch-mean')
            ax.plot(h.epoch,h.validation_MSE,color='black',ls='--',label='Full Validation')
            ax.axvline(h.best_epoch.iloc[-1],color='gray',ls=':',lw=1)
            ax.set_yscale('log');ax.set(xlabel='Fresh epoch',ylabel='Standardized DeltaV MSE',title=f'{LABELS[name]} Seed{seed}')
    axes[0,0].legend(fontsize=8);save(run,'B/03_real_training_curves',fig,hist)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
    for ax,seed in zip(axes,s.INITIAL_SEEDS):
        for name in NAMES:
            p=path.loc[(path.seed==seed)&(path.candidate==name)]
            segments(ax,p,'g',xcolumn='origin_age_h',color=COLORS[name],label=LABELS[name],lw=1)
        ax.set(xlabel='Actual origin operating time (h)',ylabel='History-only sigmoid g',ylim=(0,1),title=f'Seed{seed}; fusion meaning differs')
    axes[0].legend(fontsize=8);save(run,'B/04_history_gate',fig,path)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    labels={'fusion_to_lstm_rms':'RMS(fused) / RMS(LSTM)',
        'cnn_to_lstm_contribution_ratio':'RMS(CNN supplement) / RMS(LSTM)',
        'path_cosine':'Cosine similarity (LSTM, CNN)'}
    for ax,key in zip(axes,labels):
        ax.boxplot([path.loc[path.candidate==name,key] for name in NAMES],tick_labels=['Native','Residual v2'],showfliers=True)
        if key=='cnn_to_lstm_contribution_ratio':ax.axhline(.25,color='gray',ls=':',label='v2 upper bound')
        ax.set(ylabel=labels[key],title='Validation: 3 models × 218 origins')
    axes[1].legend(fontsize=8);save(run,'B/05_path_scales_and_bound',fig,path)
    intervention=[];fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
    for ax,seed in zip(axes,s.INITIAL_SEEDS):
        for i,mode in enumerate(['normal','cnn_off','lstm_only']):
            q=(table.loc[table.seed==seed,['candidate','RMSE_mV']] if mode=='normal' else
               cf.loc[(cf.seed==seed)&(cf['mode']==mode),['candidate','RMSE_mV']]).set_index('candidate').loc[NAMES].reset_index()
            q['seed']=seed;q['mode']=mode;intervention.append(q)
            ax.bar(np.arange(2)+(i-1)*.24,q.RMSE_mV,.24,label=mode,edgecolor='black',lw=.6)
        ax.set_xticks(np.arange(2),['Native','Residual v2']);ax.set(ylabel='Validation RMSE (mV)',title=f'Seed{seed}: inference-only intervention')
    axes[0].legend(fontsize=8);save(run,'B/06_CNN_off_counterfactual',fig,pd.concat(intervention,ignore_index=True))
    step=pd.DataFrame(steps);fig,ax=plt.subplots(figsize=(9,4.5));plot=[]
    for name in NAMES:
        p=step.loc[step.candidate==name].groupby('step').RMSE_mV.agg(['mean','std']).reset_index();p['candidate']=name;plot.append(p)
        ax.plot(p.step,p['mean'],color=COLORS[name],label=LABELS[name]);ax.fill_between(p.step,p['mean']-p['std'],p['mean']+p['std'],color=COLORS[name],alpha=.12)
    ax.set(xlabel='Forecast step (0.5 h each)',ylabel='Validation DeltaV RMSE (mV)',title='Equal three-seed means ± sample SD; not confidence intervals')
    ax.legend();save(run,'B/07_24step_RMSE',fig,pd.concat(plot,ignore_index=True))
    fig,ax=plt.subplots(figsize=(12,4.5));fixed=[]
    for name,p in points.items():
        q=p.loc[p.horizon_step==24].assign(candidate=name);fixed.append(q)
        if name==NAMES[0]:segments(ax,q,'true_voltage_V',color='black',label='Observed',lw=1.2)
        segments(ax,q,'predicted_voltage_V',color=COLORS[name],label=LABELS[name],lw=1)
    ax.set(xlabel='Actual target operating time (h)',ylabel='Absolute voltage (V)',title='Validation Seed202: fixed Step24; 218 forecasts, gaps preserved')
    ax.legend();save(run,'B/08_fixed_Step24_voltage',fig,pd.concat(fixed,ignore_index=True))
    all_values=np.concatenate([p.true_voltage_V.to_numpy() for p in points.values()]+[p.predicted_voltage_V.to_numpy() for p in points.values()])
    lo,hi=all_values.min(),all_values.max();pad=(hi-lo)*.04
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharex=True,sharey=True)
    for ax,(name,p) in zip(axes,points.items()):
        ax.scatter(p.true_voltage_V,p.predicted_voltage_V,s=4,alpha=.3,color=COLORS[name])
        ax.plot([lo,hi],[lo,hi],color='black',ls='--',lw=1);ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad)
        ax.set(xlabel='Observed absolute voltage (V)',ylabel='Predicted absolute voltage (V)',title=f'{LABELS[name]}: Seed202, 5232 Val queries')
    scatter=pd.concat([p.assign(candidate=n) for n,p in points.items()],ignore_index=True)
    save(run,'B/09_full_scatter',fig,scatter)
    fig=plt.figure(figsize=(11,4.4));grid=fig.add_gridspec(1,3,width_ratios=[1,1,.05])
    first=fig.add_subplot(grid[0,0]);second=fig.add_subplot(grid[0,1],sharex=first,sharey=first)
    axes=[first,second];cax=fig.add_subplot(grid[0,2])
    density=np.concatenate([p.target_actual_current_density_A_cm2.to_numpy() for p in points.values()])
    for ax,(name,p) in zip(axes,points.items()):
        artist=ax.scatter(p.true_voltage_V,p.predicted_voltage_V,s=4,alpha=.4,c=p.target_actual_current_density_A_cm2,
            cmap='viridis',vmin=density.min(),vmax=density.max())
        ax.plot([lo,hi],[lo,hi],color='black',ls='--',lw=1);ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad)
        ax.set(xlabel='Observed absolute voltage (V)',ylabel='Predicted absolute voltage (V)',title=LABELS[name]+'; Validation Seed202')
    fig.colorbar(artist,cax=cax,label='Observed target current density (A/cm²); post-hoc only')
    save(run,'B/10_current_colored_scatter',fig,scatter)
    groups=[];fig,axes=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
    for ax,group in zip(axes,['event','non_event']):
        for i,name in enumerate(NAMES):
            q=table.loc[table.candidate==name].set_index('seed').loc[list(s.INITIAL_SEEDS)]
            ax.bar(x+(i-.5)*width,q[group+'_RMSE_mV'],width,color=COLORS[name],label=LABELS[name],edgecolor='black',lw=.6)
            groups.extend(dict(candidate=name,seed=int(seed),group=group,RMSE_mV=float(row[group+'_RMSE_mV']),
                queries=int(row[group+'_queries']),unique_targets=int(row[group+'_unique_targets'])) for seed,row in q.iterrows())
        ax.set_xticks(x,s.INITIAL_SEEDS);ax.set(xlabel='Predeclared seed',ylabel='Validation raw DeltaV RMSE (mV)',title=group+' targets; fixed Train-defined rule')
    axes[0].legend(fontsize=8);save(run,'B/11_event_non_event_errors',fig,groups)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
    for ax,name in zip(axes,NAMES):
        h=hist.loc[(hist.candidate==name)&(hist.seed==202)]
        for key,color in [('lstm_encoder_parameter_grad_norm','#0F4D92'),('cnn_parameter_grad_norm','#42949E'),('gate_parameter_grad_norm','#B56C36')]:
            ax.plot(h.epoch,h[key],color=color,label=key.replace('_parameter_grad_norm',''),lw=1)
        ax.set_yscale('symlog',linthresh=1e-7);ax.set(xlabel='Fresh epoch',ylabel='Pre-clip gradient L2 norm',title=LABELS[name]+'; Seed202 first Train batch')
    axes[0].legend(fontsize=8);save(run,'B/12_path_training_gradients',fig,hist.loc[hist.seed==202])
    repeats=compare(run)
    if repeats.same_configuration.eq(True).all():
        fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True);repeat_curves=[]
        for ax,seed in zip(axes,s.INITIAL_SEEDS):
            for stage,name,label,color in [('A','A0','A0 planned first run','#0F4D92'),('B','B_control','B planned control repeat','#B56C36')]:
                h=f.read_csv(run/f'models/{stage}/{name}_seed{seed}/training_history.csv').assign(seed=seed,repeat=label);repeat_curves.append(h)
                ax.plot(h.epoch,h.validation_MSE,label=label,color=color,lw=1)
            ax.set_yscale('log');ax.set(xlabel='Fresh epoch',ylabel='Full Validation standardized MSE',title=f'Seed{seed}: same config, new initialization call')
        axes[0].legend(fontsize=8);save(run,'B/13_planned_repeat_variation',fig,pd.concat(repeat_curves,ignore_index=True))
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True);time_data=[]
    for name,p in points.items():
        q=p.loc[p.horizon_step==24].assign(candidate=name);time_data.append(q)
        segments(axes[0],q,'absolute_error_mV',color=COLORS[name],label=LABELS[name],lw=1)
        if name==NAMES[0]:
            segments(axes[1],q,'zero_current_fraction',color='black',lw=1)
            segments(axes[2],q,'target_actual_current_density_A_cm2',color='black',lw=1)
    axes[0].set(ylabel='Absolute error (mV)',title='Validation Seed202, Step24: actual future conditions are post-hoc diagnostics')
    axes[0].legend();axes[1].set(ylabel='Zero-current fraction');axes[2].set(ylabel='Current density (A/cm²)',xlabel='Actual target operating time (h)')
    save(run,'B/14_time_and_conditions',fig,pd.concat(time_data,ignore_index=True))
    cost=table[['candidate','seed','parameters','best_epoch','stop_epoch','training_seconds']].copy()
    csv(run/'tables/B_PARAMETER_AND_COST.csv',cost)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for i,name in enumerate(NAMES):
        q=cost.loc[cost.candidate==name].set_index('seed').loc[list(s.INITIAL_SEEDS)]
        axes[0].bar(x+(i-.5)*width,q.training_seconds/60,width,label=LABELS[name],color=COLORS[name],edgecolor='black',lw=.6)
        axes[1].bar(x+(i-.5)*width,q.stop_epoch,width,color=COLORS[name],edgecolor='black',lw=.6)
    for ax in axes:ax.set_xticks(x,s.INITIAL_SEEDS);ax.set_xlabel('Predeclared seed')
    axes[0].set(ylabel='Observed wall time (min)',title='Includes actual logging/plotting');axes[1].set(ylabel='Stop epoch',title='Same 450/40 policy; actual work differs')
    axes[0].legend(fontsize=8);save(run,'B/15_training_cost',fig,cost)
    rows=''.join(f'| {r.candidate} | {int(r.seed)} | {r.RMSE_mV:.6f} | {r.MAE_mV:.6f} | {r.R2:.9f} | {r.P99_mV:.6f} | {int(r.best_epoch)} |\n' for r in table.itertuples())
    mean_rows=''.join(f'| {r.candidate} | {r.RMSE_mV_mean:.6f} ± {r.RMSE_mV_SD:.6f} | {r.MAE_mV_mean:.6f} ± {r.MAE_mV_SD:.6f} | {r.R2_mean:.9f} |\n' for r in means.itertuples())
    text(run/'02_阶段B_残差融合研究.md',f'''# 阶段B：真实受控融合对照

六次新训练完成。F2-v2初步资格为{result['qualified']}，还须独立核验全部新权重及参数变化。与新训练原融合对照相比，RMSE配对改善{result['matched_control_decision']['paired_wins']}/3，平均比值{result['matched_control_decision']['mean_RMSE_ratio']:.6f}；相对原A0，配对改善{result['A0_reference_decision']['paired_wins']}/3，平均比值{result['A0_reference_decision']['mean_RMSE_ratio']:.6f}。完整守门结果保存在B_RESULT.json，没有因平均值较好就忽略单个种子恶化。

## 融合究竟改了什么

原式H=g HL+(1-g) HC；唯一新式H=HL+0.25(1-g) RMS(HL)/max(RMS(HC),1e-6) HC。0.25及1e-6在第一步B训练之前冻结。归一化仅看本例48步已知历史，反向梯度通过归一化及两路径。新旧模型均332436参数，初始权重键和值逐点一致；只有fuse_encoder被覆盖，TFT其他层与未来decoder相对时间坐标不变。没有把普通LSTM改名，HL仍是完整TFT的一部分。

新式保留HL系数为1，让CNN在RMS最多0.25倍HL的范围内作补充。g现在调节CNN补充，而非凸组合中的LSTM比例；不能将两种模型相同数值的g解释成相同总路径贡献。CNN补充范数不是预测误差、物理电压或贡献百分比。即使新式有效，也只检验这个受控融合设计，不能从向量RMS缩小直接推导原式“丢信息”。

HL具体指原生post_lstm_gate_encoder及post_lstm_add_norm_encoder之后的历史编码序列，不是未经处理的LSTM hidden/cell。原生hidden/cell仍给decoder初始化。每个窗口只有一个历史g标量，经历史memory影响24步输出；并非24个独立未来工况门控。

虽然state_dict键形状兼容，v2权重必须按checkpoint中的configuration调用残差融合定义，不能用原融合类加载后冒称同一模型。本阶段load_new同时验证配置、Seed、阶段、最佳Epoch及checkpoint SHA；不修改任何正式主程序或原历史权重映射。

## 全部Validation结果：5232查询/218起点/264唯一目标

| 方法 | Seed | RMSE(mV) | MAE(mV) | R²(原始ΔV) | P99(mV) | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
{rows}
| 方法 | 三种子RMSE均值±SD(mV) | MAE均值±SD(mV) | R²均值 |
|---|---:|---:|---:|
{mean_rows}

普通MSE、原始学习率/Dropout、AdamW、450/40政策、输入、标签、掩码、scaler均匹配。Train曲线为Dropout开启的batch-mean无权重MSE，Validation曲线为完整验证集MSE；最佳权重评估的Train RMSE另存在各METRICS.csv，不能混为同一种曲线。

## CNN是否真在工作

```text
{f.read_csv(run/'tables/B_CNN_PARTICIPATION.csv').to_string(index=False)}
```

关闭CNN补充后，v2的Validation RMSE在{result['CNN_off_worse_seeds']}/3种子变差；实际预测变化、补充幅度、训练路径梯度均保存。独立审计另外核查CNN各核、LSTM和门控从初始到最佳权重的变化及真实梯度。若CNN有梯度和预测影响但整体效果未优于原融合，结论应是“参与了、尚无可靠升级收益”，不能把参与等同于提升。

反事实只加载本次模型、临时改变推理融合，不改checkpoint，不是重新训练消融。原式CNN-off输出g HL，新式CNN-off输出HL，干预含义不同；图B06同时展示原式LSTM-only=HL作为补充，不能把它命名为独立普通LSTM。v2的CNN-off与LSTM-only相同，属预期数学结果。

## 计划内重复与普通CUDA波动

```text
{repeats.to_string(index=False)}
```

A0和B_control是事先安排的独立训练，若配置相同才作为重复性诊断；没有额外重跑、没有择优替换A0。共同Epoch的初始化、输入、采样顺序、CPU/CUDA RNG摘要和权重状态分别比较。即使随机状态相同，普通CUDA也不保证所有反向规约逐位确定；不能因为最终不同就直接归咎于随机种子未设置。首个不同Epoch及Loss差值保留，已知严格确定性插值反向限制没有通过替换网络规避。

本表只说明这些计划内实例的实际波动，样本量不足以估计一般重复运行分布，也不能选择更好的对照重复作为新增模型。主要资格要求同时对新配对对照和原A0成立。

## 训练成本及科学范围

```text
{cost.to_string(index=False)}
```

training_seconds是包含实时日志、绘图和同步的墙钟时间，不是独占CUDA kernel计时。相同轮数上限/耐心值不保证早停轮次、更新数和耗时相同。额外归一化无可训练参数，但会增加少量前向/反向运算。

所有图来自本次真实CSV，300dpi PNG/PDF/绘图CSV齐全。Step24时间曲线是固定步长，缺口保留；全部24步用于主指标，不因绘图减少查询。电流、零电流占比与事件身份只用于Validation事后分组，不作为未来decoder输入。P95/P99/最大误差严格复原冻结float32误差路径，另列float64精度差异。

本阶段没有构造或评价新后留窗口。是否进入事件加权及六种子训练须在B独立核验后依据固定门槛决定；没有合格前置候选就停止升级，不强行开展大规模训练。三种子均值±SD不是置信区间或显著性证据。
''')

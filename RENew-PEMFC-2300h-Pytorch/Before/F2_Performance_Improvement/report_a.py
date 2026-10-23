"""Stage A tables and actual CSV figures. Does not change qualification rules."""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display,Image
import frozen_f2 as f
import settings as s
from io_utils import csv,text,metrics

COLORS={'A0':'#0F4D92','A1':'#42949E','A2':'#B56C36'}
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save(run,name,fig,data):
    path=run/'figures'/name;path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(pad=1.4);fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    csv(path.with_name(path.name+'_data.csv'),data)
    display(Image(filename=str(path.with_suffix('.png'))))


def segments(ax,p,column,xcolumn='target_age_h',**kwargs):
    p=p.sort_values(xcolumn);group=p[xcolumn].diff().sub(.5).abs().gt(1e-6).cumsum()
    for n,(_,g) in enumerate(p.groupby(group)):
        options=kwargs.copy()
        if n:options['label']=None
        ax.plot(g[xcolumn],g[column],**options)


def write(run,table,means,result):
    histories=[];paths=[];counter=[];steps=[];points={};stability=[]
    for seed in s.INITIAL_SEEDS:
        for candidate in s.CONFIGS_A:
            folder=run/f'models/A/{candidate}_seed{seed}'
            h=f.read_csv(folder/'training_history.csv');h['seed']=seed;h['candidate']=candidate;histories.append(h)
            v=h.validation_MSE.to_numpy(float);m=f.read_csv(folder/'METRICS.csv').set_index('partition')
            stability.append(dict(candidate=candidate,seed=seed,Validation_spike_fraction_2x=float(np.mean(v[1:]>2*v[:-1])),
                maximum_preclip_global_gradient=float(h.gradient_norm_max.max()),mean_fraction_batches_clipped=float(h.clipped_fraction.mean()),
                first_batch_CNN_gradient_median=float(h.cnn_parameter_grad_norm.median()),
                first_batch_LSTM_gradient_median=float(h.lstm_encoder_parameter_grad_norm.median()),
                Train_eval_RMSE_mV=float(m.loc['TRAIN','RMSE_mV']),Validation_RMSE_mV=float(m.loc['VALIDATION','RMSE_mV']),
                best_epoch=int(h.best_epoch.iloc[-1]),stop_epoch=len(h)))
            p=f.read_csv(folder/'paths_VALIDATION.csv');p['seed']=seed;p['candidate']=candidate;paths.append(p)
            cf=f.read_csv(folder/'COUNTERFACTUAL_VALIDATION.csv');cf['seed']=seed;cf['candidate']=candidate;counter.append(cf)
            q=f.read_csv(folder/'predictions_VALIDATION.csv')
            for step,g in q.groupby('horizon_step'):steps.append(dict(candidate=candidate,seed=seed,step=int(step),**metrics(g)))
            if seed==202:points[candidate]=q
    hist=pd.concat(histories,ignore_index=True);path=pd.concat(paths,ignore_index=True);cf=pd.concat(counter,ignore_index=True)
    csv(run/'tables/A_TRAINING_DIAGNOSTICS.csv',hist);csv(run/'tables/A_GATE_AND_PATHS_VALIDATION.csv',path)
    csv(run/'tables/A_COUNTERFACTUAL_VALIDATION.csv',cf);csv(run/'tables/A_STEP_METRICS.csv',steps)
    csv(run/'tables/A_STABILITY_PER_SEED.csv',stability)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2));x=np.arange(3);width=.24
    for ax,key in zip(axes,['RMSE_mV','MAE_mV','P99_mV']):
        for index,candidate in enumerate(s.CONFIGS_A):
            values=table.loc[table.candidate==candidate].set_index('seed').loc[list(s.INITIAL_SEEDS)]
            ax.bar(x+(index-1)*width,values[key],width,label=candidate,color=COLORS[candidate],edgecolor='black',lw=.6)
        ax.set_xticks(x,s.INITIAL_SEEDS);ax.set(xlabel='Predeclared seed',ylabel=key.replace('_mV',' (mV)'),title='Validation: 5232 queries per model')
    axes[0].legend();save(run,'A/01_candidate_Validation',fig,table)
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True,sharey=True)
    for row,seed in enumerate(s.INITIAL_SEEDS):
        for column,candidate in enumerate(s.CONFIGS_A):
            h=hist.loc[(hist.seed==seed)&(hist.candidate==candidate)];ax=axes[row,column]
            ax.plot(h.epoch,h.train_unweighted_batch_MSE,label='Train batch-mean',color=COLORS[candidate])
            ax.plot(h.epoch,h.validation_MSE,label='Full Validation',color='black',ls='--')
            ax.axvline(h.best_epoch.iloc[-1],color='gray',ls=':',lw=1)
            ax.set_yscale('log');ax.set(xlabel='Fresh epoch',ylabel='Standardized DeltaV MSE',title=f'{candidate} Seed{seed}')
    axes[0,0].legend(fontsize=8);save(run,'A/02_real_training_curves',fig,hist)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,seed in zip(axes,s.INITIAL_SEEDS):
        for candidate in s.CONFIGS_A:
            p=path.loc[(path.seed==seed)&(path.candidate==candidate)]
            segments(ax,p,'g',xcolumn='origin_age_h',color=COLORS[candidate],label=candidate,lw=1)
        ax.set(xlabel='Actual origin operating time (h)',ylabel='Native gate g',ylim=(0,1),title=f'Validation Seed{seed}')
    axes[0].legend();save(run,'A/03_gate_behavior',fig,path)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,['fusion_to_lstm_rms','cnn_to_lstm_contribution_ratio','path_cosine']):
        ax.boxplot([path.loc[path.candidate==candidate,key] for candidate in s.CONFIGS_A],tick_labels=list(s.CONFIGS_A),showfliers=True)
        labels={'fusion_to_lstm_rms':'RMS(fused) / RMS(LSTM)',
            'cnn_to_lstm_contribution_ratio':'RMS(weighted CNN) / RMS(LSTM)',
            'path_cosine':'Cosine similarity (LSTM, CNN)'}
        ax.set(ylabel=labels[key],title='Three seeds, 218 origins each')
    save(run,'A/04_path_scales',fig,path)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
    for ax,candidate in zip(axes,s.CONFIGS_A):
        h=hist.loc[(hist.seed==202)&(hist.candidate==candidate)]
        for key,color in [('lstm_encoder_parameter_grad_norm','#0F4D92'),('cnn_parameter_grad_norm','#42949E'),('gate_parameter_grad_norm','#B56C36')]:
            ax.plot(h.epoch,h[key],label=key.replace('_parameter_grad_norm',''),color=color,lw=1)
        ax.set_yscale('symlog',linthresh=1e-7);ax.set(xlabel='Fresh epoch',ylabel='Pre-clip gradient L2 norm',title=f'{candidate} Seed202: first Train batch')
    axes[0].legend(fontsize=8);save(run,'A/05_training_path_gradients',fig,hist.loc[hist.seed==202])
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True);cf_data=[]
    for ax,seed in zip(axes,s.INITIAL_SEEDS):
        for i,mode in enumerate(['normal','cnn_off','lstm_only']):
            if mode=='normal':values=table.loc[table.seed==seed,['candidate','RMSE_mV']].copy()
            else:values=cf.loc[(cf.seed==seed)&(cf['mode']==mode),['candidate','RMSE_mV']].copy()
            values=values.set_index('candidate').loc[list(s.CONFIGS_A)].reset_index();values['mode']=mode;values['seed']=seed;cf_data.append(values)
            ax.bar(np.arange(3)+(i-1)*.24,values.RMSE_mV,.24,label=mode,edgecolor='black',lw=.6)
        ax.set_xticks(np.arange(3),list(s.CONFIGS_A));ax.set(xlabel='Same saved model; inference-only interventions',ylabel='Validation RMSE (mV)',title=f'Seed{seed}')
    axes[0].legend(fontsize=8);save(run,'A/06_inference_counterfactuals',fig,pd.concat(cf_data,ignore_index=True))
    st=pd.DataFrame(steps);fig,ax=plt.subplots(figsize=(9,4.5));st_plot=[]
    for candidate,g in st.groupby('candidate'):
        p=g.groupby('step').RMSE_mV.agg(['mean','std']).reset_index();p['candidate']=candidate;st_plot.append(p)
        ax.plot(p.step,p['mean'],color=COLORS[candidate],label=candidate)
        ax.fill_between(p.step,p['mean']-p['std'],p['mean']+p['std'],color=COLORS[candidate],alpha=.12)
    ax.set(xlabel='Forecast step (0.5 h each)',ylabel='Validation RMSE of raw DeltaV (mV)',title='Equal three-seed means ± sample SD; not confidence intervals')
    ax.legend();save(run,'A/07_Validation_horizon_RMSE',fig,pd.concat(st_plot,ignore_index=True))
    fig,ax=plt.subplots(figsize=(12,4.5));fixed=[]
    for candidate,p in points.items():
        q=p.loc[p.horizon_step==24].copy();q['candidate']=candidate;fixed.append(q)
        if candidate=='A0':segments(ax,q,'true_voltage_V',color='black',label='Observed',lw=1.2)
        segments(ax,q,'predicted_voltage_V',color=COLORS[candidate],label=candidate,lw=1)
    ax.set(xlabel='Actual target operating time (h)',ylabel='Absolute voltage (V)',title='Validation Seed202: fixed Step24, 218 forecasts; gaps remain gaps')
    ax.legend();save(run,'A/08_absolute_voltage_Step24',fig,pd.concat(fixed,ignore_index=True))
    all_values=np.concatenate([p.true_voltage_V.to_numpy() for p in points.values()]+[p.predicted_voltage_V.to_numpy() for p in points.values()])
    lo,hi=all_values.min(),all_values.max();pad=(hi-lo)*.04
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharex=True,sharey=True)
    for ax,(candidate,p) in zip(axes,points.items()):
        ax.scatter(p.true_voltage_V,p.predicted_voltage_V,s=4,alpha=.25,color=COLORS[candidate])
        ax.plot([lo,hi],[lo,hi],color='black',ls='--',lw=1);ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad)
        ax.set(xlabel='Observed absolute voltage (V)',ylabel='Predicted absolute voltage (V)',title=f'{candidate} Seed202: 5232 Val queries')
    save(run,'A/09_Validation_scatter',fig,pd.concat([p.assign(candidate=k) for k,p in points.items()],ignore_index=True))
    fig,ax=plt.subplots(figsize=(9,4.2))
    for candidate in s.CONFIGS_A:
        ax.hist(path.loc[path.candidate==candidate,'g'],bins=np.linspace(0,1,51),density=True,
            histtype='step',lw=1.6,label=candidate,color=COLORS[candidate])
    ax.set(xlabel='History-only native gate g',ylabel='Descriptive density',xlim=(0,1),
        title='Validation gate distributions: 3 models × 218 origins per candidate')
    ax.legend();save(run,'A/10_gate_distributions',fig,path[['candidate','seed','origin_row','origin_age_h','g']])
    old=f.read_csv(run/'historical/TRAIN_VALIDATION_METRICS.csv').groupby(['model','partition'])[['RMSE_mV','MAE_mV','R2']].mean()
    old_stability=f.read_csv(run/'historical/TRAINING_STABILITY.csv')
    rows=''.join(f'| {r.candidate} | {int(r.seed)} | {r.RMSE_mV:.6f} | {r.MAE_mV:.6f} | {r.R2:.9f} | {r.P99_mV:.6f} | {int(r.best_epoch)} | {int(r.stop_epoch)} |\n' for r in table.itertuples())
    mean_rows=''.join(f'| {r.candidate} | {r.RMSE_mV_mean:.6f} ± {r.RMSE_mV_SD:.6f} | {r.MAE_mV_mean:.6f} ± {r.MAE_mV_SD:.6f} | {r.R2_mean:.9f} |\n' for r in means.itertuples())
    dec='\n'.join(f"- {name}：{'达到预定一致性门槛' if v['qualified'] else '未达到升级门槛'}；RMSE配对改善{v['paired_wins']}/3；平均比值{v['mean_RMSE_ratio']:.6f}；逐项条件见A_RESULT.json。" for name,v in result['decisions'].items())
    plain='\n'.join(f"{name}的三种子平均Validation RMSE相对本轮A0变化为{100*(v['mean_RMSE_ratio']-1):+.2f}%，但仅{v['paired_wins']}/3个种子改善。" for name,v in result['decisions'].items())
    text(run/'01_阶段A_训练配置研究.md',f'''# 阶段A：先说结果

九次原生F2新训练已经完成。初步合格候选为{result['qualified_candidates']}，下一阶段冻结的训练配置为{result['selected_training_configuration']}。未把普通LSTM替换或改名为F2，没有改结构。

{plain}

平均值变好不等于训练已经稳定，也不等于所有种子都改善。是否升级严格依照训练前规则；没有为取得正面结论重跑某个差种子。后续同配置的新训练对照如果出现波动，也不能反过来替换本轮A0结果。

{dec}

## 旧结果只读诊断

旧六种子的Train/Validation真实预测严格重放一致；输入/标签和checkpoint身份通过，历史权重未更新。三项均值如下，不能与后留成绩混作同一区间比较：

```text
{old.to_string()}
```

```text
{old_stability.to_string(index=False)}
```

旧日志的gradient min/max不能恢复每批裁剪比例，因此旧clipped_fraction记NA；新日志记录真实每批裁剪比例。历史梯度诊断只在train模式、Dropout开启的一批Train数据反向传播，不创建optimizer，更新为0，不冒充完整新训练。最初eval模式反向被cuDNN拒绝的错误及原脚本已保留；没有修改网络或将该错误误称为严格确定性算子错误。新训练另外逐Epoch记录第一批Train的路径梯度、全局梯度范围、真实参数变化。

## 全部九次真实结果

| 方案 | Seed | Val RMSE(mV) | Val MAE(mV) | Val R²(ΔV) | Val P99(mV) | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}
| 方案 | 三种子RMSE均值±SD(mV) | MAE均值±SD(mV) | R²均值 |
|---|---:|---:|---:|
{mean_rows}

固定训练诊断汇总（spike为相邻Epoch的Validation MSE超过前一轮2倍；梯度为裁剪前；Train eval使用本次最佳权重、Dropout关闭，不等于训练中Dropout开启的batch-mean曲线）：

```text
{pd.DataFrame(stability).to_string(index=False)}
```

数值精度说明：三项主指标显式用float64误差累计。冻结代码的P95/P99及最大误差先在float32预测/标签列上计算误差，再做分位数插值；独立核验同时给出全float64误差分位数及差值，并严格重建原float32路径。不会用数值容差放过时间、样本、标签或权重身份错位，也不修改已冻结的尾部筛选指标。

梯度范数大小还受参数量、路径及批次影响。即使F2范数比普通LSTM大，也不能直接称为梯度爆炸；应结合是否有限、裁剪比例和受控配置变化判断。

A0学习率0.0009605449687795866、Dropout=.1；A1仅学习率=.0006；A2仅Dropout=.05。450轮上限、patience40、MSE、AdamW、weight_decay、batch32、clip1全部相同。三个方案各自从头训练，相同种子的初始化及共同Epoch采样顺序严格一致。增加只读探针不改变原生前向公式，不能因此承诺跨运行CUDA逐位确定性。

## 融合与训练状态

原式为`H=g*H_LSTM+(1-g)*H_CNN`。本轮A0 Train门控中位数{result['A0_median_gate']:.6f}，融合/LSTM路径RMS均值{result['A0_mean_fusion_to_LSTM_RMS']:.6f}。这是隐空间数值尺度，不是物理量，也不能单独证明融合损害精度；后续层可能补偿尺度。

两路径方向不同时，凸组合的RMS比原路径小是正常几何现象；并不自动意味着信息丢失。CSV中的`cnn_to_lstm_contribution_ratio`严格定义为`RMS((1-g)*H_CNN)/RMS(H_LSTM)`，分母是未加权的LSTM隐状态基准；它不是两条已加权路径的比值，也不是预测贡献百分比。`weighted_lstm_rms`才是`RMS(g*H_LSTM)`。各列保留原始代码名称以便复算，解读必须遵循这个定义。

融合位置是post-LSTM的历史encoder状态，在与未改动的decoder状态拼接之前。原生LSTM的hidden/cell仍用于decoder初始化，输出头的decoder残差也保留。不能将历史encoder的g直接解读为整个模型中LSTM信息的总占比，更不能由RMS缩小就认定精度降低。

三组使用相同训练政策和轮数上限，实际早停轮次、更新数及墙钟耗时可以不同；training_seconds包含实时日志/绘图，不是独占CUDA kernel计时。不能为慢收敛候选追加轮数，或假定尚未执行的更长训练必定更好。

阶段B衰减假设的研究资格：{result['B_motivation_qualified']}；条件为{result['B_motivation_checks']}。若不满足，停止此路线，不为了凑残差融合而放宽阈值。A1/A2若仅个别种子改善，也不能据此认定学习率或Dropout是主要原因。

原生TFT路径不是普通LSTM对照模型。推理中只保留TFT的LSTM隐路径，仍有TFT的VSN/Attention/decoder/head，并且共享已经训练的权重；它不是从头训练的普通LSTM或独立结构消融。所有反事实仅用于Validation诊断，未改checkpoint。

## 科学边界与下一步

本阶段只看Train/Validation，没有构造、推理或计算新后留成绩。当前最多只有三种子的探索性一致性证据，不做显著性宣称、不追加种子挽救失败配置。最终是否优于历史普通LSTM的后留成绩，必须等唯一方案冻结并完成六个全新模型后才允许判断；若没有合格方案，该项记NA。

图A01～A10与对应CSV展示真实验证误差、实时训练日志、门控、路径尺度、梯度、反事实和逐步误差。误差指标是原始ΔV；绝对电压图使用未经EWMA的原始起点电压恢复。各预测步长和所有异常查询均保留。
''')
    text(run/'A_README.md','''# 阶段A导航

先读01_阶段A_训练配置研究.md；完整数值在tables/A_VALIDATION_PER_SEED.csv及A_VALIDATION_MEAN_SD.csv，固定判据与结果在PROTOCOL.json及A_RESULT.json。

figures/A：01验证集三指标、02九次真实训练曲线、03门控、04路径尺度、05训练梯度、06推理反事实、07逐步误差、08固定Step24绝对电压、09全部Validation查询散点、10门控分布。全部PNG300dpi/PDF/绘图CSV。

models/A每个实例独立保存本次最佳checkpoint、日志、Train/Validation预测及门控路径、验证反事实、身份审计。historical只读诊断均为Train/Validation；没有读取旧后留预测文件。
''')

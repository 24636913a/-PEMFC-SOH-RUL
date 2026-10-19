"""Publication figures from real CSVs; no training, filtering or temporal smoothing."""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image,display
import core as c
from io_utils import csv,text

COLORS={'LSTM':'#B64342','F2':'#0F4D92','TRAIN':'#0F4D92','VALIDATION':'#42949E','TEST':'#B64342'}
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':11,'axes.linewidth':1.2,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def save(run,name,fig,data=None,show=False):
    path=run/'figures'/name;path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(pad=1.4);fig.savefig(path.with_suffix('.png'),dpi=300,facecolor='white',bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),facecolor='white',bbox_inches='tight');plt.close(fig)
    if data is not None:csv(path.with_name(path.name+'_data.csv'),data)
    if show:display(Image(filename=str(path.with_suffix('.png'))))
    return path.with_suffix('.png')


def discontinuous(ax,p,column,color,label,**kwargs):
    p=p.sort_values('target_age_h')
    groups=p.target_age_h.diff().sub(.5).abs().gt(1e-6).cumsum()
    for index,(_,g) in enumerate(p.groupby(groups)):
        ax.plot(g.target_age_h,g[column],color=color,label=label if index==0 else None,**kwargs)


def p1_figures(run,frame,point_sets):
    out=run/'P1_existing_F2';steps=c.read_csv(out/'steps.csv');mean=c.read_csv(out/'step_mean.csv')
    fig,ax=plt.subplots(figsize=(9,4))
    for seed,g in steps.groupby('seed'):ax.plot(g.step,g.RMSE_mV,color='gray',alpha=.4,lw=.9,label='Individual seed' if seed==42 else None)
    ax.plot(mean.step,mean.RMSE_mV,color=COLORS['F2'],lw=2,label='Equal six-seed mean')
    ax.set(xlabel='Forecast step (0.5 h each)',ylabel='RMSE of raw DeltaV (mV)',title='Existing F2: exploratory chronological holdout, 527 origins / seed')
    ax.legend();save(run,'P1/01_existing_step_RMSE',fig,steps,True)
    p=next(x for x in point_sets if int(x.seed.iloc[0])==202)
    fig,ax=plt.subplots(figsize=(10,4));ax.scatter(p.target_age_h,p.absolute_error_mV,s=3,alpha=.3,color=COLORS['F2'])
    ax.set(xlabel='Actual operating time (h)',ylabel='Absolute query error (mV)',title='Existing F2 Seed202: all 12648 holdout queries retained')
    save(run,'P1/02_existing_absolute_error_time',fig,p[['target_row','target_age_h','horizon_step','absolute_error_mV']])
    local=c.read_csv(out/'local_10h.csv');fig,ax=plt.subplots(figsize=(10,4))
    for seed,g in local.groupby('seed'):ax.plot((g.start_h+g.end_h)/2,g.RMSE_mV,lw=1,label=str(seed))
    ax.set(xlabel='10 h operating-time bin centre (h)',ylabel='Query-weighted RMSE (mV)',title='Existing F2: fixed-bin local errors; no temporal smoothing')
    ax.legend(ncol=3);save(run,'P1/03_existing_local_RMSE',fig,local)
    fixed=p.loc[p.horizon_step==24];fig,ax=plt.subplots(figsize=(11,4))
    discontinuous(ax,fixed,'true_voltage_V','black','Observed',lw=1.3)
    discontinuous(ax,fixed,'predicted_voltage_V',COLORS['F2'],'F2 Step24',lw=1.,alpha=.8)
    ax.set(xlabel='Actual target operating time (h)',ylabel='Absolute voltage (V)',title='Existing F2 Seed202: fixed Step24, 527 forecasts; gaps remain gaps')
    ax.legend();save(run,'P1/04_existing_voltage_Step24',fig,fixed)
    distribution=c.read_csv(out/'distribution_plot_data.csv');variables=distribution.variable.unique()
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,var in zip(axes.flat,variables):
        for part,g in distribution.loc[distribution.variable==var].groupby('partition'):
            values=np.sort(g.value.to_numpy());ax.plot(values,np.arange(1,len(values)+1)/len(values),label=('Exploratory holdout' if part=='TEST' else part.title())+f' (n={len(values)})',color=COLORS[part])
        ax.set(xlabel={'cell_voltage_mean':'Raw-bin mean voltage (V)','actual_current_density_A_cm2_mean':'Current density (A cm$^{-2}$)','zero_current_fraction':'Zero-current fraction','target_DeltaV_queries':'Raw DeltaV per forecast query (V)'}[var],ylabel='Empirical CDF')
    axes.flat[0].legend();fig.suptitle('Single 70:10:20 split: raw-bin distributions and query-level DeltaV',y=.995)
    save(run,'P1/05_partition_distribution_ECDF',fig,distribution)
    fig,ax=plt.subplots(figsize=(9,4))
    for points in point_sets:
        values=np.sort(points.absolute_error_mV);ax.plot(values,np.arange(1,len(values)+1)/len(values),alpha=.75,label=str(points.seed.iloc[0]))
    ax.set_xscale('symlog',linthresh=1);ax.set(xlabel='Absolute query error (mV; symlog axis includes zero)',ylabel='Empirical CDF',title='Existing F2 holdout: all errors, including extremes')
    ax.legend(ncol=3);save(run,'P1/06_error_distribution',fig,pd.concat([z[['seed','absolute_error_mV']] for z in point_sets]))
    targets=c.read_csv(out/'target_SSE_contribution.csv').groupby(['target_row','target_age_h']).sum_squared_error_mV2.mean().sort_values(ascending=False)
    selected=[]
    for (row,hour),value in targets.items():
        if all(abs(row-r['target_row'])>48 for r in selected):selected.append(dict(target_row=int(row),target_age_h=float(hour),mean_SSE_mV2=float(value)))
        if len(selected)==2:break
    csv(out/'diagnostic_events.csv',selected)
    extreme_figure(run,frame,{'F2':p},selected,'P1/07_extreme_events')


def extreme_figure(run,frame,points,events,name):
    fig,axes=plt.subplots(len(events),3,figsize=(14,3.2*len(events)),squeeze=False);data=[]
    for index,event in enumerate(events):
        row=event['target_row'];raw=frame.iloc[max(3074,row-24):min(len(frame),row+25)].copy()
        raw['target_row']=raw.index;raw['event_centre_row']=row;raw['target_age_h']=raw.bin_end_ophrs2_h
        raw_plot=raw[['target_row','target_age_h','cell_voltage_mean','actual_current_density_A_cm2_mean','zero_current_fraction','load_transition_count','event_centre_row']].copy()
        discontinuous(axes[index,0],raw,'cell_voltage_mean','black','Observed',lw=1.5)
        for kind,p in points.items():
            g=p.loc[(p.horizon_step==24)&p.target_row.between(raw.index.min(),raw.index.max())]
            raw_plot[kind+'_Step24_predicted_voltage_V']=raw_plot.target_row.map(g.set_index('target_row').predicted_voltage_V)
            discontinuous(axes[index,0],g,'predicted_voltage_V',COLORS[kind],kind+' Step24',lw=1,marker='.',ms=3)
        data.append(raw_plot)
        axes[index,1].plot(raw.target_age_h,raw.actual_current_density_A_cm2_mean,color='#42949E',label='Mean current density')
        axes[index,2].plot(raw.target_age_h,raw.zero_current_fraction,color='#767676',label='Zero-current fraction')
        secondary=axes[index,2].twinx();secondary.plot(raw.target_age_h,raw.load_transition_count,color='#B64342',ls=':',label='Load transitions');secondary.set_ylabel('Transitions / bin',color='#B64342')
        for ax in axes[index]:ax.axvline(event['target_age_h'],color='gray',ls='--',lw=.8);ax.set_xlabel('Actual operating time (h)')
        axes[index,0].set_ylabel('Absolute voltage (V)');axes[index,1].set_ylabel('Current density (A cm$^{-2}$)');axes[index,2].set_ylabel('Zero-current fraction')
        axes[index,0].legend(fontsize=8);axes[index,0].set_title(f'P1-selected adverse event: target row {row}')
    save(run,name,fig,pd.concat(data,ignore_index=True))


def p1_report(run,summary):
    out=run/'P1_existing_F2';steps=c.read_csv(out/'step_mean.csv');tails=c.read_csv(out/'tails.csv')
    largest=c.read_csv(out/'largest_20_queries_per_seed.csv')
    maxima=largest.loc[largest.groupby('seed').absolute_error_mV.idxmax(),['seed','origin_row','target_row','target_age_h','horizon_step','true_voltage_V','predicted_voltage_V','absolute_error_mV','zero_current_fraction','load_transition_count']]
    tail_text=tails[['seed','P95_abs_error_mV','P99_abs_error_mV','max_abs_error_mV','top1pct_SSE_fraction','top5pct_SSE_fraction']].to_string(index=False)
    maxima_text=maxima.to_string(index=False)
    case42=maxima.loc[maxima.seed==42].iloc[0]
    parts=c.read_csv(out/'partition_metrics.csv').groupby('partition')[['RMSE_mV','MAE_mV','R2']].mean()
    condition=c.read_csv(out/'conditions.csv');condition=condition.loc[condition.present==True].groupby('condition')[['RMSE_mV','SSE_fraction']].mean()
    step_text=''.join(f"| {h} | {steps.loc[steps.step==h,'RMSE_mV'].iloc[0]:.6f} | {steps.loc[steps.step==h,'MAE_mV'].iloc[0]:.6f} |\n" for h in [1,6,12,24])
    text(run/'01_F2_Error_Diagnosis.md',f'''# F2后期误差诊断：先说主要发现

既有六种子Top10后留平均RMSE为{summary['six_seed_mean']['RMSE_mV']:.6f} mV、MAE为{summary['six_seed_mean']['MAE_mV']:.6f} mV。误差并非每一时刻都大：最大的1%查询贡献平均{summary['mean_top1pct_SSE_fraction']*100:.2f}%平方误差，最大的5%贡献{summary['mean_top5pct_SSE_fraction']*100:.2f}%。异常点全部保留。

固定Step24均值RMSE为{steps.loc[steps.step==24,'RMSE_mV'].iloc[0]:.6f} mV；Step1为{steps.loc[steps.step==1,'RMSE_mV'].iloc[0]:.6f} mV。24步完整曲线和逐种子表已保存，不能只报告有利的某一步。

## 时间顺序与重叠查询

后留527个起点产生12648个查询，只有619个唯一目标时间。相同目标会被多个起点/提前量重复预测；原RMSE按查询加权，覆盖次数多的目标权重更高，不能把12648条当作独立实验。另列target-time-equal RMSE与固定Step24，二者是辅助口径，不替换正式成绩。

四个固定原始行区间[3074,3266,3458,3650,3843]只作描述性分段，不重划训练数据。局部RMSE按固定10h运行时钟分箱，空档不补点。原始电压图固定Step24，每个起点只取同一步，不拼接重叠预测窗口制造连续单次预测。

| 步长 | 六种子平均RMSE（mV） | 平均MAE（mV） |
|---|---:|---:|
{step_text}
## 尾部与最大误差记录

各种子的P95/P99、最大误差及平方误差贡献（误差列单位mV，SSE_fraction为比例）：

```text
{tail_text}
```

每个种子的最大误差查询、真实/预测电压及目标实际工况（工况仅事后解释）：

```text
{maxima_text}
```

大误差并非全部出现在低真实电压上。例如旧Seed42在{case42.target_age_h:g}h、Step{int(case42.horizon_step)}的真实电压约{case42.true_voltage_V:.3f}V、预测约{case42.predicted_voltage_V:.3f}V，说明相邻高低运行状态的判断也是问题。当前数据支持状态事件附近误差集中，不能仅凭这些记录认定某个模块导致了系统性提前或延后。

## 分区与工况证据

六种子分区均值（R²始终对应原始ΔV）：

```text
{parts.to_string()}
```

按Train预先固定阈值的事后工况组：

```text
{condition.to_string()}
```

这些组彼此重叠，SSE份额不能相加。low_voltage使用Train电压P05，high_transition使用Train转换次数P90，large_voltage_jump使用Train同segment相邻半小时电压跳变P95，any_zero_current为零电流占比>0。所有目标真实工况仅用于事后解释，没有进入预测器。

电压、电流密度、零电流占比和ΔV的分布统计与ECDF已输出。后期误差与状态切换/零电流混合/稀有电压跳变是否共同出现，应结合条件组、事件前后±24点和逐目标SSE表判断；不能凭相关性认定不可逆退化、门控失效或未来切换时间可从历史准确获知。P1最高SSE的两个分离事件按固定距离规则选择，仅作不利案例诊断，不作为选模依据。

## 审计边界

旧预测起点、目标、标签和原始三项指标均已严格复算一致，没有删除异常点，也未训练模型。数据跨度与合法窗口不完全均匀是已知数据资格问题，不等于行号错位。P0进一步通过运行时钟、日历时钟、segment及边界检查；没有据此发现预测时间错位或泄漏的证据。

P1统计不能单独证明F2架构造成后期误差，也不能证明更换损失必有收益。P2将用相同合法输入与目标从头训练普通LSTM作整体模型对照。

文件索引：P1_existing_F2/steps.csv、step_mean.csv、quarters.csv、local_10h.csv、tails.csv、largest_20_queries_per_seed.csv、target_SSE_contribution.csv、conditions.csv、overlap.csv、partition_distributions.csv、partition_metrics.csv；图在figures/P1/。P1的缓存只用于历史诊断，不填充本轮新模型结果。
''')


def comparison_figures(run,frame,windows):
    """P3: seed means are metric means, never ensembles or stitched windows."""
    paired=c.read_csv(run/'tables/PAIRED_DIFFERENCES.csv');means=c.read_csv(run/'tables/MODEL_MEAN_SD.csv')
    seeds=paired.seed.to_numpy();x=np.arange(len(seeds));width=.35
    for metric,ylabel,name in [('RMSE_mV','RMSE of raw DeltaV (mV)','01_six_seed_RMSE'),('R2','R² of raw DeltaV','02_six_seed_R2')]:
        fig,ax=plt.subplots(figsize=(10,4.5))
        for offset,kind in [(-.5,'LSTM'),(.5,'F2')]:
            ax.bar(x+offset*width,paired[kind+'_'+metric],width,label=kind,color=COLORS[kind],edgecolor='black',lw=.7,hatch='//' if kind=='LSTM' else None)
        ax.set_xticks(x,seeds);ax.set(xlabel='Predeclared seed',ylabel=ylabel,title='Fresh models: exploratory chronological holdout, 12648 queries / model')
        ax.legend();save(run,'P3/'+name,fig,paired,True)
    fig,ax=plt.subplots(figsize=(6,4.5));ordered=means.set_index('model').loc[['LSTM','F2']]
    ax.bar(['LSTM','F2'],ordered.RMSE_mV_mean,yerr=ordered.RMSE_mV_sample_sd,capsize=7,
        color=[COLORS[k] for k in ['LSTM','F2']],edgecolor='black',lw=.7)
    ax.set(ylabel='RMSE of raw DeltaV (mV)',title='Equal six-seed mean ± sample SD (not confidence interval)')
    save(run,'P3/03_mean_RMSE_SD',fig,means)
    steps=c.read_csv(run/'tables/STEP_METRICS.csv');fig,ax=plt.subplots(figsize=(9,4.5));step_plot=[]
    for kind,g in steps.groupby('model'):
        stat=g.groupby('step').RMSE_mV.agg(['mean','std']).reset_index();stat['model']=kind;step_plot.append(stat)
        ax.plot(stat.step,stat['mean'],label=kind,color=COLORS[kind],lw=2)
        ax.fill_between(stat.step,stat['mean']-stat['std'],stat['mean']+stat['std'],color=COLORS[kind],alpha=.12)
    ax.set(xlabel='Forecast step (0.5 h each; Step24 = 12 h)',ylabel='RMSE of raw DeltaV (mV)',title='Fresh models: equal-seed mean and descriptive between-seed SD')
    ax.legend();save(run,'P3/04_horizon_RMSE',fig,pd.concat(step_plot,ignore_index=True),True)
    points={kind:c.read_csv(run/f'models/formal/{kind}_seed202/predictions_TEST.csv') for kind in ['LSTM','F2']}
    fixed={kind:p.loc[p.horizon_step==24].copy() for kind,p in points.items()}
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,sharey=True)
    for ax,(kind,p) in zip(axes,fixed.items()):
        discontinuous(ax,p,'true_voltage_V','black','Observed',lw=1.2)
        discontinuous(ax,p,'predicted_voltage_V',COLORS[kind],kind+' Step24',lw=1.,alpha=.8)
        ax.set(ylabel='Absolute voltage (V)',title=kind+' Seed202: fixed Step24, 527 origins');ax.legend()
    axes[-1].set_xlabel('Actual target operating time (h)')
    save(run,'P3/05_absolute_voltage_Step24',fig,pd.concat([v.assign(model=k) for k,v in fixed.items()],ignore_index=True),True)
    extent=np.concatenate([v.true_voltage_V.to_numpy() for v in points.values()]+[v.predicted_voltage_V.to_numpy() for v in points.values()])
    low,high=float(extent.min()),float(extent.max());pad=(high-low)*.03
    for colored,name in [(False,'06_scatter_all_queries'),(True,'07_scatter_actual_current')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True,sharey=True)
        colour_limits=(float(points['F2'].target_actual_current_density_A_cm2.min()),float(points['F2'].target_actual_current_density_A_cm2.max()))
        for ax,(kind,p) in zip(axes,points.items()):
            options=dict(c=p.target_actual_current_density_A_cm2,cmap='viridis',vmin=colour_limits[0],vmax=colour_limits[1]) if colored else dict(color=COLORS[kind])
            artist=ax.scatter(p.true_voltage_V,p.predicted_voltage_V,s=5,alpha=.27,**options)
            ax.plot([low,high],[low,high],color='black',ls='--',lw=1);ax.set_xlim(low-pad,high+pad);ax.set_ylim(low-pad,high+pad)
            ax.set(xlabel='Observed absolute voltage (V)',ylabel='Predicted absolute voltage (V)',title=kind+' Seed202: all 12648 queries')
        if colored:
            fig.subplots_adjust(right=.85);bar=fig.add_axes([.89,.18,.02,.65]);fig.colorbar(artist,cax=bar,label='Actual target current density (A cm$^{-2}$); post-hoc only')
            # Dedicated colorbar axes need manual layout, not automatic panel resizing.
            fig.savefig(run/'figures/P3'/f'{name}.png',dpi=300,facecolor='white');fig.savefig(run/'figures/P3'/f'{name}.pdf',facecolor='white');plt.close(fig)
            csv(run/'figures/P3'/f'{name}_data.csv',pd.concat([v.assign(model=k) for k,v in points.items()],ignore_index=True))
            display(Image(filename=str(run/'figures/P3'/f'{name}.png')))
        else:save(run,'P3/'+name,fig,pd.concat([v.assign(model=k) for k,v in points.items()],ignore_index=True),True)
    target=c.read_csv(run/'formal_diagnostics/TARGET_ERRORS.csv');fig,ax=plt.subplots(figsize=(11,4.5));plot=[]
    for kind,g in target.groupby('model'):
        stat=g.groupby(['target_row','target_age_h']).mean_squared_error_mV2.mean().reset_index();stat['RMSE_mV']=np.sqrt(stat.mean_squared_error_mV2);stat['model']=kind;plot.append(stat)
        ax.scatter(stat.target_age_h,stat.RMSE_mV,s=7,alpha=.55,color=COLORS[kind],label=kind)
    ax.set(xlabel='Actual target operating time (h)',ylabel='RMSE across origins and seeds at each target (mV)',title='Fresh holdout errors: 619 target times; no temporal smoothing')
    ax.legend();save(run,'P3/08_error_over_time',fig,pd.concat(plot,ignore_index=True))
    events=c.read_csv(run/'P1_existing_F2/diagnostic_events.csv').to_dict('records')
    extreme_figure(run,frame,points,events,'P3/09_adverse_events_and_load')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True);history_plot=[]
    for ax,kind in zip(axes,['LSTM','F2']):
        history=c.read_csv(run/f'models/formal/{kind}_seed202/training_history.csv');history['model']=kind;history_plot.append(history)
        ax.plot(history.epoch,history.train_standardized_MSE,label='Train batch-mean MSE',color=COLORS[kind])
        ax.plot(history.epoch,history.validation_standardized_MSE,label='Full Validation MSE',color='black',ls='--')
        ax.axvline(history.best_epoch.iloc[-1],color='gray',ls=':',label='Selected best epoch')
        ax.set_yscale('log');ax.set(xlabel='Actual training epoch',ylabel='Standardized DeltaV MSE (log scale)',title=kind+' Seed202: new logs');ax.legend(fontsize=8)
    save(run,'P3/10_training_loss',fig,pd.concat(history_plot,ignore_index=True),True)
    origin=int(windows['TEST'].origins[len(windows['TEST'].origins)//2]);fig,ax=plt.subplots(figsize=(9,4.5));typical=[]
    for kind,p in points.items():
        g=p.loc[p.origin_row==origin];typical.append(g.assign(model=kind))
        if kind=='LSTM':ax.plot(g.horizon_step*.5,g.true_voltage_V,color='black',marker='.',label='Observed')
        ax.plot(g.horizon_step*.5,g.predicted_voltage_V,color=COLORS[kind],ls='--',label=kind)
    ax.set(xlabel='Forecast lead (h)',ylabel='Absolute voltage (V)',title=f'Seed202, predeclared median legal origin row{origin}; no window selection by error')
    ax.legend();save(run,'P3/11_fixed_origin_24step',fig,pd.concat(typical,ignore_index=True),True)

    # Explicitly expose each model's worst query, using complete paired windows.
    # The Step24 context plot above is not a plot of every horizon's maximum error.
    maxima=c.read_csv(run/'formal_diagnostics/MAX_ERROR_CASES.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharex=True,sharey=True);worst_plot=[]
    for ax,owner in zip(axes,['LSTM','F2']):
        case=maxima.loc[(maxima.model==owner)&(maxima.seed==202)].iloc[0]
        origin=int(case.origin_row)
        for kind,p in points.items():
            g=p.loc[p.origin_row==origin].copy();g['model']=kind;g['selected_for']=owner
            g['selected_target_row']=int(case.target_row);g['selection_rule']='Seed202 largest absolute query error of each model'
            worst_plot.append(g)
            if kind=='LSTM':ax.plot(g.horizon_step*.5,g.true_voltage_V,color='black',marker='.',label='Observed')
            ax.plot(g.horizon_step*.5,g.predicted_voltage_V,color=COLORS[kind],ls='--',marker='.',ms=3,label=kind)
        ax.axvline(case.horizon_step*.5,color='gray',ls=':',lw=1)
        ax.set(xlabel='Forecast lead (h)',ylabel='Absolute voltage (V)',
            title=f"Selected for {owner}: origin {origin}, {case.origin_age_h:g} h\nWorst query Step{int(case.horizon_step)}: {case.absolute_error_mV:.2f} mV")
        ax.set_ylim(low-pad,high+pad);ax.legend(fontsize=9)
    save(run,'P3/12_paired_worst_query_windows',fig,pd.concat(worst_plot,ignore_index=True),True)

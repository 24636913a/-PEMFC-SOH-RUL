"""Build tables, publication figures 01-05 and evidence-bounded Chinese reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import STUDY,OLD,OUT,load_cfg,load_cache,query_truth,sha

FIG=STUDY/'figures'
COLORS={'Persistence':'#767676','LSTM':'#B64342','F2':'#0F4D92','F2_EventWeighted':'#42949E'}
LABELS={'Persistence':'Persistence','LSTM':'LSTM','F2':'Native F2','F2_EventWeighted':'Event-weighted F2'}
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,'axes.linewidth':1.5,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42})

def write_report(pattern,text):
    (STUDY/pattern).write_text(text.rstrip()+'\n',encoding='utf-8',newline='\n')

def reload_disclosure():
    return ('独立复现保留了一个未通过的CPU后端探针：A2完整CPU重载曾超过原定1e-5V容差，'
        '该次完整最大差未序列化，记录为null，不能伪造为0；另行固定前64个起点诊断的最大差为'
        '1.4508604783e-5V（约0.0145mV），它不是完整集最大差。原因尚未被唯一确定，不能只凭推测归因。'
        '未放宽该容差、未修改已保存预测、未重新训练。随后在训练间隙以原GPU环境、字节一致的float32输入、'
        '相同batch16和CuDNN开关独立重放全部60秒及600秒网格，采用更严格的1e-7V判据；'
        '实际每个模型的全量最大差、SHA身份与执行记录见outputs/review_checkpoint_gpu_reloads.json。'
        '重放仅推理，0次反向传播和优化器更新，不计为新训练。本轮只验证了记录GPU环境中的checkpoint复现，'
        '没有证明CPU跨后端逐值一致。失败与后续核验均保留于outputs/review_checkpoint_cpu_probe.json及独立验收文件。')

def publish(fig,name,data):
    FIG.mkdir(parents=True,exist_ok=True);fig.tight_layout(pad=1.5)
    fig.savefig(FIG/f'{name}.png',dpi=300,facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',facecolor='white')
    data.to_csv(FIG/f'{name}_plot_data.csv',index=False,lineterminator='\n');plt.close(fig)

def collect():
    frames=[];states=[];runs={}
    for directory in sorted((STUDY/'runs').glob('*')):
        state=directory/'training_state.json'
        if state.exists():
            d=json.loads(state.read_text(encoding='utf-8'))
            if d['status']!='COMPLETE':raise RuntimeError('Incomplete run '+str(directory))
            states.append(d);runs[d['id']]=directory
        if (directory/'metrics.csv').exists():
            t=pd.read_csv(directory/'metrics.csv');t['run']=directory.name;frames.append(t)
    df=pd.concat(frames,ignore_index=True)
    df.to_csv(OUT/'all_metrics.csv',index=False)
    return df,states,runs

def main_table(df,grid='60s'):
    selected=df[(df.grid==grid)&(df.scope=='ALL_QUERIES')]
    records=[]
    for (h,model),sub in selected.groupby(['horizon_seconds','model'],sort=False):
        full=sub[sub.stratum=='ALL'].iloc[0];event=sub[sub.stratum=='EVENT'].iloc[0];normal=sub[sub.stratum=='NORMAL'].iloc[0]
        records.append(dict(horizon_seconds=h,model=model,origins=int(full.origins),queries=int(full.queries),
            RMSE_mV=full.RMSE_V*1000,MAE_mV=full.MAE_V*1000,R2_delta=full.R2_delta,
            event_RMSE_mV=event.RMSE_V*1000,normal_MAE_mV=normal.MAE_V*1000,
            P95_mV=full.P95_abs_V*1000,P99_mV=full.P99_abs_V*1000))
    return pd.DataFrame(records).sort_values(['horizon_seconds','model'])

def markdown_table(table):
    cols=['horizon_seconds','model','RMSE_mV','MAE_mV','R2_delta','event_RMSE_mV','normal_MAE_mV','P95_mV','P99_mV']
    names=['跨度(s)','模型','全RMSE(mV)','全MAE(mV)','ΔV R²','事件RMSE(mV)','普通MAE(mV)','P95(mV)','P99(mV)']
    lines=['| '+' | '.join(names)+' |','|'+'---|'*len(cols)]
    for _,row in table.iterrows():
        display={'Persistence':'Persistence（保持法）','LSTM':'LSTM','F2':'原生F2','F2_EventWeighted':'事件加权F2'}
        values=[str(int(row[c])) if c=='horizon_seconds' else display.get(str(row[c]),str(row[c])) if c=='model' else f'{row[c]:.6f}' for c in cols]
        lines.append('| '+' | '.join(values)+' |')
    if (120 in table.horizon_seconds.values and
        len(table[(table.horizon_seconds==30)&(table.model=='F2_EventWeighted')]) and
        not len(table[(table.horizon_seconds==120)&(table.model=='F2_EventWeighted')])):
        lines.append('| 120 | 事件加权F2 | 未执行：B1未达门槛 | — | — | — | — | — | — |')
    return '\n'.join(lines)

def figures(df,states,runs,a):
    table=main_table(df);main=df[(df.grid=='60s')&(df.scope=='ALL_QUERIES')]
    gate_path=OUT/'B1_acceptance.json'
    gate=json.loads(gate_path.read_text(encoding='utf-8')) if gate_path.exists() else {}
    b2_note='B2 not run: B1 preset gate failed' if gate.get('pass') and not gate.get('B2_allowed') else 'B2: not run unless B1 passes'
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,h in zip(axes,[30,120]):
        sub=table[table.horizon_seconds==h]
        models=[m for m in COLORS if m in sub.model.values]
        vals=[float(sub[sub.model==m].RMSE_mV.iloc[0]) for m in models]
        bars=ax.bar(np.arange(len(models)),vals,color=[COLORS[m] for m in models],edgecolor='black',linewidth=.8)
        ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_xticks(range(len(models)),[LABELS[m].replace(' ','\n') for m in models])
        ax.set(title=f'All validation queries: H={h}s',ylabel='Voltage RMSE (mV)');ax.set_ylim(0,max(vals)*1.19)
        if h==120 and 'F2_EventWeighted' not in models:ax.text(.02,.95,b2_note,transform=ax.transAxes,va='top',fontsize=9)
    publish(fig,'01_overall_validation_RMSE',table)
    leads=[];predictions={}
    for name,run in runs.items():
        z=np.load(run/'predictions_VALIDATION_60s.npz');state=next(x for x in states if x['id']==name)
        model=state['model'] if not name.startswith('B') else 'F2_EventWeighted';h=state['horizon'];predictions[name]=z
        err=z['prediction_delta_V']-z['truth_delta_V']
        for lead in range(h):leads.append(dict(run=name,model=model,horizon_seconds=h,lead_seconds=lead+1,
            RMSE_mV=float(np.sqrt(np.mean(err[:,lead]**2))*1000),MAE_mV=float(np.mean(np.abs(err[:,lead]))*1000)))
    for h in [30,120]:
        z=np.load(STUDY/f'runs/Persistence_H{h}/predictions_VALIDATION_60s.npz');e=-z['truth_delta_V']
        for lead in range(h):leads.append(dict(run='Persistence',model='Persistence',horizon_seconds=h,lead_seconds=lead+1,
            RMSE_mV=float(np.sqrt(np.mean(e[:,lead]**2))*1000),MAE_mV=float(np.mean(np.abs(e[:,lead]))*1000)))
    lead_df=pd.DataFrame(leads);fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,h in zip(axes,[30,120]):
        for model in COLORS:
            d=lead_df[(lead_df.horizon_seconds==h)&(lead_df.model==model)]
            if len(d):ax.plot(d.lead_seconds,d.RMSE_mV,label=LABELS[model],color=COLORS[model],linewidth=2)
        ax.set(title=f'H={h}s; every forecast lead',xlabel='Actual future step (~s)',ylabel='RMSE (mV)');ax.legend(fontsize=9)
    publish(fig,'02_error_by_forecast_lead',lead_df)
    fig,axes=plt.subplots(2,2,figsize=(12,8));strata=[]
    for row,h in enumerate([30,120]):
        for col,(stratum,field,unit) in enumerate([('EVENT','RMSE_V','RMSE'),('NORMAL','MAE_V','MAE')]):
            sub=main[(main.horizon_seconds==h)&(main.stratum==stratum)]
            models=[m for m in COLORS if m in sub.model.values];values=[float(sub[sub.model==m][field].iloc[0])*1000 for m in models]
            ax=axes[row,col];bars=ax.bar(range(len(models)),values,color=[COLORS[m] for m in models],edgecolor='black',linewidth=.8)
            ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_xticks(range(len(models)),[LABELS[m].replace(' ','\n') for m in models],fontsize=10)
            ax.set(title=f'H={h}s; {stratum}',ylabel=unit+' (mV)');ax.set_ylim(0,max(values)*1.2)
            for m,val in zip(models,values):strata.append(dict(horizon=h,model=m,stratum=stratum,metric=unit,value_mV=val))
    publish(fig,'03_event_and_normal_error',pd.DataFrame(strata))
    histories=[];fig,axes=plt.subplots(3,2,figsize=(12,11),squeeze=False)
    for ax,state in zip(axes.ravel(),states):
        run=runs[state['id']];d=pd.read_csv(run/'training_history.csv');d['run']=state['id'];histories.append(d)
        train_label='Train weighted objective' if state['id'].startswith('B') else 'Train unweighted MSE'
        ax.plot(d.epoch,d.train_loss_standardized,label=train_label,color='#0F4D92')
        ax.plot(d.epoch,d.validation_loss_standardized,label='Validation unweighted MSE',color='#B64342')
        ax.axvline(state['best_epoch'],color='#767676',linestyle='--',linewidth=1)
        ax.set(title=f'{state["id"]}: {state["model"]}, H={state["horizon"]}s',xlabel='Epoch',ylabel='Standardized loss')
        ax.legend(fontsize=9)
    for ax in axes.ravel()[len(states):]:ax.axis('off');ax.text(.1,.5,b2_note.replace(': ','\n'),transform=ax.transAxes)
    publish(fig,'04_training_validation_loss',pd.concat(histories,ignore_index=True))
    # First chronological eligible event used by predeclared diagnostics, never lowest-error cases.
    cases=pd.read_csv(OUT/'diagnostics/predeclared_branch_response_events.csv');cases=cases[cases.partition=='VALIDATION']
    event=cases.iloc[0];fig,axes=plt.subplots(1,2,figsize=(13,4.7));plot=[]
    for ax,h in zip(axes,[30,120]):
        template=predictions['A1' if h==30 else 'A3'];origins=template['origins'];eligible=np.flatnonzero((origins<int(event.first_row))&(origins+h>=int(event.first_row)))
        previous=np.flatnonzero(origins<int(event.first_row))
        ix=int(eligible[-1]) if len(eligible) else int(previous[-1]) if len(previous) else 0
        origin=int(origins[ix]);base=float(a['voltage_V'][origin]);lead=np.arange(1,h+1)
        truth=base+template['truth_delta_V'][ix];ax.plot(lead,truth,color='black',label='Raw measured voltage',linewidth=2.1)
        ax.axhline(base,color=COLORS['Persistence'],linestyle=':',label='Persistence')
        for name,z in predictions.items():
            state=next(x for x in states if x['id']==name)
            if state['horizon']!=h:continue
            model='F2_EventWeighted' if name.startswith('B') else state['model'];assert int(z['origins'][ix])==origin
            p=base+z['prediction_delta_V'][ix];ax.plot(lead,p,color=COLORS[model],label=LABELS[model],linewidth=1.6)
            for j in range(h):plot.append(dict(run=name,horizon_seconds=h,origin_row=origin,origin_time_ms=int(z['origin_time_ms'][ix]),
                target_row=int(z['targets'][ix,j]),target_time_ms=int(z['target_time_ms'][ix,j]),lead_seconds=j+1,truth_V=float(truth[j]),prediction_V=float(p[j]),
                persistence_V=base,event_id=int(event.event_id),trigger_inside_horizon=bool(len(eligible)),
                selection='first_predeclared_validation_episode; latest_grid_origin_before_trigger; no_error_selection'))
        subtitle='' if len(eligible) else '\nTrigger is outside this forecast horizon'
        ax.set(title=f'Predeclared event {int(event.event_id)}, H={h}s'+subtitle,xlabel='Forecast lead (~s)',ylabel='Ten-cell mean voltage (V)');ax.legend(fontsize=8)
    publish(fig,'05_real_voltage_and_predictions',pd.DataFrame(plot))

def reports(df,states,runs,native_only=False):
    main=main_table(df);main.to_csv(OUT/'validation_main_results.csv',index=False)
    count_records=[]
    for d in states:
        row={k:d[k] for k in ['id','model','horizon','parameters','stop_epoch','best_epoch','optimizer_steps',
            'training_and_evaluation_wall_seconds','training_cuda_seconds','GPU_peak_reserved_MiB','initial_state_sha256','checkpoint_sha256']}
        hist=pd.read_csv(runs[d['id']]/'training_history.csv')
        row['selected_checkpoint_optimizer_steps']=int(hist.loc[hist.epoch==d['best_epoch'],'optimizer_steps'].iloc[0])
        row['reached_epoch_cap']=d['stop_epoch']==50
        row['best_in_last_three_budget_epochs']=d['stop_epoch']==50 and d['best_epoch']>=48
        tail=hist.tail(3)
        row['last_three_validation_RMSE_slope_V_per_epoch']=float(np.polyfit(tail.epoch,tail.validation_RMSE_V,1)[0]) if len(tail)>1 else None
        count_records.append(row)
    counts=pd.DataFrame(count_records)
    counts.to_csv(OUT/'training_summary.csv',index=False)
    changes=[]
    for h,rid in [(30,'A2'),(120,'A4')]:
        old=pd.read_csv(OUT/f'historical_same_queries_H{h}.csv')
        old=old[(old.model=='Historical_F2')&(old.scope=='ALL_QUERIES')&(old.stratum=='ALL')].iloc[0]
        new=df[(df.model=='F2')&(df.horizon_seconds==h)&(df.grid=='600s')&(df.scope=='ALL_QUERIES')&(df.stratum=='ALL')].iloc[0]
        old_strata=pd.read_csv(OUT/f'historical_same_queries_H{h}.csv')
        old_strata=old_strata[(old_strata.model=='Historical_F2')&(old_strata.scope=='ALL_QUERIES')].set_index('stratum')
        new_strata=df[(df.model=='F2')&(df.horizon_seconds==h)&(df.grid=='600s')&(df.scope=='ALL_QUERIES')].set_index('stratum')
        changes.append(dict(horizon=h,old_RMSE_mV=old.RMSE_V*1000,new_RMSE_mV=new.RMSE_V*1000,
            RMSE_relative_change_percent=(new.RMSE_V/old.RMSE_V-1)*100,old_R2_delta=old.R2_delta,new_R2_delta=new.R2_delta,
            old_event_RMSE_mV=old_strata.loc['EVENT','RMSE_V']*1000,new_event_RMSE_mV=new_strata.loc['EVENT','RMSE_V']*1000,
            old_normal_MAE_mV=old_strata.loc['NORMAL','MAE_V']*1000,new_normal_MAE_mV=new_strata.loc['NORMAL','MAE_V']*1000,
            same_origin_count=int(new.origins),same_query_count=int(new.queries)))
    changes=pd.DataFrame(changes);changes.to_csv(OUT/'F2_budget_sampling_change_same600s.csv',index=False)
    detail='\n'.join(f'- {int(r.horizon)}秒同{int(r.same_origin_count)}个600秒起点：旧F2 {r.old_RMSE_mV:.6f}mV，新F2 {r.new_RMSE_mV:.6f}mV，RMSE'+
        ('降低' if r.RMSE_relative_change_percent<0 else '升高（变差）')+f'{abs(r.RMSE_relative_change_percent):.3f}%；ΔV R² {r.old_R2_delta:.6f}→{r.new_R2_delta:.6f}。事件RMSE {r.old_event_RMSE_mV:.3f}→{r.new_event_RMSE_mV:.3f}mV，普通MAE {r.old_normal_MAE_mV:.3f}→{r.new_normal_MAE_mV:.3f}mV。' for r in changes.itertuples())
    train_lines='\n'.join(f'- {d["id"]} {d["model"]} H{d["horizon"]}：停止{d["stop_epoch"]}轮，最佳{d["best_epoch"]}轮，共{d["optimizer_steps"]}次真实更新；最佳checkpoint在第{int(counts.loc[counts.id==d["id"],"selected_checkpoint_optimizer_steps"].iloc[0])}次更新保存，预留显存峰值{d["GPU_peak_reserved_MiB"]:.1f}MiB。'+
        ('达到50轮上限，未追加预算；'+('末端仍取得新低，预算可能限制，尚未证明平台。' if d['best_epoch']>=48 else '最佳轮较早，不能据此承诺追加轮数有益。') if d['stop_epoch']==50 else '按连续10轮无改善规则早停。') for d in states if d['id'].startswith('A'))
    native_table=main[main.model!='F2_EventWeighted']
    text=f'''# 原生F2充分训练结果

本标题表示执行了升级的固定训练预算，不等于已经证明充分收敛。8192个共同Train起点、最大50轮、patience10；A1–A4均为Seed202从头训练，普通LSTM与完整原生F2共享输入、起点、归一化、目标、优化设置和评价口径，参数量并不相等。

## 主Validation：60秒起点网格

{markdown_table(native_table)}

所有起点×全部提前量均参与主指标；没有把普通区间替换全样本。R²是ΔV决定系数。两个跨度是不同预测任务，不能用30秒成绩对120秒或旧12小时成绩证明提高。

## 相同600秒查询上的新旧F2比较

{detail}

这是“训练数据覆盖＋训练预算”的组合变化，不是分别识别8192个起点或50轮的因果效应。原研究1125个Validation起点中有1个因新完整信息支持同分区规则不合法，新旧均只评价剩余1124个；不能直接用旧全部1125起点表或新60秒网格声称改善。

## 实际运行与收敛

{train_lines}

完整逐轮Train目标、未加权Validation MSE/RMSE/MAE、CUDA计时间隔、wall时间及显存见各run的training_history.csv。Train主动补充事件，Validation是完整无标签筛选网格；训练模式还含原生dropout，因此两条Loss的绝对高度不能直接当作同分布过拟合证据。B模型的Train目标另有权重，图中特别标明，Validation始终未加权。训练梯度日志固定采样每轮首/末更新，不声称记录每一次梯度张量。branch_parameter_updates.json对最佳checkpoint逐分组核对初始化后的参数位移；初始化参数SHA表和完整初始化SHA另存。最佳轮接近预算上限仅说明预算仍可能限制优化，不保证继续训练会改善泛化；最佳轮明显早于停止轮则需先考虑泛化与采样。

## 限制与披露

{reload_disclosure()}

一次Seed202、共同宽度但不同参数量、固定采样和有限工况不支持统计显著性、稳定架构优势或充分收敛断言。本轮不使用后留结果选模型、不加模块、不重新搜索种子。AI辅助实现；证据权威为outputs表和runs真实日志、checkpoint及逐目标预测。
'''
    write_report('02_原生F2充分训练结果.md',text)
    if native_only:return
    gate=json.loads((OUT/'B1_acceptance.json').read_text(encoding='utf-8'))
    b2=any(d['id']=='B2' for d in states)
    assert b2==bool(gate['B2_allowed']),'Final report requires exactly the authorized conditional B2 decision'
    train_origins=pd.read_csv(OUT/'selected_train_origins.csv').origin.to_numpy(np.int64)
    _,train_arrays=load_cache();composition=[];composition_lines=[]
    for horizon in [30,120]:
        _,truth,event=query_truth(train_arrays,train_origins,horizon,load_cfg())
        event_count=int(event.sum());total=int(event.size);weighted_executed=horizon==30 or b2
        share=3*event_count/(total+2*event_count)
        composition.append(dict(horizon_seconds=horizon,origins=len(train_origins),queries=total,
            event_queries=event_count,event_query_fraction=event_count/total,
            weighted_trial_executed=weighted_executed,
            event_share_of_loss_weight=share if weighted_executed else None))
        composition_lines.append(f'- {horizon}秒：{total}个Train目标查询，其中事件{event_count}个（{event_count/total*100:.3f}%）；'+
            (f'固定1:3下事件占总损失权重{share*100:.3f}%。' if weighted_executed else 'B2未执行，不报告其实际加权训练效果。'))
    pd.DataFrame(composition).to_csv(OUT/'training_target_event_weight_composition.csv',index=False)
    ordinary=df[(df.grid=='60s')&(df.scope=='ALL_QUERIES')&(df.stratum=='NORMAL')&df.model.isin(['F2','F2_EventWeighted'])]
    ordinary_lines=['| 跨度(s) | 模型 | 普通RMSE(mV) | 普通MAE(mV) |','|---|---|---|---|']
    for row in ordinary.sort_values(['horizon_seconds','model']).itertuples():
        ordinary_lines.append(f'| {row.horizon_seconds} | '+('事件加权F2' if row.model=='F2_EventWeighted' else '原生F2')+
            f' | {row.RMSE_V*1000:.6f} | {row.MAE_V*1000:.6f} |')
    write_report('03_事件敏感训练结果.md',f'''# 事件敏感训练结果

B1只把A2的训练目标换成固定事件权重：普通1、事件3，整个有效batch计算sum(w*SE)/sum(w)。两者初始化SHA、训练起点、样本顺序、架构、优化设置和最大预算相同；早停依各自未加权全Validation RMSE决定，不能要求实际停止轮相同。标签仅作监督，不进入预测输入。权重3来自用户预设，文献未证明最优，详见notes/literature_support.md。

## 固定训练目标组成

{chr(10).join(composition_lines)}

以上总权重占比是全体已选Train查询的描述，不代表每个有效batch的相同占比，也不是实际梯度贡献比例。Train事件补充和损失加权叠加；Validation保持未加权完整网格。详见outputs/training_target_event_weight_composition.csv。

## 实测对照

{markdown_table(main[(main.horizon_seconds==30)&main.model.isin(['F2','F2_EventWeighted'])])}

普通区间直接对照如下，不替代上面的全样本结果：

{chr(10).join(ordinary_lines)}

## 独立准入判定

独立核验完整记录如下（数值门槛用未舍入结果）：

`pass=true`仅表示数据、实现与指标完整性核验通过，不表示性能门槛通过；是否允许B2由`B2_allowed`决定。本轮该字段为{str(gate['B2_allowed']).lower()}，事件RMSE降幅{gate['relative_event_RMSE_reduction']*100:.6f}%{'达到' if gate['criteria']['event_RMSE_reduction_at_least_5pct'] else '未达到'}5%。

其中reviewer_GPU_calls=0仅表示下面准入公式的计算不调用GPU，不表示整次核验没有GPU重载。完整checkpoint重载的真实GPU推理批数见review_checkpoint2.json和review_checkpoint_gpu_reloads.json，均为0次优化器更新。

```json
{json.dumps(gate,ensure_ascii=False,indent=2)}
```

120秒事件加权B2：{'已通过B1全部条件后执行一次，结果列于总表。' if b2 else '未执行；B1未满足全部预设条件，按协议取消，不尝试其他权重或损失。'}

{markdown_table(main[(main.horizon_seconds==120)&main.model.isin(['F2','F2_EventWeighted'])])}

每提前量误差、P95/P99、普通与事件查询及物理事件代理统计全部保留，见outputs/all_metrics.csv、figures/02和09及outputs/diagnostics。事件定向采样与损失加权的影响叠加，普通区间代价不能省略。无论B1成功或失败，权重、日志和全部预测都保留。

## 限制与披露

这是单种子条件性探索，不是方法稳定优于原生F2的结论。未执行的B2没有任何伪造指标。AI辅助实现和核验；真实数值以CSV/NPZ、checkpoint和独立验收为准。
''')
    diag_dir=OUT/'diagnostics'
    tail=pd.read_csv(diag_dir/'extreme_error_summary.csv').set_index('run')
    tail_description=(f'30秒加权前后，最大1%查询的绝对平方误差和从{tail.loc["A2","top_1pct_SSE_V2"]:.6f}降至'
        f'{tail.loc["B1","top_1pct_SSE_V2"]:.6f}V²，但其占全部平方误差比例从'
        f'{tail.loc["A2","top_1pct_SSE_fraction"]*100:.3f}%升至{tail.loc["B1","top_1pct_SSE_fraction"]*100:.3f}%。'
        '这是总误差也下降后产生的占比变化，不能把占比升高误写成极端误差绝对值升高；两种量均须保留。')
    summaries=[]
    for filename in ['extreme_error_summary.csv','error_by_event_category.csv','error_by_current_condition.csv','precursor_fixed_grid_summary.csv','precursor_event_alignment_summary.csv']:
        p=diag_dir/filename
        if p.exists() and p.stat().st_size>2:summaries.append('### '+filename+'\n\n```csv\n'+p.read_text(encoding='utf-8')+'```')
    write_report('04_完整误差分析与图表.md',f'''# 完整误差分析与图表

## 统一主结果

{markdown_table(main)}

完整ALL/EVENT/NORMAL、60s/600s、全查询/去重目标秒表见outputs/all_metrics.csv。同跨度所有模型使用同一评价集合，不删除突变；H30/H120合法起点数的微小差异已在00说明，不能跨任务混比。120秒重叠查询不是独立重复；辅助去重固定保留最新origin的一条真实query，改变提前量分布，不替代主表。

## 事件、工况、先兆和尾部贡献

事件阈值、quiet-gap120秒合并和先兆规则在新训练前冻结。零流仅指|I|≤0.1A观测代理，不等于硬件关机。分类优先带载→低/零流、恢复带载、电流/命令变化、温压变化、未超过已定义观测变化阈值；±30秒分类是事后解释，不作为预测输入。

本轮供气相关描述限于冻结输入中的阳/阴极入口压力、温度与运行模式，未建立所有气体流量、湿度或化学计量变化的独立真值。分类有先后优先级：没有episode最终归入温压类，不代表温压或供气没有变化、没有影响；也不能把该代理类当作全部气路状态已被识别。为避免看完误差后改规则，本轮不新增事后分类阈值。

每个预测查询最多归属一个最近已发生触发的episode；普通查询单独保留，不能把误差全硬归给物理事件。最大1%按全部查询平方误差排序，报告占全部SSE比例；不同起点命中同事件不算多个独立实验。

{tail_description}

先兆在Train固定60秒诊断网格拟合99.5%阈值，Validation原样检验。−120/−60/−30秒是发生前的历史线索，0秒是发生当时，+30秒是事后确认；后两者不能倒灌进预测输入。该诊断网格为所有完整120秒未来合法起点，与H30主评价末端合法性略有不同，不能偷偷混用计数。

{chr(10).join(summaries)}

## 门控及两条路径

outputs/diagnostics/*_gate_and_paths.json报告全部主Validation起点的门控分布；*_native_path_response.csv仅展示事先声明的首3个合法Train及首3个合法Validation事件，各7个时间快照，未按误差挑选。LSTM与CNN隐表示的RMS、夹角与变化不是各自独立电压预测贡献，也不是因果归因。门控有变化、不饱和只能排除简单失活，不能证明工况权重正确。

## 已暴露历史案例的边界

1873–1877h与2149–2154h只沿用旧研究01/02报告的解释性发现：低均值聚合桶混有5006条零流与1804条非零流记录，OpHrs2冻结会让一个运行小时桶跨越远大于30分钟的日历区间。这说明时钟与混合状态能解释部分旧大误差，不证明所有极端误差都是聚合造成。本轮未对这些后留区间新增模型推理，也未用它们制定采样、损失、先兆规则或模型超参数；不能将此类已知案例称作新盲测。

## 图像目录

01_overall_validation_RMSE：两跨度全查询RMSE；02_error_by_forecast_lead：每个提前量；03_event_and_normal_error：事件与普通代价；04_training_validation_loss：真实训练验证曲线；05_real_voltage_and_predictions：预声明案例真实与预测电压；06_current_voltage_sync：电流/电压同步；07_F2_gate_and_path_response：门控与隐路径；08_extreme_error_contribution：尾部贡献；09_event_weighted_comparison：事件加权前后；10_training_coverage：覆盖与事件重复。

每图位于figures/，保存300dpi PNG、PDF和同stem_plot_data.csv。图中没有用好看的区间替代全样本数值。全部原始目标DateTime和行号在各模型NPZ。

## 局限与披露

{reload_disclosure()}

相邻记录可能反映共同测量开关；无已知未来控制计划时，历史关联不保证能确定切换时刻。事件episode是规则代理，不是独立硬件真值。单种子、重叠评价、已暴露历史数据与固定网格限制泛化解释。AI辅助实现；图表直接来自保存的真实数据和模型输出。
''')
    answers=[]
    answers.append('1. 增加起点和预算后，F2提高多少？必须在相同600秒查询上看：\n\n'+detail+'\n\n数据与预算一起改变，不能拆分归因。')
    for question,h in [(2,30),(3,120)]:
        sub=main[(main.horizon_seconds==h)&main.model.isin(['LSTM','F2'])].set_index('model')
        f=float(sub.loc['F2','RMSE_mV']);l=float(sub.loc['LSTM','RMSE_mV'])
        answer=f'{question}. F2在{h}秒是否超过LSTM？原生F2 {f:.6f}mV，LSTM {l:.6f}mV；本轮全样本RMSE'+('F2更低。' if f<l else 'LSTM更低，不能宣称原生F2超过。')
        answer+=f' 事件RMSE（F2/LSTM）为{sub.loc["F2","event_RMSE_mV"]:.3f}/{sub.loc["LSTM","event_RMSE_mV"]:.3f}mV，全MAE为{sub.loc["F2","MAE_mV"]:.3f}/{sub.loc["LSTM","MAE_mV"]:.3f}mV；全RMSE较低不等于各项、各工况都更好。'
        weighted_row=main[(main.horizon_seconds==h)&(main.model=='F2_EventWeighted')]
        if len(weighted_row):
            score=float(weighted_row.RMSE_mV.iloc[0]);answer+=f' 事件加权F2为{score:.6f}mV，'+('低于' if score<l else '未低于')+'LSTM；仍只是本次单种子证据。'
        answers.append(answer)
    best=float(main.RMSE_mV.min());answers.append(f'4. 全样本RMSE是否低于10mV？本轮主Validation最低为{best:.6f}mV，'+('低于10mV，但仅限这次冻结实验。' if best<10 else '未达到。普通区间的低误差不能冒充全样本。'))
    answers.append('5. R²是否提高？以相同600秒查询的ΔV R²对照为准：\n\n'+'\n'.join(f'- {int(r.horizon)}秒：{r.old_R2_delta:.6f}→{r.new_R2_delta:.6f}。' for r in changes.itertuples())+'\n\n主60秒网格R²另见总表，二者不混比。')
    a2=main[(main.horizon_seconds==30)&(main.model=='F2')].iloc[0];b1=main[(main.horizon_seconds==30)&(main.model=='F2_EventWeighted')].iloc[0]
    answers.append(f'6. 事件加权是否减少极端误差？30秒事件RMSE {a2.event_RMSE_mV:.6f}→{b1.event_RMSE_mV:.6f}mV；'+('下降不代表所有门槛都通过，' if b1.event_RMSE_mV<a2.event_RMSE_mV else '本轮并未降低这一指标，')+'B1最终准入以独立判定为准。P99和最大1%贡献见04，不能只选一个有利尾部指标。')
    answers[-1]+=f' 全样本P95 {a2.P95_mV:.6f}→{b1.P95_mV:.6f}mV，P99 {a2.P99_mV:.6f}→{b1.P99_mV:.6f}mV，尾部并非必然同步改善。全RMSE相对下降{gate["relative_all_RMSE_reduction"]*100:.3f}%，事件RMSE相对下降{gate["relative_event_RMSE_reduction"]*100:.3f}%；'+('全部门槛通过，才开展B2。' if b2 else '未同时通过全部预设门槛，因此取消B2，而不是改门槛或换权重。')
    event_tail=df[(df.horizon_seconds==30)&(df.grid=='60s')&(df.scope=='ALL_QUERIES')&(df.stratum=='EVENT')].set_index('model')
    answers[-1]+=f' 仅事件子集的P99为{event_tail.loc["F2","P99_abs_V"]*1000:.6f}→{event_tail.loc["F2_EventWeighted","P99_abs_V"]*1000:.6f}mV；它与全体查询P99的统计集合不同，不能互换。'
    answers[-1]+=' '+tail_description
    answers.append(f'7. 普通时刻误差增加了吗？30秒普通MAE {a2.normal_MAE_mV:.6f}→{b1.normal_MAE_mV:.6f}mV，相对变化{(b1.normal_MAE_mV/a2.normal_MAE_mV-1)*100:+.3f}%。'+('本轮普通MAE也下降，未观察到以该指标变差来换取全RMSE收益。' if b1.normal_MAE_mV<=a2.normal_MAE_mV else '这项普通区间代价不能用全RMSE改善掩盖。'))
    if b2:
        a4=main[(main.horizon_seconds==120)&(main.model=='F2')].iloc[0];b=main[(main.horizon_seconds==120)&(main.model=='F2_EventWeighted')].iloc[0]
        answers[-2]+=f' 120秒事件RMSE为{a4.event_RMSE_mV:.6f}→{b.event_RMSE_mV:.6f}mV，'+('降低。' if b.event_RMSE_mV<a4.event_RMSE_mV else '未降低。')
        answers[-1]+=f' 120秒普通MAE为{a4.normal_MAE_mV:.6f}→{b.normal_MAE_mV:.6f}mV，相对变化{(b.normal_MAE_mV/a4.normal_MAE_mV-1)*100:+.3f}%。'
    category=diag_dir/'error_by_event_category.csv'
    category_names={'loaded_to_low_or_zero':'带载进入低/零流','low_or_zero_to_loaded':'低/零流恢复带载',
        'current_or_command_change':'电流/命令变化','temperature_or_pressure_change':'温度/压力变化',
        'no_above_threshold_observed_control_change':'未触发已定义观测控制变化规则'}
    assert category.exists() and category.stat().st_size>2,'Final category diagnosis required'
    cat=pd.read_csv(category);sentences=[]
    for h,rid in [(30,'A2'),(120,'A4'),(30,'B1')]:
        rows=cat[(cat.run==rid)&(cat.episodes>0)]
        worst=rows.loc[rows.RMSE_V.idxmax()];largest=rows.loc[rows.SSE_V2.idxmax()]
        model_label='事件加权F2' if rid=='B1' else '原生F2'
        sentences.append(f'{h}秒{model_label}：观测到类别RMSE最高的是{category_names.get(worst.category,worst.category)}，{worst.RMSE_V*1000:.3f}mV，覆盖{int(worst.episodes)}个episode、{int(worst.queries)}个查询；总SSE最大贡献类别是{category_names.get(largest.category,largest.category)}。')
    answers.append('8. 哪类事件最难？'+' '.join(sentences)+' 类别小样本、重叠查询和规则代理限制排名稳定性；这不是硬件故障的因果分类。逐episode全表保留，不以单个最大异常代替类别结论。')
    precursor=pd.read_csv(diag_dir/'precursor_fixed_grid_summary.csv')
    prec=precursor[(precursor.partition=='VALIDATION')&(precursor.future_seconds==30)].iloc[0]
    answers.append(f'9. 历史能提前预知吗？以固定30秒先兆规则为例，全部诊断窗口本来有{prec.positive_rate*100:.2f}%出现未来跳变；发出警报后真的出现跳变的比例为{prec.precision*100:.2f}%，真实跳变窗口被提前警报覆盖{prec.recall*100:.2f}%，未跳变窗口误报{prec.false_positive_rate*100:.2f}%。因此有弱关联但不足以可靠提前获知时刻；精确率也依赖这里的事件频率，不能当作跨工况不变的可预测性。跳变当时才出现的线索不能算提前预测。没有证据证明未来30/120秒的外部控制计划在起点已知。该固定规则失败也不等于证明任何历史预测方法都不可能；发生前/当时/事后及不同窗口的完整统计见04。')
    direction=('事件加权通过本轮30秒全部门槛，值得把事件辅助或多尺度列为下一轮小规模消融候选，但不据此直接采用新模块。'
        if gate['B2_allowed'] else '本轮固定事件加权未通过全部门槛，暂不支持立刻叠加辅助头或多尺度模块来追逐更低数字。')
    answers.append('10. 下一阶段值得加事件辅助或多尺度吗？'+direction+
        '优先核查训练收敛与可用工况/命令描述；历史先兆较弱时，辅助任务不能凭空提供未知的外部切换时间。'
        '当前F2已含原生CausalMultiScaleHistoryBranch多尺度因果CNN，不能说目前完全没有多尺度。后续应先检验现有支路是否有效，或单独研究更长历史的多时间分辨率表示。'
        '若另立冻结协议做辅助任务或多尺度消融，应保留原始逐秒目标和快速历史路径，只考虑压缩较远历史表示，不以聚合均值替代突变。'
        '数小时跨度需另设任务与保持法基线，不能拿本轮30秒成绩充当数小时预测能力。')
    literature=(STUDY/'notes/literature_support.md').read_text(encoding='utf-8')
    literature='\n'.join('##'+line if line.startswith('#') else line for line in literature.splitlines())
    tradeoffs=[]
    for h in [30,120]:
        ref=main[(main.horizon_seconds==h)&(main.model=='Persistence')].iloc[0]
        f=main[(main.horizon_seconds==h)&(main.model=='F2')].iloc[0]
        tradeoffs.append(f'- {h}秒：原生F2/保持法的全RMSE为{f.RMSE_mV:.3f}/{ref.RMSE_mV:.3f}mV；全MAE为{f.MAE_mV:.3f}/{ref.MAE_mV:.3f}mV，普通MAE为{f.normal_MAE_mV:.3f}/{ref.normal_MAE_mV:.3f}mV。不能因为RMSE较低就宣称全面更好。')
    write_report('05_最终研究总结与下一阶段建议.md','# 最终研究总结与下一阶段建议\n\n'+
        f'本轮实际完成{len(states)}个新训练实例，Seed202，共{sum(d["stop_epoch"] for d in states)}轮、{sum(d["optimizer_steps"] for d in states)}次优化器更新。Persistence没有训练。B2'+('经独立门槛允许后执行。' if b2 else '按B1门槛取消。')+
        '\n\n'+markdown_table(main)+
        '\n\n读表说明：1mV等于0.001V。RMSE对少数大错误更敏感，MAE是平均绝对偏差，二者越小越好。30秒行综合了每次起点之后第1至30秒的全部预测，不是只算第30秒；120秒同理。R²衡量未来电压变化量ΔV的解释程度，不是绝对电压R²，更不是健康状态或剩余寿命。'+
        '\n\n对电压保持法的优势与代价要分开看：\n\n'+'\n'.join(tradeoffs)+'\n\n## 十个问题的直接回答\n\n'+'\n\n'.join(answers)+
        '\n\n## 适用边界\n\n这是单种子探索，不宣称架构稳定胜负、最优参数、SOH估计或RUL。没有IEEE2014，没有新后留盲测，没有未来控制输入，也没有为了低于10mV追加试验。旧研究和原始数据只读保护。AI辅助实现与报告；真实日志、原始时间戳、逐查询预测和独立核验为数值依据。\n\n'+reload_disclosure()+'\n\n## 文献依据与不能外推之处\n\n'+literature)
    save_summary=dict(trained_instances=len(states),seed=202,total_epochs=sum(d['stop_epoch'] for d in states),
        optimizer_steps=sum(d['optimizer_steps'] for d in states),B2_executed=b2,protocol_sha256=sha(OUT/'frozen_protocol.json'),
        validation_results=main.to_dict('records'),historical_comparison_same600s=changes.to_dict('records'))
    from core import save
    save('outputs/final_execution_summary.json',save_summary)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-only',action='store_true');args=parser.parse_args()
    df,states,runs=collect();ids={d['id'] for d in states}
    assert {'A1','A2','A3','A4'}<=ids
    if not args.native_only:assert 'B1' in ids and (OUT/'B1_acceptance.json').exists()
    reports(df,states,runs,args.native_only)
    if not args.native_only:
        _,a=load_cache();figures(df,states,runs,a)
    print('REPORTS_GENERATED',len(states),'native_only',args.native_only,flush=True)

if __name__=='__main__':main()

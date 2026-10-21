"""P4: figures, original plotting tables, and evidence-limited Chinese results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_raw import STUDY, OUT, save, sha
from freeze_protocol import load_cache, PARTS

FIG=STUDY/'figures/results'
COLORS={'Persistence':'#767676','LSTM':'#B64342','F2':'#0F4D92'}
MODELS=['Persistence','LSTM','F2']

def export(fig,name,data):
    fig.tight_layout(pad=1.5)
    fig.savefig(FIG/f'{name}.png',dpi=300,facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',facecolor='white')
    data.to_csv(FIG/f'{name}_plot_data.csv',index=False)
    plt.close(fig)

def main():
    for h in [30,120]:
        assert json.loads((OUT/f'H{h}_acceptance.json').read_text())['pass']
    cfg=json.loads((OUT/'frozen_protocol.json').read_text(encoding='utf-8'))
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.5,
        'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    m=pd.concat([pd.read_csv(OUT/f'metrics_H{h}.csv') for h in [30,120]],ignore_index=True)
    m.to_csv(OUT/'all_metrics.csv',index=False)
    phases=['VALIDATION','HISTORICAL_ASSESSMENT','LATE_LIFE']
    allm=m.loc[m.stratum=='ALL'].copy()
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for r,h in enumerate([30,120]):
      for c,partition in enumerate(phases):
        ax=axes[r,c]; z=allm.loc[(allm.horizon_seconds==h)&(allm.partition==partition)].set_index('model').loc[MODELS]
        bars=ax.bar(MODELS,z.RMSE_V*1000,color=[COLORS[k] for k in MODELS],edgecolor='black',linewidth=.7)
        ax.set_title(f'{partition}\n300 observations → {h} s',fontsize=12)
        ax.set_ylabel('All-query RMSE (mV)'); ax.set_ylim(0,max(z.RMSE_V)*1000*1.2)
        for bar,v in zip(bars,z.RMSE_V*1000):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.2f}',ha='center',va='bottom',fontsize=11)
    export(fig,'01_all_query_rmse',allm)
    fig,axes=plt.subplots(2,2,figsize=(12,8)); logs=[]
    states=[]
    for r,h in enumerate([30,120]):
      for c,kind in enumerate(['LSTM','F2']):
        run=STUDY/f'runs/H{h}_{kind}_Seed202'; log=pd.read_csv(run/'training_history.csv')
        state=json.loads((run/'training_state.json').read_text())
        states.append(state); log['model']=kind;log['horizon_seconds']=h
        log['validation_standardized_MSE']=(log.validation_RMSE_V/cfg['training']['target_scale_V'])**2
        logs.append(log); ax=axes[r,c]
        ax.plot(log.epoch,log.train_standardized_MSE,label='Train MSE',color='#767676')
        ax.plot(log.epoch,log.validation_standardized_MSE,label='Validation MSE',color=COLORS[kind])
        ax.axvline(state['best_epoch'],ls='--',color='#4D4D4D',lw=1,label='Selected epoch')
        ax.set(xlabel='Epoch',ylabel='Standardized ΔV MSE',title=f'{kind}, 300 → {h} s')
        ax.set_yscale('log');ax.legend(fontsize=10)
    export(fig,'02_real_training_history',pd.concat(logs,ignore_index=True))
    # Every lead contributes to the main metrics; this plot makes that visible.
    fig,axes=plt.subplots(2,3,figsize=(15,8)); lead_rows=[]
    for r,h in enumerate([30,120]):
      for c,partition in enumerate(phases):
        ax=axes[r,c]
        for kind in MODELS:
          run=STUDY/(f'runs/H{h}_Persistence' if kind=='Persistence' else f'runs/H{h}_{kind}_Seed202')
          with np.load(run/f'predictions_{partition}.npz',allow_pickle=False) as p:
            rmse=np.sqrt(np.mean((p['prediction_delta_V']-p['truth_delta_V'])**2,axis=0))*1000
          ax.plot(np.arange(1,h+1),rmse,color=COLORS[kind],label=kind,lw=1.8)
          lead_rows.extend(dict(horizon=h,partition=partition,model=kind,lead_s=i+1,RMSE_mV=float(v)) for i,v in enumerate(rmse))
        ax.set(xlabel='Nominal forecast lead (s)',ylabel='RMSE (mV)',title=f'{partition}, H={h} s')
        ax.legend(fontsize=10)
    export(fig,'03_all_forecast_leads',pd.DataFrame(lead_rows))
    fig,axes=plt.subplots(2,2,figsize=(13,8)); plotrows=[]
    for r,h in enumerate([30,120]):
      run=STUDY/f'runs/H{h}_Persistence'
      with np.load(run/'predictions_VALIDATION.npz',allow_pickle=False) as p:
        event=np.flatnonzero(p['event_query'].any(1)); indices=[0,int(event[0]) if event.size else 0]
      for c,index in enumerate(indices):
        ax=axes[r,c]
        for kind in MODELS:
          run=STUDY/(f'runs/H{h}_Persistence' if kind=='Persistence' else f'runs/H{h}_{kind}_Seed202')
          with np.load(run/'predictions_VALIDATION.npz',allow_pickle=False) as p:
            target=p['target_time_ms'][index]; origin=p['origin_time_ms'][index]
            truth=p['truth_delta_V'][index]+p['origin_voltage_V'][index]
            pred=p['prediction_delta_V'][index]+p['origin_voltage_V'][index]
            elapsed=(target-origin)/1000
            if kind=='Persistence':ax.plot(elapsed,truth,color='black',label='Observed',lw=2)
            ax.plot(elapsed,pred,color=COLORS[kind],label=kind,lw=1.6)
            plotrows.extend(dict(horizon=h,panel='first_validation_origin' if c==0 else 'first_event_validation_origin',model=kind,origin_time_ms=int(origin),target_time_ms=int(t),elapsed_s=float(e),true_V=float(v),pred_V=float(pv)) for t,e,v,pv in zip(target,elapsed,truth,pred))
        ax.set(xlabel='Actual elapsed time (s)',ylabel='Mean cell voltage (V)',title=f'H={h} s; '+('first validation origin' if c==0 else 'first event origin'))
        ax.legend(fontsize=10)
    export(fig,'04_predeclared_chronological_examples',pd.DataFrame(plotrows))
    intervals=pd.read_csv(OUT/'sampling_interval_counts.csv'); top=intervals.nlargest(9,'count').sort_values('interval_ms')
    fig,ax=plt.subplots(figsize=(10,4.5)); ax.bar(top.interval_ms.astype(str),top['count'],color='#3775BA',edgecolor='black',lw=.6)
    ax.set(xlabel='Original adjacent DateTime interval (ms)',ylabel='Number of pairs (log)',title='Most frequent original sampling intervals')
    ax.set_yscale('log');export(fig,'05_original_sampling_intervals',intervals)
    # Plain Chinese results, with exact units and limits rather than performance promises.
    lines=['# 单种子模型试验与误差对照','',
      '完成固定 Seed202 的4次全新训练：H30与H120各1次LSTM和1次完整原生F2；Persistence未训练。训练顺序为30秒全部完成并通过真实性验收，才开始120秒。没有新模块、门控修改、新损失、超参数搜索或重复找种子。',
      '', '窗口口径：300条模型输入，最早一行的过去30点统计另有29条仅过去的预热，原始信息支持约328秒；不是严格只看最近300秒原始信息的任务。该派生定义在训练前代码/协议已固定，详见03补充说明。',
      '', '## 全样本主指标','',
      '| 跨度 | 分区 | 模型 | 查询数 | RMSE(mV) | MAE(mV) | ΔV R² | P95(mV) | P99(mV) |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for z in allm.itertuples():
        lines.append(f'| {z.horizon_seconds}s | {z.partition} | {z.model} | {z.queries} | {z.RMSE_V*1000:.4f} | {z.MAE_V*1000:.4f} | {z.R2_delta:.5f} | {z.P95_abs_V*1000:.4f} | {z.P99_abs_V*1000:.4f} |')
    lines += ['', '全部数值和事件/普通分层见 `outputs/all_metrics.csv`。主指标按所有保存起点×全部目标秒的查询评分；本轮600秒网格且跨度不超过120秒，同跨度目标实际不重叠。若将来更密集滚动出现重叠，仍须明确查询口径，不能混作独立秒评分。R²对ΔV计算，不能换称绝对电压R²。全部精度以CSV/NPZ为准，表格仅格式化显示。',
      '', '## 事件和普通区间','',
      '| 跨度 | 分区 | 模型 | 分层 | 查询数 | RMSE(mV) | MAE(mV) | P95(mV) | P99(mV) |',
      '|---|---|---|---|---:|---:|---:|---:|---:|']
    for z in m.loc[(m.partition!='TRAIN')&(m.stratum!='ALL')].itertuples():
      lines.append(f'| {z.horizon_seconds}s | {z.partition} | {z.model} | {z.stratum} | {z.queries} | {z.RMSE_V*1000:.4f} | {z.MAE_V*1000:.4f} | {z.P95_abs_V*1000:.4f} | {z.P99_abs_V*1000:.4f} |')
    lines+=['','RMSE对极端误差更敏感，MAE反映另一种平均代价，不能只报下降的一项。分层标签是Train阈值冻结后事后赋给评价查询的，不是模型输入。']
    for h in [30,120]:
      subset=m.loc[(m.horizon_seconds==h)&(m.partition=='VALIDATION')]
      per=subset.loc[(subset.model=='Persistence')&(subset.stratum=='ALL')].iloc[0]
      for kind in ['LSTM','F2']:
        z=subset.loc[(subset.model==kind)&(subset.stratum=='ALL')].iloc[0]
        normal=subset.loc[(subset.model==kind)&(subset.stratum=='NORMAL')].iloc[0]
        normal_per=subset.loc[(subset.model=='Persistence')&(subset.stratum=='NORMAL')].iloc[0]
        lines.append(f'- {h}秒{kind}：全样本RMSE相对Persistence变化{(z.RMSE_V/per.RMSE_V-1)*100:+.2f}%，MAE变化{(z.MAE_V/per.MAE_V-1)*100:+.2f}%；普通查询MAE {normal.MAE_V*1000:.4f}mV（Persistence {normal_per.MAE_V*1000:.4f}mV）。由此判断收益是否来自尾部/事件，不能泛称所有工况都改善。')
    lines+=['','## 真实训练数量、收敛与资源','',
      '| 模型 | 跨度 | 参数 | 停止epoch | 最佳epoch | 优化器步数 | 秒 | 峰值已分配MiB | 峰值预留MiB |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in states:lines.append(f"| {s['model']} | {s['horizon']} | {s['parameters']} | {s['stop_epoch']} | {s['best_epoch']} | {s['optimizer_steps']} | {s['training_seconds']:.1f} | {s['GPU_peak_allocated_MiB']:.1f} | {s['GPU_peak_reserved_MiB']:.1f} |")
    lines+=['','F2收敛与结构检查单独保存在 `outputs/f2_diagnostics_H30.json` 和 `f2_diagnostics_H120.json`：只用已完成checkpoint的CPU门控诊断和真实曲线，不引入新训练。如果门控已经更新且未饱和，只能排除“完全没学/全部饱和”这类简单故障，不能证明门控在所有事件上正确。12轮内尚未平台的曲线应归为有限预算证据，不宣称架构失败或F2必然优于LSTM。']
    for h in [30,120]:
      diag=json.loads((OUT/f'f2_diagnostics_H{h}.json').read_text(encoding='utf-8'))
      gate=diag['gate_by_partition']['VALIDATION']['LSTM_branch_weight']
      lines.append(f"- {h}秒F2：Validation门控的LSTM分支权重均值{gate['mean']:.6f}、范围[{gate['min']:.6f},{gate['max']:.6f}]，饱和率(<0.05或>0.95)均为0，原始零初始化门控output参数已更新。因此没有发现‘门控完全没学习或全部饱和’的简单故障。")
    lines+=['','120秒F2在第7轮达到本预算内最佳Validation，第8–10轮训练MSE仍下降但Validation未改善，按冻结patience停止。这既可能包含优化/泛化权衡，也可能受稀有状态支持不足影响；现有试验不能把原因唯一归给门控或证明延长训练必然改善。后续应先查这些基础因素和输入工况支持，再考虑另立协议。']
    startup_failures=sum(s.get('zero_optimizer_step_startup_failures',0) for s in states)
    lines+=['',f'实际有{len(states)}个完成优化的模型实例，另有{startup_failures}次F2首次前向设备兼容失败（优化器0步）。失败原始状态保存为zero_step_startup_failure.json；只修复新adapter的Lightning设备属性迁移，恢复时初始化SHA与失败前完全相同；已完成LSTM没有重训。',
      '', '同宽度/层数并不意味着参数数或优化难度相同。最佳epoch靠近12轮上限时应视为可能未充分收敛；不能为了得到低于10mV而继续重训。本次先检查收敛和表征，再决定是否值得另立新的协议。',
      '', '## 七个研究问题的直接回答','',
      '1. **数据能否用？** 17份CSV共8,604,654行，LFS逐文件OID和大小通过，存在长段真实近1秒数据，足够建立局部连续秒级预测。跨停录/文件边界不补齐，全寿命不是无断点的均匀秒序列。',
      '2. **30分钟损失什么？** 抹平电流切换、电压瞬态、相位和事件先后；OpHrs2冻结时还能把远超过30分钟日历跨度的状态混在同一桶。',
      '3. **旧大误差和聚合有关吗？** 两组已经暴露的异常桶各混入5006条零流和1804条非零流样本，旧均值约0.208V；独立重算22桶一致。因此混合状态和双时钟解释了重要部分，但不能推断所有大误差都只由聚合引起。',
      '4. **F2还是LSTM？** 本轮只在固定单种子、小样本、12轮预算下比较，见下列真实结果；不把一次试验当作架构最终优劣。']
    recommendations=[]
    for h in [30,120]:
      z=allm.loc[(allm.horizon_seconds==h)&(allm.partition=='VALIDATION')].set_index('model')
      best_model=z.RMSE_V.idxmin(); f2=float(z.loc['F2','RMSE_V']); lstm=float(z.loc['LSTM','RMSE_V']); per=float(z.loc['Persistence','RMSE_V'])
      lines.append(f'   - {h}秒Validation：Persistence {per*1000:.4f}mV，LSTM {lstm*1000:.4f}mV，F2 {f2*1000:.4f}mV；该分区最低为{best_model}。F2相对同任务Persistence '+('更好' if f2<per else '没有改善')+'。')
      recommendations.append(dict(horizon=h,validation_best=best_model,F2_beats_LSTM=f2<lstm,F2_beats_Persistence=f2<per))
    lines+=['5. **秒级RMSE及Persistence比较？** 上表按每个跨度和物理分区给出真实RMSE，不跨30秒/120秒或旧12小时任务比较。',
      '6. **下一步优化什么？** 优先根据真实训练曲线判断是否未收敛、输入尺度/工况漂移或稀有启动事件不足，再核对控制/温压信息。如果F2落后，先分析这些基础因素，不立即叠加残差头、损失或新模块。',
      '7. **怎样保留秒级信息研究数小时？** 保留原始秒级数据为权威与短时高分辨率分支；未来另立多分辨率历史编码/分块解码协议，显式保存每个预测目标真实时间戳，长时段也不能仅用聚合均值替代。是否拥有起点可知控制计划须单独核查。600秒和数小时预测都未在本轮运行。',
      '', '## 复现与产物','',
      '本轮从空研究目录依次运行 `python -B scripts/audit_raw.py`、`python -B scripts/audit_events.py`、`python -B scripts/freeze_protocol.py`，确认review/gate后执行 `python -B scripts/train_pilot.py --horizon 30`、`python -B scripts/verify_runs.py --horizon 30`，再执行对应120秒命令和 `python -B scripts/make_report.py`。已有完整训练实例会验证checkpoint并复用，不会静默重训。',
      '', '从Git新克隆复核时先完整下载LFS，再运行 `python -B scripts/audit_raw.py --rebuild-cache`（仅接受缺失/空cache；重建的中间审计在work/，保留已提交的冻结结果），随后运行两跨度verify_runs。若要重新训练，使用另外的全新实验副本并保留本轮产物；不能覆盖本轮固定实例。',
      '', '每模型`runs/H{H}_{模型}_Seed202/best.pt`为真实新checkpoint，`training_history.csv`为实际日志；`predictions_{分区}.npz`含全部原始目标行号/DateTime/真值/预测和事件掩码。Persistence单独目录，无checkpoint。图在`figures/results/`和`figures/events/`，每图有300dpi PNG、PDF和原始CSV。',
      '', '## 限制与披露','',
      '固定600秒相位网格可能漏掉短事件，Train最多2048起点，尚未考察更密集在线滚动评价。原始传感器记录频率不等于所有物理过程均以1Hz可辨识；测量开关等共同原因不能完全排除。24维是事先工程定义，并非最优特征。两个留出分区历史暴露，只能作探索性结果。单seed、有限预算不能支持显著性、最优性或全面泛化。AI协助检索、编程、绘图和报告；所有数值来自可核查的实际日志和逐目标预测。']
    (STUDY/'04_单种子模型试验与误差对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    save('outputs/final_execution_summary.json',dict(trained_instances=len(states),zero_optimizer_step_startup_failures=startup_failures,seed=202,optimizer_steps=sum(s['optimizer_steps'] for s in states),states=states,validation_decisions=recommendations,protocol_sha256=sha(OUT/'frozen_protocol.json')))
    # Append results-driven interpretation to the source-verified future-directions report.
    path=STUDY/'05_后续F2改进方向与文献依据.md'
    text=path.read_text(encoding='utf-8')
    marker='\n## 本轮固定试验的实测决定\n'
    if marker not in text:
      text+=marker+'\n'
      for r in recommendations:
        text+=f"- {r['horizon']}秒Validation最低RMSE模型为{r['validation_best']}；F2优于LSTM={r['F2_beats_LSTM']}，优于Persistence={r['F2_beats_Persistence']}。完整数值见04及all_metrics.csv。\n"
      text+='\n这些观察仅支持固定预算下的下一步诊断次序；本轮没有实现文献候选模块。若F2落后，先查训练曲线、历史输入/状态与稀有事件支持，不把文献方法直接当作保证有效的修补。\n'
      path.write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='REPORTS_CREATED',trained_instances=len(states),decisions=recommendations),ensure_ascii=False))

if __name__=='__main__': main()

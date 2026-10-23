"""P1: the user-specified historical Seed202 Top10 cohort, read-only."""
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,REPO,REFERENCE,SOURCE,CONFIG,utc,sha,read,dump,csv,read_csv,metrics,history_snapshot
from data import freeze_windows,add_events

def mdtable(f):
    cols=list(f.columns)
    def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))+'\n'

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    cfg=read(CONFIG)
    dump(out/'REGISTRATION.json',dict(created_UTC=utc(),protocol_sha256=sha(CONFIG),data_sha256=sha(SOURCE),
        git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        reference_prediction_sha256=sha(REFERENCE/'predictions_TEST.csv'),reference_checkpoint_sha256=sha(REFERENCE/'best_model.pt')))
    dump(HERE/'outputs/ACTIVE_RUN.json',dict(path=str(out.resolve())))
    history_snapshot(out)
    counts=freeze_windows(out);print(counts[['horizon_steps','partition','legal_origins','queries']].to_string(index=False),flush=True)
    p=read_csv(REFERENCE/'predictions_TEST.csv')
    assert p.seed.eq(202).all() and p.feature_mode.eq('top10').all() and p.partition.eq('TEST').all()
    assert len(p)==12648 and p.origin_row.nunique()==527
    assert not p.duplicated(['origin_row','horizon_step','target_row']).any()
    assert (p.target_row==p.origin_row+p.horizon_step).all()
    w=read_csv(out/'windows/H24_HOLDOUT.csv')
    np.testing.assert_array_equal(p.origin_row,np.repeat(w.origin_row.to_numpy(int),24))
    np.testing.assert_array_equal(p.horizon_step,np.tile(np.arange(1,25),len(w)))
    f=pd.read_csv(SOURCE);idx=p.target_row.to_numpy(int);o=p.origin_row.to_numpy(int)
    np.testing.assert_array_equal(p.target_age_h,f.bin_end_ophrs2_h.to_numpy(float)[idx])
    np.testing.assert_array_equal(p.origin_age_h,f.bin_end_ophrs2_h.to_numpy(float)[o])
    np.testing.assert_array_equal(p.true_voltage_V,f.cell_voltage_mean.to_numpy(float)[idx])
    np.testing.assert_array_equal(p.origin_raw_voltage_V,f.cell_voltage_mean.to_numpy(float)[o])
    np.testing.assert_array_equal(p.y_true_delta_V,(p.true_voltage_V-p.origin_raw_voltage_V).to_numpy(np.float32))
    np.testing.assert_array_equal(p.predicted_voltage_V,p.origin_raw_voltage_V+p.y_pred_delta_V)
    ledger=read(REFERENCE/'TRAINING_LEDGER.json')
    assert ledger['best_checkpoint_sha256']==sha(REFERENCE/'best_model.pt') and ledger['best_epoch']==122
    thresholds=read(out/'TRAIN_EVENT_THRESHOLDS.json');p=add_events(f,p,thresholds)
    prefix=[];steps=[];events=[];targets=[]
    for h in [6,12,24]:
        g=p[p.horizon_step<=h].copy();row=dict(source='Historical F2 Seed202, selected cohort',prefix_steps=h,**metrics(g));prefix.append(row)
        g['error_mV']=(g.y_pred_delta_V-g.y_true_delta_V)*1000;g['squared_error_V2']=((g.y_pred_delta_V-g.y_true_delta_V)**2)
        target=g.groupby(['target_row','target_age_h'],as_index=False).agg(queries=('squared_error_V2','size'),SSE_V2=('squared_error_V2','sum'),MSE_V2=('squared_error_V2','mean'))
        target['SSE_fraction']=target.SSE_V2/target.SSE_V2.sum();target['prefix_steps']=h;targets.append(target)
        for key in ['event_any','event_high_zero','event_high_switching','event_voltage_jump','event_start_stop']:
            for flag,group in g.groupby(key):events.append(dict(prefix_steps=h,event=key,value=bool(flag),**metrics(group)))
    for step,g in p.groupby('horizon_step'):steps.append(dict(horizon_step=int(step),lead_h=step*.5,**metrics(g)))
    main=prefix[-1]
    for key,value in cfg['historical_expected'].items():
        name=key.replace('_rounded6','');assert round(main[name],6)==value,(key,main[name],value)
    csv(out/'P1_HISTORICAL_PREFIX_METRICS.csv',prefix);csv(out/'P1_HISTORICAL_STEP_METRICS.csv',steps)
    csv(out/'P1_HISTORICAL_FIXED_STEPS.csv',pd.DataFrame(steps).query('horizon_step in [1,6,12,24]'))
    csv(out/'P1_HISTORICAL_EVENT_METRICS.csv',events);csv(out/'P1_HISTORICAL_TARGET_SSE.csv',pd.concat(targets))
    csv(out/'P1_HISTORICAL_POINTS.csv',p)
    top=pd.concat(targets).query('prefix_steps==24').sort_values('SSE_V2',ascending=False).head(12)
    csv(out/'P1_HISTORICAL_TOP_TARGETS.csv',top)
    points_error=p.copy();points_error['SSE_V2']=(p.y_pred_delta_V-p.y_true_delta_V)**2
    csv(out/'P1_HISTORICAL_ORIGIN_SSE.csv',points_error.groupby(['origin_row','origin_age_h'],as_index=False).agg(queries=('SSE_V2','size'),SSE_V2=('SSE_V2','sum')))
    result=dict(PASS=True,full24=main,original_labels_timestamps_and_raw_voltages_exact=True,
        historical_checkpoint_readonly=True,historical_checkpoint_best_epoch=122,reference_path=str(REFERENCE),
        reference_predictions_sha256=sha(REFERENCE/'predictions_TEST.csv'),training_runs=0,
        note='P1 prefixes truncate one already-trained H24 model; not independent H6/H12 training. Selected seed and previously exposed holdout; exploratory only.')
    dump(out/'P1_HISTORICAL_AUDIT.json',result)
    reports=out/'reports';reports.mkdir(exist_ok=True)
    text=f'''# 01 现有24步预测截断分析

准确复算指定20261019T055734_218949Z_seedall_compare批次、Seed202/Top10，没有加载权重重训。全24步恢复RMSE **{main['RMSE_mV']:.9f}mV**、MAE **{main['MAE_mV']:.9f}mV**、R² **{main['R2']:.9f}**，分别与指令给出的六位舍入一致；12648查询、527起点、619唯一目标时间，全部保留。

{mdtable(pd.DataFrame(prefix)[['prefix_steps','RMSE_mV','MAE_mV','R2','P95_mV','P99_mV','max_abs_mV','top1pct_SSE_fraction','target_time_equal_RMSE_mV','queries','unique_targets']])}

这是同一个48→24模型的前缀截断，不能冒充另行训练48→6/12。真实目标ΔV采用历史float32标签，指标用float64计算；绝对电压从原始、未EWMA的起点电压恢复，尺度没有重复还原。以target_row=origin_row+horizon_step逐点检查，源数据电压及实际运行时钟严格一致，预测CSV和原checkpoint SHA已保存。

{mdtable(pd.DataFrame(steps).query('horizon_step in [1,6,12,24]')[['horizon_step','lead_h','RMSE_mV','MAE_mV','R2']])}

最大绝对误差1%查询对总SSE的贡献见上表；唯一目标时间先对同一目标各查询的MSE平均，再对目标等权计算，不是预测集成。逐目标SSE贡献、每个预测起点贡献和全部逐点数据分别保存为 P1_HISTORICAL_TARGET_SSE.csv、P1_HISTORICAL_ORIGIN_SSE.csv、P1_HISTORICAL_POINTS.csv。重叠查询不是独立实验。

{mdtable(top[['target_row','target_age_h','queries','SSE_fraction']])}

工况分组阈值只从Train确定，未来真实启停/电流/电压跳变仅用于事后诊断，不进入模型输入。分组是可重叠的描述性条件，不能把相关性解释为门控因果失效。完整事件及尾部统计见P1_HISTORICAL_EVENT_METRICS.csv。历史12h误差不是另一批F2/LSTM六种子均值，也不与过去长期趋势代理混比。

该Seed202已经按后留成绩选中；本轮不再切换种子，不声称无偏盲测或随机初始化总体稳定性。P1通过才允许进入最多6次新训练。
'''
    (reports/'01_现有24步预测截断分析.md').write_text(text,encoding='utf-8')
    delivered=HERE/'reports/01_现有24步预测截断分析.md'
    if not delivered.exists():
        delivered.parent.mkdir(exist_ok=True)
        delivered.write_text(text,encoding='utf-8')
    print('P1 PASS: exact historical cohort recovered',main['RMSE_mV'],main['MAE_mV'],main['R2'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)

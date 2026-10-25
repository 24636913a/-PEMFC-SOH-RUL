"""准备性时间配对与资格审计；不训练迁移，不重新筛选原始记录。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from 运行低阶实验 import csv,js,sha
WORK=Path(__file__).resolve().parents[1];GUIDE=WORK.parents[1];TAB=WORK/'输出结果/表格'

def main():
    cfg=json.loads((WORK/'配置/阶段2B冻结协议.json').read_text(encoding='utf-8'))
    diag=pd.read_csv(WORK/'数据准备/稀疏诊断观测.csv');rows=[]
    sources={k:pd.read_csv(WORK/'数据准备'/f) for k,f in [('FC1','FC1小时观测.csv'),('RealHyFC','RealHyFC循环观测.csv')]}
    timing=pd.read_csv(GUIDE/'基础预测实验/输出结果/表格/300和600小时诊断可用性表.csv')
    bounds=json.loads((GUIDE/'基础预测实验/输出结果/Real数据准备记录.json').read_text(encoding='utf-8'))['selected']['bounds']
    for dataset,info in cfg['datasets'].items():
        obs=sources[dataset];obs=obs[(obs.scheme==info['scheme'])&obs.valid].sort_values('time')
        for _,r in diag[(diag['电堆']==dataset)&(diag.scheme==('FC1_10' if dataset=='FC1' else 'B60'))].iterrows():
            rec=dict(dataset=dataset,diagnostic=r['文件名'],nominal_h=r['标称小时'],primary=bool(r.primary),
                diagnostic_supported=bool(r.valid),clock=r['时钟资格'],actual_completed=r['实际测试结束'],
                operating_upper_h=r['累计小时上界'],D_ref=float(r.D_ref_V),D_low=float(r.D_low_V),
                release_known=False,availability='DETECTION_COMPLETED_THEN_AVAILABLE_OFFLINE_ASSUMPTION',
                origin_id=np.nan,operating_available_h=np.nan,operating_available_calendar='',lag_h=np.nan,
                operating_D=np.nan,train_pair=False,strict_clock_pair=False,reason='')
            if dataset=='FC1':
                pre=obs[obs.time<=r['标称小时']];eligible_train=r['标称小时']<=obs[obs.partition=='Train'].time.max()
                if len(pre):
                    o=pre.iloc[-1];lag=float(r['标称小时']-o.time)
                    rec.update(origin_id=int(o.id),operating_available_h=o.time,lag_h=lag,operating_D=o.D)
                    rec['train_pair']=bool(eligible_train and o.partition=='Train' and lag<=48 and r.valid and r.primary)
                    rec['reason']='NOMINAL_CLOCK_ONLY_NO_ACTUAL_DIAGNOSTIC_COMPLETION'
                else:rec['reason']='NO_PRIOR_OPERATING_OBSERVATION'
            else:
                end=pd.Timestamp(r['实际测试结束']);pre=obs[obs.available_ns<=end.value]
                eligible_train=np.isfinite(r['累计小时上界']) and r['累计小时上界']<=bounds[1]
                if len(pre):
                    o=pre.iloc[-1];lag=(end.value-int(o.available_ns))/3.6e12
                    rec.update(origin_id=int(o.id),operating_available_h=o.available_h,
                               operating_available_calendar=pd.Timestamp(int(o.available_ns)).isoformat(),lag_h=lag,operating_D=o.D)
                    rec['strict_clock_pair']=bool(lag<=48 and r.valid)
                    rec['train_pair']=bool(rec['strict_clock_pair'] and eligible_train and o.partition=='Train' and r.primary)
                    rec['reason']='ELIGIBLE_COMPLETION_ASSUMPTION_ONLY' if rec['strict_clock_pair'] else 'PRIOR_OBSERVATION_TOO_OLD_OR_DIAGNOSTIC_UNSUPPORTED'
                else:rec['reason']='NO_OPERATING_OBSERVATION_AVAILABLE_BEFORE_DIAGNOSTIC_END'
            rows.append(rec)
    pairs=pd.DataFrame(rows);csv(TAB/'诊断与运行历史配对及排除.csv',pairs)
    qualification=[];parameters=[]
    for dataset in cfg['datasets']:
        allpairs=pairs[pairs.dataset==dataset];tr=allpairs[allpairs.train_pair]
        strict=tr[tr.strict_clock_pair]
        for dims in [1,2]:
            p=dims+1;minimum=3*p;usable=strict;X=usable[['D_ref','D_low']].to_numpy()[:,:dims]
            rank=0;cond=np.nan
            if len(X)>1 and np.all(np.std(X,axis=0)>1e-14):
                A=np.c_[np.ones(len(X)),(X-X.mean(0))/X.std(0)];rank=int(np.linalg.matrix_rank(A));cond=float(np.linalg.cond(A))
            ok=len(usable)>=minimum and rank==p and cond<=100
            reason=[]
            if dataset=='FC1':reason.append('ACTUAL_DIAGNOSTIC_COMPLETION_UNKNOWN')
            if len(usable)<minimum:reason.append('INSUFFICIENT_DISTINCT_TRAIN_DIAGNOSTICS')
            if rank<p or not np.isfinite(cond) or cond>100:reason.append('DESIGN_RANK_OR_CONDITION_NOT_QUALIFIED')
            transitions=max(0,len(usable)-1)
            qualification.append(dict(dataset=dataset,dimension=dims,nominal_or_completion_train_pairs=len(tr),
                exact_clock_train_pairs=len(strict),min_distinct_pairs=minimum,train_rank=rank,condition=cond,
                train_completed_diagnostic_transitions=transitions,observation_mapping='YES' if ok else 'NO',
                state_evolution='YES' if ok and transitions>=6 else 'NO',
                status='QUALIFIED' if ok and transitions>=6 else 'NOT_QUALIFIED',reason='|'.join(reason or ['LESS_THAN_6_TRAIN_DIAGNOSTIC_TRANSITIONS']),
                extra_input='仅真实已完成诊断；没有未来教师或插值SOH',coefficients='NA' if not ok else 'SEE_MODEL_RECORD'))
            if ok:
                mu=X.mean(0);sd=X.std(0);A=np.c_[np.ones(len(X)),(X-mu)/sd]
                beta=np.linalg.lstsq(A,usable.operating_D,rcond=None)[0]
                parameters.append(dict(dataset=dataset,dimension=dims,mean=mu.tolist(),scale=sd.tolist(),coef=beta.tolist(),
                    diagnostic_files=usable.diagnostic.tolist(),identity='TRAIN_ONLY_EMPIRICAL_MAPPING_NOT_ELECTROCHEMICAL_LAW'))
    csv(TAB/'一维二维状态演化资格.csv',pd.DataFrame(qualification));js(WORK/'输出结果/模型记录/观测映射资格与参数.json',parameters or dict(coefficients=None,status='NA_NO_QUALIFIED_MAPPING'))
    real=sources['RealHyFC'];real=real[(real.scheme=='B60')&real.valid]
    dh=diag[(diag['电堆']=='RealHyFC')&(diag.scheme=='B60')&diag.primary].copy()
    shifted=dh[dh['标称小时']==600].iloc[0];upper=float(shifted['累计小时上界']);completed=pd.Timestamp(shifted['实际测试结束']).value
    later=real[(real.available_h>=upper)&(real.available_ns>=completed)].sort_values('available_h')
    late=later.iloc[0] if len(later) else None
    cases=[('STRICT_300',300.,None),('STRICT_600',600.,None),('SHIFTED_AFTER_600_COMPLETED',float(late.available_h) if late is not None else np.nan,
        int(late.available_ns) if late is not None else None)]
    transfer=[]
    for name,cut,clock in cases:
        avail=dh[dh['累计小时上界'].le(cut)]
        if clock is not None:avail=avail[pd.to_datetime(avail['实际测试结束']).map(lambda x:x.value)<=clock]
        past=real[real.available_h<=cut]
        if clock is not None:past=past[past.available_ns<=clock]
        transfer.append(dict(scenario=name,operating_cutoff_h=cut,exact_cutoff_identity='RUNNING_HOUR_CUTOFF_NOT_NOMINAL_FILENAME' if clock is None else f'cycle_id={int(late.id)}; calendar={pd.Timestamp(clock).isoformat()}',
            completed_main_diagnostics=';'.join(avail['文件名']),n_completed=len(avail),qualified_operating_cycles=len(past),
            temperature_identity='B60_EXPLORATORY',result_release_time='UNKNOWN',
            common_evaluation_start_h=cases[2][1],migration='NOT_RUN',
            caveat='时间完成资格不等于稳态/观测映射或迁移资格；300模型必须自行传播到共同后移评价起点，不使用300后真实观测'))
    csv(TAB/'后续300_600及后移起点协议.csv',pd.DataFrame(transfer))
    csv(TAB/'阶段2A时间资格原表副本.csv',timing)
    js(WORK/'输出结果/观测映射资格完成.json',dict(status='COMPLETED_NO_TRANSFER_TRAINING',
        pairs=len(pairs),train_counts=pairs.groupby('dataset').train_pair.sum().to_dict(),
        sources={k:sha(WORK/'数据准备'/f) for k,f in [('FC1','FC1小时观测.csv'),('RealHyFC','RealHyFC循环观测.csv')]},
        definition='实测邻近性只有统计关联；FC1名义时钟不升级；真实诊断发布时间未知'))
    print('历史配对、映射拒绝/资格与迁移时间协议已保存。')


if __name__=='__main__':main()

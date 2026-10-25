"""冻结数据合同→Train拟合→Validation选候选→探索性Assessment。默认不覆盖。"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,platform
import numpy as np
import pandas as pd
from 低阶演化核心 import fit_train,predict_schedule,scores,objective,history,estimate,forecast

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while c:=f.read(8388608):h.update(c)
    return h.hexdigest()

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def js(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(p,x):x.to_csv(p,index=False,encoding='utf-8-sig')


def continuous_experiment(work,out,cfg,guide):
    tab=out/'表格';records=out/'模型记录';allmet=[];chosen={};allstates=[];allcandidate=[]
    for dataset,info in cfg['datasets'].items():
        print(dataset+'：拟合Train速率和噪声尺度，执行冻结Validation候选',flush=True)
        obs=pd.read_csv(work/'数据准备'/('FC1小时观测.csv' if dataset=='FC1' else 'RealHyFC循环观测.csv'))
        obs=obs[obs.scheme==info['scheme']].sort_values('id')
        train=obs[obs.partition=='Train'];par=fit_train(train,info['lookback'],info['min_history'])
        js(records/(dataset+'_Train拟合参数.json'),par)
        baseline=pd.read_csv(guide/'基础预测实验/输出结果/表格'/f'{dataset}基础预测逐起点结果.csv')
        baseline=baseline[baseline.partition.isin(['Validation','Assessment'])].copy()
        schedule=baseline[baseline.method=='Persistence'].copy()
        assert not schedule.empty
        val=schedule[schedule.partition=='Validation'];assess=schedule[schedule.partition=='Assessment']
        assert val.origin_time.min()>=par['train_last_time']
        assert assess.origin_time.min()>=val.target_time.max(),'Assessment模型选择时间隔离失败'
        csv(tab/(dataset+'_冻结预测查询索引.csv'),schedule.drop(columns=['pred_v','pred_D','pred_SOH','method']))
        predictions=[baseline];selection=[]
        for method,candidates in [('BoundedRate',cfg['C']['candidates']),('SlowReversible',cfg['D'][dataset+'_tau'])]:
            trials=[]
            for i,setting in enumerate(candidates):
                pp,ss=predict_schedule(val,obs,par,method,setting)
                obj=objective(pp)
                trials.append((obj[0],obj[1],i,setting))
                selection.append(dict(dataset=dataset,method=method,candidate_index=i,setting=setting,
                    equal_horizon_RMSE_mV=obj[0],equal_horizon_MAE_mV=obj[1],parameters_fit_only='Train',selection_labels='Validation'))
                allcandidate.append(pp.assign(candidate_index=i))
            _,_,i,setting=min(trials)
            chosen[dataset+'_'+method]=dict(setting=setting,candidate_index=i,par=par)
            for constrained in [True,False]:
                pp,ss=predict_schedule(schedule,obs,par,method,setting,constrained)
                predictions.append(pp);allstates.extend(ss)
        pred=pd.concat(predictions,ignore_index=True)
        csv(tab/f'{dataset}全部方法逐点预测.csv',pred)
        met=scores(pred,['dataset','scheme','partition','horizon','method']);allmet.append(met)
        csv(tab/f'{dataset}各跨度指标.csv',met);csv(tab/(dataset+'_Validation候选选择.csv'),pd.DataFrame(selection))
        # Same prespecified complete-block eligibility including NA horizons and unavailable A60.
        na=[]
        for scheme in obs.scheme.unique() if dataset=='FC1' else ['A60','B60']:
            for part in ['Validation','Assessment']:
                for h in info['horizons']:
                    if scheme=='A60' or not ((met.partition==part)&(met.horizon==h)&(met.scheme==scheme)).any():
                        for method in ['Persistence','LocalLinear','BoundedRate','SlowReversible']:
                            na.append(dict(dataset=dataset,scheme=scheme,partition=part,horizon=h,method=method,n=0,
                                           reason='FROZEN_2A_NO_QUALIFIED_COMPLETE_BLOCK_OR_BASE'))
        csv(tab/(dataset+'_缺失资格NA.csv'),pd.DataFrame(na))
        # Additional fixed-origin forecast, never pooled with complete-block metrics.
        origin=float(train[train.valid].time.max()) if dataset=='FC1' else float(assess.origin_time.min())
        hist=history(obs,origin,info['lookback']);o=obs[obs.time==origin].iloc[0]
        extra=[];h=168 if dataset=='FC1' else 25
        for method in ['BoundedRate','SlowReversible']:
            if dataset=='FC1':
                extra.append(dict(dataset=dataset,method=method,origin_id=int(o.id),origin_time=origin,
                    step=np.nan,target_time=np.nan,pred_v=np.nan,true_v=np.nan,
                    status='NA_VALIDATION_SELECTED_CONFIGURATION_NOT_YET_AVAILABLE_AT_TRAIN_END',
                    model_selection_available=False,information_note='不执行使用未来Validation选择的Train末端额外预测；长期图改用合法RUL独立内层选择起点'))
                continue
            setting=chosen[dataset+'_'+method]['setting'];s=estimate(hist,par,method,setting)
            for step in range(1,h+1):
                y,l,r=forecast(s,float(step));t=obs[obs.time==origin+step];truth=t.v.iloc[0] if len(t) and t.valid.iloc[0] else np.nan
                extra.append(dict(dataset=dataset,method=method,origin_id=int(o.id),origin_time=origin,step=step,
                    target_time=origin+step,pred_v=float(o.base_V-y),slow_D=float(l),reversible_D=float(r),true_v=truth,
                    status='FIXED_ORIGIN_DESCRIPTIVE_NOT_ROLLING_COMPLETE_BLOCK',model_selection_available=True,
                    information_note='仅展示固定起点后已冻结传播；不纳入主完整块指标'))
        csv(tab/(dataset+'_固定起点额外外推.csv'),pd.DataFrame(extra))
    js(records/'冻结选择结果.json',chosen);js(records/'逐起点状态与约束记录.json',allstates)
    csv(tab/'Validation全部候选逐点预测.csv',pd.concat(allcandidate,ignore_index=True))
    met=pd.concat(allmet,ignore_index=True);csv(tab/'全部模型公平比较.csv',met)
    csv(tab/'SOH预测误差.csv',met[['dataset','scheme','partition','horizon','method','n','origins','RMSE_SOH_pp','MAE_SOH_pp']])
    return chosen


def rul_experiment(work,out,cfg,guide):
    print('FC1：400/500/600h各自重新拟合80%/20%，不借后续Train参数',flush=True)
    fc=pd.read_csv(work/'数据准备/FC1小时观测.csv');fc=fc[fc.valid].sort_values('time');base=fc.base_V.iloc[0]
    r=cfg['RUL'];results=[];curves=[];logs=[]
    for origin in r['origins']:
        hist=fc[fc.time<=origin];n=int(.8*len(hist));tr=hist.iloc[:n];va=hist.iloc[n:];inner=float(tr.time.max())
        innerpar=fit_train(tr,48,24)
        for method,candidates in [('BoundedRate',cfg['C']['candidates']),('SlowReversible',cfg['D']['FC1_tau'])]:
            choices=[]
            for idx,setting in enumerate(candidates):
                hh=history(tr,inner,48);ss=estimate(hh,innerpar,method,setting)
                py=forecast(ss,va.time.to_numpy()-inner)[0];e=py-va.D.to_numpy()
                obj=(float(np.sqrt(np.mean(e**2))),float(np.mean(abs(e))),idx,setting)
                choices.append(obj)
                logs.append(dict(origin_h=origin,method=method,setting=setting,inner_train_cutoff=inner,
                    inner_val_first=float(va.time.min()),inner_val_last=float(va.time.max()),RMSE_V=obj[0],MAE_V=obj[1],
                    parameter_train_ids=tr.id.tolist(),validation_ids=va.id.tolist(),innerpar=innerpar))
            setting=min(choices)[3];par=fit_train(hist,48,24);state=estimate(history(hist,origin,48),par,method,setting)
            logs.append(dict(origin_h=origin,method=method,chosen=setting,final_fit_ids=hist.id.tolist(),
                             par=par,state=state,phase='REFIT_ONLY_PRE_ORIGIN'))
            future=np.arange(origin+1,origin+r['horizon']+1,dtype=float)
            yh,slow,rev=forecast(state,future-state['last_time'])
            for t,y,l,v in zip(future,yh,slow,rev):curves.append(dict(origin_h=origin,method=method,time_h=t,
                pred_D_V=y,pred_v=base-y,slow_D_V=l,reversible_D_V=v,base_V=base,setting=setting,history_last_h=hist.time.max()))
            for drop in r['drops']:
                ac=fc[fc.D>=drop*base];actual=float(ac.time.iloc[0]) if len(ac) else np.nan
                before=fc[fc.time<actual] if len(ac) else fc;prev=float(before.time.max()) if len(before) else np.nan
                pc=future[yh>=drop*base];pred=float(pc[0]) if len(pc) else np.nan
                eligible=bool(hist.D.iloc[-1]<drop*base and (not np.isfinite(actual) or actual>origin))
                both=eligible and np.isfinite(actual) and np.isfinite(pred)
                results.append(dict(origin_h=origin,method=method,drop=drop,main=drop==r['main'],threshold_V=base*(1-drop),base_V=base,
                    history_last_h=float(hist.time.max()),actual_first_cross_h=actual,actual_previous_qualified_h=prev,
                    actual_right_censored=not np.isfinite(actual),actual_observation_end_h=float(fc.time.max()),
                    pred_first_cross_h=pred,pred_right_censored=not np.isfinite(pred),prediction_end_h=origin+r['horizon'],
                    origin_eligible=eligible,status='CONDITIONAL_THRESHOLD_RUL' if both else ('CENSORED' if eligible else 'ORIGIN_ALREADY_CROSSED_OR_BELOW'),
                    actual_RUL_h=actual-origin if eligible and np.isfinite(actual) else np.nan,
                    pred_RUL_h=pred-origin if eligible and np.isfinite(pred) else np.nan,
                    abs_error_h=abs(pred-actual) if both else np.nan,
                    relative_error_pct=abs(pred-actual)/(actual-origin)*100 if both else np.nan,
                    meaning='OBSERVED_CONDITION_THRESHOLD_NOT_CERTIFIED_EOL',setting=setting))
    baseline=pd.read_csv(guide/'基础预测实验/输出结果/表格/FC1_RUL阈值与删失结果.csv')
    oldcurves=pd.read_csv(guide/'基础预测实验/输出结果/表格/FC1_RUL外推曲线.csv')
    csv(out/'表格/FC1条件RUL误差与删失.csv',pd.concat([baseline,pd.DataFrame(results)],ignore_index=True))
    csv(out/'表格/FC1条件RUL外推曲线.csv',pd.concat([oldcurves,pd.DataFrame(curves)],ignore_index=True))
    js(out/'模型记录/RUL各起点拟合选择记录.json',logs)
    # Recovery episodes are observations, not a revised definition of first crossing.
    episodes=[];below=False;last=None;begin=None
    for row in fc.itertuples():
        now=row.D>=r['main']*base
        if now and not below:begin=row.time
        if below and not now:episodes.append(dict(first_below_h=begin,next_above_h=row.time,
            last_below_observed_h=last,interval_contains_missing=bool(((~pd.read_csv(work/'数据准备/FC1小时观测.csv').set_index('time').valid).loc[begin:row.time]).any()),
            interpretation='首次越界之后可再次高于阈值；不是不可逆EOL证据'))
        below=now;last=row.time
    if below:episodes.append(dict(first_below_h=begin,next_above_h=np.nan,last_below_observed_h=last,interpretation='记录结束仍在阈值下'))
    csv(out/'表格/FC1_809h越界与后续回升.csv',pd.DataFrame(episodes))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parents[3]/'物理约束退化演化/阶段2B');parser.add_argument('--output',type=Path)
    args=parser.parse_args();work=args.work.resolve();guide=next(p for p in work.parents if p.name=='PEMFC指导')
    out=(args.output or work/'输出结果/正式版本复算'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve();out.mkdir(exist_ok=True,parents=True)
    if (out/'计算完成.json').exists():raise RuntimeError('完成目录禁止覆盖，请用--output指定新目录')
    for name in ['表格','模型记录']:(out/name).mkdir(exist_ok=True)
    cfg=json.loads((Path(__file__).resolve().parents[1]/'配置/阶段2B冻结协议.json').read_text(encoding='utf-8'))
    frozen=json.loads((Path(__file__).resolve().parents[1]/'配置/运行前保护快照.json').read_text(encoding='utf-8'))
    assert sha(Path(__file__).resolve().parents[1]/'配置/阶段2B冻结协议.json')==frozen['protocol_sha256']
    contract=json.loads((Path(__file__).resolve().parents[1]/'配置/冻结输入身份.json').read_text(encoding='utf-8'))
    contract={k.replace('\\','/'):v for k,v in contract.items()}
    for rel,h in contract.items():assert sha(guide/rel)==h,rel
    for name in ['FC1小时观测.csv','RealHyFC循环观测.csv','稀疏诊断观测.csv']:
        assert sha(work/'数据准备'/name)==contract['基础预测实验/数据准备/'+name]
    continuous_experiment(work,out,cfg,guide);rul_experiment(work,out,cfg,guide)
    js(out/'计算完成.json',dict(UTC=datetime.now(timezone.utc).isoformat(),status='CPU_LOW_ORDER_CALCULATIONS_COMPLETED',
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,protocol_sha256=frozen['protocol_sha256'],
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['低阶演化核心.py','运行低阶实验.py']},
        outputs={str(p.relative_to(out)):sha(p) for folder in [out/'表格',out/'模型记录'] for p in folder.glob('*') if p.is_file()}))
    print('低阶模型拟合、选择和实际预测完成。',flush=True)


if __name__=='__main__':main()

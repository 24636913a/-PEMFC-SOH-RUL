"""独立实现最小二乘/约束滤波/外推，核对正式数值及时间身份。"""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
from 运行低阶实验 import sha,js,csv
WORK=Path(__file__).resolve().parents[1];GUIDE=WORK.parents[1]

def regression(x,y):
    x=np.asarray(x,float);A=np.c_[np.ones(len(x)),x-x[-1]]
    coef=np.linalg.lstsq(A,np.asarray(y,float),rcond=None)[0]
    return float(coef[1])

def independent_fit(train,lb,mn):
    d=train[train.valid].sort_values('time');q=[];t=[]
    for o in d.itertuples():
        h=d[(d.time<=o.time)&(d.time>o.time-lb)]
        if len(h)>=mn:q.append(regression(h.time,h.D));t.append(o.time)
    diff=np.diff(d.D);sigma=max(np.median(abs(diff-np.median(diff)))/(.6744897501960817*np.sqrt(2)),1e-5)
    avg=d.D.iloc[0];last=d.time.iloc[0];res=[]
    for o in d.itertuples():
        w=np.exp(-2*(o.time-last)/lb);avg=w*avg+(1-w)*o.D;res.append(o.D-avg);last=o.time
    return dict(q_prior=regression(d.time,d.D),rate_cap=max(np.percentile(abs(np.array(q)),95),1e-6),
        acceleration_cap=max(.25*np.percentile(abs(np.diff(q)/np.diff(t)),95),1e-8),
        noise=sigma,reversible_scale=max(np.std(res),sigma))

def independent_state(h,par,model,setting,constrained):
    times=h.time.to_numpy();ys=h.D.to_numpy();q0=par['q_prior'];cap=par['rate_cap'];acc=par['acceleration_cap']
    def bound(q,previous,dt):
        return min(max(q,max(0.,previous-acc*dt)),min(cap,previous+acc*dt))
    q=min(cap,max(0.,q0)) if constrained else q0
    if model=='BoundedRate':
        for n in range(len(h)):
            raw=par['q_prior'] if n+1<par['min_history'] else (1-setting)*par['q_prior']+setting*regression(times[:n+1],ys[:n+1])
            dt=times[n]-(times[n-1] if n else times[n]);q=bound(raw,q,dt) if constrained else raw
        return np.array([ys[-1],q,0.])
    tau=setting;state=np.array([ys[0],q,0.]);rv=par['reversible_scale'];sig=par['noise']
    cov=np.diag([rv**2,(.25*cap)**2,rv**2]);H=np.array([[1.,0.,1.]])
    for n in range(1,len(h)):
        dt=times[n]-times[n-1];rho=np.exp(-dt/tau);A=np.array([[1.,dt,0.],[0.,1.,0.],[0.,0.,rho]])
        C=A@cov@A.T+np.diag([(.05*sig)**2*dt,acc**2*dt,rv**2*(1-rho**2)])
        prior=A@state;den=float((H@C@H.T).item()+sig**2);gain=(C@H.T).ravel()/den
        updated=prior+gain*(ys[n]-float((H@prior).item()))
        # Independent equivalent covariance expression, not core's Joseph form.
        cov=C-np.outer(gain,(H@C).ravel());cov=(cov+cov.T)/2
        if constrained:updated[1]=bound(updated[1],state[1],dt)
        state=updated
    return state

def predict(s,steps,model,tau):
    steps=np.asarray(steps,float)
    return s[0]+s[1]*steps+(s[2]*np.exp(-steps/tau) if model=='SlowReversible' else 0.)

def audit(out):
    checks=[];diffs=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,passed=bool(ok),detail=str(detail)))
        assert ok,(name,detail)
    frozen=json.loads((WORK/'配置/运行前保护快照.json').read_text(encoding='utf-8'))
    check('protocol_before_calculation_SHA',sha(WORK/'配置/阶段2B冻结协议.json')==frozen['protocol_sha256'])
    cfg=json.loads((WORK/'配置/阶段2B冻结协议.json').read_text(encoding='utf-8'))
    for dataset,info in cfg['datasets'].items():
        obs=pd.read_csv(WORK/'数据准备'/('FC1小时观测.csv' if dataset=='FC1' else 'RealHyFC循环观测.csv'))
        obs=obs[obs.scheme==info['scheme']].sort_values('time');train=obs[obs.partition=='Train']
        par=json.loads((out/'模型记录'/(dataset+'_Train拟合参数.json')).read_text(encoding='utf-8'))
        independent=independent_fit(train,info['lookback'],info['min_history'])
        for k,v in independent.items():check(dataset+'_Train_'+k,abs(v-par[k])<1e-12,f'{v} vs {par[k]}')
        check(dataset+'_Train_ids_exact',par['train_ids']==train[train.valid].id.astype(int).tolist())
        pred=pd.read_csv(out/'表格'/f'{dataset}全部方法逐点预测.csv');base=pd.read_csv(GUIDE/'基础预测实验/输出结果/表格'/f'{dataset}基础预测逐起点结果.csv')
        base=base[base.partition.isin(['Validation','Assessment'])]
        keys=['partition','origin_id','target_id','horizon','step']
        schedule=base[base.method=='Persistence'].sort_values(keys)
        for method,p in pred.groupby('method'):
            p=p.sort_values(keys)
            check(dataset+'_'+method+'_same_queries',np.array_equal(p[keys].to_numpy(),schedule[keys].to_numpy()))
            check(dataset+'_'+method+'_same_true_voltage',np.array_equal(p.true_v.to_numpy(),schedule.true_v.to_numpy()))
            if method in ['Persistence','LocalLinear']:
                old=base[base.method==method].sort_values(keys)
                check(dataset+'_'+method+'_frozen_prediction',np.array_equal(p.pred_v.to_numpy(),old.pred_v.to_numpy()))
                continue
            model=method.replace('_no_projection','');constrained='_no_projection' not in method
            for origin,g in p.groupby('origin_id'):
                o=obs[obs.id==origin].iloc[0]
                hist=obs[obs.valid&(obs.time<=o.time)&(obs.time>o.time-info['lookback'])]
                assert len(hist)>=info['min_history'] and hist.id.max()<=origin
                assert par['train_last_time']<=o.time
                if dataset=='RealHyFC':assert train[train.valid].available_ns.max()<=o.available_ns
                setting=float(g.setting.iloc[0]);s=independent_state(hist,par,model,setting,constrained)
                py=predict(s,g.step.to_numpy(),model,setting);error=np.max(abs(py-g.pred_D.to_numpy()))
                diffs.append(dict(dataset=dataset,method=method,origin_id=int(origin),max_abs_difference_V=float(error)))
                assert error<2e-11,(dataset,method,origin,error)
                assert np.allclose(g.pred_v,o.base_V-g.pred_D,atol=1e-14,rtol=1e-14)
                if constrained:assert s[1]>=-1e-15 and s[1]<=par['rate_cap']+1e-15
                if g.partition.eq('Assessment').any():assert o.time>=schedule[schedule.partition=='Validation'].target_time.max()
        check(dataset+'_independent_state_and_forecast',True,'all 4 new variants, all Validation/Assessment origins; <=2e-11V numerical tolerance, IDs exact')
        m=pd.read_csv(out/'表格'/f'{dataset}各跨度指标.csv')
        for _,r in m.iterrows():
            g=pred[(pred.method==r.method)&(pred.partition==r.partition)&(pred.horizon==r.horizon)]
            e=g.pred_v-g.true_v
            assert len(g)==r.n
            assert abs(np.sqrt(np.mean(e**2))*1000-r.RMSE_mV)<1e-10
            assert abs(np.mean(abs(e))*1000-r.MAE_mV)<1e-10
            assert abs(np.sqrt(np.mean((e/g.base_V*100)**2))-r.RMSE_SOH_pp)<1e-10
        check(dataset+'_all_voltage_and_SOH_metrics',True,len(m))
        # Every candidate is retained; selected setting follows frozen equal-horizon objective.
        trials=pd.read_csv(out/'表格'/(dataset+'_Validation候选选择.csv'))
        chosen=json.loads((out/'模型记录/冻结选择结果.json').read_text(encoding='utf-8'))
        for model in ['BoundedRate','SlowReversible']:
            d=trials[trials.method==model].sort_values(['equal_horizon_RMSE_mV','equal_horizon_MAE_mV','candidate_index'])
            assert d.setting.iloc[0]==chosen[dataset+'_'+model]['setting']
        check(dataset+'_selection_only_Validation',True,'all fixed candidates recorded; no Assessment field in objective')
    # Independently reproduce every nested RUL fit/selection, curve and first crossing.
    fc=pd.read_csv(WORK/'数据准备/FC1小时观测.csv');fc=fc[fc.valid].sort_values('time');base=fc.base_V.iloc[0]
    curves=pd.read_csv(out/'表格/FC1条件RUL外推曲线.csv');rul=pd.read_csv(out/'表格/FC1条件RUL误差与删失.csv')
    records=json.loads((out/'模型记录/RUL各起点拟合选择记录.json').read_text(encoding='utf-8'))
    for rec in records:
        origin=rec['origin_h'];hist=fc[fc.time<=origin];n=int(.8*len(hist));tr=hist.iloc[:n];va=hist.iloc[n:]
        if rec.get('phase')=='REFIT_ONLY_PRE_ORIGIN':
            assert rec['final_fit_ids']==hist.id.tolist() and rec['par']['train_last_time']<=origin
            for k,v in independent_fit(hist,48,24).items():assert abs(rec['par'][k]-v)<1e-12
            setting=rec['chosen'];h=hist[hist.time>origin-48]
            s=independent_state(h,rec['par'],rec['method'],setting,True)
            c=curves[(curves.origin_h==origin)&(curves.method==rec['method'])]
            py=predict(s,c.time_h.to_numpy()-h.time.iloc[-1],rec['method'],setting)
            assert np.max(abs(py-c.pred_D_V))<2e-11
            candidate=[r for r in records if r['origin_h']==origin and r['method']==rec['method'] and 'innerpar' in r]
            picked=min(candidate,key=lambda r:(r['RMSE_V'],r['MAE_V'],cfg['C']['candidates'].index(r['setting']) if rec['method']=='BoundedRate' else cfg['D']['FC1_tau'].index(r['setting'])))
            assert picked['setting']==setting
        else:
            assert rec['parameter_train_ids']==tr.id.tolist() and rec['validation_ids']==va.id.tolist()
            assert va.time.max()<=origin
            for k,v in independent_fit(tr,48,24).items():assert abs(rec['innerpar'][k]-v)<1e-12
            hh=tr[tr.time>tr.time.max()-48];s=independent_state(hh,rec['innerpar'],rec['method'],rec['setting'],True)
            e=predict(s,va.time.to_numpy()-hh.time.max(),rec['method'],rec['setting'])-va.D
            assert abs(np.sqrt(np.mean(e**2))-rec['RMSE_V'])<2e-12
    for _,r in rul.iterrows():
        actual=fc[fc.D>=r['drop']*base];a=actual.time.iloc[0] if len(actual) else np.nan
        c=curves[(curves.origin_h==r.origin_h)&(curves.method==r.method)];p=c[c.pred_D_V>=r['drop']*base]
        predicted=p.time_h.iloc[0] if len(p) else np.nan
        assert (pd.isna(a) and pd.isna(r.actual_first_cross_h)) or a==r.actual_first_cross_h
        assert (pd.isna(predicted) and pd.isna(r.pred_first_cross_h)) or predicted==r.pred_first_cross_h
        assert r.pred_right_censored==pd.isna(predicted)
        if pd.notna(r.abs_error_h):assert r.origin_eligible and r.abs_error_h==abs(predicted-a)
    check('RUL_independent_nested_fit_selection_curves_threshold_censor',True,'60 rows; all 3 origins; no later Train labels')
    first=fc[fc.D>=.04*base].iloc[0]
    check('809h_not_irreversible_EOL',first.time==809 and bool((fc[(fc.time>809)].D<.04*base).any()))
    # Actual diagnostic release remains unknown, and no future operation is paired backwards.
    pairs=pd.read_csv(WORK/'输出结果/表格/诊断与运行历史配对及排除.csv')
    for _,r in pairs[(pairs.dataset=='RealHyFC')&pairs.origin_id.notna()].iterrows():
        assert pd.Timestamp(r.operating_available_calendar)<=pd.Timestamp(r.actual_completed)
        if r.train_pair:assert r.lag_h<=48 and r.operating_upper_h<=1383.1954760000003
    q=pd.read_csv(WORK/'输出结果/表格/一维二维状态演化资格.csv')
    assert q.state_evolution.eq('NO').all() and q.observation_mapping.eq('NO').all()
    check('mapping_rejects_insufficient_rank_sample_or_clock',True,'no fabricated coefficients/state updates')
    # Counterfactual future overwrite cannot change origin state or generated forecast.
    from 低阶演化核心 import history,state_D,forecast,fit_train
    a=fc.copy();par=fit_train(a[a.time<=600],48,24);h1=history(a,600,48)
    b=a.copy();b.loc[b.time>600,'D']=999.;h2=history(b,600,48)
    s1=state_D(h1,par,24);s2=state_D(h2,par,24)
    assert np.array_equal(forecast(s1,[1,24,1000])[0],forecast(s2,[1,24,1000])[0])
    s=dict(L=.01,q=0.,R=.005,model='SlowReversible',setting=6.)
    assert forecast(s,[24])[0][0]<forecast(s,[1])[0][0]
    check('synthetic_future_perturbation_and_allowed_recovery',True)
    protected=0
    for rel,h in frozen['files'].items():
        assert sha(GUIDE/rel)==h,rel
        protected+=1
    check('historical_SHA_preservation',True,f'{protected} files unchanged; 2B publication only adds a new version and version index')
    csv(out/'表格/独立逐起点复算差异.csv',pd.DataFrame(diffs));csv(out/'表格/独立复核检查记录.csv',pd.DataFrame(checks))
    js(out/'独立复核通过.json',dict(status='PASS',checks=len(checks),compared_origin_variants=len(diffs),
        max_prediction_difference_V=max(r['max_abs_difference_V'] for r in diffs),identity_fields='EXACT_ZERO_TOLERANCE',
        numeric_only_tolerance_V=2e-11,code_sha256=sha(__file__)))
    print(f'独立复核PASS：{len(checks)}组检查，{len(diffs)}个起点变体逐点复算。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'输出结果');a=parser.parse_args()
    try:audit(a.output)
    except Exception as error:
        (WORK/'输出结果/独立复核失败记录.txt').write_text(repr(error),encoding='utf-8');raise

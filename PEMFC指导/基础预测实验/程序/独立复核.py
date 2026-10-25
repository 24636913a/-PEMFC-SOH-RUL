"""不用预测核心的独立复算：通用lstsq/增广岭回归/直接误差公式。"""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while c:=f.read(8*1024*1024):h.update(c)
    return h.hexdigest()

def ols(x,y,positive=False):
    x,y=np.asarray(x,float),np.asarray(y,float)
    co=np.linalg.lstsq(np.c_[np.ones(len(x)),x-x[-1]],y,rcond=None)[0]
    if positive and co[1]<0:co=np.array([y.mean(),0.])
    return lambda z:co[0]+co[1]*(np.asarray(z)-x[-1])

def run(out):
    checks=[]
    def check(name,condition,detail=''):
        checks.append(dict(check=name,passed=bool(condition),detail=str(detail)))
        if not condition:raise AssertionError((name,detail))
    def close(name,a,b,tol=2e-12):
        a,b=np.asarray(a,float),np.asarray(b,float)
        diff=float(np.max(abs(a-b))) if a.size else 0.
        check(name,np.allclose(a,b,atol=tol,rtol=1e-12,equal_nan=True),diff)
    freeze=json.loads((WORK/'配置/执行前冻结.json').read_text(encoding='utf-8'))
    check('frozen_protocol_SHA',sha(WORK/'配置/阶段2A冻结协议.json')==freeze['协议SHA256'])
    contract=json.loads((WORK/'配置/预测输入合同.json').read_text(encoding='utf-8'))
    for name,h in contract.items():check('observation_SHA_'+name,sha(WORK/'数据准备'/name)==h)
    for dataset,lb,mn,nb in [('FC1',48,24,10),('RealHyFC',12,6,5)]:
        obs=pd.read_csv(WORK/'数据准备'/('FC1小时观测.csv' if dataset=='FC1' else 'RealHyFC循环观测.csv'))
        pred=pd.read_csv(out/'表格'/f'{dataset}基础预测逐起点结果.csv')
        maxdiff=0.;groups=0
        for (scheme,origin,h),p in pred.groupby(['scheme','origin_id','horizon']):
            d=obs[obs.scheme==scheme].set_index('id');o=d.loc[origin]
            hist=d[(d.index<=origin)&(d.index>origin-lb)&d.valid]
            base=d[d.valid&(d.partition=='Train')].head(nb)
            assert len(base)==nb and base.index.max()<=origin
            assert len(hist)>=mn
            pp=p.drop_duplicates('target_id').sort_values('target_id')
            assert pp.target_id.tolist()==list(range(int(origin)+1,int(origin)+h+1))
            targets=d.loc[pp.target_id]
            assert targets.valid.all() and targets.partition.nunique()==1 and targets.partition.iloc[0]!='CROSS_BOUNDARY'
            f=ols(hist.time,hist.v)
            for method,p0 in p.groupby('method'):
                rec=f(d.loc[p0.target_id].time) if method=='LocalLinear' else np.repeat(o.v,len(p0))
                maxdiff=max(maxdiff,float(max(abs(np.asarray(rec)-p0.pred_v))))
                assert np.allclose(rec,p0.pred_v,atol=2e-12,rtol=1e-12)
                assert np.array_equal(d.loc[p0.target_id].v.to_numpy(),p0.true_v.to_numpy())
            groups+=1
        check(dataset+'_all_origins_causal_full_target_blocks',True,f'{groups} blocks; max independent voltage difference {maxdiff:.3g} V')
        met=pd.read_csv(out/'表格'/f'{dataset}连续预测指标.csv')
        for _,r in met[met.n>0].iterrows():
            p=pred[(pred.scheme==r.scheme)&(pred.partition==r.partition)&(pred.horizon==r.horizon)&(pred.method==r.method)]
            err=p.pred_v-p.true_v
            close('RMSE_'+str((dataset,r.scheme,r.partition,r.horizon,r.method)),np.sqrt((err**2).mean())*1000,r.RMSE_mV)
            close('SOH_'+str((dataset,r.scheme,r.partition,r.horizon,r.method)),np.sqrt(np.mean((err/p.base_V*100)**2)),r.RMSE_SOH_pp)
            assert len(p)==r.n
    obs=pd.read_csv(WORK/'数据准备/稀疏诊断观测.csv')
    sp=pd.read_csv(out/'表格/稀疏诊断全部预测与NA.csv')
    logs=json.loads((out/'模型记录/稀疏回归逐起点参数.json').read_text(encoding='utf-8'))
    diffs=[]
    for rec in logs:
        if rec['reason']!='OK':continue
        d=obs[(obs['电堆']==rec['dataset'])&(obs.scheme==rec['scheme'])&obs.primary]
        d=d.sort_values('标称小时' if rec['dataset']=='FC1' else '实际测试结束').reset_index(drop=True)
        tr=rec['historical_transitions'];assert all(b<=rec['origin_id'] and b==a+1 for a,b in tr)
        dims=1 if rec['method']=='Ridge1D' else 2
        X=np.array([[d.iloc[a].D_ref_V,d.iloc[a].D_low_V][:dims] for a,b in tr]);y=np.array([d.iloc[b].D_ref_V-d.iloc[a].D_ref_V for a,b in tr])
        mu=X.mean(0);sd=X.std(0);A=np.c_[np.ones(len(X)),(X-mu)/sd]
        aug=np.r_[A,np.c_[np.zeros(dims),np.eye(dims)*np.sqrt(rec['alpha'])]]
        co=np.linalg.lstsq(aug,np.r_[y,np.zeros(dims)],rcond=None)[0]
        close('Ridge_standardization_'+str((rec['dataset'],rec['origin_id'],dims)),mu,rec['mean'])
        close('Ridge_coefficients_'+str((rec['dataset'],rec['origin_id'],dims)),co,rec['coef'])
        target=sp[(sp.dataset==rec['dataset'])&(sp.scheme==rec['scheme'])&(sp.origin_id==rec['origin_id'])&(sp.method==rec['method'])]
        if target.pred_D.notna().all():
            o=d.iloc[rec['origin_id']];x=np.array([o.D_ref_V,o.D_low_V])[:dims]
            val=o.D_ref_V+np.r_[1,(x-mu)/sd]@co
            diffs.append(abs(val-target.pred_D.iloc[0]));assert diffs[-1]<1e-12
    check('sparse_augmented_least_squares_recalculation',True,f'{len(diffs)} predictions; max difference {max(diffs,default=0):.3g} V')
    fair=pd.read_csv(out/'表格/一维二维健康特征公平对比.csv')
    for key,g in fair.groupby(['dataset','scheme','origin_id']):
        assert g.target_id.nunique()==1 and g.true_D.nunique()==1 and g.method.nunique()==4
        assert g.n_past_transitions.nunique()==1 and g.training_label_last_id.max()<=key[-1]
    check('fair_same_targets_cutoffs_and_transition_sets',True,f'{len(fair)//4} targets; not independent stacks')
    # Recompute RUL selection without using core formula or future thresholds.
    fc=pd.read_csv(WORK/'数据准备/FC1小时观测.csv');fc=fc[fc.valid].sort_values('time');base=fc.base_V.iloc[0]
    rr=pd.read_csv(out/'表格/FC1_RUL阈值与删失结果.csv');cur=pd.read_csv(out/'表格/FC1_RUL外推曲线.csv')
    for origin in [400,500,600]:
        history=fc[fc.time<=origin];n=int(.8*len(history));train=history.iloc[:n];val=history.iloc[n:];t=train.time.max()
        cand=[]
        for w in [24,48,72]:
            sub=train[train.time>t-w]
            if len(sub)<24:continue
            e=ols(sub.time,sub.D,True)(val.time)-val.D
            cand.append((np.sqrt(np.mean(e**2)),np.mean(abs(e)),w))
        selected=min(cand)[2]
        for method in ['Persistence','LocalLinear']:
            cc=cur[(cur.origin_h==origin)&(cur.method==method)]
            if method=='Persistence':expected=np.repeat(history.D.iloc[-1],1000)
            else:
                sub=history[history.time>origin-selected];expected=ols(sub.time,sub.D,True)(cc.time_h)
                assert cc.selected_window_h.eq(selected).all()
            close(f'RUL_curve_{origin}_{method}',expected,cc.pred_D_V)
            for _,r in rr[(rr.origin_h==origin)&(rr.method==method)].iterrows():
                actual=fc[fc.D>=r['drop']*base];p=cc[cc.pred_D_V>=r['drop']*base]
                ac=actual.time.iloc[0] if len(actual) else np.nan;pc=p.time_h.iloc[0] if len(p) else np.nan
                assert (pd.isna(ac) and pd.isna(r.actual_first_cross_h)) or ac==r.actual_first_cross_h
                assert (pd.isna(pc) and pd.isna(r.pred_first_cross_h)) or pc==r.pred_first_cross_h
                assert r.pred_right_censored==pd.isna(pc)
                if pd.notna(r.abs_error_h):assert r.origin_eligible and r.abs_error_h==abs(pc-ac)
    check('RUL_nested_selection_first_crossing_and_censoring',True,'all 30 threshold/origin/method records')
    # Baselines must independently reproduce exactly the published input scales.
    b=fc[(fc.partition=='Train')].head(10)
    close('FC1_ten_hour_baseline',np.median(b.stack_V)/5,base)
    real=pd.read_csv(WORK/'数据准备/RealHyFC循环观测.csv')
    d=real[(real.scheme=='B60')&real.valid&(real.partition=='Train')].head(5)
    close('Real_five_cycle_baseline',np.median(d.v),d.base_V.iloc[0])
    # Past preparation frozen features are reused exactly, not reselected.
    old=pd.read_csv(WORK.parent/'健康特征提取/共同负载特征研究/输出结果/表格/路线B双参考特征.csv')
    for _,r in obs.iterrows():
        o=old[(old['文件名']==r['文件名'])&(old['方案']==r.scheme)].iloc[0]
        assert o['双参考支持']==r.valid
        assert (pd.isna(o.D_ref_V) and pd.isna(r.D_ref_V)) or o.D_ref_V==r.D_ref_V
        assert (pd.isna(o.D_low_V) and pd.isna(r.D_low_V)) or o.D_low_V==r.D_low_V
    check('frozen_stage1_2_features_unchanged',True)
    # Strong synthetic tests for forbidden interpolation/missing-block eligibility and small-sample rank.
    from 基础预测核心 import continuous,ridge_fit,linear
    x=pd.DataFrame(dict(id=np.arange(1,81),time=np.arange(1,81),partition=['Train']*80,
        scheme=['S']*80,valid=[True]*80,v=np.linspace(.7,.65,80),base_V=[.7]*80))
    x.loc[x.id==60,'valid']=False
    p,ex,_=continuous(x,'FC1',48,24,[24]);assert not ((p.origin_id<60)&(p.target_id>=60)).any()
    assert ex.reason.str.contains('MISSING_OR_REJECTED').any()
    assert ridge_fit(np.ones((8,2)),np.arange(8),1,6)[0] is None
    a,b=linear([0,1,2],[3,2,1],True);assert b==0
    check('synthetic_missing_block_rank_and_nonnegative_RUL',True)
    # Independent dynamic-programming LCS verifies the full-life extension implementation.
    from 准备基础数据 import lcs_bit
    rng=np.random.default_rng(20261025)
    for _ in range(30):
        a=rng.integers(0,6,20).tolist();b=rng.integers(0,6,25).tolist();dp=np.zeros((21,26),int)
        for i in range(20):
            for j in range(25):dp[i+1,j+1]=dp[i,j]+1 if a[i]==b[j] else max(dp[i,j+1],dp[i+1,j])
        assert lcs_bit(a,b)==dp[-1,-1]
    check('cycle_LCS_bitset_vs_independent_DP',True,'30 fixed synthetic sequences')
    cycles=pd.read_csv(WORK/'输出结果/表格/完整运行周期身份.csv')
    assert cycles.id.tolist()==list(range(1,201))
    assert np.array_equal(cycles.end_ns.to_numpy()[:-1],cycles.start_ns.to_numpy()[1:])
    complete=cycles.loaded_duration_h.between(8,11)&cycles.sequence_fraction.ge(.9)
    assert np.array_equal(cycles.complete.to_numpy(),complete.to_numpy())
    check('full_life_cycle_identity_completeness',True,'200 closed candidates; 189 sequence/duration eligible; first156 exact historical identity audited in preparation')
    # All historical files except explicitly backed-up formal README remain byte-identical.
    protected=0
    for rel,h in freeze['历史SHA256'].items():
        f=WORK.parent/rel
        if rel.replace('\\','/')=='完整代码/使用说明.md':continue
        assert f.exists() and sha(f)==h,(rel,'HISTORICAL_FILE_CHANGED')
        protected+=1
    check('historical_preservation_SHA256',True,f'{protected} original files unchanged; formal README separately backed up')
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'输出结果');args=parser.parse_args()
    try:
        result=run(args.output)
    except Exception as error:
        (WORK/'输出结果/独立复核失败记录.txt').write_text(repr(error),encoding='utf-8');raise
    pd.DataFrame(result).to_csv(args.output/'表格/独立复核记录.csv',index=False,encoding='utf-8-sig')
    (args.output/'独立复核通过.json').write_text(json.dumps(dict(status='PASS',checks=len(result),
        code_SHA256=sha(__file__)),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'独立复核通过：{len(result)}项，包括逐点预测、因果身份、内部选择和历史保护。')

"""独立复算：不导入迁移核心，不用保存的参数/状态产生预测。"""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd

TASK=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def qfit(x,y):
    x=np.asarray(x,float);A=np.column_stack([np.ones(len(x)),x-x[0]])
    return float(np.linalg.lstsq(A,np.asarray(y,float),rcond=None)[0][1])
def subset(d,end,source=False):
    last=d[d.id==end].iloc[0]
    return d[(d.h<=last.h)&((d.h>last.h-48) if source else (d.id>end-12))]
def refit(d,source=False):
    ref=1 if source else 12;minimum=24 if source else 6;base=d.base_V.iloc[0];rates=[];times=[]
    for r in d.itertuples():
        w=subset(d,r.id,source)
        if len(w)>=minimum:rates.append(qfit(w.h,w.d));times.append(r.h)
    cap=max(np.percentile(np.abs(rates),95),1e-6/base/ref)
    accel=max(.25*np.percentile(abs(np.diff(rates)/np.diff(times)),95),1e-8/base/ref**2)
    dif=np.diff(d.d);noise=max(np.median(abs(dif-np.median(dif)))/(.6744897501960817*math.sqrt(2)),1e-5/base)
    a=float(d.d.iloc[0]);last=d.h.iloc[0];res=[]
    for r in d.itertuples():
        decay=math.exp(-(r.h-last)/(24 if source else 72));a=decay*a+(1-decay)*r.d;res.append(r.d-a);last=r.h
    return dict(q=qfit(d.h,d.d),cap=cap,accel=accel,noise=noise,rev=max(np.std(res),noise))
def clip(q,old,dt,p):return min(min(p['cap'],old+p['accel']*dt),max(max(0,old-p['accel']*dt),q))
def cstate(d,p,k):
    q=min(p['cap'],max(0,p['q']));last=d.h.iloc[0]
    for i,r in enumerate(d.itertuples()):
        raw=p['q'] if i+1<6 else k*qfit(d.h.iloc[:i+1],d.d.iloc[:i+1])+(1-k)*p['q']
        q=clip(raw,q,r.h-last,p);last=r.h
    return np.array([d.d.iloc[-1],q,0.])
def dstate(d,p,tau):
    state=np.array([d.d.iloc[0],min(p['cap'],max(0,p['q'])),0.]);cov=np.diag([p['rev']**2,(p['cap']/4)**2,p['rev']**2]);last=d.h.iloc[0];h=np.array([[1.,0,1.]])
    for r in d.iloc[1:].itertuples():
        dt=r.h-last;dec=math.exp(-dt/tau);f=np.array([[1,dt,0],[0,1,0],[0,0,dec]],float)
        process=np.diag([(p['noise']/20)**2*dt/12,(p['accel']*12)**2*dt/12,p['rev']**2*(1-dec*dec)])
        xp=f@state;pp=f@cov@f.T+process;innovation=float((h@pp@h.T).item()+p['noise']**2);gain=pp@h.T/innovation
        updated=xp+gain[:,0]*(r.d-float((h@xp).item()))
        # 独立使用线性滤波等价协方差形式；与主程序Joseph形式比较。
        cov=pp-gain@h@pp;updated[1]=clip(updated[1],state[1],dt,p);state=updated;last=r.h
    return state
def forward(state,dt,tau=None):
    return state[0]+state[1]*np.asarray(dt)+(0 if tau is None else state[2]*np.exp(-np.asarray(dt)/tau))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=TASK/'输出结果/正式计算');a=ap.parse_args();run=a.run
    cfg=json.loads((TASK/'配置/阶段2C冻结协议.json').read_text(encoding='utf-8'));data=TASK/'数据准备'
    fc=pd.read_csv(data/'FC1小时观测.csv');real=pd.read_csv(data/'RealHyFC循环观测.csv');real=real[real.scheme=='B60'].copy()
    fc['h']=fc.time;real['h']=real.available_h
    # 独立由原始冻结电压/base构造相对损失，而不是读取主程序中间d。
    for d in [fc,real]:d['d']=1-d.v/d.base_V
    source=refit(fc[fc.valid&(fc.partition=='Train')],True);sq=min(source['cap'],max(0,source['q']))
    pred=pd.read_csv(run/'表格/各方法逐起点预测.csv');savedmetrics=pd.read_csv(run/'表格/电压_SOH及损失误差.csv');checks=[];diffs=[]
    def check(name,passed,detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
        if not passed:raise AssertionError(name+': '+str(detail))
    check('预登记SHA未改变',sha(TASK/'配置/阶段2C冻结协议.json')==json.loads((TASK/'配置/预登记时间与SHA256.json').read_text(encoding='utf-8'))['protocol_sha256'])
    check('源域仅Train且669有效点',len(fc[fc.valid&(fc.partition=='Train')])==669)
    common=real[(real.id>51)&(real.operating_start_h>=629.2038)];truth=common[common.valid].copy();check('共同目标141且未按误差筛除',len(truth)==141)
    for sc in cfg['scenarios']:
        name=sc['name'];cut=sc['cutoff_h'];hist=real[real.valid&(real.h<=cut)].sort_values('h');p=refit(hist);anchor=int(real[real.h<=cut].id.max());n=int(.8*len(hist));tr=hist.iloc[:n];va=hist.iloc[n:];pi=refit(tr);innerhist=subset(tr,int(tr.id.iloc[-1]));chosen={}
        for typ,settings in [('Bounded',cfg['bounded_k']),('SlowReversible',cfg['reversible_tau_h'])]:
            opts=[]
            for j,s in enumerate(settings):
                state=cstate(innerhist,pi,s) if typ=='Bounded' else dstate(innerhist,pi,s)
                e=forward(state,12*(va.id.to_numpy()-int(tr.id.iloc[-1])),None if typ=='Bounded' else s)-va.d.to_numpy()
                opts.append((np.sqrt(np.mean(e**2)),np.mean(abs(e)),j,s))
            chosen[typ]=min(opts)[3]
        local=subset(hist,int(hist.id.iloc[-1]));tq=min(p['cap'],max(0,p['q']));states={'A_Persistence':(np.array([hist.d.iloc[-1],0,0]),None),'A_TargetBounded':(np.array([hist.d.iloc[-1],tq,0]),None),
          'B_FC1Direct':(np.array([hist.d.iloc[-1],sq,0]),None),'C_FC1PriorAdapted':(np.array([hist.d.iloc[-1],.25*sq+.75*tq,0]),None),
          'D_Target2B_Bounded':(cstate(local,p,chosen['Bounded']),None),'D_Target2B_SlowReversible':(dstate(local,p,chosen['SlowReversible']),chosen['SlowReversible'])}
        record=json.loads((run/f'模型记录/{name}_参数与状态.json').read_text(encoding='utf-8'))
        check(name+'历史量与身份',record['n_history']==len(hist) and record['history_ids']==hist.id.astype(int).tolist())
        check(name+'候选选择独立一致',record['selected']==chosen)
        check(name+'训练不越截止',hist.h.max()<=cut and tr.h.max()<va.h.min())
        for method,(s,tau) in states.items():
            got=pred[(pred.scenario==name)&(pred.method==method)].sort_values('target_id');check(name+method+'精确目标身份',got.target_id.tolist()==truth.id.astype(int).tolist())
            lead=(cut-hist.h.iloc[-1])+12*(truth.id.to_numpy()-anchor);ind=forward(s,lead,tau);delta=np.abs(ind-got.pred_d.to_numpy());maxdiff=float(max(delta));diffs.append(dict(scenario=name,method=method,n=len(got),max_abs_d_difference=maxdiff))
            check(name+method+'独立预测一致',maxdiff<1e-10,maxdiff)
            check(name+method+'标签未改',np.max(abs(got.true_v.to_numpy()-truth.v.to_numpy()))<1e-14)
            check(name+method+'无未来更新',(got.future_observation_updates==0).all() and (got.nominal_lead_h==12*(got.target_id-anchor)).all())
            check(name+method+'电压SOH尺度',np.max(abs(got.pred_v-got.base_V*got.pred_soh))<1e-14)
            for row in savedmetrics[(savedmetrics.scenario==name)&(savedmetrics.method==method)&(savedmetrics.status=='EVALUABLE')].itertuples():
                if row.scope=='COMMON_ALL':g=got
                elif row.scope.startswith('LEGACY_'):g=got[got.target_legacy_partition==row.scope[7:]]
                else:g=got[got.target_id<52+int(row.scope.split('_')[2])]
                e=(g.pred_v-g.true_v).to_numpy()*1000;se=(g.pred_soh-g.true_soh).to_numpy()*100
                check(name+method+row.scope+'指标',len(g)==row.n and abs(math.sqrt(np.mean(e*e))-row.RMSE_mV)<1e-9 and abs(np.mean(abs(e))-row.MAE_mV)<1e-9 and abs(math.sqrt(np.mean(se*se))-row.RMSE_SOH_pp)<1e-10)
        na=savedmetrics[(savedmetrics.scenario==name)&(savedmetrics.scope=='COMMON_COMPLETE_25_CYCLES')]
        check(name+'25循环不补缺口',len(na)==6 and (na.status=='NA_MISSING_FROZEN_CYCLE').all())
    # 未来时间重写不会影响预测：独立反事实仅改变评估坐标，预测仅由冻结名义ID排程决定。
    g=pred.iloc[:141];check('名义时长不读取真实未来时长',not np.allclose(g.actual_lead_h,g.nominal_lead_h))
    gains=pd.read_csv(run/'表格/迁移收益与负迁移.csv')
    for r in gains.itertuples():
        mt=savedmetrics[(savedmetrics.scenario==r.scenario)&(savedmetrics.scope==r.scope)].set_index('method')
        expected=100*(1-mt.loc[r.transfer,'RMSE_mV']/mt.loc[r.reference,'RMSE_mV'])
        check('迁移收益复算_'+str(r.Index),abs(expected-r.RMSE_gain_pct)<1e-9)
    for row in pd.read_csv(TASK/'配置/历史成果保护SHA256.csv').itertuples():check('历史SHA_'+str(row.Index),sha(TASK.parents[2]/row.path)==row.sha256)
    pd.DataFrame(diffs).to_csv(run/'表格/独立逐模型复算差异.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(run/'表格/独立数值复核检查.csv',index=False,encoding='utf-8-sig')
    result=dict(status='PASS',checks=len(checks),n_independent_model_forecasts=len(diffs),max_abs_d_difference=max(x['max_abs_d_difference'] for x in diffs),historical_files_unchanged=1050,method='独立lstsq、独立滤波与协方差更新、独立传播和指标；未导入生产核心；全部旧SHA核对')
    (run/'独立复核通过.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

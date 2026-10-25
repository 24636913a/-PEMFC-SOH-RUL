"""无文件副作用的低阶迁移数学核心。状态损失无量纲，速率统一 /h。"""
import numpy as np
import pandas as pd

H=np.array([1.,0.,1.])
def slope(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);z=x-x.mean()
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.sum(z*z)<=0:raise ValueError('INVALID_OLS')
    q=float(z@(y-y.mean())/(z@z));return float(y.mean()-q*x.mean()),q

def prepare(fc,real):
    fc=fc.copy();real=real[real.scheme=='B60'].copy()
    for df,coord in [(fc,'time'),(real,'available_h')]:
        df['h']=df[coord].astype(float);df['d']=1-df.SOH
        good=df[df.valid]
        assert np.allclose(good.d,good.D/good.base_V,rtol=0,atol=1e-12)
        assert good.id.is_unique and np.isfinite(good[['h','v','d']]).all().all()
    return fc.sort_values('h'),real.sort_values('id')

def window(obs,last_id,dataset):
    end=obs.loc[obs.id==last_id].iloc[0]
    return obs[(obs.h<=end.h)&((obs.h>end.h-48) if dataset=='FC1' else (obs.id>int(last_id)-12))].sort_values('h')

def fit(obs,dataset):
    obs=obs[obs.valid].sort_values('h');minimum=24 if dataset=='FC1' else 6;ref=1. if dataset=='FC1' else 12.
    if len(obs)<max(2*minimum,24 if dataset=='FC1' else 12):raise ValueError('INSUFFICIENT_FIT_HISTORY')
    rates=[];times=[];ids=[]
    for r in obs.itertuples():
        w=window(obs,r.id,dataset)
        if len(w)>=minimum:
            rates.append(slope(w.h,w.d)[1]);times.append(float(r.h));ids.append(int(r.id))
    if len(rates)<2:raise ValueError('INSUFFICIENT_LOCAL_RATES')
    base=float(obs.base_V.iloc[0]);a,q=slope(obs.h,obs.d)
    cap=max(float(np.quantile(np.abs(rates),.95)),1e-6/(base*ref))
    accel=max(.25*float(np.quantile(np.abs(np.diff(rates)/np.diff(times)),.95)),1e-8/(base*ref**2))
    diff=np.diff(obs.d);noise=max(float(np.median(abs(diff-np.median(diff)))/(.6744897501960817*np.sqrt(2))),1e-5/base)
    tc=24. if dataset=='FC1' else 72.;smooth=float(obs.d.iloc[0]);last=float(obs.h.iloc[0]);res=[]
    for r in obs.itertuples():
        smooth+=(1-np.exp(-(r.h-last)/tc))*(r.d-smooth);res.append(r.d-smooth);last=r.h
    return dict(dataset=dataset,n_fit=len(obs),fit_ids=obs.id.astype(int).tolist(),first_h=float(obs.h.min()),last_h=float(obs.h.max()),
      base_V=base,q_intercept=a,q_prior=q,q_bounded=float(np.clip(q,0,cap)),rate_cap=cap,acceleration_cap=accel,
      noise=noise,reversible_scale=max(float(np.std(res)),noise),dt_ref_h=ref,min_history=minimum,
      local_rates=rates,local_rate_h=times,local_rate_ids=ids,unit='dimensionless relative loss / actual past operating hour')

def project(q,old,dt,p):
    lo=max(0,old-p['acceleration_cap']*dt);hi=min(p['rate_cap'],old+p['acceleration_cap']*dt)
    return float(np.clip(q,lo,hi))

def state_bounded(hist,p,k):
    hist=hist.sort_values('h');q=float(np.clip(p['q_prior'],0,p['rate_cap']));last=float(hist.h.iloc[0]);clipped=0
    for i,r in enumerate(hist.itertuples()):
        sub=hist.iloc[:i+1];raw=p['q_prior']
        if len(sub)>=p['min_history']:raw=k*slope(sub.h,sub.d)[1]+(1-k)*p['q_prior']
        q1=project(raw,q,r.h-last,p);clipped+=int(abs(q1-raw)>1e-15);q=q1;last=r.h
    return dict(L=float(hist.d.iloc[-1]),q=q,R=0.,tau_h=None,last_h=last,n_history=len(hist),history_ids=hist.id.astype(int).tolist(),clipped_steps=clipped,measurement_updates=len(hist))

def state_reversible(hist,p,tau):
    hist=hist.sort_values('h');q=float(np.clip(p['q_prior'],0,p['rate_cap']));x=np.array([hist.d.iloc[0],q,0.]);rs=p['reversible_scale'];noise=p['noise'];ref=p['dt_ref_h']
    P=np.diag([rs**2,(.25*p['rate_cap'])**2,rs**2]);last=float(hist.h.iloc[0]);clipped=0
    for r in hist.iloc[1:].itertuples():
        dt=float(r.h-last);decay=np.exp(-dt/tau);F=np.array([[1,dt,0],[0,1,0],[0,0,decay]],float)
        Q=np.diag([(.05*noise)**2*dt/ref,(p['acceleration_cap']*ref)**2*dt/ref,rs**2*(1-decay**2)])
        old=float(x[1]);xp=F@x;Pp=F@P@F.T+Q;K=Pp@H/float(H@Pp@H+noise**2)
        x=xp+K*(r.d-H@xp);I=np.eye(3)-np.outer(K,H);P=I@Pp@I.T+np.outer(K,K)*noise**2
        raw=float(x[1]);x[1]=project(raw,old,dt,p);clipped+=int(abs(raw-x[1])>1e-15);last=float(r.h)
    # q状态按一个参考步长缩放后再检查，避免条件数仅由时间单位改变。
    F=np.array([[1,1,0],[0,1,0],[0,0,np.exp(-ref/tau)]],float);O=np.vstack([H,H@F,H@F@F]);cond=float(np.linalg.cond(O))
    if np.linalg.matrix_rank(O)!=3 or cond>1e6:raise ValueError('OBSERVABILITY_FAILED')
    return dict(L=float(x[0]),q=float(x[1]),R=float(x[2]),tau_h=float(tau),last_h=last,n_history=len(hist),history_ids=hist.id.astype(int).tolist(),clipped_steps=clipped,measurement_updates=len(hist)-1,P=P.tolist(),observability_condition=cond)

def predict(s,lead):
    lead=np.asarray(lead,float)
    if np.any(lead < -1e-9):raise ValueError('NEGATIVE_PROPAGATION_TIME')
    slow=s['L']+s['q']*lead;rev=np.zeros_like(lead) if s['tau_h'] is None else s['R']*np.exp(-lead/s['tau_h'])
    return slow+rev,slow,rev

def propagate_to_cut(s,cut):
    lead=float(cut-s['last_h']);d,l,r=predict(s,lead)
    return dict(s,L=float(l),R=float(r),last_h=float(cut),known_past_propagation_h=lead)

def tune(history,protocol):
    history=history.sort_values('h');n=int(np.floor(.8*len(history)));tr=history.iloc[:n];va=history.iloc[n:]
    p=fit(tr,'RealHyFC');hist=window(tr,int(tr.id.iloc[-1]),'RealHyFC')
    if len(hist)<6:raise ValueError('INNER_STATE_TOO_SHORT')
    lead=12*(va.id.to_numpy()-int(tr.id.iloc[-1]));records=[];selected={}
    for name,candidates in [('Bounded',protocol['bounded_k']),('SlowReversible',protocol['reversible_tau_h'])]:
        values=[]
        for order,setting in enumerate(candidates):
            s=state_bounded(hist,p,setting) if name=='Bounded' else state_reversible(hist,p,setting)
            y=predict(s,lead)[0];e=y-va.d.to_numpy();rmse=float(np.sqrt(np.mean(e*e)));mae=float(np.mean(abs(e)))
            values.append((rmse,mae,order,setting))
            records.append(dict(model=name,setting=setting,RMSE_d=rmse,MAE_d=mae,n_train=len(tr),n_validation=len(va),train_last_h=float(tr.h.iloc[-1]),first_validation_h=float(va.h.iloc[0]),train_last_id=int(tr.id.iloc[-1]),validation_ids=';'.join(map(str,va.id.astype(int))),nominal_plan='12h per raw-cycle step, actual future lead not input'))
        selected[name]=min(values)[3]
    return selected,records

def build_models(history,source,cut,protocol):
    if history.empty or history.h.max()>cut or not history.valid.all():raise ValueError('HISTORY_AFTER_CUTOFF_OR_INVALID')
    p=fit(history,'RealHyFC');selected,inner=tune(history,protocol);last=history.iloc[-1]
    def scalar(q):return dict(L=float(last.d),q=float(q),R=0.,tau_h=None,last_h=float(last.h),n_history=len(history),history_ids=history.id.astype(int).tolist(),measurement_updates=0,clipped_steps=0)
    w=protocol['prior_weight_source'];local=window(history,int(last.id),'RealHyFC')
    if len(local)<6:raise ValueError('STATE_HISTORY_TOO_SHORT')
    states={'A_Persistence':scalar(0),'A_TargetBounded':scalar(p['q_bounded']),'B_FC1Direct':scalar(source['q_bounded']),
      'C_FC1PriorAdapted':scalar(w*source['q_bounded']+(1-w)*p['q_bounded']),
      'D_Target2B_Bounded':state_bounded(local,p,selected['Bounded']),
      'D_Target2B_SlowReversible':state_reversible(local,p,selected['SlowReversible'])}
    out={}
    for name,state in states.items():out[name]=dict(at_last_observation=state,at_cutoff=propagate_to_cut(state,cut))
    return p,selected,inner,out

def metrics(g):
    e=(g.pred_v-g.true_v).to_numpy();s=(g.pred_soh-g.true_soh).to_numpy()*100;de=(g.pred_d-g.true_d).to_numpy()
    return dict(n=len(g),unique_targets=g.target_id.nunique(),origins=1,RMSE_mV=float(np.sqrt(np.mean(e**2))*1000),MAE_mV=float(np.mean(abs(e))*1000),
      RMSE_SOH_pp=float(np.sqrt(np.mean(s*s))),MAE_SOH_pp=float(np.mean(abs(s))),RMSE_d=float(np.sqrt(np.mean(de*de))),MAE_d=float(np.mean(abs(de))),
      P95_mV=float(np.quantile(abs(e),.95)*1000),max_mV=float(max(abs(e))*1000),actual_lead_min_h=float(g.actual_lead_h.min()),actual_lead_max_h=float(g.actual_lead_h.max()),nominal_lead_min_h=float(g.nominal_lead_h.min()),nominal_lead_max_h=float(g.nominal_lead_h.max()))

"""阶段2B低阶演化：只拟合训练统计；每起点因果窗状态，无神经网络。"""
import numpy as np
import pandas as pd

H=np.array([1.,0.,1.])

def ols(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float);xm=x.mean();ym=y.mean()
    den=np.sum((x-xm)**2);b=float(np.sum((x-xm)*(y-ym))/den) if den>0 else 0.
    return float(ym-b*xm),b


def history(obs,origin,lookback):
    return obs[obs.valid & (obs.time<=origin)&(obs.time>origin-lookback)].sort_values('time')


def fit_train(train,lookback,minimum):
    """Train数据直接统计估计，记录全部拟合身份，不读取Validation。"""
    d=train[train.valid].sort_values('time');qs=[];ts=[]
    if len(d)<max(minimum*2,24 if lookback==48 else 12):
        raise ValueError('INSUFFICIENT_TRAIN_OBSERVATIONS')
    for row in d.itertuples():
        sub=history(d,row.time,lookback)
        if len(sub)>=minimum:
            qs.append(ols(sub.time,sub.D)[1]);ts.append(row.time)
    if len(qs)<2:raise ValueError('INSUFFICIENT_TRAIN_LOCAL_RATES')
    a,q0=ols(d.time,d.D);qcap=max(float(np.quantile(np.abs(qs),.95)),1e-6)
    accel=max(.25*float(np.quantile(np.abs(np.diff(qs)/np.diff(ts)),.95)),1e-8)
    dif=np.diff(d.D);noise=max(float(np.median(abs(dif-np.median(dif)))/(.6744897501960817*np.sqrt(2))),1e-5)
    slow=float(d.D.iloc[0]);res=[];last=float(d.time.iloc[0]);tc=lookback/2
    for row in d.itertuples():
        gamma=1-np.exp(-(row.time-last)/tc)
        slow+=gamma*(row.D-slow);res.append(row.D-slow);last=row.time
    reversible=max(float(np.std(res)),noise)
    return dict(q_prior=q0,rate_cap=qcap,acceleration_cap=accel,noise=noise,reversible_scale=reversible,
        n_train=len(d),train_first_time=float(d.time.min()),train_last_time=float(d.time.max()),
        train_ids=d.id.astype(int).tolist(),local_rate_times=list(map(float,ts)),local_rates=list(map(float,qs)),
        lookback=lookback,min_history=minimum,loss_unit='V',coordinate_unit='hour' if lookback==48 else 'physical_cycle')


def project(q,old,dt,par):
    lo=max(0.,old-par['acceleration_cap']*dt);hi=min(par['rate_cap'],old+par['acceleration_cap']*dt)
    return float(np.clip(q,lo,hi))


def state_C(hist,par,k,constrained=True):
    hist=hist.sort_values('time');q=float(par['q_prior'])
    if constrained:q=float(np.clip(q,0,par['rate_cap']))
    previous=float(hist.time.iloc[0]);clipped=0
    for n,row in enumerate(hist.itertuples()):
        prefix=hist.iloc[:n+1];raw=par['q_prior']
        if len(prefix)>=par['min_history']:raw=k*ols(prefix.time,prefix.D)[1]+(1-k)*par['q_prior']
        candidate=project(raw,q,row.time-previous,par) if constrained else raw
        clipped+=int(abs(candidate-raw)>1e-15);q=float(candidate);previous=row.time
    return dict(L=float(hist.D.iloc[-1]),q=q,R=0.,last_time=float(hist.time.iloc[-1]),
                n_history=len(hist),clipped_steps=clipped,updates=len(hist),history_ids=hist.id.astype(int).tolist(),
                model='BoundedRate',setting=float(k),constrained=constrained,origin_residual_V=0.)


def transition(dt,tau):
    return np.array([[1.,dt,0.],[0.,1.,0.],[0.,0.,np.exp(-dt/tau)]])


def state_D(hist,par,tau,constrained=True):
    hist=hist.sort_values('time');q0=par['q_prior']
    if constrained:q0=float(np.clip(q0,0,par['rate_cap']))
    x=np.array([hist.D.iloc[0],q0,0.]);rs=par['reversible_scale'];noise=par['noise'];cap=par['rate_cap']
    P=np.diag([rs**2,(.25*cap)**2,rs**2]);last=float(hist.time.iloc[0]);clipped=0
    # Initial point defines state; each subsequent point is one genuine causal observation update.
    for row in hist.iloc[1:].itertuples():
        dt=float(row.time-last);F=transition(dt,tau)
        Q=np.diag([(.05*noise)**2*dt,par['acceleration_cap']**2*dt,rs**2*(1-np.exp(-2*dt/tau))])
        old=float(x[1]);xp=F@x;Pp=F@P@F.T+Q
        S=float(H@Pp@H+noise**2);K=Pp@H/S
        x=xp+K*(row.D-H@xp);I=np.eye(3)-np.outer(K,H)
        P=I@Pp@I.T+np.outer(K,K)*noise**2
        raw=float(x[1])
        if constrained:x[1]=project(raw,old,dt,par)
        clipped+=int(abs(x[1]-raw)>1e-15);last=float(row.time)
    F=transition(1.,tau);O=np.vstack([H,H@F,H@F@F]);rank=np.linalg.matrix_rank(O);cond=np.linalg.cond(O)
    if rank!=3 or cond>1e6:raise ValueError('UNOBSERVABLE_OR_ILL_CONDITIONED_STATE')
    return dict(L=float(x[0]),q=float(x[1]),R=float(x[2]),last_time=last,n_history=len(hist),
                clipped_steps=clipped,updates=len(hist)-1,history_ids=hist.id.astype(int).tolist(),
                model='SlowReversible',setting=float(tau),constrained=constrained,
                origin_residual_V=float(hist.D.iloc[-1]-(x[0]+x[2])),observability_rank=int(rank),
                observability_condition=float(cond),P=P.tolist())


def forecast(state,offset):
    """offset为已知相对小时或循环步；没有真实未来工况/时间输入。"""
    offset=np.asarray(offset,float)
    slow=state['L']+state['q']*offset
    rev=state['R']*np.exp(-offset/state['setting']) if state['model']=='SlowReversible' else np.zeros_like(offset)
    return slow+rev,slow,rev


def estimate(hist,par,method,setting,constrained=True):
    return state_C(hist,par,setting,constrained) if method=='BoundedRate' else state_D(hist,par,setting,constrained)


def predict_schedule(schedule,obs,par,method,setting,constrained=True):
    rows=[];states=[]
    for origin,p in schedule.groupby('origin_id',sort=True):
        o=obs[obs.id==origin].iloc[0];hist=history(obs,o.time,par['lookback'])
        if len(hist)<par['min_history']:raise ValueError('SCHEDULE_HISTORY_CHANGED')
        s=estimate(hist,par,method,setting,constrained)
        state=dict(s,origin_id=int(origin),origin_time=float(o.time),dataset=p.dataset.iloc[0],scheme=o.scheme,
                   parameters_train_cutoff=par['train_last_time'])
        states.append(state)
        for r in p.to_dict('records'):
            # Schedule target_time is only identity; actual lead_h is not used by forecast.
            offsets=float(r['step'])
            pred,slow,rev=forecast(s,offsets)
            name=method if constrained else method+'_no_projection'
            r.update(method=name,pred_D=float(pred),pred_v=float(o.base_V-pred),pred_SOH=float(1-pred/o.base_V),
                slow_D=float(slow),reversible_D=float(rev),q=float(s['q']),setting=setting,
                training_cutoff=par['train_last_time'],history_first_id=int(hist.id.min()),
                history_last_id=int(hist.id.max()),n_history=len(hist))
            rows.append(r)
    return pd.DataFrame(rows),states


def scores(pred,keys):
    out=[]
    if pred.empty:return pd.DataFrame(columns=keys+['n','origins','RMSE_mV','MAE_mV','RMSE_SOH_pp','MAE_SOH_pp'])
    for key,g in pred.groupby(keys,dropna=False,sort=True):
        if not isinstance(key,tuple):key=(key,)
        e=g.pred_v.to_numpy()-g.true_v.to_numpy();se=e/g.base_V.to_numpy()*100
        out.append(dict(zip(keys,key),n=len(g),origins=g.origin_id.nunique(),unique_targets=g.target_id.nunique(),
            RMSE_mV=float(np.sqrt(np.mean(e**2))*1000),MAE_mV=float(np.mean(abs(e))*1000),
            P95_mV=float(np.quantile(abs(e),.95)*1000),max_mV=float(max(abs(e))*1000),
            RMSE_SOH_pp=float(np.sqrt(np.mean(se**2))),MAE_SOH_pp=float(np.mean(abs(se)))))
    return pd.DataFrame(out)


def objective(pred):
    m=scores(pred,['horizon'])
    eligible=m[m.origins>=3]
    if eligible.empty:raise ValueError('NO_VALIDATION_HORIZON_WITH_3_ORIGINS')
    return float(eligible.RMSE_mV.mean()),float(eligible.MAE_mV.mean())

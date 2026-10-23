"""One preregistered H2 CPU experiment; no H1/H3 fit entry point."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import math,sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from common import *
def feature(d,o):
 g=d[(d.measurement_end_h>=o.measurement_end_h-96)&(d.measurement_end_h<=o.measurement_end_h)&(d.conservative_available_end_h<=o.conservative_available_end_h)].sort_values('measurement_end_h')
 x=g.measurement_end_h.to_numpy();y=g.R.to_numpy();span=float(np.ptp(x));slope=float((x-x.mean())@(y-y.mean())/sum((x-x.mean())**2)) if len(x)>1 else 0.
 return dict(zip(FEATURES,[float(o.R),float(y.mean()),float(y.std()),slope,float(o.R-np.median(y))])),len(g),span,g.raw_cycle_id.astype(int).tolist()
def cohort(cfg):
 d=read('identity').sort_values('measurement_end_h');ids=d.set_index('raw_cycle_id');cat=read('catalog').set_index('raw_cycle_id');m=read('measurement');m=m[m.stable_window.eq('middle_50')].set_index('cycle_interval_id');rows=[];ex=[];ev=[]
 for o in d.itertuples():
  x,n,span,history=feature(d,o);ev.append({'origin_raw_cycle_id':int(o.raw_cycle_id),'history_raw_ids':history,'n':n,'span_h':span,**x})
  for fold,(lo,mid,hi) in cfg['folds'].items():
   target=int(o.raw_cycle_id)+2;reason=[];key=f'H2_{fold}_{int(o.raw_cycle_id)}_{target}'
   if n<5 or span<67.2:reason.append('96H_HISTORY_NOT_QUALIFIED')
   if target not in ids.index:reason.append('NO_MEASURED_TARGET')
   if any(raw not in cat.index or not bool(cat.loc[raw,'complete_cycle']) for raw in range(int(o.raw_cycle_id),target+1)):reason.append('INCOMPLETE_PHYSICAL_CYCLE')
   if target in ids.index:
    t=ids.loc[target];role='train' if lo<=t.measurement_start_h and t.conservative_available_end_h<mid else 'validation' if mid<=t.measurement_start_h and t.conservative_available_end_h<hi else None
    if role is None:reason.append('TARGET_OUTSIDE_ROLE')
   if reason:ex.append({'sample_id':key,'fold':fold,'origin_raw_cycle_id':int(o.raw_cycle_id),'target_raw_cycle_id':target,'reason':'|'.join(reason)});continue
   oc=m.loc[o.cycle_interval_id,'measurement_end_datetime'];tc=m.loc[t.cycle_interval_id,'measurement_end_datetime']
   rows.append({'sample_id':key,'H':2,'fold':fold,'role':role,'origin_raw_cycle_id':int(o.raw_cycle_id),'target_raw_cycle_id':target,
    'origin_cycle_interval_id':o.cycle_interval_id,'target_cycle_interval_id':t.cycle_interval_id,'origin_available_h':float(o.conservative_available_end_h),'target_available_h':float(t.conservative_available_end_h),
    'origin_actual_end_h':float(o.measurement_end_h),'target_actual_start_h':float(t.measurement_start_h),'target_actual_end_h':float(t.measurement_end_h),
    'origin_calendar':oc,'target_calendar':tc,'actual_lead_h_AUDIT_ONLY':float(t.measurement_end_h-o.measurement_end_h),
    'calendar_lead_h_AUDIT_ONLY':(pd.Timestamp(tc)-pd.Timestamp(oc)).total_seconds()/3600,'history_n':n,'history_span_h':span,'target_R':float(t.R),**x})
 frame=pd.DataFrame(rows);audit=[];used=[]
 for fold in cfg['folds']:
  a=frame[frame.fold.eq(fold)];val=a[a.role.eq('validation')].sort_values('origin_available_h');train=a[a.role.eq('train')].sort_values('target_actual_end_h');cut=float(val.origin_available_h.min()) if len(val) else cfg['folds'][fold][1]
  bad=train[train.target_available_h>cut];train=train[train.target_available_h<=cut];selection=train.iloc[-math.ceil(.15*len(train)):];inner=train.iloc[:-len(selection)];ic=float(selection.origin_available_h.min()) if len(selection) else -1;purged=inner[inner.target_available_h>ic];inner=inner[inner.target_available_h<=ic]
  for r in bad.itertuples():ex.append({'sample_id':r.sample_id,'fold':fold,'origin_raw_cycle_id':r.origin_raw_cycle_id,'target_raw_cycle_id':r.target_raw_cycle_id,'reason':'OUTER_TRAIN_LABEL_UNAVAILABLE'})
  audit.append({'fold':fold,'Train_n':len(train),'Validation_n':len(val),'inner_fit_n':len(inner),'inner_selection_n':len(selection),'inner_purged_n':len(purged),'outer_label_max_h':float(train.target_available_h.max()) if len(train) else None,'first_validation_origin_h':cut,'inner_cutoff_h':ic,'inner_label_max_h':float(inner.target_available_h.max()) if len(inner) else None,'qualified':bool(len(val)>0 and len(inner)>=2 and len(selection)>0)})
  for tag,z in [('TRAIN',train),('VALIDATION',val),('INNER_FIT',inner),('INNER_SELECTION',selection),('INNER_PURGED',purged)]:cs(f'data_contracts/H2_{fold}_{tag}.csv',z)
  used.extend([train,val])
 cs('outputs/H2_ALL_CANDIDATES.csv',frame);cs('outputs/H2_USED_SAMPLES.csv',pd.concat(used));cs('outputs/H2_EXCLUSIONS.csv',pd.DataFrame(ex));cs('outputs/H2_ELIGIBILITY.csv',pd.DataFrame(audit));js('data_contracts/H2_HISTORY_IDENTITIES.json',ev)
 return audit
def fit(train,alpha):
 s=StandardScaler().fit(train[FEATURES].to_numpy());f=Ridge(alpha=alpha,solver='cholesky').fit(s.transform(train[FEATURES].to_numpy()),(train.target_R-train.origin_R).to_numpy())
 return {'alpha':alpha,'scaler_mean':s.mean_.tolist(),'scaler_scale':s.scale_.tolist(),'coef':f.coef_.tolist(),'intercept':float(f.intercept_),'sample_ids':train.sample_id.tolist()}
def predict(d,p):return d.origin_R.to_numpy()+((d[FEATURES].to_numpy()-p['scaler_mean'])/p['scaler_scale'])@p['coef']+p['intercept']
def run():
 cfg=config();assert not (P/'outputs/H2_COMPLETED.json').exists(),'Do not repeat completed H2 fits'
 audits=cohort(cfg)
 if not all(a['qualified'] for a in audits):
  js('outputs/H2_COMPLETED.json',{'status':'THREE_FOLD_NOT_EVALUABLE','new_fits':0});return
 # Eligibility decision recorded before any H2 fitting/score is computed.
 js('configs/H2_PREFIT_AUTHORIZATION.json',{'eligibility_all3':True,'reason':'Complete actual1-3cycle task presentation; all3folds andinnerchronology qualified','protocol_sha256':sha(P/'configs/FROZEN_PROTOCOL.json'),'audits':audits})
 predictions=[];models=[]
 for a in audits:
  fold=a['fold'];token=f'models/H2_{fold}.json';done=P/token
  if done.exists():p=jl(token)
  else:
   train=pd.read_csv(P/f'data_contracts/H2_{fold}_TRAIN.csv',float_precision='round_trip');inner=pd.read_csv(P/f'data_contracts/H2_{fold}_INNER_FIT.csv',float_precision='round_trip');selection=pd.read_csv(P/f'data_contracts/H2_{fold}_INNER_SELECTION.csv',float_precision='round_trip');grid=[]
   for alpha in cfg['alpha_grid']:
    cp=f'models/checkpoints/H2_{fold}_alpha{alpha}.json'
    if (P/cp).exists():r=jl(cp)
    else:
     r=fit(inner,alpha);r['selection_MAE_R']=float(abs(predict(selection,r)-selection.target_R).mean());js(cp,r)
    grid.append(r)
   best=min(grid,key=lambda r:(r['selection_MAE_R'],r['alpha']));p=fit(train,best['alpha']);p.update(fold=fold,H=2,registration='NEW_PREREGISTERED_H2',inner_grid=grid,inner_selection_ids=selection.sample_id.tolist(),audit=a);js(token,p)
  val=pd.read_csv(P/f'data_contracts/H2_{fold}_VALIDATION.csv',float_precision='round_trip');models.append(p)
  for method,pr in [('PERSISTENCE',val.origin_R.to_numpy()),('RIDGE96',predict(val,p))]:
   z=val.copy();z['method']=method;z['predicted_R']=pr;z['error_mVeq']=(pr-z.target_R)*VREF*1000;predictions.append(z)
  cs(f'outputs/H2_{fold}_PREDICTIONS.csv',pd.concat(predictions[-2:]));print('H2',fold,'n',len(val),'alpha',p['alpha'],flush=True)
 pred=pd.concat(predictions);cs('outputs/H2_PREDICTIONS.csv',pred);ms=[]
 for (fold,m),g in pred.groupby(['fold','method']):ms.append({'fold':fold,'method':m,**metric(g.target_R,g.predicted_R)})
 metrics=pd.DataFrame(ms);cs('outputs/H2_FOLD_METRICS.csv',metrics);eq=metrics.groupby('method')[['MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']].mean().reset_index();cs('outputs/H2_EQUAL_FOLD_MEANS.csv',eq)
 leads=pred[pred.method.eq('PERSISTENCE')].groupby('fold').agg(n=('target_R','size'),actual_hours_min=('actual_lead_h_AUDIT_ONLY','min'),actual_hours_max=('actual_lead_h_AUDIT_ONLY','max'),calendar_hours_min=('calendar_lead_h_AUDIT_ONLY','min'),calendar_hours_max=('calendar_lead_h_AUDIT_ONLY','max')).reset_index();cs('outputs/H2_ACTUAL_LEADS.csv',leads)
 js('outputs/H2_COMPLETED.json',{'status':'COMPLETED','new_inner_fits':12,'new_final_fits':3,'H1_H3_refits':0,'CPU':True,'prediction_rows':len(pred)})
 js('outputs/READ_LEDGER_H2.json',sorted(set(ACCESS)));config();print(eq.to_string(index=False))
if __name__=='__main__':firewall();run()

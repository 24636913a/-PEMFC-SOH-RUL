"""Independent identity/feature/normal-equation/prediction/metric checks. No sklearn fits."""
import json,hashlib,math
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent
FEATURES=['origin_R','mean_R','std_R','slope_R_per_h','origin_minus_median_R'];V=.71383023635
def read(name):return pd.read_csv(P/name,float_precision='round_trip')
def load(name):return json.loads((P/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def assertclose(a,b,tol=1e-10):assert np.max(abs(np.asarray(a)-np.asarray(b)))<tol,(a,b)
def metrics(y,p):
 e=(np.asarray(p)-np.asarray(y))*V*1000
 if not len(e):return {'n':0}
 return dict(n=len(e),MAE_mVeq=float(abs(e).mean()),RMSE_mVeq=float(np.sqrt(np.mean(e**2))),P95_mVeq=float(np.percentile(abs(e),95)),max_mVeq=float(abs(e).max()),bias_mVeq=float(e.mean()))
def run():
 cfg=load('configs/FROZEN_PROTOCOL.json');assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 for name,h in cfg['source_sha256'].items():assert sha(ROOT/name)==h,name
 for name,h in cfg['calculation_code_sha256'].items():assert sha(P/name)==h,name
 assert sha(P/'scripts/resume_and_finalize.py')==load('configs/RESUME_CODE_IDENTITY.json')['source_code_sha256']
 for row in read('inputs/SOURCE_SHA256.csv').itertuples():assert sha(P/f'inputs/{row.key}{Path(row.path).suffix}')==row.sha256
 ident=read('inputs/identity.csv').sort_values('measurement_end_h');cat=read('inputs/catalog.csv').set_index('raw_cycle_id');meas=read('inputs/measurement.csv');meas=meas[meas.stable_window.eq('middle_50')];m=meas.set_index('cycle_interval_id')
 counts=load('outputs/TRAJECTORY_COUNTS.json');assert len(ident)==counts['causal_forecast_observations']==131;assert len(meas)==141;assert len(read('inputs/partial.csv'))==6
 traj=read('outputs/REAL_P1S_OBSERVATIONS.csv');assert len(traj)==147 and traj.cycle_interval_id.is_unique
 assert not traj.exact_stable_end_calendar.notna().any();assert traj.Development.all()
 features={};cadence={};availability={};rowids={};feature_diffs=[]
 hist=load('outputs/HISTORY_FEATURE_IDENTITIES.json')
 for row in ident.itertuples(index=False):
  t=row.measurement_end_h;g=ident[(ident.measurement_end_h>=t-96)&(ident.measurement_end_h<=t)&(ident.conservative_available_end_h<=row.conservative_available_end_h)]
  x=g.measurement_end_h.to_numpy();y=g.R.to_numpy();slope=0. if len(g)<2 else float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2))
  f=np.array([row.R,y.mean(),y.std(ddof=0),slope,row.R-np.median(y)]);features[int(row.raw_cycle_id)]=f
  gaps=[x[i]-x[i-1] for i in range(1,len(g)) if g.raw_cycle_id.iloc[i]-g.raw_cycle_id.iloc[i-1]==1 and x[i]>x[i-1]]
  cadence[int(row.raw_cycle_id)]=float(np.median(gaps)) if gaps else None;availability[int(row.raw_cycle_id)]=float(row.conservative_available_end_h);rowids[int(row.raw_cycle_id)]=row
  saved=next(r for r in hist if r['origin_raw_cycle_id']==row.raw_cycle_id);assert saved['history_ids']==g.cycle_interval_id.tolist();assert saved['eligible']==bool(len(g)>=5 and np.ptp(x)>=67.2 and gaps)
  diff=float(np.max(abs(f-np.array([saved[k] for k in FEATURES]))));assert diff<1e-12;feature_diffs.append(diff)
  assertclose(row.R,m.loc[row.cycle_interval_id,'P1S_voltage_V']/V,1e-12)
 used=read('outputs/USED_SAMPLES.csv');u=used.set_index('sample_id');newmodels=load('models/NEW_H6_H12_RIDGE_PARAMETERS.json');native=load('inputs/native_parameters.json');records=[]
 # Validate every saved new coefficient vector against its actual normal equations and every alpha choice.
 for par in newmodels:
  for rec,kind in [(par,'final')]+[(r,'inner') for r in par['inner_records']]:
   train=u.loc[rec['sample_ids']];x=train[FEATURES].to_numpy();mean=x.mean(0);sd=x.std(0);sd=np.where(sd==0,1,sd);assertclose(mean,rec['scaler_mean'],1e-12);assertclose(sd,rec['scaler_scale'],1e-12)
   z=(x-mean)/sd;y=(train.target_R-train.origin_R).to_numpy();xc=z-z.mean(0);yc=y-y.mean();coef=np.array(rec['coef']);res=(xc.T@xc+rec['alpha']*np.eye(5))@coef-xc.T@yc
   assert max(abs(res))<1e-9;assertclose(rec['intercept'],y.mean()-z.mean(0)@coef,1e-12)
   if kind=='inner':assert train.target_available_h.max()<=par['inner_cutoff_h']
   records.append({'H':par['horizon'],'fold':par['fold'],'role':kind,'alpha':rec['alpha'],'n':len(train),'normal_equation_max_residual':float(max(abs(res)))})
  select=u.loc[par['inner_selection_ids']];scores=[]
  for rec in par['inner_records']:
   rp=select.origin_R.to_numpy()+((select[FEATURES].to_numpy()-rec['scaler_mean'])/rec['scaler_scale'])@rec['coef']+rec['intercept'];score=float(abs(rp-select.target_R.to_numpy()).mean());assertclose(score,rec['inner_MAE_R'],1e-12);scores.append((score,rec['alpha']))
  assert min(scores)[1]==par['alpha']
 pred=read('outputs/PREDICTIONS.csv');assert not pred.predicted_R.isna().any();assert not pred.duplicated(['sample_id','method']).any();maxdiff=0.
 for (H,fold,method),g in pred.groupby(['H','fold','method']):
  if method=='RIDGE96':
   p=next(r for r in native if r['horizon']==f'H{H}' and r['fold']==fold) if H in [1,3] else next(r for r in newmodels if r['horizon']==H and r['fold']==fold)
   mean=p['train_scaler_mean'] if H in [1,3] else p['scaler_mean'];scale=p['train_scaler_scale'] if H in [1,3] else p['scaler_scale'];coef=p['coef_standardized_X'] if H in [1,3] else p['coef'];b=p['intercept_deltaR'] if H in [1,3] else p['intercept']
   expected=g.origin_R.to_numpy()+((g[FEATURES].to_numpy()-mean)/scale)@coef+b
  elif method=='PERSISTENCE':expected=g.origin_R.to_numpy()
  else:expected=np.array([rowids[int(r.origin_raw_cycle_id)].R+features[int(r.origin_raw_cycle_id)][3]*H*cadence[int(r.origin_raw_cycle_id)] for r in g.itertuples()])
  diff=float(np.max(abs(expected-g.predicted_R.to_numpy())));maxdiff=max(maxdiff,diff);assert diff<1e-12
  for row in g.itertuples():
   assert row.target_raw_cycle_id-row.origin_raw_cycle_id==H
   assert all(bool(cat.loc[r,'complete_cycle']) for r in range(row.origin_raw_cycle_id,row.target_raw_cycle_id+1))
   target=rowids[int(row.target_raw_cycle_id)];origin=rowids[int(row.origin_raw_cycle_id)];lo,mid,hi=cfg['folds'][fold]
   assert mid<=target.measurement_start_h and target.conservative_available_end_h<hi
   assertclose(row.target_R,target.R,1e-12);assertclose(row.actual_lead_h_AUDIT_ONLY,target.measurement_end_h-origin.measurement_end_h,1e-9)
   assertclose(row.estimated_lead_h,H*cadence[int(row.origin_raw_cycle_id)],1e-9)
   assertclose(row.error_mVeq,(row.predicted_R-row.target_R)*V*1000,1e-10)
  tr=read(f'inputs/H{H}_{fold}_USED_TRAIN.csv');assert tr.target_available_h.max()<=g.origin_available_h.min()
  if H in [6,12] and method=='RIDGE96':assert par is not None
 # Independently recompute old188 pointwise predictions and frozen per-fold metrics.
 old=read('inputs/native_predictions.csv');oldmetrics=read('inputs/native_metrics.csv')
 for (h,f,model),g in old.groupby(['horizon','fold','model']):
  p=next(v for v in native if v['horizon']==h and v['fold']==f);x=np.array([features[int(raw)] for raw in g.origin_raw_cycle_id]);rp=x[:,0] if model=='PERSISTENCE' else x[:,0]+((x-p['train_scaler_mean'])/p['train_scaler_scale'])@p['coef_standardized_X']+p['intercept_deltaR']
  assertclose(rp,g.predicted_R_P1S_causal.to_numpy(),1e-12);metric=metrics(g.true_R_P1S_causal,rp);saved=oldmetrics[oldmetrics.horizon.eq(h)&oldmetrics.fold.eq(f)&oldmetrics.model.eq(model)].iloc[0]
  assertclose(metric['MAE_mVeq'],saved.MAE_mV_equivalent,1e-9);assertclose(metric['RMSE_mVeq'],saved.RMSE_mV_equivalent,1e-9)
 fm=read('outputs/FOLD_METRICS.csv');common=read('outputs/COMMON_ORIGIN_PREDICTIONS.csv')
 for r in fm.itertuples():
  source=pred if r.scope=='own_horizon_cohort' else common;g=source[source.H.eq(r.H)&source.fold.eq(r.fold)&source.method.eq(r.method)];mtr=metrics(g.target_R,g.predicted_R);assert mtr['n']==r.n
  if len(g):
   for key in ['MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']:assertclose(mtr[key],getattr(r,key),1e-10)
  else:assert all(pd.isna(getattr(r,key)) for key in ['MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq'])
 for row in read('outputs/THREE_FOLD_EQUAL_MEANS.csv').itertuples():
  g=fm[fm.scope.eq(row.scope)&fm.H.eq(row.H)&fm.method.eq(row.method)];full=len(g)==3 and (g.n>0).all();assert row.threefold_evaluable==full
  for k in ['MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']:
   if full:assertclose(getattr(row,k),g[k].mean(),1e-10)
   else:assert pd.isna(getattr(row,k))
 for row in read('outputs/POOLED_STRATUM_DIAGNOSTICS.csv').itertuples():
  g=pred[pred.H.eq(row.H)&pred.method.eq(row.method)];g=g[g[row.stratum].eq(row.state)];computed=metrics(g.target_R,g.predicted_R);assert len(g)==row.n
  if len(g):assertclose(computed['MAE_mVeq'],row.MAE_mVeq,1e-10)
  else:assert pd.isna(row.MAE_mVeq)
 assert not len(pred[pred.H.eq(12)&pred.fold.isin(['fold_1','fold_2'])&pred.method.eq('RIDGE96')]);assert not len(pred[pred.H.eq(12)&pred.fold.eq('fold_2')])
 tft=read('outputs/TFT_PARAMETER_TIME_AND_HEALTH_SUPPORT.csv');assert tft[tft.role.eq('train')].parameters_available_by_origin.sum()==0
 phys=read('outputs/REAL_TRAIN_PHYSICS_IDENTIFIABILITY.csv');assert phys.ohmic_design_rank.eq(1).all();assert phys.real_certified_physics_loss.isna().all();assert phys.real_certified_physics_model_parameter_gradient.isna().all()
 pd.DataFrame(records).to_csv(P/'outputs/INDEPENDENT_NORMAL_EQUATION_AUDIT.csv',index=False,lineterminator='\n')
 result={'PASS':True,'implementation':'Independent feature construction, NumPy normal-equation KKT, explicit point formulas and quantiles; no sklearn fit/import of production calculations',
  'prediction_rows_verified':len(pred),'native_predictions_verified':len(old),'new_parameter_vectors_verified':len(records),'feature_max_difference':max(feature_diffs),'prediction_max_difference_R':maxdiff,
  'missing_fold_means_are_NA':True,'source_SHA_unchanged':True,'no_Test':True,'no_refitting':True}
 (P/'outputs/INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result)
if __name__=='__main__':run()

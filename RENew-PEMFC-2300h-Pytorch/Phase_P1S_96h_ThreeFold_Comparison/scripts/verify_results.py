"""Independent NumPy algebra and direct-key checks; no sklearn fitting or tuning."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]
def read(rel):return pd.read_csv(P/rel,float_precision='round_trip')
def check():
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
 for name,rel in cfg['sources'].items():assert hashlib.sha256((P.parent/rel).read_bytes()).hexdigest()==cfg['source_sha256'][name]
 for rel,sha in cfg['code_sha256'].items():assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==sha
 f=read('inputs/L96_REUSED_FEATURES.csv');legal=read('outputs/LEGAL_SAMPLES.csv');splits=read('outputs/INNER_SPLITS.csv');pred=read('outputs/PREDICTIONS.csv');metrics=read('outputs/FOLD_METRICS.csv')
 identity=pd.read_csv(P.parent/cfg['sources']['identity'],float_precision='round_trip').set_index('cycle_interval_id')
 ids=json.loads((P/'inputs/L96_TIME_IDENTITIES.json').read_text(encoding='utf-8'));par=json.loads((P/'outputs/TRAINED_PARAMETERS.json').read_text(encoding='utf-8'))
 for a in ids:
  row=f[(f.horizon==a['horizon'])&(f.fold==a['fold'])&(f.role==a['role'])&(f.sample_index==a['sample_index'])].iloc[0]
  hist=identity.loc[a['cycle_ids']].sort_values('measurement_end_h');x=hist.measurement_end_h.to_numpy();y=hist.R.to_numpy();xc=x-x.mean()
  slope=float(xc@(y-y.mean())/(xc@xc)) if len(x)>1 else 0.
  independently=np.array([y[-1],np.mean(y),np.std(y),slope,y[-1]-np.median(y)])
  assert np.allclose(independently,row[cfg['feature_names']].to_numpy(float),atol=2e-14,rtol=0)
  expected=identity[(identity.measurement_end_h<=row.origin_p1s_measurement_end_ophrs2_h+1e-9)&(identity.measurement_end_h>=row.origin_p1s_measurement_end_ophrs2_h-96-1e-9)].sort_values('measurement_end_h').index.tolist()
  assert expected==a['cycle_ids']
  assert bool(row.L96_eligible)==(len(x)>=5 and np.ptp(x)>=67.2-1e-9)
 max_difference=0.;grid_checks=0
 for a in par:
  t=legal[(legal.horizon==a['horizon'])&(legal.fold==a['fold'])&(legal.role=='train')].sort_values('target_p1s_measurement_end_ophrs2_h')
  v=legal[(legal.horizon==a['horizon'])&(legal.fold==a['fold'])&(legal.role=='validation')].sort_values('target_p1s_measurement_end_ophrs2_h')
  fi=splits[(splits.horizon==a['horizon'])&(splits.fold==a['fold'])&(splits.inner_role=='inner_fit')].sort_values('target_p1s_measurement_end_ophrs2_h')
  se=splits[(splits.horizon==a['horizon'])&(splits.fold==a['fold'])&(splits.inner_role=='inner_selection')].sort_values('target_p1s_measurement_end_ophrs2_h')
  assert t.target_p1s_measurement_end_ophrs2_h.max()<=v.origin_p1s_measurement_end_ophrs2_h.min()+1e-9
  assert fi.target_p1s_measurement_end_ophrs2_h.max()<=se.origin_p1s_measurement_end_ophrs2_h.min()+1e-9
  assert t.sample_index.tolist()==a['train_samples'] and v.sample_index.tolist()==a['validation_samples']
  assert len(se)==int(np.ceil(.15*len(t)))
  for scope,data in [('inner',fi),('train',t)]:
   X=data[cfg['feature_names']].to_numpy();scale=np.std(X,axis=0);scale[scale==0]=1.
   assert np.allclose(X.mean(axis=0),a[scope+'_scaler_mean'],rtol=0,atol=1e-14)
   assert np.allclose(scale,a[scope+'_scaler_scale'],rtol=0,atol=1e-14)
  X=fi[cfg['feature_names']].to_numpy();mu=X.mean(axis=0);sd=X.std(axis=0);sd[sd==0]=1.;X=(X-mu)/sd;Xc=X-X.mean(axis=0)
  y=(fi.target_R_P1S_causal-fi.origin_R_P1S_causal).to_numpy();yc=y-y.mean();score=[]
  for g in a['alpha_grid_scores']:
   coef=np.linalg.solve(Xc.T@Xc+g['alpha']*np.eye(5),Xc.T@yc)
   result=((se[cfg['feature_names']].to_numpy()-mu)/sd-X.mean(axis=0))@coef+y.mean()
   mae=float(np.abs(result-(se.target_R_P1S_causal-se.origin_R_P1S_causal)).mean());assert abs(mae-g['inner_MAE_R'])<1e-12;score.append((mae,g['alpha']));grid_checks+=1
  assert min(score)[1]==a['alpha']
  # Reconstruct predictions from serialized parameters, without fitting again.
  result=v.origin_R_P1S_causal.to_numpy()+((v[cfg['feature_names']].to_numpy()-np.array(a['train_scaler_mean']))/np.array(a['train_scaler_scale']))@np.array(a['coef_standardized_X'])+a['intercept_deltaR']
  saved=pred[(pred.horizon==a['horizon'])&(pred.fold==a['fold'])&(pred.model=='RIDGE_96H')].sort_values('target_p1s_measurement_end_ophrs2_h')
  diff=float(np.max(np.abs(result-saved.predicted_R_P1S_causal)));assert diff<1e-12;max_difference=max(max_difference,diff)
 for row in metrics.itertuples():
  g=pred[(pred.horizon==row.horizon)&(pred.fold==row.fold)&(pred.model==row.model)];e=(g.predicted_R_P1S_causal-g.true_R_P1S_causal).to_numpy();ae=np.abs(e);assert len(e)==row.n
  vals={'MAE_R':ae.mean(),'RMSE_R':np.sqrt(np.mean(e**2)),'P95_abs_error_R':np.quantile(ae,.95),'max_abs_error_R':ae.max()}
  for col,val in vals.items():assert abs(getattr(row,col)-val)<1e-14
  for col in vals:
   mvcol=col.replace('_R','_mV_equivalent');assert abs(getattr(row,mvcol)-vals[col]*cfg['Vref_V']*1000)<1e-12
  if row.model=='PERSISTENCE':assert np.array_equal(g.predicted_R_P1S_causal,g.origin_R_P1S_causal)
 matches=read('outputs/HISTORICAL_MATCHED_PREDICTIONS.csv');oldmetrics=read('outputs/HISTORICAL_INTERSECTION_METRICS.csv')
 for (h,m,seed),g in matches.groupby(['horizon','historical_model','historical_seed']):
  names=['H1_baseline','H1_timesnet'] if h=='H1' else ['H3_history'];src=pd.concat([pd.read_csv(P.parent/cfg['sources'][n],float_precision='round_trip') for n in names]);src['seed_label']=src.seed.fillna(-1).astype(int)
  src=src[(src.model==m)&(src.seed_label==seed)]
  key=['fold','origin_raw_cycle_id','target_raw_cycle_id'];current=pred[(pred.horizon==h)&(pred.model=='RIDGE_96H')]
  independently=current.merge(src,on=key,suffixes=('_new','_hist'),validate='one_to_one');assert len(independently)==len(g)
  assert set(map(tuple,g[key].to_numpy()))==set(map(tuple,independently[key].to_numpy()))
  assert np.max(np.abs(g.true_R_P1S_causal_current-g.true_R_P1S_causal_old))<1e-12
  for row in oldmetrics[(oldmetrics.horizon==h)&(oldmetrics.historical_model==m)&(oldmetrics.seed==seed)].itertuples():
   z=g[g.fold==row.fold];col='predicted_R_P1S_causal_current' if row.method=='CURRENT_RIDGE_96H' else 'predicted_R_P1S_causal_old'
   ae=np.abs(z[col]-z.true_R_P1S_causal_current);assert row.n==len(z);assert abs(row.MAE_mV_equivalent-ae.mean()*cfg['Vref_V']*1000)<1e-12
 report={'status':'PASS','feature_vectors_verified':len(ids),'alpha_scores_algebraically_verified':grid_checks,'prediction_rows_checked':len(pred),'max_serialized_prediction_difference_R':max_difference,'historical_matched_rows':len(matches),'source_and_code_hashes_unchanged':True,'verification_uses_no_sklearn_fit':True}
 (P/'outputs/INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':check()

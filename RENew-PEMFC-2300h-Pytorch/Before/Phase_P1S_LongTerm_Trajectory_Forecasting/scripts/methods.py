"""Fixed96h input, causal cadence trend, inherited Ridge algorithm for NEW H6/H12 only."""
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from common import FEATURES
def features(identity,origin):
 t=float(origin.measurement_end_h)
 h=identity[(identity.measurement_end_h>=t-96)&(identity.measurement_end_h<=t)&(identity.conservative_available_end_h<=origin.conservative_available_end_h)].sort_values('measurement_end_h')
 x=h.measurement_end_h.to_numpy(float);y=h.R.to_numpy(float);span=float(np.ptp(x)) if len(h) else 0.
 slope=float((x-x.mean())@(y-y.mean())/np.sum((x-x.mean())**2)) if len(h)>1 else 0.
 gaps=h.measurement_end_h.diff().to_numpy();adj=h.raw_cycle_id.diff().to_numpy()==1;gap=gaps[adj & (gaps>0)]
 cadence=float(np.median(gap)) if len(gap) else None
 return dict(zip(FEATURES,[float(origin.R),float(y.mean()),float(y.std(ddof=0)),slope,float(origin.R-np.median(y))])),{
  'history_ids':h.cycle_interval_id.tolist(),'observations':len(h),'span_h':span,'eligible':bool(len(h)>=5 and span>=67.2 and cadence is not None),
  'cadence_h_per_cycle':cadence,'cadence_adjacent_pairs':int(len(gap)),'actual_history_end_max_h':float(x.max()),
  'available_history_end_max_h':float(h.conservative_available_end_h.max())}
def trend(origin_R,slope_R_per_h,known_horizon_cycles,past_cadence_h):
 return origin_R+slope_R_per_h*known_horizon_cycles*past_cadence_h
def replay(frame,par):
 if 'features' in par and par['features'][0].startswith('L96_'):
  mean=par['train_scaler_mean'];scale=par['train_scaler_scale'];coef=par['coef_standardized_X'];intercept=par['intercept_deltaR']
 else:mean=par['scaler_mean'];scale=par['scaler_scale'];coef=par['coef'];intercept=par['intercept']
 return frame.origin_R.to_numpy()+((frame[FEATURES].to_numpy()-np.array(mean))/np.array(scale))@np.array(coef)+intercept
def fit_record(train,alpha):
 scale=StandardScaler().fit(train[FEATURES].to_numpy());x=scale.transform(train[FEATURES].to_numpy());y=(train.target_R-train.origin_R).to_numpy()
 fit=Ridge(alpha=alpha,solver='cholesky',fit_intercept=True).fit(x,y)
 return dict(alpha=alpha,scaler_mean=scale.mean_.tolist(),scaler_scale=scale.scale_.tolist(),coef=fit.coef_.tolist(),intercept=float(fit.intercept_),
  sample_ids=train.sample_id.tolist(),max_label_available_h=float(train.target_available_h.max()),target_standardized=False)
def train_new(train,horizon,fold,grid):
 assert horizon in [6,12],'H1/H3 must never be fitted again'
 train=train.sort_values('target_actual_end_h').copy();n=math.ceil(.15*len(train));selection=train.iloc[-n:];initial=train.iloc[:-n]
 cutoff=float(selection.origin_available_h.min());purged=initial[initial.target_available_h>cutoff];inner=initial[initial.target_available_h<=cutoff]
 assert len(inner)>=2 and len(selection)>=1,'INSUFFICIENT_INNER_LABEL_ISOLATION'
 records=[]
 for alpha in grid:
  r=fit_record(inner,alpha);pred=replay(selection,r);r['inner_MAE_R']=float(np.mean(abs(pred-selection.target_R.to_numpy())));records.append(r)
 chosen=min(records,key=lambda a:(a['inner_MAE_R'],a['alpha']));final=fit_record(train,chosen['alpha'])
 final.update(horizon=horizon,fold=fold,registration='NEW_EXPERIMENT_NOT_FROZEN_NATIVE_H1_H3',
  inner_records=records,inner_selection_ids=selection.sample_id.tolist(),inner_purged_ids=purged.sample_id.tolist(),inner_cutoff_h=cutoff,
  final_fits=1,inner_fits=len(grid))
 return final

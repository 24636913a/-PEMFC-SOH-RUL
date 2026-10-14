"""Independent event selection and least-squares reconstruction; no neural fits."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]
def main():
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
 source=P.parent/'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['source_sha256'][str(source.relative_to(P.parent)).replace('\\','/')]
 s=pd.read_csv(source,float_precision='round_trip').sort_values('measurement_end_h')
 data=dict(np.load(P/'inputs/MATCHED_INPUTS.npz',allow_pickle=False));details=json.loads((P/'inputs/P1S_HISTORY_TIME_IDENTITIES.json').read_text(encoding='utf-8'))
 scaler=json.loads((P/'inputs/HEALTH_TRAIN_SCALER.json').read_text());alignment=pd.read_csv(P/'inputs/P1S_TFT_ALIGNMENT.csv',float_precision='round_trip')
 checked=0;difference=0.
 for a in details:
  t=a['origin_time_h'];q=s[(s.measurement_end_h>=t-96-1e-9)&(s.measurement_end_h<=t+1e-9)&(s.conservative_available_end_h<=t+1e-9)]
  assert a['raw_cycle_ids']==q.raw_cycle_id.astype(int).tolist()
  if not a['eligible']:continue
  assert t>=171.82091 and len(q)>=5 and np.ptp(q.measurement_end_h)>=67.2-1e-9
  x=q.measurement_end_h.to_numpy();y=q.R.to_numpy();design=np.column_stack([np.ones(len(x)),x-x.mean()])
  coefficient=np.linalg.lstsq(design,y-y.mean(),rcond=None)[0][1]
  vector=np.array([y[-1],np.average(y),np.std(y,ddof=0),coefficient,y[-1]-np.quantile(y,.5)])
  role=a['role'];idx=np.flatnonzero(data[role+'_origins']==a['origin_row']);assert len(idx)==1
  saved=data[role+'_health_raw'][idx[0]];diff=float(np.max(np.abs(vector-saved)));assert diff<2e-14;difference=max(difference,diff);checked+=1
 mu=data['train_health_raw'].mean(axis=0);sd=data['train_health_raw'].std(axis=0);sd=np.where(sd>0,sd,1.)
 assert np.array_equal(mu,scaler['mean']) and np.array_equal(sd,scaler['scale'])
 for role in ['train','validation']:
  assert np.array_equal(((data[role+'_health_raw']-mu)/sd).astype(np.float32),data[role+'_health'])
 assert checked==1048
 tr=alignment[alignment.role=='train'];val=alignment[(alignment.role=='validation')&alignment.eligible]
 assert tr.last_target_time_h.max()<val.origin_time_h.min()
 report=dict(status='PASS',all_origin_event_selections_checked=len(details),qualified_health_vectors_checked=checked,
  max_feature_difference=difference,independent_slope_uses_lstsq=True,health_scaler_matches_only_train=True,
  full_original_Train_labels_and_scaler_history_before_first_matched_validation_origin=True,
  Assessment_Test_or_raw_rebuild=False,neural_optimizer_steps=0)
 (P/'outputs/HEALTH_INPUT_INDEPENDENT_VERIFICATION.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report,indent=2))
if __name__=='__main__':main()

"""Exact named, Development-only historical caches. No raw, Test or model training."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
OLD='Phase_Integrated_SOH_Physics_RUL_Research/'
SOURCES={
 'cache':'Phase_P1S_96h_ThreeFold_Comparison/outputs/PREDICTIONS.csv',
 'ridge_parameters':'Phase_P1S_96h_ThreeFold_Comparison/outputs/TRAINED_PARAMETERS.json',
 'ridge_time':'Phase_P1S_96h_ThreeFold_Comparison/outputs/PARAMETER_TIME_AUDIT.csv',
 'ridge_fold_metrics':'Phase_P1S_96h_ThreeFold_Comparison/outputs/FOLD_METRICS.csv',
 'links':OLD+'outputs/P1S_TO_SOH_EXACT_FORECAST_LINKS.csv',
 'calibration':OLD+'models/PAST_ONLY_AFFINE_PARAMETERS.json',
 'old_predictions':OLD+'outputs/SOH_EVENT_PREDICTIONS.csv',
 'old_metrics':OLD+'outputs/SOH_EVENT_METRICS.csv',
 'refs':'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv',
 'events':OLD+'outputs/SOH_EVENT_LEVEL_ELIGIBILITY.csv',
 'timing':'Phase_PAV_PAA_Anchor_Validity_Qualification/outputs/POLARIZATION_TIME_PROVENANCE.csv',
 'identity':'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',
 'measurement':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv'}
METHODS=['M0_REFERENCE_PERSISTENCE','M_ORIGIN_P1S_CALIBRATED','M_RIDGE_P1S_CALIBRATED']
ACCESS=[]
def sha(q):return hashlib.sha256(Path(q).read_bytes()).hexdigest()
def firewall():
 allowed={(ROOT/s).resolve() for s in SOURCES.values()}
 def audit(event,args):
  if event!='open' or isinstance(args[0],int):return
  raw,mode,flags=args;q=Path(os.fsdecode(raw)).resolve()
  write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if q.is_relative_to(REPO) and not q.is_relative_to(P):
   if write or q not in allowed:raise PermissionError('NON_ALLOWLISTED_REPOSITORY_ACCESS: '+str(q))
   ACCESS.append(q.relative_to(ROOT).as_posix())
 sys.addaudithook(audit)
def csv_read(q):return pd.read_csv(q,float_precision='round_trip')
def source(k):return csv_read(ROOT/SOURCES[k])
def load(k):return json.loads((ROOT/SOURCES[k]).read_text(encoding='utf-8'))
def output(name):
 q=(P/name).resolve();assert q.is_relative_to(P);q.parent.mkdir(parents=True,exist_ok=True);return q
def js(name,obj):output(name).write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def cs(name,d):d.to_csv(output(name),index=False,float_format='%.17g',lineterminator='\n')
def md(name,s):output(name).write_bytes((s.strip()+'\n').encode('utf-8'))
def config():
 c=load_local('configs/FROZEN_PROTOCOL.json');assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 for s,h in c['source_sha256'].items():assert sha(ROOT/s)==h,s
 for s,h in c['audit_code_sha256'].items():assert sha(P/s)==h,s
 return c
def load_local(name):return json.loads((P/name).read_text(encoding='utf-8'))
def methods(prev_ref,origin_R,predicted_R,a,b):
 # No target observation, future lead, target calendar or new fitting is accepted.
 return dict(zip(METHODS,[float(prev_ref),float(a*origin_R+b),float(a*predicted_R+b)]))
def decomposition(y,origin_R,predicted_R,target_R,a,b):
 calibration=(a*target_R+b-y)*100
 proxy=a*(predicted_R-target_R)*100
 origin_proxy=a*(origin_R-target_R)*100
 return {'calibration_at_actual_target_SOH_pp':calibration,'ridge_proxy_error_translated_SOH_pp':proxy,
  'origin_proxy_error_translated_SOH_pp':origin_proxy,'ridge_final_error_SOH_pp':calibration+proxy,
  'origin_final_error_SOH_pp':calibration+origin_proxy,'ridge_step_delta_SOH_pp':a*(predicted_R-origin_R)*100,
  'ridge_error_cancels_calibration':bool(calibration*proxy<0),'origin_error_cancels_calibration':bool(calibration*origin_proxy<0),
  'ridge_cancellation_magnitude_SOH_pp':abs(calibration)+abs(proxy)-abs(calibration+proxy),
  'origin_cancellation_magnitude_SOH_pp':abs(calibration)+abs(origin_proxy)-abs(calibration+origin_proxy)}

"""New-stage output guard and read-only imports of the frozen Phase H code."""
from pathlib import Path
import os,sys,json,hashlib,time,subprocess,platform,traceback
import numpy as np
import pandas as pd
import torch
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
H=ROOT/'Phase_H_Determinism_MultiSeed_Stability'
sys.path.insert(0,str(H/'scripts'));import h_common as h
BASE='bca03706f06673e60b5ee5f2cfa82d33e17033db'
SEEDS=[101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197]
def sha(f):
 f=Path(f)
 if f.name=='causal_features_30min.csv':raise RuntimeError('FULL_N1_HASH_FIREWALL')
 return hashlib.sha256(f.read_bytes()).hexdigest()
def js(name,obj):
 f=(P/name).resolve()
 if not f.is_relative_to(P):raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.writing')
 with tmp.open('w',encoding='utf-8',newline='\n') as s:s.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');s.flush();os.fsync(s.fileno())
 os.replace(tmp,f)
def rd(f):return pd.read_csv(f,float_precision='round_trip')
def cs(name,d):
 f=P/name;f.parent.mkdir(parents=True,exist_ok=True);d.to_csv(f,index=False,float_format='%.17g',lineterminator='\n')
def cfg():
 f=P/'configs/frozen_seed_screening.json';assert sha(f)==f.with_suffix('.sha256').read_text().strip(),'FROZEN_CONFIG_IDENTITY_MISMATCH'
 c=json.loads(f.read_text());assert c['seeds']==SEEDS
 for path,digest in c['critical_code_sha256'].items():assert sha(P/path)==digest,('TRAINING_CODE_IDENTITY_MISMATCH',path)
 for path,digest in c['source_sha256'].items():assert sha(Path(path) if Path(path).is_absolute() else REPO/path)==digest,('HISTORICAL_SOURCE_IDENTITY_MISMATCH',path)
 for path,digest in c['data_sha256'].items():assert sha(P/path)==digest,('SNAPSHOT_IDENTITY_MISMATCH',path)
 return c
def data():return np.load(P/'outputs/audits/FROZEN_H_DATA.npz',allow_pickle=False)
def metric(d):
 e=d.y_pred_delta_V.to_numpy('float64')-d.y_true_delta_V.to_numpy('float64');y=d.y_true_delta_V.to_numpy('float64');ae=1000*abs(e)
 return dict(RMSE_mV=float(1000*np.sqrt(np.mean(e**2))),MAE_mV=float(ae.mean()),R2=float(1-np.dot(e,e)/np.dot(y-y.mean(),y-y.mean())),
  endpoint_RMSE_mV=float(1000*np.sqrt(np.mean(e[d.horizon_step.eq(24)]**2))),target_time_equal_RMSE_mV=float(np.sqrt(pd.DataFrame({'target':d.target_row,'se':(1000*e)**2}).groupby('target').se.mean().mean())),
  P95_mV=float(np.percentile(ae,95)),P99_mV=float(np.percentile(ae,99)),max_abs_error_mV=float(ae.max()),n=len(d),origins=int(d.origin_row.nunique()),distinct_target_times=int(d.target_row.nunique()))
def append_event(event):
 f=P/'outputs/execution_ledger/events.jsonl';f.parent.mkdir(parents=True,exist_ok=True)
 with f.open('a',encoding='utf-8',newline='\n') as s:s.write(json.dumps(dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event),ensure_ascii=False,allow_nan=False)+'\n');s.flush();os.fsync(s.fileno())
def ranking():
 rows=[]
 for seed in SEEDS:
  f=P/f'outputs/metrics/seed_{seed}.json'
  if f.exists():
   r=json.loads(f.read_text())
   if r['status']=='COMPLETED':rows.append(r)
 d=pd.DataFrame(rows)
 if len(d):
  d=d.sort_values(['RMSE_mV','seed'],kind='stable').reset_index(drop=True);d.insert(0,'rank',np.arange(1,len(d)+1));cs('outputs/F2_SEED_RANKING.csv',d);cs('outputs/F2_TOP5_SEEDS.csv',d.head(5))
 return d

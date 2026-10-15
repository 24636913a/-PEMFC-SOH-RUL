"""Read-only historical sources; all writes restricted to this phase."""
import hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]; ROOT=P.parent; REPO=ROOT.parent
N1=ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
BRIDGE=ROOT/'Phase_TFT_DynamicVoltage_to_ReferenceSOH'
H=ROOT/'Phase_H_Determinism_MultiSeed_Stability'
META=['bin_start_ophrs2_h','bin_end_ophrs2_h','first_datetime','last_datetime','split','bin_valid_for_windows']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return pd.read_csv(p,float_precision='round_trip')
def cs(name,d):
 q=P/name; q.parent.mkdir(parents=True,exist_ok=True); d.to_csv(q,index=False,float_format='%.17g',lineterminator='\n')
def js(name,d):
 q=P/name; q.parent.mkdir(parents=True,exist_ok=True); q.write_bytes((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def md(name,s):
 q=P/name; q.parent.mkdir(parents=True,exist_ok=True); q.write_bytes((s.strip()+'\n').encode())
def selected_extent(meta):
 if len(meta)!=3007: raise ValueError('Historical prefix must have 3007 identities')
 ids=np.flatnonzero(meta.bin_end_ophrs2_h.to_numpy()<1819.997738)
 if not np.array_equal(ids,np.arange(len(ids))) or len(ids)!=3005: raise PermissionError('Unexpected historical mask')
 if meta.iloc[ids]['split'].astype(str).str.contains('test',case=False).any(): raise PermissionError('Test values forbidden')
 return ids
def historical_development():
 # Authority is always first3007 metadata + cutoff, never a hardcoded3005 shortcut.
 meta=pd.read_csv(N1,nrows=3007,usecols=META)
 ids=selected_extent(meta)
 # Excluded rows include a Test identity: parse no numeric feature from them.
 b=pd.read_csv(N1,nrows=len(ids)); assert len(b)==len(ids)
 return meta,ids,b
def metrics(e):
 e=np.asarray(e,float)
 return dict(n=len(e),MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e*e))),P95=float(np.quantile(abs(e),.95)),MAX=float(abs(e).max()))

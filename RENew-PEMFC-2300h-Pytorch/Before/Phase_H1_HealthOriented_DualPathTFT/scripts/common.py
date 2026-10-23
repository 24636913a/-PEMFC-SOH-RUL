"""New event task only. Historical code/checkpoints/data are immutable."""
from pathlib import Path
import json,hashlib,os,subprocess,sys,random,time
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
AUDIT=ROOT/'Phase_HealthOriented_Voltage_Target_Audit';PAA=ROOT/'Phase_PAA_P1_StateToken_PreAnchor_Alignment';RCR=ROOT/'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit'
VREF=.71383023635;BOUNDARY=1819.997738
BASE='7f822d2f9f7e5df47d1fb6da6d846dfa52c5034a'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def rd(f):return pd.read_csv(f,float_precision='round_trip')
def js(name,obj):
 f=(P/name).resolve();assert f.is_relative_to(P);f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.writing')
 with tmp.open('w',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,f)
def cs(name,d):
 f=P/name;f.parent.mkdir(parents=True,exist_ok=True);d.to_csv(f,index=False,float_format='%.17g',lineterminator='\n')
def md(name,s):
 f=P/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((s.strip()+'\n').encode())
def digest_array(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
def set_seed(seed):
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def protected_pass():
 d=rd(P/'data_contracts/SOURCE_SHA256.csv');return all(sha(ROOT/r.path)==r.sha256 for r in d.itertuples())
def load_cfg():
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip();return cfg
def prep(x,record):
 y=x.copy()
 for k in range(1,48):y[:,k,0]=.3*x[:,k,0]+.7*y[:,k-1,0]
 y=((y-np.asarray(record['x_mean']))/np.asarray(record['x_scale'])).astype('float32')
 y[:,:,[i for i in range(14) if i not in record['top10']]]=0
 return y
def fit_preprocessing(x,y,feature_names):
 # Same Spearman-at-origin rule as prior stage, now one H1 target and Train-only.
 scores=[]
 for j in range(14):
  a=pd.Series(x[:,-1,j]);z=pd.Series(y)
  rho=float(a.rank().corr(z.rank())) if a.nunique()>1 and z.nunique()>1 else 0.
  scores.append(dict(index=j,feature=feature_names[j],rho=rho,score=abs(rho)))
 ranking=sorted(scores,key=lambda r:(-r['score'],r['index']));top=[r['index'] for r in ranking[:10]]
 if 0 not in top:top[-1]=0
 flat=x.astype('float64').reshape(-1,14);scale=flat.std(axis=0);scale=np.where(scale>1e-8,scale,1.)
 yscale=float(np.std(y));assert yscale>1e-8
 return dict(x_mean=flat.mean(axis=0).tolist(),x_scale=scale.tolist(),top10=sorted(top),spearman=ranking,
  y_mean_V=float(np.mean(y)),y_scale_V=yscale,fit_observations=len(x),fit_raw_history_sha256=digest_array(x),fit_y_sha256=digest_array(y))

import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
Q='Phase_Polarization_Physics_to_Health_Qualification';H='REAL_Paper_File/Integrated_Research_Route_Closure'
SOURCES={'dwells':f'{Q}/outputs/POLARIZATION_DWELLS.csv','metadata':f'{Q}/outputs/POLARIZATION_QUALIFICATION.csv','slopes':f'{Q}/outputs/NUMERICAL_SLOPES.csv','protocol':f'{Q}/configs/FROZEN_PROTOCOL.json','verification':f'{Q}/outputs/INDEPENDENT_VERIFICATION.json','measurement':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv','identity':'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv','h_contract':f'{H}/H123_FREEZE.json','h_preds':f'{H}/H123_PREDICTIONS.csv','h_sources':f'{H}/SOURCE_INDEX.csv','tft_arch':'Phase_F_TFT_DualPath_RegimeFusion/diagrams/F2_REGIME_FUSION_ARCHITECTURE.md'}
for n in [42,52,62]:SOURCES[f'F2_seed{n}']=f'Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed{n}_best.pt'
SOURCES['F2_original']='Phase_F_TFT_DualPath_RegimeFusion/outputs/checkpoints/F2_best_seed42.pt'
EVENTS=['BoL','300h','600h','850h','1150h','1500h','1750h'];FEATURES=['j_design_A_cm2','direction_ascending','context'];ACCESS=[]
def sha(q):
 h=hashlib.sha256()
 with Path(q).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def firewall():
 allow={(ROOT/v).resolve() for v in SOURCES.values()}
 def hook(event,args):
  if event!='open' or isinstance(args[0],int):return
  q=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:];write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if q.is_relative_to(REPO) and not q.is_relative_to(P):
   if write or q not in allow:raise PermissionError('UNLISTED_READ_OR_HISTORICAL_WRITE '+str(q))
   ACCESS.append(q.relative_to(ROOT).as_posix())
 sys.addaudithook(hook)
def out(s):
 q=(P/s).resolve();assert q.is_relative_to(P);q.parent.mkdir(parents=True,exist_ok=True);return q
def js(s,d):out(s).write_bytes((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def jl(s):return json.loads((P/s).read_text(encoding='utf-8'))
def rd(k):return pd.read_csv(ROOT/SOURCES[k],float_precision='round_trip')
def cs(s,d):d.to_csv(out(s),index=False,float_format='%.17g',lineterminator='\n')
def md(s,t):out(s).write_bytes((t.strip()+'\n').encode('utf-8'))
def config():
 c=jl('configs/FROZEN_PROTOCOL.json');assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 for k,v in c['source_sha256'].items():assert sha(ROOT/SOURCES[k])==v,k
 for k,v in c['code_sha256'].items():assert sha(P/k)==v,k
 return c
def metric(y,p):
 e=(np.asarray(p)-np.asarray(y))*1000;a=abs(e)
 return {'n':len(e),'MAE_mV':float(a.mean()),'RMSE_mV':float(np.sqrt((e*e).mean())),'P95_mV':float(np.quantile(a,.95)),'max_mV':float(a.max())} if len(e) else {'n':0,'MAE_mV':None,'RMSE_mV':None,'P95_mV':None,'max_mV':None}

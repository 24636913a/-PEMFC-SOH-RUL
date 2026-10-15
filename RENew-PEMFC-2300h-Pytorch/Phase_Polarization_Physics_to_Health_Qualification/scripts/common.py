"""Exact Development allowlist and read-only historical-source firewall."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]; ROOT=P.parent; REPO=ROOT.parent
BASE='数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据'
CHECKPOINTS=['BoL','300h','600h','850h','1150h','1500h','1750h']
SOURCES={f'polar_{k}':REPO/BASE/f'Polcurve_Nominal_RHFC_{k}.csv' for k in CHECKPOINTS}
SOURCES.update({k:ROOT/v for k,v in {
 'identity':'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',
 'measurement':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',
 'catalog':'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv',
 'native_predictions':'Phase_P1S_96h_ThreeFold_Comparison/outputs/PREDICTIONS.csv',
 'native_parameters':'Phase_P1S_96h_ThreeFold_Comparison/outputs/TRAINED_PARAMETERS.json',
 'native_protocol':'Phase_P1S_96h_ThreeFold_Comparison/configs/FROZEN_PROTOCOL.json',
 'native_legal':'Phase_P1S_96h_ThreeFold_Comparison/outputs/LEGAL_SAMPLES.csv',
 'soh':'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv',
 'calibration':'Phase_Integrated_SOH_Physics_RUL_Research/models/PAST_ONLY_AFFINE_PARAMETERS.json',
 'links':'Phase_Integrated_SOH_Physics_RUL_Research/outputs/P1S_TO_SOH_EXACT_FORECAST_LINKS.csv',
 'contribution':'Phase_P1S_ForecastStep_Independent_Contribution/outputs/RESEARCH_DECISION.json',
 'eis':'Phase_RealHyFC_EIS_HFR_Physics_Foundation/outputs/RESEARCH_DECISION.json',
 'tft':'Phase_Integrated_SOH_Physics_RUL_Research/outputs/TFT_PAST_INFORMATION_AND_PARAMETER_AVAILABILITY.csv'
}.items()})
FEATURES=['origin_R','mean_R','std_R','slope_R_per_h','origin_minus_median_R']
VREF=.71383023635; SOHREF=.6989533305168152; ACCESS=[]
def sha(q):
 h=hashlib.sha256()
 with Path(q).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def firewall():
 allowed={q.resolve() for q in SOURCES.values()}
 def hook(event,args):
  if event!='open' or isinstance(args[0],int):return
  q=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
  write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if q.is_relative_to(REPO) and not q.is_relative_to(P):
   if write or q not in allowed:raise PermissionError('HISTORICAL_WRITE_OR_UNLISTED_READ: '+str(q))
   ACCESS.append(q.relative_to(REPO).as_posix())
 sys.addaudithook(hook)
def out(s):
 q=(P/s).resolve();assert q.is_relative_to(P);q.parent.mkdir(parents=True,exist_ok=True);return q
def js(s,d):out(s).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def jl(s):return json.loads((P/s).read_text(encoding='utf-8'))
def cs(s,d):d.to_csv(out(s),index=False,float_format='%.17g',lineterminator='\n')
def read(k,**kw):return pd.read_csv(SOURCES[k],float_precision='round_trip',**kw)
def sj(k):return json.loads(SOURCES[k].read_text(encoding='utf-8'))
def md(s,t):out(s).write_text(t.strip()+'\n',encoding='utf-8')
def config():
 c=jl('configs/FROZEN_PROTOCOL.json');assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 for k,h in c['source_sha256'].items():assert sha(SOURCES[k])==h,k
 for k,h in c['code_sha256'].items():assert sha(P/k)==h,k
 return c
def metric(y,p):
 e=(np.asarray(p)-np.asarray(y))*VREF*1000;a=abs(e)
 return {'n':len(e),'MAE_mVeq':float(a.mean()),'RMSE_mVeq':float(np.sqrt((e*e).mean())),'P95_mVeq':float(np.quantile(a,.95)),'max_mVeq':float(a.max())} if len(e) else {'n':0,'MAE_mVeq':None,'RMSE_mVeq':None,'P95_mVeq':None,'max_mVeq':None}

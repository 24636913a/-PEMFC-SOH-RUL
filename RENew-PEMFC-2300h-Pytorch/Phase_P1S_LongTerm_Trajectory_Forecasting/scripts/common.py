"""Named Development caches only, historical paths strictly read-only."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
SOURCES={
 'identity':'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',
 'measurement':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',
 'partial':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/PREANCHOR_PARTIAL_CYCLE_INTERVALS.csv',
 'catalog':'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv',
 'support':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_CONDITION_SUPPORT.csv',
 'native_features':'Phase_P1S_96h_ThreeFold_Comparison/inputs/L96_REUSED_FEATURES.csv',
 'native_predictions':'Phase_P1S_96h_ThreeFold_Comparison/outputs/PREDICTIONS.csv',
 'native_parameters':'Phase_P1S_96h_ThreeFold_Comparison/outputs/TRAINED_PARAMETERS.json',
 'native_metrics':'Phase_P1S_96h_ThreeFold_Comparison/outputs/FOLD_METRICS.csv',
 'native_time':'Phase_P1S_96h_ThreeFold_Comparison/outputs/PARAMETER_TIME_AUDIT.csv',
 'native_legal':'Phase_P1S_96h_ThreeFold_Comparison/outputs/LEGAL_SAMPLES.csv',
 'native_protocol':'Phase_P1S_96h_ThreeFold_Comparison/configs/FROZEN_PROTOCOL.json',
 'eis':'Phase_RealHyFC_EIS_HFR_Physics_Foundation/outputs/RESEARCH_DECISION.json',
 'eis_gates':'Phase_RealHyFC_EIS_HFR_Physics_Foundation/outputs/CPU_PHYSICS_QUALIFICATION_GATES.csv',
 'physics':'Phase_Integrated_SOH_Physics_RUL_Research/outputs/PHYSICS_CPU_DIAGNOSTICS.json',
 'tft_time':'Phase_Integrated_SOH_Physics_RUL_Research/outputs/TFT_PAST_INFORMATION_AND_PARAMETER_AVAILABILITY.csv',
 'f2_cache':'Phase_H_Determinism_MultiSeed_Stability/outputs/validation_predictions/H-F2-Seed42.csv',
 'bins':'Phase_TFT_DynamicVoltage_to_ReferenceSOH/inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',
 'soh':'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv',
 'calibration':'Phase_Integrated_SOH_Physics_RUL_Research/models/PAST_ONLY_AFFINE_PARAMETERS.json',
 'forecast_contribution':'Phase_P1S_ForecastStep_Independent_Contribution/outputs/RESEARCH_DECISION.json',
 'soh_boundary':'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md',
 'eol':'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md'}
FEATURES=['origin_R','mean_R','std_R','slope_R_per_h','origin_minus_median_R']
CONDITIONS=['current_density','anode_inlet_temperature_C','anode_inlet_pressure_kPag','anode_inlet_RH_pct','anode_hydrogen_flow_nl_min','anode_stoichiometry','cathode_stoichiometry']
VREF=.71383023635;ACCESS=[]
def sha(q):return hashlib.sha256(Path(q).read_bytes()).hexdigest()
def firewall():
 allowed={(ROOT/q).resolve() for q in SOURCES.values()}
 def hook(event,args):
  if event!='open' or isinstance(args[0],int):return
  raw,mode,flags=args;q=Path(os.fsdecode(raw)).resolve()
  write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if q.is_relative_to(REPO) and not q.is_relative_to(P):
   if write or q not in allowed:raise PermissionError('NON_ALLOWLISTED_OR_HISTORICAL_WRITE: '+str(q))
   ACCESS.append(q.relative_to(ROOT).as_posix())
 sys.addaudithook(hook)
def source(k,**kwargs):return pd.read_csv(ROOT/SOURCES[k],float_precision='round_trip',**kwargs)
def jl(k):return json.loads((ROOT/SOURCES[k]).read_text(encoding='utf-8'))
def local(k):return json.loads((P/k).read_text(encoding='utf-8'))
def out(k):
 q=(P/k).resolve();assert q.is_relative_to(P);q.parent.mkdir(parents=True,exist_ok=True);return q
def js(k,obj):out(k).write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def cs(k,d):d.to_csv(out(k),index=False,float_format='%.17g',lineterminator='\n')
def md(k,s):out(k).write_bytes((s.strip()+'\n').encode('utf-8'))
def config():
 c=local('configs/FROZEN_PROTOCOL.json');assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 for s,h in c['source_sha256'].items():assert sha(ROOT/s)==h,s
 for s,h in c['calculation_code_sha256'].items():assert sha(P/s)==h,s
 return c
def metric(y,p):
 e=(np.asarray(p,float)-np.asarray(y,float))*VREF*1000;a=abs(e)
 if not len(e):return dict(n=0,MAE_mVeq=None,RMSE_mVeq=None,P95_mVeq=None,max_mVeq=None,bias_mVeq=None)
 return dict(n=len(e),MAE_mVeq=float(a.mean()),RMSE_mVeq=float(np.sqrt(np.mean(e*e))),P95_mVeq=float(np.quantile(a,.95)),max_mVeq=float(a.max()),bias_mVeq=float(e.mean()))

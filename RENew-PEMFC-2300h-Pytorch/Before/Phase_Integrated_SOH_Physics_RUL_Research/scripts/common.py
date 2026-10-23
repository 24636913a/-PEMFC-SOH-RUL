"""Named Development artifacts only; never loads a mixed N1 or raw target table."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]; ROOT=P.parent; REPO=ROOT.parent
SOURCES={
 'refs':'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv',
 'timing':'Phase_PAV_PAA_Anchor_Validity_Qualification/outputs/POLARIZATION_TIME_PROVENANCE.csv',
 'support':'Phase_PAV_PAA_Anchor_Validity_Qualification/outputs/ANCHOR_MEDIAN_SUPPORT_PROVENANCE.csv',
 'measure':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',
 'identity':'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',
 'preds':'Phase_P1S_96h_ThreeFold_Comparison/outputs/PREDICTIONS.csv',
 'samples':'Phase_P1S_96h_ThreeFold_Comparison/outputs/LEGAL_SAMPLES.csv',
 'params':'Phase_P1S_96h_ThreeFold_Comparison/outputs/TRAINED_PARAMETERS.json',
 'parameter_time':'Phase_P1S_96h_ThreeFold_Comparison/outputs/PARAMETER_TIME_AUDIT.csv',
 'p1metrics':'Phase_P1S_96h_ThreeFold_Comparison/outputs/THREE_FOLD_EQUAL_MEANS.csv',
 'p1folds':'Phase_P1S_96h_ThreeFold_Comparison/outputs/FOLD_METRICS.csv',
 'ahc':'Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/outputs/B2_CHRONOLOGICAL_PREFIX_EVALUATION.csv',
 'bins':'Phase_TFT_DynamicVoltage_to_ReferenceSOH/inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',
 'o1metrics':'Phase_O1_Nine_Model_Validation_Robustness_Audit/outputs/O1_NINE_MODEL_VALIDATION_METRICS.csv',
 'o1equal':'Phase_O1_Nine_Model_Validation_Robustness_Audit/outputs/O1_EQUAL_SEED_SUMMARY.csv',
 'eis':'Phase_RealHyFC_EIS_HFR_Physics_Foundation/outputs/RESEARCH_DECISION.json',
 'eisprovenance':'Phase_RealHyFC_EIS_HFR_Physics_Foundation/inputs/PHYSICAL_METHOD_EVIDENCE_LEDGER.csv',
 'long_predictions':'Phase_SOH_LongHorizon_Future_Forecasting/outputs/PREDICTIONS_BEFORE_EVALUATION.json',
}
REVIEWS=[
 'Phase_H_Determinism_MultiSeed_Stability/README.md',
 'Phase_P2_Final_Gate_Ablation_Training/outputs/P2_FINAL_RESEARCH_DECISION.json',
 'Phase_P3_Full450_Epoch_Sensitivity/outputs/P3_FINAL_SENSITIVITY_DECISION.json',
 'Phase_O1_Nine_Model_Validation_Robustness_Audit/reports/O1_FINAL_RESEARCH_DECISION.md',
 'Phase_P1S_96h_ThreeFold_Comparison/00_实验总结.md',
 'Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/reports/FINAL_AFFINE_CALIBRATION_CHARACTERIZATION.md',
 'Phase_PAV_PAA_Anchor_Validity_Qualification/reports/FINAL_PAA_VALIDITY_QUALIFICATION.md',
 'Phase_SOH_LongHorizon_Future_Forecasting/01_数据与预测任务定义.md',
 'Phase_TFT_DynamicVoltage_to_ReferenceSOH/outputs/FINAL_DECISION.json',
 'Phase_RealHyFC_EIS_HFR_Physics_Foundation/09_SOH_RUL研究条件与下一步.md',
 'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md',
 'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md',
]
for model in ['F0','F1','F2']:
 for seed in [42,52,62]:SOURCES[f'{model}_{seed}']=f'Phase_H_Determinism_MultiSeed_Stability/outputs/validation_predictions/H-{model}-Seed{seed}.csv'
ACCESS=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def firewall():
 allowed={(ROOT/v).resolve() for v in list(SOURCES.values())+REVIEWS}
 def hook(event,args):
  if event!='open' or isinstance(args[0],int):return
  raw,mode,flags=args; path=Path(os.fsdecode(raw)).resolve()
  write=bool((isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
  if path.is_relative_to(REPO) and not path.is_relative_to(P):
   if write or path not in allowed:raise PermissionError('OUTSIDE_PHASE_OR_SOURCE_WHITELIST: '+str(path))
   ACCESS.append({'path':path.relative_to(ROOT).as_posix(),'mode':'READ_ONLY'})
 sys.addaudithook(hook)
def js(name,obj):
 q=(P/name).resolve();assert q.is_relative_to(P);q.parent.mkdir(parents=True,exist_ok=True)
 q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def cs(name,d):
 q=(P/name).resolve();assert q.is_relative_to(P);q.parent.mkdir(parents=True,exist_ok=True)
 d.to_csv(q,index=False,float_format='%.17g',lineterminator='\n')
def md(name,s):
 q=(P/name).resolve();assert q.is_relative_to(P);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s.strip()+'\n',encoding='utf-8')
def source(k):return pd.read_csv(ROOT/SOURCES[k])
def load_json(k):return json.loads((ROOT/SOURCES[k]).read_text(encoding='utf-8'))
def config():
 q=P/'configs/FROZEN_PROTOCOL.json';assert sha(q)==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 c=json.loads(q.read_text(encoding='utf-8'))
 for name,h in c['source_sha256'].items():assert sha(ROOT/name)==h,name
 return c
def metrics(y,p):
 e=(np.asarray(p)-np.asarray(y))*100
 return {'n':len(e),'MAE_SOH_pp':float(np.abs(e).mean()),'RMSE_SOH_pp':float(np.sqrt(np.mean(e*e))),
  'P95_abs_SOH_pp':float(np.quantile(np.abs(e),.95)),'max_abs_SOH_pp':float(np.abs(e).max())}

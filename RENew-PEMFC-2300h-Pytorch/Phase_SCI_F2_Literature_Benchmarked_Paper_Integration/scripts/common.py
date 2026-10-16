"""Cache-only paper integration. No torch, fitting, old runner or raw-data loader."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
BASE='e3dc5844cde3e4d2e39c4fdba9baeb575e879f33'
H=ROOT/'Phase_H_Determinism_MultiSeed_Stability'
I2=ROOT/'Phase_I2_Frozen_Assessment_Evaluation'
J=ROOT/'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit'
O=ROOT/'Phase_O_F2_Paper_FullText_Novelty_Audit'
O1=ROOT/'Phase_O1_Nine_Model_Validation_Robustness_Audit'
P2=ROOT/'Phase_P2_Final_Gate_Ablation_Training'
R1=ROOT/'Phase_SCI_F2_Overnight_Seed_Screening'
R2=ROOT/'Phase_SCI_F2_Seed_Screening_Round2'
SIX=ROOT/'Phase_SCI_F2_SixSeed_Gate_Mechanism_Audit'
SEEDS=[42,52,62];SELECTED=[42,103,193,202,210,212]
CONDITIONS={'current_variability':('actual_current_density_A_cm2_std',9),'load_transitions':('load_transition_count',11),'load_level':('actual_current_density_A_cm2_mean',8)}
def sha(p):
    p=Path(p);assert p.name!='causal_features_30min.csv','FULL_MIXED_RAW_ACCESS_FORBIDDEN'
    return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(p):return pd.read_csv(p,float_precision='round_trip')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dst(p):
    q=(P/p).resolve();assert q.is_relative_to(P),'HISTORY_WRITE_FORBIDDEN'
    q.parent.mkdir(parents=True,exist_ok=True);return q
def text(p,s):dst(p).write_text(s,encoding='utf-8')
def js(p,o):text(p,json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cs(p,f):pd.DataFrame(f).to_csv(dst(p),index=False,float_format='%.17g',lineterminator='\n')
def mdtable(f,digits=6):
    def val(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}g}'
        return str(v).replace('|','\\|').replace('\n',' ')
    return '|'+ '|'.join(map(str,f.columns))+'|\n|'+'|'.join(['---']*len(f.columns))+'|\n'+'\n'.join('|'+ '|'.join(val(v) for v in r)+'|' for r in f.itertuples(index=False,name=None))
def metrics(f):
    y=f.y_true_delta_V.to_numpy(dtype='float64');p=f.y_pred_delta_V.to_numpy(dtype='float64');e=(p-y)*1000;a=abs(e);z=e[f.horizon_step.to_numpy()==24]
    target=pd.DataFrame({'t':f.target_row,'sq':e*e}).groupby('t').sq.mean()
    return dict(n=len(f),origins=int(f.origin_row.nunique()),unique_targets=int(f.target_row.nunique()),RMSE_mV=float(np.sqrt(np.mean(e*e))),
        MAE_mV=float(np.mean(a)),R2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)),endpoint_RMSE_mV=float(np.sqrt(np.mean(z*z))) if len(z) else None,
        target_time_equal_RMSE_mV=float(np.sqrt(target.mean())),P95_mV=float(np.quantile(a,.95)),P99_mV=float(np.quantile(a,.99)),max_abs_error_mV=float(max(a)))
def verified_config():
    c=read(P/'outputs/audits/FROZEN_ANALYSIS.json');assert sha(P/'outputs/audits/FROZEN_ANALYSIS.json')==(P/'outputs/audits/FROZEN_ANALYSIS.sha256').read_text().strip()
    for row in c['sources']:assert sha(REPO/row['path'])==row['sha256'],('SOURCE_CHANGED',row['path'])
    return c

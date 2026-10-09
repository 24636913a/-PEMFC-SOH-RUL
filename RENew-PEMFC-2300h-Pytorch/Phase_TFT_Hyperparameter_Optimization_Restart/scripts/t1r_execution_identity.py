"""Frozen execution-code identity for one authorized T1-R run."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from t1r_common import PHASE, REPO, T1RFailure

RELATIVE=(
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/scripts/run_t1r.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/scripts/t1r_real_adapter.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/scripts/t1r_training.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/scripts/t1r_optuna.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/scripts/t1r_evaluation.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/scripts/t1r_decision.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Hyperparameter_Optimization_Restart/scripts/t1r_common.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Original_Baseline/models/canonical_tft.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_TFT_Original_Baseline/scripts/tft_common.py',
 'RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/scripts/backbone_common.py',
)
MANIFEST=PHASE/'configs'/'T1R_EXECUTION_CODE_IDENTITY.json'
def digest(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def payload():
    files={p:digest(REPO/p) for p in RELATIVE}
    return {'identity_version':1,'files':files}
def overall(value:dict)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_manifest():
    value=payload(); value['manifest_sha256']=overall(value); MANIFEST.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return value
def verify_manifest(expected:str|None=None):
    if not MANIFEST.is_file():raise T1RFailure('EXECUTION_IDENTITY_MANIFEST_MISSING')
    value=json.loads(MANIFEST.read_text(encoding='utf-8')); observed=value.pop('manifest_sha256',None)
    if observed!=overall(value) or value!=payload() or (expected is not None and observed!=expected):raise T1RFailure('EXECUTION_CODE_IDENTITY_MISMATCH')
    value['manifest_sha256']=observed; return value
if __name__=='__main__': print(json.dumps(write_manifest(),indent=2))

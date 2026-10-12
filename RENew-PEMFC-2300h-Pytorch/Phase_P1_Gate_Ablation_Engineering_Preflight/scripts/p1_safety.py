"""Repo read firewall, non-executable protocol and optimizer-step guard."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

PHASE=Path(__file__).resolve().parents[1]
REPO=PHASE.parent.parent
ROOT='RENew-PEMFC-2300h-Pytorch/'
BASE='f4a47e51fd733878873fc5bbdea9e7caec1c1055'
SOURCE_NAMES=(
 'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml',
 'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.sha256',
 'Phase_H_Determinism_MultiSeed_Stability/scripts/run_h_training.py',
 'Phase_H_Determinism_MultiSeed_Stability/scripts/h_common.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/__init__.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/causal_multiscale_branch.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/frozen_feature_schema.py',
 'Phase_TFT_Original_Baseline/models/canonical_tft.py',
 'Phase_TFT_FullFeature_CausalFilter_Comparison/configs/PREREGISTERED_E_COMPARISON.yaml',
 'Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py',
 'Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts/run_c_ablation.py',
 'Phase_P_Final_Innovation_and_Experiment_Planning/protocol/P_PROPOSED_GATE_PROTOCOL.json',
 'Phase_P_Final_Innovation_and_Experiment_Planning/protocol/P_PROPOSED_GATE_PROTOCOL.sha256',
 'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json',
)
ALLOWED={str((REPO/(ROOT+p)).resolve()).casefold() for p in SOURCE_NAMES}
EVENTS=[]
OPTIMIZER_ATTEMPTS=[]
_installed=False


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def permitted_open(path,writing=False):
    if isinstance(path,int):
        return
    path=Path(os.fsdecode(path)).resolve()
    if path.is_relative_to(PHASE.resolve()):
        return
    if path.is_relative_to(REPO.resolve()):
        if writing or str(path).casefold() not in ALLOWED:
            raise PermissionError('P1_REPO_IO_NOT_ALLOWED:'+str(path))
        return
    # Real raw sources can reside outside the repository, too.
    parts=[x.casefold() for x in path.parts]
    if '数据' in parts or path.name.startswith(('Loadcycle_','Polcurve_')) or path.suffix.lower() in ('.pt','.npz','.parquet','.csv'):
        raise PermissionError('P1_EXTERNAL_DATA_IO_NOT_ALLOWED:'+str(path))


def install_repo_firewall():
    global _installed
    if _installed:
        return
    def hook(event,args):
        if event!='open' or isinstance(args[0],int):
            return
        path,mode,flags=args
        writing=bool((isinstance(mode,str) and any(s in mode for s in ('w','a','x','+'))) or
                     (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        try:
            permitted_open(path,writing)
        except PermissionError:
            EVENTS.append({'path':os.fsdecode(path),'writing':writing,'action':'DENIED'})
            raise
        if str(REPO).casefold() in os.fsdecode(path).casefold():
            EVENTS.append({'path':os.fsdecode(path),'writing':writing,'action':'ALLOWED'})
    sys.addaudithook(hook);_installed=True


def forbid_optimizer_steps():
    import torch
    import inspect
    original=[]
    def blocked(*args,**kwargs):
        OPTIMIZER_ATTEMPTS.append('BLOCKED_SYNTHETIC_FIREWALL_ATTEMPT')
        raise PermissionError('P1_OPTIMIZER_STEP_PROHIBITED')
    for name in dir(torch.optim):
        cls=getattr(torch.optim,name)
        if inspect.isclass(cls) and issubclass(cls,torch.optim.Optimizer) and hasattr(cls,'step'):
            original.append((cls,cls.step));cls.step=blocked
    return original


def check_h_early_stop(source):
    tree=ast.parse(source)
    match=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.If):
            continue
        test=node.test
        if isinstance(test,ast.Compare) and isinstance(test.left,ast.Name) and test.left.id=='rmse' and len(test.ops)==1 and isinstance(test.ops[0],ast.Lt):
            rhs=test.comparators[0]
            if isinstance(rhs,ast.BinOp) and isinstance(rhs.op,ast.Sub) and isinstance(rhs.left,ast.Name) and rhs.left.id=='best' and isinstance(rhs.right,ast.Constant) and rhs.right.value==1e-12:
                match.append(ast.get_source_segment(source,node).splitlines()[0])
    if len(match)!=1 or "'validation_standardized_RMSE':rmse" not in source or 'pred-vy' not in source:
        raise RuntimeError('H_STANDARDIZED_EARLY_STOP_NOT_VERIFIED')
    return {'condition':match[0],'monitor':'validation_standardized_RMSE','comparison':'< best - 1e-12',
            'mean_square_in_standardized_space':True}


def require_preflight_only(protocol):
    if protocol['training_authorized'] is not False or protocol['mode']!='ENGINEERING_PREFLIGHT_ONLY':
        raise PermissionError('P1_REAL_TRAINING_AUTHORITY_REJECTED')
    if protocol['optimizer_steps_allowed']!=0 or protocol['real_data_load_allowed'] is not False:
        raise PermissionError('P1_UNSAFE_EXECUTION_SCOPE')

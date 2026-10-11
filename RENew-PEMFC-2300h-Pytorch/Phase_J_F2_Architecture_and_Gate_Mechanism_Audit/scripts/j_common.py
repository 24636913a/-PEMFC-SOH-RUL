"""Phase J: narrow read authority, Validation-only H Path A, no training APIs."""
from __future__ import annotations
import ast
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml

PHASE = Path(__file__).resolve().parents[1]
PROJECT, REPO = PHASE.parent, PHASE.parent.parent
BASE = '427d4dc402f083a5b9e094b12981c5e231df8100'
IR = PROJECT/'Phase_IR_Validation_Reload_Parity_Audit'
I2 = PROJECT/'Phase_I2_Frozen_Assessment_Evaluation'
sys.path.insert(0, str(IR/'scripts'))
import ir_common as ir
i, h, e, c, H = ir.i, ir.h, ir.e, ir.c, ir.H
from i_metrics import project_metrics, independent_metrics
from models.regime_aware_dualpath_tft import REGIME_FEATURE_NAMES
INDICES = [11, 13, 9, 6, 8]
SEEDS = [42, 52, 62]
OUT = PHASE/'outputs'
PROTOCOL = PHASE/'configs/J_FROZEN_AUDIT_PROTOCOL.yaml'
SHA = PROTOCOL.with_suffix('.sha256')
INPUTS = OUT/'validation_parity/J_VALIDATION_INPUTS.npz'
LIMITS = ['F2_VALIDATION_MULTISEED_GAIN_OBSERVED', 'F2_REUSED_ASSESSMENT_MULTISEED_GAIN_OBSERVED',
 'PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS', 'C1_D0_E0_REPRODUCIBILITY_UNRESOLVED',
 'TIMESNET_FREEZE', 'BACKBONE_SEARCH_CLOSED']

def historical_paths():
    paths = list(ir.ALLOW) + [ir.PROTOCOL, ir.SHA, IR/'outputs/IR_SUMMARY.json',
        IR/'outputs/input_identity/IR_FIXED_VALIDATION_INPUT.npz',
        h.F/'reports/F_IMPLEMENTATION_AUDIT.md',
        H/'reports/H_FINAL_CONCLUSION.md', IR/'reports/IR_FINAL_RECOMMENDATION.md',
        I2/'configs/I2_ASSESSMENT_PROTOCOL.yaml', I2/'configs/I2_ASSESSMENT_PROTOCOL.sha256',
        I2/'outputs/metrics/I2_ASSESSMENT_METRICS.csv',
        I2/'outputs/metrics/I2_MULTISEED_SUMMARY.csv',
        I2/'reports/I2_FINAL_SCIENTIFIC_CONCLUSION.md']
    paths += list((IR/'scripts').glob('*.py')) + list((IR/'tests').glob('*.py'))
    paths += list((H/'tests').glob('*.py'))
    paths += list((I2/'scripts').glob('*.py')) + list((I2/'tests').glob('*.py'))
    # Reject Assessment input / point-prediction caches even if upstream allows them.
    return sorted(set(p.resolve() for p in paths if p not in i.F_PRED))

def allowed(path, *, prefix=False):
    p = Path(path).resolve()
    if p == c.N1.resolve():
        if prefix: return p
        raise RuntimeError('FULL_N1_READ_OR_HASH_PROHIBITED')
    if p.is_relative_to(PHASE) or p in historical_paths(): return p
    raise RuntimeError('J_SOURCE_NOT_ALLOWLISTED:'+str(p))

def digest(path): return hashlib.sha256(allowed(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(allowed(path).read_text(encoding='utf-8'))
def read_csv(path): return pd.read_csv(allowed(path), float_precision='round_trip')
def write_target(path):
    p = Path(path).resolve()
    if not p.is_relative_to(PHASE): raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
def text_write(path, text): write_target(path).write_text(text, encoding='utf-8', newline='\n')
def json_write(path, data):
    p=write_target(path); tmp=p.with_suffix(p.suffix+'.writing')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def csv_write(path, data): data.to_csv(write_target(path),index=False,float_format='%.17g',lineterminator='\n')
def npz_write(path, **arrays): np.savez_compressed(write_target(path),**arrays)

def own_code():
    files={p.relative_to(REPO).as_posix():digest(p) for d in ['scripts','tests'] for p in sorted((PHASE/d).glob('*.py'))}
    return {'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}

def verify_records(records):
    for name, expected in records.items():
        p=Path(name) if Path(name).is_absolute() else REPO/name
        if digest(p)!=expected: raise RuntimeError('J_SOURCE_IDENTITY_CHANGED:'+name)

def historical_identity():
    hc, ip = ir.historical_protocols()
    ir.load_protocol()
    p=I2/'configs/I2_ASSESSMENT_PROTOCOL.yaml'
    if digest(p)!=(p.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]):
        raise RuntimeError('I2_PROTOCOL_CHANGED')
    cfg=yaml.safe_load(p.read_text(encoding='utf-8'))
    # Only code identity and explicit small metric/report authorities are read.
    verify_records(cfg['code_identity']['files'])
    return hc, ip

def load_protocol():
    if digest(PROTOCOL)!=SHA.read_text(encoding='utf-8').split()[0]: raise RuntimeError('J_PROTOCOL_CHANGED')
    cfg=yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    if own_code()!=cfg['code_identity']: raise RuntimeError('J_CODE_IDENTITY_CHANGED')
    verify_records(cfg['sources']); verify_records(cfg['prepared_inputs'])
    historical_identity()
    return cfg

@contextmanager
def prefix_guard(access):
    old=pd.read_csv; old_bytes=Path.read_bytes
    def bounded(path,*args,**kw):
        p=Path(path).resolve()
        if p==c.N1.resolve():
            if kw.get('nrows')!=3007: raise RuntimeError('UNBOUNDED_N1_READ')
            access.append({'path':str(p),'nrows':3007,'purpose':'frozen logical Development prefix'})
        else: allowed(p)
        return old(path,*args,**kw)
    def guarded_bytes(path):
        if path.resolve()==c.N1.resolve(): raise RuntimeError('FULL_N1_BYTES_PROHIBITED')
        return old_bytes(path)
    with patch.object(pd,'read_csv',bounded), patch.object(Path,'read_bytes',guarded_bytes): yield

def read_dev():
    access=[]
    with prefix_guard(access): dev=c.load_development({})
    if len(dev)!=3005 or len(access)!=1 or not (dev.bin_end_ophrs2_h<1819.997738).all():
        raise RuntimeError('DEVELOPMENT_PREFIX_BOUNDARY_MISMATCH')
    return dev, access

def validate_keys(origins, targets, ages):
    if len(origins)!=205 or len(set(origins.tolist()))!=205 or targets.shape!=(205,24):
        raise RuntimeError('VALIDATION_KEY_COUNT_OR_DUPLICATE')
    if not np.array_equal(targets,origins[:,None]+np.arange(1,25)[None,:]): raise RuntimeError('TARGET_KEY_IDENTITY_MISMATCH')
    if (origins<2103).any() or (targets>=2404).any() or not (ages<1819.997738).all(): raise RuntimeError('VALIDATION_BOUNDARY_VIOLATION')
    if not (np.diff(origins)>0).all(): raise RuntimeError('VALIDATION_ORDER_MISMATCH')

def validate_gates(g, contexts, n):
    if g.shape!=(n,) or contexts.shape!=(n,5): raise RuntimeError('GATE_COUNT_OR_CONTEXT_SHAPE_MISMATCH')
    if not np.isfinite(g).all() or not np.isfinite(contexts).all() or not ((g>=0)&(g<=1)).all():
        raise RuntimeError('GATE_NONFINITE_OR_OUT_OF_RANGE')

def verify_state(audit):
    if not (audit['all_state_entries_equal'] and audit['requires_grad_unchanged']
        and audit['all_parameter_grad_None'] and all(audit['requires_grad_before'].values())):
        raise RuntimeError('STATE_OR_GRADIENT_FLAGS_CHANGED')

def passive_predict(model, x, device):
    gs=[]; zs=[]
    def capture(_m,args,out):
        gs.append(out.detach().cpu().numpy().reshape(-1).copy())
        zs.append(args[0].detach().cpu().numpy().copy())
        return None
    handle=model.regime_gate.register_forward_hook(capture)
    try: p=e.predict(model,x,32,device)
    finally: handle.remove()
    g,z=np.concatenate(gs),np.concatenate(zs)
    validate_gates(g,z,len(x))
    return p,g,z

def fixed_predict(model, x, device):
    handle=model.regime_gate.register_forward_hook(lambda _m,_a,out:torch.full_like(out,0.5))
    try: return e.predict(model,x,32,device)
    finally: handle.remove()

def hooks_empty(model):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())

@contextmanager
def no_training_guard():
    def forbidden(*_args,**_kw): raise RuntimeError('J_TRAINING_OPERATION_PROHIBITED')
    with patch.object(torch.Tensor,'backward',forbidden), patch.object(torch.optim.Optimizer,'__init__',forbidden): yield

def load_inputs():
    with np.load(allowed(INPUTS),allow_pickle=False) as ar: return {k:ar[k].copy() for k in ar.files}

def load_cuda_model(seed,cfg):
    hc=h.load_protocol();env=h.configure_normal(hc)
    manifest=cfg['checkpoints'][f'H-F2-Seed{seed}']
    ck=torch.load(allowed(REPO/manifest['path']),map_location='cpu',weights_only=True)
    model=h.model('F2',hc['hyperparameters'],torch.device('cuda'))
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    if not all(p.requires_grad and p.grad is None for p in model.parameters()) or not hooks_empty(model):
        raise RuntimeError('ORIGINAL_FLAGS_OR_HOOKS_INVALID')
    return model,ck,env

def prediction_frame(data, prediction, seed, mode, cfg):
    truth=read_csv(OUT/'validation_parity/J_VALIDATION_TRUTH.csv')
    frame=truth.copy()
    frame['y_pred_delta_V']=(prediction*data['target_scale_V'].item()).ravel().astype(np.float64)
    frame['seed']=seed; frame['run_id']=f'J_F2_Seed{seed}'; frame['mode']=mode
    frame['checkpoint_sha256']=cfg['checkpoints'][f'H-F2-Seed{seed}']['sha256']
    frame['protocol_sha256']=digest(PROTOCOL)
    frame['input_fingerprint']=cfg['input_fingerprints']['validation_x_sha256']
    return frame

def validate_frame(frame, truth, seed, mode, cfg):
    keys=['origin_row','target_row','horizon_step']
    if len(frame)!=4920 or frame.duplicated(keys).any(): raise RuntimeError('PREDICTION_KEYS_INVALID')
    if not np.array_equal(frame[keys].to_numpy(),truth[keys].to_numpy()): raise RuntimeError('PREDICTION_KEYS_MISMATCH')
    for key in ['y_true_delta_V','target_age_h']:
        if not np.array_equal(frame[key].to_numpy(),truth[key].to_numpy()): raise RuntimeError('PREDICTION_TRUTH_MISMATCH')
    for key,value in {'seed':seed,'mode':mode,'run_id':f'J_F2_Seed{seed}',
        'protocol_sha256':digest(PROTOCOL),'checkpoint_sha256':cfg['checkpoints'][f'H-F2-Seed{seed}']['sha256'],
        'input_fingerprint':cfg['input_fingerprints']['validation_x_sha256']}.items():
        if not (frame[key]==value).all(): raise RuntimeError('PREDICTION_IDENTITY_MISMATCH:'+key)
    if not np.isfinite(frame[['y_true_delta_V','y_pred_delta_V']].to_numpy()).all(): raise RuntimeError('PREDICTION_NONFINITE')
    if not frame.target_row.between(2103,2403).all() or not (frame.target_age_h<1819.997738).all(): raise RuntimeError('PREDICTION_BOUNDARY_FAILURE')
    return True

def gate_statistics(g):
    validate_gates(np.asarray(g),np.zeros((len(g),5)),len(g))
    return {'n':len(g),'mean':float(np.mean(g)),'median':float(np.median(g)), 'SD':float(np.std(g,ddof=1)),
        'min':float(np.min(g)),'max':float(np.max(g)),
        **{'P'+str(q):float(np.quantile(g,q/100)) for q in [5,25,75,95]},
        'IQR':float(np.quantile(g,.75)-np.quantile(g,.25)),
        'low_lt_0p1_fraction':float(np.mean(g<.1)), 'middle_0p4_to_0p6_fraction':float(np.mean((g>=.4)&(g<=.6))),
        'high_gt_0p9_fraction':float(np.mean(g>.9)), 'temporal_range':float(np.ptp(g)),
        'mean_absolute_deviation_from_0p5':float(np.mean(np.abs(g-.5)))}

def gate_decision(stats, comparisons):
    dynamic=(stats.SD>1e-3)&(stats.temporal_range>1e-2)
    affected=comparisons.prediction_difference_RMSE_mV>1e-3
    beneficial=comparisons.learned_minus_fixed_RMSE_mV<-1e-3
    if dynamic.all() and affected.all() and beneficial.all(): return 'PHASE_J_GATE_DYNAMIC_AND_PREDICTIVE_EFFECT_OBSERVED'
    if dynamic.any() and affected.any(): return 'PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED'
    return 'PHASE_J_GATE_MECHANISM_NOT_SUBSTANTIATED'

def static_audit():
    banned={'fit','backward','step','train','predict_frozen','inference_mode','requires_grad_', 'prepare_data','prepare','top10_scores','spearman_scores','AdamW','Adam','SGD','create_study'}
    calls=[]
    for path in (PHASE/'scripts').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                if name in banned: raise RuntimeError('J_FORBIDDEN_EXECUTION_CALL:'+name)
                if name in ['build_windows','predict','read_dev','read_csv','digest']: calls.append({'file':path.name,'line':node.lineno,'call':name})
    return {'pass':True,'banned_calls':sorted(banned),'allowed_inherited_calls':calls,
        'Assessment_windows_constructed':False,'Assessment_inference':False,'training':False,'Test_accessed':False}

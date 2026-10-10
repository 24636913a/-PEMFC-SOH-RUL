"""Read-only historical authorities and phase-local writes; no full N1 hash."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
BASE = '11f5d23fe075ee68288bf34bce6277644c963d09'
E = PROJECT / 'Phase_TFT_FullFeature_CausalFilter_Comparison'
F = PROJECT / 'Phase_F_TFT_DualPath_RegimeFusion'
C = PROJECT / 'Phase_TFT_SingleSplit_Feature_Filter_Ablation'
T0 = PROJECT / 'Phase_TFT_Original_Baseline/models/canonical_tft.py'
OUT = PHASE / 'outputs'
PROTOCOL = PHASE / 'configs/G_REPLAY_PROTOCOL.yaml'
SHA = PROTOCOL.with_suffix('.sha256')
HISTORICAL = [
    E/'scripts/run_e_comparison.py', F/'scripts/run_f_training.py', T0,
    C/'scripts/run_c_ablation.py', E/'configs/PREREGISTERED_E_COMPARISON.yaml',
    F/'configs/F0_F2_EXECUTION_PROTOCOL.yaml',
    E/'outputs/training_logs/E2_seed42.json', E/'outputs/E_PRETRAIN_AUDIT.json',
    E/'outputs/E_COMMON_SCALER.json', E/'outputs/checkpoints/E2_Top10_EWMA_seed42.pt',
    E/'outputs/E2_predictions_seed42.csv', F/'outputs/audits/F_ENVIRONMENT.json',
    F/'outputs/audits/F_COMMON_SCALER.json', F/'outputs/audits/F_DATA_ACCESS_AUDIT.json',
    F/'outputs/audits/F_FORMAL_INITIALIZATION_AUDIT.json', F/'outputs/audits/F0_FIRST_BATCH.json',
    F/'outputs/training_logs/F0_epochs_seed42.csv', F/'outputs/training_logs/F0_run_seed42.json',
    F/'outputs/checkpoints/F0_best_seed42.pt', F/'outputs/predictions/F0_assessment_seed42.csv',
]

def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(PHASE):
        raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def file_hash(path):
    path = Path(path).resolve()
    if path.name == 'causal_features_30min.csv':
        raise RuntimeError('FULL_N1_READ_OR_HASH_PROHIBITED')
    if not path.is_relative_to(PHASE) and path not in [p.resolve() for p in HISTORICAL]:
        raise RuntimeError('SOURCE_NOT_ALLOWLISTED:'+str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def identity():
    paths = HISTORICAL + sorted((PHASE/'scripts').glob('*.py')) + sorted((PHASE/'tests').glob('*.py'))
    return {str(p.relative_to(REPO)).replace('\\','/'): file_hash(p) for p in paths}

def import_e():
    sys.path.insert(0, str(E/'scripts'))
    import run_e_comparison
    return run_e_comparison

def array_hash(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def tensor_tree_hash(value):
    """Exact deterministic summary of tensors, RNG or optimizer; no RNG use."""
    import torch
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str((a.shape, str(a.dtype))).encode()); h.update(a.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(repr(key).encode()); visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x: visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()

def compare_batches(a, b):
    """Exact equality; chronological phase of first divergent evidence."""
    if len(a) != len(b): raise RuntimeError('BATCH_RECORD_COUNT_MISMATCH')
    order = ['origins', 'input_sha256', 'label_sha256', 'parameters_before',
             'rng_before', 'optimizer_before', 'forward_sha256', 'rng_after_forward',
             'loss', 'gradient_before_clip', 'gradient_norm', 'gradient_after_clip',
             'rng_after_backward', 'parameters_after', 'optimizer_after', 'rng_after_step']
    first = None; differences = {k: 0 for k in order}
    for x,y in zip(a,b):
        if (x['epoch'],x['batch']) != (y['epoch'],y['batch']):
            raise RuntimeError('BATCH_KEY_ORDER_MISMATCH')
        for k in order:
            if x[k] != y[k]:
                differences[k] += 1
                if first is None: first = {'epoch': x['epoch'], 'batch': x['batch'], 'field': k,
                    'same_origins': x['origins'] == y['origins'], 'left': x[k], 'right': y[k]}
    return {'identical': first is None, 'records': len(a), 'first_divergence': first, 'field_difference_counts': differences}

from __future__ import annotations
import hashlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

PHASE=Path(__file__).resolve().parents[1];PROJECT=PHASE.parent;REPO=PROJECT.parent
BASE='dea436c7520ca9962e7aa3005af1faaedbc786f2'
F=PROJECT/'Phase_F_TFT_DualPath_RegimeFusion'
E=PROJECT/'Phase_TFT_FullFeature_CausalFilter_Comparison'
C=PROJECT/'Phase_TFT_SingleSplit_Feature_Filter_Ablation'
T0=PROJECT/'Phase_TFT_Original_Baseline/models/canonical_tft.py'
OUT=PHASE/'outputs';PROTOCOL=PHASE/'configs/H_FROZEN_PROTOCOL.yaml';SHA=PROTOCOL.with_suffix('.sha256')
sys.path.insert(0,str(E/'scripts'));import run_e_comparison as e
sys.path.insert(0,str(F));from models import DualPathStrictHistoryTFT,RegimeAwareDualPathTFT
from canonical_tft import CanonicalStrictHistoryTFT
COUNTS={'F0':323483,'F1':332379,'F2':332436}
HISTORY=[E/'scripts/run_e_comparison.py',C/'scripts/run_c_ablation.py',T0,
         F/'configs/F0_F2_EXECUTION_PROTOCOL.yaml',E/'outputs/E_COMMON_SCALER.json',
         F/'outputs/audits/F_DATA_ACCESS_AUDIT.json',F/'outputs/audits/F_FORMAL_INITIALIZATION_AUDIT.json']
HISTORY+=sorted((F/'models').glob('*.py'))

def atomic_json(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(PHASE):raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.writing')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def file_hash(path):
    path=Path(path).resolve()
    if path.name=='causal_features_30min.csv':raise RuntimeError('FULL_N1_HASH_PROHIBITED')
    allow=[p.resolve() for p in HISTORY]+upstream_files()
    if not path.is_relative_to(PHASE) and path not in allow:raise RuntimeError('SOURCE_NOT_ALLOWLISTED:'+str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()

def upstream_files():
    from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
    from pytorch_forecasting.models.temporal_fusion_transformer import sub_modules
    return [Path(inspect.getsourcefile(TemporalFusionTransformer)).resolve(),Path(inspect.getsourcefile(sub_modules)).resolve()]

def identity():
    paths=HISTORY+sorted((PHASE/'scripts').glob('*.py'))+sorted((PHASE/'tests').glob('*.py'))+upstream_files()
    records={str(p.relative_to(REPO)).replace('\\','/') if p.is_relative_to(REPO) else str(p):file_hash(p) for p in paths}
    return {'files':records,'sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}

def array_hash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes());return h.hexdigest()

def tree_hash(x):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous().numpy();h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
        elif isinstance(v,np.ndarray):h.update(array_hash(v).encode())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for a in v:visit(a)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()

def rng_summary(gen=None):
    import random
    return {'cpu':tree_hash(torch.get_rng_state()),'cuda':tree_hash(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
            'loader':tree_hash(gen.get_state()) if gen is not None else None,
            'numpy':tree_hash(np.random.get_state()),'python':tree_hash(random.getstate())}

def environment():
    import pytorch_forecasting as pf
    return {'python':platform.python_version(),'torch':str(torch.__version__),'pytorch_forecasting':pf.__version__,
        'CUDA':torch.version.cuda,'cuDNN':torch.backends.cudnn.version(),
        'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'CPU_threads':torch.get_num_threads(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,
        'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}

def load_protocol():
    if file_hash(PROTOCOL)!=SHA.read_text(encoding='utf-8').split()[0]:raise RuntimeError('H_PROTOCOL_HASH_MISMATCH')
    cfg=yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    if identity()!=cfg['code_identity']:raise RuntimeError('H_EXECUTION_IDENTITY_CHANGED')
    return cfg

def configure_normal(cfg):
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED_NO_DEVICE_SUBSTITUTION')
    p=cfg['normal_runtime'];torch.set_num_threads(p['CPU_threads'])
    torch.use_deterministic_algorithms(False,warn_only=False)
    torch.backends.cudnn.deterministic=p['cudnn_deterministic'];torch.backends.cudnn.benchmark=p['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=p['TF32_matmul'];torch.backends.cudnn.allow_tf32=p['TF32_cudnn']
    observed=environment()
    for k,v in p.items():
        if k!='claim' and observed[k]!=v:raise RuntimeError('NORMAL_ENVIRONMENT_MISMATCH:'+k)
    return observed

def model(group,p,device):
    classes={'F0':CanonicalStrictHistoryTFT,'F1':DualPathStrictHistoryTFT,'F2':RegimeAwareDualPathTFT}
    m=classes[group](hidden_size=p['hidden_size'],hidden_continuous_size=p['hidden_continuous_size'],attention_head_size=4,lstm_layers=1,dropout=p['dropout']).to(device)
    if sum(t.numel() for t in m.parameters() if t.requires_grad)!=COUNTS[group]:raise RuntimeError('MODEL_PARAMETER_COUNT_CHANGED')
    return m

def prepare(cfg):
    ecfg,_=e.load_protocol();dev=e.c.load_development(ecfg)
    train=e.c.build_windows(dev,ecfg,0,2103);val=e.c.build_windows(dev,ecfg,2103,2404)
    if (len(dev),len(train.origins),len(val.origins))!=(3005,1380,205):raise RuntimeError('H_DATA_COUNT_MISMATCH')
    if set(train.targets.ravel())&set(val.targets.ravel()):raise RuntimeError('TARGET_ROW_OVERLAP')
    top10=cfg['preprocessing']['Top10_indices']
    if top10!=[0,3,13,11,4,9,6,7,10,8]:raise RuntimeError('FROZEN_FEATURE_INDICES_CHANGED')
    # Consume frozen Top10; do not rerun a feature-selection experiment.
    scaler=e.CommonScaler().fit(train.x);old=read_json(E/'outputs/E_COMMON_SCALER.json')
    if not np.array_equal(scaler.mean,np.array(old['mean'])) or not np.array_equal(scaler.scale,np.array(old['scale'])):raise RuntimeError('SCALER_AUTHORITY_MISMATCH')
    tx,vx=e.prep(train.x,scaler,top10,'EWMA'),e.prep(val.x,scaler,top10,'EWMA')
    fp={'train_raw_x_sha256':array_hash(train.x),'train_preprocessed_x_sha256':array_hash(tx),
        'validation_x_sha256':array_hash(vx),'train_y_sha256':array_hash(train.y),
        'validation_y_sha256':array_hash(val.y),'scaler_sha256':array_hash(np.stack([scaler.mean,scaler.scale])),
        'train_origins_sha256':array_hash(train.origins),'validation_origins_sha256':array_hash(val.origins),
        'train_target_rows_sha256':array_hash(train.targets),'validation_target_rows_sha256':array_hash(val.targets)}
    oldf=read_json(F/'outputs/audits/F_DATA_ACCESS_AUDIT.json')
    for k in ['train_raw_x_sha256','train_preprocessed_x_sha256','validation_x_sha256','train_y_sha256','scaler_sha256']:
        if fp[k]!=oldf[k]:raise RuntimeError('HISTORICAL_INPUT_HASH_MISMATCH:'+k)
    fp.update(read_nrows=3007,development_rows=3005,train_windows=1380,validation_windows=205,Top10_indices=top10,
        target_scale_V=scaler.target_scale,assessment_windows_constructed=False,new_assessment_inference=False,test_accessed=False,
        full_N1_read_or_hash=False,development_prefix_fingerprint_sha256=hashlib.sha256(dev.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()).hexdigest())
    return train,val,scaler,tx,vx,fp

def repeat_gate(values):
    if len(values) not in (2,3) or not np.isfinite(values).all():raise RuntimeError('REPEAT_METRICS_INVALID')
    mean=float(np.mean(values));d=float(max(values)-min(values));t=max(.5,.10*mean)
    return {'RMSE_values_mV':list(values),'absolute_spread_mV':d,'relative_spread':d/mean,'threshold_mV':t,'pass':d<=t}

"""Shared read-only authority and query definitions; no training on import."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
STUDY = Path(__file__).resolve().parents[1]
OLD = STUDY.parent / 'RealHyFC_Raw1s_Dynamic_Study'
REPO = STUDY.parent.parent
OUT = STUDY / 'outputs'
PARTS = ['TRAIN','VALIDATION','HISTORICAL_ASSESSMENT','LATE_LIFE']
DTYPES = {'x':'float32','time_ms':'int64','age_h':'float64','file_id':'uint16',
          'source_row':'int32','segment':'int32','valid':'uint8','voltage_V':'float64'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,obj):
    p=STUDY/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def load_cfg():return json.loads((OUT/'frozen_protocol.json').read_text(encoding='utf-8'))

def load_cache():
    schema=json.loads((OLD/'outputs/cache_schema.json').read_text(encoding='utf-8'))
    n=schema['rows']
    return schema,{k:np.memmap(OLD/'cache'/f'{k}.bin',mode='r',dtype=d,
        shape=(n,24) if k=='x' else (n,)) for k,d in DTYPES.items()}

def model_module():
    key='frozen_raw1s_native_adapter'
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,OLD/'scripts/models.py')
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
    return sys.modules[key]

def model_factory(kind,h):return model_module().create_model(kind,24,300,h)

def query_truth(a,origins,h,cfg):
    targets=origins[:,None]+np.arange(1,h+1,dtype=np.int64)
    v=a['voltage_V'];truth=np.asarray(v[targets]-v[origins,None])
    dv=np.diff(np.column_stack([v[origins],v[targets]]),axis=1)
    event=np.maximum.accumulate(np.abs(dv)>cfg['voltage_event']['threshold_V'],axis=1)
    return targets,truth,event

def metrics(y,p):
    y=np.asarray(y,dtype=np.float64).ravel();p=np.asarray(p,dtype=np.float64).ravel()
    if not len(y):return dict(queries=0,RMSE_V=None,MAE_V=None,R2_delta=None,P95_abs_V=None,P99_abs_V=None)
    e=p-y;sse=float(np.dot(e,e));den=float(np.sum((y-y.mean())**2));ae=np.abs(e)
    return dict(queries=len(y),RMSE_V=math.sqrt(sse/len(y)),MAE_V=float(ae.mean()),
        R2_delta=1-sse/den if den>0 else None,P95_abs_V=float(np.quantile(ae,.95)),P99_abs_V=float(np.quantile(ae,.99)))

def latest_query_indices(targets):
    """One actual query per target: latest origin (=shortest lead), never an ensemble."""
    flat=np.asarray(targets).ravel()
    _,reverse_index=np.unique(flat[::-1],return_index=True)
    return len(flat)-1-reverse_index

def metric_rows(model,h,grid,origins,targets,truth,pred,event):
    rows=[]
    for scope,selection in [('ALL_QUERIES',np.arange(truth.size)),('UNIQUE_TARGET_LATEST_ORIGIN',latest_query_indices(targets))]:
        y=truth.ravel()[selection];p=pred.ravel()[selection];ev=event.ravel()[selection]
        for stratum,mask in [('ALL',np.ones(len(y),bool)),('EVENT',ev),('NORMAL',~ev)]:
            rows.append(dict(model=model,horizon_seconds=h,partition='VALIDATION',grid=grid,scope=scope,
                stratum=stratum,origins=len(origins),unique_target_seconds=len(np.unique(targets)),**metrics(y[mask],p[mask])))
    return rows

def assert_integrity(cfg=None,raw=False):
    cfg=load_cfg() if cfg is None else cfg
    for name,expected in cfg['historical_artifact_sha256'].items():
        assert sha(OLD/name)==expected,('Historical artifact changed',name)
    assert sha(STUDY.parent/'RealHyFC_F2_Main.py')==cfg['historical_code_sha256']
    for name,expected in cfg.get('frozen_file_sha256',{}).items():assert sha(STUDY/name)==expected,name
    if raw:
        schema,_=load_cache()
        for f in schema['source_files']:
            p=REPO/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['file']
    return True

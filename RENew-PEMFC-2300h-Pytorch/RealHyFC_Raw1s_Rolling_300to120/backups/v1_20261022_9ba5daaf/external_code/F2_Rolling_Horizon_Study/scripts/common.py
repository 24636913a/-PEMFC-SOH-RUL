"""Side-effect-free paths, IO, metric formulas and history protection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math
import subprocess
import time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
REPO=ROOT.parent
SOURCE=ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
REFERENCE=ROOT/'REAL_Paper_File/F2主程序/F2_Training_Outputs/20261019T055734_218949Z_seedall_compare/seed202/top10'
CONFIG=HERE/'configs/FROZEN_PROTOCOL.json'

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.writing')
    tmp.write_text(text,encoding='utf-8')
    for i in range(8):
        try:tmp.replace(path);return
        except PermissionError:
            if i==7:raise
            time.sleep(.1*(i+1))
def dump(path,value):atomic(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def csv(path,frame):
    if not isinstance(frame,pd.DataFrame):frame=pd.DataFrame(frame)
    atomic(path,frame.to_csv(index=False,float_format='%.17g',lineterminator='\n'))
def read_csv(path):return pd.read_csv(path,float_precision='round_trip')
def metrics(points):
    y=points.y_true_delta_V.to_numpy(float);err=points.y_pred_delta_V.to_numpy(float)-y
    n=len(y);sse=math.fsum(float(x)**2 for x in err);mean=math.fsum(map(float,y))/n
    sst=math.fsum((float(x)-mean)**2 for x in y)
    target=pd.DataFrame(dict(target=points.target_row,error2=err**2)).groupby('target',sort=True).error2.mean()
    ab=np.abs(err)*1000;top=max(1,math.ceil(.01*n))
    return dict(RMSE_mV=math.sqrt(sse/n)*1000,MAE_mV=math.fsum(abs(float(x)) for x in err)/n*1000,
        R2=1-sse/sst if sst else None,P95_mV=float(np.quantile(ab,.95)),P99_mV=float(np.quantile(ab,.99)),
        max_abs_mV=float(ab.max()),top1pct_queries=top,top1pct_SSE_fraction=float(np.sum(np.partition(err**2,n-top)[n-top:])/sse) if sse else 0.,
        target_time_equal_RMSE_mV=float(np.sqrt(target.mean())*1000),queries=n,origins=int(points.origin_row.nunique()),unique_targets=int(points.target_row.nunique()))
def history_snapshot(out):
    paths=subprocess.check_output(['git','ls-files','-z','--','RENew-PEMFC-2300h-Pytorch'],cwd=REPO).decode().split('\0')
    files={};missing=[]
    for p in paths:
        if not p or '/F2_Rolling_Horizon_Study/' in p:continue
        if not (REPO/p).is_file():missing.append(p);continue
        files[p]=sha(REPO/p)
        if len(files)%10000==0:print('Historical protection snapshot',len(files),flush=True)
    dump(out/'HISTORY_BEFORE.json',files);dump(out/'PREEXISTING_MISSING_FILES.json',missing)
    return len(files)

def verify_history(out):
    before=read(out/'HISTORY_BEFORE.json');changed=[]
    for i,(p,expected) in enumerate(before.items(),1):
        if not (REPO/p).is_file() or sha(REPO/p)!=expected:changed.append(p)
        if i%10000==0:print('Historical protection recheck',i,flush=True)
    missing=read(out/'PREEXISTING_MISSING_FILES.json')
    restored=[p for p in missing if (REPO/p).is_file()]
    dump(out/'HISTORY_PROTECTION_FINAL.json',dict(PASS=not changed and not restored,existing_files_checked=len(before),changed=changed,preexisting_missing=len(missing),restored=restored))
    assert not changed and not restored
    return len(before)

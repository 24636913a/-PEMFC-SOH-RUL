"""Local, inference-only audit utilities. No historical output writer is invoked."""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]; ROOT=P.parent; REPO=ROOT.parent
R2=ROOT/'Phase_SCI_F2_Seed_Screening_Round2'
H=ROOT/'Phase_H_Determinism_MultiSeed_Stability'
J=ROOT/'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit'
SEEDS=[42,103,193,202,210,212]
BASE='dae9331ec660b3e91e23c2064d9c547b26b46393'
FEATURES=['load_transition_count','zero_current_fraction','actual_current_density_A_cm2_std','actual_current_A_std','actual_current_density_A_cm2_mean']
INDICES=[11,13,9,6,8]
def sha(p):
    p=Path(p); assert p.name!='causal_features_30min.csv','FULL_RAW_READ_FORBIDDEN'
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rd(p): return pd.read_csv(p,float_precision='round_trip')
def dest(p):
    p=(P/p).resolve(); assert p.is_relative_to(P),'HISTORICAL_WRITE_FORBIDDEN'
    p.parent.mkdir(parents=True,exist_ok=True); return p
def js(p,o): dest(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def cs(p,o): pd.DataFrame(o).to_csv(dest(p),index=False,float_format='%.17g',lineterminator='\n')
def txt(p,s): dest(p).write_text(s,encoding='utf-8')
def cfg():
    q=P/'configs/frozen_protocol.json'; assert sha(q)==(q.with_suffix('.sha256')).read_text().strip()
    c=read(q)
    for k,v in c['input_sha256'].items():
        assert sha(REPO/k)==v,('SOURCE_CHANGED',k)
    return c
def metrics(f):
    e=(f.y_pred_delta_V-f.y_true_delta_V).to_numpy()*1000.; a=np.abs(e)
    y=f.y_true_delta_V.to_numpy()*1000.; z=e[f.horizon_step.to_numpy()==24]
    t=pd.DataFrame({'target':f.target_row,'e2':e*e}).groupby('target').e2.mean().mean()
    return dict(n=len(f),origins=int(f.origin_row.nunique()),target_times=int(f.target_row.nunique()),
        RMSE_mV=float(np.sqrt(np.mean(e*e))),MAE_mV=float(a.mean()),R2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),
        endpoint_RMSE_mV=float(np.sqrt(np.mean(z*z))),target_time_equal_RMSE_mV=float(np.sqrt(t)),
        P95_mV=float(np.percentile(a,95)),P99_mV=float(np.percentile(a,99)),max_abs_error_mV=float(a.max()))
def derangements(n,seeds):
    rows=[]
    for s in seeds:
        rng=np.random.default_rng(s)
        while True:
            p=rng.permutation(n)
            if np.all(p!=np.arange(n)): break
        rows.extend(dict(permutation_seed=s,window=i,source_window=int(v)) for i,v in enumerate(p))
    return pd.DataFrame(rows)
def historical_inventory():
    """Hash Git-tracked checkpoints and selected Development screening files only."""
    blocks=subprocess.check_output(['git','ls-tree','-r','-z',BASE],cwd=REPO).split(b'\0'); rows=[]
    for b in blocks:
        if not b:continue
        head,raw=b.split(b'\t',1); name=raw.decode(); path=REPO/name
        weight=path.suffix.lower() in ['.pt','.pth','.ckpt']
        screen=any('/'+x+'/' in '/'+name for x in ['Phase_SCI_F2_Overnight_Seed_Screening','Phase_SCI_F2_Seed_Screening_Round2'])
        if not (weight or screen):continue
        data=path.read_bytes(); blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert blob==head.decode().split()[-1],('HISTORY_DIFFERS_FROM_BASE',name)
        rows.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),checkpoint=weight))
    return pd.DataFrame(rows)
def get_h():
    sys.path.insert(0,str(H/'scripts')); import h_common as h
    return h

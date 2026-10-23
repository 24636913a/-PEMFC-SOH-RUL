"""Isolated exploratory evaluation: writes are restricted to this new phase."""
from pathlib import Path
import sys,os,json,hashlib,time,subprocess
import numpy as np
import pandas as pd
import torch,yaml
P=Path(__file__).resolve().parents[1]; ROOT=P.parent; REPO=ROOT.parent
H=ROOT/'Phase_H_Determinism_MultiSeed_Stability'
B=ROOT/'Phase_SCI_F2_LSTM_TCN_Fair_Baselines'
R2=ROOT/'Phase_SCI_F2_Seed_Screening_Round2'
SOURCE=ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
MODELS=['F0','F1','F2','LSTM','TCN']; SEEDS=[42,52,62]
TOP10=[0,3,13,11,4,9,6,7,10,8]
FEATURES=['cell_voltage_mean','cell_voltage_std','cell_voltage_range','cell_voltage_min','cell_voltage_max','actual_current_A_mean','actual_current_A_std','actual_current_A_peak_to_peak','actual_current_density_A_cm2_mean','actual_current_density_A_cm2_std','actual_current_density_A_cm2_peak_to_peak','load_transition_count','operation_point_transitions','zero_current_fraction']
PARTS=['TRAIN','VALIDATION','HISTORICAL_ASSESSMENT','LATE_LIFE']

def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
def path(n):
    f=(P/n).resolve();assert f.is_relative_to(P),'HISTORY_WRITE_FORBIDDEN';f.parent.mkdir(parents=True,exist_ok=True);return f
def write(n,s):
    f=path(n);t=f.with_suffix(f.suffix+'.writing')
    with t.open('w',encoding='utf-8',newline='\n') as q:q.write(s);q.flush();os.fsync(q.fileno())
    for delay in [.02,.05,.1,.2,.5,1,None]:
        try:os.replace(t,f);return
        except PermissionError:
            if delay is None:raise
            time.sleep(delay)
def js(n,o):write(n,json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))
def rd(f):return pd.read_csv(f,float_precision='round_trip')
def cs(n,o):write(n,pd.DataFrame(o).to_csv(index=False,float_format='%.17g',lineterminator='\n'))
def event(kind,**kw):
    f=path('outputs/execution_ledger/events.jsonl')
    with f.open('a',encoding='utf-8',newline='\n') as q:q.write(json.dumps(dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw),ensure_ascii=False)+'\n');q.flush();os.fsync(q.fileno())
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO).decode('utf-8').strip()
def data(part):return np.load(P/f'outputs/audits/{part}_WINDOWS.npz',allow_pickle=False)
def inventory():return rd(P/'outputs/audits/CHECKPOINT_INVENTORY.csv')
def protocol():
    f=P/'configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml';c=yaml.safe_load(f.read_text(encoding='utf-8'))
    assert sha(f)==f.with_suffix('.sha256').read_text().strip()
    for k,v in c['frozen_sha256'].items():assert sha(REPO/k)==v,('FROZEN_IDENTITY_CHANGED',k)
    return c
def snapshot_history():
    """Protect every tracked pre-existing file, including Phase L and once-only ledgers."""
    names=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode('utf-8').split('\0')
    rows=[]
    for n in names:
        if n and (REPO/n).is_file() and not (REPO/n).resolve().is_relative_to(P):rows.append(dict(path=n,sha256=sha(REPO/n)))
    cs('outputs/audits/HISTORY_BEFORE.csv',rows);return len(rows)
def protect():
    rows=rd(P/'outputs/audits/HISTORY_BEFORE.csv')
    for r in rows.itertuples():assert sha(REPO/r.path)==r.sha256,('HISTORY_CHANGED',r.path)
    return len(rows)

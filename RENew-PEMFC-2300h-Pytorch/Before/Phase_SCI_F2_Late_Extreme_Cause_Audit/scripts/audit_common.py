"""New diagnostic writes only. All scientific results are exploratory, no fitting."""
from pathlib import Path
import sys,os,json,hashlib,time,subprocess
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
FULL=ROOT/'Phase_SCI_RealHyFC_Full_Chronological_Evaluation'
MODELS=['F0','F1','F2','LSTM','TCN'];SEEDS=[42,52,62]
EVENTS={'E1':(3112,3113),'E2':(3627,3628)}
PARTS=['TRAIN','VALIDATION','HISTORICAL_ASSESSMENT','LATE_LIFE']
def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as s:
        for b in iter(lambda:s.read(1048576),b''):h.update(b)
    return h.hexdigest()
def ah(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
def path(n):
    f=(P/n).resolve();assert f.is_relative_to(P);f.parent.mkdir(parents=True,exist_ok=True);return f
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
def data(part):return np.load(FULL/f'outputs/audits/{part}_WINDOWS.npz',allow_pickle=False)
def raw():return rd(FULL/'outputs/audits/FULL_ROW_IDENTITY.csv')
def pred(model,seed,part='LATE_LIFE'):return rd(FULL/f'outputs/predictions/{part}_{model}_seed{seed}.csv')
def check_protocol():
    p=P/'configs/DIAGNOSTIC_PROTOCOL.json';c=read(p);assert sha(p)==p.with_suffix('.sha256').read_text().strip()
    for f,h in c['source_sha256'].items():assert sha(REPO/f)==h,('SOURCE_IDENTITY_CHANGED',f)
    return c
def prepare():
    assert not (P/'configs/DIAGNOSTIC_PROTOCOL.json').exists(),'NO_REFREEZE'
    f=raw();assert len(f)==3843 and np.array_equal(f.row_id,np.arange(3843))
    for rows,ages in [((3112,3113),(1873.5,1874.)),((3627,3628),(2149.5,2150.))]:assert np.array_equal(f.loc[list(rows),'bin_end_ophrs2_h'],ages)
    files=[FULL/'outputs/audits/FULL_ROW_IDENTITY.csv',FULL/'outputs/audits/CHECKPOINT_INVENTORY.csv',ROOT/'RealHyFC_F2_Main.py']
    files+=list((FULL/'outputs/audits').glob('*_WINDOWS.npz'))
    for m in MODELS:
        for s in SEEDS:files.append(FULL/f'outputs/predictions/LATE_LIFE_{m}_seed{s}.csv')
    files+=list((P/'scripts').glob('*.py'))
    js('configs/DIAGNOSTIC_PROTOCOL.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),status='EXPLORATORY_DIAGNOSTIC_AFTER_EVENTS_WERE_KNOWN_NOT_PREREGISTERED_CONFIRMATION',events=EVENTS,event_context='24 records before first core row and 24 after second; 50 per event',SSE='all original queries; also sum of mean SE per unique target',pair_inversion='predicted first voltage < predicted second for same origin, while actual first > second; no threshold',timing='for each origin covering first-2..second+2, argmin predicted minus argmin actual row; no shifted predictions used for formal scoring',analog='majority zero-current >0.5, powered flanks zero_current_fraction=0; physics definition only, no error selection',nearest_history='RMS standardized masked historical 48x14 tensor; five closest legal Train histories, no future labels used in distance',gate='observe only original forward hooks; no gate replacement, no intervention, no gradients',source_sha256={f.relative_to(REPO).as_posix():sha(f) for f in files}))
    write('configs/DIAGNOSTIC_PROTOCOL.sha256',sha(P/'configs/DIAGNOSTIC_PROTOCOL.json')+'\n')

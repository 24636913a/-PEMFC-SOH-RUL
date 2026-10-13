"""P3-only authority, bounded Development access, original H preprocessing."""
from __future__ import annotations
import ast
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import yaml

PHASE=Path(__file__).resolve().parents[1];PROJECT=PHASE.parent;REPO=PROJECT.parent
BASE='c9b7ecce5e76c9fcdcc74dac414ea0c7ad1d85c0'
P1=PROJECT/'Phase_P1_Gate_Ablation_Engineering_Preflight'
H=PROJECT/'Phase_H_Determinism_MultiSeed_Stability'
F=PROJECT/'Phase_F_TFT_DualPath_RegimeFusion'
E=PROJECT/'Phase_TFT_FullFeature_CausalFilter_Comparison'
C=PROJECT/'Phase_TFT_SingleSplit_Feature_Filter_Ablation'
N1=PROJECT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
OUT=PHASE/'outputs';PROTOCOL=PHASE/'configs/P3_FROZEN_FULL450_PROTOCOL.json'
RUNS=[(f'P3-{g}-Seed{s}',g,s) for s,order in ((42,('G0','G1','G2')),(52,('G1','G2','G0')),(62,('G2','G0','G1'))) for g in order]
STATIC=[P1/'scripts/gate_ablation_models.py',P1/'scripts/p1_checks.py',
        P1/'protocol/P1_EXECUTION_PROTOCOL_DRAFT.json',P1/'protocol/P1_EXECUTION_PROTOCOL_DRAFT.sha256',
        P1/'outputs/P1_EXECUTION_AUDIT.json',P1/'outputs/P1_SYNTHETIC_MODEL_EVIDENCE.json',
        H/'configs/H_FROZEN_PROTOCOL.yaml',H/'configs/H_FROZEN_PROTOCOL.sha256',
        H/'scripts/h_common.py',H/'scripts/run_h_training.py',H/'scripts/crash_safe_recorder.py',
        E/'scripts/run_e_comparison.py',E/'configs/PREREGISTERED_E_COMPARISON.yaml',
        E/'configs/PREREGISTERED_E_COMPARISON.sha256',E/'outputs/E_COMMON_SCALER.json',
        C/'scripts/run_c_ablation.py',F/'outputs/audits/F_DATA_ACCESS_AUDIT.json',
        PROJECT/'Phase_TFT_Original_Baseline/models/canonical_tft.py',
        PROJECT/'Phase_P_Final_Innovation_and_Experiment_Planning/protocol/P_PROPOSED_GATE_PROTOCOL.json']
STATIC+=sorted((F/'models').glob('*.py'))
P2=PROJECT/'Phase_P2_Final_Gate_Ablation_Training'
STATIC+=[P2/'configs/P2_FROZEN_EXECUTION_PROTOCOL.json',P2/'configs/P2_FROZEN_EXECUTION_PROTOCOL.sha256']
STATIC+=[PROJECT/'Phase_O1_Nine_Model_Validation_Robustness_Audit/reports/O1_FINAL_RESEARCH_DECISION.md']
for run_id,_,_ in RUNS:
    STATIC += [P2/'outputs/training_logs'/run_id.replace('P3-','P2-')/name for name in ('initial.json','result.json','epochs.json')]
ALLOW={str(p.resolve()).casefold() for p in STATIC}
OPEN_EVENTS=[];PREFIX_READS=[];_prefix_open=False;_installed=False

def utc():return datetime.now(timezone.utc).isoformat()
def sha_bytes(payload):return hashlib.sha256(payload).hexdigest()
def relative(path):return str(Path(path).resolve().relative_to(REPO)).replace('\\','/')

def permitted_open(path,writing=False):
    if isinstance(path,int):return
    p=Path(os.fsdecode(path)).resolve()
    if p.is_relative_to(PHASE):return
    if p==N1.resolve():
        if writing or not _prefix_open:raise PermissionError('N1_BOUNDED_READER_ONLY')
        return
    if p.is_relative_to(REPO):
        if writing or str(p).casefold() not in ALLOW:raise PermissionError('P3_REPO_ACCESS_DENIED:'+str(p))
    elif '数据' in p.parts or p.name.startswith(('Loadcycle_','Polcurve_')) or p.suffix.lower() in ('.pt','.npz','.parquet','.csv'):
        raise PermissionError('P3_EXTERNAL_REAL_DATA_DENIED:'+str(p))

def validate_read_csv_request(path,kwargs):
    p=Path(path).resolve()
    if p==N1.resolve():
        if kwargs.get('nrows')!=3007 or kwargs.get('chunksize') is not None or kwargs.get('iterator'):
            raise PermissionError('N1_READ_CONTRACT_REQUIRES_NROWS_3007')
    elif not p.is_relative_to(PHASE):
        raise PermissionError('NON_P3_CSV_ACCESS_DENIED:'+str(p))

def install_firewall():
    global _installed
    if _installed:return
    def hook(event,args):
        if event!='open' or isinstance(args[0],int):return
        path,mode,flags=args
        writing=bool((isinstance(mode,str) and any(x in mode for x in ('w','a','x','+'))) or
                     (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        try:permitted_open(path,writing)
        except PermissionError:
            OPEN_EVENTS.append({'path':os.fsdecode(path),'writing':writing,'action':'DENIED'});raise
        if str(REPO).casefold() in os.fsdecode(path).casefold():
            OPEN_EVENTS.append({'path':os.fsdecode(path),'writing':writing,'action':'ALLOWED'})
    sys.addaudithook(hook)
    original=pd.read_csv
    def bounded(path,*args,**kwargs):
        global _prefix_open
        validate_read_csv_request(path,kwargs)
        is_n1=Path(path).resolve()==N1.resolve()
        previous=_prefix_open;_prefix_open=is_n1
        try:
            result=original(path,*args,**kwargs)
            if is_n1:PREFIX_READS.append({'nrows':3007,'rows_read':len(result),'purpose':'prescribed Development prefix only'})
            return result
        finally:_prefix_open=previous
    pd.read_csv=bounded;_installed=True

def file_hash(path):
    p=Path(path).resolve()
    if p==N1.resolve():raise PermissionError('FULL_N1_HASH_PROHIBITED')
    permitted_open(p)
    h=hashlib.sha256()
    with p.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def atomic_json(path,payload):
    path=Path(path).resolve()
    if not path.is_relative_to(PHASE):raise PermissionError('P3_OUTPUT_ONLY')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.writing')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)

def verify_static_source(observed,git_blob,name,registered_sha=None):
    # Git text=auto stores LF while H froze the original checkout file bytes.
    # Only CRLF/LF representation is canonicalized for the separate Git check.
    if observed.replace(b'\r\n',b'\n')!=git_blob.replace(b'\r\n',b'\n'):
        raise RuntimeError('FROZEN_SOURCE_CHANGED:'+name)
    if registered_sha is not None and sha_bytes(observed)!=registered_sha:
        raise RuntimeError('H_FROZEN_WORKTREE_BYTE_HASH_MISMATCH:'+name)

def execution_identity():
    records={}
    hcfg=yaml.safe_load((H/'configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
    for path in STATIC:
        observed=path.read_bytes();name=relative(path)
        verify_static_source(observed,git('show',BASE+':'+name),name,hcfg['code_identity']['files'].get(name))
        records[name]=sha_bytes(observed)
    for path,expected in hcfg['code_identity']['files'].items():
        if path.startswith('C:'):
            actual=file_hash(path)
            if actual!=expected:raise RuntimeError('UPSTREAM_TFT_SOURCE_MISMATCH:'+path)
            records[path]=actual
    for folder in ('scripts','tests'):
        for path in sorted((PHASE/folder).glob('*.py')):records[relative(path)]=file_hash(path)
    return {'files':records,'sha256':sha_bytes(json.dumps(records,sort_keys=True).encode())}

def load_protocol():
    cfg=read_json(PROTOCOL);digest=file_hash(PROTOCOL)
    if digest!=PROTOCOL.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]:raise RuntimeError('P3_PROTOCOL_HASH_MISMATCH')
    if cfg['authorization']!='USER_EXPLICIT_P3_NINE_RUN_TRAINING' or cfg['maximum_runs']!=9:raise RuntimeError('P3_AUTHORIZATION_MISMATCH')
    if cfg['code_identity']!=execution_identity():raise RuntimeError('P3_EXECUTION_CODE_IDENTITY_CHANGED')
    return cfg,digest

def legacy_tools():
    sys.path.insert(0,str(P1/'scripts'))
    a=importlib.import_module('gate_ablation_models')
    sys.path.insert(0,str(E/'scripts'))
    e=importlib.import_module('run_e_comparison')
    array_hash,_=a.extract_function(H/'scripts/h_common.py','array_hash',{'hashlib':hashlib,'np':np})
    return a,e,array_hash

def validate_windows(train,val):
    if (len(train.origins),len(val.origins))!=(1380,205):raise RuntimeError('WINDOW_COUNT_MISMATCH')
    if set(train.targets.ravel())&set(val.targets.ravel()):raise RuntimeError('TARGET_ROW_OVERLAP')
    if not np.array_equal(val.targets,val.origins[:,None]+np.arange(1,25)):raise RuntimeError('VALIDATION_TARGET_KEY_MISMATCH')
    if not {2247,2248}.issubset(set(val.targets.ravel())):raise RuntimeError('REGISTERED_EXTREME_TARGETS_MISSING')

def prepare(cfg):
    a,e,array_hash=legacy_tools()
    original,_=a.extract_function(H/'scripts/h_common.py','prepare',
        {'e':e,'np':np,'array_hash':array_hash,'read_json':read_json,'E':E,'F':F,'hashlib':hashlib})
    train,val,scaler,tx,vx,fp=original(cfg['H'])
    validate_windows(train,val)
    fp.update(real_source=relative(N1),Assessment_windows_constructed=False,Test_accessed=False,
              allowed_prefix_reads=list(PREFIX_READS),full_N1_hash_performed=False)
    return a,e,array_hash,train,val,scaler,tx,vx,fp

def validate_common_sampler(seed,epoch,order):
    path=OUT/'audits/sampler_authority'/f'seed{seed}_epoch{epoch:03d}.json'
    record={'seed':seed,'epoch':epoch,'origin_order':order,'origin_order_sha256':sha_bytes(np.asarray(order,dtype=np.int64).tobytes())}
    if path.exists():
        if read_json(path)!=record:raise RuntimeError('COMMON_EPOCH_SAMPLER_ORDER_MISMATCH')
    else:atomic_json(path,record)
    return record['origin_order_sha256']

def checkpoint_write(path,payload):
    path=Path(path).resolve()
    if not path.is_relative_to((OUT/'checkpoints').resolve()):raise PermissionError('P3_CHECKPOINT_ONLY')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pt.writing')
    with tmp.open('wb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def require_checkpoint_identity(ck,run_id,group,seed,protocol_sha,code_sha,fp):
    expected={'run_id':run_id,'model':group,'seed':seed,'protocol_sha256':protocol_sha,'code_identity_sha256':code_sha,'input_fingerprints':fp}
    for key,value in expected.items():
        if ck.get(key)!=value:raise RuntimeError('P3_CHECKPOINT_IDENTITY_MISMATCH:'+key)

def runtime_access_audit():
    return {'prefix_reads':PREFIX_READS,'full_N1_hash':False,'Assessment_accessed':False,'Test_accessed':False,
            'historic_checkpoint_opened':False,'open_events':OPEN_EVENTS}
def verify_p2_initialization(group,seed,initial,rng,identity):
    expected=read_json(P2/'outputs/training_logs'/f'P2-{group}-Seed{seed}'/'initial.json')
    if (initial,rng,identity)!=(expected['initial_model_sha256'],expected['initial_training_rng'],expected['parameter_identity']):
        raise RuntimeError('P2_INITIAL_WEIGHT_OR_RNG_IDENTITY_MISMATCH')

def first_hypothetical_stop(previous,epoch,stale):
    return previous if previous is not None else (epoch if stale>=40 else None)

def verified_backup(source):
    import shutil
    source=Path(source).resolve()
    if not source.is_relative_to(OUT/'checkpoints'):raise PermissionError('ONLY_NEW_P3_CHECKPOINT_BACKUP')
    dest=PHASE/'backups/checkpoints'/source.name
    if dest.exists():raise RuntimeError('BACKUP_ALREADY_EXISTS_NO_OVERWRITE')
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,dest)
    digest=file_hash(source)
    verify_backup_hash(digest,file_hash(dest))
    return {'path':str(dest.relative_to(PHASE)).replace('\\\\','/'),'sha256':digest,'verified':True,
            'storage_scope':'INDEPENDENT_DIRECTORY_SAME_E_VOLUME_NOT_OFF_DEVICE'}
def verify_backup_hash(expected,observed):
    if expected!=observed:raise RuntimeError('LOCAL_BACKUP_HASH_MISMATCH')


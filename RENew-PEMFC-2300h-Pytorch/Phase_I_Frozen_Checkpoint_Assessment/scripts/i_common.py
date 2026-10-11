"""Phase I read-only dependencies; no scaler fit, training, or target selection."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml

PHASE=Path(__file__).resolve().parents[1]; PROJECT=PHASE.parent; REPO=PROJECT.parent
BASE='1b83d05b94a4952d0a41df00142db710278830b4'
H=PROJECT/'Phase_H_Determinism_MultiSeed_Stability'
sys.path.insert(0,str(H/'scripts')); import h_common as h
e=h.e; c=e.c
OUT=PHASE/'outputs'; PROTOCOL=PHASE/'configs/I_ASSESSMENT_PROTOCOL.yaml'; SHA=PROTOCOL.with_suffix('.sha256')
LEDGER=OUT/'execution_ledger/I_EXECUTION_LEDGER.json'
RUNS=[{'run_id':f'I_{g}_Seed{s}','model':g,'seed':s,'H_run_id':f'H-{g}-Seed{s}'} for g in ['F0','F1','F2'] for s in [42,52,62]]
KEYS=['origin_row','target_row','horizon_step']
F_PRED=[h.F/'outputs/predictions'/f'{g}_assessment_seed42.csv' for g in ['F0','F1','F2']]
H_FILES=[H/'configs/H_FROZEN_PROTOCOL.yaml',H/'configs/H_FROZEN_PROTOCOL.sha256',
 H/'outputs/metrics/H_VALIDATION_METRICS.csv',H/'outputs/audits/H_FINAL_ARTIFACT_INTEGRITY_AUDIT.json',
 H/'outputs/audits/H_INITIALIZATION_AUTHORITY.json',H/'outputs/audits/H_EXECUTION_LEDGER.json']
E_FILES=[e.PROTOCOL,e.PROTOCOL_SHA,e.PHASE/'outputs/E_COMMON_SCALER.json']
READ_ALLOWLIST=H_FILES+E_FILES+F_PRED+list((H/'scripts').glob('*.py'))+h.HISTORY+h.upstream_files()
for r in RUNS:
 READ_ALLOWLIST.extend([H/'outputs/checkpoints'/f"{r['H_run_id']}_best.pt",H/'outputs/training_logs'/r['H_run_id']/'result.json',H/'outputs/training_logs'/r['H_run_id']/'input_audit.json'])

def check_read(path, *, prefix=False):
 p=Path(path).resolve()
 if p==c.N1.resolve():
  if not prefix: raise RuntimeError('FULL_N1_READ_OR_HASH_PROHIBITED')
  return p
 if p.is_relative_to(PHASE) or p in [x.resolve() for x in READ_ALLOWLIST]: return p
 raise RuntimeError('SOURCE_NOT_ALLOWLISTED:'+str(p))

def digest(path): return hashlib.sha256(check_read(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(check_read(path).read_text(encoding='utf-8'))
def atomic_json(path,value):
 p=Path(path).resolve()
 if not p.is_relative_to(PHASE): raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.writing')
 with t.open('w',encoding='utf-8',newline='\n') as f:
  f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)

def csv_write(path,frame):
 p=Path(path).resolve()
 if not p.is_relative_to(PHASE): raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
 p.parent.mkdir(parents=True,exist_ok=True)
 frame.to_csv(p,index=False,float_format='%.17g',lineterminator='\n')

def code_identity():
 paths=sorted((PHASE/'scripts').glob('*.py'))+sorted((PHASE/'tests').glob('*.py'))
 records={p.relative_to(REPO).as_posix():digest(p) for p in paths}
 records.update(h.identity()['files'])
 return {'files':records,'sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}

def load_protocol():
 if digest(PROTOCOL)!=SHA.read_text(encoding='utf-8').split()[0]:raise RuntimeError('I_PROTOCOL_HASH_MISMATCH')
 cfg=yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
 if code_identity()!=cfg['code_identity']:raise RuntimeError('I_EXECUTION_CODE_IDENTITY_CHANGED')
 for p,sha in cfg['sources'].items():
  if digest(REPO/p)!=sha:raise RuntimeError('I_FROZEN_SOURCE_CHANGED:'+p)
 h.load_protocol()
 return cfg

def read_dev():
 """Execute the inherited bounded reader, guarding the real pandas call."""
 original=pd.read_csv; access=[]
 def bounded(path,*args,**kw):
  p=Path(path).resolve()
  if p==c.N1.resolve():
   if kw.get('nrows')!=3007:raise RuntimeError('UNBOUNDED_N1_READ')
   access.append({'source':str(p),'nrows':3007})
  else:check_read(p)
  return original(path,*args,**kw)
 with patch.object(pd,'read_csv',bounded):dev=c.load_development({})
 if len(access)!=1 or len(dev)!=3005 or not (dev.bin_end_ophrs2_h<1819.997738).all():raise RuntimeError('I_PREFIX_BOUNDARY_MISMATCH')
 return dev,access

def frozen_scaler():
 saved=read_json(e.PHASE/'outputs/E_COMMON_SCALER.json')
 s=e.CommonScaler();s.mean=np.asarray(saved['mean'],dtype=np.float64);s.scale=np.asarray(saved['scale'],dtype=np.float64);s.samples=saved['flat_samples']
 if s.mean.shape!=(14,) or s.scale.shape!=(14,) or not np.isfinite(s.scale).all() or not (s.scale>0).all():raise RuntimeError('FROZEN_SCALER_INVALID')
 if s.target_scale!=saved['target_scale_V']:raise RuntimeError('TARGET_SCALE_MISMATCH')
 return s

def prepare_data():
 ecfg,_=e.load_protocol();hc=h.load_protocol();dev,access=read_dev();s=frozen_scaler()
 val=c.build_windows(dev,ecfg,2103,2404);assess=c.build_windows(dev,ecfg,2404,3005)
 if (len(val.origins),len(assess.origins))!=(205,378):raise RuntimeError('I_WINDOW_COUNT_MISMATCH')
 if set(val.targets.ravel())&set(assess.targets.ravel()):raise RuntimeError('TARGET_ISOLATION_FAILURE')
 vx=e.prep(val.x,s,hc['preprocessing']['Top10_indices'],'EWMA');ax=e.prep(assess.x,s,hc['preprocessing']['Top10_indices'],'EWMA')
 fp={'assessment_raw_x_sha256':h.array_hash(assess.x),'assessment_input_sha256':h.array_hash(ax),
 'assessment_labels_sha256':h.array_hash(assess.y),'assessment_origins_sha256':h.array_hash(assess.origins),
 'assessment_target_rows_sha256':h.array_hash(assess.targets),'validation_x_sha256':h.array_hash(vx),
 'validation_y_sha256':h.array_hash(val.y),'scaler_sha256':h.array_hash(np.stack([s.mean,s.scale])),
 'development_prefix_fingerprint_sha256':hashlib.sha256(dev.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()).hexdigest(),
 'read_nrows':3007,'development_rows':3005,'assessment_origins':378,'records_per_checkpoint':9072,
 'target_scale_V':s.target_scale,'scaler_refitted':False,'full_N1_hash_performed':False,'formal_test_accessed':False}
 href=read_json(H/'outputs/training_logs/H-F0-Seed42/input_audit.json')
 for key in ['validation_x_sha256','validation_y_sha256','scaler_sha256','development_prefix_fingerprint_sha256']:
  if fp[key]!=href[key]:raise RuntimeError('I_H_DATA_IDENTITY_MISMATCH:'+key)
 origin=np.repeat(assess.origins,24);targets=assess.targets.ravel();steps=np.tile(np.arange(1,25),len(assess.origins))
 truth=pd.DataFrame({'origin_row':origin,'target_row':targets,'horizon_step':steps,
 'origin_age_h':np.repeat(assess.ages,24),'target_age_h':dev.bin_end_ophrs2_h.iloc[targets].to_numpy(),
 'y_true_delta_V':assess.y.ravel().astype(np.float64)})
 if not (targets==origin+steps).all() or not truth.target_row.between(2404,3004).all() or not (truth.target_age_h<1819.997738).all():raise RuntimeError('I_TARGET_BOUNDARY_FAILURE')
 historic=[]
 for p in F_PRED:
  old=pd.read_csv(check_read(p),usecols=KEYS+['y_true_delta_V','target_age_h'],float_precision='round_trip').sort_values(['origin_row','horizon_step']).reset_index(drop=True)
  if len(old)!=9072 or not np.array_equal(old[KEYS].to_numpy(),truth[KEYS].to_numpy()):raise RuntimeError('F_ASSESSMENT_KEY_MISMATCH')
  td=float(np.max(np.abs(old.y_true_delta_V-truth.y_true_delta_V)));ad=float(np.max(np.abs(old.target_age_h-truth.target_age_h)))
  if td>1e-15 or ad>1e-9:raise RuntimeError('F_ASSESSMENT_TRUTH_MISMATCH')
  historic.append({'source':p.relative_to(REPO).as_posix(),'keys_exact':True,'max_truth_difference_V':td,'max_age_difference_h':ad})
 return {'vx':vx,'ax':ax,'truth':truth,'fingerprints':fp,'prefix_access':access,'historical_truth_audit':historic,'scale':s.target_scale}

def configure_environment():
 hc=h.load_protocol(); env=h.configure_normal(hc)
 return env

def verify_checkpoint(spec,expected=None):
 hc=h.load_protocol();path=H/'outputs/checkpoints'/f"{spec['H_run_id']}_best.pt";check_read(path)
 observed=digest(path);record=read_json(H/'outputs/training_logs'/spec['H_run_id']/'result.json')
 integrity=read_json(H/'outputs/audits/H_FINAL_ARTIFACT_INTEGRITY_AUDIT.json')
 item=[r for r in integrity['formal_checkpoints'] if r['run_id']==spec['H_run_id']]
 if len(item)!=1 or not item[0]['pass'] or observed!=item[0]['checkpoint_sha256'] or observed!=record['checkpoint_reload']['file_sha256']:raise RuntimeError('CHECKPOINT_FILE_HASH_MISMATCH')
 ck=torch.load(path,map_location='cpu',weights_only=True)
 metrics=pd.read_csv(check_read(H/'outputs/metrics/H_VALIDATION_METRICS.csv'),float_precision='round_trip')
 row=metrics.loc[metrics.run_id==spec['H_run_id']]
 if len(row)!=1:raise RuntimeError('H_METRIC_IDENTITY_MISSING')
 row=row.iloc[0]
 if row.role!='MULTISEED_PRIMARY' or row.model!=spec['model'] or int(row.seed)!=spec['seed'] or int(row.best_epoch)!=int(record['best_epoch']) or row.status!='COMPLETED':raise RuntimeError('H_METRIC_IDENTITY_MISMATCH')
 fields={'run_id':spec['H_run_id'],'model':spec['model'],'seed':spec['seed'],'protocol_sha256':h.file_hash(h.PROTOCOL),'code_identity_sha256':hc['code_identity']['sha256'],'best_epoch':int(record['best_epoch'])}
 if any(ck[k]!=v for k,v in fields.items()) or record['role']!='MULTISEED_PRIMARY':raise RuntimeError('CHECKPOINT_METADATA_MISMATCH')
 if ck['input_fingerprints']!=read_json(H/'outputs/training_logs'/spec['H_run_id']/'input_audit.json'):raise RuntimeError('CHECKPOINT_INPUT_FINGERPRINT_MISMATCH')
 if ck['target_scale_V']!=frozen_scaler().target_scale or row.protocol_sha256!=ck['protocol_sha256'] or row.code_identity_sha256!=ck['code_identity_sha256']:raise RuntimeError('CHECKPOINT_SCALE_OR_CODE_MISMATCH')
 init=read_json(H/'outputs/audits/H_INITIALIZATION_AUTHORITY.json')['initialization_sha256'][str(spec['seed'])][spec['model']]
 if ck['initialization_sha256']!=init or not record['checkpoint_reload']['pass']:raise RuntimeError('CHECKPOINT_INITIALIZATION_OR_RELOAD_MISMATCH')
 m=h.model(spec['model'],hc['hyperparameters'],torch.device('cpu'));schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in m.state_dict().items()}
 saved_schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in ck['state_dict'].items()}
 if schema!=saved_schema or not all(torch.isfinite(v).all() for v in ck['state_dict'].values()):raise RuntimeError('CHECKPOINT_STATE_SCHEMA_MISMATCH')
 m.load_state_dict(ck['state_dict'],strict=True)
 manifest={**spec,'path':path.relative_to(REPO).as_posix(),'sha256':observed,'best_epoch':int(ck['best_epoch']),
 'parameter_count':h.COUNTS[spec['model']],'state_schema':schema,'H_protocol_sha256':ck['protocol_sha256'],
 'H_code_identity_sha256':ck['code_identity_sha256'],'input_fingerprints':ck['input_fingerprints'],
 'target_scale_V':ck['target_scale_V'],'metadata_pass':True}
 if expected is not None and manifest!=expected:raise RuntimeError('FROZEN_CHECKPOINT_MANIFEST_CHANGED')
 return m,ck,manifest

def predict_frozen(m,x, *, gate=False):
 m.eval()
 for p in m.parameters():p.requires_grad_(False)
 before=e.state_digest(m);values=[];gates=[];handle=None
 if gate:handle=m.regime_gate.register_forward_hook(lambda _m,_a,out:gates.append(out.detach().cpu().numpy().reshape(-1).copy()))
 device=next(m.parameters()).device
 try:
  with torch.inference_mode():
   for first in range(0,len(x),32):
    out=m.predict_delta(torch.from_numpy(x[first:first+32]).to(device))
    if out.shape!=(min(32,len(x)-first),24) or not torch.isfinite(out).all():raise RuntimeError('I_INFERENCE_OUTPUT_INVALID')
    values.append(out.cpu().numpy().copy())
 finally:
  if handle is not None:handle.remove()
 if e.state_digest(m)!=before or any(p.grad is not None or p.requires_grad for p in m.parameters()):raise RuntimeError('I_PARAMETER_UPDATE_DETECTED')
 g=np.concatenate(gates) if gates else None
 if gate and (g is None or len(g)!=len(x) or not np.isfinite(g).all() or not ((g>=0)&(g<=1)).all()):raise RuntimeError('I_GATE_CAPTURE_INVALID')
 return np.concatenate(values),g

def validate_predictions(frame,truth,spec,protocol_sha,checkpoint_sha,input_sha,expected_records=9072):
 if len(frame)!=expected_records or frame.duplicated(KEYS).any():raise RuntimeError('PREDICTION_COUNT_OR_DUPLICATE_KEYS')
 f=frame.sort_values(['origin_row','horizon_step']).reset_index(drop=True);t=truth.sort_values(['origin_row','horizon_step']).reset_index(drop=True)
 for k,v in {'run_id':spec['run_id'],'model':spec['model'],'seed':spec['seed'],'protocol_sha256':protocol_sha,'checkpoint_sha256':checkpoint_sha,'assessment_input_fingerprint':input_sha}.items():
  if not (f[k]==v).all():raise RuntimeError('PREDICTION_IDENTITY_MISMATCH:'+k)
 if not np.array_equal(f[KEYS].to_numpy(),t[KEYS].to_numpy()) or not np.array_equal(f.y_true_delta_V.to_numpy(),t.y_true_delta_V.to_numpy()):raise RuntimeError('PREDICTION_KEYS_OR_TRUTH_MISMATCH')
 if not np.array_equal(f.target_age_h.to_numpy(),t.target_age_h.to_numpy()):raise RuntimeError('PREDICTION_TARGET_AGE_MISMATCH')
 if not f.target_row.between(2404,3004).all() or not (f.target_row==f.origin_row+f.horizon_step).all() or not (f.target_age_h<1819.997738).all():raise RuntimeError('PREDICTION_BOUNDARY_VIOLATION')
 if not np.isfinite(f[['y_true_delta_V','y_pred_delta_V']].to_numpy()).all():raise RuntimeError('PREDICTION_NONFINITE')
 counts=f.groupby('origin_row').horizon_step.agg(list)
 if not all(v==list(range(1,25)) for v in counts):raise RuntimeError('INCOMPLETE_HORIZONS')
 return f

def assert_fresh(path=LEDGER):
 if Path(path).exists():raise RuntimeError('ASSESSMENT_ONE_SHOT_ALREADY_ENTERED_NO_RERUN')

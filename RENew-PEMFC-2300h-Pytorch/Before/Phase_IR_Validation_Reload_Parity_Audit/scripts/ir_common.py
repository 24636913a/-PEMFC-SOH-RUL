"""Read-only H/I dependencies. Never invoke their preparation or run entry points."""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

PHASE=Path(__file__).resolve().parents[1];PROJECT=PHASE.parent;REPO=PROJECT.parent
BASE='27d3001f0838d7fa8ee311516c5eaf9bdb2c63ec'
I=PROJECT/'Phase_I_Frozen_Checkpoint_Assessment'
sys.path.insert(0,str(I/'scripts'));import i_common as i
h=i.h;e=i.e;c=i.c;H=i.H
OUT=PHASE/'outputs';PROTOCOL=PHASE/'configs/IR_AUDIT_PROTOCOL.yaml';SHA=PROTOCOL.with_suffix('.sha256')
LEDGER=OUT/'execution_logs/IR_EXECUTION_LEDGER.json'
RUNS=i.RUNS;PATHS=['A','B','C','D','E']
I_PRESERVED=[i.PROTOCOL,i.SHA,i.LEDGER,I/'reports/I_EXECUTION_BLOCKED.md',I/'reports/I_FINAL_SCIENTIFIC_CONCLUSION.md']
ALLOW=[p for p in i.READ_ALLOWLIST if p not in i.F_PRED]+I_PRESERVED+list((I/'scripts').glob('*.py'))+list((I/'tests').glob('*.py'))

def allowed(path):
 p=Path(path).resolve()
 if p==c.N1.resolve():raise RuntimeError('FULL_N1_ACCESS_PROHIBITED')
 if p.is_relative_to(PHASE) or p in [x.resolve() for x in ALLOW]:return p
 raise RuntimeError('IR_SOURCE_NOT_ALLOWLISTED:'+str(p))

def digest(path):return hashlib.sha256(allowed(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(allowed(path).read_text(encoding='utf-8'))
def json_write(path,value):
 path=Path(path).resolve()
 if not path.is_relative_to(PHASE):raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.writing')
 with tmp.open('w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def csv_write(path,frame):
 path=Path(path).resolve()
 if not path.is_relative_to(PHASE):raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
 path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')

def own_code():
 files={p.relative_to(REPO).as_posix():digest(p) for p in sorted((PHASE/'scripts').glob('*.py'))+sorted((PHASE/'tests').glob('*.py'))}
 return {'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}

def historical_protocols():
 # Do not call I.load_protocol(): its broad source list includes Assessment caches.
 if digest(i.PROTOCOL)!=i.SHA.read_text(encoding='utf-8').split()[0]:raise RuntimeError('I_PROTOCOL_IDENTITY_CHANGED')
 ip=yaml.safe_load(i.PROTOCOL.read_text(encoding='utf-8'))
 if i.code_identity()!=ip['code_identity']:raise RuntimeError('I_CODE_IDENTITY_CHANGED')
 if read_json(i.LEDGER)['state']!='BLOCKED':raise RuntimeError('I_BLOCKED_LEDGER_CHANGED')
 return h.load_protocol(),ip

def load_protocol():
 if digest(PROTOCOL)!=SHA.read_text(encoding='utf-8').split()[0]:raise RuntimeError('IR_PROTOCOL_IDENTITY_CHANGED')
 cfg=yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
 if own_code()!=cfg['code_identity']:raise RuntimeError('IR_CODE_IDENTITY_CHANGED')
 for path,sha in cfg['sources'].items():
  if digest(Path(path) if Path(path).is_absolute() else REPO/path)!=sha:raise RuntimeError('IR_SOURCE_IDENTITY_CHANGED:'+path)
 historical_protocols();return cfg

def validation_input():
 hc,_=historical_protocols();ecfg,_=e.load_protocol();dev,access=i.read_dev();scaler=i.frozen_scaler()
 val=c.build_windows(dev,ecfg,2103,2404)
 if len(val.origins)!=205:raise RuntimeError('VALIDATION_UNIVERSE_MISMATCH')
 x=e.prep(val.x,scaler,hc['preprocessing']['Top10_indices'],'EWMA')
 fp={'validation_x_sha256':h.array_hash(x),'validation_y_sha256':h.array_hash(val.y),'validation_origins_sha256':h.array_hash(val.origins),'validation_target_rows_sha256':h.array_hash(val.targets),'scaler_sha256':h.array_hash(np.stack([scaler.mean,scaler.scale])),'development_prefix_fingerprint_sha256':hashlib.sha256(dev.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()).hexdigest()}
 for spec in RUNS:
  old=read_json(H/'outputs/training_logs'/spec['H_run_id']/'input_audit.json')
  for key in fp:
   if fp[key]!=old[key]:raise RuntimeError('H_INPUT_IDENTITY_MISMATCH:'+spec['H_run_id']+':'+key)
 x=x[:32].copy();y=val.y[:32].copy();origins=val.origins[:32].copy();targets=val.targets[:32].copy()
 if x.shape!=(32,48,14) or x.dtype!=np.float32 or not np.isfinite(x).all() or (targets>=2404).any():raise RuntimeError('FIXED_VALIDATION_CONTRACT_INVALID')
 return x,y,origins,targets,{'historical_fingerprints':fp,'fixed_input_sha256':h.array_hash(x),'fixed_origins_sha256':h.array_hash(origins),'fixed_targets_sha256':h.array_hash(targets),'read_nrows':3007,'development_rows':3005,'access':access,'target_scale_V':scaler.target_scale,'scaler_refitted':False,'Assessment_windows_constructed':False,'Assessment_prediction_cache_read':False,'Test_accessed':False,'full_N1_hash':False}

def static_audit():
 forbidden={'fit','backward','step','train','prepare_data','prepare','execute','train_model','fit_condition'}
 for path in (PHASE/'scripts').glob('*.py'):
  for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
   if isinstance(node,ast.Call):
    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
    if name in forbidden:raise RuntimeError('IR_FORBIDDEN_EXECUTION_CALL:'+name)
 return {'pass':True,'forbidden_calls':sorted(forbidden),'scope':'all IR production scripts; read-only inherited APIs explicitly invoked'}

def difference_statistics(actual,saved,scale):
 if actual.shape!=saved.shape or actual.shape!=(32,24):raise RuntimeError('OUTPUT_SHAPE_IDENTITY_MISMATCH')
 finite=bool(np.isfinite(actual).all() and np.isfinite(saved).all())
 nan_inf={'actual_nan':int(np.isnan(actual).sum()),'actual_inf':int(np.isinf(actual).sum()),'saved_nan':int(np.isnan(saved).sum()),'saved_inf':int(np.isinf(saved).sum())}
 result={'array_equal':bool(np.array_equal(actual,saved)),'unequal_elements':int((actual!=saved).sum()),'unequal_fraction':float((actual!=saved).mean()),'finite':finite,**nan_inf}
 if not finite:return {**result,'max_abs_difference_standardized':None,'mean_abs_difference_standardized':None,'difference_RMSE_standardized':None,'max_abs_difference_mV':None,'difference_RMSE_mV':None}
 d=actual.astype(np.float64)-saved.astype(np.float64);position=np.unravel_index(np.abs(d).argmax(),d.shape)
 result.update(max_abs_difference_standardized=float(np.abs(d).max()),mean_abs_difference_standardized=float(np.abs(d).mean()),difference_RMSE_standardized=float(np.sqrt(np.mean(d*d))),max_abs_difference_mV=float(np.abs(d).max()*scale*1000),difference_RMSE_mV=float(np.sqrt(np.mean(d*d))*scale*1000),max_difference_batch_index_zero_based=0,max_difference_sample_index_zero_based=int(position[0]),max_difference_horizon_step=int(position[1]+1))
 for tolerance in [1e-8,1e-7,1e-6,1e-5]:result['fraction_abs_difference_gt_'+str(tolerance)+'_standardized']=float((np.abs(d)>tolerance).mean())
 return result

def snapshot(model):
 return {key:t.detach().cpu().clone() for key,t in model.state_dict().items()}

def state_audit(model,before,flags):
 after=snapshot(model);parameter_names={k for k,_ in model.named_parameters()};buffer_names={k for k,_ in model.named_buffers()}
 rows=[{'name':k,'kind':'parameter' if k in parameter_names else 'buffer' if k in buffer_names else 'state_entry','shape':list(v.shape),'dtype':str(v.dtype),'equal':bool(torch.equal(v,after[k])),'before_sha256':h.array_hash(v.numpy()),'after_sha256':h.array_hash(after[k].numpy())} for k,v in before.items()]
 flags_after={k:p.requires_grad for k,p in model.named_parameters()}
 return {'all_state_entries_equal':all(r['equal'] for r in rows),'state_entries':rows,'requires_grad_before':flags,'requires_grad_after':flags_after,'requires_grad_unchanged':flags==flags_after,'all_parameter_grad_None':all(p.grad is None for p in model.parameters())}

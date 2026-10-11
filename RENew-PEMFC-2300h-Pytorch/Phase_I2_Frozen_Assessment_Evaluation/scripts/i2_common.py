"""Independent I2 authority and guarded writes; inherited H Path A only."""
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

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
BASE = '58ae0657f2b953a684178da6724c2e44203b114c'
I = PROJECT / 'Phase_I_Frozen_Checkpoint_Assessment'
IR = PROJECT / 'Phase_IR_Validation_Reload_Parity_Audit'
sys.path.insert(0, str(I / 'scripts'))
import i_common as i
from i_metrics import project_metrics, independent_metrics, diagnostics
sys.path.insert(0, str(IR / 'scripts'))
import ir_common as ir
h, e, c, H = i.h, i.e, i.c, i.H
OUT = PHASE / 'outputs'
PROTOCOL = PHASE / 'configs/I2_ASSESSMENT_PROTOCOL.yaml'
SHA = PROTOCOL.with_suffix('.sha256')
LEDGER = OUT / 'execution_ledger/I2_EXECUTION_LEDGER.json'
RUNS = [{**s, 'run_id': s['run_id'].replace('I_', 'I2_', 1)} for s in i.RUNS]
HISTORY_LIMITS = ['PHASE_I_ORIGINAL_EXECUTION_BLOCKED', 'PHASE_IR_H_AUTHORITY_PARITY_VERIFIED',
 'PHASE_IR_I_EXECUTION_SEMANTICS_DIFFERENCE_OBSERVED',
 'PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS', 'C1_D0_E0_REPRODUCIBILITY_UNRESOLVED',
 'TIMESNET_FREEZE', 'BACKBONE_SEARCH_CLOSED']
ALLOW = i.READ_ALLOWLIST + ir.ALLOW + [ir.PROTOCOL, ir.SHA, IR/'outputs/IR_SUMMARY.json',
 IR/'outputs/input_identity/IR_FIXED_VALIDATION_INPUT.npz',
 I/'outputs/preparation_audit/I_ASSESSMENT_KEY_TRUTH_AUTHORITY.csv']
ALLOW += list((I/'scripts').glob('*.py')) + list((I/'tests').glob('*.py'))
ALLOW += list((IR/'scripts').glob('*.py')) + list((IR/'tests').glob('*.py'))
ALLOW += list((IR/'outputs/input_identity').glob('*.npz'))

def allowed(path):
 p = Path(path).resolve()
 if p == c.N1.resolve(): raise RuntimeError('FULL_N1_ACCESS_OR_HASH_PROHIBITED')
 if p.is_relative_to(PHASE) or p in {q.resolve() for q in ALLOW}: return p
 raise RuntimeError('I2_SOURCE_NOT_ALLOWLISTED:'+str(p))

def digest(path): return hashlib.sha256(allowed(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(allowed(path).read_text(encoding='utf-8'))
def read_csv(path): return pd.read_csv(allowed(path), float_precision='round_trip')
def write_target(path):
 p=Path(path).resolve()
 if not p.is_relative_to(PHASE): raise RuntimeError('HISTORICAL_WRITE_PROHIBITED')
 p.parent.mkdir(parents=True, exist_ok=True)
 return p
def text_write(path, value): write_target(path).write_text(value, encoding='utf-8', newline='\n')
def json_write(path, value):
 p=write_target(path); tmp=p.with_suffix(p.suffix+'.writing')
 with tmp.open('w', encoding='utf-8', newline='\n') as f:
  f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,p)
def csv_write(path, frame): frame.to_csv(write_target(path), index=False, float_format='%.17g', lineterminator='\n')
def npz_write(path, **arrays): np.savez_compressed(write_target(path), **arrays)

def own_code():
 files={str(p.relative_to(REPO).as_posix()):digest(p) for d in ['scripts','tests'] for p in sorted((PHASE/d).glob('*.py'))}
 files.update(i.code_identity()['files']); files.update(ir.own_code()['files'])
 return {'files':files, 'sha256':hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}

def verify_hashes(records):
 for path, expected in records.items():
  p=Path(path) if Path(path).is_absolute() else REPO/path
  if digest(p)!=expected: raise RuntimeError('I2_FROZEN_SOURCE_CHANGED:'+path)

def load_protocol():
 if digest(PROTOCOL)!=SHA.read_text(encoding='utf-8').split()[0]: raise RuntimeError('I2_PROTOCOL_CHANGED')
 cfg=yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
 if own_code()!=cfg['code_identity']: raise RuntimeError('I2_CODE_IDENTITY_CHANGED')
 verify_hashes(cfg['sources']); verify_hashes(cfg['prepared_inputs'])
 i.load_protocol(); ir.load_protocol()
 if read_json(i.LEDGER)['state']!='BLOCKED': raise RuntimeError('ORIGINAL_I_BLOCKED_LEDGER_CHANGED')
 return cfg

def assert_fresh(path=LEDGER):
 if Path(path).exists(): raise RuntimeError('I2_ONE_SHOT_CONSUMED_NO_RETRY')

def validate_frame(frame, truth, spec, cfg):
 return i.validate_predictions(frame,truth,spec,digest(PROTOCOL),cfg['checkpoints_by_run'][spec['run_id']]['sha256'],
  cfg['input_fingerprints']['assessment_input_sha256'])

def verify_state(audit):
 if not (audit['all_state_entries_equal'] and audit['requires_grad_unchanged']
  and audit['all_parameter_grad_None'] and all(audit['requires_grad_before'].values())):
  raise RuntimeError('I2_STATE_OR_EXECUTION_FLAGS_CHANGED')

def static_audit():
 banned={'fit','backward','step','train','predict_frozen','inference_mode','requires_grad_', 'prepare', 'load_development'}
 for path in (PHASE/'scripts').glob('*.py'):
  for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
   if isinstance(node,ast.Call):
    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
    if name in banned: raise RuntimeError('I2_FORBIDDEN_CALL:'+name)
 return {'pass':True,'banned_calls':sorted(banned),'inherited_prefix_reader':'i.prepare_data -> i.read_dev -> c.load_development(nrows=3007)'}

def scientific_decision(paired):
 a=int((paired.D20_F2_minus_F0_mV<0).sum()); b=int((paired.D21_F2_minus_F1_mV<0).sum())
 if a==3 and b==3: primary='F2_REUSED_ASSESSMENT_MULTISEED_GAIN_OBSERVED'
 elif b==3: primary='F2_GATE_GAIN_WITHOUT_NATIVE_SUPERIORITY'
 elif a in (1,2) or b in (1,2): primary='F2_ASSESSMENT_SEED_DEPENDENCE_OBSERVED'
 else: primary='F2_VALIDATION_GAIN_NOT_TRANSFERRED'
 extra=[]
 if a in (1,2) or b in (1,2): extra.append('F2_ASSESSMENT_SEED_DEPENDENCE_OBSERVED')
 if a==0 or paired.D20_F2_minus_F0_mV.mean()>=0: extra.append('F2_VALIDATION_GAIN_NOT_TRANSFERRED')
 return primary, sorted(set(extra)-{primary})

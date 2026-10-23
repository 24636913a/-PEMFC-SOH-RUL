"""One shot: all nine Path A Validation checks precede any Assessment forward."""
import argparse
import datetime
import subprocess
import traceback
from unittest.mock import patch
from i2_common import *

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def forbidden(*args,**kwargs): raise RuntimeError('TRAINING_OPERATION_PROHIBITED')

def worker(spec,mode):
 cfg=load_protocol(); ledger=read_json(LEDGER)
 required='VALIDATION_RUNNING' if mode=='validation' else 'ASSESSMENT_RUNNING'
 if ledger['state']!=required: raise RuntimeError('WORKER_NOT_AUTHORIZED_IN_LEDGER_STATE')
 folder=OUT/('validation_reload_audit' if mode=='validation' else 'predictions')
 auditpath=folder/(spec['run_id']+'_'+mode+'_audit.json')
 rawpath=folder/(spec['run_id']+'_'+mode+'.npz')
 if auditpath.exists() or rawpath.exists(): raise RuntimeError('WORKER_OUTPUT_ALREADY_EXISTS_NO_RETRY')
 env=i.configure_environment(); hc=h.load_protocol()
 old=next(r for r in i.RUNS if r['H_run_id']==spec['H_run_id'])
 _,ck,manifest=i.verify_checkpoint(old)
 manifest['run_id']=spec['run_id']
 if manifest!=cfg['checkpoints_by_run'][spec['run_id']]: raise RuntimeError('CHECKPOINT_CHANGED')
 device=torch.device('cuda')
 model=h.model(spec['model'],hc['hyperparameters'],device)
 model.load_state_dict(ck['state_dict'],strict=True); model.eval()
 before=ir.snapshot(model); flags={k:p.requires_grad for k,p in model.named_parameters()}
 before_hash=e.state_digest(model)
 with np.load(allowed(OUT/'inputs/I2_INPUTS.npz'),allow_pickle=False) as saved:
  x=saved['validation_x' if mode=='validation' else 'assessment_x'].copy()
 expected=(32,48,14) if mode=='validation' else (378,48,14)
 if x.shape!=expected or x.dtype!=np.float32 or not np.isfinite(x).all(): raise RuntimeError('WORKER_INPUT_INVALID')
 with patch.object(torch.Tensor,'backward',forbidden), patch.object(torch.autograd,'backward',forbidden), patch.object(torch.optim.Optimizer,'__init__',forbidden):
  actual=e.predict(model,x,32,device)
 if actual.shape!=(expected[0],24) or actual.dtype!=np.float32 or not np.isfinite(actual).all(): raise RuntimeError('OUTPUT_CONTRACT_INVALID')
 voltage=actual*cfg['target_scale_V']
 # Raw bytes and all differences persisted before exact Validation assertion.
 reference=ck['fixed_validation_prediction'].cpu().numpy() if mode=='validation' else np.empty((0,),dtype=np.float32)
 npz_write(rawpath,standardized_prediction=actual,voltage_prediction=voltage,reference_standardized=reference)
 state=ir.state_audit(model,before,flags)
 audit={**spec,'mode':mode,'environment':env,'protocol_sha256':digest(PROTOCOL),
  'code_identity_sha256':cfg['code_identity']['sha256'],'checkpoint_sha256':manifest['sha256'],
  'helper':'e.predict(model,x,32,device)','path':'H Path A','state_dict_before_sha256':before_hash,
  'state_dict_after_sha256':e.state_digest(model),'state_audit':state,'raw_npz_sha256':digest(rawpath),
  'backward_calls':0,'optimizer_created':False,'optimizer_steps':0,'gate_capture':'NOT_COLLECTED',
  'formal_test_accessed':False,'new_training':False,'completed_utc':now()}
 if mode=='validation': audit['difference_statistics']=ir.difference_statistics(actual,reference,cfg['target_scale_V'])
 json_write(auditpath,audit)
 verify_state(state)
 if before_hash!=audit['state_dict_after_sha256']: raise RuntimeError('STATE_DIGEST_CHANGED')
 if mode=='validation':
  if not np.array_equal(actual,reference): raise RuntimeError('PHASE_I2_VALIDATION_PARITY_BLOCKED:'+spec['run_id'])
 else:
  truth=read_csv(OUT/'inputs/I2_ASSESSMENT_KEY_TRUTH.csv'); f=truth.copy()
  f['y_pred_delta_V']=voltage.ravel().astype(np.float64)
  for k,v in {**spec,'protocol_sha256':digest(PROTOCOL),'checkpoint_sha256':manifest['sha256'],
   'assessment_input_fingerprint':cfg['input_fingerprints']['assessment_input_sha256']}.items(): f[k]=v
  validate_frame(f,truth,spec,cfg)
  csvpath=folder/(spec['run_id']+'_assessment.csv'); csv_write(csvpath,f)
  audit['csv_sha256']=digest(csvpath); audit['records']=len(f); json_write(auditpath,audit)
 print(spec['run_id'],mode,'PASS',flush=True)

def execute():
 assert_fresh(); cfg=load_protocol()
 ledger={'state':'VALIDATION_RUNNING','started_utc':now(),'protocol_sha256':digest(PROTOCOL),
  'code_identity_sha256':cfg['code_identity']['sha256'],'completed_validation':[], 'completed_assessment':[],
  'transitions':[],'formal_test_accessed':False,'new_training':False,'retry_allowed':False}
 json_write(LEDGER,ledger)
 try:
  for mode in ['validation','assessment']:
   if mode=='assessment':
    if len(ledger['completed_validation'])!=9: raise RuntimeError('NINE_VALIDATION_GATES_NOT_PASS')
    ledger['state']='ASSESSMENT_RUNNING'; ledger['transitions'].append({'state':ledger['state'],'utc':now()}); json_write(LEDGER,ledger)
   for spec in RUNS:
    logfile=OUT/'execution_ledger'/(spec['run_id']+'_'+mode+'.log'); write_target(logfile)
    with logfile.open('x',encoding='utf-8') as log:
     process=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',spec['run_id'],'--mode',mode],cwd=PHASE,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode: raise RuntimeError('WORKER_FAILED:'+spec['run_id']+':'+mode+'; log='+str(logfile))
    ledger['completed_'+mode].append(spec['run_id']); json_write(LEDGER,ledger)
    print(spec['run_id'],mode,'PASS',flush=True)
  ledger['state']='INDEPENDENT_AUDIT_RUNNING';json_write(LEDGER,ledger)
  from audit_i2_predictions import audit_all
  audit=audit_all(cfg)
  from analyze_i2_results import analyze
  result=analyze(cfg)
  from plot_i2_results import plot_all
  plot_all()
  load_protocol()
  ledger.update(state='COMPLETED',completed_utc=now(),artifact_integrity=audit,scientific_status=result['scientific_status'])
  json_write(LEDGER,ledger)
  print('I2_COMPLETED',result['scientific_status'],flush=True)
 except Exception as ex:
  stage=ledger['state'];ledger.update(state='BLOCKED',failure_stage=stage,exception=repr(ex),traceback=traceback.format_exc(),stopped_utc=now(),
   scientific_status='PHASE_I2_VALIDATION_PARITY_BLOCKED' if stage=='VALIDATION_RUNNING' else 'PHASE_I2_TECHNICAL_BLOCKED')
  json_write(LEDGER,ledger)
  text_write(PHASE/'reports/I2_EXECUTION_BLOCKED.md','# I2 fail-closed\n\nNo retry permitted.\n\n'+ledger['traceback'])
  raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',choices=[r['run_id'] for r in RUNS]);parser.add_argument('--mode',choices=['validation','assessment'])
 args=parser.parse_args()
 if args.worker and args.mode: worker(next(r for r in RUNS if r['run_id']==args.worker),args.mode)
 elif args.execute: execute()
 else: parser.error('Use --execute once, or authorized internal worker')

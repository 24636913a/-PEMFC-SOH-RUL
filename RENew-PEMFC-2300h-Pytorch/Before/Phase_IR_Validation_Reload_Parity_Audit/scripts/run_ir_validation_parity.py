"""One finite 9 x 5 audit; every controlled path runs in a fresh Python process."""
import argparse
import datetime
import subprocess
import traceback
from unittest.mock import patch
from ir_common import *

def freeze():
 if PROTOCOL.exists() or LEDGER.exists():raise RuntimeError('IR_ALREADY_FROZEN_OR_STARTED')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=BASE:raise RuntimeError('BASE_COMMIT_CHANGED')
 if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()!='main':raise RuntimeError('MAIN_REQUIRED')
 if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True):raise RuntimeError('TRACKED_TREE_DIRTY')
 hc,ip=historical_protocols();static=static_audit();x,y,origins,targets,data=validation_input()
 folder=OUT/'input_identity';folder.mkdir(parents=True,exist_ok=True)
 input_path=folder/'IR_FIXED_VALIDATION_INPUT.npz';np.savez_compressed(input_path,x=x,y=y,origins=origins,targets=targets)
 csv_write(folder/'IR_FIXED_VALIDATION_ORIGINS.csv',pd.DataFrame({'sample_index_zero_based':np.arange(32),'origin_row':origins,'first_target_row':targets[:,0],'last_target_row':targets[:,-1]}))
 records=[];reference_paths=[]
 for spec,old in zip(RUNS,ip['checkpoints']):
  model,ck,identity=i.verify_checkpoint(spec,old)
  reference=ck['fixed_validation_prediction'].cpu().numpy()
  if reference.shape!=(32,24) or reference.dtype!=np.float32 or not np.isfinite(reference).all():raise RuntimeError('SAVED_FIXED_PREDICTION_INVALID')
  ref=folder/(spec['H_run_id']+'_SAVED_REFERENCE.npz');np.savez_compressed(ref,saved=reference);reference_paths.append(ref)
  records.append({**spec,'checkpoint_path':identity['path'],'checkpoint_sha256':identity['sha256'],'best_epoch':identity['best_epoch'],'parameter_count':identity['parameter_count'],'state_schema_sha256':hashlib.sha256(json.dumps(identity['state_schema'],sort_keys=True).encode()).hexdigest(),'target_scale_V':identity['target_scale_V'],'saved_prediction_sha256':h.array_hash(reference),'reference_path':ref.relative_to(REPO).as_posix()})
  del model,ck
 sources={p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p):digest(p) for p in ALLOW+[input_path]+reference_paths}
 env=h.configure_normal(hc)
 cfg={'phase':PHASE.name,'base_commit':BASE,'authorization':'USER_EXPLICIT_VALIDATION_ONLY_NINE_CHECKPOINT_FIVE_PATH_AUDIT','code_identity':own_code(),'sources':sources,'environment':env,'H_protocol_sha256':digest(h.PROTOCOL),'I_protocol_sha256':digest(i.PROTOCOL),'I_BLOCKED_ledger_sha256':digest(i.LEDGER),'data_identity':data,'input_archive_sha256':digest(input_path),'checkpoints':records,'paths':{
 'A':{'construct_load':'CUDA construct then strict original state load','context':'torch.no_grad','requires_grad':'original','predictor':'original e.predict -> c.predict'},
 'B':{'construct_load':'same as A','context':'torch.inference_mode','requires_grad':'original','predictor':'same one-batch predict_delta arithmetic loop'},
 'C':{'construct_load':'same as A','context':'torch.no_grad','requires_grad':False,'predictor':'original e.predict -> c.predict'},
 'D':{'construct_load':'CPU construct/load then CUDA transfer','context':'torch.no_grad','requires_grad':'original','predictor':'original e.predict -> c.predict'},
 'E':{'construct_load':'CPU construct/load then CUDA transfer','context':'torch.inference_mode','requires_grad':False,'predictor':'unmodified original I.predict_frozen'}},
 'budget':{'checkpoint_count':9,'paths_per_checkpoint':5,'total_worker_forward_calls':45,'fixed_origins':32,'horizon':24,'batch_size':32,'automatic_retries':0},
 'parity_rule':'np.array_equal(actual, saved) ONLY; diagnostic tolerances cannot change PASS',
 'diagnostic_tolerances_standardized':[1e-8,1e-7,1e-6,1e-5],
 'numeric_scale':'standardized DeltaV differences * frozen target_scale_V * 1000 = mV',
 'scope':{'training':False,'Assessment_window_construction':False,'Assessment_prediction_cache_read':False,'Assessment_inference':False,'Test_access':False,'scaler_fit':False,'H_I_modification':False,'checkpoint_modification':False},
 'interpretation':'single-variable controlled differences establish execution-context dependence for these checkpoints, not a changed equation or unique GPU kernel mechanism; no performance ranking',
 'stops':['identity mismatch stops all further paths','numerical mismatches are retained without weakening exact gate','path exceptions recorded without device fallback','one authorized matrix only; no automatic Phase I restart'],
 'static_code_audit':static}
 PROTOCOL.parent.mkdir(parents=True,exist_ok=True);PROTOCOL.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8',newline='\n');SHA.write_text(digest(PROTOCOL)+'  IR_AUDIT_PROTOCOL.yaml\n',encoding='utf-8',newline='\n')
 json_write(folder/'IR_INPUT_IDENTITY.json',data);json_write(folder/'IR_CHECKPOINT_IDENTITY.json',records)
 print('IR_PROTOCOL_FROZEN '+digest(PROTOCOL),flush=True)

def worker(run_id,path_id):
 cfg=load_protocol();spec=next(r for r in RUNS if r['H_run_id']==run_id);record=next(r for r in cfg['checkpoints'] if r['H_run_id']==run_id)
 target=OUT/'path_comparisons'/f'{run_id}_Path{path_id}.json'
 if target.exists():raise RuntimeError('PATH_ALREADY_EXECUTED_NO_RETRY')
 result={**spec,'path_id':path_id,'checkpoint_sha256':record['checkpoint_sha256'],'protocol_sha256':digest(PROTOCOL),'input_sha256':cfg['data_identity']['fixed_input_sha256'],'status':'STARTED','exception':None,'traceback':None}
 try:
  hc,ip=historical_protocols();env=h.configure_normal(hc)
  if env!=cfg['environment']:raise RuntimeError('IR_ENVIRONMENT_IDENTITY_MISMATCH')
  archive=OUT/'input_identity/IR_FIXED_VALIDATION_INPUT.npz'
  with np.load(allowed(archive),allow_pickle=False) as z:x=z['x'].copy();y=z['y'].copy();origins=z['origins'].copy()
  if h.array_hash(x)!=result['input_sha256']:raise RuntimeError('IR_FIXED_INPUT_IDENTITY_MISMATCH')
  _,ck,_=i.verify_checkpoint(spec,next(r for r in ip['checkpoints'] if r['H_run_id']==run_id))
  saved=ck['fixed_validation_prediction'].cpu().numpy().copy()
  if h.array_hash(saved)!=record['saved_prediction_sha256']:raise RuntimeError('IR_SAVED_PREDICTION_IDENTITY_MISMATCH')
  device=torch.device('cuda');model=h.model(spec['model'],hc['hyperparameters'],device if path_id in ['A','B','C'] else torch.device('cpu'))
  model.load_state_dict(ck['state_dict'],strict=True)
  if path_id in ['D','E']:model.to(device)
  model.eval();original_flags={k:p.requires_grad for k,p in model.named_parameters()}
  if path_id in ['C','E']:
   for p in model.parameters():p.requires_grad_(False)
  before=snapshot(model);before_hash=e.state_digest(model);flags={k:p.requires_grad for k,p in model.named_parameters()}
  counters={'Tensor.backward':0,'autograd.backward':0,'Optimizer.created':0}
  def prohibited(name):
   def block(*args,**kwargs):counters[name]+=1;raise RuntimeError('READ_ONLY_OPERATION_VIOLATION:'+name)
   return block
  with patch.object(torch.Tensor,'backward',prohibited('Tensor.backward')),patch.object(torch.autograd,'backward',prohibited('autograd.backward')),patch.object(torch.optim.Optimizer,'__init__',prohibited('Optimizer.created')):
   if path_id in ['A','C','D']:actual=e.predict(model,x,32,device)
   elif path_id=='B':
    with torch.inference_mode():actual=model.predict_delta(torch.from_numpy(x).to(device)).cpu().numpy().copy()
   elif path_id=='E':actual,_=i.predict_frozen(model,x)
   else:raise RuntimeError('UNREGISTERED_PATH')
  # Persist raw arrays and differences before classifying parity or asserting state integrity.
  prediction_path=OUT/'path_comparisons'/f'{run_id}_Path{path_id}.npz';prediction_path.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(prediction_path,actual=actual,saved=saved,difference=actual.astype(np.float64)-saved.astype(np.float64),origins=origins)
  statistics=difference_statistics(actual,saved,record['target_scale_V'])
  result.update(statistics);result.update(device=str(next(model.parameters()).device),input_dtype=str(x.dtype),output_dtype=str(actual.dtype),prediction_shape=list(actual.shape),output_sha256=h.array_hash(actual),prediction_archive_sha256=digest(prediction_path),environment=env)
  if statistics['finite']:
   factor=record['target_scale_V']*1000;result['fixed32_actual_RMSE_mV']=float(np.sqrt(np.mean((actual.astype(float)*record['target_scale_V']-y.astype(float))**2))*1000);result['fixed32_saved_RMSE_mV']=float(np.sqrt(np.mean((saved.astype(float)*record['target_scale_V']-y.astype(float))**2))*1000);result['fixed32_RMSE_delta_mV']=result['fixed32_actual_RMSE_mV']-result['fixed32_saved_RMSE_mV'];result['max_difference_origin_row']=int(origins[statistics['max_difference_sample_index_zero_based']])
  result['status']='NUMERIC_COMPARISON_SAVED';json_write(target,result)
  state=state_audit(model,before,flags);state.update(state_sha256_before=before_hash,state_sha256_after=e.state_digest(model),original_constructed_requires_grad=original_flags,operation_counters=counters,backward_executed=any(counters[k] for k in ['Tensor.backward','autograd.backward']),optimizer_created=bool(counters['Optimizer.created']),optimizer_step_executed=False,model_eval_before=True,model_eval_after=not model.training)
  json_write(OUT/'parameter_state_audit'/f'{run_id}_Path{path_id}.json',state)
  result['state_audit_pass']=bool(state['all_state_entries_equal'] and state['requires_grad_unchanged'] and state['all_parameter_grad_None'] and before_hash==state['state_sha256_after'] and not any(counters.values()))
  result['status']='COMPARED' if result['state_audit_pass'] and result['finite'] else 'STATE_OR_NUMERICAL_BLOCKED';json_write(target,result)
  print(run_id+' '+path_id+' exact='+str(result['array_equal'])+' max_diff_mV='+str(result['max_abs_difference_mV']),flush=True)
 except Exception as exc:
  result.update(status='EXCEPTION',exception=repr(exc),traceback=traceback.format_exc());json_write(target,result);print(result['traceback'],flush=True)
  if any(term in str(exc) for term in ['IDENTITY','HASH','SCHEMA','CHANGED','ALLOWLIST','METADATA']):sys.exit(2)
  sys.exit(1)

def run_matrix():
 if LEDGER.exists():raise RuntimeError('IR_MATRIX_ALREADY_STARTED_NO_AUTOMATIC_REPEAT')
 cfg=load_protocol();ledger={'status':'RUNNING','protocol_sha256':digest(PROTOCOL),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paths':[],'Assessment_inference':False,'Test_accessed':False,'new_training':False};json_write(LEDGER,ledger)
 try:
  tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=PHASE,capture_output=True,text=True,encoding='utf-8',errors='replace')
  json_write(OUT/'execution_logs/IR_REGRESSION_TESTS.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,'synthetic_only':True})
  if tests.returncode:raise RuntimeError('IR_REGRESSION_TESTS_FAILED')
  for spec in RUNS:
   for path_id in PATHS:
    args=[sys.executable,'-B','scripts/run_ir_validation_parity.py','--worker',spec['H_run_id'],'--path',path_id]
    process=subprocess.run(args,cwd=PHASE,capture_output=True,text=True,encoding='utf-8',errors='replace')
    log={'command':args,'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr};json_write(OUT/'execution_logs'/f"{spec['H_run_id']}_Path{path_id}.json",log)
    ledger['paths'].append({'run_id':spec['H_run_id'],'path_id':path_id,'returncode':process.returncode});json_write(LEDGER,ledger)
    print(process.stdout.strip(),flush=True)
    if process.returncode==2:raise RuntimeError('PHASE_IR_IDENTITY_BLOCKED')
  from audit_ir_differences import audit_all
  from analyze_ir_root_cause import analyze
  audit_all(cfg);summary=analyze(cfg);load_protocol();ledger['status']='COMPLETED';ledger['scientific_status']=summary['primary_status'];json_write(LEDGER,ledger);print('IR_COMPLETE '+summary['primary_status'],flush=True)
 except Exception as exc:
  ledger.update(status='BLOCKED',exception=repr(exc),traceback=traceback.format_exc());json_write(LEDGER,ledger);raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',choices=[r['H_run_id'] for r in RUNS]);parser.add_argument('--path',choices=PATHS);args=parser.parse_args()
 if args.freeze:freeze()
 elif args.run:run_matrix()
 else:
  if not args.path:parser.error('--worker requires --path')
  worker(args.worker,args.path)

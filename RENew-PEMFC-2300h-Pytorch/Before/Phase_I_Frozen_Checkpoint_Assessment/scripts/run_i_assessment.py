"""Single authorized inference of nine frozen H weights; no training paths."""
import argparse
import datetime
import time
import traceback
from i_common import *
from i_metrics import project_metrics
from prepare_i_assessment import static_inference_audit
from audit_i_predictions import audit_all

def execute():
 assert_fresh();cfg=load_protocol();ledger={'state':'PREPARED','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':digest(PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256'],'completed_runs':[],'transitions':[],'formal_test_accessed':False,'new_training':False}
 def state(name):
  ledger['state']=name;ledger['transitions'].append({'state':name,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});atomic_json(LEDGER,ledger)
 state('PREPARED')
 try:
  test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=PHASE,capture_output=True,text=True)
  log=OUT/'preparation_audit/I_SYNTHETIC_TEST_LOG.txt';log.parent.mkdir(parents=True,exist_ok=True);log.write_text(test.stdout+test.stderr,encoding='utf-8')
  if test.returncode:raise RuntimeError('I_SYNTHETIC_REGRESSION_TEST_FAILED')
  static_inference_audit();data=prepare_data()
  if data['fingerprints']!=cfg['input_fingerprints']:raise RuntimeError('FROZEN_DATA_IDENTITY_CHANGED')
  env=configure_environment()
  if env!=cfg['environment']:raise RuntimeError('FROZEN_ENVIRONMENT_CHANGED')
  models={};reloads=[];synthetic=np.random.default_rng(2026).normal(size=(2,48,14)).astype(np.float32)
  for spec,identity in zip(RUNS,cfg['checkpoints']):
   m,ck,_=verify_checkpoint(spec,identity);m.to('cuda')
   val,_=predict_frozen(m,data['vx'][:32]);saved=ck['fixed_validation_prediction'].cpu().numpy()
   if not np.array_equal(val,saved):raise RuntimeError('FROZEN_VALIDATION_RELOAD_MISMATCH:'+spec['run_id'])
   smoke,gate=predict_frozen(m,synthetic,gate=spec['model']=='F2')
   reloads.append({**spec,'validation_reload_max_difference':float(np.max(np.abs(val-saved))),'validation_reload_pass':True,'synthetic_shape':list(smoke.shape),'synthetic_gate_count':0 if gate is None else len(gate),'parameter_update':False,'checkpoint_sha256':identity['sha256']})
   models[spec['run_id']]=m.cpu()
  csv_write(OUT/'preparation_audit/I_CHECKPOINT_RELOAD_AND_SMOKE.csv',pd.DataFrame(reloads))
  atomic_json(OUT/'preparation_audit/I_RUNTIME_FIREWALL.json',{'read_nrows':3007,'development_rows':3005,'development_filter':'bin_end_ophrs2_h < 1819.997738','full_N1_file_hash':False,'formal_test_accessed':False,'scaler_refitted':False,'prefix_access':data['prefix_access'],'synthetic_tests_pass':True,'assessment_before_protocol_freeze':False})
  state('ALL_CHECKPOINTS_VERIFIED');state('ASSESSMENT_RUNNING')
  for spec,identity in zip(RUNS,cfg['checkpoints']):
   ledger['current_run']=spec['run_id'];atomic_json(LEDGER,ledger);start=time.perf_counter();m=models[spec['run_id']].to('cuda');before=e.state_digest(m)
   pred,gate=predict_frozen(m,data['ax'],gate=spec['model']=='F2')
   frame=data['truth'].copy();frame['y_pred_delta_V']=(pred.astype(np.float64)*data['scale']).ravel()
   for key,value in spec.items():frame[key]=value
   frame['protocol_sha256']=digest(PROTOCOL);frame['checkpoint_sha256']=identity['sha256'];frame['assessment_input_fingerprint']=cfg['input_fingerprints']['assessment_input_sha256']
   frame=validate_predictions(frame,data['truth'],spec,digest(PROTOCOL),identity['sha256'],cfg['input_fingerprints']['assessment_input_sha256'])
   path=OUT/'predictions'/f"{spec['run_id']}_assessment.csv";csv_write(path,frame)
   reopened=pd.read_csv(path,float_precision='round_trip');validate_predictions(reopened,data['truth'],spec,digest(PROTOCOL),identity['sha256'],cfg['input_fingerprints']['assessment_input_sha256'])
   if not np.array_equal(reopened.y_pred_delta_V.to_numpy(),frame.y_pred_delta_V.to_numpy()):raise RuntimeError('PREDICTION_SERIALIZATION_MISMATCH')
   gate_sha=None
   if gate is not None:
    g=pd.DataFrame({'origin_row':frame.origin_row.unique(),'origin_age_h':frame.groupby('origin_row',sort=True).origin_age_h.first().to_numpy(),'gate':gate})
    for key,value in spec.items():g[key]=value
    g['protocol_sha256']=digest(PROTOCOL);g['checkpoint_sha256']=identity['sha256'];gp=OUT/'diagnostic'/f"{spec['run_id']}_gates.csv";csv_write(gp,g);gate_sha=digest(gp)
   metrics=project_metrics(reopened);atomic_json(OUT/'execution_ledger'/f"{spec['run_id']}_result.json",{**spec,'status':'COMPLETED','metrics':metrics,'records':len(reopened),'prediction_file_sha256':digest(path),'gate_file_sha256':gate_sha,'state_sha256_before':before,'state_sha256_after':e.state_digest(m),'parameters_updated':0,'inference_seconds':time.perf_counter()-start,'checkpoint_sha256':identity['sha256'],'protocol_sha256':digest(PROTOCOL)})
   m.cpu();ledger['completed_runs'].append(spec['run_id']);atomic_json(LEDGER,ledger)
   print(spec['run_id']+' completed; records='+str(len(frame))+'; RMSE_mV='+str(metrics['RMSE_mV']),flush=True)
  state('ALL_PREDICTIONS_SAVED');audit_all(cfg);state('METRICS_AUDITED')
  from analyze_i_results import analyze
  from plot_i_results import plot_all
  summary=analyze(cfg);plot_all();load_protocol();state('COMPLETED')
  print('PHASE_I_COMPLETED '+summary['scientific_status'],flush=True)
 except Exception as exc:
  ledger['exception']=repr(exc);ledger['failure_stage']=ledger['state'];ledger['scientific_status']='PHASE_I_EXECUTION_BLOCKED';ledger['traceback']=traceback.format_exc();state('BLOCKED')
  (PHASE/'reports').mkdir(exist_ok=True);(PHASE/'reports/I_EXECUTION_BLOCKED.md').write_text('PHASE_I_EXECUTION_BLOCKED\n\n'+ledger['traceback'],encoding='utf-8')
  raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True);parser.parse_args();execute()

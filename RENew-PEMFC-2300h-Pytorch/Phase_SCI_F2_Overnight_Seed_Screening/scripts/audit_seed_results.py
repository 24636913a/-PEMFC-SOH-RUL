"""Read-only independent metric/key/scale/checkpoint audit. No optimizer."""
from screen_common import *
from repair_identity import verify_repair
import math
def independent(d):
 error=[float(p)-float(y) for p,y in zip(d.y_pred_delta_V,d.y_true_delta_V)];n=len(error);se=[e*e for e in error];y=[float(x) for x in d.y_true_delta_V];mean=math.fsum(y)/n
 target={}
 for t,e in zip(d.target_row,error):target.setdefault(int(t),[]).append(e*e*1e6)
 end=[e for e,step in zip(error,d.horizon_step) if step==24];ae=sorted(abs(e)*1000 for e in error)
 def quantile(p):
  at=(n-1)*p;lo=int(at);hi=min(lo+1,n-1);return ae[lo]+(ae[hi]-ae[lo])*(at-lo)
 return dict(RMSE_mV=1000*math.sqrt(math.fsum(se)/n),MAE_mV=math.fsum(ae)/n,R2=1-math.fsum(se)/math.fsum((x-mean)**2 for x in y),
  endpoint_RMSE_mV=1000*math.sqrt(math.fsum(e*e for e in end)/len(end)),target_time_equal_RMSE_mV=math.sqrt(math.fsum(math.fsum(a)/len(a) for a in target.values())/len(target)),P95_mV=quantile(.95),P99_mV=quantile(.99),max_abs_error_mV=ae[-1])
def main():
 c=cfg();verify_repair();h.configure_normal(c);d=data();truth=rd(P/'outputs/audits/VALIDATION_KEY_TRUTH.csv');audit=[];horizon=[]
 progress=P/'outputs/execution_ledger/PROGRESS_ENGINEERING_REPAIR.json';state=json.loads(progress.read_text());assert not any(r['status']=='STARTED' for r in state['seeds'].values()),'TRAINING_NOT_TERMINAL'
 scale=json.loads((P/'outputs/audits/SCALER.json').read_text())['target_scale_V']
 initializations=[]
 for seed in SEEDS:
  f=P/f'outputs/metrics/seed_{seed}.json'
  if not f.exists():continue
  assert state['seeds'][str(seed)]['status']=='COMPLETED',('LEDGER_RESULT_STATUS_MISMATCH',seed)
  r=json.loads(f.read_text());assert r['status']=='COMPLETED' and r['seed']==seed
  assert r['predictions']==f'outputs/predictions/F2_seed_{seed}.csv' and r['checkpoint']==f'outputs/checkpoints/F2_seed_{seed}_best.pt','ARTIFACT_PATH_FIREWALL'
  frame=rd(P/r['predictions'])
  assert len(frame)==4920 and not frame.duplicated(['origin_row','target_row','horizon_step']).any()
  assert np.array_equal(frame[truth.columns].to_numpy(),truth.to_numpy()),('EXACT_KEY_TRUTH_MISMATCH',seed)
  assert np.array_equal(frame.target_age_h.to_numpy(),d['val_target_age_h'].ravel())
  assert np.array_equal(frame.origin_raw_voltage_V.to_numpy(),np.repeat(d['val_origin_raw_voltage_V'],24))
  assert np.array_equal(frame.predicted_voltage_V.to_numpy(),frame.origin_raw_voltage_V.to_numpy()+frame.y_pred_delta_V.to_numpy())
  assert np.array_equal(frame.true_label_voltage_V.to_numpy(),frame.origin_raw_voltage_V.to_numpy()+frame.y_true_delta_V.to_numpy())
  assert np.array_equal(frame.true_raw_target_voltage_V.to_numpy(),d['val_target_raw_voltage_V'].ravel())
  calc=independent(frame);maxdiff=max(abs(calc[k]-r[k]) for k in calc);assert maxdiff<1e-10,('METRIC_MISMATCH',seed,maxdiff)
  log=json.loads((P/f'outputs/training_logs/seed_{seed}/epochs.json').read_text());best=math.inf;epoch=0;stale=0
  for row in log:
   if row['validation_standardized_RMSE']<best-1e-12:best=row['validation_standardized_RMSE'];epoch=row['epoch'];stale=0
   else:stale+=1
  assert epoch==r['best_epoch'] and len(log)==r['stop_epoch']
  assert len(log)==450 or stale==40
  assert sha(P/r['checkpoint'])==r['checkpoint_sha256'] and sha(P/r['predictions'])==r['prediction_sha256']
  ck=torch.load(P/r['checkpoint'],map_location='cpu',weights_only=True)
  assert ck['protocol_sha256']==sha(P/'configs/frozen_seed_screening.json') and ck['seed']==seed and ck['best_epoch']==epoch
  assert ck['initialization_sha256']==r['initialization_sha256'];initializations.append(r['initialization_sha256'])
  model=h.model('F2',c['hyperparameters'],torch.device('cuda'));model.load_state_dict(ck['state_dict']);out=h.e.predict(model,d['vx'],32,torch.device('cuda'))*scale
  assert np.array_equal(out.ravel().astype('float64'),frame.y_pred_delta_V.to_numpy()),('SAME_GPU_PREDICTION_REPLAY_MISMATCH',seed,float(abs(out.ravel()-frame.y_pred_delta_V.to_numpy()).max()))
  del model;torch.cuda.empty_cache()
  audit.append(dict(seed=seed,status='PASS',points=len(frame),key_truth_exact=True,max_metric_difference=maxdiff,same_GPU_replay_exact=True,checkpoint_sha256=r['checkpoint_sha256'],frozen_float32_DeltaV_label_max_rounding_V=float(abs(frame.true_label_voltage_V-frame.true_raw_target_voltage_V).max())))
  for step,g in frame.groupby('horizon_step'):
   ee=[float(p)-float(y) for p,y in zip(g.y_pred_delta_V,g.y_true_delta_V)]
   horizon.append(dict(seed=seed,horizon_step=int(step),horizon_h=float(step*.5),RMSE_mV=1000*math.sqrt(math.fsum(e*e for e in ee)/len(ee)),MAE_mV=1000*math.fsum(abs(e) for e in ee)/len(ee),n=len(g)))
  print(f'AUDIT seed{seed} PASS',flush=True)
 assert len(initializations)==len(set(initializations)),'INDEPENDENT_INITIALIZATIONS_NOT_UNIQUE'
 if not audit:
  js('outputs/audits/INDEPENDENT_AUDIT.json',dict(status='NO_COMPLETED_SEEDS',optimizer_steps=0));return
 cs('outputs/audits/INDEPENDENT_PER_SEED.csv',pd.DataFrame(audit));cs('outputs/F2_HORIZON_METRICS.csv',pd.DataFrame(horizon))
 ranking=rd(P/'outputs/F2_SEED_RANKING.csv');assert ranking.seed.tolist()==[r['seed'] for r in sorted([json.loads(f.read_text()) for f in (P/'outputs/metrics').glob('seed_*.json')],key=lambda r:(r['RMSE_mV'],r['seed']))]
 cfg();verify_repair();js('outputs/audits/INDEPENDENT_AUDIT.json',dict(status='PASS',completed_seeds=len(audit),predictions_checked=len(audit)*4920,all_keys_truth_exact=True,all_GPU_replays_exact=True,max_metric_difference=max(r['max_metric_difference'] for r in audit),optimizer_steps=0,Test=False,Assessment=False))
if __name__=='__main__':main()

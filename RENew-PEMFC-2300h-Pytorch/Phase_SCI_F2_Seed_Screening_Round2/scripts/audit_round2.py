"""Independent math.fsum metrics, exact keys and full checkpoint GPU replay."""
from screen_common import *
from reliable_io import js
import math

def independent(frame):
 e=[float(p)-float(y) for p,y in zip(frame.y_pred_delta_V,frame.y_true_delta_V)];n=len(e);y=[float(x) for x in frame.y_true_delta_V];mu=math.fsum(y)/n
 ae=sorted(abs(x)*1000 for x in e);end=[x for x,s in zip(e,frame.horizon_step) if s==24];first=[x for x,s in zip(e,frame.horizon_step) if s==1];target={}
 for t,x in zip(frame.target_row,e):target.setdefault(int(t),[]).append(x*x*1e6)
 def q(p):
  at=(n-1)*p;lo=int(at);hi=min(lo+1,n-1);return ae[lo]+(ae[hi]-ae[lo])*(at-lo)
 return dict(RMSE_mV=1000*math.sqrt(math.fsum(x*x for x in e)/n),MAE_mV=math.fsum(ae)/n,R2=1-math.fsum(x*x for x in e)/math.fsum((x-mu)**2 for x in y),step1_RMSE_mV=1000*math.sqrt(math.fsum(x*x for x in first)/len(first)),endpoint_RMSE_mV=1000*math.sqrt(math.fsum(x*x for x in end)/len(end)),target_time_equal_RMSE_mV=math.sqrt(math.fsum(math.fsum(a)/len(a) for a in target.values())/len(target)),P95_mV=q(.95),P99_mV=q(.99),max_abs_error_mV=ae[-1])

def protect_round1():
 table=rd(P/'outputs/audits/ROUND1_PROTECTION.csv')
 for row in table.itertuples():assert sha(REPO/row.path)==row.sha256,('ROUND1_FILE_CHANGED',row.path)
 return len(table)

def main():
 c=cfg();state=json.loads((P/'outputs/execution_ledger/PROGRESS.json').read_text());assert state['status']!='RUNNING'
 assert set(map(int,state['seeds']))==set(SEEDS) and not any(x['status']=='STARTED' for x in state['seeds'].values())
 protected=protect_round1();h.configure_normal(c);d=data();truth=rd(P/'outputs/audits/VALIDATION_KEY_TRUTH.csv');scale=json.loads((P/'outputs/audits/SCALER.json').read_text())['target_scale_V']
 audit=[];horizons=[];init=[];qualified=[]
 for seed in SEEDS:
  f=P/f'outputs/metrics/seed_{seed}.json';entry=state['seeds'][str(seed)]
  if entry['status']!='COMPLETED':assert not f.exists(),('NONCOMPLETED_HAS_FORMAL_METRICS',seed);continue
  r=json.loads(f.read_text());assert r['status']=='COMPLETED' and r['seed']==seed
  assert r['checkpoint']==f'outputs/checkpoints/F2_seed_{seed}_best.pt' and r['predictions']==f'outputs/predictions/F2_seed_{seed}.csv'
  frame=rd(P/r['predictions']);assert len(frame)==4920 and not frame.duplicated(['origin_row','target_row','horizon_step']).any()
  assert np.array_equal(frame[truth.columns].to_numpy(),truth.to_numpy()),('EXACT_TRUTH_KEY_MISMATCH',seed)
  for col,expected in [('target_age_h',d['val_target_age_h'].ravel()),('origin_age_h',np.repeat(d['val_origin_age_h'],24)),('origin_raw_voltage_V',np.repeat(d['val_origin_raw_voltage_V'],24)),('true_raw_target_voltage_V',d['val_target_raw_voltage_V'].ravel())]:assert np.array_equal(frame[col].to_numpy(),expected),(seed,col)
  assert np.array_equal(frame.predicted_voltage_V.to_numpy(),frame.origin_raw_voltage_V.to_numpy()+frame.y_pred_delta_V.to_numpy())
  assert np.array_equal(frame.true_label_voltage_V.to_numpy(),frame.origin_raw_voltage_V.to_numpy()+frame.y_true_delta_V.to_numpy())
  calc=independent(frame);diff=max(abs(calc[k]-r[k]) for k in calc);assert diff<1e-10,(seed,diff)
  rule=qualification(calc);assert rule==qualification(r)
  assert all(entry[k]==v for k,v in rule.items())
  if rule['qualified']:qualified.append(seed)
  epochs=json.loads((P/f'outputs/training_logs/seed_{seed}/epochs.json').read_text());best=math.inf;best_epoch=0;stale=0
  for row in epochs:
   if row['validation_standardized_RMSE']<best-1e-12:best=row['validation_standardized_RMSE'];best_epoch=row['epoch'];stale=0
   else:stale+=1
  assert best_epoch==r['best_epoch'] and len(epochs)==r['stop_epoch']
  assert len(epochs)==450 or stale==40
  assert abs(best*scale*1000-r['RMSE_mV'])<1e-5
  assert sha(P/r['checkpoint'])==r['checkpoint_sha256'] and sha(P/r['predictions'])==r['prediction_sha256']
  ck=torch.load(P/r['checkpoint'],map_location='cpu',weights_only=True)
  assert all(torch.isfinite(v).all() for v in ck['state_dict'].values() if torch.is_tensor(v)),('NONFINITE_CHECKPOINT',seed)
  assert ck['protocol_sha256']==sha(P/'configs/frozen_seed_screening.json') and ck['seed']==seed and ck['best_epoch']==best_epoch
  assert ck['input_fingerprints']==json.loads((P/'outputs/audits/DATA_FINGERPRINTS.json').read_text())
  assert ck['initialization_sha256']==r['initialization_sha256'];init.append(r['initialization_sha256'])
  m=h.model('F2',c['hyperparameters'],torch.device('cuda'));m.load_state_dict(ck['state_dict']);out=h.e.predict(m,d['vx'],32,torch.device('cuda'))*scale
  assert np.array_equal(out.ravel().astype('float64'),frame.y_pred_delta_V.to_numpy()),('GPU_REPLAY_MISMATCH',seed)
  del m;torch.cuda.empty_cache()
  for step,g in frame.groupby('horizon_step'):
   ee=[float(p)-float(y) for p,y in zip(g.y_pred_delta_V,g.y_true_delta_V)]
   row=dict(seed=seed,horizon_step=int(step),horizon_h=step*.5,RMSE_mV=1000*math.sqrt(math.fsum(x*x for x in ee)/len(ee)),MAE_mV=1000*math.fsum(abs(x) for x in ee)/len(ee),n=len(g))
   saved=next(x for x in r['horizon_metrics'] if x['horizon_step']==step)
   assert abs(row['RMSE_mV']-saved['RMSE_mV'])<1e-10 and abs(row['MAE_mV']-saved['MAE_mV'])<1e-10
   horizons.append(row)
  audit.append(dict(seed=seed,status='PASS',points=len(frame),max_metric_difference=diff,exact_keys_truth=True,same_GPU_replay_exact=True,qualification=rule['selection_status'],checkpoint_sha256=r['checkpoint_sha256'],float32_label_rounding_max_V=float(abs(frame.true_label_voltage_V-frame.true_raw_target_voltage_V).max())))
  print(f'AUDIT seed{seed} PASS {rule["selection_status"]}',flush=True)
 assert len(init)==len(set(init)) and qualified==state['qualified_seeds'] and len(qualified)<=3
 if len(qualified)==3:assert all(state['seeds'][str(s)]['status']=='NOT_STARTED' for s in SEEDS if s>qualified[-1])
 # Chronological execution must be a prefix, never skip to a special seed.
 entered=[s for s in SEEDS if state['seeds'][str(s)]['status']!='NOT_STARTED']
 assert entered==SEEDS[:len(entered)]
 historical=rd(P/'outputs/HISTORICAL_THREE_CANDIDATES.csv');historical_audit=[]
 for r in historical.to_dict('records'):
  frame=rd(P/r['predictions']);calc=independent(frame);diff=max(abs(calc[k]-r[k]) for k in calc);assert diff<1e-10
  assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['source_prediction'])==r['source_prediction_sha256']
  ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
  assert all(torch.isfinite(v).all() for v in ck['state_dict'].values() if torch.is_tensor(v)),('NONFINITE_HISTORICAL_CHECKPOINT',r['seed'])
  m=h.model('F2',c['hyperparameters'],torch.device('cuda'));m.load_state_dict(ck['state_dict']);out=h.e.predict(m,d['vx'],32,torch.device('cuda'))*scale
  assert np.array_equal(out.ravel().astype('float64'),frame.y_pred_delta_V.to_numpy())
  del m;torch.cuda.empty_cache();historical_audit.append(dict(seed=r['seed'],status='PASS',max_metric_difference=diff,same_GPU_replay_exact=True))
 if audit:cs('outputs/audits/INDEPENDENT_PER_SEED.csv',pd.DataFrame(audit));cs('outputs/ROUND2_HORIZON_INDEPENDENT.csv',pd.DataFrame(horizons))
 cs('outputs/audits/INDEPENDENT_HISTORICAL.csv',pd.DataFrame(historical_audit));rank=ranking()
 if len(rank):assert rank.seed.tolist()==[r['seed'] for r in sorted([json.loads((P/f'outputs/metrics/seed_{s}.json').read_text()) for s in entered if state['seeds'][str(s)]['status']=='COMPLETED'],key=lambda r:(r['RMSE_mV'],r['seed']))]
 cfg();js('outputs/audits/INDEPENDENT_AUDIT.json',dict(status='PASS',completed_seeds=len(audit),new_prediction_points_checked=len(audit)*4920,historical_points_checked=14760,all_GPU_replays_exact=True,all_keys_truth_exact=True,max_metric_difference=max([r['max_metric_difference'] for r in audit+historical_audit]),round1_files_protected=protected,qualified=qualified,optimizer_steps=0,Test=False,Assessment=False))

if __name__=='__main__':main()

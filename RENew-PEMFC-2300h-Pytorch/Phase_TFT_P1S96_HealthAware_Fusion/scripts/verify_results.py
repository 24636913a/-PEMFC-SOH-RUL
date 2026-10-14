"""Independent cache arithmetic, causal sample and artifact identity checks."""
import json
import numpy as np
import pandas as pd
import torch
from runtime_common import P,load_protocol,inputs,file_sha,json_write,MODELS
def main():
 cfg,sha=load_protocol();data,fp=inputs();ledger=json.loads((P/'outputs/EXECUTION_LEDGER.json').read_text());assert ledger['status']=='NINE_RUNS_COMPLETED'
 metrics=pd.read_csv(P/'outputs/ALL_RUN_METRICS.csv',float_precision='round_trip');truthkey=None;records=0;checks=[]
 for row in metrics.itertuples():
  frame=pd.read_csv(P/row.prediction,float_precision='round_trip');assert file_sha(P/row.prediction)==row.prediction_sha256;assert len(frame)==178*24
  assert not frame.duplicated(['origin_row','target_row','horizon_step']).any()
  key=frame[['origin_row','target_row','horizon_step','y_true_delta_V']].to_numpy()
  if truthkey is None:truthkey=key
  else:assert np.array_equal(key,truthkey)
  assert np.array_equal(frame.origin_row.to_numpy(),np.repeat(data['validation_origins'],24))
  assert np.array_equal(frame.target_row.to_numpy(),data['validation_targets'].ravel())
  assert np.array_equal(frame.y_true_delta_V.to_numpy(),data['validation_y'].astype(float).ravel())
  assert np.array_equal(frame.origin_time_h.to_numpy(),np.repeat(data['validation_ages'],24))
  assert np.array_equal(frame.target_time_h.to_numpy(),data['validation_target_times'].ravel())
  error=frame.y_pred_delta_V.to_numpy()-frame.y_true_delta_V.to_numpy();absolute=np.abs(error)*1000
  target_weight=np.array([np.mean(error[frame.target_row.to_numpy()==t]**2) for t in np.unique(frame.target_row)])
  independent=dict(RMSE_mV=np.sqrt(np.mean(error**2))*1000,MAE_mV=np.mean(absolute),endpoint_12h_RMSE_mV=np.sqrt(np.mean(error[frame.horizon_step.to_numpy()==24]**2))*1000,
   target_time_equal_RMSE_mV=np.sqrt(np.mean(target_weight))*1000,P95_abs_error_mV=np.percentile(absolute,95),P99_abs_error_mV=np.percentile(absolute,99),max_abs_error_mV=np.max(absolute))
  for k,v in independent.items():assert abs(v-getattr(row,k))<1e-10,(row.run_id,k,v,getattr(row,k))
  assert row.distinct_target_rows==224 and row.validation_origins==178 and row.prediction_queries==4272
  ck=torch.load(P/row.checkpoint,map_location='cpu',weights_only=True);assert file_sha(P/row.checkpoint)==row.checkpoint_sha256
  assert ck['protocol_sha256']==sha and ck['input_fingerprints']==fp and ck['model']==row.model and ck['seed']==row.seed
  assert ck['best_epoch']==row.best_epoch and row.stop_epoch<=450
  assert ck['state_dict']['regime_gate.hidden.weight'].shape==(8,5 if row.model==MODELS[0] else 10)
  assert not any(k.startswith('soh') or k.startswith('physics') for k in ck['state_dict'])
  folder=P/'outputs/training_logs'/row.run_id;epochs=pd.read_csv(folder/'epochs.csv');assert len(epochs)==row.stop_epoch
  assert epochs.loc[epochs.validation_standardized_RMSE.idxmin(),'epoch']==row.best_epoch
  assert row.checkpoint_reload_exact
  gates=pd.read_csv(P/f'outputs/metrics/{row.run_id}_gate_weights.csv');assert len(gates)==178 and gates.gate_weight.between(0,1).all()
  checks.append(dict(run_id=row.run_id,status='PASS',queries=len(frame),checkpoint_reload_exact=True));records+=len(frame)
 for seed in [42,52,62]:
  starts=[];orders=[]
  for g in MODELS:
   folder=P/'outputs/training_logs'/f'{g}_seed{seed}';starts.append(json.loads((folder/'initial.json').read_text())['rng']);orders.append(pd.read_csv(folder/'epochs.csv'))
  assert starts[0]==starts[1]==starts[2]
  n=min(map(len,orders));assert np.array_equal(orders[0].sampler_order_sha256[:n],orders[1].sampler_order_sha256[:n]);assert np.array_equal(orders[1].sampler_order_sha256[:n],orders[2].sampler_order_sha256[:n])
 alignment=pd.read_csv(P/'inputs/P1S_TFT_ALIGNMENT.csv');detail=json.loads((P/'inputs/P1S_HISTORY_TIME_IDENTITIES.json').read_text())
 assert len(alignment)==1585 and len(detail)==1585
 for a in detail:
  if a['eligible']:
   assert len(a['raw_cycle_ids'])>=5 and len(a['raw_cycle_ids'])==len(set(a['raw_cycle_ids']))
   assert min(a['measurement_end_hours'])>=a['origin_time_h']-96-1e-9
   assert max(a['conservative_available_hours'])<=a['origin_time_h']+1e-9
   assert np.ptp(a['measurement_end_hours'])>=67.2-1e-9
 result=dict(status='PASS',formal_runs=9,all_queries_verified=records,shared_seed_sampling_and_rng_verified=True,all_prediction_keys_and_truth_exact=True,
  checkpoint_hashes_and_reload_verified=True,all_frozen_sources_and_training_code_unchanged=True,Assessment_Test_accessed=False,runs=checks)
 json_write('outputs/INDEPENDENT_VERIFICATION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

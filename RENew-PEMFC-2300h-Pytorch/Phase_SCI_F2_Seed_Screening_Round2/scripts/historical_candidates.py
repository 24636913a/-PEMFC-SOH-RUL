"""Read-only historical 42/103/193 validation audit; no optimizer or fitting."""
from screen_common import *
from reliable_io import js

def build_historical(c,d,scale):
 rows=[];horizons=[];truth=rd(P/'outputs/audits/VALIDATION_KEY_TRUTH.csv')
 for seed in [42,103,193]:
  stage=H if seed==42 else OLD
  prediction=stage/(f'outputs/validation_predictions/H-F2-Seed{seed}.csv' if seed==42 else f'outputs/predictions/F2_seed_{seed}.csv')
  checkpoint=stage/(f'outputs/checkpoints/H-F2-Seed{seed}_best.pt' if seed==42 else f'outputs/checkpoints/F2_seed_{seed}_best.pt')
  frame=rd(prediction);assert len(frame)==4920 and np.array_equal(frame[truth.columns].to_numpy(),truth.to_numpy())
  ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
  assert ck['seed']==seed and ck['input_fingerprints']==json.loads((P/'outputs/audits/DATA_FINGERPRINTS.json').read_text())
  assert ck['target_scale_V']==scale
  expected_protocol=c['historical_protocol_sha256'] if seed==42 else sha(OLD/'configs/frozen_seed_screening.json')
  assert ck['protocol_sha256']==expected_protocol
  model=h.model('F2',c['hyperparameters'],torch.device('cuda'));model.load_state_dict(ck['state_dict']);out=h.e.predict(model,d['vx'],32,torch.device('cuda'))*scale
  assert np.array_equal(out.ravel().astype('float64'),frame.y_pred_delta_V.to_numpy()),('HISTORICAL_GPU_REPLAY_MISMATCH',seed)
  del model;torch.cuda.empty_cache()
  # This new canonical cache adds time/raw-voltage identities, leaving originals unchanged.
  frame['origin_age_h']=np.repeat(d['val_origin_age_h'],24);frame['target_age_h']=d['val_target_age_h'].ravel();frame['origin_raw_voltage_V']=np.repeat(d['val_origin_raw_voltage_V'],24)
  frame['true_raw_target_voltage_V']=d['val_target_raw_voltage_V'].ravel();frame['true_label_voltage_V']=frame.origin_raw_voltage_V+frame.y_true_delta_V;frame['predicted_voltage_V']=frame.origin_raw_voltage_V+frame.y_pred_delta_V
  dest=f'outputs/audits/historical_seed_{seed}_canonical.csv';cs(dest,frame)
  met=metric(frame);ee=frame.loc[frame.horizon_step.eq(1),'y_pred_delta_V'].to_numpy()-frame.loc[frame.horizon_step.eq(1),'y_true_delta_V'].to_numpy()
  rows.append(dict(seed=seed,source='HISTORICAL_SELECTED',**met,**qualification(met),step1_RMSE_mV=float(1000*np.sqrt(np.mean(ee**2))),best_epoch=ck['best_epoch'],source_prediction=str(prediction),source_prediction_sha256=sha(prediction),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),predictions=dest,prediction_sha256=sha(P/dest),protocol_sha256=expected_protocol,same_GPU_replay_exact=True))
  for step,g in frame.groupby('horizon_step'):
   e=g.y_pred_delta_V.to_numpy()-g.y_true_delta_V.to_numpy();horizons.append(dict(seed=seed,horizon_step=int(step),horizon_h=step*.5,RMSE_mV=float(1000*np.sqrt(np.mean(e**2))),MAE_mV=float(1000*np.mean(abs(e))),n=len(g)))
 cs('outputs/HISTORICAL_THREE_CANDIDATES.csv',pd.DataFrame(rows));cs('outputs/HISTORICAL_HORIZON_METRICS.csv',pd.DataFrame(horizons))
 js('outputs/audits/HISTORICAL_REPLAY.json',dict(status='PASS',seeds=[42,103,193],points=14760,exact_key_truth=True,all_same_GPU_replays_exact=True,original_files_modified=False,optimizer_steps=0))
 print(pd.DataFrame(rows)[['seed','RMSE_mV','endpoint_RMSE_mV','max_abs_error_mV','selection_status']].to_string(index=False),flush=True)

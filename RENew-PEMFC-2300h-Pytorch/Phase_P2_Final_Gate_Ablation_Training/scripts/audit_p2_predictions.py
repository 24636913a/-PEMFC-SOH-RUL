"""Prediction-query metrics, target SSE robustness and exact cache identity."""
from __future__ import annotations
import numpy as np
import pandas as pd
from p2_common import *

KEYS=['origin_row','target_row','horizon_step']
def metric_bundle(frame):
    y=frame.y_true_delta_V.to_numpy(np.float64);pred=frame.y_pred_delta_V.to_numpy(np.float64)
    if not np.isfinite(y).all() or not np.isfinite(pred).all():raise RuntimeError('NONFINITE_PREDICTION')
    err=(pred-y)*1000.;sq=err**2
    target=frame[KEYS].copy();target['SSE_query_mV2']=sq
    means=target.groupby('target_row',sort=True).SSE_query_mV2.mean()
    totals=target.groupby('target_row',sort=True).SSE_query_mV2.sum().sort_values(ascending=False)
    denominator=float(np.sum((y-y.mean())**2))
    endpoint=frame.horizon_step.to_numpy()==24
    metrics={'RMSE_mV':float(np.sqrt(sq.mean())),'MAE_mV':float(np.abs(err).mean()),
             'R2':float(1-np.sum((pred-y)**2)/denominator),'endpoint_12h_RMSE_mV':float(np.sqrt(sq[endpoint].mean())),
             'P95_abs_error_mV':float(np.quantile(np.abs(err),.95)),'P99_abs_error_mV':float(np.quantile(np.abs(err),.99)),
             'max_abs_error_mV':float(np.abs(err).max()),'target_time_equal_RMSE_mV':float(np.sqrt(means.mean())),
             **{f'top{k}_target_SSE_share':float(totals.head(k).sum()/totals.sum()) for k in (1,5,10)}}
    horizons=[{'horizon_step':h,'forecast_hours':.5*h,'RMSE_mV':float(np.sqrt(sq[frame.horizon_step.to_numpy()==h].mean()))}
              for h in range(1,25)]
    target_details=target.groupby('target_row',sort=True).SSE_query_mV2.agg(['count','sum','mean']).reset_index()
    target_details.columns=['target_row','multiplicity','SSE_mV2','mean_squared_error_mV2']
    return metrics,horizons,target_details

def validate_frame(frame,truth,run_id,group,seed,protocol_sha):
    if len(frame)!=4920 or frame.duplicated(KEYS).any():raise RuntimeError('PREDICTION_RECORD_COUNT_OR_DUPLICATE_KEY')
    for key,value in (('run_id',run_id),('model',group),('seed',seed),('protocol_sha256',protocol_sha)):
        if not frame[key].eq(value).all():raise RuntimeError('PREDICTION_IDENTITY_MISMATCH:'+key)
    if not np.array_equal(frame[KEYS+['y_true_delta_V']].to_numpy(),truth[KEYS+['y_true_delta_V']].to_numpy()):
        raise RuntimeError('PREDICTION_KEYS_OR_TRUTH_MISMATCH')
    if not np.array_equal(frame.target_row.to_numpy(),frame.origin_row.to_numpy()+frame.horizon_step.to_numpy()):
        raise RuntimeError('TARGET_QUERY_IDENTITY_MISMATCH')
    if frame.target_row.max()>=2404 or frame.origin_row.min()<2103:raise RuntimeError('NON_VALIDATION_QUERY')
    if not {2247,2248}.issubset(set(frame.target_row)):raise RuntimeError('EXTREME_TARGET_DELETION')
    return metric_bundle(frame)

def audit_completed(cfg,protocol_sha,require_all=False):
    truth=pd.read_csv(OUT/'audits/P2_VALIDATION_KEY_TRUTH_AUTHORITY.csv',float_precision='round_trip')
    results=[]
    for run_id,group,seed in RUNS:
        result_path=OUT/'training_logs'/run_id/'result.json'
        if not result_path.exists():
            if require_all:raise RuntimeError('MISSING_FORMAL_RUN:'+run_id)
            continue
        result=read_json(result_path)
        if result['status']!='COMPLETED' or result['protocol_sha256']!=protocol_sha or result['code_identity_sha256']!=cfg['code_identity']['sha256']:
            raise RuntimeError('FORMAL_RESULT_IDENTITY_MISMATCH')
        ck=PHASE/result['checkpoint'];pred=PHASE/result['validation_prediction']
        if file_hash(ck)!=result['checkpoint_sha256'] or file_hash(pred)!=result['prediction_sha256']:raise RuntimeError('ARTIFACT_HASH_MISMATCH')
        frame=pd.read_csv(pred,float_precision='round_trip');metrics,horizons,targets=validate_frame(frame,truth,run_id,group,seed,protocol_sha)
        for key,value in metrics.items():
            if not np.isclose(value,result[key],atol=1e-10,rtol=0):raise RuntimeError('METRIC_CACHE_RECALCULATION_MISMATCH:'+key)
        epoch_records=read_json(OUT/'training_logs'/run_id/'epochs.json')
        if len(epoch_records)!=result['stop_epoch'] or not epoch_records:raise RuntimeError('EPOCH_LOG_INCOMPLETE')
        for row in epoch_records:
            expected=read_json(OUT/'audits/sampler_authority'/f'seed{seed}_epoch{row["epoch"]:03d}.json')
            if row['sampler_origin_order_sha256']!=expected['origin_order_sha256']:raise RuntimeError('RECORDED_SAMPLER_MISMATCH')
        gates=pd.read_csv(OUT/'metrics'/f'{run_id}_gate_weights.csv',float_precision='round_trip')
        if len(gates)!=205 or not np.array_equal(gates.origin_row.to_numpy(),frame.origin_row.unique()):raise RuntimeError('GATE_CACHE_INCOMPLETE')
        results.append({'run_id':run_id,'status':'PASS','queries':4920,'origins':205,'target_rows':int(frame.target_row.nunique()),
                        'cache_metric_recalculation':True,'checkpoint_hash':True,'sampler_epochs':len(epoch_records)})
    audit={'status':'PASS','completed_runs':len(results),'require_all_nine':require_all,'runs':results,'Test_accessed':False}
    atomic_json(OUT/'audits/P2_ARTIFACT_INTEGRITY_AUDIT.json',audit)
    return audit

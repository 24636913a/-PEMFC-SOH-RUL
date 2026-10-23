"""Prediction-query metrics, target SSE robustness and exact cache identity."""
from __future__ import annotations
import numpy as np
import pandas as pd
from p3_common import *

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


def require_full450_history(records):
    if [r['epoch'] for r in records]!=list(range(1,451)):raise RuntimeError('FULL450_HISTORY_INCOMPLETE')
    first=None
    for row in records:
        first=first_hypothetical_stop(first,row['epoch'],row['stale_epochs'])
        if row['first_hypothetical_patience40_epoch']!=first:raise RuntimeError('HYPOTHETICAL_STOP_HISTORY_MISMATCH')
    return first

def audit_completed(cfg,protocol_sha,require_all=False):
    truth=pd.read_csv(OUT/'audits/P3_VALIDATION_KEY_TRUTH_AUTHORITY.csv',float_precision='round_trip')
    audits=[]
    for run_id,group,seed in RUNS:
        result_path=OUT/'training_logs'/run_id/'result.json'
        if not result_path.exists():
            if require_all:raise RuntimeError('MISSING_FORMAL_RUN:'+run_id)
            continue
        result=read_json(result_path)
        history=read_json(OUT/'training_logs'/run_id/'epochs.json')
        first=require_full450_history(history)
        if result['status']!='COMPLETED' or result['stop_epoch']!=450:raise RuntimeError('FORMAL_RUN_NOT_450_COMPLETE')
        if result['protocol_sha256']!=protocol_sha or result['code_identity_sha256']!=cfg['code_identity']['sha256']:raise RuntimeError('FORMAL_RESULT_IDENTITY_MISMATCH')
        if {r['checkpoint_role'] for r in result['checkpoint_results']}!={'best','last450'} or len(result['checkpoint_results'])!=2:raise RuntimeError('TWO_CHECKPOINT_ROLES_REQUIRED')
        initial=read_json(OUT/'training_logs'/run_id/'initial.json')
        verify_p2_initialization(group,seed,initial['initial_model_sha256'],initial['initial_training_rng'],initial['parameter_identity'])
        for row in history:
            expected=read_json(OUT/'audits/sampler_authority'/f'seed{seed}_epoch{row["epoch"]:03d}.json')
            if row['sampler_origin_order_sha256']!=expected['origin_order_sha256']:raise RuntimeError('RECORDED_SAMPLER_MISMATCH')
        for r in result['checkpoint_results']:
            role=r['checkpoint_role']
            if r['run_id']!=run_id or r['model']!=group or r['seed']!=seed or r['protocol_sha256']!=protocol_sha or r['code_identity_sha256']!=cfg['code_identity']['sha256']:raise RuntimeError('CHECKPOINT_RESULT_IDENTITY_MISMATCH')
            if r['first_hypothetical_patience40_epoch']!=first:raise RuntimeError('STOP_METADATA_MISMATCH')
            expected_epoch=result['best_epoch'] if role=='best' else 450
            if r['selected_epoch']!=expected_epoch:raise RuntimeError('CHECKPOINT_SELECTED_EPOCH_MISMATCH')
            ck=PHASE/r['checkpoint'];pred=PHASE/r['validation_prediction'];backup=PHASE/r['checkpoint_backup']['path']
            if file_hash(ck)!=r['checkpoint_sha256'] or file_hash(pred)!=r['prediction_sha256']:raise RuntimeError('ARTIFACT_HASH_MISMATCH')
            verify_backup_hash(r['checkpoint_sha256'],file_hash(backup))
            frame=pd.read_csv(pred,float_precision='round_trip')
            if not frame.checkpoint_role.eq(role).all():raise RuntimeError('PREDICTION_CHECKPOINT_ROLE_MISMATCH')
            metrics,_,_=validate_frame(frame,truth,run_id,group,seed,protocol_sha)
            for key,value in metrics.items():
                if not np.isclose(value,r[key],atol=1e-10,rtol=0):raise RuntimeError('METRIC_CACHE_RECALCULATION_MISMATCH:'+key)
            gates=pd.read_csv(OUT/'metrics'/f'{run_id}_{role}_gate_weights.csv',float_precision='round_trip')
            if len(gates)!=205 or not np.array_equal(gates.origin_row.to_numpy(),frame.origin_row.unique()):raise RuntimeError('GATE_CACHE_INCOMPLETE')
            audits.append({'run_id':run_id,'role':role,'status':'PASS','queries':4920,'origins':205,'target_rows':int(frame.target_row.nunique()),
                          'metric_recalculation':True,'checkpoint_and_backup_hash':True,'sampler_epochs':450,'P2_initial_identity':True})
    audit={'status':'PASS','completed_runs':len(audits)//2,'checkpoint_roles_audited':len(audits),'all_nine_required':require_all,'runs':audits,'Test_accessed':False}
    atomic_json(OUT/'audits/P3_ARTIFACT_INTEGRITY_AUDIT.json',audit)
    return audit


"""Registered cache-only interface correction; original protocol/code/blocked ledger preserved.

No model construction/forward or original wrapper retry is performed here.
Original H primary table has only RMSE/MAE/R2; other metrics are calculated
from the hash-pinned H Validation CSV rather than guessed or silently skipped.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import j_common as j

MANIFEST=j.PHASE/'configs/J_CACHE_ONLY_COMPLETION_MANIFEST.yaml'

def registered_code():
    paths=sorted((j.PHASE/'cache_completion').glob('*.py'))
    files={p.relative_to(j.REPO).as_posix():j.digest(p) for p in paths}
    files.update(j.own_code()['files'])
    return {'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}

def read_h_validation(path,cfg):
    p=Path(path).resolve();records=cfg['additional_Validation_sources']
    name=p.relative_to(j.REPO).as_posix() if p.is_relative_to(j.REPO) else str(p)
    if name not in records:raise RuntimeError('COMPLETION_SOURCE_NOT_ALLOWLISTED')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=records[name]:raise RuntimeError('H_VALIDATION_CACHE_HASH_MISMATCH')
    return pd.read_csv(p,float_precision='round_trip')

def check_h_reference(current,reference,stored_row):
    keys=['origin_row','target_row','horizon_step']
    a=current.sort_values(['origin_row','horizon_step']).reset_index(drop=True)
    b=reference.sort_values(['origin_row','horizon_step']).reset_index(drop=True)
    if len(b)!=4920 or b.duplicated(keys).any() or not np.array_equal(a[keys].to_numpy(),b[keys].to_numpy()):
        raise RuntimeError('ORIGINAL_H_VALIDATION_KEYS_MISMATCH')
    for key in ['y_true_delta_V','y_pred_delta_V']:
        if not np.array_equal(a[key].to_numpy(),b[key].to_numpy()):raise RuntimeError('ORIGINAL_H_VALIDATION_POINT_MISMATCH:'+key)
    metrics=j.project_metrics(b)
    for key in ['RMSE_mV','MAE_mV','R2']:
        if abs(metrics[key]-float(stored_row[key]))>1e-10:raise RuntimeError('ORIGINAL_H_PRIMARY_METRIC_MISMATCH:'+key)
    return metrics

def freeze():
    cfg=j.load_protocol()
    if MANIFEST.exists():raise RuntimeError('CACHE_COMPLETION_ALREADY_REGISTERED')
    hist=j.read_csv(j.H/'outputs/metrics/H_VALIDATION_METRICS.csv');sources={}
    for seed in j.SEEDS:
        row=hist.loc[hist.run_id.eq(f'H-F2-Seed{seed}')].iloc[0]
        path=(j.H/str(row.validation_prediction)).resolve()
        if path.parent!=(j.H/'outputs/validation_predictions').resolve() or path.name!=f'H-F2-Seed{seed}.csv':raise RuntimeError('H_VALIDATION_SOURCE_IDENTITY_INVALID')
        observed=hashlib.sha256(path.read_bytes()).hexdigest()
        if observed!=row.prediction_file_sha256:raise RuntimeError('H_VALIDATION_CACHE_HASH_MISMATCH')
        sources[path.relative_to(j.REPO).as_posix()]=observed
    value={'scope':'CACHE_ONLY_COMPLETION_NO_MODEL_FORWARD','original_protocol_sha256':j.digest(j.PROTOCOL),
        'original_code_identity_sha256':cfg['code_identity']['sha256'],'complete_execution_identity':registered_code(),
        'additional_Validation_sources':sources,
        'reason':'original H metric table stores RMSE/MAE/R2 only; compare all points and secondary metrics to pinned H Validation caches',
        'unchanged':['model','checkpoint','input','thresholds','metric definitions','decision rules','original protocol','original code','blocked wrapper ledger'],
        'new_training':False,'new_inference':False,'Test_access':False,
        'original_wrapper_status':'BLOCKED_REPORT_INTERFACE_KEYERROR_WITH_SECONDARY_TERMINAL_ENCODING_ERROR',
        'retry_original_wrapper':False,'scientific_decision_after_cache_integrity_only':True}
    j.text_write(MANIFEST,yaml.safe_dump(value,sort_keys=False,allow_unicode=True))
    j.text_write(MANIFEST.with_suffix('.sha256'),j.digest(MANIFEST)+'  J_CACHE_ONLY_COMPLETION_MANIFEST.yaml\n')
    print('Cache-only completion registered:',j.digest(MANIFEST),'complete code:',value['complete_execution_identity']['sha256'])

def load():
    j.load_protocol()
    if j.digest(MANIFEST)!=MANIFEST.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]:raise RuntimeError('CACHE_COMPLETION_MANIFEST_CHANGED')
    cfg=yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))
    if registered_code()!=cfg['complete_execution_identity']:raise RuntimeError('CACHE_COMPLETION_CODE_CHANGED')
    if j.digest(j.PROTOCOL)!=cfg['original_protocol_sha256']:raise RuntimeError('ORIGINAL_J_PROTOCOL_CHANGED')
    return cfg

def complete():
    cfg=load();original=j.load_protocol();data=j.load_inputs()
    truth=j.read_csv(j.OUT/'validation_parity/J_VALIDATION_TRUTH.csv')
    j.validate_keys(data['origins'],data['targets'],truth.target_age_h.to_numpy())
    hist=j.read_csv(j.H/'outputs/metrics/H_VALIDATION_METRICS.csv')
    rows=[];checks=[];hchecks=[];gates_all=[]
    for seed in j.SEEDS:
        for stage in ['parity','full','fixed']:
            token=j.read_json(j.OUT/f'integrity_audit/J_{stage}_Seed{seed}_EXECUTION.json')
            if token['status']!='COMPLETED' or token['protocol_sha256']!=j.digest(j.PROTOCOL):raise RuntimeError('INCOMPLETE_J_INFERENCE_STAGE')
            j.verify_state(j.read_json(j.OUT/f'integrity_audit/J_{stage}_Seed{seed}_STATE_AUDIT.json'))
        parity=j.read_json(j.OUT/f'validation_parity/J_F2_Seed{seed}_FIXED32_PARITY.json')
        restoration=j.read_json(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_RESTORATION_AUDIT.json')
        if not parity['pass'] or not restoration['pass']:raise RuntimeError('PARITY_OR_RESTORATION_FAILED')
        gates=j.read_csv(j.OUT/f'gate_values/J_F2_Seed{seed}_Gates.csv')
        j.validate_gates(gates.g.to_numpy(),gates[[n+'_standardized_mean' for n in j.REGIME_FEATURE_NAMES]].to_numpy(),205)
        if not np.array_equal(gates.origin_row.to_numpy(),data['origins']):raise RuntimeError('GATE_ORIGIN_MISMATCH')
        for key,value in {'seed':seed,'run_id':f'J_F2_Seed{seed}','protocol_sha256':j.digest(j.PROTOCOL),
            'input_fingerprint':original['input_fingerprints']['validation_x_sha256'],
            'checkpoint_sha256':original['checkpoints'][f'H-F2-Seed{seed}']['sha256']}.items():
            if not (gates[key]==value).all():raise RuntimeError('GATE_IDENTITY_MISMATCH:'+key)
        gates_all.append(gates)
        with np.load(j.allowed(j.OUT/f'validation_parity/J_full_Seed{seed}_PREDICTIONS.npz'),allow_pickle=False) as ar:learned=ar['normal'].copy();g=ar['g'].copy()
        if not np.array_equal(g.astype(np.float64),gates.g.to_numpy()):raise RuntimeError('GATE_CSV_NPZ_MISMATCH')
        with np.load(j.allowed(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_FIXED_DIAGNOSTIC.npz'),allow_pickle=False) as ar:fixed=ar['fixed'].copy();restored=ar['restored'].copy()
        if not np.array_equal(restored,learned):raise RuntimeError('RESTORATION_CACHE_MISMATCH')
        for mode,part,pred in [('learned','LEARNED',learned),('fixed_0p5','FIXED0P5',fixed)]:
            frame=j.read_csv(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_{part}_PREDICTIONS.csv')
            j.validate_frame(frame,truth,seed,mode,original)
            if not np.array_equal(frame.y_pred_delta_V.to_numpy(),(pred*data['target_scale_V'].item()).ravel().astype(np.float64)):raise RuntimeError('CSV_NPZ_PREDICTION_MISMATCH')
            metrics=j.project_metrics(frame);independent=j.independent_metrics(frame)
            for key,value in metrics.items():
                delta=abs(value-independent[key])
                if delta>1e-10:raise RuntimeError('INDEPENDENT_METRIC_MISMATCH:'+key)
                checks.append({'seed':seed,'mode':mode,'metric':key,'project':value,'independent':independent[key],'difference':delta,'pass':True})
            if mode=='learned':
                old=hist.loc[hist.run_id.eq(f'H-F2-Seed{seed}')].iloc[0]
                reference=read_h_validation(j.H/str(old.validation_prediction),cfg)
                reference_metrics=check_h_reference(frame,reference,old)
                for key,value in metrics.items():
                    delta=abs(value-reference_metrics[key])
                    if delta>1e-10:raise RuntimeError('H_VALIDATION_CACHE_METRIC_CHANGED')
                    hchecks.append({'seed':seed,'metric':key,'current':value,'original_H_cache':reference_metrics[key],'difference':delta,'all_4920_keys_truth_and_prediction_exact':True,'pass':True})
            rows.append({'seed':seed,'mode':mode,'n_origins':205,'n_queries':4920,**metrics})
    j.csv_write(j.OUT/'fixed_gate_diagnostic/J_VALIDATION_METRICS.csv',pd.DataFrame(rows))
    j.csv_write(j.OUT/'integrity_audit/J_INDEPENDENT_METRIC_AUDIT.csv',pd.DataFrame(checks))
    j.csv_write(j.OUT/'integrity_audit/J_ORIGINAL_H_FULL_VALIDATION_CACHE_AUDIT.csv',pd.DataFrame(hchecks))
    j.csv_write(j.OUT/'gate_values/J_ALL_F2_GATES.csv',pd.concat(gates_all,ignore_index=True))
    proof={'pass':True,'nine_checkpoint_hashes_unchanged':True,'fixed32_exact_H_parity':'3/3',
        'passive_hook_bitwise_parity':'3/3','restored_prediction_bitwise_parity':'3/3','full_205_H_prediction_exact':'3/3',
        'gate_records':615,'CSV_prediction_records':29520,'independent_metric_checks':len(checks),
        'max_independent_metric_difference':max(r['difference'] for r in checks),'parameter_buffer_audits':'9/9 PASS',
        'new_training':False,'optimizer_created':False,'backward_executed':False,'parameters_updated':0,
        'requires_grad_flags_preserved':True,'new_Assessment_inference':False,'Assessment_prediction_cache_read':False,
        'formal_Test_accessed':False,'checkpoint_modified':False,'full_N1_hash_performed':False,'scaler_refitted':False,
        'protocol_sha256':j.digest(j.PROTOCOL),'code_identity_sha256':original['code_identity']['sha256'],
        'cache_completion_manifest_sha256':j.digest(MANIFEST),'complete_execution_identity_sha256':cfg['complete_execution_identity']['sha256'],
        'original_wrapper_blocked_record_preserved':True,'original_inference_repeated':False,'cache_only_completion':True}
    j.json_write(j.OUT/'integrity_audit/J_FINAL_READONLY_AUDIT.json',proof)
    j.json_write(j.OUT/'integrity_audit/J_CACHE_ONLY_COMPLETION_RECORD.json',{'status':'CACHE_ONLY_INTEGRITY_COMPLETED',**proof})
    print('Cache integrity PASS; original 205-origin H predictions exact3/3; independent metrics48/48')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--complete',action='store_true');args=p.parse_args()
    if args.freeze:freeze()
    elif args.complete:complete()

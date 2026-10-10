"""Independent scalar metric recalculation from round-trip persisted CSV only."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
PHASE=Path(__file__).resolve().parents[1];OUT=PHASE/'outputs'

def check_frame(frame,authority,run_id,group,seed,sha):
    if len(frame)!=4920 or len(authority)!=4920 or frame.duplicated(['origin_row','horizon_step','target_row']).any():raise RuntimeError('PREDICTION_RECORD_OR_DUPLICATE_FAILURE')
    if not (frame.run_id==run_id).all() or not (frame.model==group).all() or not (frame.seed==seed).all() or not (frame.protocol_sha256==sha).all():raise RuntimeError('PREDICTION_IDENTITY_FAILURE')
    if not np.isfinite(frame[['y_true_delta_V','y_pred_delta_V']].to_numpy()).all():raise RuntimeError('PREDICTION_NONFINITE')
    if not (frame.target_row==frame.origin_row+frame.horizon_step).all() or not frame.target_row.between(2103,2403).all():raise RuntimeError('TARGET_BOUNDARY_FAILURE')
    if frame.origin_row.nunique()!=205 or not (frame.groupby('origin_row').horizon_step.nunique()==24).all() or set(frame.horizon_step)!=set(range(1,25)):
        raise RuntimeError('HORIZON_COVERAGE_FAILURE')
    cols=['origin_row','target_row','horizon_step','y_true_delta_V'];order=['origin_row','horizon_step']
    if not np.array_equal(frame.sort_values(order)[cols].to_numpy(),authority.sort_values(order)[cols].to_numpy()):raise RuntimeError('EXACT_KEY_TRUTH_FAILURE')
    y=frame.y_true_delta_V.to_numpy(float);p=frame.y_pred_delta_V.to_numpy(float);errors=p-y;n=len(y)
    sse=math.fsum(float(v)*float(v) for v in errors);mae=math.fsum(abs(float(v)) for v in errors)/n
    mean=math.fsum(float(v) for v in y)/n;sst=math.fsum((float(v)-mean)**2 for v in y)
    return {'RMSE_mV':math.sqrt(sse/n)*1000,'MAE_mV':mae*1000,'R2':1-sse/sst}

def audit():
    sha=hashlib.sha256((PHASE/'configs/H_FROZEN_PROTOCOL.yaml').read_bytes()).hexdigest()
    if sha!=(PHASE/'configs/H_FROZEN_PROTOCOL.sha256').read_text().split()[0]:raise RuntimeError('AUDIT_PROTOCOL_HASH_FAILURE')
    authority=pd.read_csv(OUT/'audits/H_VALIDATION_KEY_TRUTH_AUTHORITY.csv',float_precision='round_trip')
    ledger=json.loads((OUT/'audits/H_EXECUTION_LEDGER.json').read_text(encoding='utf-8'));rows=[]
    cfg=yaml.safe_load((PHASE/'configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
    planned={r['run_id']:r for r in cfg['formal_runs']}
    for name,entry in ledger['runs'].items():
        if entry['status']!='COMPLETED':continue
        result=json.loads((OUT/'training_logs'/name/'result.json').read_text(encoding='utf-8'))
        if result['status']!='COMPLETED':raise RuntimeError('RESULT_STATUS_FAILURE')
        if name not in planned or result['model']!=planned[name]['model'] or result['seed']!=planned[name]['seed'] or result['protocol_sha256']!=sha or result['code_identity_sha256']!=cfg['code_identity']['sha256']:
            raise RuntimeError('RESULT_FROZEN_IDENTITY_FAILURE')
        path=PHASE/result['validation_prediction'];frame=pd.read_csv(path,float_precision='round_trip')
        metrics=check_frame(frame,authority,name,result['model'],result['seed'],sha)
        for k,v in metrics.items():
            if abs(v-result[k])>1e-10:raise RuntimeError('INDEPENDENT_METRIC_RECALCULATION_FAILURE:'+k)
        rows.append({**result,'independent_RMSE_mV':metrics['RMSE_mV'],'independent_MAE_mV':metrics['MAE_mV'],'independent_R2':metrics['R2'],'records':4920,
            'prediction_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    target=OUT/'metrics';target.mkdir(exist_ok=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='checkpoint_reload'} for r in rows]).to_csv(target/'H_VALIDATION_METRICS.csv',index=False)
    (OUT/'audits/H_INDEPENDENT_PREDICTION_AUDIT.json').write_text(json.dumps({'pass':True,'runs':len(rows),'method':'scalar math.fsum from round-trip CSV',
        'records_per_run':4920,'exact_keys_and_truth':True,'metric_tolerance':1e-10,'test_accessed':False},indent=2)+'\n',encoding='utf-8')
    return rows

if __name__=='__main__':print(json.dumps({'audited_runs':len(audit()),'pass':True}))

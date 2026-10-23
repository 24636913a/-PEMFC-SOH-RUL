"""Post-execution identity, state, key, CSV/NPZ and independent metric audit."""
import numpy as np
import pandas as pd
import j_common as j

def audit():
    cfg=j.load_protocol();j.static_audit();data=j.load_inputs()
    truth=j.read_csv(j.OUT/'validation_parity/J_VALIDATION_TRUTH.csv')
    j.validate_keys(data['origins'],data['targets'],truth.target_age_h.to_numpy())
    rows=[];cachechecks=[];allg=[]
    historic=j.read_csv(j.H/'outputs/metrics/H_VALIDATION_METRICS.csv')
    for seed in j.SEEDS:
        for stage in ['parity','full','fixed']:
            if j.read_json(j.OUT/f'integrity_audit/J_{stage}_Seed{seed}_EXECUTION.json')['status']!='COMPLETED': raise RuntimeError('INCOMPLETE_J_STAGE')
            j.verify_state(j.read_json(j.OUT/f'integrity_audit/J_{stage}_Seed{seed}_STATE_AUDIT.json'))
        if not j.read_json(j.OUT/f'validation_parity/J_F2_Seed{seed}_FIXED32_PARITY.json')['pass']: raise RuntimeError('PARITY_NOT_PASSED')
        if not j.read_json(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_RESTORATION_AUDIT.json')['pass']: raise RuntimeError('RESTORATION_NOT_PASSED')
        gates=j.read_csv(j.OUT/f'gate_values/J_F2_Seed{seed}_Gates.csv')
        j.validate_gates(gates.g.to_numpy(),gates[[n+'_standardized_mean' for n in j.REGIME_FEATURE_NAMES]].to_numpy(),205)
        if not np.array_equal(gates.origin_row.to_numpy(),data['origins']): raise RuntimeError('GATE_ORIGIN_MISMATCH')
        for key,value in {'seed':seed,'run_id':f'J_F2_Seed{seed}','protocol_sha256':j.digest(j.PROTOCOL),
            'input_fingerprint':cfg['input_fingerprints']['validation_x_sha256'],
            'checkpoint_sha256':cfg['checkpoints'][f'H-F2-Seed{seed}']['sha256']}.items():
            if not (gates[key]==value).all(): raise RuntimeError('GATE_IDENTITY_MISMATCH:'+key)
        allg.append(gates)
        with np.load(j.allowed(j.OUT/f'validation_parity/J_full_Seed{seed}_PREDICTIONS.npz'),allow_pickle=False) as ar: learned=ar['normal'].copy();g=ar['g'].copy()
        if not np.array_equal(g.astype(np.float64),gates.g.to_numpy()): raise RuntimeError('GATE_CSV_NPZ_MISMATCH')
        with np.load(j.allowed(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_FIXED_DIAGNOSTIC.npz'),allow_pickle=False) as ar: fixed=ar['fixed'].copy();restored=ar['restored'].copy()
        if not np.array_equal(restored,learned): raise RuntimeError('RESTORATION_CACHE_MISMATCH')
        for mode,filepart,prediction in [('learned','LEARNED',learned),('fixed_0p5','FIXED0P5',fixed)]:
            frame=j.read_csv(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_{filepart}_PREDICTIONS.csv')
            j.validate_frame(frame,truth,seed,mode,cfg)
            if not np.array_equal(frame.y_pred_delta_V.to_numpy(),(prediction*data['target_scale_V'].item()).ravel().astype(np.float64)): raise RuntimeError('CSV_NPZ_PREDICTION_MISMATCH')
            metrics=j.project_metrics(frame); independent=j.independent_metrics(frame)
            for key,value in metrics.items():
                delta=abs(value-independent[key])
                if delta>1e-10: raise RuntimeError('INDEPENDENT_METRIC_AUDIT_MISMATCH:'+key)
                cachechecks.append({'seed':seed,'mode':mode,'metric':key,'project':value,'independent':independent[key],'difference':delta,'pass':True})
            if mode=='learned':
                old=historic.loc[historic.run_id.eq(f'H-F2-Seed{seed}')].iloc[0]
                for key,value in metrics.items():
                    if abs(value-float(old[key]))>1e-10: raise RuntimeError('H_FULL_VALIDATION_METRIC_CHANGED:'+key)
            rows.append({'seed':seed,'mode':mode,'n_origins':205,'n_queries':4920,**metrics})
    frame=pd.DataFrame(rows)
    j.csv_write(j.OUT/'fixed_gate_diagnostic/J_VALIDATION_METRICS.csv',frame)
    j.csv_write(j.OUT/'integrity_audit/J_INDEPENDENT_METRIC_AUDIT.csv',pd.DataFrame(cachechecks))
    j.csv_write(j.OUT/'gate_values/J_ALL_F2_GATES.csv',pd.concat(allg,ignore_index=True))
    proof={'pass':True,'nine_checkpoint_hashes_unchanged':True,'fixed32_exact_H_parity':'3/3',
        'passive_hook_bitwise_parity':'3/3','restored_prediction_bitwise_parity':'3/3','gate_records':615,
        'CSV_prediction_records':29520,'independent_metric_checks':len(cachechecks),
        'max_independent_metric_difference':max(r['difference'] for r in cachechecks),'parameter_buffer_audits':'9/9 PASS',
        'new_training':False,'optimizer_created':False,'backward_executed':False,'parameters_updated':0,
        'requires_grad_flags_preserved':True,'new_Assessment_inference':False,'Assessment_prediction_cache_read':False,
        'formal_Test_accessed':False,'checkpoint_modified':False,'full_N1_hash_performed':False,'scaler_refitted':False,
        'protocol_sha256':j.digest(j.PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256']}
    j.json_write(j.OUT/'integrity_audit/J_FINAL_READONLY_AUDIT.json',proof)
    print('J readonly and independent cache audit PASS:',proof)
    return proof

if __name__=='__main__':audit()

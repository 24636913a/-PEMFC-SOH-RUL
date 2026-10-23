"""Fresh worker: fixed32 parity or full Validation passive-gate capture."""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import j_common as j

def execute(seed,stage):
    cfg=j.load_protocol();data=j.load_inputs()
    if stage=='full':
        for s in j.SEEDS:
            p=j.read_json(j.OUT/f'validation_parity/J_F2_Seed{s}_FIXED32_PARITY.json')
            if not p['pass']: raise RuntimeError('ALL_THREE_EXACT_PARITIES_REQUIRED')
    token=j.OUT/f'integrity_audit/J_{stage}_Seed{seed}_EXECUTION.json'
    if token.exists(): raise RuntimeError('J_INFERENCE_STAGE_ALREADY_ENTERED')
    j.json_write(token,{'status':'STARTED','seed':seed,'stage':stage,'pid':os.getpid(),'protocol_sha256':j.digest(j.PROTOCOL)})
    try:
        with j.no_training_guard():
            model,ck,env=j.load_cuda_model(seed,cfg)
            before=j.ir.snapshot(model);flags={k:p.requires_grad for k,p in model.named_parameters()}
            x=data['x'][:32] if stage=='parity' else data['x']
            normal=j.e.predict(model,x,32,torch.device('cuda'))
            if stage=='parity':
                saved=ck['fixed_validation_prediction'].numpy()
                result=j.ir.difference_statistics(normal,saved,float(data['target_scale_V']))
                j.json_write(j.OUT/f'validation_parity/J_F2_Seed{seed}_NOHOOK_PARITY.json',result)
                if not result['array_equal']: raise RuntimeError('EXACT_H_VALIDATION_PARITY_FAILED')
            hooked,g,z=j.passive_predict(model,x,torch.device('cuda'))
            hook_exact=bool(np.array_equal(normal,hooked))
            if not hook_exact: raise RuntimeError('PASSIVE_HOOK_CHANGED_PREDICTION')
            context_expected=x[:,:,j.INDICES].astype(np.float64).mean(1)
            if not np.allclose(z,context_expected,atol=cfg['numeric_tolerances']['standardized_context_atol'],rtol=1e-6):
                raise RuntimeError('HISTORICAL_CONTEXT_AGGREGATION_MISMATCH')
            raw_inverse=z.astype(np.float64)*data['scaler_scale'][j.INDICES]+data['scaler_mean'][j.INDICES]
            raw=data['raw_context'][:len(x)]
            if not np.allclose(raw_inverse,raw,atol=cfg['numeric_tolerances']['raw_context_inverse_atol'],rtol=cfg['numeric_tolerances']['raw_context_inverse_rtol']):
                raise RuntimeError('PHYSICAL_CONTEXT_INVERSE_MISMATCH')
            state=j.ir.state_audit(model,before,flags);j.verify_state(state)
            if not j.hooks_empty(model): raise RuntimeError('PASSIVE_HOOK_NOT_REMOVED')
            j.json_write(j.OUT/f'integrity_audit/J_{stage}_Seed{seed}_STATE_AUDIT.json',state)
            j.npz_write(j.OUT/f'validation_parity/J_{stage}_Seed{seed}_PREDICTIONS.npz',normal=normal,hooked=hooked,g=g,context=z)
            record={'pass':True,'seed':seed,'stage':stage,'origins':len(x),'no_hook_exact_H_reference':stage=='parity',
                'passive_hook_exact':True,'max_context_difference_standardized':float(np.max(np.abs(z-context_expected))),
                'max_context_inverse_difference_raw':float(np.max(np.abs(raw_inverse-raw))),
                'state_dict_unchanged':True,'requires_grad_unchanged':True,'all_grad_None':True,'hooks_removed':True,'environment':env}
            if stage=='parity':
                j.json_write(j.OUT/f'validation_parity/J_F2_Seed{seed}_FIXED32_PARITY.json',record)
            else:
                frame=j.prediction_frame(data,normal,seed,'learned',cfg)
                j.csv_write(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_LEARNED_PREDICTIONS.csv',frame)
                gates=pd.DataFrame({'run_id':f'J_F2_Seed{seed}','seed':seed,'origin_row':data['origins'],
                    'origin_age_h':data['ages'],'g':g,'checkpoint_sha256':cfg['checkpoints'][f'H-F2-Seed{seed}']['sha256'],
                    'input_fingerprint':cfg['input_fingerprints']['validation_x_sha256'],'protocol_sha256':j.digest(j.PROTOCOL)})
                for k,name in enumerate(j.REGIME_FEATURE_NAMES):
                    gates[name+'_standardized_mean']=z[:,k];gates[name+'_raw_mean']=raw[:,k]
                    gates[name+'_inverse_raw_mean']=raw_inverse[:,k]
                j.csv_write(j.OUT/f'gate_values/J_F2_Seed{seed}_Gates.csv',gates)
                j.json_write(j.OUT/f'gate_values/J_F2_Seed{seed}_CAPTURE_AUDIT.json',record)
        j.json_write(token,{'status':'COMPLETED','seed':seed,'stage':stage,'pid':os.getpid(),'protocol_sha256':j.digest(j.PROTOCOL),'result':record})
        print('J',stage,'seed',seed,'PASS; origins',len(x),'state unchanged')
    except Exception as exc:
        j.json_write(token,{'status':'BLOCKED','seed':seed,'stage':stage,'exception':repr(exc),'pid':os.getpid()})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=j.SEEDS,required=True);p.add_argument('--stage',choices=['parity','full'],required=True)
    a=p.parse_args();execute(a.seed,a.stage)

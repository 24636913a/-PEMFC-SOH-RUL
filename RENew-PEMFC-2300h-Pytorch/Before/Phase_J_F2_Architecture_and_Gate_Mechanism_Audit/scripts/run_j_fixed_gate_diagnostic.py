"""Only intervention allowed: temporarily return g=.5, then verify restoration."""
import argparse
import os
import numpy as np
import torch
import j_common as j

def execute(seed):
    cfg=j.load_protocol();data=j.load_inputs()
    for s in j.SEEDS:
        if not j.read_json(j.OUT/f'gate_values/J_F2_Seed{s}_CAPTURE_AUDIT.json')['pass']:
            raise RuntimeError('ALL_PASSIVE_CAPTURES_REQUIRED')
    token=j.OUT/f'integrity_audit/J_fixed_Seed{seed}_EXECUTION.json'
    if token.exists(): raise RuntimeError('J_FIXED_DIAGNOSTIC_ALREADY_ENTERED')
    j.json_write(token,{'status':'STARTED','seed':seed,'pid':os.getpid(),'protocol_sha256':j.digest(j.PROTOCOL)})
    try:
        with j.no_training_guard():
            model,_,env=j.load_cuda_model(seed,cfg)
            before=j.ir.snapshot(model);flags={k:p.requires_grad for k,p in model.named_parameters()}
            with np.load(j.allowed(j.OUT/f'validation_parity/J_full_Seed{seed}_PREDICTIONS.npz'),allow_pickle=False) as ar:
                normal=ar['normal'].copy()
            fixed=j.fixed_predict(model,data['x'],torch.device('cuda'))
            if not j.hooks_empty(model): raise RuntimeError('FIXED_HOOK_NOT_REMOVED')
            restored=j.e.predict(model,data['x'],32,torch.device('cuda'))
            if not np.array_equal(restored,normal): raise RuntimeError('NORMAL_PREDICTION_NOT_EXACTLY_RESTORED')
            state=j.ir.state_audit(model,before,flags);j.verify_state(state)
            j.json_write(j.OUT/f'integrity_audit/J_fixed_Seed{seed}_STATE_AUDIT.json',state)
            j.npz_write(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_FIXED_DIAGNOSTIC.npz',fixed=fixed,restored=restored)
            j.csv_write(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_FIXED0P5_PREDICTIONS.csv',j.prediction_frame(data,fixed,seed,'fixed_0p5',cfg))
            result={'pass':True,'seed':seed,'origins':205,'state_dict_unchanged':True,'requires_grad_unchanged':True,
                'all_grad_None':True,'hooks_removed':True,'normal_prediction_exactly_restored':True,'fixed_gate':.5,
                'environment':env,'new_training':False,'new_checkpoint_saved':False}
            j.json_write(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_RESTORATION_AUDIT.json',result)
        j.json_write(token,{'status':'COMPLETED','seed':seed,'pid':os.getpid(),'result':result,'protocol_sha256':j.digest(j.PROTOCOL)})
        print('J fixed 0.5 seed',seed,'PASS; restored bitwise exact; state unchanged')
    except Exception as exc:
        j.json_write(token,{'status':'BLOCKED','seed':seed,'exception':repr(exc),'pid':os.getpid()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=j.SEEDS,required=True);execute(p.parse_args().seed)

"""Read-only confirmation that each native F2 path changed during fresh training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from audit_data import HERE,csv,dump
from train_longterm import new_model,state_sha

def run():
    root=Path(json.loads((HERE/'outputs/ACTIVE_PILOT.json').read_text(encoding='utf-8'))['path'])
    rows=[]
    for d in sorted(root.glob('*_seed202_*h')):
        info=json.loads((d/'COMPLETED.json').read_text(encoding='utf-8'))
        model=new_model(info['model'],info['horizon_steps'],202)
        assert state_sha(model)==info['initial_state_sha256']
        initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
        saved=torch.load(d/'best.pt',map_location='cpu',weights_only=False)['state_dict']
        groups=['history_branch.','regime_gate.','lstm_encoder.','lstm_decoder.','OTHER'] if info['model']=='F2' else ['lstm.','head.']
        for group in groups:
            names=[k for k in initial if (not any(k.startswith(p) for p in groups if p!='OTHER'))] if group=='OTHER' else [k for k in initial if k.startswith(group)]
            count=sum(initial[k].numel() for k in names)
            diff=sum(float(torch.sum((saved[k].double()-initial[k].double())**2)) for k in names)
            changed=sum(int(torch.count_nonzero(saved[k]!=initial[k])) for k in names)
            assert count>0 and changed>0
            rows.append(dict(model=info['model'],seed=202,horizon_h=info['horizon_steps']*.5,path_group=group,
                parameters=count,changed_parameter_elements=changed,parameter_delta_L2=np.sqrt(diff)))
    csv(root/'INDEPENDENT_PARAMETER_UPDATES.csv',pd.DataFrame(rows))
    dump(root/'PARAMETER_UPDATE_AUDIT.json',dict(PASS=True,initial_states_reconstructed_exactly=True,
        all_registered_path_groups_changed=True,backward_calls=0,optimizer_updates=0,
        interpretation='Parameter participation evidence, not a causal gain ablation'))
    print('Read-only parameter participation PASS')

if __name__=='__main__':run()

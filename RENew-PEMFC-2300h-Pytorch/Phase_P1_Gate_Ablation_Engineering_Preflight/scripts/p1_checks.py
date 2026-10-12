"""Actual synthetic model/RNG/reload checks. All results are technical only."""
import json
from pathlib import Path
import numpy as np
import torch

import gate_ablation_models as a


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def check_parameter_identity(cohort):
    counts={};core_states={};gate_states={};flags={}
    for group,model in cohort.items():
        counts[group]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        require(counts[group]=={'G0':332379,'G1':332436,'G2':332436}[group],'PARAMETER_COUNT_MISMATCH')
        require(all(p.requires_grad for p in model.parameters()),'REQUIRES_GRAD_CHANGED')
        core,gate=a.split_state(model);core_states[group]=core;gate_states[group]=gate
        flags[group]={k:v.requires_grad for k,v in model.named_parameters()}
    for group in ('G1','G2'):
        require(set(core_states[group])==set(core_states['G0']),'CORE_KEY_SET_MISMATCH')
        for key,v in core_states['G0'].items():
            require(v.shape==core_states[group][key].shape and torch.equal(v,core_states[group][key]),'CORE_INIT_MISMATCH:'+key)
    require(not gate_states['G0'],'G0_DUMMY_GATE_PRESENT')
    require(set(gate_states['G1'])==set(gate_states['G2']),'GATE_SCHEMA_MISMATCH')
    require(sum(v.numel() for v in gate_states['G1'].values())==57,'GATE_COUNT_MISMATCH')
    require(all(torch.equal(v,gate_states['G2'][k]) for k,v in gate_states['G1'].items()),'GATE_INIT_MISMATCH')
    require({k:v for k,v in flags['G1'].items() if k.startswith('regime_gate.')}=={k:v for k,v in flags['G2'].items() if k.startswith('regime_gate.')},'GATE_TRAINABLE_FLAGS_MISMATCH')
    return {'counts':counts,'core_sha256':{g:a.state_digest(s) for g,s in core_states.items()},
            'gate_sha256':{g:a.state_digest(s) for g,s in gate_states.items()},'all_original_requires_grad_unchanged':True}


def check_seed(seed,config,protocol_sha,destination):
    device=torch.device('cuda');a.set_seed(seed)
    caller_rng=a.rng_digest(a.rng_snapshot())
    cohort=a.build_cohort(seed,config,device)
    require(a.rng_digest(a.rng_snapshot())==caller_rng,'CONSTRUCTION_CHANGED_CALLER_RNG')
    identity=check_parameter_identity(cohort)
    reverse=a.build_cohort(seed,config,device,order=('G2','G1','G0'))
    reverse_identity=check_parameter_identity(reverse)
    require(identity['core_sha256']==reverse_identity['core_sha256'] and identity['gate_sha256']==reverse_identity['gate_sha256'],'CONSTRUCTION_ORDER_CHANGED_INIT')
    require(a.rng_digest(a.rng_snapshot())==caller_rng,'REVERSE_CONSTRUCTION_CHANGED_RNG')
    del reverse
    states={};orders={};dropout={}
    for group in ('G0','G1','G2'):
        a.set_seed(seed)
        _,generator=a.make_index_loader(seed)
        states[group]=a.rng_digest(a.rng_snapshot(),generator)
        orders[group]=a.sampler_two_epochs(seed)
        # A dropout probe under saved/restored RNG, no model fitting.
        saved=a.rng_snapshot()
        probe=torch.nn.functional.dropout(torch.ones(128,device=device),p=.1,training=True)
        dropout[group]=a.tensor_digest(probe)
        a.restore_rng(saved)
    require(states['G0']==states['G1']==states['G2'],'PRETRAIN_RNG_MISMATCH')
    require(orders['G0']==orders['G1']==orders['G2'],'SAMPLER_ORDER_MISMATCH')
    require(dropout['G0']==dropout['G1']==dropout['G2'],'DROPOUT_RNG_MISMATCH')
    x=a.synthetic_input();encoder=cohort['G2'].build_input(torch.from_numpy(x[:4]).to(device))['encoder_cont']
    contexts={g:cohort[g].regime_context(encoder).detach().cpu() for g in ('G1','G2')}
    require(torch.equal(contexts['G1'],torch.ones_like(contexts['G1'])),'G1_NOT_CONSTANT_CONTEXT')
    require(not torch.equal(contexts['G2'][0],contexts['G2'][1]),'G2_CONTEXT_NOT_HISTORY_DEPENDENT')
    expected=encoder[:,:,cohort['G2'].regime_indices].mean(1).detach().cpu()
    require(torch.equal(contexts['G2'],expected),'G2_CONTEXT_AGGREGATION_CHANGED')
    for group in ('G1','G2'):
        gate=cohort[group].regime_weight(encoder).detach()
        require(torch.equal(gate,torch.full_like(gate,.5)),'INITIAL_GATE_NOT_HALF')
    predictions={};reloads={};model_checks={}
    for group in ('G0','G1','G2'):
        model=cohort[group]
        before=a.state_digest(model.state_dict());flags={k:p.requires_grad for k,p in model.named_parameters()}
        prediction=a.predict_synthetic(model,x,device)
        require(prediction.shape==(65,24) and prediction.dtype==np.float32 and np.isfinite(prediction).all(),'SYNTHETIC_OUTPUT_CONTRACT')
        require(not model.training,'H_PREDICT_DID_NOT_SET_EVAL')
        require(flags=={k:p.requires_grad for k,p in model.named_parameters()},'PREDICT_CHANGED_REQUIRES_GRAD')
        path=destination/f'seed{seed}'/f'{group}_synthetic_initial.pt'
        file_sha=a.save_synthetic_checkpoint(path,model,group,seed,protocol_sha)
        reloaded=a.reload_synthetic_checkpoint(path,group,seed,config,device,protocol_sha)
        pred_reload=a.predict_synthetic(reloaded,x,device)
        require(np.array_equal(prediction,pred_reload),'SYNTHETIC_RELOAD_PREDICTION_NOT_EXACT')
        require(before==a.state_digest(model.state_dict()),'PREDICT_OR_SAVE_CHANGED_STATE')
        predictions[group]=prediction
        reloads[group]={'path':str(path.relative_to(a.PHASE)),'sha256':file_sha,'exact_prediction_match':True,
                        'prediction_sha256':a.digest_bytes(prediction.tobytes()),'max_abs_difference':0.,'batch_size':32,'rows':65,'horizon':24}
        # Non-optimizing train-mode backward. No optimizer is constructed.
        a.set_seed(seed);model.train();model.zero_grad(set_to_none=True)
        output=model.predict_delta(torch.from_numpy(x[:2]).to(device))
        loss=torch.nn.functional.mse_loss(output,torch.zeros_like(output));loss.backward()
        gradients={k:p.grad for k,p in model.named_parameters() if p.grad is not None}
        require(gradients and all(torch.isfinite(g).all() for g in gradients.values()),'SYNTHETIC_GRADIENT_NONFINITE')
        require(any(float(g.abs().sum())>0 for k,g in gradients.items() if not k.startswith('regime_gate.')),'CORE_HAS_NO_GRADIENT')
        if group!='G0':
            require('regime_gate.output.weight' in gradients and float(gradients['regime_gate.output.weight'].abs().sum())>0,'GATE_OUTPUT_HAS_NO_GRADIENT')
        require(before==a.state_digest(model.state_dict()),'BACKWARD_UPDATED_MODEL_STATE')
        model_checks[group]={'finite_nonzero_core_gradient':True,'gate_data_gradient_present':group!='G0',
                             'optimizer_steps':0,'state_unchanged_after_backward':True}
        model.zero_grad(set_to_none=True)
        del reloaded
    differences={g:float(np.max(np.abs(predictions[g]-predictions['G0']))) for g in ('G1','G2')}
    require(all(v<=1e-7 for v in differences.values()),'INITIAL_PREDICTION_PARITY_FAIL')
    # Test the different context contracts beyond their initially-zero output.
    for group in ('G1','G2'):
        with torch.no_grad():
            cohort[group].regime_gate.hidden.weight.fill_(.1);cohort[group].regime_gate.hidden.bias.zero_()
            cohort[group].regime_gate.output.weight.fill_(.2);cohort[group].regime_gate.output.bias.zero_()
    altered=encoder.clone();altered[0,:,cohort['G2'].regime_indices]=0.;altered[1,:,cohort['G2'].regime_indices]=1.
    g1=cohort['G1'].regime_weight(altered).detach();g2=cohort['G2'].regime_weight(altered).detach()
    require(torch.equal(g1,torch.full_like(g1,g1[0].item())),'G1_NOT_GLOBAL_SHARED')
    require(not torch.equal(g2[0],g2[1]),'G2_NOT_RESPONSIVE_TO_SYNTHETIC_CONTEXT')
    np.savez(destination/f'seed{seed}_synthetic_predictions.npz',**predictions)
    result={'seed':seed,'parameter_identity':identity,'construction_order_independent':True,
            'training_start_rng_sha256':states,'sampler':orders,'dropout_probe_sha256':dropout,
            'initial_gate_half':True,'G1_global_context_only':True,'G2_original_history_context':True,
            'initial_prediction_max_abs_difference':differences,
            'initial_prediction_exact':{g:bool(np.array_equal(predictions[g],predictions['G0'])) for g in ('G1','G2')},
            'reload':reloads,'gradients':model_checks,'synthetic_only':True}
    del cohort
    return result

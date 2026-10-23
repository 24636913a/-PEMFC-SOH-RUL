"""Inspect real module schemas/counts and prepare a frozen Validation audit."""
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import yaml
import j_common as j

def architecture(hc):
    rows=[]; shapes=[]
    for name in ['F0','F1','F2']:
        model=j.h.model(name,hc['hyperparameters'],torch.device('cpu'))
        rows.append({'model':name,'class':type(model).__name__,'parameters':sum(p.numel() for p in model.parameters()),
            'branch_parameters':sum(p.numel() for k,p in model.named_parameters() if k.startswith('history_branch.')),
            'gate_parameters':sum(p.numel() for k,p in model.named_parameters() if k.startswith('regime_gate.'))})
        shapes += [{'model':name,'name':k,'shape':str(tuple(p.shape)),'parameters':p.numel()} for k,p in model.named_parameters()]
        if model.decoder_variables!=['relative_forecast_step']: raise RuntimeError('DECODER_FIREWALL_CHANGED')
    if [r['parameters'] for r in rows]!=[323483,332379,332436]: raise RuntimeError('ARCHITECTURE_COUNT_MISMATCH')
    if rows[1]['branch_parameters']!=8896 or rows[2]['gate_parameters']!=57: raise RuntimeError('MODULE_COUNT_MISMATCH')
    f2=j.h.model('F2',hc['hyperparameters'],torch.device('cpu'))
    if list(f2.regime_indices)!=j.INDICES or not set(j.INDICES).issubset(hc['preprocessing']['Top10_indices']): raise RuntimeError('GATE_CONTEXT_SCHEMA_MISMATCH')
    branch=f2.history_branch
    convs=[]
    for name in ['short','medium']:
        mod=getattr(branch,name); conv=mod.conv
        convs.append({'branch':name,'kernel_size':conv.kernel_size[0],'dilation':conv.dilation[0],
            'left_padding':mod.left_padding,'right_padding':0,'groups':conv.groups,
            'receptive_field_points':1+(conv.kernel_size[0]-1)*conv.dilation[0],
            'span_h':(conv.kernel_size[0]-1)*conv.dilation[0]*.5})
    j.csv_write(j.OUT/'architecture/J_PARAMETER_COUNTS.csv',pd.DataFrame(rows))
    j.csv_write(j.OUT/'architecture/J_PARAMETER_SCHEMA.csv',pd.DataFrame(shapes))
    j.csv_write(j.OUT/'architecture/J_CAUSAL_CONV_SCHEMA.csv',pd.DataFrame(convs))
    features=hc['task']['raw_f2_feature_order']
    j.csv_write(j.OUT/'architecture/J_GATE_FEATURE_SCHEMA.csv',pd.DataFrame([
        {'feature':features[idx],'raw_f2_index':idx,'Top10_retained':True,'aggregation':'mean of 48 standardized masked historical bins',
            'scaler':'frozen E_COMMON_SCALER, unfiltered Train histories','raw_unit':unit}
        for idx,unit in zip(j.INDICES,['count/bin','fraction','A/cm^2','A','A/cm^2'])]))
    j.json_write(j.OUT/'architecture/J_ARCHITECTURE_AUDIT.json',{'pass':True,'parameters':rows,
        'gate_hidden':[5,8],'gate_output':[8,1],'gate_broadcast':'[B,1,1] shared across history features/time',
        'conv_input':'encoder VSN [B,48,64]','conv_output':'[B,48,64]',
        'fusion_lstm_input':'post-LSTM encoder GLU+AddNorm [B,48,64]',
        'decoder_state_handoff':'original LSTM encoder h/c; fusion does not change h/c',
        'decoder_input':'deterministic relative step only; 14 physical placeholders zero and excluded from decoder VSN',
        'convolution_activation':'none before dropout/residual/LayerNorm','new_architecture':False})

def freeze():
    if j.PROTOCOL.exists(): raise RuntimeError('J_PROTOCOL_ALREADY_FROZEN')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=j.REPO,text=True).strip()
    if head!=j.BASE: raise RuntimeError('BASE_COMMIT_CHANGED')
    hc,ip=j.historical_identity(); architecture(hc)
    manifests={}
    reference={row['run_id']:row for row in ip['checkpoints']}
    if len(reference)!=9 or set(reference)!={row['run_id'] for row in j.i.RUNS}: raise RuntimeError('I_CHECKPOINT_AUTHORITY_SCHEMA_MISMATCH')
    for spec in j.i.RUNS:
        _,_,manifest=j.i.verify_checkpoint(spec,reference[spec['run_id']])
        manifests[spec['H_run_id']]=manifest
    j.json_write(j.OUT/'checkpoint_identity/J_NINE_CHECKPOINT_IDENTITY.json',manifests)
    dev,access=j.read_dev(); ecfg,_=j.e.load_protocol(); scaler=j.i.frozen_scaler()
    train=j.c.build_windows(dev,ecfg,0,2103); val=j.c.build_windows(dev,ecfg,2103,2404)
    if len(train.origins)!=1380: raise RuntimeError('TRAIN_CONTEXT_COUNT_MISMATCH')
    x=j.e.prep(val.x,scaler,hc['preprocessing']['Top10_indices'],'EWMA')
    j.validate_keys(val.origins,val.targets,dev.bin_end_ophrs2_h.iloc[val.targets.ravel()].to_numpy())
    fp={'validation_x_sha256':j.h.array_hash(x),'validation_y_sha256':j.h.array_hash(val.y),
        'validation_origins_sha256':j.h.array_hash(val.origins),'validation_target_rows_sha256':j.h.array_hash(val.targets),
        'scaler_sha256':j.h.array_hash(np.stack([scaler.mean,scaler.scale])),
        'development_prefix_fingerprint_sha256':j.hashlib.sha256(dev.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()).hexdigest()}
    for manifest in manifests.values():
        for key,value in fp.items():
            if manifest['input_fingerprints'][key]!=value: raise RuntimeError('H_INPUT_IDENTITY_MISMATCH:'+key)
    with np.load(j.allowed(j.IR/'outputs/input_identity/IR_FIXED_VALIDATION_INPUT.npz'),allow_pickle=False) as fixed:
        if not np.array_equal(fixed['x'],x[:32]): raise RuntimeError('IR_FIXED_INPUT_IDENTITY_MISMATCH')
    raw_context=val.x[:,:,j.INDICES].astype(np.float64).mean(1)
    train_context=train.x[:,:,j.INDICES].astype(np.float64).mean(1)
    thresholds=[]
    for k,name in enumerate(j.REGIME_FEATURE_NAMES):
        q1,q2=np.quantile(train_context[:,k],[1/3,2/3])
        thresholds.append({'feature':name,'raw_f2_index':j.INDICES[k],'low_upper':float(q1),'high_lower':float(q2),
            'source':'1380 legal Train historical-window means','ties':'LOW <= q1; MID q1 < x <= q2; HIGH > q2',
            'degenerate_quantiles':bool(q1==q2)})
    j.csv_write(j.OUT/'gate_diagnostics/J_TRAIN_ONLY_CONDITION_THRESHOLDS.csv',pd.DataFrame(thresholds))
    j.npz_write(j.INPUTS,x=x,y=val.y,origins=val.origins,targets=val.targets,ages=val.ages,
        raw_context=raw_context,train_context=train_context,scaler_mean=scaler.mean,scaler_scale=scaler.scale,target_scale_V=np.asarray(scaler.target_scale))
    truth=pd.DataFrame({'origin_row':np.repeat(val.origins,24),'target_row':val.targets.ravel(),
        'horizon_step':np.tile(np.arange(1,25),205),'origin_age_h':np.repeat(val.ages,24),
        'target_age_h':dev.bin_end_ophrs2_h.iloc[val.targets.ravel()].to_numpy(),'y_true_delta_V':val.y.ravel().astype(np.float64)})
    j.csv_write(j.OUT/'validation_parity/J_VALIDATION_TRUTH.csv',truth)
    j.json_write(j.OUT/'integrity_audit/J_PREFIX_ACCESS_AUDIT.json',{'access':access,'read_nrows':3007,'development_rows':3005,
        'fingerprint_type':'DEVELOPMENT_LOGICAL_PREFIX_ONLY','development_prefix_fingerprint_sha256':fp['development_prefix_fingerprint_sha256'],
        'full_N1_hash':False,'Assessment_windows_constructed':False,'Test_accessed':False,'scaler_refitted':False})
    stat=j.static_audit(); j.json_write(j.OUT/'integrity_audit/J_STATIC_EXECUTION_AUDIT.json',stat)
    sources={p.relative_to(j.REPO).as_posix() if p.is_relative_to(j.REPO) else str(p):j.digest(p) for p in j.historical_paths()}
    prepared={p.relative_to(j.REPO).as_posix():j.digest(p) for p in [j.INPUTS,j.OUT/'validation_parity/J_VALIDATION_TRUTH.csv',j.OUT/'gate_diagnostics/J_TRAIN_ONLY_CONDITION_THRESHOLDS.csv']}
    cfg={'phase':j.PHASE.name,'base_commit':j.BASE,'status':'FROZEN_BEFORE_NEW_VALIDATION_GATE_INFERENCE',
        'code_identity':j.own_code(),'sources':sources,'prepared_inputs':prepared,'checkpoints':manifests,
        'input_fingerprints':fp,'hyperparameters':hc['hyperparameters'],'preprocessing':hc['preprocessing'],
        'scope':{'Train':'condition grouping only; no new fit/feature selection','Validation':[2103,2404],
            'Validation_origins':205,'Assessment_inference':False,'formal_Test_access':False,'training':False,'architecture_modification':False},
        'inference':{'authority':'H Path A: CUDA construction, strict state load, eval, original requires_grad flags, e.predict(batch=32), no_grad',
            'exact_fixed32_parity':'np.array_equal, all 3 F2 before gate analysis','passive_hook_parity':'bitwise exact',
            'per_seed_calls':['fixed32 no-hook','fixed32 passive hook','205 no-hook','205 passive hook','205 fixed g=.5','205 restored learned gate'],
            'Assessment_calls':0,'gradient_updates':0,'fixed_gate_only':.5},
        'descriptive_rules':{'low':'g < 0.1','middle':'0.4 <= g <= 0.6','high':'g > 0.9',
            'dynamic':'sample SD > 0.001 AND range > 0.01; PROJECT_ANALYSIS_DESIGN',
            'prediction_affected':'prediction difference RMSE > 0.001 mV',
            'predictive_effect_supported':'all 3 seeds dynamic, affected and learned-minus-fixed RMSE < -0.001 mV',
            'threshold_purpose':'descriptive classification, never tuning or seed selection',
            'condition_groups':thresholds,'correlations':'Spearman descriptive only; no independence/significance claims'},
        'numeric_tolerances':{'cached_metric_abs':1e-10,'raw_context_inverse_atol':1e-5,'raw_context_inverse_rtol':1e-6,
            'standardized_context_atol':1e-6,'fixed_parity':'EXACT'},
        'limitations':j.LIMITS,'stop_rule':'no training, no Assessment inference, no Test, no automatic next phase'}
    j.text_write(j.PROTOCOL,yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    j.text_write(j.SHA,j.digest(j.PROTOCOL)+'  J_FROZEN_AUDIT_PROTOCOL.yaml\n')
    print('J protocol frozen:',j.digest(j.PROTOCOL),'code:',cfg['code_identity']['sha256'],'checkpoint identities:9/9')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze: freeze()
    else: architecture({'hyperparameters':j.load_protocol()['hyperparameters'],'preprocessing':j.load_protocol()['preprocessing'], 'task':j.h.load_protocol()['task']})

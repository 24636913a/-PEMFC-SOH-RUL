"""Registered eleven-run maximum; fixed Validation gates, no retries."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from h_common import *
from crash_safe_recorder import CrashSafeRecorder
from torch.utils.data import DataLoader,TensorDataset

RUNS=[('H-F0-Repeat-A','F0',42),('H-F0-Repeat-B','F0',42)]+[(f'H-{g}-Seed{s}',g,s) for g in ['F0','F1','F2'] for s in [42,52,62]]

def freeze():
    if PROTOCOL.exists() or (OUT/'audits/H_EXECUTION_LEDGER.json').exists():raise RuntimeError('NO_REFREEZE_OR_RESTART')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=BASE:raise RuntimeError('BASE_COMMIT_CHANGED')
    f=yaml.safe_load((F/'configs/F0_F2_EXECUTION_PROTOCOL.yaml').read_text(encoding='utf-8'))
    runtime=environment();runtime.update(CPU_threads=1,cudnn_deterministic=True,cudnn_benchmark=False,
        torch_deterministic_algorithms=False,warn_only=False,claim='CUDA_STRICT_BITWISE_DETERMINISM_NOT_CLAIMED')
    cfg={'phase':PHASE.name,'base_commit':BASE,'authorization':'USER_EXPLICIT_PHASE_H_FULL_EXECUTION',
        'status':'FROZEN_BEFORE_PROBES_AND_FORMAL_TRAINING','scope':{'train_validation_only':True,'test_access':'PROHIBITED',
            'new_assessment_inference':'PROHIBITED','assessment_windows':'PROHIBITED','architecture_modification':'PROHIBITED',
            'hyperparameter_search':'PROHIBITED','new_backbone':'PROHIBITED'},
        'task':f['task'],'preprocessing':f['preprocessing'],'hyperparameters':f['hyperparameters'],
        'data':{'read_nrows':3007,'filter':'bin_end_ophrs2_h < 1819.997738','development_rows':3005,
            'train_rows':[0,2103],'validation_rows':[2103,2404],'train_windows':1380,'validation_windows':205,
            'full_N1_hash':'PROHIBITED','Top10_selection':'consume frozen Train-derived indices; no new selection'},
        'normal_runtime':runtime,'strict_probes':{'separate_processes':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
            'torch_deterministic_algorithms':True,'warn_only':False,'optimizer_step':False,
            'known_cuda_interpolation_failure_does_not_block_normal_training':True},
        'formal_runs':[{'run_id':n,'model':g,'seed':s} for n,g,s in RUNS],
        'maximum_formal_runs':11,'seeds':[42,52,62],'dataloader':{'shuffle':True,'batch_size':32,'num_workers':0,'generator_seed':'run seed'},
        'early_stopping':'validation_standardized_RMSE < best - 1e-12; patience40; max450',
        'repeat_gate':{'difference':'max(values)-min(values)','threshold_mV':'max(0.5,0.10*mean(values))',
            'two_repeat_failure':'PHASE_H_SAME_SEED_MATERIAL_INSTABILITY',
            'three_repeat_failure':'PHASE_H_F0_SEED42_REPEATABILITY_BLOCKED','failure_stops_all_remaining_runs':True},
        'multi_seed_statistics':{'include':'only nine designated F0/F1/F2 Seed42/52/62 runs',
            'exclude':'Repeat-A/B from n=3','RMSE_std':'sample ddof=1','no_significance_claims':True,
            'paired_difference_direction':'F1-F0; F2-F1; F2-F0; negative means minuend/new model better'},
        'architecture_decision':{'all3_F2_better_F0_and_F1':'F2_VALIDATION_MULTISEED_GAIN_OBSERVED',
            'majority_F2_better_F1_but_not_all3_both':'F2_VS_F1_MULTISEED_SIGNAL_ONLY',
            'otherwise':'F2_STRUCTURAL_GAIN_NOT_STABLE'},
        'prediction_audit':{'records_per_run':4920,'exact_keys_truth':True,'metric_absolute_tolerance':1e-10,
            'checkpoint_reload_fixed_validation_exact':True},
        'crash_recorder':'every batch stage; fsync journal, fsynced temporary state + atomic replace; no unbounded tensor copies',
        'stops':['data/truth/model mismatch','nonfinite loss or gradient','normal CUDA failure or OOM','reload or cache audit failure',
            'repeat threshold exceeded','Test access','interrupted run requires review; no automatic reinitialization'],
        'history_limits':['TIMESNET_FREEZE','BACKBONE_SEARCH_CLOSED','PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS','C1_D0_E0_REPRODUCIBILITY_UNRESOLVED'],
        'code_identity':identity()}
    PROTOCOL.parent.mkdir(parents=True,exist_ok=True)
    # LF bytes frozen identically in working tree and Git; no post-run rewrite.
    with PROTOCOL.open('w',encoding='utf-8',newline='\n') as stream:stream.write(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    digest=file_hash(PROTOCOL)
    with SHA.open('w',encoding='utf-8',newline='\n') as stream:stream.write(digest+'  H_FROZEN_PROTOCOL.yaml\n')
    atomic_json(OUT/'audits/H_PROTOCOL_FREEZE.json',{'protocol_sha256':digest,'code_identity_sha256':cfg['code_identity']['sha256'],
        'frozen_before_real_training':True,'base_commit':BASE,'normal_defaults_explicitly_frozen_before_results':True})
    print('H_PROTOCOL_FROZEN '+digest,flush=True)

def initialization(cfg):
    result={}
    for seed in [42,52,62]:
        models={}
        for group in ['F0','F1','F2']:e.set_seed(seed);models[group]=model(group,cfg['hyperparameters'],torch.device('cpu'))
        for g in ['F1','F2']:
            if not all(torch.equal(v,models[g].state_dict()[k]) for k,v in models['F0'].state_dict().items()):raise RuntimeError('SHARED_NATIVE_INIT_MISMATCH')
        if not all(torch.equal(v,models['F2'].state_dict()[k]) for k,v in models['F1'].state_dict().items()):raise RuntimeError('SHARED_BRANCH_INIT_MISMATCH')
        with torch.no_grad():gate=models['F2'].regime_weight(torch.zeros(2,48,15))
        if not torch.equal(gate,torch.full_like(gate,.5)):raise RuntimeError('F2_INITIAL_GATE_NOT_HALF')
        result[str(seed)]={g:e.state_digest(m) for g,m in models.items()}
    atomic_json(OUT/'audits/H_INITIALIZATION_AUTHORITY.json',{'initialization_sha256':result,'shared_native_equal':True,'shared_branch_equal':True,'F2_gate_half':True,
        'F0_trainable_parameter_count':323483,'F0_state_dict_element_count':sum(v.numel() for v in models['F0'].state_dict().values())})

def preflight(cfg):
    for device in ['cuda','cpu']:
        for kind in ['interpolation','tft']:
            subprocess.run([sys.executable,'-B',str(PHASE/'scripts/probe_determinism.py'),'--strict','--device',device,'--kind',kind],cwd=PHASE,check=True)
            status=read_json(OUT/'compatibility'/f'{device}_strict_{kind}/result.json')['status']
            if device=='cuda' and status not in ['PASS','CUDA_STRICT_DETERMINISM_UNSUPPORTED']:raise RuntimeError('UNEXPECTED_STRICT_CUDA_FAILURE')
            # CPU strict result is saved regardless; normal CUDA is independent.
    tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=PHASE,text=True,capture_output=True)
    atomic_json(OUT/'audits/H_REGRESSION_TESTS.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,'synthetic_only':True})
    if tests.returncode:raise RuntimeError('CRASH_SAFE_OR_FIREWALL_TEST_FAILURE')
    subprocess.run([sys.executable,'-B',str(PHASE/'scripts/probe_determinism.py'),'--smoke'],cwd=PHASE,check=True)
    initialization(cfg)

def checkpoint_reload(path,group,cfg,device,vx,sha,fp):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['protocol_sha256']!=sha or ck['code_identity_sha256']!=cfg['code_identity']['sha256'] or ck['input_fingerprints']!=fp:raise RuntimeError('CHECKPOINT_IDENTITY_MISMATCH')
    reloaded=model(group,cfg['hyperparameters'],device);reloaded.load_state_dict(ck['state_dict'],strict=True)
    actual=e.predict(reloaded,vx[:32],32,device);expected=ck['fixed_validation_prediction'].numpy()
    if not np.array_equal(actual,expected):raise RuntimeError('CHECKPOINT_RELOAD_PARITY_FAILURE')
    return reloaded,{'pass':True,'max_difference':float(np.max(abs(actual-expected))),'file_sha256':file_hash(path)}

def worker(run_id):
    group,seed=next((g,s) for n,g,s in RUNS if n==run_id)
    folder=OUT/'training_logs'/run_id;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run_state.json').exists():raise RuntimeError('RUN_ID_ALREADY_EXISTS_NO_RESTART')
    atomic_json(folder/'run_state.json',{'status':'TRAINING','run_id':run_id,'last_completed_epoch':0})
    try:
        cfg=load_protocol();env=configure_normal(cfg);sha=file_hash(PROTOCOL)
        train,val,scaler,tx,vx,fp=prepare(cfg);atomic_json(folder/'input_audit.json',fp)
        atomic_json(folder/'environment.json',{**env,'seed':seed,'base_commit':BASE,'code_identity_sha256':cfg['code_identity']['sha256']})
        e.set_seed(seed);device=torch.device('cuda');p=cfg['hyperparameters'];m=model(group,p,device)
        init=e.state_digest(m);authority=read_json(OUT/'audits/H_INITIALIZATION_AUTHORITY.json')
        if init!=authority['initialization_sha256'][str(seed)][group]:raise RuntimeError('RUN_INITIALIZATION_MISMATCH')
        opt=torch.optim.AdamW(m.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
        ty,vy=train.y/scaler.target_scale,val.y/scaler.target_scale
        gen=torch.Generator().manual_seed(seed)
        loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty.astype(np.float32)),torch.arange(len(tx))),batch_size=32,shuffle=True,generator=gen,num_workers=0)
        atomic_json(folder/'initial.json',{'initialization_sha256':init,'rng':rng_summary(gen),'seed':seed})
        recorder=CrashSafeRecorder(folder,run_id);best=math.inf;best_epoch=stale=0;history=[]
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter();ckpath=OUT/'checkpoints'/f'{run_id}_best.pt'
        ckpath.parent.mkdir(parents=True,exist_ok=True)
        for epoch in range(1,451):
            t=time.perf_counter();m.train();losses=[];order=[]
            for batch,(xb,yb,ids) in enumerate(loader):
                origins=train.origins[ids.numpy()].tolist();order.extend(origins)
                xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
                recorder.emit('BATCH_STARTED',epoch=epoch,batch=batch,origins=origins,input_sha256=array_hash(xb.cpu().numpy()),label_sha256=array_hash(yb.cpu().numpy()),
                    model_state_sha256=tree_hash(m.state_dict()),rng=rng_summary(gen))
                pred=m.predict_delta(xb);a=pred.detach().cpu().numpy()
                recorder.emit('FORWARD_COMPLETED',output_sha256=array_hash(a),output_shape=list(a.shape),output_mean=float(a.mean()),output_std=float(a.std()))
                loss=torch.nn.functional.mse_loss(pred,yb)
                if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
                recorder.emit('LOSS_COMPUTED',loss=float(loss.detach().cpu()))
                if epoch==1 and batch==0:atomic_json(folder/'first_batch.json',recorder.state)
                recorder.emit('BACKWARD_STARTED')
                try:loss.backward()
                except Exception as exc:recorder.emit('BACKWARD_FAILED',exception=repr(exc),traceback=traceback.format_exc());raise
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
                if not torch.isfinite(norm) or any(not torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None):raise RuntimeError('NONFINITE_GRADIENT')
                recorder.emit('BACKWARD_COMPLETED',gradient_norm=float(norm.detach().cpu()))
                opt.step();recorder.emit('OPTIMIZER_STEP_COMPLETED',parameters_after_sha256=tree_hash(m.state_dict()))
                losses.append(float(loss.detach().cpu()))
            pred=e.predict(m,vx,32,device);rmse=float(np.sqrt(np.mean((pred-vy)**2)))
            if not math.isfinite(rmse):raise RuntimeError('NONFINITE_VALIDATION')
            if rmse<best-1e-12:
                best,best_epoch,stale=rmse,epoch,0
                tmp=ckpath.with_suffix('.pt.writing')
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'run_id':run_id,'model':group,'seed':seed,
                    'protocol_sha256':sha,'code_identity_sha256':cfg['code_identity']['sha256'],'best_epoch':best_epoch,
                    'target_scale_V':scaler.target_scale,'input_fingerprints':fp,'initialization_sha256':init,
                    'fixed_validation_prediction':torch.from_numpy(pred[:32].copy())},tmp);os.replace(tmp,ckpath)
            else:stale+=1
            row={'epoch':epoch,'train_standardized_MSE':float(np.mean(losses)),'validation_standardized_RMSE':rmse,
                'validation_RMSE_mV':rmse*scaler.target_scale*1000,'best_epoch':best_epoch,'stale_epochs':stale,
                'epoch_time_s':time.perf_counter()-t,'sampler_origin_order_sha256':array_hash(np.asarray(order)),
                'rng_after_validation':rng_summary(gen)}
            history.append(row);atomic_json(folder/'epochs.json',history)
            pd.DataFrame([{k:v for k,v in r.items() if k!='rng_after_validation'} for r in history]).to_csv(folder/'epochs.csv',index=False)
            atomic_json(folder/'run_state.json',{'status':'TRAINING','run_id':run_id,'last_completed_epoch':epoch,'best_epoch':best_epoch})
            print(f'{run_id} epoch={epoch} val_RMSE_mV={row["validation_RMSE_mV"]:.6f} best={best_epoch} stale={stale}',flush=True)
            if stale>=40:break
        training_time=time.perf_counter()-started;peak=int(torch.cuda.max_memory_allocated())
        del m,opt
        m,reload=checkpoint_reload(ckpath,group,cfg,device,vx,sha,fp)
        pred=e.predict(m,vx,32,device)*scaler.target_scale
        records=[]
        for i,origin in enumerate(val.origins):
            for h in range(24):records.append({'run_id':run_id,'model':group,'seed':seed,'origin_row':int(origin),'target_row':int(val.targets[i,h]),
                'horizon_step':h+1,'y_true_delta_V':float(val.y[i,h]),'y_pred_delta_V':float(pred[i,h]),'protocol_sha256':sha})
        frame=pd.DataFrame(records);path=OUT/'validation_predictions'/f'{run_id}.csv';path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,float_format='%.17g')
        metrics=e.numpy_metrics(frame)
        result={'run_id':run_id,'model':group,'seed':seed,'role':'REPEATABILITY_ONLY' if 'Repeat' in run_id else 'MULTISEED_PRIMARY',
            **metrics,'best_epoch':best_epoch,'stop_epoch':epoch,'training_time_s':training_time,'parameter_count':COUNTS[group],
            'peak_GPU_memory_bytes':peak,'initialization_sha256':init,'checkpoint_reload':reload,
            'protocol_sha256':sha,'code_identity_sha256':cfg['code_identity']['sha256'],'checkpoint':str(ckpath.relative_to(PHASE)),
            'validation_prediction':str(path.relative_to(PHASE)),'status':'COMPLETED'}
        atomic_json(folder/'result.json',result);atomic_json(folder/'run_state.json',{'status':'COMPLETED','run_id':run_id,'last_completed_epoch':epoch})
        # Query key/truth authority: only Validation, fixed independently of predictions.
        authority_path=OUT/'audits/H_VALIDATION_KEY_TRUTH_AUTHORITY.csv'
        truth=frame[['origin_row','target_row','horizon_step','y_true_delta_V']]
        if not authority_path.exists():truth.to_csv(authority_path,index=False,float_format='%.17g')
        else:
            old=pd.read_csv(authority_path,float_precision='round_trip')
            if not np.array_equal(old.to_numpy(),truth.to_numpy()):raise RuntimeError('VALIDATION_TRUTH_OR_KEYS_CHANGED')
    except Exception as exc:
        atomic_json(folder/'failure.json',{'status':'INTERRUPTED_REQUIRES_REVIEW','exception':repr(exc),'traceback':traceback.format_exc()})
        prior=read_json(folder/'run_state.json');prior.update(status='INTERRUPTED_REQUIRES_REVIEW');atomic_json(folder/'run_state.json',prior)
        raise

def run():
    cfg=load_protocol();ledger_path=OUT/'audits/H_EXECUTION_LEDGER.json'
    if ledger_path.exists():raise RuntimeError('H_RUN_ALREADY_ENTERED_NO_AUTOMATIC_RESTART')
    ledger={'status':'PRECHECKS','runs':{},'formal_runs_started':0,'test_accessed':False,'new_assessment_inference':False}
    atomic_json(ledger_path,ledger)
    try:
        preflight(cfg);ledger['status']='TRAINING';atomic_json(ledger_path,ledger)
        for index,(name,group,seed) in enumerate(RUNS):
            ledger['runs'][name]={'status':'STARTED','seed':seed,'model':group};ledger['formal_runs_started']+=1;atomic_json(ledger_path,ledger)
            folder=OUT/'training_logs'/name;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'console.log').open('w',encoding='utf-8') as log:
                proc=subprocess.run([sys.executable,'-B',__file__,'--worker',name],cwd=PHASE,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:raise RuntimeError('FORMAL_RUN_INTERRUPTED:'+name)
            result=read_json(folder/'result.json');ledger['runs'][name]={'status':'COMPLETED','RMSE_mV':result['RMSE_mV']};atomic_json(ledger_path,ledger)
            from audit_h_predictions import audit
            audit() # Every completed cache checked before eligibility for next run.
            print(name+' completed RMSE_mV='+str(result['RMSE_mV']),flush=True)
            if index in (1,2):
                values=[read_json(OUT/'training_logs'/n/'result.json')['RMSE_mV'] for n,_,_ in RUNS[:index+1]]
                gate=repeat_gate(values);atomic_json(OUT/'repeatability'/f'{index+1}_run_gate.json',gate)
                if not gate['pass']:
                    ledger.update(status='PHASE_H_REPEATABILITY_BLOCKED',stop_reason='PHASE_H_SAME_SEED_MATERIAL_INSTABILITY' if index==1 else 'PHASE_H_F0_SEED42_REPEATABILITY_BLOCKED')
                    atomic_json(ledger_path,ledger);return
        load_protocol();ledger['status']='PHASE_H_MULTISEED_VALIDATION_COMPLETED';atomic_json(ledger_path,ledger)
    except Exception as exc:
        ledger.update(status='PHASE_H_TECHNICAL_BLOCKED',exception=repr(exc),traceback=traceback.format_exc());atomic_json(ledger_path,ledger)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();p=parser.add_mutually_exclusive_group(required=True);p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--worker',choices=[n for n,_,_ in RUNS]);a=parser.parse_args()
    if a.freeze:freeze()
    elif a.run:run()
    else:worker(a.worker)

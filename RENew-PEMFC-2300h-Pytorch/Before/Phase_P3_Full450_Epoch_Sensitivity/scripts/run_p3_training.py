"""Exactly nine serial fresh runs, fail-closed, no restart or hidden retries."""
from __future__ import annotations
import argparse
import math
import time
import traceback
from p3_common import *
from audit_p3_predictions import validate_frame,metric_bundle,audit_completed
from torch.utils.data import DataLoader,TensorDataset

def freeze():
    if PROTOCOL.exists() or (OUT/'audits/P3_EXECUTION_LEDGER.json').exists():raise RuntimeError('NO_REFREEZE_OR_RESTART')
    if git('rev-parse','HEAD').decode().strip()!=BASE:raise RuntimeError('BASE_COMMIT_CHANGED')
    if git('status','--porcelain','--untracked-files=no').decode().strip():raise RuntimeError('TRACKED_WORKTREE_DIRTY')
    h=yaml.safe_load((H/'configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
    if file_hash(H/'configs/H_FROZEN_PROTOCOL.yaml')!=(H/'configs/H_FROZEN_PROTOCOL.sha256').read_text().split()[0]:raise RuntimeError('H_PROTOCOL_SHA_MISMATCH')
    p1=read_json(P1/'protocol/P1_EXECUTION_PROTOCOL_DRAFT.json')
    if file_hash(P1/'protocol/P1_EXECUTION_PROTOCOL_DRAFT.json')!=(P1/'protocol/P1_EXECUTION_PROTOCOL_DRAFT.sha256').read_text().split()[0]:raise RuntimeError('P1_PROTOCOL_SHA_MISMATCH')
    if p1['training_authorized'] is not False or read_json(P1/'outputs/P1_EXECUTION_AUDIT.json')['final_status']!='GATE_TRAINING_TECHNICALLY_READY_FOR_AUTHORIZATION':raise RuntimeError('P1_PREFLIGHT_IDENTITY_MISMATCH')
    p=read_json(PROJECT/'Phase_P_Final_Innovation_and_Experiment_Planning/protocol/P_PROPOSED_GATE_PROTOCOL.json')
    cfg={'phase':PHASE.name,'base_commit':BASE,'authorization':'USER_EXPLICIT_P3_NINE_RUN_TRAINING',
         'training_authorized':True,'maximum_runs':9,'frozen_UTC':utc(),'H':h,
         'P1_protocol_sha256':file_hash(P1/'protocol/P1_EXECUTION_PROTOCOL_DRAFT.json'),
         'groups':p['groups'],'runs':[{'run_id':r,'model':g,'seed':s} for r,g,s in RUNS],
         'initialization':p1['initialization'],'prediction':p1['prediction'],
         'early_stopping':{**p1['early_stopping'],'break_on_patience':False,'record_first_hypothetical_patience40':True},
         'experiment_nature':'POST_P2_EXPLORATORY_EPOCH_SENSITIVITY',
         'required_epochs_each_run':450,'checkpoint_roles':['Best-of-450','Last-450'],
         'P2_authority_sha256':file_hash(P2/'configs/P2_FROZEN_EXECUTION_PROTOCOL.json'),
         'CUDA_strict_bitwise_determinism_claimed':False,'success_criteria':p['success_proposal'],
         'evaluation':p['evaluation'],'scope':{'train_validation_only':True,'Assessment':'PROHIBITED','Test':'PROHIBITED',
            'historical_checkpoint_load':'PROHIBITED','maximum_training_runs':9,'retries':'PROHIBITED',
            'optimizer_search':False,'new_gate_or_backbone':False,'Physics_F2':'PAUSED','SOH_RUL':'NOT_AUTHORIZED'},
         'stop_rules':['source/data/scaler/truth/key mismatch','nonfinite loss/gradient/predictions','CUDA/OOM failure',
                       'checkpoint/reload/cache/artifact failure','common epoch sampler mismatch','interruption: preserve and stop'],
         'checkpoint_policy':{'directory':'outputs/checkpoints','atomic_fsync_save':True,'sole_copy_never_deleted':True,
            'git_large_weight_policy':'local ignored files; all path/hash/backup status recorded; no deletion',
            'backup_status':'VERIFIED_INDEPENDENT_DIRECTORY_SAME_VOLUME_BACKUP_REQUIRED'},
         'code_identity':execution_identity()}
    atomic_json(PROTOCOL,cfg)
    PROTOCOL.with_suffix('.sha256').write_text(file_hash(PROTOCOL)+'  '+PROTOCOL.name+'\n',encoding='utf-8',newline='\n')
    print('P3_PROTOCOL_FROZEN '+file_hash(PROTOCOL),flush=True)

def fresh_model(group,seed,cfg,device):
    a,_,_=legacy_tools();a.set_seed(seed)
    cohort=a.build_cohort(seed,cfg['H'],device)
    from p1_checks import check_parameter_identity
    identity=check_parameter_identity(cohort)
    expected=read_json(P1/'outputs/P1_SYNTHETIC_MODEL_EVIDENCE.json')[str(seed)]['parameter_identity']
    if identity!=expected:raise RuntimeError('P1_FRESH_INITIALIZATION_NOT_REPRODUCED')
    model=cohort[group]
    for other in tuple(cohort):
        if other!=group:del cohort[other]
    return model,identity

def worker(run_id):
    folder=OUT/'training_logs'/run_id
    ledger=read_json(OUT/'audits/P3_EXECUTION_LEDGER.json')
    if ledger.get('active_run')!=run_id or ledger['status']!='TRAINING':raise RuntimeError('WORKER_NOT_AUTHORIZED_BY_P3_LEDGER')
    if (folder/'run_state.json').exists():raise RuntimeError('RUN_ALREADY_STARTED_NO_REINITIALIZATION')
    group,seed=next((g,s) for r,g,s in RUNS if r==run_id)
    atomic_json(folder/'run_state.json',{'status':'STARTED','run_id':run_id,'model':group,'seed':seed,'last_completed_epoch':0,'started_UTC':utc()})
    try:
        cfg,protocol_sha=load_protocol();a,e,array_hash,train,val,scaler,tx,vx,fp=prepare(cfg)
        env,ast_hash=a.configure_runtime(cfg['H']);device=torch.device('cuda');p=cfg['H']['hyperparameters']
        expected=read_json(OUT/'audits/P3_DATA_AUTHORITY.json')
        if fp!=expected:raise RuntimeError('RUN_DATA_IDENTITY_MISMATCH')
        m,identity=fresh_model(group,seed,cfg,device);initial=a.state_digest(m.state_dict())
        opt=torch.optim.AdamW(m.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
        gen=torch.Generator().manual_seed(seed)
        rng=a.rng_digest(a.rng_snapshot(),gen)
        p1rng=read_json(P1/'outputs/P1_SYNTHETIC_MODEL_EVIDENCE.json')[str(seed)]['training_start_rng_sha256'][group]
        if rng!=p1rng:raise RuntimeError('PRETRAIN_CPU_CUDA_LOADER_RNG_MISMATCH')
        verify_p2_initialization(group,seed,initial,rng,identity)
        atomic_json(folder/'initial.json',{'run_id':run_id,'model':group,'seed':seed,'parameter_identity':identity,
            'initial_model_sha256':initial,'initial_training_rng':rng,'protocol_sha256':protocol_sha,
            'code_identity_sha256':cfg['code_identity']['sha256'],'no_historical_warm_start':True})
        atomic_json(folder/'environment.json',{**env,'pure_function_AST_hashes':ast_hash})
        ty,vy=train.y/scaler.target_scale,val.y/scaler.target_scale
        loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty.astype(np.float32)),torch.arange(len(tx))),
                          batch_size=32,shuffle=True,generator=gen,num_workers=0)
        source=ast.parse((H/'scripts/crash_safe_recorder.py').read_text(encoding='utf-8'))
        node=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='CrashSafeRecorder')
        namespace={'Path':Path,'json':json,'os':os,'atomic_json':atomic_json}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(H/'scripts/crash_safe_recorder.py'),'exec'),namespace)
        recorder=namespace['CrashSafeRecorder'](folder,run_id)
        best=math.inf;best_epoch=stale=0;first_stop=None;best_at_first_stop=None;history=[];flags={k:v.requires_grad for k,v in m.named_parameters()}
        ckpath=OUT/'checkpoints'/f'{run_id}_best.pt'
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for epoch in range(1,451):
            epoch_start=time.perf_counter();m.train();losses=[];order=[]
            for batch,(xb,yb,ids) in enumerate(loader):
                origins=train.origins[ids.numpy()].tolist();order.extend(origins)
                xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
                recorder.emit('BATCH_STARTED',epoch=epoch,batch=batch,origins=origins,
                              model_state_sha256=a.state_digest(m.state_dict()),rng=a.rng_digest(a.rng_snapshot(),gen),
                              input_sha256=array_hash(xb.cpu().numpy()),label_sha256=array_hash(yb.cpu().numpy()))
                pred=m.predict_delta(xb)
                if not torch.isfinite(pred).all():raise RuntimeError('NONFINITE_TRAIN_PREDICTION')
                loss=torch.nn.functional.mse_loss(pred,yb)
                if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
                recorder.emit('FORWARD_AND_LOSS_COMPLETED',loss=float(loss.detach().cpu()),output_sha256=array_hash(pred.detach().cpu().numpy()))
                recorder.emit('BACKWARD_STARTED');loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
                if not torch.isfinite(norm) or any(not torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None):raise RuntimeError('NONFINITE_GRADIENT')
                recorder.emit('BACKWARD_COMPLETED',gradient_norm=float(norm.detach().cpu()))
                opt.step();recorder.emit('OPTIMIZER_STEP_COMPLETED',model_after_sha256=a.state_digest(m.state_dict()))
                losses.append(float(loss.detach().cpu()))
            order_sha=validate_common_sampler(seed,epoch,order)
            if sorted(order)!=sorted(train.origins.tolist()):raise RuntimeError('ORIGIN_SAMPLER_NOT_COMPLETE_PERMUTATION')
            pred=a.predict_synthetic(m,vx,device)
            rmse=float(np.sqrt(np.mean((pred-vy)**2)))
            if not math.isfinite(rmse):raise RuntimeError('NONFINITE_VALIDATION')
            if a.early_stop_improves(rmse,best):
                best,best_epoch,stale=rmse,epoch,0
                checkpoint_write(ckpath,{'state_dict':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},
                    'run_id':run_id,'model':group,'seed':seed,'protocol_sha256':protocol_sha,
                    'code_identity_sha256':cfg['code_identity']['sha256'],'input_fingerprints':fp,
                    'initial_model_sha256':initial,'initial_training_rng':rng,'best_epoch':best_epoch,
                    'validation_standardized_RMSE':best,'target_scale_V':scaler.target_scale,
                    'fixed_validation_prediction':torch.from_numpy(pred[:32].copy())})
            else:stale+=1
            row={'epoch':epoch,'train_standardized_MSE':float(np.mean(losses)),
                 'validation_standardized_RMSE':rmse,'validation_RMSE_mV':rmse*scaler.target_scale*1000,
                 'best_epoch':best_epoch,'stale_epochs':stale,'sampler_origin_order_sha256':order_sha,
                 'epoch_time_s':time.perf_counter()-epoch_start,'rng_after_validation':a.rng_digest(a.rng_snapshot(),gen)}
            first_stop_before=first_stop
            first_stop=first_hypothetical_stop(first_stop,epoch,stale)
            if first_stop_before is None and first_stop is not None:
                best_at_first_stop={'best_epoch':best_epoch,'best_standardized_RMSE':best}
            row['first_hypothetical_patience40_epoch']=first_stop
            history.append(row);atomic_json(folder/'epochs.json',history)
            pd.DataFrame([{k:v for k,v in r.items() if k!='rng_after_validation'} for r in history]).to_csv(folder/'epochs.csv',index=False)
            atomic_json(folder/'run_state.json',{'status':'TRAINING','run_id':run_id,'model':group,'seed':seed,
                'last_completed_epoch':epoch,'best_epoch':best_epoch,'stale':stale,'last_UTC':utc()})
            print(f'{run_id} epoch={epoch} RMSE_mV={row["validation_RMSE_mV"]:.6f} best={best_epoch} stale={stale} time_s={row["epoch_time_s"]:.2f}',flush=True)
            # Patience is monitored but NEVER terminates P3.
        lastpath=OUT/'checkpoints'/f'{run_id}_last450.pt'
        checkpoint_write(lastpath,{'state_dict':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},
            'run_id':run_id,'model':group,'seed':seed,'protocol_sha256':protocol_sha,
            'code_identity_sha256':cfg['code_identity']['sha256'],'input_fingerprints':fp,
            'initial_model_sha256':initial,'initial_training_rng':rng,'best_epoch':450,
            'validation_standardized_RMSE':rmse,'target_scale_V':scaler.target_scale,
            'fixed_validation_prediction':torch.from_numpy(pred[:32].copy())})
        training_time=time.perf_counter()-started;peak=int(torch.cuda.max_memory_allocated())
        if flags!={k:v.requires_grad for k,v in m.named_parameters()}:raise RuntimeError('REQUIRES_GRAD_CHANGED')
        del m,opt
        role_results=[]
        for role,current_checkpoint in (('best',ckpath),('last450',lastpath)):
            ckpath=current_checkpoint
            ck=torch.load(ckpath,map_location='cpu',weights_only=True)
            require_checkpoint_identity(ck,run_id,group,seed,protocol_sha,cfg['code_identity']['sha256'],fp)
            with a.isolated_rng(seed):m=a.constructor(group,cfg['H']).to(device)
            m.load_state_dict(ck['state_dict'],strict=True)
            fixed=a.predict_synthetic(m,vx[:32],device)
            if not np.array_equal(fixed,ck['fixed_validation_prediction'].numpy()):raise RuntimeError('CHECKPOINT_RELOAD_NOT_EXACT')
            pred=a.predict_synthetic(m,vx,device)*scaler.target_scale
            if not np.isfinite(pred).all():raise RuntimeError('NONFINITE_FINAL_PREDICTION')
            records=[]
            for i,origin in enumerate(val.origins):
                for h in range(24):records.append({'run_id':run_id,'model':group,'seed':seed,'origin_row':int(origin),
                    'target_row':int(val.targets[i,h]),'horizon_step':h+1,'origin_age_h':float(val.ages[i]),
                    'y_true_delta_V':float(val.y[i,h]),'y_pred_delta_V':float(pred[i,h]),'protocol_sha256':protocol_sha})
            frame=pd.DataFrame(records);frame['checkpoint_role']=role;truth=pd.read_csv(OUT/'audits/P3_VALIDATION_KEY_TRUTH_AUTHORITY.csv',float_precision='round_trip')
            metrics,horizons,targets=validate_frame(frame,truth,run_id,group,seed,protocol_sha)
            inherited=e.numpy_metrics(frame)
            if abs(inherited['RMSE_mV']-metrics['RMSE_mV'])>1e-10 or abs(inherited['MAE_mV']-metrics['MAE_mV'])>1e-10:raise RuntimeError('H_METRIC_SEMANTICS_MISMATCH')
            predpath=OUT/'predictions'/f'{run_id}_{role}.csv';predpath.parent.mkdir(parents=True,exist_ok=True)
            frame.to_csv(predpath,index=False,float_format='%.17g')
            reread=pd.read_csv(predpath,float_precision='round_trip')
            if not np.array_equal(frame[KEY_COLUMNS].to_numpy(),reread[KEY_COLUMNS].to_numpy()):raise RuntimeError('PREDICTION_CSV_ROUND_TRIP_NOT_EXACT')
            validate_frame(reread,truth,run_id,group,seed,protocol_sha)
            pd.DataFrame([{**x,'run_id':run_id,'model':group,'seed':seed} for x in horizons]).to_csv(OUT/'metrics'/f'{run_id}_{role}_horizons.csv',index=False,float_format='%.17g')
            targets.to_csv(OUT/'metrics'/f'{run_id}_{role}_target_SSE.csv',index=False,float_format='%.17g')
            gate_records=[]
            with torch.no_grad():
                for i,origin in enumerate(val.origins):
                    if group=='G0':weight=.5;context=[None]*5
                    else:
                        encoder=m.build_input(torch.from_numpy(vx[i:i+1]).to(device))['encoder_cont']
                        context=m.regime_context(encoder).cpu().numpy()[0].tolist();weight=float(m.regime_weight(encoder).item())
                    gate_records.append({'run_id':run_id,'model':group,'seed':seed,'origin_row':int(origin),'gate_weight':weight,
                                         **{f'context_{j}':v for j,v in enumerate(context)}})
            pd.DataFrame(gate_records).to_csv(OUT/'metrics'/f'{run_id}_{role}_gate_weights.csv',index=False,float_format='%.17g')
            result={'run_id':run_id,'model':group,'seed':seed,**metrics,'status':'COMPLETED','best_epoch':best_epoch,
                'stop_epoch':epoch,'stop_reason':'COMPLETE_450_NO_PATIENCE_BREAK','checkpoint_role':role,'selected_epoch':ck['best_epoch'],
                'first_hypothetical_patience40_epoch':first_stop,'best_at_first_hypothetical_stop':best_at_first_stop,
                'new_best_after_hypothetical_stop':first_stop is not None and best_epoch>first_stop,'training_time_s':training_time,
                'parameter_count':sum(v.numel() for v in m.parameters() if v.requires_grad),'peak_GPU_memory_bytes':peak,
                'initial_model_sha256':initial,'protocol_sha256':protocol_sha,'code_identity_sha256':cfg['code_identity']['sha256'],
                'checkpoint':str(ckpath.relative_to(PHASE)).replace('\\','/'),'checkpoint_sha256':file_hash(ckpath),
                'validation_prediction':str(predpath.relative_to(PHASE)).replace('\\','/'),'prediction_sha256':file_hash(predpath),
                'checkpoint_reload_exact':True,'checkpoint_backup_status':'VERIFIED_SEPARATE_DIRECTORY_SAME_VOLUME','checkpoint_backup':verified_backup(ckpath),
                'Test_accessed':False,'Assessment_accessed':False}
            atomic_json(folder/f'{role}_result.json',result)
            role_results.append(result)
            del m
        atomic_json(folder/'result.json',{**role_results[0],'checkpoint_results':role_results})
        atomic_json(folder/'run_state.json',{'status':'COMPLETED','run_id':run_id,'last_completed_epoch':epoch,'best_epoch':best_epoch,'completed_UTC':utc()})
    except Exception:
        failure={'status':'P3_TECHNICAL_FAILURE','run_id':run_id,'exception':traceback.format_exc(),'stopped_UTC':utc()}
        atomic_json(folder/'failure.json',failure)
        if (folder/'run_state.json').exists():
            state=read_json(folder/'run_state.json');state['status']='P3_TECHNICAL_FAILURE';atomic_json(folder/'run_state.json',state)
        raise
    finally:atomic_json(folder/'data_access_audit.json',runtime_access_audit())

KEY_COLUMNS=['origin_row','target_row','horizon_step','y_true_delta_V','y_pred_delta_V']

def run():
    ledger_path=OUT/'audits/P3_EXECUTION_LEDGER.json'
    if ledger_path.exists():raise RuntimeError('P3_LEDGER_EXISTS_NO_AUTOMATIC_RESTART')
    ledger={'status':'PRECHECKS','runs':{},'formal_runs_started':0,'formal_runs_completed':0,'Test_accessed':False,'Assessment_accessed':False,'created_UTC':utc()}
    atomic_json(ledger_path,ledger)
    try:
        cfg,protocol_sha=load_protocol()
        tests=subprocess.run([sys.executable,'-B',str(PHASE/'scripts/run_p3_tests.py')],cwd=PHASE,capture_output=True,text=True)
        atomic_json(OUT/'audits/P3_EXECUTION_GUARD_TESTS.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,'synthetic_only':True})
        if tests.returncode:raise RuntimeError('P3_GUARD_TEST_FAILURE')
        a,e,array_hash,train,val,scaler,tx,vx,fp=prepare(cfg)
        a.configure_runtime(cfg['H']);atomic_json(OUT/'audits/P3_DATA_AUTHORITY.json',fp)
        for name in ('metrics','predictions','checkpoints','training_logs'):(OUT/name).mkdir(parents=True,exist_ok=True)
        records=[{'origin_row':int(o),'target_row':int(val.targets[i,h]),'horizon_step':h+1,'y_true_delta_V':float(val.y[i,h])}
                 for i,o in enumerate(val.origins) for h in range(24)]
        truthpath=OUT/'audits/P3_VALIDATION_KEY_TRUTH_AUTHORITY.csv'
        pd.DataFrame(records).to_csv(truthpath,index=False,float_format='%.17g')
        ledger.update(status='TRAINING',protocol_sha256=protocol_sha,code_identity_sha256=cfg['code_identity']['sha256']);atomic_json(ledger_path,ledger)
        for run_id,group,seed in RUNS:
            ledger['active_run']=run_id;ledger['formal_runs_started']+=1;ledger['runs'][run_id]={'status':'STARTED','model':group,'seed':seed};atomic_json(ledger_path,ledger)
            folder=OUT/'training_logs'/run_id;folder.mkdir(parents=True,exist_ok=True)
            print('START '+run_id,flush=True)
            with (folder/'console.log').open('w',encoding='utf-8',newline='\n') as console:
                proc=subprocess.run([sys.executable,'-B',__file__,'--worker',run_id],cwd=PHASE,stdout=console,stderr=subprocess.STDOUT)
            if proc.returncode:raise RuntimeError('FORMAL_RUN_FAILED_NO_RETRY:'+run_id)
            result=read_json(folder/'result.json');audit_completed(cfg,protocol_sha)
            ledger['formal_runs_completed']+=1;ledger['runs'][run_id]={'status':'COMPLETED','RMSE_mV':result['RMSE_mV']};atomic_json(ledger_path,ledger)
            print('COMPLETED '+run_id+' RMSE_mV='+str(result['RMSE_mV']),flush=True)
        load_protocol();audit_completed(cfg,protocol_sha,require_all=True)
        from analyze_p3_results import analyze
        analyze(cfg,protocol_sha)
        ledger.update(status='P3_NINE_RUNS_COMPLETED',active_run=None,completed_UTC=utc());atomic_json(ledger_path,ledger)
    except Exception:
        ledger.update(status='P3_TECHNICAL_FAILURE',exception=traceback.format_exc(),stopped_UTC=utc());atomic_json(ledger_path,ledger)
        from analyze_p3_results import technical_failure_report
        technical_failure_report(ledger)
        raise
    finally:atomic_json(OUT/'audits/P3_PARENT_DATA_ACCESS_AUDIT.json',runtime_access_audit())

if __name__=='__main__':
    parser=argparse.ArgumentParser();options=parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--freeze',action='store_true');options.add_argument('--run',action='store_true');options.add_argument('--worker',choices=[r for r,_,_ in RUNS])
    args=parser.parse_args();install_firewall()
    if args.freeze:freeze()
    elif args.run:run()
    else:worker(args.worker)

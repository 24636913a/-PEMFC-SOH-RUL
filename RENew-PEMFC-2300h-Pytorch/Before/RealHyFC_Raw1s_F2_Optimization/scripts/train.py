"""Serial bounded A1..A4/B1/(conditional B2); explicit opt-in only."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import random
import time
import traceback
import numpy as np
import pandas as pd
import torch
from core import (STUDY,OLD,OUT,load_cfg,load_cache,model_factory,model_module,save,sha,
    query_truth,metrics,metric_rows,assert_integrity)

def configure():
    random.seed(202);np.random.seed(202);torch.manual_seed(202);torch.cuda.manual_seed_all(202)
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True

def state_sha(state):
    h=hashlib.sha256()
    for value in state.values():h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def branch_name(name):
    if name.startswith('lstm.'):return 'LSTM_BASELINE'
    if name.startswith('head.'):return 'BASELINE_HEAD'
    for prefix,label in [('native.lstm_encoder.','F2_LSTM_ENCODER'),('native.lstm_decoder.','F2_LSTM_DECODER'),
                         ('native.history_branch.','F2_CAUSAL_CNN'),('native.regime_gate.','F2_REGIME_GATE'),
                         ('native.output_layer.','F2_NATIVE_OUTPUT')]:
        if name.startswith(prefix):return label
    return 'F2_REMAINING_TFT'

def branch_updates(model,initial):
    groups={}
    for name,p in model.named_parameters():
        now=p.detach().cpu();before=initial[name];d=(now-before).double()
        g=groups.setdefault(branch_name(name),dict(parameters=0,changed_elements=0,squared_L2_change=0.))
        g['parameters']+=p.numel();g['changed_elements']+=int(torch.count_nonzero(now!=before))
        g['squared_L2_change']+=float(torch.sum(d*d))
    for g in groups.values():g['L2_change']=g['squared_L2_change']**.5
    return groups

class Windows:
    def __init__(self,a,origins,h,cfg):
        self.origins=np.asarray(origins,dtype=np.int64);self.h=h
        mean=np.array(cfg['scaler']['mean'],np.float32);scale=np.array(cfg['scaler']['scale'],np.float32)
        # CPU-only cache of exact causal inputs; source memmaps remain strictly read-only.
        raw=a['x'][self.origins[:,None]+np.arange(-299,1)]
        self.x=torch.from_numpy(np.ascontiguousarray((raw-mean)/scale))
        self.targets,self.truth,self.event=query_truth(a,self.origins,h,cfg)
        self.y=torch.from_numpy((self.truth/cfg['training']['target_scale_V']).astype(np.float32))
        self.weights=torch.from_numpy(np.where(self.event,3.,1.).astype(np.float32))
        assert torch.isfinite(self.x).all() and torch.isfinite(self.y).all()
    def __len__(self):return len(self.origins)

def predict(model,windows,batch):
    pieces=[];model.eval()
    with torch.inference_mode():
        for left in range(0,len(windows),batch):
            p=model(windows.x[left:left+batch].cuda()).cpu().numpy()
            assert np.isfinite(p).all();pieces.append(p)
    return np.concatenate(pieces).astype(np.float64)

def write_predictions(run,label,kind,h,grid,a,origins,pred,cfg):
    targets,truth,event=query_truth(a,origins,h,cfg)
    origin_ms=np.asarray(a['time_ms'][origins]);target_ms=np.asarray(a['time_ms'][targets])
    assert np.max(np.abs(target_ms-origin_ms[:,None]-np.arange(1,h+1)*1000))<=100
    np.savez_compressed(run/f'predictions_{label}_{grid}.npz',origins=origins,targets=targets,
        origin_time_ms=origin_ms,target_time_ms=target_ms,origin_voltage_V=np.asarray(a['voltage_V'][origins]),
        truth_delta_V=truth,prediction_delta_V=pred,event_query=event,
        target_file_id=np.asarray(a['file_id'][targets]),target_source_row=np.asarray(a['source_row'][targets]))
    return metric_rows(kind,h,grid,origins,targets,truth,pred,event)

def evaluate(run,kind,model,h,a,cfg,primary_pred=None):
    records=[]
    for grid in ['60s','600s']:
        suffix='' if grid=='60s' else '_600s'
        origins=np.load(OUT/f'origins_H{h}{suffix}.npz')['VALIDATION']
        if kind=='Persistence':pred=np.zeros((len(origins),h),dtype=np.float64)
        elif grid=='60s' and primary_pred is not None:pred=primary_pred
        else:pred=predict(model,Windows(a,origins,h,cfg),cfg['training']['batch_size'])*cfg['training']['target_scale_V']
        records+=write_predictions(run,'VALIDATION',kind,h,grid,a,origins,pred,cfg)
    pd.DataFrame(records).to_csv(run/'metrics.csv',index=False)
    return records

def auxiliary_baselines(h,a,cfg):
    run=STUDY/f'runs/Persistence_H{h}'
    if not run.exists():
        run.mkdir(parents=True);evaluate(run,'Persistence',None,h,a,cfg)
    rows=[];kept=np.load(OUT/f'origins_H{h}_600s.npz')['VALIDATION']
    for kind in ['Persistence','LSTM','F2']:
        previous=OLD/f'runs/H{h}_{kind}{"" if kind=="Persistence" else "_Seed202"}'
        z=np.load(previous/'predictions_VALIDATION.npz');index=np.searchsorted(z['origins'],kept)
        assert np.array_equal(z['origins'][index],kept)
        rows+=metric_rows('Historical_'+kind,h,'600s',kept,z['targets'][index],z['truth_delta_V'][index],z['prediction_delta_V'][index],z['event_query'][index])
    pd.DataFrame(rows).to_csv(OUT/f'historical_same_queries_H{h}.csv',index=False)

def run_path(spec):return STUDY/f'runs/{spec["id"]}_{spec["model"]}_H{spec["horizon"]}'

def train_one(run_id,microbatch=None):
    cfg=load_cfg();assert_integrity(cfg)
    assert json.loads((OUT/'review_checkpoint1.json').read_text(encoding='utf-8'))['decision']=='PASS'
    for n,h in cfg['implementation_sha256'].items():assert sha(STUDY/n)==h,n
    specs={s['id']:s for s in cfg['runs']};spec=specs[run_id];run=run_path(spec)
    if run.exists():
        state=json.loads((run/'training_state.json').read_text(encoding='utf-8'))
        assert state['status']=='COMPLETE','Existing partial/failed run requires explicit investigation, not automatic retraining'
        assert state['protocol_sha256']==sha(OUT/'frozen_protocol.json') and sha(run/'best.pt')==state['checkpoint_sha256']
        print('REUSE_COMPLETE_NO_TRAINING',run_id,flush=True);return
    prior=['A1','A2','A3','A4','B1','B2'];position=prior.index(run_id)
    for name in prior[:position]:
        p=run_path(specs[name])/'training_state.json'
        assert p.exists() and json.loads(p.read_text())['status']=='COMPLETE',('Prior run incomplete',name)
    if run_id=='B2':
        gate=json.loads((OUT/'B1_acceptance.json').read_text(encoding='utf-8'))
        assert gate['pass'] and gate['B2_allowed'],'B1 independent acceptance and all performance gates required'
    assert torch.cuda.is_available();configure();h=spec['horizon'];kind=spec['model'];weighted=spec['loss']=='EVENT_WEIGHTED_MSE'
    batch=cfg['training']['batch_size'] if microbatch is None else microbatch
    assert batch in [1,2,4,8,16] and 32%batch==0
    _,a=load_cache();orig=np.load(OUT/f'origins_H{h}.npz')
    train=Windows(a,orig['TRAIN'],h,cfg);val=Windows(a,orig['VALIDATION'],h,cfg)
    auxiliary_baselines(h,a,cfg)
    model=model_factory(kind,h).cuda();initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    init_sha=state_sha(model.state_dict());settings=cfg['training'];yscale=settings['target_scale_V']
    run.mkdir(parents=True)
    manifest=dict(status='RUNNING',id=run_id,model=kind,horizon=h,seed=202,initial_state_sha256=init_sha,
        protocol_sha256=sha(OUT/'frozen_protocol.json'),parameters=sum(p.numel() for p in model.parameters()),
        loss=spec['loss'],microbatch=batch,effective_batch=32,optimizer_steps=0,instance_count=1,
        training_origins=len(train),validation_origins=len(val),data_order='torch.randperm with separate CPU generator Seed202 per epoch')
    save(str((run/'config.json').relative_to(STUDY)),dict(protocol=cfg,run=spec,microbatch=batch))
    save(str((run/'initial_parameter_sha256.json').relative_to(STUDY)),{n:hashlib.sha256(p.numpy().tobytes()).hexdigest() for n,p in initial.items()})
    save(str((run/'training_state.json').relative_to(STUDY)),manifest)
    paired={'B1':'A2','B2':'A4'}
    if run_id in paired:
        comparison=json.loads((run_path(specs[paired[run_id]])/'training_state.json').read_text())
        assert init_sha==comparison['initial_state_sha256'],'Weighted loss must start from same untrained parameters'
    opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    generator=torch.Generator().manual_seed(202);best=float('inf');best_epoch=0;stale=0;steps=0
    log=[];gradient_log=[];started=time.time();torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(1,settings['max_epochs']+1):
            epoch_start=time.time();model.train();order=torch.randperm(len(train),generator=generator)
            numerator_total=0.;denominator_total=0.;train_cuda_ms=0.
            for start in range(0,len(train),32):
                indices=order[start:start+32];opt.zero_grad(set_to_none=True)
                # One global denominator for the full effective batch, including variable event mix.
                denominator=float(train.weights[indices].sum()) if weighted else len(indices)*h
                clock_start=torch.cuda.Event(enable_timing=True);clock_end=torch.cuda.Event(enable_timing=True)
                clock_start.record()
                for offset in range(0,len(indices),batch):
                    ix=indices[offset:offset+batch];xb=train.x[ix].cuda();yb=train.y[ix].cuda()
                    w=train.weights[ix].cuda() if weighted else 1.
                    squared=(model(xb)-yb).square();numerator=(squared*w).sum()
                    assert bool(torch.isfinite(numerator)),'Nonfinite training loss'
                    (numerator/denominator).backward();numerator_total+=float(numerator.detach())
                if start==0 or start+32>=len(train):
                    grouped={}
                    for name,p in model.named_parameters():
                        if p.grad is not None:grouped.setdefault(branch_name(name),[]).append(p.grad.detach().norm().square())
                    for name,values in grouped.items():
                        norm=float(torch.stack(values).sum().sqrt())
                        gradient_log.append(dict(epoch=epoch,step=steps+1,branch=name,preclip_gradient_L2=norm,
                            sample='FIRST_UPDATE' if start==0 else 'LAST_UPDATE'))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm'])
                assert bool(torch.isfinite(norm)),'Nonfinite gradient'
                opt.step();steps+=1;clock_end.record();clock_end.synchronize()
                train_cuda_ms+=clock_start.elapsed_time(clock_end);denominator_total+=denominator
            val_start=torch.cuda.Event(enable_timing=True);val_end=torch.cuda.Event(enable_timing=True)
            val_start.record();pred=predict(model,val,batch)*yscale;val_end.record();val_end.synchronize()
            scores=metrics(val.truth,pred);score=scores['RMSE_V'];assert np.isfinite(score)
            if score<best-1e-12:
                best=score;best_epoch=epoch;stale=0
                torch.save(dict(state_dict={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
                    id=run_id,model=kind,horizon=h,epoch=epoch,seed=202,initial_state_sha256=init_sha,
                    protocol_sha256=manifest['protocol_sha256'],validation_RMSE_V=best),run/'best.pt')
                np.save(run/'best_validation_prediction.npy',pred)
            else:stale+=1
            row=dict(epoch=epoch,train_loss_standardized=numerator_total/denominator_total,
                validation_loss_standardized=(score/yscale)**2,validation_RMSE_V=score,validation_MAE_V=scores['MAE_V'],
                best_epoch=best_epoch,optimizer_steps=steps,epoch_wall_seconds=time.time()-epoch_start,
                training_cuda_seconds=train_cuda_ms/1000,validation_cuda_interval_seconds=val_start.elapsed_time(val_end)/1000,
                GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
            assert row['GPU_peak_reserved_MiB']<=8192
            log.append(row);pd.DataFrame(log).to_csv(run/'training_history.csv',index=False)
            pd.DataFrame(gradient_log).to_csv(run/'gradient_log.csv',index=False)
            manifest.update(optimizer_steps=steps,completed_epoch=epoch,best_epoch=best_epoch)
            save(str((run/'training_state.json').relative_to(STUDY)),manifest)
            print(json.dumps(dict(stage='TRAIN',id=run_id,**row)),flush=True)
            if stale>=settings['patience']:break
        torch.save(dict(state_dict={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},epoch=epoch),run/'final_epoch.pt')
        checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict']);model.cuda()
        reloaded=predict(model,val,batch)*yscale;saved=np.load(run/'best_validation_prediction.npy')
        reload_difference=float(np.max(np.abs(reloaded-saved)));assert reload_difference<=1e-7
        updates=branch_updates(model,initial)
        save(str((run/'branch_parameter_updates.json').relative_to(STUDY)),updates)
        evaluate(run,kind if not weighted else 'F2_EventWeighted',model,h,a,cfg,primary_pred=reloaded)
        manifest.update(status='COMPLETE',stop_epoch=epoch,best_epoch=best_epoch,optimizer_steps=steps,
            best_validation_RMSE_V=best,training_and_evaluation_wall_seconds=time.time()-started,
            training_cuda_seconds=sum(r['training_cuda_seconds'] for r in log),checkpoint_sha256=sha(run/'best.pt'),
            final_epoch_checkpoint_sha256=sha(run/'final_epoch.pt'),reload_max_difference_V=reload_difference,
            GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            all_branch_groups_parameter_updates=all(g['changed_elements']>0 for g in updates.values()))
        save(str((run/'training_state.json').relative_to(STUDY)),manifest)
        print('COMPLETE',run_id,json.dumps(manifest),flush=True)
    except Exception as exc:
        manifest.update(status='FAILED',optimizer_steps=steps,error=repr(exc),traceback=traceback.format_exc())
        save(str((run/'training_state.json').relative_to(STUDY)),manifest);raise
    finally:
        del model,opt;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',choices=['A1','A2','A3','A4','B1','B2'],required=True)
    parser.add_argument('--microbatch',type=int,default=None);args=parser.parse_args()
    train_one(args.run,args.microbatch)

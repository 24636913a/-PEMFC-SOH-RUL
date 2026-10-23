"""Four-instance maximum, fixed Seed202, serial single-GPU direct forecasting."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from audit_raw import STUDY, OUT, save, sha
from freeze_protocol import load_cache, metrics, PARTS
from models import create_model, source_identity

class RawWindows(Dataset):
    def __init__(self,arrays,origins,horizon,config):
        self.a=arrays; self.origins=origins; self.h=horizon
        self.mean=np.array(config['scaler']['mean'],np.float32)
        self.scale=np.array(config['scaler']['scale'],np.float32)
        self.yscale=config['training']['target_scale_V']
    def __len__(self): return len(self.origins)
    def __getitem__(self,i):
        origin=int(self.origins[i]); x=np.array(self.a['x'][origin-299:origin+1],copy=True)
        x=(x-self.mean)/self.scale
        v=self.a['voltage_V']; y=np.array((v[origin+1:origin+1+self.h]-v[origin])/self.yscale,dtype=np.float32)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        return torch.from_numpy(x),torch.from_numpy(y)

def configure(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1); torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True

def predict(model,dataset,batch):
    model.eval(); pieces=[]
    with torch.no_grad():
        for xb,_ in DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0):
            p=model(xb.to('cuda')).detach().cpu().numpy()
            assert np.isfinite(p).all()
            pieces.append(p)
    return np.concatenate(pieces)

def evaluate(kind,model,arrays,origin_sets,horizon,cfg,run):
    records=[]; yscale=cfg['training']['target_scale_V']
    for partition in PARTS:
        origins=origin_sets[partition]; ds=RawWindows(arrays,origins,horizon,cfg)
        pred=np.zeros((len(origins),horizon),np.float64) if kind=='Persistence' else predict(model,ds,cfg['training']['batch_size']).astype(np.float64)*yscale
        targets=origins[:,None]+np.arange(1,horizon+1)
        v=arrays['voltage_V']; truth=v[targets]-v[origins,None]
        target_ms=np.asarray(arrays['time_ms'][targets]); origin_ms=np.asarray(arrays['time_ms'][origins])
        assert np.max(np.abs(target_ms-origin_ms[:,None]-np.arange(1,horizon+1)*1000))<=100
        steps=np.diff(np.column_stack([v[origins],v[targets]]),axis=1)
        event=np.maximum.accumulate(np.abs(steps)>cfg['voltage_event']['threshold_V'],axis=1)
        np.savez_compressed(run/f'predictions_{partition}.npz',
            origins=origins,targets=targets,origin_time_ms=origin_ms,target_time_ms=target_ms,
            origin_voltage_V=np.asarray(v[origins]),truth_delta_V=truth,prediction_delta_V=pred,
            event_query=event,target_file_id=np.asarray(arrays['file_id'][targets]),
            target_source_row=np.asarray(arrays['source_row'][targets]))
        for name,mask in [('ALL',np.ones(truth.shape,bool)),('EVENT',event),('NORMAL',~event)]:
            rec=dict(horizon_seconds=horizon,model=kind,seed=None if kind=='Persistence' else 202,
                partition=partition,stratum=name,origins=len(origins),**metrics(truth[mask],pred[mask]))
            records.append(rec)
        print(json.dumps(dict(stage='EVALUATE',horizon=horizon,model=kind,partition=partition,metrics=records[-3]),ensure_ascii=False),flush=True)
    pd.DataFrame(records).to_csv(run/'metrics.csv',index=False)
    return records

def train_one(kind,arrays,origin_sets,horizon,cfg):
    run=STUDY/f'runs/H{horizon}_{kind}_Seed202'
    previous=None
    if run.exists():
        previous=json.loads((run/'training_state.json').read_text(encoding='utf-8'))
        assert previous['protocol_sha256']==sha(OUT/'frozen_protocol.json')
        if previous['status']=='COMPLETE':
            assert sha(run/'best.pt')==previous['checkpoint_sha256']
            return pd.read_csv(run/'metrics.csv').to_dict('records')
        # A pre-optimizer device-integration failure is not a second learned run.
        assert previous['status']=='FAILED' and previous['optimizer_steps']==0
        failure=run/'zero_step_startup_failure.json'
        assert not failure.exists(), 'Only one documented zero-step compatibility recovery allowed'
        save(str(failure.relative_to(STUDY)),previous)
    else: run.mkdir(parents=True)
    configure(202); model=create_model(kind,24,300,horizon).cuda()
    settings=cfg['training']; batch=settings['batch_size']; accumulate=settings['gradient_accumulation']
    train_ds=RawWindows(arrays,origin_sets['TRAIN'],horizon,cfg)
    val_ds=RawWindows(arrays,origin_sets['VALIDATION'],horizon,cfg)
    loader=DataLoader(train_ds,batch_size=batch,shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(202))
    opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    val_orig=origin_sets['VALIDATION']; v=arrays['voltage_V']
    y_val=v[val_orig[:,None]+np.arange(1,horizon+1)]-v[val_orig,None]
    yscale=settings['target_scale_V']; best=float('inf'); stale=0; best_epoch=0; steps=0
    history=[]; started=time.time(); initial_sha=hashlib.sha256()
    for tensor in model.state_dict().values(): initial_sha.update(tensor.detach().cpu().numpy().tobytes())
    manifest=dict(status='RUNNING',model=kind,seed=202,horizon=horizon,initial_state_sha256=initial_sha.hexdigest(),
        parameters=sum(p.numel() for p in model.parameters()),protocol_sha256=sha(OUT/'frozen_protocol.json'),
        optimizer_steps=0,instance_count=1)
    if previous:
        assert manifest['initial_state_sha256']==previous['initial_state_sha256'], 'Zero-step recovery must preserve initialization'
        manifest['zero_optimizer_step_startup_failures']=1
        manifest['startup_recovery']='device tracking in Lightning submodule; no task/weights/hyperparameter change'
    save(str((run/'training_state.json').relative_to(STUDY)),manifest)
    torch.cuda.reset_peak_memory_stats()
    try:
      for epoch in range(1,settings['max_epochs']+1):
        model.train(); opt.zero_grad(set_to_none=True); total_loss=0.; seen=0; epoch_start=time.time()
        for b,(xb,yb) in enumerate(loader):
            xb=xb.cuda(); yb=yb.cuda(); pred=model(xb)
            loss=torch.nn.functional.mse_loss(pred,yb)
            assert bool(torch.isfinite(loss)), 'Nonfinite loss'
            (loss/accumulate).backward(); total_loss+=float(loss.detach())*len(xb); seen+=len(xb)
            if (b+1)%accumulate==0 or b+1==len(loader):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm'])
                assert bool(torch.isfinite(norm)), 'Nonfinite gradient'
                opt.step(); opt.zero_grad(set_to_none=True); steps+=1
        val_pred=predict(model,val_ds,batch).astype(np.float64)*yscale
        score=metrics(y_val,val_pred)['RMSE_V']; assert np.isfinite(score)
        if score<best-1e-12:
            best=score; stale=0; best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,epoch=epoch,seed=202,model=kind,horizon=horizon,
                protocol_sha256=manifest['protocol_sha256'],validation_rmse_V=best),run/'best.pt')
            np.save(run/'best_validation_prediction.npy',val_pred)
        else: stale+=1
        rec=dict(epoch=epoch,train_standardized_MSE=total_loss/seen,validation_RMSE_V=score,
            best_epoch=best_epoch,optimizer_steps=steps,epoch_seconds=time.time()-epoch_start,
            GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
        history.append(rec); pd.DataFrame(history).to_csv(run/'training_history.csv',index=False)
        manifest.update(completed_epoch=epoch,optimizer_steps=steps)
        save(str((run/'training_state.json').relative_to(STUDY)),manifest)
        print(json.dumps(dict(stage='TRAIN',horizon=horizon,model=kind,**rec)),flush=True)
        if stale>=settings['patience']: break
      checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
      model.load_state_dict(checkpoint['state_dict']); model.cuda()
      reload_prediction=predict(model,val_ds,batch).astype(np.float64)*yscale
      cached=np.load(run/'best_validation_prediction.npy'); reload_diff=float(np.max(np.abs(cached-reload_prediction)))
      assert reload_diff<=1e-7, 'Checkpoint reload prediction mismatch'
      manifest.update(status='COMPLETE',stop_epoch=epoch,best_epoch=best_epoch,
        best_validation_RMSE_V=best,training_seconds=time.time()-started,
        checkpoint_sha256=sha(run/'best.pt'),reload_max_difference_V=reload_diff,
        GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
      records=evaluate(kind,model,arrays,origin_sets,horizon,cfg,run)
      manifest.update(GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                      GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
      save(str((run/'training_state.json').relative_to(STUDY)),manifest)
    except Exception as error:
      manifest.update(status='FAILED',optimizer_steps=steps,error=repr(error),traceback=traceback.format_exc())
      save(str((run/'training_state.json').relative_to(STUDY)),manifest)
      raise
    finally:
      del model,opt; gc.collect(); torch.cuda.empty_cache()
    return records

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--horizon',type=int,choices=[30,120],required=True)
    args=parser.parse_args(); horizon=args.horizon
    cfg=json.loads((OUT/'frozen_protocol.json').read_text(encoding='utf-8'))
    assert torch.cuda.is_available()
    assert cfg['seed']==202 and cfg['max_trained_instances']==4
    assert sha(STUDY.parent/'RealHyFC_F2_Main.py')==cfg['historical_code_sha256']
    assert sha(OUT/f'origins_H{horizon}.npz')==cfg['sampling_hashes'][str(horizon)]
    review=json.loads((OUT/'review_checkpoint1.json').read_text(encoding='utf-8'))
    assert review['decision']=='PASS', 'Independent P0-P2 gate has not passed'
    assert (STUDY/'01_30分钟聚合导致的信息损失分析.md').exists() and (STUDY/'02_秒级突变事件与可预测性研究.md').exists()
    if horizon==120:
      gate=json.loads((OUT/'H30_acceptance.json').read_text(encoding='utf-8'))
      assert gate['pass'], 'H30 protocol/metrics/resource acceptance required before H120'
    origins=np.load(OUT/f'origins_H{horizon}.npz',allow_pickle=False)
    _,arrays=load_cache(); assert 'voltage_V' in arrays
    run=STUDY/f'runs/H{horizon}_Persistence'
    if run.exists():
        assert (run/'metrics.csv').exists(), 'Partial persistence output requires inspection'
        records=pd.read_csv(run/'metrics.csv').to_dict('records')
    else:
        run.mkdir(parents=True)
        records=evaluate('Persistence',None,arrays,origins,horizon,cfg,run)
    for kind in ['LSTM','F2']: records.extend(train_one(kind,arrays,origins,horizon,cfg))
    pd.DataFrame(records).to_csv(OUT/f'metrics_H{horizon}.csv',index=False)
    save(f'outputs/H{horizon}_completion.json',dict(status='COMPLETED',trained_instances=2,seed=202,models=['LSTM','F2'],persistence_fitted=False))
    print('HORIZON_COMPLETED',horizon,flush=True)

if __name__=='__main__': main()

"""Independent eligibility/metrics, original-weight replay and training ledger audit.

No backward, optimizer updates, model choice or new experiments. Replaying
holdout to verify saved predictions is not a second selection/evaluation round.
"""
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from common import HERE,CONFIG,SOURCE,sha,array_sha,read,read_csv,dump,csv,verify_history
from data import FEATURES,PARTS,prepare
from train import new_model,predict,state_sha,TRAIN_SOURCES
from models import configure

def independent_windows(out):
    f=pd.read_csv(SOURCE);v=f.bin_valid_for_windows.to_numpy()==1;seg=f.temporal_segment_id.to_numpy();age=f.bin_end_ophrs2_h.to_numpy(float)
    raw=f[FEATURES].to_numpy(float);first=pd.to_datetime(f.first_datetime).astype('int64').to_numpy()/1e9
    last=pd.to_datetime(f.last_datetime).astype('int64').to_numpy()/1e9
    bad=(~v)|~np.isfinite(raw).all(1)
    breaks=np.r_[False,(seg[1:]!=seg[:-1])|~np.isclose(np.diff(age),.5,atol=1e-6,rtol=0)|((first[1:]-last[:-1])<0)|((first[1:]-last[:-1])>1.01)]
    bp=np.r_[0,np.cumsum(bad)];ep=np.r_[0,np.cumsum(breaks)];checks=[]
    for h in [6,12,24]:
        for part,(s,e) in PARTS.items():
            origins=[]
            for o in range(max(s,47),e-h):
                a=o-47;b=o+h
                if bp[b+1]-bp[a]==0 and ep[b+1]-ep[a+1]==0:origins.append(o)
            w=read_csv(out/f'windows/H{h}_{part}.csv');np.testing.assert_array_equal(origins,w.origin_row)
            checks.append(dict(horizon_steps=h,partition=part,origins=len(origins),PASS=True))
    csv(out/'INDEPENDENT_WINDOW_CHECKS.csv',checks)
    return checks

def independent_metric(p):
    y=p.y_true_delta_V.to_numpy(float);error=p.y_pred_delta_V.to_numpy(float)-y
    rmse=float(np.linalg.norm(error)/math.sqrt(len(error))*1000)
    mae=float(np.mean(np.abs(error))*1000);r2=float(1-np.dot(error,error)/np.dot(y-y.mean(),y-y.mean()))
    a=np.abs(error)*1000;top=max(1,int(np.ceil(len(p)*.01)))
    return dict(RMSE_mV=rmse,MAE_mV=mae,R2=r2,P95_mV=float(np.percentile(a,95)),P99_mV=float(np.percentile(a,99)),
        max_abs_mV=float(a.max()),top1pct_SSE_fraction=float(np.sort(error**2)[-top:].sum()/np.dot(error,error)),
        target_time_equal_RMSE_mV=float(np.sqrt(pd.DataFrame(dict(t=p.target_row,e2=error**2)).groupby('t').e2.mean().mean())*1000))

def verify(out):
    assert read(out/'HOLDOUT_EVALUATION_COMPLETED.json')['PASS'];configure()
    windowchecks=independent_windows(out);records=[];gradsummary=[]
    registered=read(out/'COHORT_BEFORE_TRAINING.json')
    for name in TRAIN_SOURCES:assert sha(HERE/'scripts'/name)==registered['code_sha256'][name],name
    for kind,h in read(CONFIG)['training_jobs']:
        job=out/f'models/{kind}_H{h}';info=read(job/'COMPLETED.json');logs=read(job/'epoch_log.json')
        assert info['seed']==202 and info['initialized_from_scratch'] and not info['historical_checkpoint_loaded']
        assert info['optimizer_initial_state_empty'] and info['optimizer_steps']==info['backward_calls']>0
        assert sha(job/'best.pt')==info['best_checkpoint_sha256']
        assert info['code_sha256']==registered['code_sha256'] and info['source_sha256']==sha(SOURCE)
        prep=read(out/f'preprocess/H{h}.json');frame,d,identity=prepare(out,h,True,prep)
        for part in ['TRAIN','VALIDATION']:assert identity['partitions'][part]==prep['partitions'][part]
        # Verify scaler independently against only legal TRAIN raw histories.
        flat=d['TRAIN']['xraw'].astype(np.float64).reshape(-1,14)
        np.testing.assert_array_equal(flat.mean(0),prep['feature_mean'])
        np.testing.assert_array_equal(np.where(flat.std(0)>1e-8,flat.std(0),1.),prep['feature_std'])
        loader=DataLoader(TensorDataset(torch.arange(len(d['TRAIN']['origins']))),batch_size=32,shuffle=True,
            generator=torch.Generator().manual_seed(202),num_workers=0,drop_last=False)
        best=math.inf;bestepoch=0;wait=0
        for l in logs:
            order=np.concatenate([batch[0].numpy() for batch in loader])
            assert array_sha(order.astype(np.int64))==l['sampler_order_sha256']
            current=math.sqrt(l['validation_MSE'])
            if current<best-1e-12:best=current;bestepoch=l['epoch'];wait=0
            else:wait+=1
            assert l['best_epoch']==bestepoch and l['stale_epochs']==wait
            assert all(np.isfinite(x) for x in l['gradient_groups'].values())
            assert l['gradient_norm_min']>0 and np.isfinite(l['gradient_norm_max'])
            assert l['optimizer_steps']==l['epoch']*math.ceil(len(d['TRAIN']['origins'])/32)
        assert bestepoch==info['best_epoch'] and logs[-1]['epoch']==info['stop_epoch']
        assert info['stop_epoch']==450 or logs[-1]['stale_epochs']==40
        updates=read_csv(job/'PARAMETER_CHANGES.csv')
        for group,g in updates.groupby('path'):
            assert g.changed_elements.sum()>0,(kind,h,group)
            gradsummary.append(dict(model=kind,horizon_steps=h,path=group,changed_elements=int(g.changed_elements.sum()),
                any_epoch_positive_gradient=any(l['gradient_groups'].get(group,0)>0 for l in logs)))
        model=new_model(kind,h).cuda();assert state_sha(model)==info['initial_state_sha256']
        init=torch.load(job/'initial.pt',map_location='cpu',weights_only=False)
        for name,t in model.state_dict().items():assert torch.equal(t.detach().cpu(),init['state_dict'][name]),name
        checkpoint=torch.load(job/'best.pt',map_location='cpu',weights_only=False)
        for row in updates.itertuples():
            delta=checkpoint['state_dict'][row.parameter].cpu().double()-init['state_dict'][row.parameter].cpu().double()
            assert int(torch.count_nonzero(delta))==row.changed_elements
            assert abs(float(torch.linalg.vector_norm(delta))-row.delta_L2)<1e-12
        model.load_state_dict(checkpoint['state_dict'],strict=True);before=state_sha(model);assert before==info['best_state_sha256']
        allref=read_csv(out/'ALL_PARTITION_METRICS.csv')
        for part,w in d.items():
            saved=read_csv(job/f'predictions_{part}.csv')
            np.testing.assert_array_equal(saved.origin_row,np.repeat(w['origins'],h))
            np.testing.assert_array_equal(saved.target_row,w['targets'].ravel())
            np.testing.assert_array_equal(saved.horizon_step,np.tile(np.arange(1,h+1),len(w['origins'])))
            np.testing.assert_array_equal(saved.target_age_h,w['target_age_h'].ravel())
            np.testing.assert_array_equal(saved.origin_raw_voltage_V,np.repeat(w['origin_voltage'],h))
            np.testing.assert_array_equal(saved.true_voltage_V,w['true_voltage'].ravel())
            np.testing.assert_array_equal(saved.y_true_delta_V,w['y'].ravel())
            pred=predict(model,w['x'])*prep['target_scale_V']
            np.testing.assert_array_equal(pred.ravel(),saved.y_pred_delta_V)
            np.testing.assert_array_equal(saved.predicted_voltage_V,saved.origin_raw_voltage_V+saved.y_pred_delta_V)
            for scope,g in [('ALL_H_LEGAL',saved),('COMMON_H24_ORIGINS',saved[saved.origin_row.isin(read_csv(out/f'windows/H24_{part}.csv').origin_row)])]:
                m=independent_metric(g);ref=allref[(allref.model==kind)&(allref.horizon_steps==h)&(allref.partition==part)&(allref.scope==scope)].iloc[0]
                for k,v in m.items():assert abs(v-ref[k])<1e-9,(kind,h,part,scope,k,v,ref[k])
                records.append(dict(model=kind,horizon_steps=h,partition=part,scope=scope,checkpoint_replay_exact=True,**m))
        assert state_sha(model)==before
        del model,init,checkpoint;torch.cuda.empty_cache()
    c0=read(out/'models/C0_H24/STARTED.json');c1=read(out/'models/C1_H24/STARTED.json')
    assert c0['initial_backbone_sha256']==c1['initial_backbone_sha256']
    assert c0['initial_CPU_RNG_sha256']==c1['initial_CPU_RNG_sha256'] and c0['initial_CUDA_RNG_sha256']==c1['initial_CUDA_RNG_sha256']
    csv(out/'INDEPENDENT_RECALCULATED_METRICS.csv',records);csv(out/'INDEPENDENT_GRADIENT_AND_PATH_UPDATES.csv',gradsummary)
    dump(out/'INDEPENDENT_VERIFICATION.json',dict(PASS=True,new_checkpoints_reloaded=6,partition_prediction_replays=18,
        exact_replay_no_tolerance=True,window_configurations_verified=len(windowchecks),metric_tolerance_mV=1e-9,
        gradients_finite=True,all_parameter_groups_changed=True,initial_backbones_and_rng_C0_C1_equal=True,
        sampler_orders_and_early_stopping_replayed=True,backward_calls_in_this_verifier=0,optimizer_steps_in_this_verifier=0,
        no_model_selection_or_training_on_holdout=True))
    print('Independent verification PASS:6 checkpoints,18 exact prediction replays,9 window configurations',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--history',action='store_true');p.add_argument('--run',type=Path);a=p.parse_args()
    out=a.run or Path(read(HERE/'outputs/ACTIVE_RUN.json')['path'])
    if a.history:print('History protection PASS:',verify_history(out))
    else:verify(out)

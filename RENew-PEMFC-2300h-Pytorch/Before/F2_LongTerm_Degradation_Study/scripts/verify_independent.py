"""Independent clock counts, target reconstruction, metrics and checkpoint replay.

No fitting forecast networks, optimizer updates or backward passes. Uses
independent cumulative-sum window/target calculations and float64 arithmetic.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from audit_data import HERE,REPO,CONFIG,SOURCE,META,sha,dump,csv
from train_longterm import prepare,new_model,predict,state_sha

AUDIT=HERE/'outputs/qualification_20261020_v2'

def verify():
    out=Path(json.loads((HERE/'outputs/ACTIVE_PILOT.json').read_text(encoding='utf-8'))['path'])
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    meta=pd.read_csv(SOURCE,usecols=META)
    n=len(meta);age=meta.bin_end_ophrs2_h.to_numpy(float)
    seg=meta.temporal_segment_id.to_numpy();valid=meta.bin_valid_for_windows.to_numpy()==1
    dt0=pd.to_datetime(meta.first_datetime).astype('int64').to_numpy()/1e9
    dt1=pd.to_datetime(meta.last_datetime).astype('int64').to_numpy()/1e9
    breaks=np.r_[False,(seg[1:]!=seg[:-1])|~np.isclose(np.diff(age),.5,atol=1e-6,rtol=0)|((dt0[1:]-dt1[:-1])<0)|((dt0[1:]-dt1[:-1])>1.01)]
    bad=np.r_[0,np.cumsum(~valid)];edge=np.r_[0,np.cumsum(breaks)]
    counts=pd.read_csv(AUDIT/'WINDOW_COUNTS.csv')
    checks=[]
    for r in counts.itertuples():
        start,end=cfg['partition_rows'][r.partition];h=int(r.history_steps);k=int(r.horizon_steps)
        origins=[]
        for origin in range(max(start,h-1),end-k):
            lo=origin-h+1;hi=origin+k
            if bad[hi+1]-bad[lo]==0 and edge[hi+1]-edge[lo+1]==0:origins.append(origin)
        reference=pd.read_csv(AUDIT/f'L{h}_H{k}_{r.partition}_LEGAL.csv')
        assert origins==reference.origin_row.tolist(),(h,k,r.partition)
        assert len(origins)==r.legal_origins
        last=-1;disjoint=0;unique=set()
        for o in origins:
            unique.update(range(o+1,o+k+1))
            if o+1>last:disjoint+=1;last=o+k
        assert len(unique)==r.unique_target_rows and disjoint==r.disjoint_target_trajectories
        checks.append(dict(check='independent_clock_windows',history=h,horizon=k,partition=r.partition,PASS=True))
    fields=cfg['target_B']['covariates'];f=pd.read_csv(SOURCE,nrows=3074,usecols=fields+['cell_voltage_mean','bin_valid_for_windows'])
    v=f.cell_voltage_mean.to_numpy(float);z=f[fields].to_numpy(float);train=np.arange(len(f))<2690
    train &= f.bin_valid_for_windows.to_numpy()==1
    train &= np.isfinite(z).all(1)&np.isfinite(v)
    zm=z[train].mean(0);zs=z[train].std(0)
    X=np.column_stack([np.ones(train.sum()),(z[train]-zm)/zs])
    coeff=np.linalg.pinv(X)@v[train]
    base=np.median(z[train],axis=0)
    vb=v-(z-base)@(coeff[1:]/zs)
    support=np.isfinite(z).all(1)&(z>=z[train].min(0)).all(1)&(z<=z[train].max(0)).all(1)&valid[:3074]
    observed=pd.read_csv(AUDIT/'TARGETS_TRAIN_VALIDATION_ONLY.csv',float_precision='round_trip')
    target_diffs={}
    for name,values,ok in [('A',v,valid[:3074]),('B',vb,support)]:
        result=np.full(3074,np.nan)
        sums=np.r_[0,np.cumsum(np.where(np.isfinite(values),values,0.))]
        for i in range(47,3074):
            lo=i-47
            if ok[lo:i+1].all() and not breaks[lo+1:i+1].any() and np.isfinite(values[lo:i+1]).all():result[i]=(sums[i+1]-sums[lo])/48
        col='A_causal_mean_V' if name=='A' else 'B_causal_adjusted_mean_V'
        np.testing.assert_array_equal(np.isfinite(result),np.isfinite(observed[col]))
        diff=np.nanmax(np.abs(result-observed[col].to_numpy(float)))
        assert diff<2e-12,(name,diff)
        target_diffs[name]=float(diff)
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.use_deterministic_algorithms(False)
    artifacts=[];metricchecks=[]
    for d in sorted(out.glob('*_seed202_*h')):
        result=json.loads((d/'COMPLETED.json').read_text(encoding='utf-8'))
        assert result['initialized_from_scratch'] and result['optimizer_initial_state_empty'] and not result['history_checkpoint_loaded']
        assert result['parameters_changed'] and result['optimizer_steps']==result['backward_calls']>0
        assert sha(d/'best.pt')==result['best_checkpoint_sha256']
        logs=json.loads((d/'epoch_log.json').read_text(encoding='utf-8'))
        best=float('inf');wait=0;bestepoch=0
        for row in logs:
            if row['validation_MSE']<best-1e-12:best=row['validation_MSE'];bestepoch=row['epoch'];wait=0
            else:wait+=1
            assert row['best_epoch']==bestepoch and row['patience_counter']==wait
            assert row['gradient_norm_mean']>0 and np.isfinite(row['gradient_norm_max'])
        assert bestepoch==result['best_epoch'] and logs[-1]['epoch']==result['stop_epoch']
        data,identity=prepare(result['horizon_steps'])
        saved_identity=json.loads((d/'PREPROCESS_AND_WINDOWS.json').read_text(encoding='utf-8'))
        assert identity==saved_identity
        model=new_model(result['model'],result['horizon_steps'],202).cuda()
        checkpoint=torch.load(d/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict']);assert state_sha(model)==result['best_state_sha256']
        for part,w in data.items():
            points=pd.read_csv(d/f'{part}_PREDICTIONS.csv',float_precision='round_trip')
            np.testing.assert_array_equal(points.origin_row,np.repeat(w['origins'],result['horizon_steps']))
            np.testing.assert_array_equal(points.target_row,w['targets'].ravel())
            np.testing.assert_array_equal(points.true_delta_proxy_V,w['y'].ravel())
            np.testing.assert_array_equal(points.target_h,w['ages'][w['targets']].ravel())
            yp=predict(model,w['x']).astype(float)*identity['target_scale']
            np.testing.assert_array_equal(yp.ravel(),points.predicted_delta_proxy_V)
            y=points.true_delta_proxy_V.to_numpy(float);err=yp.ravel()-y
            rmse=float(np.linalg.norm(err)/np.sqrt(len(err))*1000);mae=float(np.abs(err).mean()*1000)
            r2=float(1-np.dot(err,err)/np.dot(y-y.mean(),y-y.mean()))
            ref=next(m for m in result['metrics'] if m['model']==result['model'] and m['partition']==part)
            assert abs(rmse-ref['RMSE_mV'])<1e-12 and abs(mae-ref['MAE_mV'])<1e-12 and abs(r2-ref['R2_delta'])<1e-12
            metricchecks.append(dict(model=result['model'],horizon_h=result['horizon_steps']*.5,partition=part,RMSE_mV=rmse,MAE_mV=mae,R2_delta=r2,checkpoint_replay_bitwise_equal=True))
        del model;torch.cuda.empty_cache()
        artifacts.append(dict(directory=d.name,checkpoint_sha256=sha(d/'best.pt'),log_sha256=sha(d/'epoch_log.json'),PASS=True))
    csv(out/'INDEPENDENT_METRIC_RECALCULATION.csv',pd.DataFrame(metricchecks))
    dump(out/'INDEPENDENT_VERIFICATION.json',dict(PASS=True,window_configurations_checked=len(checks),training_runs_checked=len(artifacts),
        target_recalculation_max_difference_V=target_diffs,metric_absolute_tolerance_mV=1e-12,
        checkpoint_prediction_replays='exact equality, no tolerance',all_training_gradients_finite=True,early_stopping_replayed=True,
        holdout_outcome_rows_read=0,artifacts=artifacts))
    print('Independent verification PASS:',len(checks),'window configurations,',len(artifacts),'trained models',flush=True)

def verify_history():
    before=json.loads((AUDIT/'HISTORY_BEFORE.json').read_text(encoding='utf-8'));changed=[]
    for i,(rel,expected) in enumerate(before.items(),1):
        p=REPO/rel
        if not p.is_file() or sha(p)!=expected:changed.append(rel)
        if i%10000==0:print('Historical SHA recheck',i,flush=True)
    missing=json.loads((AUDIT/'PREEXISTING_MISSING_TRACKED_FILES.json').read_text(encoding='utf-8'))
    newly_present=[p for p in missing if (REPO/p).is_file()]
    dump(AUDIT/'HISTORY_PROTECTION_FINAL.json',dict(PASS=not changed and not newly_present,files_hashed=len(before),changed_files=changed,
        preexisting_missing_files=len(missing),preexisting_missing_files_restored=newly_present))
    assert not changed and not newly_present
    print('Historical protection PASS',len(before),'present files unchanged;',len(missing),'pre-existing missing files recorded')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--history',action='store_true');a=p.parse_args()
    if a.history:verify_history()
    else:verify()

"""Fresh GPU pilot and predeclared conditional comparison. No old checkpoints.

Selection access is nrows=3074 only. HOLDOUT inference is deliberately not part
of the pilot. Every invocation creates a new run directory and records hashes.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import random
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import native_models as nm
from audit_data import HERE, ROOT, SOURCE, CONFIG, csv, dump, sha

AUDIT=HERE/'outputs/qualification_20261020_v2'
TRAIN_CONFIG=HERE/'configs/FROZEN_TRAINING_PROTOCOL.json'
FEATURES=nm.FEATURES
TOP10=[0,3,13,11,4,9,6,7,10,8]

def atomic_json(path,value):
    tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for attempt in range(8):
        try: tmp.replace(path);return
        except PermissionError:
            if attempt==7: raise
            time.sleep(.1*(attempt+1))

def state_sha(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def array_sha(x):
    import hashlib
    return hashlib.sha256(str((x.shape,str(x.dtype))).encode()+np.ascontiguousarray(x).tobytes()).hexdigest()

def prepare(horizon):
    f=pd.read_csv(SOURCE,nrows=3074,usecols=FEATURES+['bin_end_ophrs2_h'])
    target=pd.read_csv(AUDIT/'TARGETS_TRAIN_VALIDATION_ONLY.csv',float_precision='round_trip')
    b=target.B_causal_adjusted_mean_V.to_numpy(float)
    raw=f[FEATURES].to_numpy(float); ages=f.bin_end_ophrs2_h.to_numpy(float)
    result={}
    for part in ['TRAIN','VALIDATION']:
        w=pd.read_csv(AUDIT/f'L48_H{horizon}_{part}_LEGAL.csv')
        ok=[np.isfinite(b[r.origin_row:r.target_last_row+1]).all() and np.isfinite(raw[r.history_first_row:r.origin_row+1]).all() for r in w.itertuples()]
        w=w[ok].reset_index(drop=True)
        origins=w.origin_row.to_numpy(int);targets=origins[:,None]+np.arange(1,horizon+1)[None,:]
        x=np.stack([raw[o-47:o+1] for o in origins]).astype(np.float32)
        y=b[targets]-b[origins,None]
        result[part]=dict(w=w,origins=origins,targets=targets,xraw=x,y=y,originB=b[origins],trueB=b[targets],ages=ages)
    trainflat=result['TRAIN']['xraw'].astype(float).reshape(-1,14)
    mean=trainflat.mean(0);scale=trainflat.std(0);scale=np.where(scale>1e-8,scale,1.)
    target_scale=float(np.std(result['TRAIN']['y']))
    assert target_scale>0
    for d in result.values():
        x=d['xraw'].copy()
        for i in range(1,48):x[:,i,0]=.3*d['xraw'][:,i,0]+.7*x[:,i-1,0]
        x=((x-mean)/scale).astype(np.float32)
        x[:,:,[i for i in range(14) if i not in TOP10]]=0
        d['x']=x;d['ynorm']=(d['y']/target_scale).astype(np.float32)
    assert result['TRAIN']['targets'].max()<result['VALIDATION']['origins'].min()
    identity=dict(horizon_steps=horizon,history_steps=48,feature_mean=mean.tolist(),feature_std=scale.tolist(),target_scale=target_scale,
        target_sha256=sha(AUDIT/'TARGETS_TRAIN_VALIDATION_ONLY.csv'),source_sha256=sha(SOURCE),
        partitions={p:dict(origins=len(d['origins']),queries=d['y'].size,inputs_sha256=array_sha(d['x']),
            labels_sha256=array_sha(d['y']),origin_sha256=array_sha(d['origins']),target_sha256=array_sha(d['targets'])) for p,d in result.items()})
    return result,identity

def new_model(kind,horizon,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    nm.HORIZON=horizon
    if kind=='LSTM':return nm.PlainLSTM(horizon)
    return nm.RegimeAwareDualPathTFT(hidden_size=64,hidden_continuous_size=32,attention_head_size=4,lstm_layers=1,dropout=.1)

def metrics(points):
    true=points.true_delta_proxy_V.to_numpy(float);err=points.predicted_delta_proxy_V.to_numpy(float)-true
    sse=float(np.sum(err**2,dtype=np.float64));sst=float(np.sum((true-true.mean())**2,dtype=np.float64))
    target=pd.DataFrame(dict(target=points.target_row,error2=err**2)).groupby('target').error2.mean()
    terminal=points.horizon_step==points.horizon_step.max()
    abserr=np.abs(err)*1000
    return dict(RMSE_mV=float(np.sqrt(sse/len(err))*1000),MAE_mV=float(np.mean(abserr)),R2_delta=1-sse/sst,
        R2_absolute_proxy=1-sse/float(np.sum((points.true_proxy_V-points.true_proxy_V.mean())**2)),
        terminal_RMSE_mV=float(np.sqrt(np.mean(err[terminal]**2))*1000),
        target_time_equal_RMSE_mV=float(np.sqrt(target.mean())*1000),P95_mV=float(np.quantile(abserr,.95)),
        P99_mV=float(np.quantile(abserr,.99)),max_abs_mV=float(abserr.max()),queries=len(points),origins=points.origin_row.nunique(),
        unique_target_times=points.target_row.nunique())

def point_frame(d,pred,horizon,kind,seed,part):
    return pd.DataFrame(dict(model=kind,seed=seed,partition=part,horizon_h=horizon*.5,
        origin_row=np.repeat(d['origins'],horizon),target_row=d['targets'].ravel(),
        horizon_step=np.tile(np.arange(1,horizon+1),len(d['origins'])),
        origin_h=np.repeat(d['ages'][d['origins']],horizon),target_h=d['ages'][d['targets']].ravel(),
        origin_proxy_V=np.repeat(d['originB'],horizon),true_proxy_V=d['trueB'].ravel(),
        predicted_proxy_V=(d['originB'][:,None]+pred).ravel(),true_delta_proxy_V=d['y'].ravel(),predicted_delta_proxy_V=pred.ravel()))

def curve(log,path,live=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.semilogy([r['epoch'] for r in log],[r['train_MSE'] for r in log],label='Train',color='#0072B2')
    ax.semilogy([r['epoch'] for r in log],[r['validation_MSE'] for r in log],label='Validation',color='#D55E00')
    ax.set(xlabel='Epoch',ylabel='Standardized delta-proxy MSE',title='Fresh training; no historical log replay')
    ax.legend();ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)
    if live:
        from IPython.display import display,Image
        display(Image(filename=str(path)))

@torch.no_grad()
def predict(model,x):
    model.eval();a=[]
    for j in range(0,len(x),32):a.append(model.predict_delta(torch.from_numpy(x[j:j+32]).cuda()).cpu().numpy())
    return np.concatenate(a)

def fit(kind,seed,horizon,out,live=False):
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(TRAIN_CONFIG.read_text(encoding='utf-8'));hp=cfg['hyperparameters']
    data,identity=prepare(horizon);dump(out/'PREPROCESS_AND_WINDOWS.json',identity)
    for part,d in data.items():csv(out/f'{part}_WINDOWS.csv',d['w'])
    model=new_model(kind,horizon,seed).cuda()
    nparams=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if kind=='F2':assert nparams==332436
    initsha=state_sha(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=hp['learning_rate'],weight_decay=hp['weight_decay'])
    assert not optimizer.state
    loader=DataLoader(TensorDataset(torch.from_numpy(data['TRAIN']['x']),torch.from_numpy(data['TRAIN']['ynorm'])),
        batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    runid=dict(model=kind,seed=seed,horizon_steps=horizon,train_config_sha256=sha(TRAIN_CONFIG),qualification_sha256=sha(CONFIG),
        code_sha256=sha(Path(__file__)),native_model_code_sha256=sha(HERE/'scripts/native_models.py'),initial_state_sha256=initsha,
        parameters=nparams,initialized_from_scratch=True,history_checkpoint_loaded=False,optimizer_initial_state_empty=True,
        torch=str(torch.__version__),CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
        strict_determinism=False,ordinary_CUDA_bitwise_reproducibility_promised=False)
    dump(out/'RUN_IDENTITY.json',runid)
    print(f'Start fresh {kind}, seed {seed}, horizon {horizon*.5:g}h, {nparams} parameters',flush=True)
    best=float('inf');wait=0;log=[];steps=0;started=time.monotonic();bestepoch=0
    for epoch in range(1,hp['max_epochs']+1):
        model.train();loss_sum=0.;count=0;grad_norms=[]
        for xb,yb in loader:
            xb=xb.cuda();yb=yb.cuda();optimizer.zero_grad(set_to_none=True)
            yp=model.predict_delta(xb);loss=torch.nn.functional.mse_loss(yp,yb)
            assert torch.isfinite(loss)
            loss.backward()
            g=torch.nn.utils.clip_grad_norm_(model.parameters(),hp['gradient_clip_norm'])
            assert torch.isfinite(g)
            grad_norms.append(float(g));optimizer.step();steps+=1
            loss_sum+=float(loss.detach())*len(xb);count+=len(xb)
        validation_pred=predict(model,data['VALIDATION']['x'])
        val=float(np.mean((validation_pred-data['VALIDATION']['ynorm']).astype(float)**2))
        improved=val<best-hp['min_delta_standardized_MSE']
        if improved:
            best=val;wait=0;bestepoch=epoch
            tmp=out/'best.pt.writing'
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation_MSE=val,identity=runid),tmp)
            tmp.replace(out/'best.pt')
        else:wait+=1
        log.append(dict(epoch=epoch,train_MSE=loss_sum/count,validation_MSE=val,best_epoch=bestepoch,
            best_validation_MSE=best,patience_counter=wait,gradient_norm_mean=float(np.mean(grad_norms)),
            gradient_norm_max=float(np.max(grad_norms)),optimizer_steps=steps,state_sha256=state_sha(model),elapsed_s=time.monotonic()-started))
        # Preserve every epoch; ordinary Windows file lock retries do not alter training math.
        atomic_json(out/'epoch_log.json',log)
        if epoch==1 or epoch%10==0 or wait>=hp['patience']:
            print(f'{kind} seed{seed} {horizon*.5:g}h epoch{epoch} train={loss_sum/count:.7g} val={val:.7g} best={bestepoch} patience={wait}',flush=True)
            curve(log,out/'training_history.png',live)
        if wait>=hp['patience']:break
    finalsha=state_sha(model);assert finalsha!=initsha and steps>0
    chk=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(chk['state_dict']);bestsha=state_sha(model)
    rows=[];step_rows=[]
    for part,d in data.items():
        pred=predict(model,d['x']).astype(float)*identity['target_scale']
        points=point_frame(d,pred,horizon,kind,seed,part)
        csv(out/f'{part}_PREDICTIONS.csv',points)
        m=metrics(points);m.update(model=kind,seed=seed,horizon_h=horizon*.5,partition=part)
        rows.append(m)
        baseline=point_frame(d,np.zeros_like(pred),horizon,'Persistence',seed,part)
        bm=metrics(baseline);bm.update(model='Persistence',seed=seed,horizon_h=horizon*.5,partition=part);rows.append(bm)
        for step,g in points.groupby('horizon_step'):
            err=(g.predicted_delta_proxy_V-g.true_delta_proxy_V).to_numpy(float)*1000
            step_rows.append(dict(model=kind,seed=seed,partition=part,horizon_h=horizon*.5,step=int(step),lead_h=step*.5,
                RMSE_mV=float(np.sqrt(np.mean(err**2))),MAE_mV=float(np.mean(abs(err)))))
        if kind=='F2':
            with torch.no_grad():
                x=torch.from_numpy(d['x']).cuda();g=model.regime_weight(model.build_input(x)['encoder_cont']).cpu().numpy().ravel()
            csv(out/f'{part}_GATE.csv',pd.DataFrame(dict(origin_row=d['origins'],origin_h=d['ages'][d['origins']],g=g)))
    csv(out/'METRICS.csv',pd.DataFrame(rows));csv(out/'STEP_METRICS.csv',pd.DataFrame(step_rows));csv(out/'epoch_log.csv',pd.DataFrame(log))
    curve(log,out/'training_history.png',live)
    result=dict(**runid,best_epoch=bestepoch,stop_epoch=epoch,optimizer_steps=steps,backward_calls=steps,
        parameters_changed=True,best_state_sha256=bestsha,final_state_sha256=finalsha,best_checkpoint_sha256=sha(out/'best.pt'),
        wall_training_s=time.monotonic()-started,metrics=rows)
    atomic_json(out/'COMPLETED.json',result)
    del model,optimizer;torch.cuda.empty_cache()
    return result

def run_pilot(live=False):
    s=json.loads((AUDIT/'TARGET_SUMMARY.json').read_text(encoding='utf-8'))
    assert next(x for x in s['target_gates'] if x['target']=='B')['forecast_training_qualified']
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.use_deterministic_algorithms(False)
    assert torch.cuda.is_available()
    warnings.filterwarnings('ignore',message="Attribute '(loss|logging_metrics)' is an instance of")
    out=HERE/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_pilot202')
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'FROZEN_RUN_PROTOCOL.json',dict(training_protocol_sha256=sha(TRAIN_CONFIG),qualification_sha256=sha(CONFIG),
        training_code_sha256=sha(Path(__file__)),model_code_sha256=sha(HERE/'scripts/native_models.py'),
        seed=202,horizons_steps=[48,96],models=['LSTM','F2'],holdout_outcomes_access=False))
    dump(HERE/'outputs/ACTIVE_PILOT.json',dict(path=str(out)))
    results=[]
    try:
        for horizon in [48,96]:
            for kind in ['LSTM','F2']:
                result=fit(kind,202,horizon,out/f'{kind}_seed202_{horizon//2}h',live)
                results.append(result)
                atomic_json(out/'EXECUTION_LEDGER.json',results)
        summary=[]
        for r in results:
            m=next(x for x in r['metrics'] if x['partition']=='VALIDATION' and x['model']==r['model'])
            b=next(x for x in r['metrics'] if x['partition']=='VALIDATION' and x['model']=='Persistence')
            rmse_gain=1-m['RMSE_mV']/b['RMSE_mV'];mae_gain=1-m['MAE_mV']/b['MAE_mV']
            good=rmse_gain>=.05 and mae_gain>=.05 and m['R2_delta']>0 and m['terminal_RMSE_mV']<=b['terminal_RMSE_mV']
            summary.append(dict(model=r['model'],horizon_h=r['horizon_steps']*.5,pilot_qualified=bool(good),
                RMSE_gain_fraction=rmse_gain,MAE_gain_fraction=mae_gain,RMSE_mV=m['RMSE_mV'],MAE_mV=m['MAE_mV'],
                R2_delta=m['R2_delta'],terminal_RMSE_mV=m['terminal_RMSE_mV'],baseline_RMSE_mV=b['RMSE_mV'],
                baseline_MAE_mV=b['MAE_mV'],baseline_terminal_RMSE_mV=b['terminal_RMSE_mV']))
        csv(out/'PILOT_QUALIFICATION.csv',pd.DataFrame(summary))
        expand=all(x['pilot_qualified'] for x in summary if x['horizon_h']==48)
        atomic_json(out/'PILOT_DECISION.json',dict(expand_to_six_seeds=expand,completed_fresh_training_runs=len(results),
            holdout_evaluation_executed=False,status='PILOT_PASSED' if expand else 'PILOT_NOT_QUALIFIED_STOP_BEFORE_EXPANSION'))
        print('Pilot complete; expand_to_six_seeds =',expand,flush=True)
    except BaseException:
        (out/'FAILURE_TRACEBACK.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--live',action='store_true')
    a=parser.parse_args()
    if not a.pilot:parser.error('Specify --pilot to authorize the registered fresh-training pilot')
    print(run_pilot(a.live))

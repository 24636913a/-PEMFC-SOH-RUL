"""At most six registered, fresh GPU runs; no holdout inference during training."""
import argparse
import ctypes
import math
import time
import traceback
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from common import HERE,SOURCE,CONFIG,sha,array_sha,read,dump,csv,metrics,utc
from data import prepare,points
from models import new_model,configure

TRAIN_SOURCES=['common.py','data.py','native_models.py','models.py','train.py']

def state_sha(model,backbone_only=False):
    import hashlib
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        if backbone_only and (name.startswith('residual_head.') or name=='residual_logit'):continue
        h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def predict(model,x):
    model.eval();pred=[]
    for i in range(0,len(x),32):pred.append(model.predict_delta(torch.from_numpy(x[i:i+32]).cuda()).cpu().numpy())
    return np.concatenate(pred)

def group_name(name):
    for prefix,label in [('history_branch.','CNN'),('lstm_encoder.','LSTM_encoder'),('lstm_decoder.','LSTM_decoder'),('regime_gate.','gate'),('residual_head.','residual_head'),('residual_logit','alpha'),('lstm.','plain_LSTM'),('head.','plain_output')]:
        if name.startswith(prefix):return label
    return 'native_other'

def gradient_groups(model):
    sums={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            g=group_name(name);sums[g]=sums.get(g,0)+p.grad.detach().double().square().sum()
    return {g:float(torch.sqrt(v)) for g,v in sums.items()}

def render_loss(log,path,live=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.plot([x['epoch'] for x in log],[x['train_MSE'] for x in log],label='Train',color='#0072B2')
    ax.plot([x['epoch'] for x in log],[x['validation_MSE'] for x in log],label='Validation',color='#D55E00')
    ax.set(xlabel='Epoch',ylabel='Standardized raw DeltaV MSE',title=f"Fresh training; best epoch {log[-1]['best_epoch']}")
    ax.legend();ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)
    if live:
        from IPython.display import display,Image
        display(Image(filename=str(path)))

def save_checkpoint(path,model,info,epoch,val,steps):
    tmp=path.with_name(path.name+'.writing')
    torch.save(dict(state_dict=model.state_dict(),identity=info,best_epoch=epoch,validation_standardized_RMSE=val,optimizer_steps=steps),tmp)
    for i in range(8):
        try:tmp.replace(path);return
        except PermissionError:
            if i==7:raise
            time.sleep(.1*(i+1))

def evaluate_train_validation(out,job,model,d,frame,identity,kind,horizon):
    rows=[];steps=[]
    for part,w in d.items():
        delta=predict(model,w['x'])*identity['target_scale_V']
        p=points(frame,w,delta,horizon,kind,part);csv(job/f'predictions_{part}.csv',p)
        rows.append(dict(model=kind,horizon_steps=horizon,partition=part,scope='ALL_H_LEGAL',**metrics(p)))
        old=pd.read_csv(out/f'windows/H24_{part}.csv').origin_row
        common=p[p.origin_row.isin(old)]
        rows.append(dict(model=kind,horizon_steps=horizon,partition=part,scope='COMMON_H24_ORIGINS',**metrics(common)))
        for step,g in p.groupby('horizon_step'):steps.append(dict(model=kind,horizon_steps=horizon,partition=part,horizon_step=int(step),**metrics(g)))
    csv(job/'TRAIN_VALIDATION_METRICS.csv',rows);csv(job/'TRAIN_VALIDATION_STEP_METRICS.csv',steps)
    return rows

def fit(out,kind,horizon,live=False):
    cfg=read(CONFIG);hp=cfg['hyperparameters'];job=out/f'models/{kind}_H{horizon}'
    job.mkdir(parents=True,exist_ok=False)
    frame,d,prep=prepare(out,horizon)
    cache=out/f'preprocess/H{horizon}.json'
    if cache.exists():assert read(cache)==prep
    else:dump(cache,prep)
    model=new_model(kind,horizon).cuda();initial=state_sha(model);backbone=state_sha(model,True)
    initial_params={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=hp['learning_rate'],weight_decay=hp['weight_decay'])
    assert not optimizer.state
    info=dict(model=kind,seed=202,horizon_steps=horizon,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_state_sha256=initial,initial_backbone_sha256=backbone,initialized_from_scratch=True,historical_checkpoint_loaded=False,
        optimizer_initial_state_empty=True,protocol_sha256=sha(CONFIG),source_sha256=sha(SOURCE),
        code_sha256={name:sha(HERE/'scripts'/name) for name in TRAIN_SOURCES},preprocess_sha256=sha(cache),started_UTC=utc(),
        initial_CPU_RNG_sha256=array_sha(torch.get_rng_state().numpy()),initial_CUDA_RNG_sha256=array_sha(torch.cuda.get_rng_state().cpu().numpy()))
    dump(job/'STARTED.json',info)
    torch.save(dict(state_dict=model.state_dict(),identity=info),job/'initial.pt')
    if kind=='C1':assert read(out/'models/C0_H24/STARTED.json')['initial_backbone_sha256']==backbone
    generator=torch.Generator().manual_seed(202)
    loader=DataLoader(TensorDataset(torch.from_numpy(d['TRAIN']['x']),torch.from_numpy(d['TRAIN']['ynorm']),torch.arange(len(d['TRAIN']['origins']))),
        batch_size=hp['batch_size'],shuffle=True,generator=generator,num_workers=0,drop_last=False)
    best=math.inf;bestepoch=0;wait=0;steps=0;logs=[];start=time.perf_counter();first_update=None
    print(f'Fresh GPU training {kind} H={horizon}, Seed202; Train {len(d["TRAIN"]["origins"])} / Validation {len(d["VALIDATION"]["origins"])}; params {info["parameters"]}',flush=True)
    try:
        for epoch in range(1,hp['max_epochs']+1):
            model.train();losses=[];norms=[];order=[];gg={};t0=time.perf_counter()
            for batch,(xb,yb,idx) in enumerate(loader):
                xb=xb.cuda();yb=yb.cuda();optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.mse_loss(model.predict_delta(xb),yb)
                assert bool(torch.isfinite(loss))
                loss.backward()
                if batch==0:gg=gradient_groups(model)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),hp['gradient_clip_norm'])
                assert bool(torch.isfinite(norm)) and all(math.isfinite(x) for x in gg.values())
                optimizer.step();steps+=1
                if first_update is None:first_update=state_sha(model);assert first_update!=initial
                losses.append(float(loss.detach()));norms.append(float(norm));order.extend(idx.tolist())
            valpred=predict(model,d['VALIDATION']['x'])
            vmse=float(np.mean((valpred.astype(float)-d['VALIDATION']['ynorm'].astype(float))**2));vrmse=math.sqrt(vmse)
            assert math.isfinite(vrmse)
            if vrmse<best-1e-12:
                best=vrmse;bestepoch=epoch;wait=0;save_checkpoint(job/'best.pt',model,info,epoch,best,steps)
            else:wait+=1
            row=dict(epoch=epoch,train_MSE=float(np.mean(losses)),validation_MSE=vmse,validation_RMSE_V=vrmse*prep['target_scale_V'],
                best_epoch=bestepoch,stale_epochs=wait,optimizer_steps=steps,backward_calls=steps,gradient_norm_min=min(norms),gradient_norm_max=max(norms),
                sampler_order_sha256=array_sha(np.asarray(order,np.int64)),state_sha256=state_sha(model),epoch_s=time.perf_counter()-t0,
                gradient_groups=gg,residual_alpha=float(model.residual_alpha.detach()) if kind=='C1' else None)
            logs.append(row);dump(job/'epoch_log.json',logs)
            csv(job/'training_history.csv',[{k:v for k,v in x.items() if k!='gradient_groups'} for x in logs])
            if epoch==1 or epoch%10==0 or wait>=hp['patience'] or epoch==hp['max_epochs']:
                print(f'{kind} H{horizon} epoch{epoch} Train={row["train_MSE"]:.8g} Val={vmse:.8g} best={bestepoch} stale={wait}',flush=True)
                render_loss(logs,job/'training_history.png',live)
            if wait>=hp['patience']:break
        final=state_sha(model);assert steps>0 and final!=initial
        assert sha(SOURCE)==info['source_sha256'] and sha(CONFIG)==info['protocol_sha256']
        checkpoint=torch.load(job/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        rows=evaluate_train_validation(out,job,model,d,frame,prep,kind,horizon)
        changes=[]
        for name,p in model.named_parameters():
            delta=p.detach().cpu().double()-initial_params[name].double()
            changes.append(dict(parameter=name,path=group_name(name),elements=p.numel(),changed_elements=int(torch.count_nonzero(delta)),delta_L2=float(torch.linalg.vector_norm(delta))))
        csv(job/'PARAMETER_CHANGES.csv',changes)
        result=dict(**info,best_epoch=bestepoch,stop_epoch=epoch,optimizer_steps=steps,backward_calls=steps,
            first_update_sha256=first_update,final_state_sha256=final,best_state_sha256=state_sha(model),best_checkpoint_sha256=sha(job/'best.pt'),
            training_seconds=time.perf_counter()-start,metrics=rows,completed_UTC=utc())
        dump(job/'COMPLETED.json',result)
        render_loss(logs,job/'training_history.png',live)
        return result
    except BaseException:
        (job/'FAILURE_TRACEBACK.txt').write_text(traceback.format_exc(),encoding='utf-8')
        torch.save(dict(state_dict=model.state_dict(),optimizer_steps=steps,identity=info),job/'partial.pt')
        raise
    finally:
        del model,optimizer;torch.cuda.empty_cache()

def run(out,live=False):
    cfg=read(CONFIG);assert read(out/'P1_HISTORICAL_AUDIT.json')['PASS']
    assert read(out/'REGISTRATION.json')['protocol_sha256']==sha(CONFIG)
    assert read(out/'REGISTRATION.json')['data_sha256']==sha(SOURCE)
    warnings.filterwarnings('ignore',message="Attribute '(loss|logging_metrics)' is an instance of")
    env=configure();frozen=out/'COHORT_BEFORE_TRAINING.json'
    codes={name:sha(HERE/'scripts'/name) for name in TRAIN_SOURCES}
    if frozen.exists():
        prior=read(frozen);assert prior['code_sha256']==codes,'CODE_CHANGED_DURING_COHORT_REQUIRES_REVIEW'
    else:dump(frozen,dict(created_UTC=utc(),protocol_sha256=sha(CONFIG),code_sha256=codes,environment=env,jobs=cfg['training_jobs'],
        max_fresh_runs=6,no_holdout_inference_before_all_best_frozen=True))
    # Temporary, process-thread scoped idle-sleep prevention; no power setting changes.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    results=[]
    try:
        for kind,horizon in cfg['training_jobs']:
            job=out/f'models/{kind}_H{horizon}'
            if (job/'COMPLETED.json').exists():
                r=read(job/'COMPLETED.json');assert sha(job/'best.pt')==r['best_checkpoint_sha256']
                print('Retaining already completed job, no repeat:',job.name,flush=True)
            elif job.exists():raise RuntimeError('INCOMPLETE_JOB_PRESERVED_REQUIRES_ENGINEERING_REVIEW: '+str(job))
            else:r=fit(out,kind,horizon,live)
            results.append(r);dump(out/'EXECUTION_LEDGER.json',results)
        assert len(results)==6
        dump(out/'ALL_SIX_BEST_CHECKPOINTS_FROZEN.json',dict(created_UTC=utc(),protocol_sha256=sha(CONFIG),
            checkpoints=[dict(model=r['model'],horizon_steps=r['horizon_steps'],SHA256=r['best_checkpoint_sha256'],best_epoch=r['best_epoch']) for r in results],
            source_sha256=sha(SOURCE),new_holdout_predictions_computed=False))
        print('All six newly trained best checkpoints FROZEN. No new holdout inference performed yet.',flush=True)
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--live',action='store_true');a=p.parse_args()
    out=a.run or Path(read(HERE/'outputs/ACTIVE_RUN.json')['path']);run(out,a.live)

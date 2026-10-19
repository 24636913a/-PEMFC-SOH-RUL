"""Fresh serial training. Imports do not initialize a model or start any loop."""
import math
import os
import time
import traceback
import numpy as np
import pandas as pd
import torch
from IPython.display import Image,display
import matplotlib.pyplot as plt
import core as c
from io_utils import js,csv,text,metrics


def rng_sha():return dict(cpu=c.array_sha(torch.get_rng_state().numpy()),cuda=c.array_sha(torch.cuda.get_rng_state().cpu().numpy()))


def checkpoint(path,model,metadata):
    temporary=path.with_suffix('.pt.writing')
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},**metadata),temporary)
    for delay in [.05,.1,.2,.5,1.,None]:
        try:os.replace(temporary,path);return
        except PermissionError:
            if delay is None:raise
            time.sleep(delay)


def predict(model,x,batch_size=32):
    model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(x),batch_size):values.append(model.predict_delta(torch.from_numpy(x[start:start+batch_size]).cuda()).cpu().numpy())
    result=np.concatenate(values)
    c.require(result.shape==(len(x),24) and np.isfinite(result).all(),'PREDICTION_SHAPE_OR_FINITE')
    return result


def loss_figure(folder,history,kind,seed,handle):
    p=pd.DataFrame(history);fig,ax=plt.subplots(figsize=(9,3.5))
    ax.plot(p.epoch,p.train_standardized_MSE,label='Current Train MSE');ax.plot(p.epoch,p.validation_standardized_MSE,label='Current Validation MSE')
    ax.axvline(p.best_epoch.iloc[-1],color='gray',ls='--',lw=1,label='Best epoch')
    ax.set(xlabel='Epoch',ylabel='Standardized DeltaV MSE',title=f'{kind} Seed{seed}: fresh training')
    ax.legend();fig.tight_layout();path=folder/'training_history.png';fig.savefig(path,dpi=300,facecolor='white');plt.close(fig)
    image=Image(filename=str(path))
    if handle is None:return display(image,display_id=True)
    handle.update(image);return handle


def train_one(run,cohort,kind,seed,windows,scaler,prepared,protocol):
    c.require(not (run/'INVALIDATED_BEFORE_TRAINING.json').exists(),'INVALIDATED_RUN_NO_TRAINING')
    c.require(c.read_json(run/'P1_ACCEPTANCE.json')['status']=='PASS','P1_MUST_PASS_BEFORE_TRAINING')
    if cohort=='formal':c.require(c.read_json(run/'PILOT_GATE.json')['status']=='PASS','PILOT_REQUIRED')
    frozen=c.read_json(run/'TRAINING_CODE_BEFORE_FIRST_MODEL.json')
    for file,digest in frozen.items():c.require(c.sha(c.HERE/file)==digest,'TRAINING_CODE_CHANGED:'+file)
    for record in protocol['partitions']:
        c.require(c.array_sha(prepared[record['partition']])==record['inputs_sha256'],'MODEL_INPUT_IDENTITY')
    folder=run/'models'/cohort/f'{kind}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    model=c.new_model(kind,seed);initial=c.state_sha(model);count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c.HYPERPARAMETERS['learning_rate'],weight_decay=c.HYPERPARAMETERS['weight_decay'])
    c.require(not optimizer.state,'OPTIMIZER_NOT_FRESH')
    generator=torch.Generator().manual_seed(seed)
    train_y=(windows['TRAIN'].y/scaler.target_scale).astype('float32');val_y=(windows['VALIDATION'].y/scaler.target_scale).astype('float32')
    loader=c.DataLoader(c.TensorDataset(torch.from_numpy(prepared['TRAIN']),torch.from_numpy(train_y),torch.arange(len(train_y))),
        batch_size=32,shuffle=True,generator=generator,num_workers=0,drop_last=False)
    metadata=dict(run_id=run.name,cohort=cohort,kind=kind,seed=seed,parameters=count,initialization_sha256=initial,
        protocol_sha256=c.sha(run/'FROZEN_PROTOCOL.json'),prepared_train_sha256=c.array_sha(prepared['TRAIN']),
        fresh_initialization=True,historical_checkpoint_loaded=False,optimizer_initial_state_empty=True,initial_rng=rng_sha(),
        LSTM_internal_dropout=0. if kind=='LSTM' else None,LSTM_external_dropout=0. if kind=='LSTM' else None,
        validation_selection='minimum full standardized Validation RMSE, min_delta1e-12, patience40')
    js(folder/'BEFORE_TRAINING.json',metadata)
    best=math.inf;best_epoch=stale=steps=0;first_update=None;history=[];handle=None;started=time.perf_counter()
    print(f'Training {cohort} {kind} Seed {seed}: {count} parameters, fresh weights/AdamW',flush=True)
    try:
        for epoch in range(1,451):
            model.train();losses=[];norms=[];order=[];epoch_started=time.perf_counter()
            for xb,yb,indices in loader:
                xb,yb=xb.cuda(),yb.cuda();optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.mse_loss(model.predict_delta(xb),yb)
                c.require(bool(torch.isfinite(loss)),'NONFINITE_LOSS');loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                c.require(bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),'NONFINITE_GRADIENT')
                optimizer.step();steps+=1
                if first_update is None:first_update=c.state_sha(model);c.require(first_update!=initial,'NO_FIRST_UPDATE')
                losses.append(float(loss.detach().cpu()));norms.append(float(norm));order.extend(indices.tolist())
            validation=predict(model,prepared['VALIDATION']);mse=float(np.mean((validation.astype('float64')-val_y.astype('float64'))**2));score=math.sqrt(mse)
            c.require(math.isfinite(score),'NONFINITE_VALIDATION')
            if score<best-1e-12:
                best,best_epoch,stale=score,epoch,0
                checkpoint(folder/'best_model.pt',model,metadata|dict(best_epoch=epoch,optimizer_steps=steps,target_scale_V=scaler.target_scale))
            else:stale+=1
            history.append(dict(epoch=epoch,train_standardized_MSE=float(np.mean(losses)),validation_standardized_MSE=mse,
                best_epoch=best_epoch,stale_epochs=stale,optimizer_steps=steps,backward_calls=steps,
                gradient_norm_min=min(norms),gradient_norm_max=max(norms),sampler_order_sha256=c.array_sha(np.asarray(order,np.int64)),
                state_sha256=c.state_sha(model),rng=rng_sha(),epoch_seconds=time.perf_counter()-epoch_started))
            csv(folder/'training_history.csv',history)
            js(folder/'TRAINING_LEDGER.json',metadata|dict(status='TRAINING',epoch=epoch,best_epoch=best_epoch,optimizer_steps=steps,first_update_sha256=first_update))
            print(f'{cohort} {kind} Seed {seed} | Epoch {epoch}/450 | Train Loss {history[-1]["train_standardized_MSE"]:.6f} | Validation Loss {mse:.6f} | Best Epoch {best_epoch} | Patience {stale}/40',flush=True)
            if epoch==1 or epoch%10==0 or stale>=40 or epoch==450:handle=loss_figure(folder,history,kind,seed,handle)
            if stale>=40:print(f'Early Stopping {cohort} {kind} Seed {seed}: epoch {epoch}, best {best_epoch}',flush=True);break
        c.require(steps==55*len(history) and c.state_sha(model)!=initial,'UPDATE_COUNT')
        ledger=metadata|dict(status='TRAINED',best_epoch=best_epoch,stop_epoch=epoch,optimizer_steps=steps,backward_calls=steps,
            first_update_sha256=first_update,final_state_sha256=c.state_sha(model),best_checkpoint_sha256=c.sha(folder/'best_model.pt'),training_seconds=time.perf_counter()-started)
        js(folder/'TRAINING_LEDGER.json',ledger)
        return ledger
    except Exception:
        text(folder/'FAILURE.txt',traceback.format_exc())
        js(folder/'TRAINING_LEDGER.json',metadata|dict(status='FAILURE_PRESERVED',optimizer_steps=steps,best_epoch=best_epoch))
        try:checkpoint(folder/'partial_failure.pt',model,metadata|dict(partial_failure=True,optimizer_steps=steps))
        except Exception:text(folder/'PARTIAL_SAVE_FAILURE.txt',traceback.format_exc())
        raise
    finally:
        del model,optimizer;torch.cuda.empty_cache()


def load_new_model(run,ledger):
    path=run/'models'/ledger['cohort']/f"{ledger['kind']}_seed{ledger['seed']}"/'best_model.pt'
    c.require(c.sha(path)==ledger['best_checkpoint_sha256'],'NEW_CHECKPOINT_SHA')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    for key in ['run_id','cohort','kind','seed','best_epoch']:
        c.require(saved[key]==ledger[key],'CHECKPOINT_METADATA:'+key)
    c.require(saved['fresh_initialization'] and not saved['historical_checkpoint_loaded'],'NOT_FRESH')
    model=c.new_model(ledger['kind'],ledger['seed']);model.load_state_dict(saved['state_dict'],strict=True)
    return model


def pilot(run,frame,windows,scaler,prepared,protocol):
    js(run/'TRAINING_CODE_BEFORE_FIRST_MODEL.json',{name:c.sha(c.HERE/name) for name in ['core.py','training.py','io_utils.py']})
    ledgers=[]
    for kind in ['LSTM','F2']:
        row=train_one(run,'pilot',kind,202,windows,scaler,prepared,protocol);ledgers.append(row)
        model=load_new_model(run,row);pred=predict(model,prepared['VALIDATION'])*scaler.target_scale
        points=c.prediction_frame(frame,windows['VALIDATION'],pred,202,'top10','VALIDATION')
        folder=run/'models/pilot'/f'{kind}_seed202';csv(folder/'predictions_VALIDATION.csv',points)
        # Reload and independently repeat the new checkpoint inference, no historical predictions.
        del model;torch.cuda.empty_cache();model=load_new_model(run,row)
        replay=predict(model,prepared['VALIDATION'])*scaler.target_scale
        c.require(np.array_equal(replay,pred),'PILOT_REPLAY_CHANGED')
        js(folder/'PILOT_VALIDATION_METRICS.json',metrics(points));del model;torch.cuda.empty_cache()
    a=c.read_csv(run/'models/pilot/LSTM_seed202/training_history.csv');b=c.read_csv(run/'models/pilot/F2_seed202/training_history.csv')
    n=min(len(a),len(b));c.require(np.array_equal(a.sampler_order_sha256.iloc[:n],b.sampler_order_sha256.iloc[:n]),'PILOT_SAMPLER_NOT_PAIRED')
    js(run/'PILOT_GATE.json',dict(status='PASS',models=ledgers,new_checkpoint_replay_exact=True,same_inputs_labels_and_sampler=True,
        genuine_optimizer_steps=sum(r['optimizer_steps'] for r in ledgers),no_pilot_holdout_scores=True))
    print('PILOT_GATE_PASS: no holdout performance used; start twelve fresh formal instances',flush=True)


def formal_train(run,windows,scaler,prepared,protocol):
    rows=[]
    for seed in c.SUPPORTED_SEEDS:
        for kind in ['LSTM','F2']:
            rows.append(train_one(run,'formal',kind,seed,windows,scaler,prepared,protocol))
            js(run/'FORMAL_TRAINING_PROGRESS.json',dict(completed=len(rows),models=rows))
        a=c.read_csv(run/f'models/formal/LSTM_seed{seed}/training_history.csv');b=c.read_csv(run/f'models/formal/F2_seed{seed}/training_history.csv');n=min(len(a),len(b))
        c.require(np.array_equal(a.sampler_order_sha256.iloc[:n],b.sampler_order_sha256.iloc[:n]),'FORMAL_SAMPLER_NOT_PAIRED')
    js(run/'ALL_FORMAL_WEIGHTS_FROZEN.json',dict(status='ALL_12_TRAINED',models=rows,holdout_metrics_observed=False))
    print('All twelve formal best checkpoints frozen; now allow chronological holdout scoring',flush=True)
    return rows


def formal_evaluate(run,frame,windows,scaler,prepared,ledgers):
    c.require(c.read_json(run/'ALL_FORMAL_WEIGHTS_FROZEN.json')['status']=='ALL_12_TRAINED','ALL_MODELS_MUST_BE_FROZEN')
    rows=[];steps=[]
    for ledger in ledgers:
        kind,seed=ledger['kind'],ledger['seed'];model=load_new_model(run,ledger)
        for partition,w in windows.items():
            print(f'New inference {kind} Seed {seed}: {partition}, {len(w.origins)} windows',flush=True)
            pred=predict(model,prepared[partition])*scaler.target_scale
            p=c.prediction_frame(frame,w,pred,seed,'top10',partition);p['model']=kind
            c.require(np.array_equal(p.y_true_delta_V,w.y.ravel().astype(float)),'EVALUATION_LABEL_IDENTITY')
            csv(run/f'models/formal/{kind}_seed{seed}/predictions_{partition}.csv',p)
            rows.append(dict(model=kind,seed=seed,partition=partition,**metrics(p)))
            if partition=='TEST':
                for step,g in p.groupby('horizon_step'):steps.append(dict(model=kind,seed=seed,step=int(step),**metrics(g)))
        del model;torch.cuda.empty_cache()
    table=pd.DataFrame(rows);holdout=table.loc[table.partition=='TEST'].copy();means=[]
    for kind,g in holdout.groupby('model'):
        c.require(sorted(g.seed.tolist())==sorted(c.SUPPORTED_SEEDS),'INCOMPLETE_FORMAL_SEEDS')
        means.append(dict(model=kind,n=6,**{key+'_mean':math.fsum(g[key])/6 for key in ['RMSE_mV','MAE_mV','R2']},
            **{key+'_sample_sd':float(g[key].std(ddof=1)) for key in ['RMSE_mV','MAE_mV','R2']}))
    paired=[]
    for seed in c.SUPPORTED_SEEDS:
        a=holdout.loc[(holdout.model=='LSTM')&(holdout.seed==seed)].iloc[0];b=holdout.loc[(holdout.model=='F2')&(holdout.seed==seed)].iloc[0]
        paired.append(dict(seed=seed,**{kind+'_'+key:float(row[key]) for kind,row in [('LSTM',a),('F2',b)] for key in ['RMSE_mV','MAE_mV','R2']},
            **{key+'_F2_minus_LSTM':float(b[key]-a[key]) for key in ['RMSE_mV','MAE_mV','R2']}))
    csv(run/'tables/PARTITION_METRICS.csv',table);csv(run/'tables/HOLDOUT_PER_SEED.csv',holdout)
    csv(run/'tables/MODEL_MEAN_SD.csv',means);csv(run/'tables/PAIRED_DIFFERENCES.csv',paired);csv(run/'tables/STEP_METRICS.csv',steps)
    display(holdout);display(pd.DataFrame(means));display(pd.DataFrame(paired))
    return table,pd.DataFrame(means),pd.DataFrame(paired)

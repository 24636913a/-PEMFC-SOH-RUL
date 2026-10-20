"""One fresh F2 instance. Only Train/Validation exist in this training API."""
import math
import os
import time
import traceback
import numpy as np
import pandas as pd
import torch
from IPython.display import display,Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import frozen_f2 as f
import diagnostics as d
import settings as s
from io_utils import js,csv,text,metrics,enrich
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save_checkpoint(path,model,metadata):
    temporary=path.with_suffix('.pt.writing')
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},**metadata),temporary)
    for delay in [.05,.1,.2,.5,1.,None]:
        try:os.replace(temporary,path);return
        except PermissionError:
            if delay is None:raise
            time.sleep(delay)


def rng():
    return dict(cpu=f.array_sha(torch.get_rng_state().numpy()),cuda=f.array_sha(torch.cuda.get_rng_state().cpu().numpy()))


def refresh(folder,history,stage,candidate,seed,handle):
    frame=pd.DataFrame(history);fig,ax=plt.subplots(figsize=(9,3.8))
    ax.plot(frame.epoch,frame.train_unweighted_batch_MSE,label='Train unweighted batch-mean MSE',color='#0F4D92')
    ax.plot(frame.epoch,frame.validation_MSE,label='Full Validation MSE',color='black',ls='--')
    if frame.weighted_loss.iloc[0]:ax.plot(frame.epoch,frame.train_optimized_loss,label='Train weighted objective',color='#42949E',alpha=.8)
    ax.axvline(frame.best_epoch.iloc[-1],ls=':',color='gray',label='Selected best epoch')
    ax.set(xlabel='Fresh training epoch',ylabel='Standardized DeltaV MSE',title=f'{stage}/{candidate} Seed{seed}: current run')
    ax.set_yscale('log');ax.legend(fontsize=8);fig.tight_layout()
    path=folder/'training_history.png';fig.savefig(path,dpi=300,bbox_inches='tight')
    fig.savefig(folder/'training_history.pdf',bbox_inches='tight');plt.close(fig)
    value=Image(filename=str(path))
    if handle is None:return display(value,display_id=True)
    handle.update(value);return handle


def grouped_metrics(points,frame,threshold):
    p=enrich(points,frame);e=event_mask(frame,threshold)[p.target_row.to_numpy(int)]
    result=metrics(p)
    result.update(P95_mV=float(p.absolute_error_mV.quantile(.95)),P99_mV=float(p.absolute_error_mV.quantile(.99)),
        max_abs_mV=float(p.absolute_error_mV.max()))
    for key,mask in [('event',e),('non_event',~e)]:
        result.update({key+'_'+k:v for k,v in metrics(p.loc[mask]).items()})
    return result


def event_mask(frame,threshold):
    voltage=frame.cell_voltage_mean.to_numpy(float);jump=np.r_[np.nan,np.diff(voltage)]
    same=frame.temporal_segment_id.eq(frame.temporal_segment_id.shift()).to_numpy() & frame.bin_end_ophrs2_h.diff().sub(.5).abs().lt(1e-6).to_numpy()
    z=frame.zero_current_fraction.to_numpy(float)
    return ((frame.startup_flag.to_numpy()>0)|(frame.shutdown_flag.to_numpy()>0)|((z>0)&(z<1))|
        (same & (abs(jump)>=threshold)))


def train_one(run,stage,candidate,seed,config,frame,windows,scaler,prepared):
    f.require(set(windows)==set(prepared)=={'TRAIN','VALIDATION'},'SELECTION_DATA_ONLY')
    frozen=f.read_json(run/'CODE_BEFORE_A.json')
    for name,digest in frozen.items():f.require(f.sha(f.HERE/name)==digest,'FROZEN_TRAINING_CODE_CHANGED:'+name)
    protocol=f.read_json(run/'PROTOCOL.json')
    for part in windows:
        f.require(f.array_sha(prepared[part])==protocol['partitions'][part]['input_sha256'],'INPUT_IDENTITY')
    folder=run/'models'/stage/f'{candidate}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    model=d.new_model(seed,config);initial=f.state_sha(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=s.WEIGHT_DECAY)
    f.require(not optimizer.state,'OPTIMIZER_NOT_FRESH')
    labels={part:(w.y/scaler.target_scale).astype('float32') for part,w in windows.items()}
    weighted=config['loss']=='EVENT_MSE'
    if weighted:
        f.require(stage in ['C','D'],'NO_PREMATURE_WEIGHTING')
        e=event_mask(frame,protocol['event_definition']['Train_jump_P95_V'])[windows['TRAIN'].targets]
        weights=(1+e.astype('float32'));weights/=weights.mean()
    else:weights=np.ones_like(labels['TRAIN'])
    generator=torch.Generator().manual_seed(seed)
    loader=f.DataLoader(f.TensorDataset(torch.from_numpy(prepared['TRAIN']),torch.from_numpy(labels['TRAIN']),
        torch.from_numpy(weights),torch.arange(len(labels['TRAIN']))),batch_size=32,shuffle=True,
        generator=generator,num_workers=0,drop_last=False)
    metadata=dict(run_id=run.name,stage=stage,candidate=candidate,seed=seed,configuration=config,
        parameters=sum(p.numel() for p in model.parameters()),initialization_sha256=initial,
        fresh_initialization=True,historical_checkpoint_loaded=False,optimizer_initial_state_empty=True,
        initial_rng=rng(),protocol_sha256=f.sha(run/'PROTOCOL.json'),
        train_inputs_sha256=f.array_sha(prepared['TRAIN']),validation_inputs_sha256=f.array_sha(prepared['VALIDATION']),
        target_scale_V=scaler.target_scale,loss_weight_sha256=f.array_sha(weights),
        early_stopping='minimum unweighted full Validation RMSE; min_delta 1e-12, patience40')
    js(folder/'BEFORE_TRAINING.json',metadata)
    history=[];best=math.inf;best_epoch=stale=steps=0;first=None;handle=None;started=time.perf_counter()
    print(f'{stage}/{candidate} Seed{seed}: FRESH GPU training, {metadata["parameters"]} parameters, lr={config["learning_rate"]}, dropout={config["dropout"]}',flush=True)
    try:
        for epoch in range(1,451):
            model.train();f.require(model._fusion_mode=='normal','TRAINING_FUSION_NOT_NORMAL')
            optimized=[];unweighted=[];norms=[];order=[];probe={};epoch_start=time.perf_counter()
            for batch,(xb,yb,wb,indices) in enumerate(loader):
                xb,yb,wb=xb.cuda(),yb.cuda(),wb.cuda();optimizer.zero_grad(set_to_none=True)
                predicted=model.predict_delta(xb)
                plain=torch.nn.functional.mse_loss(predicted,yb)
                loss=(torch.nn.functional.mse_loss(predicted,yb,reduction='none')*wb).mean() if weighted else plain
                f.require(bool(torch.isfinite(loss)),'NONFINITE_LOSS');loss.backward()
                if batch==0:
                    probe=d.gradient_row(model)
                    probe.update({k:float(np.mean(v)) for k,v in d.path_rows(model).items()})
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                f.require(bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),'NONFINITE_GRADIENT')
                optimizer.step();steps+=1
                if first is None:first=f.state_sha(model);f.require(first!=initial,'NO_OPTIMIZER_UPDATE')
                optimized.append(float(loss.detach().cpu()));unweighted.append(float(plain.detach().cpu()))
                norms.append(float(norm));order.extend(indices.tolist());d.clear(model)
            prediction,_=d.predict(model,prepared['VALIDATION'])
            mse=float(np.mean((prediction.astype('float64')-labels['VALIDATION'].astype('float64'))**2));score=math.sqrt(mse)
            f.require(math.isfinite(score),'NONFINITE_VALIDATION')
            if score<best-1e-12:
                best,best_epoch,stale=score,epoch,0
                save_checkpoint(folder/'best_model.pt',model,metadata|dict(best_epoch=epoch,optimizer_steps=steps))
            else:stale+=1
            history.append(dict(epoch=epoch,train_optimized_loss=float(np.mean(optimized)),train_unweighted_batch_MSE=float(np.mean(unweighted)),
                weighted_loss=weighted,validation_MSE=mse,best_epoch=best_epoch,stale_epochs=stale,
                optimizer_steps=steps,backward_calls=steps,gradient_norm_min=min(norms),gradient_norm_max=max(norms),
                clipped_fraction=float(np.mean(np.asarray(norms)>1)),sampler_order_sha256=f.array_sha(np.asarray(order,np.int64)),
                state_sha256=f.state_sha(model),rng=rng(),epoch_seconds=time.perf_counter()-epoch_start,**probe))
            csv(folder/'training_history.csv',history)
            js(folder/'TRAINING_LEDGER.json',metadata|dict(status='TRAINING',epoch=epoch,best_epoch=best_epoch,optimizer_steps=steps,first_update_sha256=first))
            print(f'{stage}/{candidate} Seed{seed} | Epoch {epoch}/450 | Train {history[-1]["train_unweighted_batch_MSE"]:.6f} | Val {mse:.6f} | Best {best_epoch} | Patience {stale}/40',flush=True)
            if epoch==1 or epoch%10==0 or stale>=40 or epoch==450:handle=refresh(folder,history,stage,candidate,seed,handle)
            if stale>=40:break
        f.require(steps==55*len(history),'UPDATE_COUNT')
        ledger=metadata|dict(status='TRAINED',stop_epoch=epoch,best_epoch=best_epoch,optimizer_steps=steps,backward_calls=steps,
            first_update_sha256=first,final_state_sha256=f.state_sha(model),best_checkpoint_sha256=f.sha(folder/'best_model.pt'),
            training_seconds=time.perf_counter()-started)
        js(folder/'TRAINING_LEDGER.json',ledger)
    except Exception:
        text(folder/'FAILURE.txt',traceback.format_exc())
        save_checkpoint(folder/'partial_failure.pt',model,metadata|dict(optimizer_steps=steps,partial_failure=True))
        js(folder/'TRAINING_LEDGER.json',metadata|dict(status='FAILURE_PRESERVED',optimizer_steps=steps));raise
    finally:
        del model,optimizer;torch.cuda.empty_cache()
    evaluate_one(run,ledger,frame,windows,scaler,prepared)
    return ledger


def load_new(run,ledger):
    folder=run/'models'/ledger['stage']/f"{ledger['candidate']}_seed{ledger['seed']}"
    f.require(f.sha(folder/'best_model.pt')==ledger['best_checkpoint_sha256'],'CHECKPOINT_SHA')
    saved=torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)
    for key in ['run_id','stage','candidate','seed','configuration','initialization_sha256','best_epoch']:
        f.require(saved[key]==ledger[key],'CHECKPOINT_IDENTITY:'+key)
    model=d.new_model(ledger['seed'],ledger['configuration']);model.load_state_dict(saved['state_dict'],strict=True)
    return model,folder


def evaluate_one(run,ledger,frame,windows,scaler,prepared):
    model,folder=load_new(run,ledger);all_metrics=[];summaries=[]
    protocol=f.read_json(run/'PROTOCOL.json');threshold=protocol['event_definition']['Train_jump_P95_V']
    for part,w in windows.items():
        pred,paths=d.predict(model,prepared[part],observe=True);pred*=scaler.target_scale
        points=f.prediction_frame(frame,w,pred,ledger['seed'],'top10',part)
        csv(folder/f'predictions_{part}.csv',points)
        all_metrics.append(dict(partition=part,**grouped_metrics(points,frame,threshold)))
        path_frame=pd.DataFrame(paths);path_frame['origin_row']=w.origins;path_frame['origin_age_h']=w.origin_age_h
        for index,name in zip(model.regime_indices,f.REGIME_FEATURE_NAMES):
            path_frame['history_scaled_'+name]=prepared[part][:,:,index].mean(axis=1)
        csv(folder/f'paths_{part}.csv',path_frame);summaries.append(dict(partition=part,**d.summarize_paths(paths)))
        if part=='VALIDATION':
            replay,_=d.predict(model,prepared[part]);replay*=scaler.target_scale
            f.require(np.array_equal(replay,pred),'NEW_CHECKPOINT_REPLAY')
    csv(folder/'METRICS.csv',all_metrics);csv(folder/'PATH_SUMMARY.csv',summaries)
    normal=f.read_csv(folder/'predictions_VALIDATION.csv');before=f.state_sha(model);counter=[]
    try:
        for mode in ['cnn_off','lstm_only']:
            model._fusion_mode=mode;pred,_=d.predict(model,prepared['VALIDATION']);pred*=scaler.target_scale
            p=f.prediction_frame(frame,windows['VALIDATION'],pred,ledger['seed'],'top10','VALIDATION')
            csv(folder/f'counterfactual_{mode}_VALIDATION.csv',p)
            change=(p.y_pred_delta_V-normal.y_pred_delta_V).to_numpy(float)*1000
            counter.append(dict(mode=mode,**grouped_metrics(p,frame,threshold),prediction_change_RMS_mV=float(np.sqrt(np.mean(change**2))),
                max_prediction_change_mV=float(abs(change).max())))
    finally:model._fusion_mode='normal'
    f.require(f.state_sha(model)==before,'COUNTERFACTUAL_CHANGED_WEIGHTS')
    csv(folder/'COUNTERFACTUAL_VALIDATION.csv',counter)
    js(folder/'EVALUATION_AUDIT.json',dict(status='PASS',Validation_replay_exact=True,counterfactual_weights_unchanged=True,
        holdout_evaluated=False,all_Train_Validation_queries_retained=True))
    del model;torch.cuda.empty_cache()

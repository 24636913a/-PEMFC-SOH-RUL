"""One registered from-scratch GPU training; successful run IDs never re-enter."""
from baseline_common import *
import argparse
from torch.utils.data import DataLoader,TensorDataset

def main(name,seed):
    c=cfg();assert (name,seed) in RUNS
    pre=read(P/'outputs/audits/PREFLIGHT_REPORT.json');assert pre['status']=='PASS'
    run=f'{name}_seed{seed}';folder=path(f'outputs/training_logs/{run}/run_state.json').parent
    assert not (folder/'run_state.json').exists(),'RUN_ALREADY_ENTERED_NO_RETRAIN'
    js(folder.relative_to(P)/'run_state.json',dict(status='TRAINING',model=name,seed=seed,last_completed_epoch=0));event('RUN_STARTED',model=name,seed=seed)
    try:
        env=h.configure_normal(c);d=data();scale=read(P/'outputs/audits/SCALER.json')['target_scale_V'];tx,vx=d['tx'],d['vx'];ty,vy=d['train_y']/scale,d['val_y']/scale
        h.e.set_seed(seed);device=torch.device('cuda');m=model(name,device);initial=h.e.state_digest(m);count=sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert count==c['parameter_counts'][name]
        opt=torch.optim.AdamW(m.parameters(),lr=c['hyperparameters']['learning_rate'],weight_decay=c['hyperparameters']['weight_decay'])
        gen=torch.Generator().manual_seed(seed);loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty.astype(np.float32)),torch.arange(len(tx))),batch_size=32,shuffle=True,generator=gen,num_workers=0)
        js(folder.relative_to(P)/'initial.json',dict(initialization_sha256=initial,seed=seed,environment=env,rng=h.rng_summary(gen),parameter_count=count,protocol_sha256=sha(P/'configs/FROZEN_BASELINE_PROTOCOL.yaml')))
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();best=math.inf;best_epoch=stale=0;logs=[]
        checkpoint=path(f'outputs/checkpoints/{run}_best.pt');journal=folder/'batch_events.jsonl'
        for epoch in range(1,451):
            tick=time.perf_counter();m.train();losses=[];order=[];max_norm=0.
            with journal.open('a',encoding='utf-8',newline='\n') as stream:
                for batch,(xb,yb,ids) in enumerate(loader):
                    order.extend(d['train_origins'][ids.numpy()].tolist());xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
                    if epoch==1 and batch==0:js(folder.relative_to(P)/'first_batch.json',dict(input_sha256=h.array_hash(xb.cpu().numpy()),label_sha256=h.array_hash(yb.cpu().numpy()),origins=d['train_origins'][ids.numpy()].tolist(),initialization_sha256=initial))
                    output=m(xb);loss=torch.nn.functional.mse_loss(output,yb)
                    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
                    if not torch.isfinite(norm) or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise RuntimeError('NONFINITE_GRADIENT')
                    opt.step();v=float(loss.detach().cpu());losses.append(v);max_norm=max(max_norm,float(norm.detach().cpu()))
                    stream.write(json.dumps(dict(epoch=epoch,batch=batch,train_standardized_MSE=v,gradient_norm_before_clip=float(norm.detach().cpu()),sampler_ids_sha256=h.array_hash(ids.numpy())))+'\n')
                stream.flush();os.fsync(stream.fileno())
            pred=predict(m,vx,device);rmse=float(np.sqrt(np.mean((pred-vy)**2)))
            if not math.isfinite(rmse):raise RuntimeError('NONFINITE_VALIDATION')
            if rmse<best-1e-12:
                best,best_epoch,stale=rmse,epoch,0;tmp=checkpoint.with_suffix('.pt.writing')
                torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},model=name,seed=seed,best_epoch=epoch,architecture=ARCH[name],parameter_count=count,target_scale_V=scale,protocol_sha256=sha(P/'configs/FROZEN_BASELINE_PROTOCOL.yaml'),input_fingerprints=read(P/'outputs/audits/DATA_FINGERPRINTS.json'),initialization_sha256=initial,full_validation_prediction=torch.from_numpy(pred.copy())),tmp);replace(tmp,checkpoint)
            else:stale+=1
            row=dict(epoch=epoch,train_standardized_MSE=float(np.mean(losses)),validation_standardized_RMSE=rmse,validation_RMSE_mV=rmse*scale*1000,best_epoch=best_epoch,stale_epochs=stale,epoch_time_s=time.perf_counter()-tick,max_gradient_norm_before_clip=max_norm,sampler_origin_order_sha256=h.array_hash(np.asarray(order)))
            logs.append(row);cs(folder.relative_to(P)/'epochs.csv',logs);js(folder.relative_to(P)/'run_state.json',dict(status='TRAINING',model=name,seed=seed,last_completed_epoch=epoch,best_epoch=best_epoch))
            if epoch%10==0 or epoch==1:print(f'{run} epoch={epoch} RMSE={row["validation_RMSE_mV"]:.6f} best={best_epoch} stale={stale}',flush=True)
            if stale>=40:break
        torch.cuda.synchronize();duration=time.perf_counter()-started;peak=int(torch.cuda.max_memory_allocated());del m,opt;torch.cuda.empty_cache()
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert ck['model']==name and ck['seed']==seed and ck['protocol_sha256']==sha(P/'configs/FROZEN_BASELINE_PROTOCOL.yaml')
        reloaded=model(name,device);reloaded.load_state_dict(ck['state_dict'],strict=True);actual=predict(reloaded,vx,device)
        assert np.array_equal(actual,ck['full_validation_prediction'].numpy()),'CHECKPOINT_RELOAD_NOT_EXACT'
        f=frame(name,seed,actual*scale,d);cs(f'outputs/predictions/{run}.csv',f);saved=rd(P/f'outputs/predictions/{run}.csv');assert_keys(saved)
        r=dict(status='COMPLETED',model=name,seed=seed,**metrics(saved),best_epoch=best_epoch,stop_epoch=epoch,stale_epochs=stale,parameter_count=count,training_time_s=duration,peak_GPU_memory_bytes=peak,environment=env,checkpoint=checkpoint.relative_to(P).as_posix(),checkpoint_sha256=sha(checkpoint),predictions=f'outputs/predictions/{run}.csv',prediction_sha256=sha(P/f'outputs/predictions/{run}.csv'),protocol_sha256=sha(P/'configs/FROZEN_BASELINE_PROTOCOL.yaml'),initialization_sha256=initial,best_validation_standardized_RMSE=best,checkpoint_reload_all205_exact=True)
        assert abs(r['RMSE_mV']-best*scale*1000)<1e-5,'BEST_CHECKPOINT_METRIC_MISMATCH'
        r['horizon_metrics']=[dict(step=int(step),n=len(g),RMSE_mV=metrics(g)['RMSE_mV'] if step==24 else float(np.sqrt(np.mean(((g.y_pred_delta_V-g.y_true_delta_V)*1000)**2))),MAE_mV=float(np.mean(abs((g.y_pred_delta_V-g.y_true_delta_V)*1000)))) for step,g in saved.groupby('horizon_step')]
        js(f'outputs/metrics/{run}.json',r);js(folder.relative_to(P)/'run_state.json',dict(status='COMPLETED',model=name,seed=seed,last_completed_epoch=epoch));event('RUN_COMPLETED',model=name,seed=seed,RMSE_mV=r['RMSE_mV']);print(json.dumps(r,ensure_ascii=False),flush=True)
    except Exception as exc:
        js(folder.relative_to(P)/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));js(folder.relative_to(P)/'run_state.json',dict(status='FAILED',model=name,seed=seed));event('RUN_FAILED',model=name,seed=seed,error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['LSTM','TCN'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.model,a.seed)

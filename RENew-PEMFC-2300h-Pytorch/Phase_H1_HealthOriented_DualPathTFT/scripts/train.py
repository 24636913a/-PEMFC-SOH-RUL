"""Registered inner epoch selection + independent full-Train refit; no outer selection."""
from common import *
sys.path.insert(0,str(P/'models'))
from event_tft import create_model
from engineering import hash_state
import torch,traceback,datetime
from torch.utils.data import TensorDataset,DataLoader

def configure():
 if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no silent device substitution')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True

def infer(model,x,device):
 model.eval();out=[]
 with torch.no_grad():
  for i in range(0,len(x),32):out.append(model.predict_voltage(torch.from_numpy(x[i:i+32]).to(device)).cpu().numpy().reshape(-1))
 return np.concatenate(out)

def save_torch(path,obj):
 path=Path(path);assert path.resolve().is_relative_to(P);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pt.writing');torch.save(obj,tmp);os.replace(tmp,path)

def fit_pass(run_id,stage,group,seed,x,y,selection,params,budget,device):
 set_seed(seed);model=create_model(group,params,device);initial=hash_state(model)
 authority=json.loads((P/'outputs/ENGINEERING_EVIDENCE.json').read_text())['initialization'][str(seed)][group]
 assert initial==authority,'Shared initialization changed'
 optimizer=torch.optim.AdamW(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay'])
 generator=torch.Generator().manual_seed(seed);loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y.astype('float32').reshape(-1,1))),batch_size=32,shuffle=True,generator=generator,num_workers=0)
 folder=P/'outputs/training_logs'/run_id;folder.mkdir(parents=True,exist_ok=True);rows=[];best=float('inf');best_epoch=0;stale=0;started=time.perf_counter()
 log=(folder/f'{stage}_epochs.jsonl').open('w',encoding='utf-8',newline='\n');journal=(folder/f'{stage}_batch_journal.jsonl').open('w',encoding='utf-8',newline='\n')
 checkpoint=P/'outputs/checkpoints'/f'{run_id}_{stage}.pt';config_sha=sha(P/'configs/FROZEN_PROTOCOL.json')
 try:
  for epoch in range(1,budget+1):
   model.train();losses=[];maximum_grad=0.
   for batch,(xb,yb) in enumerate(loader):
    stamp=dict(stage=stage,epoch=epoch,batch=batch,inputs_sha256=digest_array(xb.numpy()))
    journal.write(json.dumps(dict(**stamp,event='BEFORE_FORWARD_BACKWARD'))+'\n');journal.flush();os.fsync(journal.fileno())
    xb=xb.to(device);yb=yb.to(device);optimizer.zero_grad(set_to_none=True);prediction=model.predict_voltage(xb)
    loss=torch.nn.functional.mse_loss(prediction,yb)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
    optimizer.step();maximum_grad=max(maximum_grad,float(norm));losses.append(float(loss.detach().cpu()))
    journal.write(json.dumps(dict(**stamp,event='OPTIMIZER_STEP_COMPLETED',loss=losses[-1],gradient_norm=float(norm)))+'\n');journal.flush()
   row=dict(epoch=epoch,train_standardized_MSE=float(np.mean(losses)),maximum_preclip_gradient_norm=maximum_grad,elapsed_seconds=time.perf_counter()-started)
   if selection is not None:
    sx,sy=selection;p=infer(model,sx,device);score=float(np.sqrt(np.mean((p-sy)**2)))
    if not np.isfinite(score):raise RuntimeError('Nonfinite inner selection RMSE')
    if score<best-1e-12:
     best=score;best_epoch=epoch;stale=0
     save_torch(checkpoint,dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},run_id=run_id,stage=stage,group=group,seed=seed,selected_epoch=epoch,protocol_sha256=config_sha))
    else:stale+=1
    row.update(inner_selection_standardized_RMSE=score,best_epoch=best_epoch,stale_epochs=stale)
   else:best_epoch=epoch
   rows.append(row);log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
   # Local ignored recovery state; no thousands of per-epoch files.
   save_torch(P/'outputs/runtime'/f'{run_id}_{stage}_resume.pt',dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),epoch=epoch,
    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state(),best_epoch=best_epoch,protocol_sha256=config_sha))
   if epoch%25==0:print(f'{run_id} {stage} epoch={epoch} best_epoch={best_epoch}',flush=True)
   if selection is not None and stale>=params['patience']:break
  if selection is None:
   save_torch(checkpoint,dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},run_id=run_id,stage=stage,group=group,seed=seed,selected_epoch=best_epoch,protocol_sha256=config_sha))
  metadata=dict(stage=stage,optimizer_epochs=len(rows),selected_epoch=best_epoch,fit_seconds=time.perf_counter()-started,initialization_sha256=initial,
   checkpoint=checkpoint.relative_to(P).as_posix(),checkpoint_sha256=sha(checkpoint),selection='Train_inner_only' if selection is not None else 'fixed_epoch_selected_in_inner; no_outer_validation_early_stop')
  if selection is not None:metadata['best_inner_standardized_RMSE']=best
  return model,metadata
 finally:log.close();journal.close()

def worker(fold,seed,group):
 cfg=load_cfg();assert protected_pass();configure();run_id=f'{fold}_{group}_seed{seed}';folder=f'outputs/training_logs/{run_id}'
 if (P/folder/'RUN_STATE.json').exists():raise RuntimeError('Run exists; never silently restart or train twice:'+run_id)
 for f,h in cfg['training_code_sha256'].items():assert sha(P/f)==h,'Frozen training code changed:'+f
 js(folder+'/RUN_STATE.json',dict(status='RUNNING',run_id=run_id,start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 started=time.perf_counter()
 try:
  samples=rd(P/'data_contracts/USED_SAMPLES.csv');g=samples.loc[samples.fold.eq(fold)];tr=g.loc[g.role.eq('train')];va=g.loc[g.role.eq('validation')]
  inner=rd(P/'data_contracts/INNER_SPLITS.csv');inner=inner.loc[inner.fold.eq(fold)];fit=tr.loc[tr.sample_key.isin(inner.loc[inner.inner_role.eq('fit'),'sample_key'])];sel=tr.loc[tr.sample_key.isin(inner.loc[inner.inner_role.eq('selection'),'sample_key'])]
  raw=np.load(P/'data_contracts/RAW_DYNAMIC_HISTORIES.npz')['x'];records=json.loads((P/'data_contracts/TRAIN_ONLY_PREPROCESSING.json').read_text())[fold]
  ip=records['inner'];op=records['outer'];params=cfg['hyperparameters'];device=torch.device('cuda')
  ix=prep(raw[fit.array_index],ip);iy=(fit.true_voltage_V.to_numpy()-ip['y_mean_V'])/ip['y_scale_V']
  sx=prep(raw[sel.array_index],ip);sy=(sel.true_voltage_V.to_numpy()-ip['y_mean_V'])/ip['y_scale_V']
  _,selection=fit_pass(run_id,'inner',group,seed,ix,iy,(sx,sy),params,params['max_epochs'],device)
  tx=prep(raw[tr.array_index],op);ty=(tr.true_voltage_V.to_numpy()-op['y_mean_V'])/op['y_scale_V']
  model,final=fit_pass(run_id,'final',group,seed,tx,ty,None,params,selection['selected_epoch'],device)
  # Final weights are frozen before outerValidation is transformed/predicted.
  path=P/final['checkpoint'];clone=create_model(group,params,device);clone.load_state_dict(torch.load(path,map_location=device,weights_only=True)['state_dict'])
  probe=tx[:min(4,len(tx))];assert np.array_equal(infer(model,probe,device),infer(clone,probe,device))
  elapsed=time.perf_counter()-started
  last_label=pd.to_datetime(tr.target_available_calendar).max();first_origin=pd.to_datetime(va.origin_confirmed_calendar).min()
  replay_available=last_label+pd.Timedelta(seconds=elapsed)
  inner_available=pd.to_datetime(fit.target_available_calendar).max()+pd.Timedelta(seconds=selection['fit_seconds'])
  if replay_available>=first_origin or inner_available>=pd.to_datetime(sel.origin_confirmed_calendar).min():raise RuntimeError('Measured training runtime does not fit historical availability interval')
  vx=prep(raw[va.array_index],op);standard=infer(clone,vx,device);voltage=standard.astype('float64')*op['y_scale_V']+op['y_mean_V']
  d=va.copy();d['run_id']=run_id;d['model']='H-'+group;d['seed']=seed;d['predicted_voltage_V']=voltage;d['standardized_output']=standard
  d['predicted_SOH_V_percent']=100*voltage/VREF;d['true_SOH_V_percent']=100*d.true_voltage_V/VREF
  d['replay_model_available_calendar']=str(replay_available);d['real_fit_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['historical_online_deployment_claimed']=False
  cs(f'outputs/predictions/{run_id}.csv',d)
  js(folder+'/COMPLETION.json',dict(status='COMPLETE',run_id=run_id,inner=selection,final=final,voltage_queries=len(d),
   total_training_seconds=elapsed,latest_Train_label_calendar=str(last_label),replay_model_available_calendar=str(replay_available),
   earliest_Validation_origin_calendar=str(first_origin),inner_model_replay_available_calendar=str(inner_available),
   CPU_wall_UTC_is_not_historical_deployment=True,checkpoint_reload_exact=True,outer_used_for_selection=False,optimizer_fits=2))
  js(folder+'/RUN_STATE.json',dict(status='COMPLETE',run_id=run_id,optimizer_fits=2));print(run_id,'COMPLETE',flush=True)
 except Exception as e:
  js(folder+'/FAILURE.json',dict(exception=repr(e),traceback=traceback.format_exc(),no_automatic_retraining=True));js(folder+'/RUN_STATE.json',dict(status='FAILED',run_id=run_id));raise

def main():
 cfg=load_cfg()
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]),sys.argv[4]);return
 assert (P/'outputs/PRETRAINING_COMMIT.json').exists(),'Commit frozen protocol before GPU training'
 for fold in ['fold_1','fold_2','fold_3']:
  for seed in [42,52,62]:
   for group in ['F0','F1','F2']:
    run=f'{fold}_{group}_seed{seed}';state=P/f'outputs/training_logs/{run}/RUN_STATE.json'
    if state.exists():
     status=json.loads(state.read_text())['status']
     if status=='COMPLETE':continue
     raise RuntimeError('Existing noncompleted run requires explicit engineering repair/resume review:'+run)
    result=subprocess.run([sys.executable,'-B',str(P/'scripts/train.py'),'worker',fold,str(seed),group],cwd=REPO)
    if result.returncode:raise RuntimeError('Worker failed; stop registered sweep')
 js('outputs/ALL_GPU_RUNS_COMPLETE.json',dict(status='COMPLETE',registered_runs=27,optimizer_fit_passes=54,outer_early_stopping=False,trained_on_GPU=True))
if __name__=='__main__':main()

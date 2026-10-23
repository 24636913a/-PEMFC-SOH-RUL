"""One new F2 seed; same scientific update and selection sequence as Phase H."""
from screen_common import *
import argparse,math
from torch.utils.data import DataLoader,TensorDataset
class Recorder:
 def __init__(self,folder,seed):self.folder=folder;self.seed=seed;self.sequence=0;self.state={}
 def emit(self,stage,**payload):
  if stage=='BATCH_STARTED':self.state={'seed':self.seed,'stages':[]}
  self.state.update(payload);self.state['stage']=stage;self.state['stages'].append(stage);self.sequence+=1;self.state['sequence']=self.sequence
  with (self.folder/'batch_events.jsonl').open('a',encoding='utf-8',newline='\n') as s:
   s.write(json.dumps({'seed':self.seed,'sequence':self.sequence,'stage':stage,'epoch':self.state.get('epoch'),'batch':self.state.get('batch'),**payload},allow_nan=False)+'\n');s.flush();os.fsync(s.fileno())
  js(self.folder.relative_to(P)/'latest_batch.json',self.state)
def main(seed):
 from repair_identity import verify_repair
 verify_repair()
 c=cfg();assert seed in c['seeds'];folder=P/f'outputs/training_logs/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 assert not (folder/'run_state.json').exists(),'RUN_ALREADY_ENTERED_NO_RETRY'
 js(folder.relative_to(P)/'run_state.json',dict(status='TRAINING',seed=seed,last_completed_epoch=0))
 try:
  env=h.configure_normal(c);d=data();scale=json.loads((P/'outputs/audits/SCALER.json').read_text())['target_scale_V'];tx=d['tx'];vx=d['vx'];ty=d['train_y']/scale;vy=d['val_y']/scale
  h.e.set_seed(seed);device=torch.device('cuda');params=c['hyperparameters'];m=h.model('F2',params,device);init=h.e.state_digest(m)
  opt=torch.optim.AdamW(m.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay']);gen=torch.Generator().manual_seed(seed)
  loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty.astype(np.float32)),torch.arange(len(tx))),batch_size=32,shuffle=True,generator=gen,num_workers=0)
  js(folder.relative_to(P)/'initial.json',dict(seed=seed,initialization_sha256=init,rng=h.rng_summary(gen),environment=env,protocol_sha256=sha(P/'configs/frozen_seed_screening.json')))
  rec=Recorder(folder,seed);best=math.inf;best_epoch=stale=0;history=[];torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
  checkpoint=P/f'outputs/checkpoints/F2_seed_{seed}_best.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
  for epoch in range(1,451):
   epoch_start=time.perf_counter();m.train();losses=[];order=[]
   for batch,(xb,yb,ids) in enumerate(loader):
    origins=d['train_origins'][ids.numpy()].tolist();order.extend(origins);xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
    rec.emit('BATCH_STARTED',epoch=epoch,batch=batch,origins=origins,input_sha256=h.array_hash(xb.cpu().numpy()),label_sha256=h.array_hash(yb.cpu().numpy()),model_state_sha256=h.tree_hash(m.state_dict()),rng=h.rng_summary(gen))
    pred=m.predict_delta(xb);a=pred.detach().cpu().numpy();rec.emit('FORWARD_COMPLETED',output_sha256=h.array_hash(a),output_shape=list(a.shape),output_mean=float(a.mean()),output_std=float(a.std()))
    loss=torch.nn.functional.mse_loss(pred,yb)
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
    rec.emit('LOSS_COMPUTED',loss=float(loss.detach().cpu()))
    if epoch==1 and batch==0:js(folder.relative_to(P)/'first_batch.json',rec.state)
    rec.emit('BACKWARD_STARTED');loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
    if not torch.isfinite(norm) or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise RuntimeError('NONFINITE_GRADIENT')
    rec.emit('BACKWARD_COMPLETED',gradient_norm=float(norm.detach().cpu()));opt.step();rec.emit('OPTIMIZER_STEP_COMPLETED',parameters_after_sha256=h.tree_hash(m.state_dict()));losses.append(float(loss.detach().cpu()))
   pred=h.e.predict(m,vx,32,device);rmse=float(np.sqrt(np.mean((pred-vy)**2)))
   if not math.isfinite(rmse):raise RuntimeError('NONFINITE_VALIDATION')
   if rmse<best-1e-12:
    best,best_epoch,stale=rmse,epoch,0;tmp=checkpoint.with_suffix('.pt.writing')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()},seed=seed,model='F2',parameter_count=332436,protocol_sha256=sha(P/'configs/frozen_seed_screening.json'),
     best_epoch=best_epoch,target_scale_V=scale,input_fingerprints=json.loads((P/'outputs/audits/DATA_FINGERPRINTS.json').read_text()),initialization_sha256=init,fixed_validation_prediction=torch.from_numpy(pred[:32].copy())),tmp);os.replace(tmp,checkpoint)
   else:stale+=1
   row=dict(epoch=epoch,train_standardized_MSE=float(np.mean(losses)),validation_standardized_RMSE=rmse,validation_RMSE_mV=rmse*scale*1000,best_epoch=best_epoch,stale_epochs=stale,epoch_time_s=time.perf_counter()-epoch_start,sampler_origin_order_sha256=h.array_hash(np.asarray(order)),rng_after_validation=h.rng_summary(gen))
   history.append(row);js(folder.relative_to(P)/'epochs.json',history);cs(folder.relative_to(P)/'epochs.csv',pd.DataFrame([{k:v for k,v in r.items() if k!='rng_after_validation'} for r in history]))
   js(folder.relative_to(P)/'run_state.json',dict(status='TRAINING',seed=seed,last_completed_epoch=epoch,best_epoch=best_epoch))
   print(f'Seed{seed} epoch={epoch} RMSE_mV={row["validation_RMSE_mV"]:.6f} best={best_epoch} stale={stale}',flush=True)
   if stale>=40:break
  duration=time.perf_counter()-started;peak=int(torch.cuda.max_memory_allocated());del m,opt
  ck=torch.load(checkpoint,map_location='cpu',weights_only=True);assert ck['seed']==seed and ck['protocol_sha256']==sha(P/'configs/frozen_seed_screening.json'),'CHECKPOINT_IDENTITY_MISMATCH'
  reloaded=h.model('F2',params,device);reloaded.load_state_dict(ck['state_dict']);fixed=h.e.predict(reloaded,vx[:32],32,device)
  assert np.array_equal(fixed,ck['fixed_validation_prediction'].numpy()),'CHECKPOINT_RELOAD_MISMATCH'
  pred=h.e.predict(reloaded,vx,32,device)*scale
  frame=pd.DataFrame(dict(seed=seed,origin_row=np.repeat(d['val_origins'],24),target_row=d['val_targets'].ravel(),horizon_step=np.tile(np.arange(1,25),len(vx)),
   origin_age_h=np.repeat(d['val_origin_age_h'],24),target_age_h=d['val_target_age_h'].ravel(),origin_raw_voltage_V=np.repeat(d['val_origin_raw_voltage_V'],24),
   y_true_delta_V=d['val_y'].ravel(),y_pred_delta_V=pred.ravel(),true_raw_target_voltage_V=d['val_target_raw_voltage_V'].ravel()))
  frame['true_label_voltage_V']=frame.origin_raw_voltage_V+frame.y_true_delta_V;frame['predicted_voltage_V']=frame.origin_raw_voltage_V+frame.y_pred_delta_V
  path=f'outputs/predictions/F2_seed_{seed}.csv';cs(path,frame);saved=rd(P/path)
  authority=rd(P/'outputs/audits/VALIDATION_KEY_TRUTH.csv');assert np.array_equal(saved[authority.columns].to_numpy(),authority.to_numpy()),'PREDICTION_TRUTH_IDENTITY_MISMATCH'
  metrics=metric(saved);official=h.e.numpy_metrics(saved)
  for k in official:assert abs(metrics[k]-official[k])<1e-10,'METRIC_MISMATCH'
  result=dict(status='COMPLETED',seed=seed,**metrics,best_epoch=best_epoch,stop_epoch=epoch,patience_early_stopped=stale>=40,training_time_s=duration,peak_GPU_memory_bytes=peak,
   checkpoint=checkpoint.relative_to(P).as_posix(),checkpoint_sha256=sha(checkpoint),predictions=path,prediction_sha256=sha(P/path),initialization_sha256=init,
   protocol_sha256=sha(P/'configs/frozen_seed_screening.json'),best_validation_RMSE_mV=best*scale*1000,checkpoint_reload_exact=True)
  assert abs(result['RMSE_mV']-best*scale*1000)<1e-5,'BEST_CHECKPOINT_METRIC_MISMATCH'
  js(f'outputs/metrics/seed_{seed}.json',result);js(folder.relative_to(P)/'run_state.json',dict(status='COMPLETED',seed=seed,last_completed_epoch=epoch));print(json.dumps(result),flush=True)
 except Exception as exc:
  js(folder.relative_to(P)/'failure.json',dict(status='FAILED',seed=seed,exception=repr(exc),traceback=traceback.format_exc()))
  js(folder.relative_to(P)/'run_state.json',dict(status='FAILED',seed=seed));raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);main(parser.parse_args().seed)

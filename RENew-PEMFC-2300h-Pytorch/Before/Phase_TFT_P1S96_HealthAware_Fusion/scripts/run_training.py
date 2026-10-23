"""Nine one-shot paired CUDA fits; exact Phase H optimizer and stopping rules."""
import os,sys,json,time,math,subprocess,traceback,hashlib,inspect
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch,yaml
from torch.utils.data import DataLoader,TensorDataset
from runtime_common import *
from metrics import measure
RUNS=[(f'{g}_seed{s}',g,s) for s in [42,52,62] for g in MODELS]
def utc():return datetime.now(timezone.utc).isoformat()
def freeze():
 path=P/'configs/FROZEN_PROTOCOL.json'
 if path.exists():raise RuntimeError('NO_REFREEZE')
 h=yaml.safe_load(H_PROTOCOL.read_text(encoding='utf-8'));qualification=json.loads((P/'outputs/CPU_QUALIFICATION.json').read_text())
 assert qualification['status']=='CPU_QUALIFIED'
 tests=(P/'logs/PRETRAIN_TESTS.log').read_text(encoding='utf-8');assert 'OK' in tests and 'FAILED' not in tests
 source=set([H_PROTOCOL,IDENTITY,CATALOG,PROJECT/'Phase_TFT_FullFeature_CausalFilter_Comparison/outputs/E_COMMON_SCALER.json',
  PROJECT/'Phase_F_TFT_DualPath_RegimeFusion/outputs/audits/F_DATA_ACCESS_AUDIT.json'])
 for rel in ['Phase_F_TFT_DualPath_RegimeFusion/models','Phase_TFT_Original_Baseline/models',
  'Phase_H_Determinism_MultiSeed_Stability/scripts','Phase_P2_Final_Gate_Ablation_Training/scripts',
  'Phase_TFT_FullFeature_CausalFilter_Comparison/scripts','Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts',
  'Phase_P1_Gate_Ablation_Engineering_Preflight/scripts']:
  source.update((PROJECT/rel).glob('*.py'))
 for rel in [
  'Phase_F_TFT_DualPath_RegimeFusion/configs/F0_F2_EXECUTION_PROTOCOL.yaml',
  'Phase_F_TFT_DualPath_RegimeFusion/reports/F_FINAL_EXPERIMENT_SUMMARY.md',
  'Phase_H_Determinism_MultiSeed_Stability/reports/H_FINAL_CONCLUSION.md',
  'Phase_P1_Gate_Ablation_Engineering_Preflight/reports/P1_FINAL_TRAINING_READINESS_DECISION.md',
  'Phase_P2_Final_Gate_Ablation_Training/configs/P2_FROZEN_EXECUTION_PROTOCOL.json',
  'Phase_P2_Final_Gate_Ablation_Training/reports/P2_FINAL_RESEARCH_DECISION.md',
  'Phase_P3_Full450_Epoch_Sensitivity/configs/P3_FROZEN_FULL450_PROTOCOL.json',
  'Phase_P3_Full450_Epoch_Sensitivity/reports/P3_SUPERVISOR_ONE_PAGE_SUMMARY.md',
  'Phase_PAA_P1_StateToken_PreAnchor_Alignment/reports/FINAL_P1S_PREANCHOR_AUDIT.md',
  'Phase_CTAE_P1S_Reference_Availability_Erratum/reports/CTA_REFERENCE_AVAILABILITY_ERRATUM.md',
  'Phase_P1S_LongHistory_ShortHorizon_Forecasting/configs/FROZEN_LONG_HISTORY_PROTOCOL.json',
  'Phase_P1S_LongHistory_ShortHorizon_Forecasting/00_研究结论_通俗中文版.md',
  'Phase_P1S_96h_ThreeFold_Comparison/configs/FROZEN_PROTOCOL.json',
  'Phase_P1S_96h_ThreeFold_Comparison/00_实验总结.md']:source.add(PROJECT/rel)
 from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer,sub_modules
 upstream=[Path(inspect.getsourcefile(TemporalFusionTransformer)),Path(inspect.getsourcefile(sub_modules))]
 files={str(p.relative_to(PROJECT)).replace('\\','/'):file_sha(p) for p in sorted(source)}
 cfg=dict(phase=P.name,UTC=utc(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
  authorization='USER_EXPLICIT_NINE_MODEL_SEED_TRAINING',H=h,model_names=MODELS,seeds=[42,52,62],formal_runs=RUNS,max_formal_runs=9,
  health=dict(window_h=96,minimum_observations=5,minimum_span_h=67.2,reference_V=.71383023635,reference_available_h=171.82091,
   availability='conservative_available_end_h <= origin bin_end_ophrs2_h; window uses actual measurement end',standardization='matched Train only',features=['latest R','96h mean','96h population std','slope on actual hours','latest minus median']),
  minimum_evaluation=dict(train=100,validation=30,unique_validation_targets=24),matched_counts=dict(train=870,validation=178,unique_validation_targets=224),
  preprocessing='Exact H full-Train unfiltered scaler, per-window EWMA .3, frozen Top10, original target scale; subset afterwards',
  architecture='Read-only subclass F2; A gate5-8-1; B/C gate10-8-1; no changes to either branch or decoder',
  initialization='Original F2 common state from seed; expanded health columns from isolated seed+100000; B/C identical; original gate-output zero => .5 initial fusion',
  paired_rng='Reset Python/NumPy/CPU/CUDA seed before training; DataLoader generator per seed; every common epoch order fingerprint must match',
  stopping='validation standardized RMSE < best - 1e-12; patience 40; max450, best checkpoint evaluation',
  success=dict(mean_RMSE_reduction_vs_each_control_at_least=.05,paired_seeds_improve_over_both_at_least=2,
   endpoint='No obvious deterioration, judged qualitatively with all paired endpoint values disclosed; no extra numerical zero-worsening threshold',
   target_time_and_tail='Report all target-equal RMSE, P95/P99/max; no hidden extreme targets',validation='Exploratory reused Development; not Test or significance'),
  source_sha256=files,execution_code_sha256={str(p.relative_to(P)).replace('\\','/'):file_sha(p) for d in ['models','scripts','tests'] for p in sorted((P/d).glob('*.py'))},
  upstream_sha256={str(p):file_sha(p) for p in upstream},input_fingerprints=json.loads((P/'inputs/INPUT_FINGERPRINTS.json').read_text()),
  no_M2_or_physics_or_SOH_head=True,no_Assessment_or_Test=True,no_Optuna_or_feature_selection=True,failures='Record and stop, no automatic fresh retries')
 json_write('configs/FROZEN_PROTOCOL.json',cfg);(P/'configs/FROZEN_PROTOCOL.sha256').write_bytes((file_sha(path)+'\n').encode())
 counts=[];init={}
 for s in cfg['seeds']:
  c=cohort(s,h);init[str(s)]={g:state_hash(m) for g,m in c.items()}
  for g,m in c.items():counts.append(dict(model=g,seed=s,total_trainable_parameters=sum(t.numel() for t in m.parameters() if t.requires_grad),gate_parameters=sum(t.numel() for t in m.regime_gate.parameters()),additional_parameters=0 if g==MODELS[0] else 40))
 json_write('outputs/INITIALIZATION_AUTHORITY.json',init);csv_write('outputs/MODEL_PARAMETER_COUNTS.csv',pd.DataFrame(counts))
 print('FROZEN',file_sha(path),flush=True)
def smoke():
 cfg,sha=load_protocol();install_firewall();env=configure(cfg['H']);assert torch.cuda.is_available()
 free,total=torch.cuda.mem_get_info();result=dict(environment=env,free_GPU_bytes=free,total_GPU_bytes=total,models={})
 c=cohort(42,cfg['H'],'cuda')
 for g,m in c.items():
  seed_all(12345);x=torch.randn(32,48,14,device='cuda');h=torch.randn(32,5,device='cuda');y=torch.randn(32,24,device='cuda')
  m.train();before=state_hash(m);opt=torch.optim.AdamW(m.parameters(),lr=cfg['H']['hyperparameters']['learning_rate'],weight_decay=cfg['H']['hyperparameters']['weight_decay']);opt.zero_grad(set_to_none=True)
  z=forward_model(m,x,h);loss=torch.nn.functional.mse_loss(z,y);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm)
  assert all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None);opt.step();assert before!=state_hash(m)
  destination=P/'outputs/preflight'/f'{g}_synthetic.pt';destination.parent.mkdir(parents=True,exist_ok=True)
  checkpoint(destination,dict(state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},synthetic_only=True))
  saved=torch.load(destination,map_location='cpu',weights_only=True);assert all(torch.equal(v.detach().cpu(),saved['state_dict'][k]) for k,v in m.state_dict().items())
  result['models'][g]=dict(status='PASS',forward_shape=list(z.shape),backward=True,optimizer_step=True,checkpoint_round_trip=True,synthetic_only=True)
 json_write('outputs/GPU_SMOKE_TEST.json',result);print('GPU_FORWARD_BACKWARD_OPTIMIZER_PASS',flush=True)
def checkpoint(path,data):
 tmp=path.with_suffix('.pt.writing');torch.save(data,tmp)
 with tmp.open('rb+') as f:os.fsync(f.fileno())
 tmp.replace(path)
def worker(run_id):
 run,g,seed=next(a for a in RUNS if a[0]==run_id);folder=P/'outputs/training_logs'/run;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'run_state.json').exists():raise RuntimeError('NO_RUN_RESTART')
 json_write(str(folder.relative_to(P)/'run_state.json'),dict(status='STARTED',run_id=run,UTC=utc()))
 try:
  cfg,sha=load_protocol();install_firewall();environment=configure(cfg['H']);data,fp=inputs();p=cfg['H']['hyperparameters'];device=torch.device('cuda')
  c=cohort(seed,cfg['H']);m=c[g].to(device);del c
  init=state_hash(m);assert init==json.loads((P/'outputs/INITIALIZATION_AUTHORITY.json').read_text())[str(seed)][g]
  # Construction order and added Linear parameters cannot alter training RNG.
  seed_all(seed);gen=torch.Generator().manual_seed(seed)
  rng=dict(cpu=ah(torch.get_rng_state().numpy()),cuda=[ah(x.cpu().numpy()) for x in torch.cuda.get_rng_state_all()],loader=ah(gen.get_state().numpy()))
  json_write(str(folder.relative_to(P)/'initial.json'),dict(initial_model_sha256=init,rng=rng,seed=seed,no_historical_warm_start=True));json_write(str(folder.relative_to(P)/'environment.json'),environment)
  scale=float(data['target_scale_V']);ty=(data['train_y']/scale).astype(np.float32);vy=data['validation_y']/scale
  loader=DataLoader(TensorDataset(torch.from_numpy(data['train_x']),torch.from_numpy(data['train_health']),torch.from_numpy(ty),torch.arange(len(ty))),batch_size=p['batch_size'],shuffle=True,generator=gen,num_workers=0)
  opt=torch.optim.AdamW(m.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay']);best=math.inf;best_epoch=stale=0;history=[]
  ckpath=P/'outputs/checkpoints'/f'{run}_best.pt';ckpath.parent.mkdir(parents=True,exist_ok=True);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  journal=folder/'batch_events.jsonl'
  for epoch in range(1,p['max_epochs']+1):
   tick=time.perf_counter();m.train();losses=[];order=[]
   with journal.open('a',encoding='utf-8',newline='\n') as stream:
    for batch,(xb,hb,yb,ids) in enumerate(loader):
     origin_ids=data['train_origins'][ids.numpy()].astype(int);order.extend(origin_ids.tolist());xb,hb,yb=xb.to(device),hb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
     json_write(str(folder.relative_to(P)/'latest_batch.json'),dict(epoch=epoch,batch=batch,stage='BEFORE_FORWARD',origin_rows=origin_ids.tolist()))
     z=forward_model(m,xb,hb);loss=torch.nn.functional.mse_loss(z,yb)
     if not torch.isfinite(loss) or not torch.isfinite(z).all():raise RuntimeError('NONFINITE_LOSS_OR_PREDICTION')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),p['gradient_clip_norm'])
     if not torch.isfinite(norm) or any(not torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None):raise RuntimeError('NONFINITE_GRADIENT')
     opt.step();losses.append(float(loss.detach().cpu()))
     stream.write(json.dumps(dict(epoch=epoch,batch=batch,stage='OPTIMIZER_STEP_COMPLETED',origin_rows=origin_ids.tolist(),loss=losses[-1],gradient_norm=float(norm.detach().cpu())))+'\n');stream.flush()
    os.fsync(stream.fileno())
   order_sha=ah(np.array(order,dtype=np.int64));order_path=P/'outputs/sampler'/f'seed{seed}.json';order_path.parent.mkdir(parents=True,exist_ok=True)
   assert sorted(order)==sorted(data['train_origins'].tolist())
   reference=json.loads(order_path.read_text()) if order_path.exists() else {}
   if str(epoch) in reference:assert reference[str(epoch)]==order_sha
   else:
    reference[str(epoch)]=order_sha;json_write(str(order_path.relative_to(P)),reference)
   vp=predict(m,data['validation_x'],data['validation_health'],device);rmse=float(np.sqrt(np.mean((vp-vy)**2)))
   if not math.isfinite(rmse):raise RuntimeError('NONFINITE_VALIDATION')
   if rmse<best-1e-12:
    best,best_epoch,stale=rmse,epoch,0
    checkpoint(ckpath,dict(state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},run_id=run,model=g,seed=seed,
     protocol_sha256=sha,input_fingerprints=fp,initial_state_sha256=init,best_epoch=best_epoch,validation_standardized_RMSE=best,
     target_scale_V=scale,fixed_validation_prediction=torch.from_numpy(vp[:32].copy())))
   else:stale+=1
   row=dict(epoch=epoch,train_standardized_MSE=float(np.mean(losses)),validation_standardized_RMSE=rmse,validation_RMSE_mV=rmse*scale*1000,
    best_epoch=best_epoch,stale_epochs=stale,sampler_order_sha256=order_sha,epoch_time_s=time.perf_counter()-tick)
   history.append(row);csv_write(str(folder.relative_to(P)/'epochs.csv'),pd.DataFrame(history));json_write(str(folder.relative_to(P)/'run_state.json'),dict(status='TRAINING',run_id=run,epoch=epoch,best_epoch=best_epoch,stale=stale,UTC=utc()))
   if epoch==1 or epoch%10==0 or stale>=p['patience']:print(f'{run} epoch={epoch} best={best_epoch} stale={stale} validation_RMSE_mV={row["validation_RMSE_mV"]:.6f}',flush=True)
   if stale>=p['patience']:break
  duration=time.perf_counter()-start;peak=int(torch.cuda.max_memory_allocated());del m,opt
  ck=torch.load(ckpath,map_location='cpu',weights_only=True);assert ck['protocol_sha256']==sha and ck['input_fingerprints']==fp
  c=cohort(seed,cfg['H']);m=c[g].to(device);del c;m.load_state_dict(ck['state_dict'],strict=True)
  fixed=predict(m,data['validation_x'][:32],data['validation_health'][:32],device);assert np.array_equal(fixed,ck['fixed_validation_prediction'].numpy())
  vp,gates=predict(m,data['validation_x'],data['validation_health'],device,return_gate=True);vp=vp*scale
  records=[dict(run_id=run,model=g,seed=seed,origin_row=int(o),target_row=int(data['validation_targets'][i,h]),horizon_step=h+1,
   origin_time_h=float(data['validation_ages'][i]),target_time_h=float(data['validation_target_times'][i,h]),
   y_true_delta_V=float(data['validation_y'][i,h]),y_pred_delta_V=float(vp[i,h]),protocol_sha256=sha) for i,o in enumerate(data['validation_origins']) for h in range(24)]
  frame=pd.DataFrame(records);predrel=f'outputs/predictions/{run}.csv';(P/'outputs/predictions').mkdir(exist_ok=True);csv_write(predrel,frame)
  reloaded=pd.read_csv(P/predrel,float_precision='round_trip');assert np.array_equal(frame[['origin_row','target_row','horizon_step','y_true_delta_V','y_pred_delta_V']].to_numpy(),reloaded[['origin_row','target_row','horizon_step','y_true_delta_V','y_pred_delta_V']].to_numpy())
  metric,horizon,target=measure(frame);(P/'outputs/metrics').mkdir(exist_ok=True);csv_write(f'outputs/metrics/{run}_horizons.csv',horizon);csv_write(f'outputs/metrics/{run}_target_equal.csv',target)
  weights=pd.DataFrame(dict(model=g,seed=seed,origin_row=data['validation_origins'],origin_time_h=data['validation_ages'],gate_weight=gates));csv_write(f'outputs/metrics/{run}_gate_weights.csv',weights)
  result=dict(run_id=run,model=g,seed=seed,status='COMPLETED',**metric,best_epoch=best_epoch,stop_epoch=epoch,training_seconds=duration,
   peak_GPU_bytes=peak,parameter_count=sum(t.numel() for t in m.parameters() if t.requires_grad),checkpoint=str(ckpath.relative_to(P)).replace('\\','/'),
   checkpoint_sha256=file_sha(ckpath),prediction=predrel,prediction_sha256=file_sha(P/predrel),checkpoint_reload_exact=True,
   gate_mean=float(gates.mean()),gate_std=float(gates.std()),gate_min=float(gates.min()),gate_max=float(gates.max()),protocol_sha256=sha,
   Assessment_accessed=False,Test_accessed=False)
  json_write(str(folder.relative_to(P)/'result.json'),result);json_write(str(folder.relative_to(P)/'run_state.json'),dict(status='COMPLETED',run_id=run,epoch=epoch,best_epoch=best_epoch,UTC=utc()))
  load_protocol();print('COMPLETED',run,metric['RMSE_mV'],flush=True)
 except Exception:
  json_write(str(folder.relative_to(P)/'failure.json'),dict(status='TECHNICAL_FAILURE',traceback=traceback.format_exc(),UTC=utc()))
  json_write(str(folder.relative_to(P)/'run_state.json'),dict(status='TECHNICAL_FAILURE',run_id=run,UTC=utc()));raise
def run_all():
 path=P/'outputs/EXECUTION_LEDGER.json'
 if path.exists():raise RuntimeError('NO_AUTOMATIC_REPEAT_OF_FORMAL_RUNS')
 cfg,sha=load_protocol();sm=json.loads((P/'outputs/GPU_SMOKE_TEST.json').read_text());assert all(x['status']=='PASS' for x in sm['models'].values())
 ledger=dict(status='TRAINING',protocol_sha256=sha,started_UTC=utc(),formal_started=0,formal_completed=0,runs={});json_write('outputs/EXECUTION_LEDGER.json',ledger)
 try:
  for run,g,seed in RUNS:
   ledger['active_run']=run;ledger['formal_started']+=1;ledger['runs'][run]=dict(status='STARTED',model=g,seed=seed);json_write('outputs/EXECUTION_LEDGER.json',ledger)
   folder=P/'outputs/training_logs'/run;folder.mkdir(parents=True,exist_ok=True);print('START',run,flush=True)
   with (folder/'console.log').open('w',encoding='utf-8',newline='\n') as f:proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker',run],cwd=P,stdout=f,stderr=subprocess.STDOUT)
   if proc.returncode:raise RuntimeError('FORMAL_RUN_FAILED_NO_RETRY:'+run)
   record=json.loads((folder/'result.json').read_text());ledger['formal_completed']+=1;ledger['runs'][run]=dict(status='COMPLETED',RMSE_mV=record['RMSE_mV']);json_write('outputs/EXECUTION_LEDGER.json',ledger)
   print('DONE',run,'RMSE_mV',record['RMSE_mV'],flush=True)
  results=pd.DataFrame([json.loads((P/'outputs/training_logs'/r/'result.json').read_text()) for r,_,_ in RUNS]);csv_write('outputs/ALL_RUN_METRICS.csv',results)
  ledger.update(status='NINE_RUNS_COMPLETED',active_run=None,completed_UTC=utc());json_write('outputs/EXECUTION_LEDGER.json',ledger)
 except Exception:
  ledger.update(status='TECHNICAL_FAILURE_NO_RETRY',traceback=traceback.format_exc(),UTC=utc());json_write('outputs/EXECUTION_LEDGER.json',ledger);raise
if __name__=='__main__':
 {'freeze':freeze,'smoke':smoke,'run':run_all,'worker':lambda:worker(sys.argv[2])}[sys.argv[1]]()

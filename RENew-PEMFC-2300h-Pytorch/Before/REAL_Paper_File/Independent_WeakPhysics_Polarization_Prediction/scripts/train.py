"""24 predeclared CPU runs max; resumable perRun, never rerun completed ID."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import time,copy,random
import torch
from common import *
from network import VoltageMLP,slope_loss,norm
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_default_dtype(torch.float64)
def tensors(frame,scale):return torch.tensor((frame[FEATURES].to_numpy()-scale['x_mean'])/scale['x_scale'],dtype=torch.float64)
def setup(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);m=VoltageMLP().double();assert sum(p.numel() for p in m.parameters())==1217;return m
def preflight(cfg,data,probes,scalers):
 if (P/'outputs/REAL_TRAIN_PREFLIGHT.json').exists():return
 records=[]
 for s in scalers:
  tr=data[data.task.eq(s['task'])&data.fold.eq(s['fold'])&data.role.eq('train')];ph=probes[probes.task.eq(s['task'])&probes.fold.eq(s['fold'])];x=tensors(tr,s);xp=tensors(ph,s);y=torch.tensor((tr.true_V.to_numpy()-s['y_mean'])/s['y_scale'])
  for seed in cfg['seeds']:
   m=setup(seed);l=(m(x)-y).square().mean();phys,dj,raw=slope_loss(m,xp,s['y_scale'],s['x_scale'][0]);dg=norm(torch.autograd.grad(l,tuple(m.parameters()),retain_graph=True));pg=norm(torch.autograd.grad(phys,tuple(m.parameters()),allow_unused=True));records.append({'task':s['task'],'fold':s['fold'],'seed':seed,'real_Train_n':len(tr),'unlabeled_probes':len(ph),'data_loss':float(l.detach()),'physics_loss':float(phys.detach()),'physics_raw_V2_per_j2':float(raw.detach()),'violation_rate':float((dj>1e-10).double().mean()),'data_grad_norm':dg,'physics_grad_norm':pg,'lambda_weighted_grad_ratio':cfg['lambda']*pg/max(dg,1e-30),'computable':True,'nontrivial_at_initialization':bool(float(phys.detach())>1e-16 and pg>1e-12)})
 cs('outputs/REAL_TRAIN_PREFLIGHT.csv',pd.DataFrame(records));js('outputs/REAL_TRAIN_PREFLIGHT.json',{'status':'COMPUTABLE','lambda_frozen':cfg['lambda'],'records':records,'no_validation_scores_used':True,'no_optimizer_steps_in_preflight':True,'inactive_initialization_is_not_a_reason_to_change_seed_or_loss':True})
def run_one(cfg,s,data,probes,seed,arm):
 runid=f"{s['task']}_{s['fold']}_{seed}_{arm}";done=P/f'outputs/runs/{runid}_COMPLETE.json'
 if done.exists():print('SKIP_COMPLETED',runid,flush=True);return
 q=data[data.task.eq(s['task'])&data.fold.eq(s['fold'])];tr=q[q.role.eq('train')];va=q[q.role.eq('validation')];ph=probes[probes.task.eq(s['task'])&probes.fold.eq(s['fold'])]
 x=tensors(tr,s);xp=tensors(ph,s);y=torch.tensor((tr.true_V.to_numpy()-s['y_mean'])/s['y_scale']);m=setup(seed);opt=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'],weight_decay=0);cp=P/f'models/{runid}_resume.pt';logpath=out(f'outputs/logs/{runid}.csv');start=0;history=[];activated=0;begin=time.perf_counter()
 if cp.exists():
  saved=torch.load(cp,map_location='cpu',weights_only=False);m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['epoch'];history=saved['history'];activated=saved['activated'];print('RESUME_SAME_RUN',runid,start,flush=True)
 else:js(f'outputs/runs/{runid}_STARTED.json',{'runid':runid,'seed':seed,'arm':arm,'protocol_SHA256':sha(P/'configs/FROZEN_PROTOCOL.json'),'Train_ids':tr.sample_id.tolist(),'Validation_ids':va.sample_id.tolist(),'parameter_count':1217,'device':'cpu','initial_state_SHA256':hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest()})
 for epoch in range(start+1,cfg['epochs']+1):
  opt.zero_grad(set_to_none=True);prediction=m(x);ld=(prediction-y).square().mean();lp,dj,raw=slope_loss(m,xp,s['y_scale'],s['x_scale'][0]);dg=norm(torch.autograd.grad(ld,tuple(m.parameters()),retain_graph=True));pg=norm(torch.autograd.grad(lp,tuple(m.parameters()),retain_graph=True,allow_unused=True));lam=cfg['lambda'] if arm=='B_PHYSICS' else 0.;total=ld+lam*lp
  if float(lp.detach())>1e-16 and pg>1e-12:activated+=1
  row={'epoch':epoch,'data_loss_normalized':float(ld.detach()),'physics_loss_normalized':float(lp.detach()),'physics_loss_raw':float(raw.detach()),'violation_rate':float((dj>1e-10).double().mean()),'max_positive_slope_ohm_cm2':float(torch.relu(dj).max().detach()),'data_grad_norm':dg,'physics_grad_norm':pg,'weighted_physics_grad_ratio':lam*pg/max(dg,1e-30),'lambda_applied':lam,'total_loss':float(total.detach())};assert all(np.isfinite(v) for v in row.values()),'NONFINITE_TECHNICAL_FAILURE';history.append(row);total.backward();opt.step()
  if epoch%500==0:
   torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history,'activated':activated},cp);cs(f'outputs/logs/{runid}.csv',pd.DataFrame(history));print('TRAIN',runid,epoch,'data',row['data_loss_normalized'],'phys',row['physics_loss_normalized'],flush=True)
 torch.save({'model':m.state_dict(),'scaler':s,'runid':runid,'seed':seed,'arm':arm,'epoch':cfg['epochs'],'protocol_SHA256':sha(P/'configs/FROZEN_PROTOCOL.json')},out(f'models/{runid}.pt'));cs(f'outputs/logs/{runid}.csv',pd.DataFrame(history))
 with torch.no_grad():pred=m(tensors(va,s)).numpy()*s['y_scale']+s['y_mean']
 result=va.copy();result['runid']=runid;result['seed']=seed;result['arm']=arm;result['predicted_V']=pred;result['error_mV']=(pred-result.true_V)*1000;cs(f'outputs/predictions/{runid}.csv',result)
 # Dense probes include only allowed validation targetcontexts for evaluation, never training.
 dense=[]
 for (event,b),v in va.groupby(['checkpoint','branch']):
  for j in np.linspace(float(v.domain_low.iloc[0]),float(v.domain_high.iloc[0]),101):dense.append({'checkpoint':event,'branch':b,'j_design_A_cm2':j,'direction_ascending':int(v.direction_ascending.iloc[0]),'context':float(v.context.iloc[0])})
 de=pd.DataFrame(dense);xs=tensors(de,s).requires_grad_(True);yp=m(xs);sl=torch.autograd.grad(yp.sum(),xs)[0][:,0].detach().numpy()*s['y_scale']/s['x_scale'][0];de['predicted_V']=yp.detach().numpy()*s['y_scale']+s['y_mean'];de['slope_ohm_cm2']=sl;de['runid']=runid;de['seed']=seed;de['arm']=arm;cs(f'outputs/probes/{runid}.csv',de)
 js(f'outputs/runs/{runid}_COMPLETE.json',{'runid':runid,'task':s['task'],'fold':s['fold'],'seed':seed,'arm':arm,'status':'COMPLETED','optimizer_steps':cfg['epochs'],'physics_positive_loss_and_gradient_epochs':activated,'weighted_physics_active_epochs':activated if arm=='B_PHYSICS' else 0,'physics_inactive_entire_training':activated==0,'seconds_current_session':time.perf_counter()-begin,'train_n':len(tr),'validation_n':len(va),'validation_label_used_for_selection':False,'weights_SHA256':sha(P/f'models/{runid}.pt'),'metrics':metric(result.true_V,result.predicted_V)})
 print('COMPLETED',runid,metric(result.true_V,result.predicted_V),flush=True)
def run():
 cfg=config();assert jl('outputs/SYNTHETIC_TEST_RESULTS.json')['status']=='PASS','Synthetic/autogradtests required before realtraining'
 data=pd.read_csv(P/'data_contracts/ALL_LEGAL_SAMPLES.csv',float_precision='round_trip');probes=pd.read_csv(P/'data_contracts/UNLABELED_TRAIN_PHYSICS_PROBES.csv',float_precision='round_trip');scalers=jl('data_contracts/TRAIN_ONLY_SCALERS.json');preflight(cfg,data,probes,scalers)
 for s in scalers:
  for seed in cfg['seeds']:
   for arm in ['A_DATA_ONLY','B_PHYSICS']:run_one(cfg,s,data,probes,seed,arm)
 config();js('outputs/TRAINING_COMPLETE.json',{'status':'COMPLETED','run_count':len(scalers)*6,'epochs_per_run':cfg['epochs'],'TFT_or_Ridge_updates':0,'GPU_runs':0,'Test_read':False});js('outputs/TRAIN_READ_LEDGER.json',sorted(set(ACCESS)))
if __name__=='__main__':firewall();run()

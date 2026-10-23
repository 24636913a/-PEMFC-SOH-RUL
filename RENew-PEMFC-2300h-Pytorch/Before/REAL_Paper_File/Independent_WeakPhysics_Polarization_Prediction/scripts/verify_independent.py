"""No training: independent NumPy forward/analytic slope and real-loss finite differences."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch
from common import *
from network import VoltageMLP,slope_loss,norm
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
def numpy_forward(x,state):
 w0=state['layers.0.weight'];b0=state['layers.0.bias'];w1=state['layers.2.weight'];b1=state['layers.2.bias'];w2=state['layers.4.weight'];b2=state['layers.4.bias']
 a=x@w0.T+b0;s=1/(1+np.exp(-a));h=a*s;der0=s+a*s*(1-s);c=h@w1.T+b1;t=1/(1+np.exp(-c));h1=c*t;der1=t+c*t*(1-t);y=(h1@w2.T+b2).ravel();d=((der0*w0[:,0])@w1.T*der1)@w2.T
 return y,d.ravel()
def numpy_state(m):return {k:v.detach().numpy().copy() for k,v in m.state_dict().items()}
def run():
 cfg=config();contracts=jl('data_contracts/TRAIN_ONLY_SCALERS.json');data=pd.read_csv(P/'data_contracts/ALL_LEGAL_SAMPLES.csv',float_precision='round_trip');ph=pd.read_csv(P/'data_contracts/UNLABELED_TRAIN_PHYSICS_PROBES.csv',float_precision='round_trip');pref=pd.read_csv(P/'outputs/REAL_TRAIN_PREFLIGHT.csv',float_precision='round_trip');pred=pd.read_csv(P/'outputs/ALL_POINT_PREDICTIONS.csv',float_precision='round_trip');probes=pd.read_csv(P/'outputs/ALL_EVALUATION_SLOPES.csv',float_precision='round_trip');checks=[];maxpred=0.;maxslope=0.;maxfd=0.;fdrows=[]
 for rel,h in cfg['data_contract_sha256'].items():assert sha(P/rel)==h,rel
 # Independent source-label and history-context identity, not merely self-consistent model outputs.
 dw=rd('dwells');stable=dw[dw.window.eq('middle50')&dw.stable_domain].set_index(['checkpoint','branch','dwell_group']);assert len(stable)==105
 measurement=rd('measurement');identity=rd('identity');measurement=measurement[(measurement.stable_window=='middle_50')&measurement.cycle_interval_id.isin(identity.cycle_interval_id)];measurement['calendar']=pd.to_datetime(measurement.measurement_end_datetime);normal=measurement[measurement.anode_inlet_temperature_C.between(74,76)&measurement.anode_inlet_RH_pct.between(39,41)].sort_values('calendar');metadata=rd('metadata').set_index('checkpoint');eventorigin=pd.read_csv(P/'data_contracts/B_EVENT_ORIGIN_IDENTITY.csv',float_precision='round_trip').set_index('checkpoint')
 for row in data.itertuples():
  original=stable.loc[(row.checkpoint,row.branch,row.dwell_group)];assert float(original.V_V)==row.true_V and row.raw_first==original.raw_first and row.raw_last==original.raw_last
  assert abs(row.j_design_A_cm2-original.set_current_A/280)<1e-14 and abs(original.j_A_cm2-row.j_design_A_cm2)<=.005
  if row.task=='A':
   train=data[data.task.eq('A')&data.role.eq('train')&data.sample_id.eq(row.context_source_id)];assert len(train)==1 and train.true_V.iloc[0]==row.context
  else:
   ev=eventorigin.loc[row.checkpoint];pool=normal[(normal.calendar>pd.Timestamp(metadata.loc[ev.previous_checkpoint,'end_calendar']))&(normal.calendar<pd.Timestamp(metadata.loc[row.checkpoint,'start_calendar']))];first=pool.iloc[0];assert first.cycle_interval_id==row.context_source_id and abs(first.P1S_operating_voltage_retention-row.context)<1e-14 and first.calendar==pd.Timestamp(row.origin_calendar)
 for s in contracts:
  tr=data[data.task.eq(s['task'])&data.fold.eq(s['fold'])&data.role.eq('train')];val=data[data.task.eq(s['task'])&data.fold.eq(s['fold'])&data.role.eq('validation')];pp=ph[ph.task.eq(s['task'])&ph.fold.eq(s['fold'])]
  assert not (set(tr.sample_id)&set(val.sample_id));assert tr.sample_id.tolist()==s['training_sample_ids'];mu=tr[FEATURES].mean().to_numpy();sd=tr[FEATURES].std(ddof=0).to_numpy(copy=True);sd[sd==0]=1.;assert np.max(abs(mu-s['x_mean']))<1e-12 and np.max(abs(sd-s['x_scale']))<1e-12
  if s['task']=='A':assert set(val.context_source_id)<=set(tr.sample_id)
  else:assert pd.to_datetime(tr.end_calendar).max()<pd.to_datetime(val.origin_calendar).min()
  xp=(pp[FEATURES].to_numpy()-s['x_mean'])/s['x_scale']
  for seed in cfg['seeds']:
   # Initialization reproducible; physicalgradient checked independently by finite differences of NumPy slope formula.
   torch.manual_seed(seed);m=VoltageMLP().double();lp,_,_=slope_loss(m,torch.tensor(xp),s['y_scale'],s['x_scale'][0]);grads=torch.autograd.grad(lp,tuple(m.parameters()),allow_unused=True);gn=norm(grads);state=numpy_state(m);_,dn=numpy_forward(xp,state);np_loss=float(np.mean(np.maximum(dn,0)**2));saved=pref[pref.task.eq(s['task'])&pref.fold.eq(s['fold'])&pref.seed.eq(seed)].iloc[0];assert abs(np_loss-saved.physics_loss)<1e-12 and abs(gn-saved.physics_grad_norm)<1e-10
   rng=np.random.default_rng(117);direction={};inner=0.
   for (name,p),grad in zip(m.named_parameters(),grads):
    z=rng.normal(size=p.shape);direction[name]=z;inner+=float((grad.detach().numpy()*z).sum()) if grad is not None else 0.
   h=1e-6
   plus={k:state[k]+h*direction[k] for k in state};minus={k:state[k]-h*direction[k] for k in state};dp=numpy_forward(xp,plus)[1];dm=numpy_forward(xp,minus)[1];finite=float((np.mean(np.maximum(dp,0)**2)-np.mean(np.maximum(dm,0)**2))/(2*h));difference=abs(inner-finite);assert difference<1e-7+abs(inner)*1e-5;maxfd=max(maxfd,difference);fdrows.append({'task':s['task'],'fold':s['fold'],'seed':seed,'real_Train_physics_loss':np_loss,'autograd_directional_parameter_derivative':inner,'independent_NumPy_finite_difference':finite,'absolute_difference':difference})
   for arm in ['A_DATA_ONLY','B_PHYSICS']:
    rid=f"{s['task']}_{s['fold']}_{seed}_{arm}";model=torch.load(P/f'models/{rid}.pt',map_location='cpu',weights_only=False);state={k:v.numpy() for k,v in model['model'].items()};g=pred[pred.runid.eq(rid)];p=probes[probes.runid.eq(rid)];assert g.sample_id.tolist()==val.sample_id.tolist()
    x=(g[FEATURES].to_numpy()-s['x_mean'])/s['x_scale'];v,d=numpy_forward(x,state);difference=float(np.max(abs(v*s['y_scale']+s['y_mean']-g.predicted_V.to_numpy())));assert difference<1e-12;maxpred=max(maxpred,difference)
    xx=(p[FEATURES].to_numpy()-s['x_mean'])/s['x_scale'];v,d=numpy_forward(xx,state);difference=float(np.max(abs(d*s['y_scale']/s['x_scale'][0]-p.slope_ohm_cm2.to_numpy())));assert difference<1e-11;maxslope=max(maxslope,difference)
    log=pd.read_csv(P/f'outputs/logs/{rid}.csv',float_precision='round_trip');complete=jl(f'outputs/runs/{rid}_COMPLETE.json');assert len(log)==cfg['epochs'] and log.epoch.tolist()==list(range(1,cfg['epochs']+1));count=int(((log.physics_loss_normalized>1e-16)&(log.physics_grad_norm>1e-12)).sum());assert count==complete['physics_positive_loss_and_gradient_epochs']
    if arm=='A_DATA_ONLY':assert (log.lambda_applied==0).all();assert np.max(abs(log.total_loss-log.data_loss_normalized))<1e-12
    else:assert (log.lambda_applied==cfg['lambda']).all();assert np.max(abs(log.total_loss-log.data_loss_normalized-cfg['lambda']*log.physics_loss_normalized))<1e-12
    start=jl(f'outputs/runs/{rid}_STARTED.json');checks.append({'runid':rid,'initial_state_SHA256':start['initial_state_SHA256'],'optimizer_steps':len(log),'weights_SHA256':sha(P/f'models/{rid}.pt'),'A_B_no_validation_selection':not complete['validation_label_used_for_selection']})
  for seed in cfg['seeds']:
   a=next(r for r in checks if r['runid']==f"{s['task']}_{s['fold']}_{seed}_A_DATA_ONLY");b=next(r for r in checks if r['runid']==f"{s['task']}_{s['fold']}_{seed}_B_PHYSICS");assert a['initial_state_SHA256']==b['initial_state_SHA256'] and a['optimizer_steps']==b['optimizer_steps']
 # Independently check all eventmetrics and primaryequalmeans, no optimizer call.
 ev=pd.read_csv(P/'outputs/EVENT_SEED_METRICS.csv',float_precision='round_trip')
 for r in ev.itertuples():
  g=pred[pred.task.eq(r.task)&pred.fold.eq(r.fold)&pred.seed.eq(r.seed)&pred.arm.eq(r.arm)&pred.checkpoint.eq(r.checkpoint)];e=(g.predicted_V-g.true_V).to_numpy()*1000;a=abs(e)
  for value,saved in [(a.mean(),r.MAE_mV),(np.sqrt((e*e).mean()),r.RMSE_mV),(np.quantile(a,.95),r.P95_mV),(a.max(),r.max_mV)]:assert abs(value-saved)<1e-9
  p=probes[probes.task.eq(r.task)&probes.fold.eq(r.fold)&probes.seed.eq(r.seed)&probes.arm.eq(r.arm)&probes.checkpoint.eq(r.checkpoint)];assert abs((p.slope_ohm_cm2>1e-10).mean()-r.violation_rate)<1e-12
 primary=pd.read_csv(P/'outputs/PRIMARY_EVENT_EQUAL_METRICS.csv',float_precision='round_trip')
 for r in primary.itertuples():
  g=ev[ev.task.eq(r.task)&ev.seed.eq(r.seed)&ev.arm.eq(r.arm)];assert abs(g.MAE_mV.mean()-r.MAE_mV)<1e-12
 cs('outputs/INDEPENDENT_REAL_PARAMETER_GRADIENT_CHECKS.csv',pd.DataFrame(fdrows));cs('outputs/INDEPENDENT_RUN_IDENTITY.csv',pd.DataFrame(checks));config();js('outputs/INDEPENDENT_VERIFICATION.json',{'status':'PASS','runs_checked':len(checks),'predictions_checked':len(pred),'evaluation_derivatives_checked':len(probes),'NumPy_prediction_max_abs_difference_V':maxpred,'NumPy_slope_max_abs_difference_ohm_cm2':maxslope,'real_parameter_gradient_fd_max_difference':maxfd,'no_fits':True,'same_AB_initial_states_and_budget':True,'sources_and_core_code_unchanged':True,'Test_read':False});print('INDEPENDENT_PASS',len(checks),len(pred),maxpred,maxslope,maxfd)
if __name__=='__main__':firewall();run()

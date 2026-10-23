"""Fixed preregistered decisions; no fitting, model selection or new losses."""
from common import *
def run():
 cfg=config();assert jl('outputs/TRAINING_COMPLETE.json')['status']=='COMPLETED'
 pred=pd.concat([pd.read_csv(q,float_precision='round_trip') for q in sorted((P/'outputs/predictions').glob('*.csv'))],ignore_index=True);probes=pd.concat([pd.read_csv(q,float_precision='round_trip') for q in sorted((P/'outputs/probes').glob('*.csv'))],ignore_index=True)
 # RunID encodes task/fold; pull metadata without reclassifying individual samples.
 records=[jl(q.relative_to(P).as_posix()) for q in sorted((P/'outputs/runs').glob('*_COMPLETE.json'))];runmeta={r['runid']:r for r in records}
 probes['task']=[runmeta[k]['task'] for k in probes.runid];probes['fold']=[runmeta[k]['fold'] for k in probes.runid];pred['interval']=pd.cut(pred.j_design_A_cm2,[0,.6,.9,2],labels=['0.36_to_0.60','0.60_to_0.90','0.90_to_1.80'],right=True).astype(str)
 cs('outputs/ALL_POINT_PREDICTIONS.csv',pred);cs('outputs/ALL_EVALUATION_SLOPES.csv',probes)
 def slope(g):
  d=g.slope_ohm_cm2.to_numpy();positive=np.maximum(d,0)
  return {'physics_probe_n':len(g),'violation_rate':float((d>1e-10).mean()),'positive_slope_RMS_ohm_cm2':float(np.sqrt(np.mean(positive**2))),'max_positive_slope_ohm_cm2':float(positive.max())}
 group=['task','fold','seed','arm','checkpoint'];met=[]
 for key,g in pred.groupby(group):
  task,fold,seed,arm,event=key;ph=probes[probes.task.eq(task)&probes.fold.eq(fold)&probes.seed.eq(seed)&probes.arm.eq(arm)&probes.checkpoint.eq(event)];met.append(dict(zip(group,key))|metric(g.true_V,g.predicted_V)|slope(ph))
 eventmet=pd.DataFrame(met);cs('outputs/EVENT_SEED_METRICS.csv',eventmet)
 cs('outputs/POOLED_HOLDOUT_METRICS.csv',pd.DataFrame([dict(zip(['task','seed','arm'],key))|metric(g.true_V,g.predicted_V) for key,g in pred.groupby(['task','seed','arm'])]))
 for extra,name in [('branch','DIRECTION_METRICS.csv'),('interval','CURRENT_INTERVAL_METRICS.csv')]:
  values=[]
  for key,g in pred.groupby(group+[extra],observed=True):values.append(dict(zip(group+[extra],key))|metric(g.true_V,g.predicted_V))
  cs('outputs/'+name,pd.DataFrame(values))
 primary=[]
 for (task,seed,arm),g in eventmet.groupby(['task','seed','arm']):
  primary.append({'task':task,'seed':int(seed),'arm':arm,'events':g.checkpoint.nunique(),'real_holdout_platforms':int(g.n.sum()),**{k:float(g[k].mean()) for k in ['MAE_mV','RMSE_mV','P95_mV','max_mV','violation_rate','positive_slope_RMS_ohm_cm2','max_positive_slope_ohm_cm2']}})
 prim=pd.DataFrame(primary);cs('outputs/PRIMARY_EVENT_EQUAL_METRICS.csv',prim)
 summary=[]
 for (task,arm),g in prim.groupby(['task','arm']):
  summary.append({'task':task,'arm':arm,'seeds':len(g),**{k+'_mean':float(g[k].mean()) for k in ['MAE_mV','RMSE_mV','P95_mV','max_mV','violation_rate','positive_slope_RMS_ohm_cm2']},**{k+'_seed_sd':float(g[k].std(ddof=1)) for k in ['MAE_mV','RMSE_mV']}})
 summ=pd.DataFrame(summary);cs('outputs/THREE_SEED_MEAN_AND_SD.csv',summ);cs('outputs/COMPLETE_RUN_LEDGER.csv',pd.DataFrame([{k:v for k,v in r.items() if k!='metrics'} for r in records]))
 decisions=[];comparisons=[]
 for task,g in prim.groupby('task'):
  a=g[g.arm.eq('A_DATA_ONLY')].set_index('seed');b=g[g.arm.eq('B_PHYSICS')].set_index('seed');wins=int((b.MAE_mV<a.MAE_mV).sum());phywins=0
  for seed in cfg['seeds']:
   ar=a.loc[seed];br=b.loc[seed];eligible=bool((ar.violation_rate>1e-8 and br.violation_rate<=ar.violation_rate*.5) or (ar.positive_slope_RMS_ohm_cm2>1e-8 and br.positive_slope_RMS_ohm_cm2<=ar.positive_slope_RMS_ohm_cm2*.5));phywins+=eligible
   comparisons.append({'task':task,'seed':seed,'MAE_data_mV':ar.MAE_mV,'MAE_physics_mV':br.MAE_mV,'MAE_gain_percent':float((ar.MAE_mV-br.MAE_mV)/ar.MAE_mV*100),'baseline_violation_rate':ar.violation_rate,'physics_violation_rate':br.violation_rate,'physics_criterion_seed_pass':eligible})
  gain=float((a.MAE_mV.mean()-b.MAE_mV.mean())/a.MAE_mV.mean()*100);notworse=bool(b.MAE_mV.mean()<=a.MAE_mV.mean()+1e-9 and b.RMSE_mV.mean()<=a.RMSE_mV.mean()+1e-9)
  eventmeans=eventmet[eventmet.task.eq(task)].groupby(['checkpoint','arm']).MAE_mV.mean().unstack();eventbad=bool((eventmeans.B_PHYSICS>eventmeans.A_DATA_ONLY*1.2+1e-9).any());active=sum(r['weighted_physics_active_epochs'] for r in records if r['task']==task and r['arm']=='B_PHYSICS')
  promising=bool(phywins>=2 and notworse and gain>=2 and wins>=2 and not eventbad);status='WEAK_PHYSICS_BRANCH_PROMISING' if promising else 'PHYSICS_DESCRIPTIVE_ONLY_NO_INCREMENT'
  decisions.append({'task':task,'status':status,'MAE_gain_percent':gain,'MAE_seed_wins':wins,'physics_reduction_seed_passes':phywins,'mean_MAE_RMSE_nonworse':notworse,'any_event_mean_MAE_worsens_over20percent':eventbad,'physics_weighted_active_epochs':active,'constraint_activity_status':'PHYSICS_CONSTRAINT_INACTIVE' if active==0 else 'PHYSICS_ACTIVATED_DURING_TRAINING','ordinary_final_all_monotone':bool((a.violation_rate==0).all()),'scope':'Measuredcurveinterpolation/reconstruction' if task=='A' else 'Nextscheduledcheckpoint conditionalnominalprotocol backtest; measurementcompletionassumption, not strictonlineSOH'})
 future_promising=any(d['task']=='B' and d['status']=='WEAK_PHYSICS_BRANCH_PROMISING' for d in decisions)
 integration='PROCEED_TO_SOH_INTEGRATION_RESEARCH' if future_promising else 'RETAIN_DESCRIPTIVE_PHYSICS_ONLY'
 cs('outputs/PAIRED_SEED_AB_COMPARISON.csv',pd.DataFrame(comparisons));js('outputs/RESEARCH_DECISION.json',{'task_decisions':decisions,'integration_decision':integration,'TFT_integration_qualified':False,'SOH_integration_qualified':False,'reason':'No incremental evidence -> retain diagnostics/ablation only. Only a promising conditionalfuturetask would justify separateSOHinterfacequalification research, never automaticfusiontraining. Bothinterfaces require past-only crossfit/independentSOHlabels and certifiedtargetconditions.','H1_H2_H3_changed':False,'TFT_changed':False,'new_RUL_training':0,'statistical_significance_claim':False})
 # Final actual loss/gradient activity histories and chronology are separate from inference metrics.
 activity=[]
 for r in records:
  log=pd.read_csv(P/f"outputs/logs/{r['runid']}.csv",float_precision='round_trip');activity.append({'runid':r['runid'],'task':r['task'],'fold':r['fold'],'seed':r['seed'],'arm':r['arm'],'epochs':len(log),'active_epochs':r['physics_positive_loss_and_gradient_epochs'],'physics_loss_max':float(log.physics_loss_normalized.max()),'physics_loss_last_preupdate':float(log.physics_loss_normalized.iloc[-1]),'physics_grad_max':float(log.physics_grad_norm.max()),'weighted_grad_ratio_max':float(log.weighted_physics_grad_ratio.max()),'weighted_grad_ratio_median':float(log.weighted_physics_grad_ratio.median()),'positive_loss_last_epoch':int(log.loc[log.physics_loss_normalized>1e-16,'epoch'].max()) if (log.physics_loss_normalized>1e-16).any() else None})
 cs('outputs/REAL_PHYSICS_ACTIVITY.csv',pd.DataFrame(activity));config();print(summ.to_string(index=False));print(json.dumps(decisions,ensure_ascii=False,indent=2))
if __name__=='__main__':firewall();run()
